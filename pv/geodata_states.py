"""Open geodata of the German states for PV module geometry (orthophoto, laser surface model / point cloud,
terrain model), fetched per park into a local cache. One function per state; all return a dict
{product: [local paths]} plus acquisition metadata where the portal exposes it.

Verified 2026-10-16 (see STATUS 2026-10-16 "Recherche Laser-Oberflächendaten"):
  Niedersachsen   STAC APIs: dop.stac / dom.stac / dgm.stac / bdom.stac .lgln.niedersachsen.de (COG, CC BY 4.0,
                  items carry the acquisition date; several years per tile -> choose the newest after `after`)
  Brandenburg     https://data.geobasis-bb.de/geobasis/daten/{als/laz,dop/rgbi_tif,bdom/tif,dgm/tif}/<prod>_33<E>-<N>.zip
                  (1 km tiles, EPSG:25833, DL-DE BY 2.0; acquisition from the file headers)
  Mecklenburg-V.  Atom feeds https://www.geodaten-mv.de/dienste/{dop20,dgm,dom}_atom -> dataset feeds with 2 km tiles
                  (EPSG:25833, CC BY 4.0): dom1_33_<E>_<N>_2_xyz.zip, dop20rgb_33_<E>_<N>_2_mv.tif, dgm1 ...
  Bayern          laser point cloud https://geodaten.bayern.de/odd_data/laser/<E>_<N>.laz (1 km, EPSG:25832, CC BY 4.0);
                  DOP20/DOM20/DGM1 see pv.geodata_bayern
  Sachsen-Anhalt  DOP20 2 km tiles direct (gfds_webshare/sec-download/LVermGeo/DOP20/32<E><N>.tif); DOM1 and DGM1 by HTTP
                  range reads out of the statewide zips (Online-Bereitstellung-LVermGeo/{DOM/DOM1,DGM/DGM1}_<1..4>.zip,
                  ~12 GB each), names dom1_32_<E>_<N>_2_st_<year>.tif (fetch_st, 2026-10-22). The point cloud is not open.
Added 2026-10-22 (phase 2a; portals checked that day):
  Nordrhein-W.    https://www.opengeodata.nrw.de/produkte/geobasis/{hm/3dm_l_las/3dm_l_las, hm/dgm1_tiff/dgm1_tiff,
                  hm/dom1_tiff/dom1_tiff, lusat/akt/dop/dop_jp2_f10}/ with XML indexes; 1 km tiles EPSG:25832, LAZ 4-10
                  points/m2, DOP10 RGBI as JPEG2000; dl-de/zero-2-0 (fetch_nw).
  Hessen          gds.hessen.de Downloadcenter, REST API /INTERSHOP/rest/WFS/HLBG-Geodaten-Site/-/downloadcenter?path=...;
                  DOM1 (laser), DGM1, DOP20 (JPG + jgw) as zip per municipality in a folder per district, 1 km tiles
                  EPSG:25832 read by range requests; flight date per tile from Metadaten_DGM-DOM_<month>.xlsx; the point
                  cloud only by shop order; dl-de/zero-2-0 (fetch_he).
  Sachsen         links per tile from the ArcGIS service of the Produktdownload app (geodienste.sachsen.de ...
                  rest_geosn_downloadlinks/MapServer/<layer>/query?where=Kachel='<E><N>'), files on the GeoCloud
                  (share ids change; the morning of 2026-10-22 all shares answered 404, in the afternoon they worked),
                  2 km tiles EPSG:25833, LSC (LAZ), DGM1, DOM1, DOP20; dl-de/by-2-0 (fetch_sn).
  Thüringen       TLBG, CC BY 4.0, 1 km tiles EPSG:25832 (LAZ, DGM1, DOP20). On 2026-10-22 geoportal.geoportal-th.de
                  served a maintenance page and geoportal.thueringen.de a bot challenge -> files by hand (fetch_th reads
                  the cache only).
Licences: attribution required (GeoBasis-DE / <Landesamt>, year)."""
import json
import os
import re
import subprocess
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET

from pyproj import Transformer

UA = "Mozilla/5.0 (X11; Linux x86_64) synthetic_re_data_generation/geodata_states"
_TO_UTM = {32: Transformer.from_crs("EPSG:4326", "EPSG:25832", always_xy=True),
           33: Transformer.from_crs("EPSG:4326", "EPSG:25833", always_xy=True)}


def tiles_for(lat: float, lon: float, zone: int, margin_m: float = 300.0, size_m: float = 1000.0) -> list:
    """All (E, N) tile keys (lower-left corner in units of size_m, as integers in km when size_m=1000)
    covering the square +-margin around the point. Unlike geodata_bayern.tiles_for it returns the full range."""
    x, y = _TO_UTM[zone].transform(lon, lat)
    es = range(int((x - margin_m) // size_m), int((x + margin_m) // size_m) + 1)
    ns = range(int((y - margin_m) // size_m), int((y + margin_m) // size_m) + 1)
    return [(e, n) for e in es for n in ns]


def _curl(url: str, path: str, timeout: int = 1800, ua: str = UA) -> str:
    if os.path.exists(path) and os.path.getsize(path) > 1000:
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    r = subprocess.run(["curl", "-s", "-f", "-L", "-m", str(timeout), "-A", ua, "-o", path + ".part", url])
    if r.returncode != 0:
        if os.path.exists(path + ".part"):
            os.remove(path + ".part")
        raise RuntimeError(f"download failed ({r.returncode}): {url}")
    os.replace(path + ".part", path)
    return path


def _get_json(url: str, data: dict = None) -> dict:
    req = urllib.request.Request(url, data=json.dumps(data).encode() if data else None,
                                 headers={"User-Agent": UA, "Content-Type": "application/json", "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def _get_text(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read().decode("utf-8", "replace")


# ----------------------------------------------------------------------------------------------- Niedersachsen
NI_STAC = {"dop20": ("https://dop.stac.lgln.niedersachsen.de/search", "DOP", "dop20_rgb"),
           "dom1": ("https://dom.stac.lgln.niedersachsen.de/search", "dom1", "dom1-tif"),
           "dgm1": ("https://dgm.stac.lgln.niedersachsen.de/search", "dgm1", None),
           "bdom20": ("https://bdom.stac.lgln.niedersachsen.de/search", "BDOM", None)}


def fetch_ni(lat: float, lon: float, out_dir: str, products=("dop20", "dom1", "dgm1"), margin_m: float = 300.0,
             after: str = None) -> dict:
    """Niedersachsen via STAC. For every product all items intersecting the bbox are listed; per tile the newest
    acquisition (datetime) is downloaded, or the newest one dated >= `after` (ISO date) when given."""
    d = margin_m / 111320.0
    bbox = [lon - d / 0.62, lat - d, lon + d / 0.62, lat + d]
    out = {}
    for prod in products:
        url, coll, asset_key = NI_STAC[prod]
        feats = _get_json(url, {"collections": [coll], "bbox": bbox, "limit": 200}).get("features", [])
        best = {}
        for f in feats:
            dt = (f.get("properties", {}).get("datetime") or "")[:10]
            if after and dt < after:
                continue
            assets = f["assets"]
            key = asset_key if asset_key in assets else next(k for k, v in assets.items() if "tiff" in (v.get("type") or ""))
            tile = re.sub(r"_(\d{4}-\d{2}-\d{2}|\d{4})$", "", f["id"])
            if tile not in best or dt > best[tile][0]:
                best[tile] = (dt, assets[key]["href"])
        paths = []
        for tile, (dt, href) in sorted(best.items()):
            paths.append(_curl(href, os.path.join(out_dir, "ni", prod, os.path.basename(href))))
        out[prod] = paths
        out[prod + "_dates"] = {os.path.basename(h): dt for dt, h in best.values()}
    return out


# ------------------------------------------------------------------------------------------------- Brandenburg
BB_BASE = "https://data.geobasis-bb.de/geobasis/daten"
BB_PRODUCTS = {"als": "als/laz/als_33{e}-{n}.zip", "dop20": "dop/rgbi_tif/dop_33{e}-{n}.zip",
               "bdom": "bdom/tif/bdom_33{e}-{n}.zip", "dgm1": "dgm/tif/dgm_33{e}-{n}.zip"}


def fetch_bb(lat: float, lon: float, out_dir: str, products=("dop20", "als", "dgm1"), margin_m: float = 300.0) -> dict:
    """Brandenburg: 1 km tiles in EPSG:25833, zipped; no acquisition date in the URL (read from the files)."""
    out = {}
    for prod in products:
        paths = []
        for e, n in tiles_for(lat, lon, 33, margin_m):
            rel = BB_PRODUCTS[prod].format(e=e, n=n)
            try:
                paths.append(_curl(f"{BB_BASE}/{rel}", os.path.join(out_dir, "bb", prod, os.path.basename(rel))))
            except RuntimeError:          # ALS coverage is partial in Brandenburg: missing tiles are skipped
                continue
        out[prod] = paths
    return out


# ---------------------------------------------------------------------------------- Mecklenburg-Vorpommern
MV_FEEDS = {"dop20": ("https://www.geodaten-mv.de/dienste/dop20_atom", "RGB", "TIFF"),
            "dom1": ("https://www.geodaten-mv.de/dienste/dom_atom", "DOM1", "XYZ"),
            "dgm1": ("https://www.geodaten-mv.de/dienste/dgm_atom", "DGM1", "XYZ")}
_ATOM = "{http://www.w3.org/2005/Atom}"


def _mv_dataset_links(prod: str) -> list:
    """[(file name, url, bbox)] of the dataset feed whose title matches the product and format."""
    service_url, want, fmt = MV_FEEDS[prod]
    root = ET.fromstring(_get_text(service_url))
    ds = [l.get("href") for l in root.iter(_ATOM + "link")
          if l.get("type") == "application/atom+xml" and l.get("rel") == "alternate" and want in (l.get("title") or "")]
    if not ds:
        raise RuntimeError(f"MV: no dataset feed for {prod}")
    links = []
    for href in ds:
        droot = ET.fromstring(_get_text(href.replace("&amp;", "&")))
        for l in droot.iter(_ATOM + "link"):
            if l.get("rel") == "section" and fmt in (l.get("title") or ""):
                m = re.search(r"file=([^&]+)", l.get("href"))
                links.append((m.group(1) if m else l.get("href"), l.get("href").replace("&amp;", "&"), l.get("bbox")))
        if links:
            break
    return links


def fetch_mv(lat: float, lon: float, out_dir: str, products=("dop20", "dom1", "dgm1"), margin_m: float = 300.0) -> dict:
    """MV: 2 km tiles in EPSG:25833 from the Atom dataset feeds; tile keys are even km values. Cache first: the feed
    is only read when a tile is missing locally (2026-10-22, the feed answered 503 for cached parks)."""
    import glob
    out = {}
    for prod in products:
        keys = [f"_33_{e * 2}_{n * 2}_2" for e, n in tiles_for(lat, lon, 33, margin_m, size_m=2000.0)]
        local = {k: sorted(f for f in glob.glob(os.path.join(out_dir, "mv", prod, f"*{k}*")) if not f.endswith(".part")) for k in keys}   # unfinished downloads do not count
        if all(local.values()):
            out[prod] = [v[0] for v in local.values()]
            continue
        links = {name: url for name, url, _ in _mv_dataset_links(prod)}
        paths = []
        for key in keys:
            cand = [nm for nm in links if key in nm]
            if not cand:
                continue
            paths.append(_curl(links[cand[0]], os.path.join(out_dir, "mv", prod, cand[0])))
        out[prod] = paths
    return out


# ------------------------------------------------------------------------------------------------------ Bayern
def fetch_by_laser(lat: float, lon: float, out_dir: str, margin_m: float = 300.0) -> dict:
    """Bayern laser point cloud, 1 km LAZ tiles in EPSG:25832 (~1 GB each)."""
    paths = []
    for e, n in tiles_for(lat, lon, 32, margin_m):
        paths.append(_curl(f"https://geodaten.bayern.de/odd_data/laser/{e}_{n}.laz", os.path.join(out_dir, "by", "laser", f"{e}_{n}.laz"), timeout=7200))
    return {"laser": paths}


FETCHERS = {"Niedersachsen": fetch_ni, "Brandenburg": fetch_bb, "Mecklenburg-Vorpommern": fetch_mv}


# ------------------------------------------------------------------------------------------ Sachsen-Anhalt
ST_DOP20 = "https://www.geodatenportal.sachsen-anhalt.de/gfds_webshare/sec-download/LVermGeo/DOP20/32{e:03d}{n:04d}.tif"
ST_DOM1_PARTS = ["https://geodatenportal.sachsen-anhalt.de/gfds_webshare/download/LVermGeo/Geodatenportal/Online-Bereitstellung-LVermGeo/DOM/DOM1_%d.zip" % i for i in (1, 2, 3, 4)]


def fetch_st_dop(lat: float, lon: float, out_dir: str, margin_m: float = 300.0) -> dict:
    """Sachsen-Anhalt DOP20 RGBI: 2 km tiles in EPSG:25832 named 32<E km><N km> (even km values), direct download
    (found via the INSPIRE OI coverage GML; the tile selector itself is browser-bound)."""
    paths = []
    for e, n in tiles_for(lat, lon, 32, margin_m, size_m=2000.0):
        try:
            paths.append(_curl(ST_DOP20.format(e=e * 2, n=n * 2), os.path.join(out_dir, "st", "dop20", f"dop20_32{e * 2:03d}{n * 2:04d}.tif"), ua="Mozilla/5.0 (X11; Linux x86_64) Firefox/128.0"))
        except RuntimeError as ex:                          # some tiles are missing on the server (404, 2026-10-22): the orthophoto only gives pitch0
            print("  ST DOP20 missing:", ex, flush=True)
    return {"dop20": paths}


class _RemoteFile:
    """Seekable read-only HTTP file (Range requests) so that zipfile can list a remote archive."""
    def __init__(self, url):
        self.url, self.pos = url, 0
        try:
            req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as r:
                self.size = int(r.headers["Content-Length"])
        except urllib.error.HTTPError as ex:              # HEAD not allowed (Hessen): size from the Content-Range of a 1-byte GET
            if ex.code != 405:
                raise
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", "Range": "bytes=0-0"})
            with urllib.request.urlopen(req, timeout=60) as r:
                self.size = int(r.headers["Content-Range"].split("/")[-1])
    def seek(self, off, whence=0):
        self.pos = (off, self.pos + off, self.size + off)[whence]; return self.pos
    def tell(self): return self.pos
    def read(self, n=-1):
        n = self.size - self.pos if n is None or n < 0 else min(n, self.size - self.pos)
        if n <= 0: return b""
        req = urllib.request.Request(self.url, headers={"User-Agent": "Mozilla/5.0", "Range": f"bytes={self.pos}-{self.pos + n - 1}"})
        with urllib.request.urlopen(req, timeout=300) as r:
            data = r.read()
        self.pos += len(data); return data
    def seekable(self): return True
    def readable(self): return True
    def writable(self): return False
    def flush(self): pass
    def close(self): pass


def st_dom1_part_for(tile_token: str) -> tuple:
    """Which statewide DOM1 zip part contains a tile whose name contains tile_token (e.g. '6065762')."""
    import zipfile
    for i, url in enumerate(ST_DOM1_PARTS, 1):
        with zipfile.ZipFile(_RemoteFile(url)) as z:
            names = [nm for nm in z.namelist() if tile_token in nm]
            if names:
                return i, names
    return None, []


def fetch_st_dom1(lat: float, lon: float, out_dir: str, margin_m: float = 300.0, product: str = "DOM1") -> dict:
    """Sachsen-Anhalt DOM1 (or DGM1 when the analogous statewide zips exist): extract the 2 km tiles covering the
    park from the statewide zips by HTTP range reads (zipfile over a seekable remote file), ~5-25 MB per tile
    instead of 12 GB per part. Tile names carry the acquisition year: dom1_32_<E>_<N>_2_st_<year>.tif."""
    import zipfile
    want = [f"_32_{e * 2}_{n * 2}_2_st" for e, n in tiles_for(lat, lon, 32, margin_m, size_m=2000.0)]
    parts = [u.replace("DOM/DOM1_", f"DGM/{product}_") if product != "DOM1" else u for u in ST_DOM1_PARTS]
    paths, dates = [], {}
    for url in parts:
        try:
            rf = _RemoteFile(url)
        except Exception:
            continue
        with zipfile.ZipFile(rf) as z:
            for nm in z.namelist():
                if any(w in nm for w in want) and nm.endswith(".tif"):
                    p = os.path.join(out_dir, "st", product.lower(), os.path.basename(nm))
                    if not os.path.exists(p):
                        os.makedirs(os.path.dirname(p), exist_ok=True)
                        with z.open(nm) as src, open(p + ".part", "wb") as dst:
                            dst.write(src.read())
                        os.replace(p + ".part", p)
                    paths.append(p); dates[os.path.basename(p)] = re.search(r"_st_(\d{4})", nm).group(1) if re.search(r"_st_(\d{4})", nm) else None
        if len(paths) >= len(want):
            break
    return {product.lower(): paths, product.lower() + "_dates": dates}


def fetch_st(lat: float, lon: float, out_dir: str, margin_m: float = 300.0) -> dict:
    """Sachsen-Anhalt complete (2026-10-22, decision E6): DOP20 tiles + DOM1 and DGM1 from the statewide zips."""
    out = fetch_st_dop(lat, lon, out_dir, margin_m)
    out.update(fetch_st_dom1(lat, lon, out_dir, margin_m, "DOM1"))
    out.update(fetch_st_dom1(lat, lon, out_dir, margin_m, "DGM1"))
    return out


def _extract_members(zip_path_or_url: str, want: list, dest: str, suffixes=(".tif",)) -> list:
    """Extract the members whose name contains one of the tokens in `want` and ends with one of `suffixes` from a
    local zip or (via HTTP range reads) a remote one; companion world files (.tfw/.jgw) are taken along."""
    import zipfile
    src = _RemoteFile(zip_path_or_url) if zip_path_or_url.startswith("http") else zip_path_or_url
    out = []
    with zipfile.ZipFile(src) as z:
        names = z.namelist()
        hit = [nm for nm in names if any(w in os.path.basename(nm) for w in want) and nm.lower().endswith(tuple(suffixes))]
        world = [nm for nm in names if os.path.splitext(nm)[0] in {os.path.splitext(h)[0] for h in hit} and nm.lower().endswith((".tfw", ".jgw", ".j2w", ".aux.xml", ".prj"))]
        for nm in hit + world:
            p = os.path.join(dest, os.path.basename(nm))
            if not os.path.exists(p):
                os.makedirs(dest, exist_ok=True)
                with z.open(nm) as s, open(p + ".part", "wb") as d:
                    d.write(s.read())
                os.replace(p + ".part", p)
            if nm in hit:
                out.append(p)
    return out


# ------------------------------------------------------------------------------------------ Nordrhein-Westfalen
NW_BASE = "https://www.opengeodata.nrw.de/produkte/geobasis/"
NW_PRODUCTS = {"laser": ("hm/3dm_l_las/3dm_l_las/", "3dm_32_{e}_{n}_1_nw"),          # LAZ 1 km, 4-10 points/m2 (3D-Messdaten)
               "dgm1": ("hm/dgm1_tiff/dgm1_tiff/", "dgm1_32_{e}_{n}_1_nw"),          # GeoTIFF, year in the name
               "dom1": ("hm/dom1_tiff/dom1_tiff/", "dom1_32_{e}_{n}_1_nw"),
               "dop20": ("lusat/akt/dop/dop_jp2_f10/", "dop10rgbi_32_{e}_{n}_1_nw")}  # DOP10 RGBI as JPEG2000 (key dop20 for the pipeline)


def _nw_listing(prod: str, out_dir: str, max_age_days: float = 30.0) -> list:
    """File names of one NRW product from the XML index of its directory (cached for max_age_days)."""
    import time
    path = os.path.join(out_dir, "nw", f"_index_{prod}.xml")
    if not os.path.exists(path) or time.time() - os.path.getmtime(path) > max_age_days * 86400:
        _curl(NW_BASE + NW_PRODUCTS[prod][0], path + ".new", timeout=600)
        os.replace(path + ".new", path)
    return [f.get("name") for f in ET.parse(path).getroot().iter("file")]


def fetch_nw(lat: float, lon: float, out_dir: str, products=("dop20", "laser", "dgm1", "dom1"), margin_m: float = 300.0) -> dict:
    """Nordrhein-Westfalen (opengeodata.nrw.de, dl-de/zero-2-0): directory indexes as XML, 1 km tiles in EPSG:25832,
    files named <prod>_32_<E km>_<N km>_1_nw[_<year>].<ext>; the newest file per tile is taken (the name carries
    the year for DGM1/DOM1/DOP; the point cloud year comes from the GPS time of the points)."""
    out = {}
    for prod in products:
        names = _nw_listing(prod, out_dir)
        paths = []
        for e, n in tiles_for(lat, lon, 32, margin_m):
            stem = NW_PRODUCTS[prod][1].format(e=e, n=n)
            cand = sorted(nm for nm in names if nm.startswith(stem + "_") or nm.startswith(stem + "."))
            if cand:
                paths.append(_curl(NW_BASE + NW_PRODUCTS[prod][0] + cand[-1], os.path.join(out_dir, "nw", prod, cand[-1])))
        out[prod] = paths
    return out


# ---------------------------------------------------------------------------------------------------- Hessen
HE_REST = "https://gds.hessen.de/INTERSHOP/rest/WFS/HLBG-Geodaten-Site/-/downloadcenter"
HE_HOST = "https://gds.hessen.de"
HE_PRODUCTS = {"dom1": ("3D-Daten/Digitales Oberflächenmodell (DOM1)", "_32_{e}_{n}_1_he", (".tif",)),
               "dgm1": ("3D-Daten/Digitales Geländemodell (DGM1)", "_32_{e}_{n}_1_he", (".tif",)),
               "dop20": ("Luftbildinformationen/Digitale Orthophotos DOP20", "_32_{e}_{n}_1_he", (".jpg", ".tif"))}


def _he_norm(s: str) -> str:
    s = re.sub(r"\s*-\s*(DOM1|DGM1|DOP20?)\s*$", "", s, flags=re.I)
    return re.sub(r"[\s_\-()]", "", s).lower().replace("ß", "ss")


def _he_list(path: str) -> dict:
    """{package name: download path} of one downloadcenter folder (all pages)."""
    import urllib.parse
    out, page = {}, 1
    while True:
        d = _get_json(f"{HE_REST}?path={urllib.parse.quote_plus(path)}&page={page}")
        sr = d["searchresult"]
        for x in sr.get("downloads", []):
            out[x["name"]] = x["downloadLink"]["uri"]
        if page * sr["paging"]["pageSize"] >= sr["paging"]["total"]:
            return out
        page += 1


def fetch_he(lat: float, lon: float, out_dir: str, landkreis: str, gemeinde: str = None, products=("dop20", "dom1", "dgm1"),
             margin_m: float = 300.0) -> dict:
    """Hessen (gds.hessen.de Downloadcenter, dl-de/zero-2-0): DOM1 (laser), DGM1 and DOP20 as zip packages per
    municipality inside a folder per district, 1 km tiles in EPSG:25832 (GeoTIFF/JPG + world file). The REST API
    behind the downloadcenter lists the packages; the tiles covering the park are read out of the packages by HTTP
    range requests (the park's municipality first, then the other packages of the district until all tiles are found).
    The point cloud is not in the downloadcenter (shop order only)."""
    import urllib.parse
    out = {}
    for prod in products:
        folder, tok, suf = HE_PRODUCTS[prod]
        want_keys = tiles_for(lat, lon, 32, margin_m)
        want = [tok.format(e=e, n=n) for e, n in want_keys]
        dest = os.path.join(out_dir, "he", prod)
        have = [p for p in (sorted(os.listdir(dest)) if os.path.isdir(dest) else []) if any(w in p for w in want) and p.lower().endswith(suf)]
        if len({w for w in want for p in have if w in p}) == len(want):
            out[prod] = [os.path.join(dest, p) for p in have]
            continue
        d = _get_json(f"{HE_REST}?path={urllib.parse.quote_plus(folder)}&navigation=all")
        kreise = []                                               # the navigation is a flat list; districts = one level below the product folder
        for nd in d["navigation"]:
            q = urllib.parse.parse_qs(urllib.parse.urlparse(nd.get("uri", "")).query).get("path", [""])[0]
            if q.startswith(folder + "/") and "/" not in q[len(folder) + 1:]:
                kreise.append(dict(name=q[len(folder) + 1:]))
        kname = next((c["name"] for c in kreise if _he_norm(landkreis) in _he_norm(c["name"]) or _he_norm(c["name"]).endswith(_he_norm(landkreis))), None)
        if kname is None:
            raise RuntimeError(f"Hessen {prod}: district {landkreis!r} not found among {[c['name'] for c in kreise]}")
        pk = _he_list(f"{folder}/{kname}")
        order = sorted(pk, key=lambda nm: 0 if gemeinde and _he_norm(nm) == _he_norm(gemeinde) else 1)
        found = []
        for nm in order:
            got = _extract_members(HE_HOST + urllib.parse.quote(pk[nm]), [w for w in want if not any(w in os.path.basename(f) for f in found)], dest, suf)
            found += got
            if len({w for w in want for p in found if w in os.path.basename(p)}) == len(want):
                break
        out[prod] = sorted(set(found))
    if "dom1" in out:                                             # acquisition dates per tile from the metadata list of the downloadcenter
        try:
            dates = _he_dates(out_dir)
            out["dom1_dates"] = {os.path.basename(p): dates.get(re.search(r"_32_(\d{3})_(\d{4})_", os.path.basename(p)).group(1) +
                                                                   re.search(r"_32_(\d{3})_(\d{4})_", os.path.basename(p)).group(2)) for p in out["dom1"]}
        except Exception as ex:
            print("  Hessen metadata list not read:", type(ex).__name__, str(ex)[:80], flush=True)
    return out


HE_META = "https://gds.hessen.de/INTERSHOP/static/WFS/HLBG-Geodaten-Site/-/HLBG-Geodaten/de_DE/Downloadcenter/Daten/3D-Daten/Metadaten_DGM-DOM_August-2026.xlsx"


def _he_dates(out_dir: str) -> dict:
    """{'<E km><N km>': 'YYYY-MM-DD'} flight date of the laser survey per 1 km tile (DGM1/DOM1 metadata list)."""
    import pandas as pd
    p = _curl(HE_META, os.path.join(out_dir, "he", "_metadaten_dgm_dom.xlsx"), timeout=600)
    x = pd.read_excel(p, header=1)
    return {str(k): str(v)[:10] for k, v in zip(x["Kachelname"], x["Bildflugdatum"])}


# ---------------------------------------------------------------------------------------------------- Sachsen
SN_LINKS = "https://geodienste.sachsen.de/ags-relay/ArcGISServer/guest/arcgis/rest/services/geosn/rest_geosn_downloadlinks/MapServer"
SN_LAYERS = {"laser": 1, "dom1": 4, "dgm1": 6, "dop20": 7}       # LSC, DOM1, DGM1, DOP_RGB


def fetch_sn(lat: float, lon: float, out_dir: str, products=("dop20", "laser", "dgm1", "dom1"), margin_m: float = 300.0) -> dict:
    """Sachsen (geodaten.sachsen.de, dl-de/by-2-0): 2 km tiles in EPSG:25833, tile number <E km><N km> (lower left,
    even km). The download links and the date per tile ("Stand") come from the ArcGIS service behind the Produktdownload
    app (rest_geosn_downloadlinks, one layer per product); the GeoCloud share ids in the links change (LSC on
    2026-10-22: rqcqdt8QMcLFUvC instead of the id on the batch page), so they are never hard-coded. Zips are kept as
    downloaded; the point cloud (LAZ inside the zip) is extracted next to it."""
    import urllib.parse
    out = {}
    for prod in products:
        paths, dates = [], {}
        for e, n in tiles_for(lat, lon, 33, margin_m, size_m=2000.0):
            tile = f"{e * 2}{n * 2}"
            q = urllib.parse.urlencode(dict(where=f"Kachel='{tile}'", outFields="Download,Stand", returnGeometry="false", f="json"))
            feats = _get_json(f"{SN_LINKS}/{SN_LAYERS[prod]}/query?{q}").get("features", [])
            if not feats:
                continue
            url, stand = feats[0]["attributes"]["Download"], feats[0]["attributes"].get("Stand")
            try:
                p = _curl(url, os.path.join(out_dir, "sn", prod, os.path.basename(url)))
            except RuntimeError as ex:                              # 2026-10-22: the LSC share answered 503 (NotFound) for every tile
                print("  Sachsen", prod, "not available:", ex, flush=True)
                continue
            if prod == "laser":
                got = _extract_members(p, [""], os.path.join(out_dir, "sn", prod), (".laz",))
                paths += got
                dates.update({os.path.basename(g): stand for g in got})
            else:
                paths.append(p)
                dates[os.path.basename(p)] = stand
        out[prod] = paths
        out[prod + "_dates"] = dates
    return out


# -------------------------------------------------------------------------------------------------- Thüringen
def fetch_th(lat: float, lon: float, out_dir: str, products=("dop20", "laser", "dgm1"), margin_m: float = 300.0) -> dict:
    """Thüringen (TLBG, CC BY 4.0; 1 km tiles in EPSG:25832, LAZ, DGM1, DOP20): on 2026-10-22 the old download
    services (geoportal.geoportal-th.de Atom feeds and gaialight download apps) answered with a maintenance page and
    the new portal geoportal.thueringen.de sits behind a bot protection, so nothing is fetched here. The tiles are
    downloaded by hand and placed into <out_dir>/th/{laser,dgm1,dop20}/ (unzipped or zipped); this function picks the
    files whose name contains _<E km>_<N km>_ of the needed tiles and raises with the list of the missing ones."""
    out, missing = {}, []
    for prod in products:
        dest = os.path.join(out_dir, "th", prod)
        files = sorted(os.listdir(dest)) if os.path.isdir(dest) else []
        paths = []
        for e, n in tiles_for(lat, lon, 32, margin_m):
            hit = [f for f in files if re.search(rf"_{e}_{n}(_|\.)", f) or f"_32{e}_{n}" in f or f"_32_{e}_{n}" in f]
            if not hit:
                missing.append(f"{prod} {e}_{n}")
                continue
            for f in hit:
                p = os.path.join(dest, f)
                if f.lower().endswith(".zip") and prod == "laser":
                    paths += _extract_members(p, [""], dest, (".laz", ".las"))
                else:
                    paths.append(p)
        out[prod] = sorted(set(paths))
    if missing:
        raise RuntimeError("Thüringen: tiles missing in the cache (download by hand): " + ", ".join(missing))
    return out
