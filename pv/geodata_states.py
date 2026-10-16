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
  Sachsen-Anhalt  tile download and WCS are browser-bound (403/500 for scripted access); not implemented here.
Licences: attribution required (GeoBasis-DE / <Landesamt>, year)."""
import json
import os
import re
import subprocess
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
    """MV: 2 km tiles in EPSG:25833 from the Atom dataset feeds; tile keys are even km values."""
    out = {}
    for prod in products:
        links = {name: url for name, url, _ in _mv_dataset_links(prod)}
        paths = []
        for e, n in tiles_for(lat, lon, 33, margin_m, size_m=2000.0):
            key = f"_33_{e * 2}_{n * 2}_2"
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
        paths.append(_curl(ST_DOP20.format(e=e * 2, n=n * 2), os.path.join(out_dir, "st", "dop20", f"dop20_32{e * 2:03d}{n * 2:04d}.tif"), ua="Mozilla/5.0 (X11; Linux x86_64) Firefox/128.0"))
    return {"dop20": paths}


class _RemoteFile:
    """Seekable read-only HTTP file (Range requests) so that zipfile can list a remote archive."""
    def __init__(self, url):
        self.url, self.pos = url, 0
        req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            self.size = int(r.headers["Content-Length"])
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
