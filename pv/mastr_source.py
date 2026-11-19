"""MaStR source of the generator: find the parsed register (parquets), else parse an unpacked export, else
download the current full export from the BNetzA and parse it. No automatic re-download: once parquets exist,
they are used until a newer export is requested explicitly (refresh=True).

Parsed files (settings mastr_dir): solar_units_<date>.parquet, solar_eeg_<date>.parquet, solar_parks_<date>.parquet.
Date selection (settings mastr_date): a fixed tag such as 20250625 (release parks_pv_v4) or "auto" = newest
complete set in mastr_dir. The BNetzA only serves the daily export of the day before and quarterly reference
exports, so an old fixed date can only come from an archived copy, never from a download.

Data licence: Datenlizenz Deutschland - Namensnennung - Version 2.0, source "Marktstammdatenregister, Bundesnetzagentur".
"""
import glob
import os
import re
import shutil
import time
import urllib.request
import warnings
import zipfile

import settings

PAGE_URL = "https://www.marktstammdatenregister.de/MaStR/Datendownload"
HOST = "https://download.marktstammdatenregister.de"
USER_AGENT = "re-data-generator/1.0"
KINDS = ("units", "eeg", "parks")
# members of the export the parser reads (pv.mastr.parse_register); the rest (~30 GB unpacked) is skipped
NEEDED = re.compile(r"^(EinheitenSolar_\d+|AnlagenEegSolar_\d+|Katalogwerte)\.xml$")
_DAILY = re.compile(r'href="(https://download\.marktstammdatenregister\.de/Gesamtdatenexport_(\d{8})_[\d.]+\.zip)"')


def parquet_paths(date: str, mastr_dir: str = None) -> dict:
    d = mastr_dir or settings.get("mastr_dir")
    return {k: os.path.join(d, f"solar_{k}_{date}.parquet") for k in KINDS}


def available_dates(mastr_dir: str = None) -> list:
    """Date tags with a complete set of parquets, oldest first."""
    d = mastr_dir or settings.get("mastr_dir")
    dates = {re.search(r"_(\d{8})\.parquet$", f).group(1) for f in glob.glob(os.path.join(d, "solar_units_*.parquet"))
             if re.search(r"_(\d{8})\.parquet$", f)}
    return sorted(x for x in dates if all(os.path.exists(p) for p in parquet_paths(x, d).values()))


def resolve_date(date: str = None, mastr_dir: str = None):
    """The date tag in use: the fixed setting, or for "auto" the newest complete set (None if there is none)."""
    date = date or settings.get("mastr_date")
    if date != "auto":
        return date
    have = available_dates(mastr_dir)
    return have[-1] if have else None


def export_date(export_dir: str) -> str:
    """Date tag of an unpacked export from its folder name (Marktstammdatenregister_<date>_... or Gesamtdatenexport_<date>_...)."""
    m = re.search(r"_(\d{8})(_|$)", os.path.basename(export_dir.rstrip("/")))
    if not m:
        raise ValueError(f"no date tag YYYYMMDD in the export folder name {export_dir}")
    return m.group(1)


def has_export(export_dir: str) -> bool:
    return (os.path.isfile(os.path.join(export_dir, "Katalogwerte.xml"))
            and bool(glob.glob(os.path.join(export_dir, "EinheitenSolar_*.xml")))
            and bool(glob.glob(os.path.join(export_dir, "AnlagenEegSolar_*.xml"))))


def current_export_url() -> tuple:
    """(url, date) of the daily full export, read from the BNetzA download page."""
    req = urllib.request.Request(PAGE_URL, headers={"User-Agent": USER_AGENT})
    html = urllib.request.urlopen(req, timeout=60).read().decode("utf-8", "replace")
    hits = _DAILY.findall(html)
    if not hits:
        raise RuntimeError(f"no daily Gesamtdatenexport link found on {PAGE_URL} (page layout changed?)")
    url, date = max(hits, key=lambda h: h[1])
    return url, date


def download(url: str, dest: str, chunk: int = 1 << 22) -> str:
    """Download url to dest with resume (dest.part); returns dest. ~3 GB for the full export."""
    if os.path.exists(dest):
        return dest
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    part = dest + ".part"
    have = os.path.getsize(part) if os.path.exists(part) else 0
    headers = {"User-Agent": USER_AGENT}
    if have:
        headers["Range"] = f"bytes={have}-"
    resp = urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120)
    if have and resp.status != 206:              # server ignored the range: start over
        have = 0
    total = have + int(resp.headers.get("Content-Length", 0))
    t0, last = time.time(), 0.0
    with open(part, "ab" if have else "wb") as f:
        while True:
            buf = resp.read(chunk)
            if not buf:
                break
            f.write(buf)
            have += len(buf)
            if time.time() - last > 30:
                last = time.time()
                print(f"  {have / 2**30:.2f} / {total / 2**30:.2f} GiB ({have / max(last - t0, 1e-9) / 2**20:.0f} MiB/s)", flush=True)
    if total and have != total:
        raise IOError(f"incomplete download {have} of {total} bytes; run again to resume ({part})")
    os.replace(part, dest)
    return dest


def extract_needed(zip_path: str, export_dir: str) -> str:
    """Unpack only the files the parser reads into export_dir."""
    os.makedirs(export_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        names = [n for n in z.namelist() if NEEDED.match(os.path.basename(n))]
        if not names:
            raise RuntimeError(f"{zip_path}: no EinheitenSolar/AnlagenEegSolar/Katalogwerte members")
        for n in names:
            target = os.path.join(export_dir, os.path.basename(n))
            if not os.path.exists(target):
                with z.open(n) as src, open(target + ".tmp", "wb") as dst:
                    shutil.copyfileobj(src, dst, 1 << 22)
                os.replace(target + ".tmp", target)
    return export_dir


def parse(export_dir: str, mastr_dir: str = None, min_kw: float = 100.0, workers: int = 12) -> str:
    """Parse an unpacked export into the three parquets; returns the date tag."""
    from pv import mastr
    date = export_date(export_dir)
    out = parquet_paths(date, mastr_dir)
    os.makedirs(os.path.dirname(out["units"]), exist_ok=True)
    relabelled = mastr.check_catalogue(mastr.load_catalogue(export_dir))
    if relabelled:
        print(f"MaStR {date}: {len(relabelled)} catalogue labels differ from the canonical ones, canonical used: {relabelled}", flush=True)
    t0 = time.time()
    units, eeg = mastr.parse_register(export_dir, min_kw, workers)
    print(f"parsed {len(units)} units, {len(eeg)} EEG plants in {time.time() - t0:.0f} s", flush=True)
    units.to_parquet(out["units"], index=False)
    eeg.to_parquet(out["eeg"], index=False)
    mastr.aggregate_parks(units, eeg).to_parquet(out["parks"], index=False)
    print(f"MaStR {date} -> {os.path.dirname(out['units'])} ({time.time() - t0:.0f} s)", flush=True)
    return date


def ensure(date: str = None, mastr_dir: str = None, export_dir: str = None, download_dir: str = None,
           allow_download: bool = True, refresh: bool = False, keep_zip: bool = False, workers: int = 12) -> str:
    """Make sure parsed MaStR parquets exist and return their date tag.
    1. parquets of the requested date (or, for "auto", of any date) present -> use them
    2. unpacked export of that date in export_dir -> parse
    3. otherwise (only for "auto" or a date equal to today's export, and allow_download) -> download, unpack, parse
    refresh=True skips step 1 for "auto" and fetches the current export even if older parquets exist."""
    mastr_dir = mastr_dir or settings.get("mastr_dir")
    want = date or settings.get("mastr_date")
    have = resolve_date(want, mastr_dir)
    if have and not refresh and all(os.path.exists(p) for p in parquet_paths(have, mastr_dir).values()):
        return have
    export_dir = export_dir or settings.get("mastr_export_dir")
    if has_export(export_dir) and (want == "auto" or export_date(export_dir) == want) and not refresh:
        return parse(export_dir, mastr_dir, workers=workers)
    if not allow_download:
        raise FileNotFoundError(f"MaStR {want}: no parquets in {mastr_dir} and no unpacked export in {export_dir}; "
                                f"download disabled")
    url, cur = current_export_url()
    if want not in ("auto", cur):
        raise FileNotFoundError(f"MaStR {want} not in {mastr_dir} and not downloadable (BNetzA serves only {cur}); "
                                f"provide the archived parquets or set mastr_date: auto")
    if have and have == cur and not refresh:
        return have
    warnings.warn(f"MaStR: downloading the full export {cur} (~3 GB) from {HOST}", stacklevel=2)
    download_dir = download_dir or os.path.join(os.path.dirname(mastr_dir.rstrip("/")), "mastr_download")
    zip_path = download(url, os.path.join(download_dir, os.path.basename(url)))
    target = os.path.join(download_dir, f"Gesamtdatenexport_{cur}")
    extract_needed(zip_path, target)
    date = parse(target, mastr_dir, workers=workers)
    shutil.rmtree(target)                         # parquets are the product; the XML is re-creatable from the zip
    if not keep_zip:
        os.remove(zip_path)
    return date
