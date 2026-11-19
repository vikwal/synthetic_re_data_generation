#!/usr/bin/env python3
"""WP0.1 — ERA5 single-level download of the round-2 variables.

Downloads sshf, blh, gwd (hourly, Germany box) as monthly NetCDF files,
2023-01 .. 2026-06, to /mnt/nvme2/synthetic/raw/round2/era5_nc/.

Modeled on /mnt/nas/era5/requestalbedo.py: resumable via completeness check,
never overwrites a valid file, logs incomplete months. Up to 4 concurrent
CDS requests (fair-use limit).

Run: synthre/bin/python scripts/round2/wp0_cds_download.py [--start 2023-01] [--end 2026-06]
"""

import os
import sys
import json
import time
import logging
import argparse
import calendar
import datetime as dt
from concurrent.futures import ThreadPoolExecutor, as_completed

import zipfile

import cdsapi
import netCDF4
import xarray as xr

OUT_DIR = "/mnt/nvme2/synthetic/raw/round2/era5_nc"
LOG_PATH = "/mnt/nvme2/synthetic/raw/round2/era5_nc/download.log"
MANIFEST = "/mnt/nvme2/synthetic/raw/round2/era5_nc/manifest.json"

DATASET = "reanalysis-era5-single-levels"
# N, W, S, E — Germany + margin
AREA = [56, 5, 47, 16]
VARIABLES = [
    "surface_sensible_heat_flux",   # sshf, J/m2 accumulated, positive downward
    "boundary_layer_height",        # blh, m, instantaneous
    "gravity_wave_dissipation",     # gwd, J/m2 accumulated (optional feature)
]
MAX_WORKERS = 4
# ERA5T lag is ~5 days; the last requested month may be partial.
ERA5T_SAFE_LAG_DAYS = 6

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler()],
)
log = logging.getLogger("wp0_cds")


def month_range(start: str, end: str):
    y, m = map(int, start.split("-"))
    ye, me = map(int, end.split("-"))
    while (y, m) <= (ye, me):
        yield y, m
        m += 1
        if m > 12:
            m, y = 1, y + 1


def expected_hours(year: int, month: int) -> int:
    ndays = calendar.monthrange(year, month)[1]
    last_full = (dt.date.today() - dt.timedelta(days=ERA5T_SAFE_LAG_DAYS))
    first = dt.date(year, month, 1)
    if first > last_full:
        return 0
    if dt.date(year, month, ndays) > last_full:
        ndays = (last_full - first).days + 1
    return ndays * 24


def normalize(path: str) -> None:
    """CDS returns a zip when accumulated (sshf, gwd) and instantaneous (blh)
    variables are requested together — merge the members into one NetCDF."""
    if not (os.path.exists(path) and zipfile.is_zipfile(path)):
        return
    tmpdir = path + ".extract"
    os.makedirs(tmpdir, exist_ok=True)
    with zipfile.ZipFile(path) as z:
        members = [n for n in z.namelist() if n.endswith(".nc")]
        z.extractall(tmpdir, members)
    parts = [xr.open_dataset(os.path.join(tmpdir, m)) for m in members]
    merged = xr.merge(parts, compat="override", join="outer")
    merged.to_netcdf(path + ".merged")
    for p in parts:
        p.close()
    for m in members:
        os.remove(os.path.join(tmpdir, m))
    os.rmdir(tmpdir)
    os.replace(path + ".merged", path)


def is_complete(path: str, year: int, month: int) -> bool:
    if not os.path.exists(path):
        return False
    try:
        normalize(path)
        with netCDF4.Dataset(path) as ds:
            tdim = None
            for name in ("valid_time", "time"):
                if name in ds.dimensions:
                    tdim = len(ds.dimensions[name])
                    break
            if tdim is None:
                return False
            varnames = set(ds.variables)
            have = {"sshf", "blh", "gwd"} & varnames
            exp = expected_hours(year, month)
            return exp > 0 and tdim >= exp and len(have) >= 2
    except Exception as e:
        log.warning("unreadable %s: %s", path, e)
        return False


def fetch_month(year: int, month: int, variables) -> tuple:
    fname = f"era5_r2_{year}_{month:02d}.nc"
    path = os.path.join(OUT_DIR, fname)
    if is_complete(path, year, month):
        return (year, month, "cached")
    if expected_hours(year, month) == 0:
        return (year, month, "not_yet_available")
    ndays = calendar.monthrange(year, month)[1]
    req = {
        "product_type": ["reanalysis"],
        "variable": variables,
        "year": [str(year)],
        "month": [f"{month:02d}"],
        "day": [f"{d:02d}" for d in range(1, ndays + 1)],
        "time": [f"{h:02d}:00" for h in range(24)],
        "area": AREA,
        "data_format": "netcdf",
        "download_format": "unarchived",
    }
    tmp = path + ".part"
    client = cdsapi.Client(quiet=True, wait_until_complete=True)
    t0 = time.time()
    client.retrieve(DATASET, req, tmp)
    os.replace(tmp, path)
    ok = is_complete(path, year, month)
    log.info("%s done in %.0fs complete=%s", fname, time.time() - t0, ok)
    return (year, month, "ok" if ok else "incomplete")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2023-01")
    ap.add_argument("--end", default="2026-06")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    months = list(month_range(args.start, args.end))
    log.info("submitting %d months, vars=%s", len(months), VARIABLES)

    # Probe: first request decides whether gwd is accepted and the licence works.
    variables = list(VARIABLES)
    y0, m0 = months[0]
    try:
        res = fetch_month(y0, m0, variables)
        log.info("probe %s-%02d -> %s", y0, m0, res[2])
    except Exception as e:
        msg = str(e)
        if "gravity_wave_dissipation" in msg or "gwd" in msg:
            log.warning("gwd rejected, dropping it: %s", msg)
            variables = [v for v in variables if v != "gravity_wave_dissipation"]
            fetch_month(y0, m0, variables)
        elif "licence" in msg.lower() or "license" in msg.lower() or "401" in msg or "403" in msg:
            log.error("BLOCKER licence/auth: %s", msg)
            sys.exit(2)
        else:
            raise

    results = {}
    failed = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(fetch_month, y, m, variables): (y, m) for y, m in months[1:]}
        for fut in as_completed(futs):
            y, m = futs[fut]
            try:
                results[f"{y}-{m:02d}"] = fut.result()[2]
            except Exception as e:
                log.error("FAILED %s-%02d: %s", y, m, e)
                failed.append((y, m))
                results[f"{y}-{m:02d}"] = f"error: {e}"

    # one retry round for transient failures
    for y, m in failed:
        try:
            results[f"{y}-{m:02d}"] = fetch_month(y, m, variables)[2]
        except Exception as e:
            log.error("RETRY FAILED %s-%02d: %s", y, m, e)

    results[f"{y0}-{m0:02d}"] = "ok"
    with open(MANIFEST, "w") as f:
        json.dump({"variables": variables, "area": AREA, "months": results,
                   "finished": dt.datetime.now().isoformat()}, f, indent=2)
    bad = {k: v for k, v in results.items() if v not in ("ok", "cached", "not_yet_available")}
    log.info("DONE. %d/%d months ok, problems: %s", len(results) - len(bad), len(results), bad or "none")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
