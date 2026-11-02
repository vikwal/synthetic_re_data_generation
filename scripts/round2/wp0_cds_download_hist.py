#!/usr/bin/env python3
"""Round 3 — ERA5 single-level download of the FULL chain variable set for
the years before the station CSVs begin (2020-01 .. 2023-06), Germany box,
hourly, one NetCDF per month, for the longitudinal aging test on the real
parks' 2020/2021-2024 meter data.

Variables (CDS names -> NetCDF short names as used by wind_era5_v2):
  10m/100m u,v wind (u10 v10 u100 v100), 10m wind gust since previous
  post-processing (fg10 -> wind_gust_10m), friction velocity (zust ->
  friction_wind), 2m temperature (t2m -> temp_2m), surface pressure (sp ->
  pressure), 2m dewpoint (d2m -> dew_point_2m), surface sensible heat flux
  (sshf), boundary layer height (blh), gravity wave dissipation (gwd).

Modeled on wp0_cds_download.py: resumable via completeness check, never
overwrites a valid file, mixed accumulated/instantaneous zip merged into
one NetCDF. Two concurrent CDS requests.

Run on l2: synthre/bin/python scripts/round2/wp0_cds_download_hist.py [--start 2020-01] [--end 2023-06]
Output: /mnt/nvme2/synthetic/raw/round2/era5_nc_hist/era5_hist_YYYY_MM.nc
"""

import os
import sys
import json
import time
import logging
import argparse
import calendar
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

import cdsapi
import netCDF4
import xarray as xr

OUT_DIR = "/mnt/nvme2/synthetic/raw/round2/era5_nc_hist"
LOG_PATH = os.path.join(OUT_DIR, "download.log")
MANIFEST = os.path.join(OUT_DIR, "manifest.json")

DATASET = "reanalysis-era5-single-levels"
AREA = [56, 5, 47, 16]  # N, W, S, E — Germany + margin (as wp0)
VARIABLES = [
    "10m_u_component_of_wind", "10m_v_component_of_wind",
    "100m_u_component_of_wind", "100m_v_component_of_wind",
    "10m_wind_gust_since_previous_post_processing",
    "friction_velocity",
    "2m_temperature", "surface_pressure", "2m_dewpoint_temperature",
    "surface_sensible_heat_flux", "boundary_layer_height",
    "gravity_wave_dissipation",
]
EXPECTED_SHORT = {"u10", "v10", "u100", "v100", "fg10", "zust", "t2m", "sp",
                  "d2m", "sshf", "blh", "gwd"}
MAX_WORKERS = 2

os.makedirs(OUT_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler()],
)
log = logging.getLogger("wp0_cds_hist")


def month_range(start: str, end: str):
    y, m = map(int, start.split("-"))
    ye, me = map(int, end.split("-"))
    while (y, m) <= (ye, me):
        yield y, m
        m += 1
        if m > 12:
            m, y = 1, y + 1


def expected_hours(year: int, month: int) -> int:
    return calendar.monthrange(year, month)[1] * 24


def normalize(path: str) -> None:
    """CDS returns a zip when accumulated (sshf, gwd, fg10) and instantaneous
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
            have = EXPECTED_SHORT & set(ds.variables)
            missing = EXPECTED_SHORT - have
            if missing:
                log.warning("%s lacks %s", os.path.basename(path), sorted(missing))
            # accumulated fields carry one extra step at the month edges
            return tdim >= expected_hours(year, month) and len(have) >= 11
    except Exception as e:
        log.warning("unreadable %s: %s", path, e)
        return False


def fetch_month(year: int, month: int) -> tuple:
    fname = f"era5_hist_{year}_{month:02d}.nc"
    path = os.path.join(OUT_DIR, fname)
    if is_complete(path, year, month):
        return (year, month, "cached")
    ndays = calendar.monthrange(year, month)[1]
    req = {
        "product_type": ["reanalysis"],
        "variable": VARIABLES,
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
    for attempt in range(3):
        try:
            client.retrieve(DATASET, req, tmp)
            break
        except Exception as e:  # transient CDS errors: retry after a pause
            log.warning("%s attempt %d failed: %s", fname, attempt + 1, e)
            time.sleep(120)
    else:
        return (year, month, "failed")
    os.replace(tmp, path)
    ok = is_complete(path, year, month)
    log.info("%s done in %.0fs complete=%s", fname, time.time() - t0, ok)
    return (year, month, "ok" if ok else "incomplete")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2020-01")
    ap.add_argument("--end", default="2023-06")
    args = ap.parse_args()
    months = list(month_range(args.start, args.end))
    log.info("requesting %d months %s..%s into %s", len(months), args.start, args.end, OUT_DIR)
    results = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(fetch_month, y, m): (y, m) for y, m in months}
        for f in as_completed(futs):
            y, m = futs[f]
            try:
                results[f"{y}-{m:02d}"] = f.result()[2]
            except Exception as e:
                log.error("%d-%02d crashed: %s", y, m, e)
                results[f"{y}-{m:02d}"] = "crashed"
            with open(MANIFEST, "w") as fh:
                json.dump(results, fh, indent=1, sort_keys=True)
    bad = {k: v for k, v in results.items() if v not in ("ok", "cached")}
    log.info("DOWNLOAD_DONE ok=%d problems=%s", len(results) - len(bad), bad or "none")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
