#!/usr/bin/env python3
"""Round 3 — station-point ERA5 series 2020-01 .. 2026-04 for the real
parks' DWD reference stations, for the longitudinal aging test.

Reads the monthly Germany NetCDFs of wp0_cds_download_hist.py (2020-01 ..
2023-06, all chain variables), picks the nearest 0.25 deg grid point per
station (same rule as wp0_extract_points.py), writes the wind_era5_v2 CSV
schema and appends the existing v2 rows (2023-07 onwards) so each station
gets one continuous file in /mnt/nvme2/synthetic/raw/wind_era5_v3/.
The generator is pointed at it through round2.era5_dir_v2.

Only the stations of the 13 validation parks by default (park id = station
id, two parks may share one station); --all extends to every v2 station.

Run on l2: synthre/bin/python scripts/round2/wp0_extract_points_hist.py [--all]
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
import xarray as xr

NC_DIR = "/mnt/nvme2/synthetic/raw/round2/era5_nc_hist"
V2_DIR = "/mnt/nvme2/synthetic/raw/wind_era5_v2"
OUT_DIR = "/mnt/nvme2/synthetic/raw/wind_era5_v3"
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
STATIONS = os.path.join(REPO, "..", "stations_master.csv")  # ~/Work/stations_master.csv, as wp0_extract_points.py

PARK_STATIONS = ["00198", "00282", "00298", "01200", "01303", "02483",
                 "04745", "05347", "05426", "07374"]

# NetCDF short name -> v2 CSV column (order = v2 schema)
COLS = [("u10", "u_wind_10m"), ("v10", "v_wind_10m"),
        ("u100", "u_wind_100m"), ("v100", "v_wind_100m"),
        ("fg10", "wind_gust_10m"), ("zust", "friction_wind"),
        ("t2m", "temp_2m"), ("sp", "pressure"), ("d2m", "dew_point_2m"),
        ("sshf", "sshf"), ("blh", "blh"), ("gwd", "gwd")]


def load_hist() -> xr.Dataset:
    files = sorted(glob.glob(os.path.join(NC_DIR, "era5_hist_*.nc")))
    assert files, "no era5_hist NetCDFs yet"
    ds = xr.open_mfdataset(files, combine="by_coords")
    tname = "valid_time" if "valid_time" in ds.coords else "time"
    print(f"months: {len(files)} | vars: {sorted(ds.data_vars)} | time: "
          f"{str(ds[tname].values[0])[:16]} .. {str(ds[tname].values[-1])[:16]}")
    return ds.rename({tname: "time"}) if tname != "time" else ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="all v2 stations, not only the park stations")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    stations = pd.read_csv(STATIONS, dtype={"station_id": str}).set_index("station_id")
    ids = (sorted(os.path.basename(f)[8:-4] for f in glob.glob(os.path.join(V2_DIR, "Station_*.csv")))
           if args.all else PARK_STATIONS)
    ds = load_hist()
    have = [(nc, col) for nc, col in COLS if nc in ds]
    missing = [col for nc, col in COLS if nc not in ds]
    if missing:
        print(f"WARNING: variables missing in the hist NetCDFs: {missing}")
    for sid in ids:
        v2_path = os.path.join(V2_DIR, f"Station_{sid}.csv")
        if not os.path.exists(v2_path) or sid not in stations.index:
            print(f"skip {sid}: no v2 file or no coordinates")
            continue
        lat, lon = float(stations.loc[sid, "latitude"]), float(stations.loc[sid, "longitude"])
        point = ds.sel(latitude=lat, longitude=lon, method="nearest")
        hist = point.to_dataframe()[[nc for nc, _ in have]].rename(columns=dict(have))
        hist.index = pd.to_datetime(hist.index, utc=True)
        hist = hist[~hist.index.duplicated(keep="first")].sort_index()
        v2 = pd.read_csv(v2_path)
        v2["timestamp"] = pd.to_datetime(v2["timestamp"], utc=True)
        v2 = v2.set_index("timestamp")
        hist = hist.loc[: v2.index.min() - pd.Timedelta(hours=1)]
        out = pd.concat([hist.reindex(columns=v2.columns), v2]).sort_index()
        out.index.name = "timestamp"
        # continuity check: no gap at the seam, hourly throughout
        idx = out.index
        gaps = int((idx[1:] - idx[:-1] != pd.Timedelta(hours=1)).sum())
        out.to_csv(os.path.join(OUT_DIR, f"Station_{sid}.csv"))
        print(f"{sid}: {idx.min():%Y-%m-%d} .. {idx.max():%Y-%m-%d}, {len(out)} rows, "
              f"hist {len(hist)} + v2 {len(v2)}, non-hourly steps {gaps}, "
              f"NaN u10 {int(out['u_wind_10m'].isna().sum())}")
    print("done")


if __name__ == "__main__":
    main()
