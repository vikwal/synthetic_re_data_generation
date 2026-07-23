#!/usr/bin/env python3
"""WP0.1b — extend the per-station ERA5 CSVs with sshf, blh, gwd.

Reads the monthly Germany NetCDFs (wp0_cds_download.py), picks the nearest
0.25 deg grid point per station, and writes the existing CSV schema plus the
three new columns to /mnt/nvme2/synthetic/raw/wind_era5_v2/.

The chain drives parks by their DWD station id, so extending the 201 station
CSVs covers stations, parks and the ML sites alike.
"""

import glob
import os

import numpy as np
import pandas as pd
import xarray as xr

NC_DIR = "/mnt/nvme2/synthetic/raw/round2/era5_nc"
SRC_DIRS = ["/mnt/nas/synthetic/raw/wind_era5",
            "/mnt/nas/synthetic/raw/wind_real_era5"]
OUT_DIR = "/mnt/nvme2/synthetic/raw/wind_era5_v2"
STATIONS = os.path.join(os.path.dirname(__file__), "..", "..", "..",
                        "stations_master.csv")
NEW_VARS = ["sshf", "blh", "gwd"]


def load_new_vars() -> xr.Dataset:
    files = sorted(glob.glob(os.path.join(NC_DIR, "era5_r2_*.nc")))
    assert files, "no era5_r2 NetCDFs yet"
    ds = xr.open_mfdataset(files, combine="by_coords")
    have = [v for v in NEW_VARS if v in ds]
    print(f"months: {len(files)} | vars: {have} | "
          f"time: {str(ds.valid_time.values[0])[:16]} .. {str(ds.valid_time.values[-1])[:16]}")
    return ds[have]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    stations = pd.read_csv(STATIONS, dtype={"station_id": str}).set_index("station_id")
    ds = load_new_vars()

    src_files = {}
    for d in SRC_DIRS:
        for f in glob.glob(os.path.join(d, "Station_*.csv")):
            sid = os.path.basename(f)[len("Station_"):-len(".csv")]
            src_files.setdefault(sid, f)

    done, skipped = 0, []
    for sid, path in sorted(src_files.items()):
        out_path = os.path.join(OUT_DIR, f"Station_{sid}.csv")
        if sid not in stations.index:
            skipped.append(sid)
            continue
        lat = float(stations.loc[sid, "latitude"])
        lon = float(stations.loc[sid, "longitude"])
        df = pd.read_csv(path)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df.set_index("timestamp", inplace=True)
        point = ds.sel(latitude=lat, longitude=lon, method="nearest")
        new = point.to_dataframe()[[v for v in NEW_VARS if v in ds]]
        new.index = pd.to_datetime(new.index, utc=True)
        merged = df.join(new, how="left")
        n_missing = int(merged["sshf"].isna().sum()) if "sshf" in merged else -1
        merged.to_csv(out_path)
        done += 1
        if done % 25 == 0:
            print(f"{done}/{len(src_files)} done (last: {sid}, sshf NaN hours: {n_missing})")
    print(f"written: {done}, skipped (no coords): {skipped}")

    # consistency spot-check: merged wind columns identical to source
    sid = sorted(src_files)[0]
    a = pd.read_csv(src_files[sid])
    b = pd.read_csv(os.path.join(OUT_DIR, f"Station_{sid}.csv"))
    assert np.allclose(a["u_wind_10m"], b["u_wind_10m"], equal_nan=True)
    print("consistency check OK")


if __name__ == "__main__":
    main()
