#!/usr/bin/env python3
"""Extract nearest ICON-D2 09Z grid-point file per M4 (site_v2) station.

Run on l2 inside the synthre venv:
    source ~/Work/synthetic_re_data_generation/synthre/bin/activate
    python3 extract_icon_d2_m4.py
"""
import os
import re
import shutil
import sys

import pandas as pd
from geopy.distance import geodesic

WIND_PARAMETER_CSV = "/mnt/nvme2/synthetic/wind/site_v2_20260714/wind_parameter.csv"
ICON_ROOT = "/mnt/lambda1/nvme1/icon-d2/parquet/ML/09"
OUT_DIR = "/mnt/nvme2/synthetic/wind/mendeley_M4_icon_d2_09z"
TARGET_DATE = pd.Timestamp("2026-04-30", tz="UTC")

FNAME_RE = re.compile(r"^(\d+)_(\d+)_(\d+)_(\d+)_ML\.parquet$")


def parse_coord(fname):
    m = FNAME_RE.match(fname)
    if not m:
        return None
    lat_i, lat_d, lon_i, lon_d = m.groups()
    lat = int(lat_i) + int(lat_d) / 10 ** len(lat_d)
    lon = int(lon_i) + int(lon_d) / 10 ** len(lon_d)
    return lat, lon


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    wp = pd.read_csv(WIND_PARAMETER_CSV, sep=";", dtype={"park_id": str})
    wp["park_id"] = wp["park_id"].str.zfill(5)
    print(f"sites in wind_parameter.csv: {len(wp)}", file=sys.stderr)

    rows = []
    for _, site in wp.iterrows():
        station_id = site["park_id"]
        site_lat = float(site["latitude"])
        site_lon = float(site["longitude"])
        station_dir = os.path.join(ICON_ROOT, station_id)

        candidates = []
        for fname in os.listdir(station_dir):
            coord = parse_coord(fname)
            if coord is None:
                continue
            dist_km = geodesic((site_lat, site_lon), coord).km
            candidates.append((dist_km, fname, coord))

        if not candidates:
            rows.append({
                "park_id": station_id, "site_latitude": site_lat, "site_longitude": site_lon,
                "matched_file": None, "grid_latitude": None, "grid_longitude": None,
                "distance_km": None, "first_starttime": None, "last_starttime": None,
                "n_starttimes": 0, "reaches_2026_04_30": False, "error": "no_candidate_files",
            })
            print(f"[{station_id}] NO CANDIDATE FILES in {station_dir}", file=sys.stderr)
            continue

        candidates.sort(key=lambda t: t[0])
        dist_km, fname, (grid_lat, grid_lon) = candidates[0]
        src_path = os.path.join(station_dir, fname)

        df = pd.read_parquet(src_path, columns=["starttime"])
        first_st = df["starttime"].min()
        last_st = df["starttime"].max()
        n_st = df["starttime"].nunique()
        reaches = bool(last_st >= TARGET_DATE - pd.Timedelta(hours=15))

        dst_name = f"icon_d2_09z_{station_id}.parquet"
        dst_path = os.path.join(OUT_DIR, dst_name)
        shutil.copy2(src_path, dst_path)

        rows.append({
            "park_id": station_id, "site_latitude": site_lat, "site_longitude": site_lon,
            "matched_file": fname, "grid_latitude": grid_lat, "grid_longitude": grid_lon,
            "distance_km": round(dist_km, 3), "first_starttime": first_st, "last_starttime": last_st,
            "n_starttimes": n_st, "reaches_2026_04_30": reaches, "error": None,
        })
        print(f"[{station_id}] -> {fname} (d={dist_km:.2f} km) last={last_st} n={n_st}", file=sys.stderr)

    cov = pd.DataFrame(rows)
    cov_path = os.path.join(OUT_DIR, "icon_d2_coverage.csv")
    cov.to_csv(cov_path, index=False)

    print("\n=== SUMMARY ===", file=sys.stderr)
    print(f"sites processed: {len(cov)}", file=sys.stderr)
    print(f"sites reaching 2026-04-30: {cov['reaches_2026_04_30'].sum()} / {len(cov)}", file=sys.stderr)
    print(f"distance_km stats:\n{cov['distance_km'].describe()}", file=sys.stderr)
    short = cov[~cov["reaches_2026_04_30"]]
    if len(short):
        print(f"\nSITES NOT REACHING TARGET DATE:\n{short[['park_id','last_starttime']]}", file=sys.stderr)
    far = cov[cov["distance_km"] > 5]
    if len(far):
        print(f"\nSITES WITH NEAREST GRID POINT > 5 km AWAY:\n{far[['park_id','distance_km']]}", file=sys.stderr)


if __name__ == "__main__":
    main()
