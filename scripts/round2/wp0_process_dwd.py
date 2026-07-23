#!/usr/bin/env python3
"""WP0.3 — hourly DWD 10 m wind from the existing 10-min parquets.

Input:  /mnt/lambda1/nvme1/synthetic/raw/wind/Station_XXXXX.parquet
        (10-min, columns: station_id, wind_speed, temperature_2m,
        relative_humidity, std_v_wind, pressure, wind_direction; UTC index)
Output: /mnt/nvme2/synthetic/raw/round2/dwd_wind_hourly/DWD_XXXXX.parquet
        hourly means, an hour requires >= 4 of 6 valid 10-min values.
"""

import glob
import os

import pandas as pd

IN_DIR = "/mnt/lambda1/nvme1/synthetic/raw/wind"
OUT_DIR = "/mnt/nvme2/synthetic/raw/round2/dwd_wind_hourly"
MIN_VALID = 4
COLS = ["wind_speed", "wind_direction", "temperature_2m", "pressure"]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    files = sorted(glob.glob(os.path.join(IN_DIR, "Station_*.parquet")))
    stats = []
    for f in files:
        sid = os.path.basename(f)[len("Station_"):-len(".parquet")]
        df = pd.read_parquet(f)
        df = df[[c for c in COLS if c in df.columns]].sort_index()
        # DWD missing markers
        df = df.mask(df <= -999)
        grouper = df.resample("1h")
        hourly = grouper.mean()
        counts = grouper.count()
        hourly = hourly.where(counts >= MIN_VALID)
        hourly.to_parquet(os.path.join(OUT_DIR, f"DWD_{sid}.parquet"))
        cov = float(hourly["wind_speed"].notna().mean()) if "wind_speed" in hourly else 0.0
        stats.append((sid, len(hourly), cov))
    print(f"stations: {len(stats)}")
    low = [(s, round(c, 3)) for s, _, c in stats if c < 0.7]
    print("coverage<70%:", low if low else "none")
    span = pd.read_parquet(os.path.join(OUT_DIR, f"DWD_{stats[0][0]}.parquet")).index
    print("range:", span.min(), "..", span.max())


if __name__ == "__main__":
    main()
