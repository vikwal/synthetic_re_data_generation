#!/usr/bin/env python3
"""WP2-A — station correction table.

Pairs hourly ERA5 10 m wind with observed DWD 10 m wind per station over the
TRAINING window (Jul 2024 - Apr 2026; the guide's 2015-2022 window was waived
by the user - no ERA5 backfill), computes RMSE -> ERA5 quality class
(Hu 2023: <=1.5 / 1.5-3 / >3 m/s) and the 13 empirical quantiles of both
series. Hard assert: training never touches Jun 2023 - Jun 2024 (validation).

Output: data/round2/station_correction_table.parquet
"""

import glob
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
from round2.correction import QUANTILES  # noqa: E402

ERA5_DIR = "/mnt/nas/synthetic/raw/wind_era5"
DWD_DIR = "/mnt/nvme2/synthetic/raw/round2/dwd_wind_hourly"
OUT = os.path.join(REPO, "data", "round2", "station_correction_table.parquet")

TRAIN = ("2024-07-01", "2026-04-30 23:00")
VALIDATION = (pd.Timestamp("2023-06-01", tz="UTC"), pd.Timestamp("2024-07-01", tz="UTC"))
MIN_COVERAGE = 0.70


def classify(rmse: float) -> int:
    if rmse <= 1.5:
        return 1
    if rmse <= 3.0:
        return 2
    return 3


def main():
    assert pd.Timestamp(TRAIN[0], tz="UTC") >= VALIDATION[1], \
        "training window overlaps the validation window"

    rows, dropped = [], []
    for f in sorted(glob.glob(os.path.join(ERA5_DIR, "Station_*.csv"))):
        sid = os.path.basename(f)[len("Station_"):-len(".csv")]
        dwd_path = os.path.join(DWD_DIR, f"DWD_{sid}.parquet")
        if not os.path.exists(dwd_path):
            dropped.append((sid, "no_dwd"))
            continue
        era = pd.read_csv(f, usecols=["timestamp", "u_wind_10m", "v_wind_10m"])
        era["timestamp"] = pd.to_datetime(era["timestamp"], utc=True)
        era.set_index("timestamp", inplace=True)
        era["v10_era5"] = np.hypot(era["u_wind_10m"], era["v_wind_10m"])
        dwd = pd.read_parquet(dwd_path)[["wind_speed"]].rename(
            columns={"wind_speed": "v10_dwd"})
        both = era[["v10_era5"]].join(dwd, how="inner").loc[TRAIN[0]:TRAIN[1]].dropna()
        n_expected = len(pd.date_range(TRAIN[0], TRAIN[1], freq="1h", tz="UTC"))
        coverage = len(both) / n_expected
        if coverage < MIN_COVERAGE:
            dropped.append((sid, f"coverage_{coverage:.2f}"))
            continue
        assert both.index.min() >= VALIDATION[1], "leakage into validation window"
        err = both["v10_era5"] - both["v10_dwd"]
        rmse = float(np.sqrt((err ** 2).mean()))
        q_era5 = both["v10_era5"].quantile(QUANTILES).values
        q_dwd = both["v10_dwd"].quantile(QUANTILES).values
        rows.append({
            "station_id": sid, "n_hours": len(both), "coverage": coverage,
            "rmse": rmse, "bias": float(err.mean()),
            "corr": float(both["v10_era5"].corr(both["v10_dwd"])),
            "era5_class": classify(rmse),
            **{f"q_era5_{q:.3f}": v for q, v in zip(QUANTILES, q_era5)},
            **{f"q_dwd_{q:.3f}": v for q, v in zip(QUANTILES, q_dwd)},
        })
    df = pd.DataFrame(rows)
    df.to_parquet(OUT)
    print(f"stations kept: {len(df)} | dropped: {len(dropped)}")
    print("class counts:", df["era5_class"].value_counts().sort_index().to_dict())
    print("rmse: mean %.2f min %.2f max %.2f" % (df.rmse.mean(), df.rmse.min(), df.rmse.max()))
    if dropped:
        print("dropped:", dropped[:10], "..." if len(dropped) > 10 else "")


if __name__ == "__main__":
    main()
