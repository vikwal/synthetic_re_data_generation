#!/usr/bin/env python3
"""WP1 — tidy per-station dataset + per-gridpoint normalization stats.

For every station with ERA5 v2 data: join hourly ERA5 (u10, v10, ws10, t2m,
blh, sp) with DWD 10 m wind obs, build the scaling-factor target
y = ws_obs / ws_era5 with validity mask, and write one parquet per station.
Normalization stats (mean/std per station/gridpoint, TRAIN period only) go to
data/era5_bc/norm_stats.json.

Run with the synthre venv:
    synthre/bin/python scripts/era5_bc/wp1_build_dataset.py

Verification printed per split: obs coverage over train/eval periods,
target median sanity (expected within [0.3, 3]).
"""

import json
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
from era5_bc import data as D  # noqa: E402
from era5_bc.config import load_config, station_processed_path  # noqa: E402


def main():
    cfg = load_config()
    per = cfg["periods"]
    os.makedirs(cfg["paths"]["processed_dir"], exist_ok=True)

    norm_stats, rows = {}, []
    for split in ("train", "val", "test"):
        for sid in cfg["stations"][f"{split}_effective"]:
            frame = D.build_station_frame(cfg, sid)
            norm_stats[sid] = D.compute_norm_stats(frame, cfg)
            frame.to_parquet(station_processed_path(cfg, sid))

            y_train = frame.loc[per["train_start"]:per["train_end"], "y"]
            rows.append({
                "station_id": sid,
                "split": split,
                "cov_train": D.obs_coverage(frame, per["train_start"], per["train_end"]),
                "cov_eval": D.obs_coverage(frame, per["eval_start"], per["eval_end"]),
                "y_median_train": float(y_train.median()),
                "y_p99_train": float(y_train.quantile(0.99)),
                "n_hours": len(frame),
            })
            print(f"{sid} [{split}] cov_train={rows[-1]['cov_train']:.2f} "
                  f"cov_eval={rows[-1]['cov_eval']:.2f} "
                  f"y_med={rows[-1]['y_median_train']:.2f}")

    stats_path = cfg["paths"]["norm_stats"]
    os.makedirs(os.path.dirname(stats_path), exist_ok=True)
    with open(stats_path, "w") as f:
        json.dump(norm_stats, f, indent=1)

    report = pd.DataFrame(rows).set_index("station_id")
    report_path = os.path.join(os.path.dirname(stats_path), "wp1_coverage.csv")
    report.to_csv(report_path)

    # ---- summary + sanity ----
    print("\n=== WP1 summary ===")
    print(report.groupby("split")[["cov_train", "cov_eval", "y_median_train"]]
          .describe().T.round(3).to_string())
    low = report[report["cov_train"] < cfg["stations"]["min_obs_coverage"]]
    if len(low):
        print(f"\nWARNING: {len(low)} stations below min_obs_coverage "
              f"{cfg['stations']['min_obs_coverage']}: "
              f"{low.index.tolist()} (excluded at window-build time)")
    # mountain summits (Zugspitze 05792, Feldberg 01346, Brocken 00722) reach
    # median factors of 3.5-3.9 — physically plausible, ERA5 cannot resolve peaks
    high = report[report["y_median_train"] > 3]
    if len(high):
        print(f"\nINFO: median factor > 3 (summit stations): {high.index.tolist()}")
    bad_y = report[(report["y_median_train"] < 0.2) | (report["y_median_train"] > 5)]
    assert len(bad_y) == 0, f"suspicious target medians: {bad_y.index.tolist()}"
    print(f"\nwrote {len(report)} stations -> {cfg['paths']['processed_dir']}")
    print(f"norm stats -> {stats_path}\nreport -> {report_path}")


if __name__ == "__main__":
    main()
