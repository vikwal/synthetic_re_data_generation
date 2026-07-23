#!/usr/bin/env python3
"""HANDOFF ask 6 — M0 (Renewables.ninja) through the same clean protocol.

Source: /mnt/nas/renewables.ninja/Wind/data/real_parks/<park>.csv
(timestamp, power_tN [kW], wind_speed_tN [m/s]; hourly, Jun 2023 - Jun 2024).
Covers 9 of the 13 parks (missing: 00298, 01200_1, 01200_2, 01303_1) — M0
comparisons run at N=9, disclosed in the export.

Appends/replaces experiment='M0' rows in results/round2/summary/scores.csv and
stores DBSCAN masks under masks/M0/ (same per-model protocol as the ladder).
"""

import glob
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
from round2 import evaluation, meterdata, outliers  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_ladder import min_cut_in  # noqa: E402

M0_DIR = "/mnt/nas/renewables.ninja/Wind/data/real_parks"
MASK_BASE = "/mnt/nvme2/synthetic/raw/round2/masks"
SUM_DIR = os.path.join(REPO, "results", "round2", "summary")
WINDOW = ("2023-06-01", "2024-06-01")


def main():
    prices = pd.read_csv(os.path.join(REPO, "data", "round2", "prices_delu.csv"),
                         index_col=0, parse_dates=True)["price_eur_mwh"]
    parks = sorted(meterdata.load_mapping()["park_id"])
    rows, missing = [], []
    for pid in parks:
        path = os.path.join(M0_DIR, f"{pid}.csv")
        if not os.path.exists(path):
            missing.append(pid)
            continue
        df = pd.read_csv(path, parse_dates=["timestamp"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df.set_index("timestamp", inplace=True)
        df = df.loc[WINDOW[0]:WINDOW[1]]
        pcols = [c for c in df.columns if c.startswith("power_t")]
        wcols = [c for c in df.columns if c.startswith("wind_speed_t")]
        synth = df[pcols].sum(axis=1) * 1000.0  # kW -> W
        v_hub = df[wcols].mean(axis=1)
        meas = meterdata.load_park_power(pid, WINDOW)
        rated = meterdata.rated_power_w(pid)

        keep = outliers.compute_mask(pid, meas, v_hub,
                                     min_cut_in=min_cut_in(pid), synth=synth)
        os.makedirs(os.path.join(MASK_BASE, "M0"), exist_ok=True)
        keep.rename("keep").to_frame().to_parquet(
            os.path.join(MASK_BASE, "M0", f"mask_{pid}.parquet"))
        share = float(1.0 - keep.mean())
        keep_b = keep.reindex(meas.index).fillna(False).astype(bool)

        for variant, (m, pr) in {"raw": (meas, None),
                                 "clean": (meas[keep_b], None),
                                 "clean_exkl_curt": (meas[keep_b], prices)}.items():
            res = evaluation.evaluate(m, synth, p_rated=rated, prices=pr)
            res.pop("acf_meas", None), res.pop("acf_synth", None)
            if variant == "clean_exkl_curt":
                res = {k.replace("_excl_curt", ""): v for k, v in res.items()
                       if (k.endswith("_excl_curt") and not k.startswith("acf"))
                       or k in ("n_hours", "window", "curtailed_share")}
            rows.append({"experiment": "M0", "park_id": pid, "variant": variant,
                         "outlier_share": share, **res})
        r2c = [r for r in rows if r["park_id"] == pid and r["variant"] == "clean"][0]["r2"]
        print(f"M0 {pid}: clean R2={r2c:.3f}, outlier share {share:.1%}")

    scores = pd.read_csv(os.path.join(SUM_DIR, "scores.csv"), dtype={"park_id": str})
    scores = scores[scores["experiment"] != "M0"]
    scores = pd.concat([scores, pd.DataFrame(rows)], ignore_index=True)
    scores.to_csv(os.path.join(SUM_DIR, "scores.csv"), index=False)
    print(f"\nM0: {len(rows)//3} parks scored (missing: {missing}) -> scores.csv")


if __name__ == "__main__":
    main()
