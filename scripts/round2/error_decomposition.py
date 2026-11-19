#!/usr/bin/env python3
"""HANDOFF ask 21b — Murphy/MSE error decomposition (brainstorm M-1).

For the main rungs, per park, on the cleaned capacity-normalized hourly
series (same protocol as score_ladder: stored DBSCAN masks, window
Jun 2023 – Jun 2024):

  MSE = bias^2 + (sigma_s - sigma_m)^2 + 2*sigma_s*sigma_m*(1 - r)

plus the raw triplet (bias_n, sigma_ratio, r). No regeneration — reads the
existing synth outputs and masks.

Outputs: results/round2/summary/error_decomposition.csv (park x rung)
         results/round2/summary/error_decomposition_medians.csv (per rung)
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
from round2 import meterdata  # noqa: E402
from round2.paper_style import PSEUDONYM, rung  # noqa: E402

SYNTH_BASE = "/mnt/nvme2/synthetic/wind/round2"
M0_DIR = "/mnt/nas/renewables.ninja/Wind/data/real_parks"
MASK_BASE = "/mnt/nvme2/synthetic/raw/round2/masks"
SUM_DIR = os.path.join(REPO, "results", "round2", "summary")
WINDOW = ("2023-06-01", "2024-06-01")

RUNGS = ["M0", "M1noage", "M2noage", "M3noage", "M4all", "M5all", "K5"]


def load_synth(exp: str, park_id: str) -> pd.Series:
    if exp == "M0":
        path = os.path.join(M0_DIR, f"{park_id}.csv")
        df = pd.read_csv(path, parse_dates=["timestamp"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df.set_index("timestamp", inplace=True)
        pcols = [c for c in df.columns if c.startswith("power_t")]
        return df.loc[WINDOW[0]:WINDOW[1], pcols].sum(axis=1) * 1000.0
    path = os.path.join(SYNTH_BASE, exp, f"synth_{park_id}.csv")
    df = pd.read_csv(path, sep=";", index_col=0, parse_dates=True) \
        .loc[WINDOW[0]:WINDOW[1]]
    return df["power_park"]


def decompose(meas_n: np.ndarray, synth_n: np.ndarray) -> dict:
    bias = float(synth_n.mean() - meas_n.mean())
    s_m, s_s = float(meas_n.std()), float(synth_n.std())
    r = float(np.corrcoef(meas_n, synth_n)[0, 1])
    mse = float(((synth_n - meas_n) ** 2).mean())
    terms = {
        "mse": mse,
        "bias_sq": bias ** 2,
        "variance_term": (s_s - s_m) ** 2,
        "phase_term": 2.0 * s_s * s_m * (1.0 - r),
        "bias_n": bias,
        "sigma_ratio": s_s / s_m if s_m > 0 else np.nan,
        "corr_r": r,
    }
    # closure check: the three terms must reproduce the MSE exactly
    terms["closure_resid"] = mse - (terms["bias_sq"] + terms["variance_term"]
                                    + terms["phase_term"])
    return terms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rungs", nargs="*", default=RUNGS)
    args = ap.parse_args()

    parks = sorted(meterdata.load_mapping()["park_id"])
    rows = []
    for exp in args.rungs:
        for pid in parks:
            try:
                synth = load_synth(exp, pid)
            except FileNotFoundError:
                continue
            meas = meterdata.load_park_power(pid, WINDOW)
            rated = meterdata.rated_power_w(pid)
            mask_path = os.path.join(MASK_BASE, exp, f"mask_{pid}.parquet")
            keep = pd.read_parquet(mask_path)["keep"]
            m = meas[keep.reindex(meas.index).fillna(False).astype(bool)]
            df = pd.concat([m.rename("meas"), synth.rename("synth")],
                           axis=1).dropna()
            terms = decompose(df["meas"].values / rated,
                              df["synth"].values / rated)
            assert abs(terms["closure_resid"]) < 1e-12, (exp, pid)
            rows.append({"experiment": exp, "rung": rung(exp),
                         "park_id": pid, "park": PSEUDONYM.get(pid, pid),
                         "n_hours": len(df), **terms})
        n = sum(1 for r in rows if r["experiment"] == exp)
        print(f"{exp}: {n} parks", flush=True)

    df = pd.DataFrame(rows).drop(columns=["closure_resid"])
    out = os.path.join(SUM_DIR, "error_decomposition.csv")
    df.round(6).to_csv(out, index=False)

    med = df.groupby(["experiment", "rung"], sort=False)[
        ["mse", "bias_sq", "variance_term", "phase_term",
         "bias_n", "sigma_ratio", "corr_r"]].median()
    for t in ("bias_sq", "variance_term", "phase_term"):
        med[f"{t}_share"] = med[t] / (med["bias_sq"] + med["variance_term"]
                                      + med["phase_term"])
    med_out = os.path.join(SUM_DIR, "error_decomposition_medians.csv")
    med.round(6).to_csv(med_out)
    print(f"\n{med.round(4).to_string()}")
    print(f"\n-> {out}\n-> {med_out}")


if __name__ == "__main__":
    main()
