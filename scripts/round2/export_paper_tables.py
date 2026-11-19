#!/usr/bin/env python3
"""Paper exports requested via HANDOFF.md (2026-07-05).

1. ladder_results_flat.csv — experiment x park x variant with R2, ER, W1,
   nRMSE, nMAE and normalized mean bias (bias_n = mean(P_synth - P_meas) /
   P_rated over the evaluated hours; positive = overestimation).
2. wilcoxon_key_comparisons.csv — exact Wilcoxon + rank-biserial + bootstrap
   CI for the manuscript's key contrasts, park level (N=13), clean and raw.
3. outlier_share_by_model.csv — DBSCAN outlier share per experiment
   (mean/max over parks) for the fairness-caveat sentence.

Outputs land in results/round2/summary/ and are copied next to the figs.
"""

import os
import shutil
import sys

import numpy as np
import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
from round2 import evaluation, meterdata  # noqa: E402

SUM_DIR = os.path.join(REPO, "results", "round2", "summary")
FIG_DIR = os.path.join(REPO, "figs", "round2", "summary")
SYNTH_BASE = "/mnt/nvme2/synthetic/wind/round2"
MASK_BASE = "/mnt/nvme2/synthetic/raw/round2/masks"
WINDOW = ("2023-06-01", "2024-06-01")

# wake fairness: single-turbine parks (00298, 05426) have no intra-park
# wakes — structural zeros in the wake pair. These comparisons get an extra
# companion row restricted to parks with >= WAKE_MIN_TURBINES turbines.
WAKE_RESTRICTED = {"M4 vs M5 (wakes)"}
WAKE_MIN_TURBINES = 3

KEY_COMPARISONS = [
    ("M1 vs M2", "M2", "M1"),
    ("M1 vs M3", "M3", "M1"),
    ("M1 vs M4", "M4", "M1"),
    ("M1 vs M5", "M5", "M1"),
    ("M1 vs M5all", "M5all", "M1"),
    ("M2 vs M3 (MOST)", "M3", "M2"),
    ("M3 vs M4 (aging, 5 parks)", "M4", "M3"),
    ("M4 vs M5 (wakes)", "M5", "M4"),
    ("weibull vs const (all parks)", "M4all", "M3allconst"),
    ("weibull+step vs weibull", "M4ball", "M4all"),
    ("no aging vs weibull (all parks)", "M4all", "M3noage"),
    ("K5 vs M5all (measured vs ERA5)", "K5", "M5all"),
    ("MK vs M5all (paper M5 vs MK)", "MK", "M5all"),
    ("MK vs K5 (identity check)", "MK", "K5"),
    ("M0 vs M1 (ninja baseline, N=9)", "M1", "M0"),
    ("M0 vs M5all (ninja vs full chain, N=9)", "M5all", "M0"),
    ("M0 vs MK (ninja vs measured path, N=9)", "MK", "M0"),
    # ask 14: per-rung aging contrasts (paper Sec 5.4 table)
    ("aging per rung: M2 (5 parks)", "M2", "M2noage"),
    ("aging per rung: M3 (5 parks)", "M3", "M3noage"),
    ("aging per rung: M4 (all parks)", "M4all", "M4all_noage"),
    ("aging per rung: M5 (all parks)", "M5all", "M5all_noage"),
    ("aging per rung: K5 (all parks)", "K5", "K5_noage"),
    # ask 19: clean ladder — per-depth aging contrast at M2 depth
    ("aging per rung: M2all (all parks)", "M2all", "M2noage"),
    ("clean-ladder M1: aged vs no aging", "M1", "M1noage"),
    # ask 18a: restrictive correction gate (branch A only <= 5 km)
    ("M5_gate5 vs M5 (restrictive gate)", "M5_gate5", "M5all"),
    ("M5_gate5 vs M5-noQM (gate5 net QM effect)", "M5_gate5", "M5all_noQM"),
    # ask 21a: similarity gate (D3) — QM step and net QM effect
    ("M2_simgate vs M1 (simgate QM step)", "M2_simgate", "M1noage"),
    ("M2_simgate vs M2 (simgate vs distance gate)", "M2_simgate", "M2noage"),
    ("M5_simgate vs M5 (simgate full chain)", "M5_simgate", "M5all"),
    ("M5_simgate vs M5-noQM (simgate net QM effect)", "M5_simgate", "M5all_noQM"),
]


M0_DIR = "/mnt/nas/renewables.ninja/Wind/data/real_parks"


def load_synth_power(exp: str, park_id: str) -> pd.Series:
    if exp == "M0":  # renewables.ninja baseline (kW, per-turbine columns)
        df = pd.read_csv(os.path.join(M0_DIR, f"{park_id}.csv"),
                         parse_dates=["timestamp"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df.set_index("timestamp", inplace=True)
        pcols = [c for c in df.columns if c.startswith("power_t")]
        return (df[pcols].sum(axis=1) * 1000.0).loc[WINDOW[0]:WINDOW[1]]
    path = os.path.join(SYNTH_BASE, exp, f"synth_{park_id}.csv")
    return pd.read_csv(path, sep=";", index_col=0, parse_dates=True) \
        .loc[WINDOW[0]:WINDOW[1], "power_park"]


def bias_n(exp: str, park_id: str, variant: str) -> float:
    """Normalized mean hourly bias over the hours the variant evaluates."""
    synth = load_synth_power(exp, park_id)
    meas = meterdata.load_park_power(park_id, WINDOW)
    both = pd.concat([meas.rename("m"), synth.rename("s")], axis=1).dropna()
    if variant != "raw":
        mask_path = os.path.join(MASK_BASE, exp, f"mask_{park_id}.parquet")
        keep = pd.read_parquet(mask_path)["keep"]
        both = both[keep.reindex(both.index).fillna(False).astype(bool)]
    rated = meterdata.rated_power_w(park_id)
    return float((both["s"] - both["m"]).mean() / rated)


def main():
    scores = pd.read_csv(os.path.join(SUM_DIR, "scores.csv"),
                         dtype={"park_id": str})

    # ---- 1. flat ladder table with bias ----
    flat = scores[["experiment", "park_id", "variant", "r2", "energy_ratio",
                   "wasserstein", "rmse_n", "mae_n", "n_hours",
                   "outlier_share"]].copy()
    flat["bias_n"] = [
        bias_n(r.experiment, r.park_id, r.variant)
        for r in flat.itertuples()
    ]
    # ask 12c: authoritative commissioning date + age in every export row
    from round2 import parkinfo
    ages = parkinfo.load().set_index("park_id")
    flat["commissioning_date"] = flat["park_id"].map(ages["commissioning_date"])
    flat["age_years_2023_12"] = flat["park_id"].map(ages["age_years_2023_12"])
    out1 = os.path.join(SUM_DIR, "ladder_results_flat.csv")
    flat.round(6).to_csv(out1, index=False)
    shutil.copy(out1, os.path.join(FIG_DIR, "ladder_results_flat.csv"))
    print(f"1. {out1} ({len(flat)} rows; bias_n = mean(P_synth-P_meas)/P_rated)")

    # ---- 2. key-comparison statistics ----
    rows = []
    for variant in ("clean", "raw"):
        sv = scores[scores["variant"] == variant].copy()
        sv["abs_er_dev"] = (sv["energy_ratio"] - 1.0).abs()
        sv = sv.set_index(["experiment", "park_id"])
        multi_ids = set(evaluation.park_turbine_counts()
                        .loc[lambda s: s >= WAKE_MIN_TURBINES].index)
        for label, a, b in KEY_COMPARISONS:
            for metric in ("r2", "abs_er_dev", "wasserstein"):
                try:
                    pa, pb = sv.loc[a][metric], sv.loc[b][metric]
                except KeyError:
                    continue
                pair = pd.concat([pa.rename("a"), pb.rename("b")], axis=1).dropna()
                variants_of_pair = [(label, pair)]
                if label in WAKE_RESTRICTED:
                    variants_of_pair.append((
                        f"{label.replace(')', '')}, n_turb>={WAKE_MIN_TURBINES})",
                        pair.loc[pair.index.isin(multi_ids)]))
                for lab, pr in variants_of_pair:
                    st = evaluation.compare_pathways(pr["a"], pr["b"])
                    rows.append({
                        "comparison": lab, "a": a, "b": b, "metric": metric,
                        "variant": variant, "n_parks": st["n_parks"],
                        "median_diff_a_minus_b": st["median_diff"],
                        "wilcoxon_p_exact": st.get("wilcoxon_p"),
                        "rank_biserial": st.get("rank_biserial"),
                        "boot_ci_lo": st.get("boot_ci_lo"),
                        "boot_ci_hi": st.get("boot_ci_hi"),
                        "sign_test_p": st.get("sign_test_p"),
                    })
    stats = pd.DataFrame(rows)
    out2 = os.path.join(SUM_DIR, "wilcoxon_key_comparisons.csv")
    stats.round(6).to_csv(out2, index=False)
    shutil.copy(out2, os.path.join(FIG_DIR, "wilcoxon_key_comparisons.csv"))
    print(f"2. {out2} ({len(stats)} rows)")

    # ---- 3. outlier share per model ----
    sh = scores[scores["variant"] == "clean"].groupby("experiment")[
        "outlier_share"].agg(["mean", "max", "min"]).round(4)
    out3 = os.path.join(SUM_DIR, "outlier_share_by_model.csv")
    sh.to_csv(out3)
    print(f"3. {out3}")
    print(sh.to_string())
    overall = scores[scores["variant"] == "clean"]["outlier_share"]
    print(f"\noverall: mean {overall.mean():.3f}, max {overall.max():.3f} "
          f"(max bei {scores.loc[overall.idxmax(), 'park_id']})")


if __name__ == "__main__":
    main()
