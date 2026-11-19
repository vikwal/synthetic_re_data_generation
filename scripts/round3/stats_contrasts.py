#!/usr/bin/env python3
"""Round 3 (reviewer 5): the paired differences behind the key contrasts.

For each contrast (A vs B, farm-level, clean variant) and metric: the 13 (or
9) paired differences, their skewness, the exact Wilcoxon signed-rank p, the
sign-test p (needs no symmetry), the Hodges-Lehmann estimate (pseudo-median
of the Walsh averages), the plain median and mean difference, the win count,
and the smallest and largest difference.

Run on l2: synthre/bin/python scripts/round3/stats_contrasts.py [--scores results/quantitative/scores.csv]
Output: results/round3/paired_contrasts.csv, paired_differences_long.csv
"""
import argparse
import itertools
import os

import numpy as np
import pandas as pd
from scipy import stats

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(REPO, "results", "round3")

CONTRASTS = [("M5all_PL", "M0", "framework vs Renewables.ninja"),
             ("M1noage", "M0", "simple ERA5 chain vs Renewables.ninja"),
             ("M2noage", "M1noage", "+ correction gate"),
             ("M2all", "M2noage", "+ aging"),
             ("M5all_PL", "M2all", "+ wakes"),
             ("M5all_PL", "M1noage", "full chain vs simple ERA5 chain"),
             ("M5all", "M5all_PL", "MOST profile vs power law"),
             ("M5all_PL", "M5all_PL_noage", "aging on/off at the full chain"),
             ("M5all_PL", "M5all_PL_noWake", "wakes on/off at the full chain")]
METRICS = {"r2": "higher", "abs_er_dev": "lower", "rmse_n": "lower", "wasserstein": "lower"}


def hodges_lehmann(d):
    w = [(a + b) / 2 for a, b in itertools.combinations_with_replacement(d, 2)]
    return float(np.median(w))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default=os.path.join(REPO, "results", "quantitative", "scores.csv"))
    ap.add_argument("--variant", default="clean")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    sc = pd.read_csv(args.scores, dtype={"park_id": str})
    sc = sc[sc.variant == args.variant].copy()
    sc["abs_er_dev"] = (sc.energy_ratio - 1).abs()
    rows, long = [], []
    for a, b, label in CONTRASTS:
        A = sc[sc.experiment == a].set_index("park_id")
        B = sc[sc.experiment == b].set_index("park_id")
        parks = sorted(set(A.index) & set(B.index))
        if len(parks) < 5:
            print(f"skip {a} vs {b}: {len(parks)} common parks")
            continue
        for metric, better in METRICS.items():
            d = (A.loc[parks, metric] - B.loc[parks, metric]).astype(float)
            sign = 1.0 if better == "higher" else -1.0
            adv = sign * d                       # > 0 means A better
            nz = adv[adv != 0]
            try:
                p_w = stats.wilcoxon(nz, alternative="two-sided", method="exact").pvalue if len(nz) else np.nan
            except Exception:
                p_w = stats.wilcoxon(nz, alternative="two-sided").pvalue if len(nz) else np.nan
            p_sign = stats.binomtest(int((nz > 0).sum()), len(nz), 0.5).pvalue if len(nz) else np.nan
            rows.append({"contrast": label, "A": a, "B": b, "metric": metric, "n": len(parks),
                         "wins_A": int((adv > 0).sum()), "ties": int((adv == 0).sum()),
                         "median_diff": float(d.median()), "mean_diff": float(d.mean()),
                         "hodges_lehmann": hodges_lehmann(d.values),
                         "min_diff": float(d.min()), "max_diff": float(d.max()),
                         "skew": float(stats.skew(d)) if len(d) > 2 else np.nan,
                         "p_wilcoxon_exact": p_w, "p_sign": p_sign,
                         "rank_biserial": float((2 * stats.rankdata(np.abs(nz.values)) * np.sign(nz.values)).sum()
                                                / (len(nz) * (len(nz) + 1))) if len(nz) else np.nan})
            for pk, val in d.items():
                long.append({"contrast": label, "metric": metric, "park_id": pk, "diff_A_minus_B": float(val)})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT, "paired_contrasts.csv"), index=False)
    pd.DataFrame(long).to_csv(os.path.join(OUT, "paired_differences_long.csv"), index=False)
    pd.set_option("display.width", 250)
    show = out[out.metric.isin(["r2", "abs_er_dev"])]
    print(show[["contrast", "metric", "n", "wins_A", "median_diff", "hodges_lehmann", "mean_diff",
                "min_diff", "max_diff", "skew", "p_wilcoxon_exact", "p_sign", "rank_biserial"]]
          .round(4).to_string(index=False))
    print("STATS_DONE")


if __name__ == "__main__":
    main()
