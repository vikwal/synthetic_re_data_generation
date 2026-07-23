#!/usr/bin/env python3
"""Component-attribution (ablation) summary from the M-ladder.

Each rung adds ONE component; the increment rung-vs-previous is the
component's contribution (guide: 'The M-ladder IS the component-attribution
answer'). Statistics: exact Wilcoxon on |ER-1| per park (N=13) + effect size.

Output: results/round2/ablation_summary.md (+ ablation_per_park.csv)
"""

import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
from round2 import evaluation  # noqa: E402

RESULTS = os.path.join(REPO, "results", "round2")
LADDER = [
    ("M1", "Round-1-Anker (ERA5 + const aging)", None),
    ("M2", "+ gated QM-Korrektur (WP2)", "M1"),
    ("M3", "+ MOST-Stabilität (WP3, adoptiert)", "M2"),
    ("M4", "+ Weibull-Aging (WP4)", "M3"),
    ("M4b", "+ EEG-Step-Variante (WP4)", "M3"),
    ("M5", "+ NOJ-Wakes (WP5)", "M4"),
]
# Wake fairness: single-turbine parks have no intra-park wakes — their paired
# diff for the wake rung is structurally 0 and dilutes median/bootstrap/counts.
# The wake rung therefore gets an ADDITIONAL row restricted to parks with
# >= MIN_TURBINES_WAKE turbines (the all-parks row stays for comparability).
WAKE_RUNG = "M5"
MIN_TURBINES_WAKE = 3


def main():
    df = pd.read_csv(os.path.join(RESULTS, "ladder_metrics.csv"),
                     dtype={"park_id": str})
    piv = {m: df.pivot_table(index="park_id", columns="experiment", values=m)
           for m in ("r2", "energy_ratio", "wasserstein", "rmse_n")}

    rows = []
    for exp, label, prev in LADDER:
        r = {"rung": exp, "component": label,
             "median_r2": piv["r2"][exp].median(),
             "median_ER": piv["energy_ratio"][exp].median(),
             "median_absERdev": (piv["energy_ratio"][exp] - 1).abs().median(),
             "median_W1": piv["wasserstein"][exp].median()}
        if prev:
            a = (piv["energy_ratio"][prev] - 1).abs()
            b = (piv["energy_ratio"][exp] - 1).abs()
            st = evaluation.compare_pathways(b, a)
            r.update({"d_absERdev_vs_prev": st["median_diff"],
                      "wilcoxon_p": st.get("wilcoxon_p"),
                      "rank_biserial": st.get("rank_biserial"),
                      "n_parks_improved": int((b < a).sum()),
                      "n_parks_changed": int((b != a).sum())})
        rows.append(r)

        if exp == WAKE_RUNG and prev:
            n_turb = evaluation.park_turbine_counts()
            multi = [p for p in b.index if n_turb.get(p, 0) >= MIN_TURBINES_WAKE]
            am, bm = a.loc[multi], b.loc[multi]
            st = evaluation.compare_pathways(bm, am)
            rows.append({
                "rung": f"{exp} (n_turb>={MIN_TURBINES_WAKE})",
                "component": f"wakes, multi-turbine parks only (N={len(multi)})",
                "median_r2": piv["r2"][exp].loc[multi].median(),
                "median_ER": piv["energy_ratio"][exp].loc[multi].median(),
                "median_absERdev": bm.median(),
                "median_W1": piv["wasserstein"][exp].loc[multi].median(),
                "d_absERdev_vs_prev": st["median_diff"],
                "wilcoxon_p": st.get("wilcoxon_p"),
                "rank_biserial": st.get("rank_biserial"),
                "n_parks_improved": int((bm < am).sum()),
                "n_parks_changed": int((bm != am).sum())})
    summary = pd.DataFrame(rows)

    per_park = pd.concat(
        {m: piv[m][[e for e, _, _ in LADDER]] for m in ("r2", "energy_ratio", "wasserstein")},
        axis=1)
    per_park.to_csv(os.path.join(RESULTS, "ablation_per_park.csv"))

    lines = [
        "# Ablation study — component attribution (M-ladder, 13 parks)",
        "",
        "Increment = rung vs. previous rung on |energy ratio - 1| per park; "
        "negative d = component reduces the energy-bias. Exact Wilcoxon, N=13.",
        "",
        summary.round(4).to_markdown(index=False),
        "",
        "Notes:",
        "- M2 changes only the 6 branch-A/B parks (7 branch-C nulls by design).",
        "- M4/M4b act only on the 5 parks with apply_ageing=True (round-1 "
        "heritage). M4b differs from M4 only at park 07374 (the single aged "
        "park older than 20 y; -0.5 % energy) -> medians identical.",
        "- Per-park values: ablation_per_park.csv; full stats: pairwise_stats.csv.",
        f"- Wake rung: extra row restricted to parks with >= "
        f"{MIN_TURBINES_WAKE} turbines — single-turbine parks (00298, 05426) "
        "have no intra-park wakes, their paired diff is structurally 0 and "
        "only dilutes median/bootstrap/counts of the all-parks test.",
    ]
    out = os.path.join(RESULTS, "ablation_summary.md")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(summary.round(4).to_string(index=False))
    print("\nwritten:", out)


if __name__ == "__main__":
    main()
