#!/usr/bin/env python3
"""HANDOFF ask 26a — age-sensitive permutation statistics (post-hoc).

Same 1000 draws as ask 25 (identical rng protocol, seed 25), same fast chain,
same masks (M5all / M2all families, clean), canonical final-gate state — but
per-park Delta|ER-1| and DeltaR2 are stored per draw and two ALTERNATIVE
statistics are evaluated (specified in the HANDOFF ask BEFORE running):

  T_mean = fleet MEAN Delta|ER-1| (tail-sensitive, unlike the ask-25 median)
  T_tail = mean Delta|ER-1| over the parks holding the three OLDEST assigned
           ages in that draw (true assignment: the three 20+yr parks)

p_perm one-sided as in ask 25: (1 + #{perm >= true}) / (n_perm + 1).

POST-HOC LABEL: these statistics were chosen AFTER seeing the ask-25
old-tail pattern; drawer / response-letter use only, no paper artifact.

Outputs -> results/round2/ask26/:
  perm_null_alt.csv       per-draw T_mean, T_tail (+ per-park deltas), models
                          M5 + M2, true rows marked (draw='true')
  perm_null_alt_hist.png  null histograms, true value marked (2x2 panels)

Usage: ask26a_perm_alt.py [--n-perm 1000] [--seed 25]
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from round2 import aging as r2_aging  # noqa: E402
from round2 import meterdata, parkinfo  # noqa: E402
from round2.chain import ParkCache  # noqa: E402
from round2.paper_style import apply_print_style  # noqa: E402
from ask25_placebo import MODELS, AgingChain, ages_from_date, clean_arrays  # noqa: E402

OUT_DIR = os.path.join(REPO, "results", "round2", "ask26")
LAM, KAP = 54.5, 2.0
POSTHOC_NOTE = ("# POST-HOC statistics (ask 26a): T_mean / T_tail were chosen "
                "AFTER seeing the ask-25 old-tail pattern; same 1000 draws & "
                "seed as ask 25; drawer / response-letter use only.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=25)
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    parks = sorted(meterdata.load_mapping()["park_id"])
    print(f"building park caches ({len(parks)} parks) ...", flush=True)
    pcs = {pid: ParkCache(pid) for pid in parks}
    dates = {pid: parkinfo.commissioning_date(pid) for pid in parks}
    # the three oldest commissioning dates define the tail membership
    oldest3 = set(sorted(dates.values())[:3])
    print("three oldest dates:", sorted(oldest3), "->",
          [p for p in parks if dates[p] in oldest3])

    df_tab = {pid: {d: r2_aging.DF_weibull(ages_from_date(pcs[pid].index, d),
                                           LAM, KAP)
                    for d in dates.values()} for pid in parks}

    chains, oks, meas_ok = {}, {}, {}
    for m, spec in MODELS.items():
        for pid in parks:
            chains[(m, pid)] = AgingChain(pcs[pid], spec["theta"])
            oks[(m, pid)], meas_ok[(m, pid)] = clean_arrays(
                pcs[pid], spec["mask_exp"])

    def er_r2(m, pid, df_vec):
        ok = oks[(m, pid)]
        p = chains[(m, pid)].power(df_vec)[ok]
        mm = meas_ok[(m, pid)]
        r2 = 1.0 - float(((mm - p) ** 2).sum()) / float(((mm - mm.mean()) ** 2).sum())
        return float(p.sum() / mm.sum()), r2

    noage = {(m, pid): er_r2(m, pid, 1.0) for m in MODELS for pid in parks}

    def row_for(m, assign, draw_label):
        """Per-park twin deltas + T_mean / T_tail for one assignment."""
        d_er, d_r2, tail = {}, {}, []
        for pid in parks:
            er, r2 = er_r2(m, pid, df_tab[pid][assign[pid]])
            er0, r20 = noage[(m, pid)]
            d_er[pid] = abs(er0 - 1.0) - abs(er - 1.0)
            d_r2[pid] = r2 - r20
            if assign[pid] in oldest3:
                tail.append(d_er[pid])
        row = {"draw": draw_label, "model": m,
               "T_mean": float(np.mean(list(d_er.values()))),
               "T_tail": float(np.mean(tail)),
               "T_median": float(np.median(list(d_er.values())))}
        row.update({f"d_aed_{pid}": d_er[pid] for pid in parks})
        row.update({f"d_r2_{pid}": d_r2[pid] for pid in parks})
        return row

    rows = [row_for(m, dates, "true") for m in MODELS]
    true_stat = {(r["model"], k): r[k] for r in rows for k in ("T_mean", "T_tail")}

    # identical draw protocol to ask 25 (seed 25): one rng, permutation per
    # draw, identity redrawn inline -> same sequence of permutations
    rng = np.random.default_rng(args.seed)
    date_list = [dates[pid] for pid in parks]
    for i in range(args.n_perm):
        perm = rng.permutation(len(parks))
        while np.array_equal(perm, np.arange(len(parks))):
            perm = rng.permutation(len(parks))
        assign = {pid: date_list[perm[j]] for j, pid in enumerate(parks)}
        for m in MODELS:
            rows.append(row_for(m, assign, i))
        if (i + 1) % 100 == 0:
            print(f"  perm {i + 1}/{args.n_perm}", flush=True)

    null = pd.DataFrame(rows)
    out_csv = os.path.join(OUT_DIR, "perm_null_alt.csv")
    with open(out_csv, "w") as f:
        f.write(POSTHOC_NOTE + "\n")
        null.to_csv(f, index=False)

    pperm = {}
    for m in MODELS:
        nm = null[(null.model == m) & (null.draw != "true")]
        for stat in ("T_mean", "T_tail"):
            tv = true_stat[(m, stat)]
            pperm[(m, stat)] = float((1 + (nm[stat] >= tv).sum()) / (len(nm) + 1))
            print(f"{m} {stat}: true = {tv:+.4f}  p_perm = {pperm[(m, stat)]:.4f} "
                  f"(null mean {nm[stat].mean():+.4f}, "
                  f"p95 {nm[stat].quantile(0.95):+.4f})")

    # consistency: ask-25 median statistic must reproduce on the same draws
    a25 = pd.read_csv(os.path.join(REPO, "results", "round2", "ask25",
                                   "perm_null.csv"))
    for m in MODELS:
        old = a25[(a25.model == m) & (a25.draw != "true")][
            "d_abs_er_dev_median"].astype(float).values
        new = null[(null.model == m) & (null.draw != "true")]["T_median"].values
        print(f"{m}: draw reproduction vs ask25 max|dT_median| = "
              f"{np.abs(old - new).max():.2e}")

    apply_print_style()
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    labels = {"T_mean": r"fleet mean $\Delta|\mathrm{ER}-1|$ (no-aging $-$ aged)",
              "T_tail": r"mean $\Delta|\mathrm{ER}-1|$, 3 oldest assigned ages"}
    for i, m in enumerate(MODELS):
        nm = null[(null.model == m) & (null.draw != "true")]
        for j, stat in enumerate(("T_mean", "T_tail")):
            ax = axes[i, j]
            ax.hist(nm[stat].astype(float), bins=40, color="#9ecae1",
                    edgecolor="white")
            tv = true_stat[(m, stat)]
            ax.axvline(tv, color="#d62728", lw=3)
            ax.text(0.97, 0.95,
                    f"{m}\ntrue = {tv:+.3f}\n$p_{{perm}}$ = "
                    f"{pperm[(m, stat)]:.3f}",
                    transform=ax.transAxes, ha="right", va="top", fontsize=17,
                    bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"))
            ax.set_xlabel(labels[stat])
            if j == 0:
                ax.set_ylabel("permutation draws")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "perm_null_alt_hist.png"), dpi=180)
    plt.close(fig)
    print(f"done -> {OUT_DIR}")


if __name__ == "__main__":
    main()
