#!/usr/bin/env python3
"""M-ladder analysis: pairwise WP1 statistics between rungs, ACF overlays,
stability-stratified error table (WP3) and BLH-flag shares.

Usage: analyze_ladder.py [--rungs M1 M2 M3 M4 M5]
Reads results/round2/ladder_metrics.csv + the per-park synth CSVs.
Writes results/round2/{pairwise_stats.csv, stability_stratified.csv,
blh_flags.csv} and figs/round2/acf_<exp>.png
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
from round2 import evaluation, meterdata  # noqa: E402

SYNTH_BASE = "/mnt/nvme2/synthetic"
RESULTS = os.path.join(REPO, "results", "round2")
WINDOW = ("2023-06-01", "2024-06-01")


def pairwise(df: pd.DataFrame, rungs):
    rows = []
    for metric in ("r2", "energy_ratio", "wasserstein", "rmse_n"):
        piv = df.pivot_table(index="park_id", columns="experiment", values=metric)
        for i, a in enumerate(rungs):
            for b in rungs[i + 1:]:
                if a not in piv or b not in piv:
                    continue
                x = piv[a] if metric != "energy_ratio" else (piv[a] - 1).abs()
                y = piv[b] if metric != "energy_ratio" else (piv[b] - 1).abs()
                stats = evaluation.compare_pathways(y, x)  # positive diff = b better? see note
                rows.append({"metric": metric if metric != "energy_ratio"
                             else "abs_energy_ratio_dev", "a": a, "b": b, **stats})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RESULTS, "pairwise_stats.csv"), index=False)
    return out


def acf_figures(rungs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from statsmodels.tsa.stattools import acf as sm_acf
    for exp in rungs:
        files = sorted(glob.glob(os.path.join(SYNTH_BASE, "wind", "round2", exp,
                                              "synth_*.csv")))
        if not files:
            continue
        fig, ax = plt.subplots(figsize=(8, 5))
        acc_m, acc_s = [], []
        for f in files:
            pid = os.path.basename(f)[len("synth_"):-len(".csv")]
            synth = pd.read_csv(f, sep=";", index_col=0, parse_dates=True)[
                "power_park"].loc[WINDOW[0]:WINDOW[1]]
            meas = meterdata.load_park_power(pid, WINDOW)
            both = pd.concat([meas.rename("m"), synth.rename("s")], axis=1).dropna()
            if len(both) < 1000:
                continue
            acc_m.append(sm_acf(both["m"], nlags=48))
            acc_s.append(sm_acf(both["s"], nlags=48))
        if not acc_m:
            continue
        lags = np.arange(49)
        for a in acc_m:
            ax.plot(lags, a, color="k", alpha=0.15)
        for a in acc_s:
            ax.plot(lags, a, color="tab:red", alpha=0.15)
        ax.plot(lags, np.mean(acc_m, axis=0), color="k", lw=2, label="measured (mean)")
        ax.plot(lags, np.mean(acc_s, axis=0), color="tab:red", lw=2,
                label=f"synthetic {exp} (mean)")
        ax.set_xlabel("lag [h]")
        ax.set_ylabel("ACF of park power")
        ax.set_title(f"Autocorrelation, measured vs synthetic — {exp}")
        ax.legend()
        fig.savefig(os.path.join(REPO, "figs", "round2", f"acf_{exp}.png"),
                    dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"acf figure for {exp}")


def stability_table(exp: str):
    """Stratified errors by stability class + BLH-flag share (needs an
    experiment run with shear='most' so the columns exist)."""
    rows, blh_rows = [], []
    for f in sorted(glob.glob(os.path.join(SYNTH_BASE, "wind", "round2", exp,
                                           "synth_*.csv"))):
        pid = os.path.basename(f)[len("synth_"):-len(".csv")]
        df = pd.read_csv(f, sep=";", index_col=0, parse_dates=True) \
            .loc[WINDOW[0]:WINDOW[1]]
        if "stability_class" not in df.columns:
            continue
        meas = meterdata.load_park_power(pid, WINDOW)
        rated = meterdata.rated_power_w(pid)
        both = pd.concat([meas.rename("m"), df["power_park"].rename("s"),
                          df["stability_class"]], axis=1).dropna()
        for cls, grp in both.groupby("stability_class"):
            err = (grp["s"] - grp["m"]) / rated
            rows.append({"park_id": pid, "stability": cls, "n": len(grp),
                         "bias_n": float(err.mean()),
                         "rmse_n": float(np.sqrt((err ** 2).mean()))})
        flag_cols = [c for c in df.columns if c.startswith("blh_flag")]
        if flag_cols:
            blh_rows.append({"park_id": pid,
                             "blh_flag_share": float(df[flag_cols].any(axis=1).mean())})
    if rows:
        st = pd.DataFrame(rows)
        st.to_csv(os.path.join(RESULTS, "stability_stratified.csv"), index=False)
        agg = st.groupby("stability").apply(
            lambda g: pd.Series({"parks": g.park_id.nunique(),
                                 "mean_bias_n": g.bias_n.mean(),
                                 "mean_rmse_n": g.rmse_n.mean(),
                                 "hours": g.n.sum()}), include_groups=False)
        print("\nstability-stratified errors:\n", agg.round(4).to_string())
    if blh_rows:
        bl = pd.DataFrame(blh_rows)
        bl.to_csv(os.path.join(RESULTS, "blh_flags.csv"), index=False)
        print("\nBLH-flag shares:\n", bl.round(3).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rungs", nargs="*", default=["M1", "M2", "M3", "M4", "M4b", "M5"])
    ap.add_argument("--stability-exp", default="M3")
    args = ap.parse_args()
    df = pd.read_csv(os.path.join(RESULTS, "ladder_metrics.csv"),
                     dtype={"park_id": str})
    rungs = [r for r in args.rungs if r in df["experiment"].unique()]
    print("rungs found:", rungs)

    med = df[df["experiment"].isin(rungs)].groupby("experiment")[
        ["r2", "energy_ratio", "wasserstein", "rmse_n"]].median()
    print("\nmedians per rung:\n", med.round(4).to_string())

    pw = pairwise(df, rungs)
    key = pw[(pw["metric"] == "abs_energy_ratio_dev")]
    print("\npairwise |ER-1| (positive median_diff = first rung worse):\n",
          key[["a", "b", "median_diff", "wilcoxon_p", "rank_biserial"]]
          .round(4).to_string(index=False))
    acf_figures(rungs)
    stability_table(args.stability_exp)

    with open(os.path.join(RESULTS, "RUN_REPORT.md"), "a") as f:
        f.write(f"\n## ladder analysis ({pd.Timestamp.now()})\n")
        f.write("### medians per rung\n" + med.round(4).to_markdown() + "\n")


if __name__ == "__main__":
    main()
