#!/usr/bin/env python3
"""Round-2 summary: component contrasts, tables and figures.

Reads results/round2/summary/scores.csv (score_ladder.py). Headline numbers =
DBSCAN-cleaned ('clean'); raw as robustness companion.

Contrasts (each: exact Wilcoxon + rank-biserial + bootstrap CI on |ER-1| and
R2; Holm correction over the four LOO headline contrasts):
  extrapolation  M5all vs M5all_PL      | incremental M3 vs M2
  aging          M5all vs M5all_noage   | shapes M3allconst/M4all/M4ball vs M3noage
  downscaling    M5all vs M5all_noQM    | incremental M2 vs M1 (per branch)
  kriging        K5 vs M5all            | K1 vs M1, K3 vs M3

Outputs: results/round2/summary/{rung_overview.csv,contrasts.csv,*.md},
         figs/round2/summary/*.png (600 dpi)
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
from round2 import evaluation  # noqa: E402

SUM_DIR = os.path.join(REPO, "results", "round2", "summary")
FIG_DIR = os.path.join(REPO, "figs", "round2", "summary")
SYNTH_BASE = "/mnt/nvme2/synthetic/wind/round2"

# park -> region (from results_summarize.ipynb, for consistent ordering)
REGION = {
    "01200_1": "SH-West", "01200_2": "SH-West", "00298": "MV",
    "04745": "NDS", "07374": "NRW-Münsterland", "01303_1": "NRW-Niederrhein",
    "01303_2": "NRW-Niederrhein", "05347": "Hessen-Nord", "02483": "Hessen-Nord",
    "00198_1": "S-Anhalt", "00198_2": "Hessen-Süd", "00282": "Thüringen",
    "05426": "RLP",
}
PARK_ORDER = sorted(REGION, key=lambda p: (REGION[p], p))

HEADLINE = [
    ("extrapolation (MOST)", "M5all", "M5all_PL"),
    ("aging (Weibull)", "M5all", "M5all_noage"),
    ("downscaling (QM)", "M5all", "M5all_noQM"),
    ("kriging (measured path)", "K5", "M5all"),
]
SECONDARY = [
    ("MOST incremental", "M3", "M2"),
    ("QM incremental", "M2", "M1"),
    ("aging const (all)", "M3allconst", "M3noage"),
    ("aging weibull (all)", "M4all", "M3noage"),
    ("aging weibull+step (all)", "M4ball", "M3noage"),
    ("kriging K1 vs M1", "K1", "M1"),
    ("kriging K3 vs M3", "K3", "M3"),
]
LADDER_M = ["M1", "M2", "M3", "M4all", "M5all"]
LADDER_K = ["K1", "K3", "K5"]
# paper rung naming (HANDOFF ask 7): server all-aging rungs = paper M4/M5
PAPER_NAME = {"M4all": "M4", "M5all": "M5", "M5all_PL": "M5-PL",
              "M5all_noQM": "M5-noQM", "M5all_noage": "M5-noage"}
def pname(r):
    return PAPER_NAME.get(r, r)


def load_scores(variant="clean") -> pd.DataFrame:
    s = pd.read_csv(os.path.join(SUM_DIR, "scores.csv"), dtype={"park_id": str})
    s = s[s["variant"] == variant].copy()
    s["abs_er_dev"] = (s["energy_ratio"] - 1.0).abs()
    return s.set_index(["experiment", "park_id"])


def paired(scores, exp_a, exp_b, metric):
    a = scores.loc[exp_a][metric] if exp_a in scores.index.get_level_values(0) else None
    b = scores.loc[exp_b][metric] if exp_b in scores.index.get_level_values(0) else None
    if a is None or b is None:
        return None
    return pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()


def contrast_row(scores, label, exp_a, exp_b):
    out = {"contrast": label, "a": exp_a, "b": exp_b}
    for metric, better in (("abs_er_dev", "lower"), ("r2", "higher")):
        pr = paired(scores, exp_a, exp_b, metric)
        if pr is None or pr.empty:
            continue
        st = evaluation.compare_pathways(pr["a"], pr["b"])
        out[f"{metric}_median_diff"] = st["median_diff"]
        out[f"{metric}_p"] = st["wilcoxon_p"]
        out[f"{metric}_rank_biserial"] = st["rank_biserial"]
        out[f"{metric}_ci"] = f"[{st['boot_ci_lo']:.3f}, {st['boot_ci_hi']:.3f}]"
        if metric == "abs_er_dev":
            out["improves"] = (st["median_diff"] < 0)
    return out


def holm(pvals):
    order = np.argsort(pvals)
    adj = np.empty(len(pvals))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(pvals) - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def fig_effect_matrix(scores, contrasts_df):
    rows = HEADLINE + SECONDARY[:2]
    mat = np.full((len(rows), len(PARK_ORDER)), np.nan)
    labels = []
    for ri, (label, a, b) in enumerate(rows):
        labels.append(label)
        pr = paired(scores, a, b, "abs_er_dev")
        if pr is None:
            continue
        for ci, pid in enumerate(PARK_ORDER):
            if pid in pr.index:
                mat[ri, ci] = pr.loc[pid, "a"] - pr.loc[pid, "b"]
    fig, ax = plt.subplots(figsize=(12, 4.5))
    vmax = np.nanmax(np.abs(mat)) or 0.1
    im = ax.pcolormesh(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(np.arange(len(PARK_ORDER)) + 0.5,
                  [f"{p}\n{REGION[p]}" for p in PARK_ORDER], fontsize=7)
    stats = contrasts_df.set_index("contrast")
    ylabels = []
    for label, a, b in rows:
        p = stats.loc[label, "abs_er_dev_p"] if label in stats.index else np.nan
        ylabels.append(f"{label}\n(p={p:.3f})" if np.isfinite(p) else label)
    ax.set_yticks(np.arange(len(rows)) + 0.5, ylabels, fontsize=8)
    fig.colorbar(im, label=r"$\Delta$ |energy ratio $-$ 1|  (blue = component improves)")
    ax.set_title("Component effect per park (DBSCAN-cleaned)")
    fig.savefig(os.path.join(FIG_DIR, "effect_matrix.png"), dpi=600,
                bbox_inches="tight")
    plt.close(fig)


def fig_ladder_spaghetti(scores):
    metrics = [("r2", "R²"), ("abs_er_dev", "|ER − 1|"), ("wasserstein", "W₁")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (metric, name) in zip(axes, metrics):
        for rungs, color in ((LADDER_M, "tab:blue"), (LADDER_K, "tab:orange")):
            xs = np.arange(len(rungs)) if rungs is LADDER_M else \
                np.arange(len(LADDER_M), len(LADDER_M) + len(rungs))
            per_park = {}
            for pid in PARK_ORDER:
                ys = [scores.loc[(r, pid), metric]
                      if (r, pid) in scores.index else np.nan for r in rungs]
                ax.plot(xs, ys, color=color, alpha=0.2, lw=0.8)
                per_park[pid] = ys
            med = np.nanmedian(np.array(list(per_park.values()), dtype=float), axis=0)
            ax.plot(xs, med, color=color, lw=2.5, marker="o",
                    label="ERA5 pathway" if rungs is LADDER_M else "measurement pathway (kriging)")
        ax.set_xticks(range(len(LADDER_M) + len(LADDER_K)),
                      [pname(r) for r in LADDER_M + LADDER_K],
                      rotation=45, fontsize=8)
        ax.set_title(name)
        ax.grid(alpha=0.3)
    axes[0].legend()
    fig.suptitle("Experiment ladder, 13 parks (thin) + median (bold) — DBSCAN-cleaned")
    fig.savefig(os.path.join(FIG_DIR, "ladder_spaghetti.png"), dpi=600,
                bbox_inches="tight")
    plt.close(fig)


def fig_dumbbells(scores):
    branches = pd.read_csv(os.path.join(REPO, "data", "round2",
                                        "branch_assignment.csv"),
                           dtype={"location_id": str})
    br = branches[branches["kind"] == "park"].set_index(
        branches.loc[branches["kind"] == "park", "location_id"]
        .str.replace("park_", "", regex=False))["branch"]
    fig, axes = plt.subplots(1, len(HEADLINE), figsize=(4 * len(HEADLINE), 5),
                             sharey=True)
    colors = {"A": "tab:green", "B": "tab:orange", "C": "tab:gray"}
    for ax, (label, a, b) in zip(axes, HEADLINE):
        pr = paired(scores, a, b, "abs_er_dev")
        if pr is None:
            continue
        for yi, pid in enumerate(PARK_ORDER):
            if pid not in pr.index:
                continue
            c = colors.get(br.get(pid, "C"), "tab:gray") if "QM" in label or \
                "downscaling" in label else "tab:blue"
            ax.plot([pr.loc[pid, "b"], pr.loc[pid, "a"]], [yi, yi], color=c,
                    lw=1.2, alpha=0.7)
            ax.scatter([pr.loc[pid, "b"]], [yi], color="lightgray", s=25,
                       zorder=3)
            ax.scatter([pr.loc[pid, "a"]], [yi], color=c, s=30, zorder=4)
        ax.set_title(f"{label}\n({pname(b)} → {pname(a)})", fontsize=9)
        ax.set_xlabel("|ER − 1|")
        ax.grid(alpha=0.3, axis="x")
    axes[0].set_yticks(range(len(PARK_ORDER)), PARK_ORDER, fontsize=8)
    fig.suptitle("Leave-one-out contrasts at the full chain (grey = without, colored = with component)")
    fig.savefig(os.path.join(FIG_DIR, "dumbbells_headline.png"), dpi=600,
                bbox_inches="tight")
    plt.close(fig)


def fig_kriging_week(scores):
    from round2 import meterdata
    pid = "04745"
    window = ("2024-02-05", "2024-02-12")
    fig, ax = plt.subplots(figsize=(11, 4))
    meas = meterdata.load_park_power(pid, window) / 1e6
    ax.plot(meas.index, meas.values, "k-", lw=1.8, label="measured")
    for exp, color in (("M5all", "tab:blue"), ("K5", "tab:orange")):
        path = os.path.join(SYNTH_BASE, exp, f"synth_{pid}.csv")
        if not os.path.exists(path):
            continue
        s = pd.read_csv(path, sep=";", index_col=0, parse_dates=True) \
            .loc[window[0]:window[1], "power_park"] / 1e6
        ax.plot(s.index, s.values, color=color, lw=1.2, alpha=0.9, label=pname(exp))
    ax.set_ylabel("Park power [MW]")
    ax.set_title(f"Example week, park {pid} (Jettebruch): full ERA5 chain vs measurement pathway")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig(os.path.join(FIG_DIR, "kriging_example_week.png"), dpi=600,
                bbox_inches="tight")
    plt.close(fig)


def fig_bar_grids(scores_clean, scores_raw):
    rungs = ["M1", "M3", "M5all", "K5"]
    for metric, name, ref in (("r2", "R²", 1.0), ("abs_er_dev", "|ER−1|", 0.0)):
        fig, axes = plt.subplots(5, 3, figsize=(13, 14), sharey=True)
        for ax, pid in zip(axes.flat, PARK_ORDER):
            vals = [scores_clean.loc[(r, pid), metric]
                    if (r, pid) in scores_clean.index else np.nan for r in rungs]
            ax.bar(range(len(rungs)), vals, color="seagreen")
            ax.axhline(ref, color="red", ls="--", lw=1)
            ax.set_xticks(range(len(rungs)), [pname(r) for r in rungs], fontsize=8)
            ax.set_title(f"{pid} ({REGION[pid]})", fontsize=9)
        for ax in axes.flat[len(PARK_ORDER):]:
            ax.axis("off")
        fig.suptitle(f"{name} per park and main rung (DBSCAN-cleaned)")
        fig.savefig(os.path.join(FIG_DIR, f"bars_{metric}.png"), dpi=600,
                    bbox_inches="tight")
        plt.close(fig)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    # tables-only: figs in FIG_DIR are owned by regen_paper_figs.py (asks
    # 13/15 style) — do not overwrite them from here
    ap.add_argument("--tables-only", action="store_true")
    cli = ap.parse_args()
    os.makedirs(SUM_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    scores_clean = load_scores("clean")
    scores_raw = load_scores("raw")

    # ---- contrast table (clean; raw companion) ----
    rows = []
    for variant, sc in (("clean", scores_clean), ("raw", scores_raw)):
        for label, a, b in HEADLINE + SECONDARY:
            r = contrast_row(sc, label, a, b)
            r["variant"] = variant
            rows.append(r)
    contrasts = pd.DataFrame(rows)
    for variant in ("clean", "raw"):
        m = (contrasts["variant"] == variant) & contrasts["contrast"].isin(
            [h[0] for h in HEADLINE])
        pv = contrasts.loc[m, "abs_er_dev_p"].values.astype(float)
        contrasts.loc[m, "abs_er_dev_p_holm"] = holm(pv)
    contrasts.to_csv(os.path.join(SUM_DIR, "contrasts.csv"), index=False)
    cc = contrasts[contrasts["variant"] == "clean"]
    print(cc[["contrast", "abs_er_dev_median_diff", "abs_er_dev_p",
              "abs_er_dev_p_holm", "abs_er_dev_rank_biserial", "improves"]]
          .round(4).to_string(index=False))

    # ---- rung overview (clean | raw medians) ----
    ov = []
    for exp in sorted(scores_clean.index.get_level_values(0).unique()):
        row = {"experiment": exp}
        for tag, sc in (("clean", scores_clean), ("raw", scores_raw)):
            if exp not in sc.index.get_level_values(0):
                continue
            sub = sc.loc[exp]
            row.update({f"r2_{tag}": sub["r2"].median(),
                        f"absERdev_{tag}": sub["abs_er_dev"].median(),
                        f"W1_{tag}": sub["wasserstein"].median(),
                        f"n_parks_{tag}": int(sub["r2"].notna().sum())})
        ov.append(row)
    overview = pd.DataFrame(ov).round(4)
    overview.to_csv(os.path.join(SUM_DIR, "rung_overview.csv"), index=False)
    with open(os.path.join(SUM_DIR, "summary.md"), "w") as f:
        f.write("# Round-2 Summary (Hauptzahlen: DBSCAN-bereinigt)\n\n")
        f.write("## Sprossen-Übersicht (Mediane)\n\n")
        f.write(overview.to_markdown(index=False) + "\n\n")
        f.write("## Kontraste (clean)\n\n")
        f.write(cc.drop(columns=["variant"]).round(4).to_markdown(index=False) + "\n")
    print("\n" + overview.to_string(index=False))

    # ---- figures ----
    if cli.tables_only:
        print("tables-only: figures skipped (owned by regen_paper_figs.py)")
        return
    fig_effect_matrix(scores_clean, cc)
    fig_ladder_spaghetti(scores_clean)
    fig_dumbbells(scores_clean)
    fig_kriging_week(scores_clean)
    fig_bar_grids(scores_clean, scores_raw)
    print(f"\nfigures -> {FIG_DIR}")


if __name__ == "__main__":
    main()
