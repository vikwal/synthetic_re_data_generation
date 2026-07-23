#!/usr/bin/env python3
"""Ask 30 — wake-effect stratification by park size + Jensen magnitude
calibration (analog of the aging old-tail analysis).

Canonical contrast throughout: paired M4all vs M5all, CLEAN variant from
results/round2/summary/scores.csv (the paper contrast; see the item-0
definition note — the earlier −0.074/p=0.054 companion row was the RAW
server-ladder M4→M5 contrast from ladder_metrics.csv).

1. Threshold sweep n_min ∈ {2,3,4,5,7} with the mandatory minimum-attainable
   two-sided p column (2/2^N).
2. Jensen magnitude calibration: realized ΔER = ER(M5all) − ER(M4all) vs the
   modeled expectation −ℓ·ER(M4all), ℓ = 1 − mean(w) at k=0.075 over the
   scoring window. Identity line, OLS slope + 95 % CI, Spearman(ℓ, realized
   |ER−1| improvement). Pseudonyms + ask-13 fonts.

Outputs (no canonical artifacts touched):
    results/round2/ask30/{wake_stratified.csv, wake_calibration.csv,
                          fig_wake_calibration.png, definition_note.md}
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
from round2 import evaluation  # noqa: E402
from round2.paper_style import PSEUDONYM, apply_print_style  # noqa: E402

OUT = os.path.join(REPO, "results", "round2", "ask30")
WAKE_DIR = "/mnt/nvme2/synthetic/raw/round2/wakes"
WAKE_K = 0.075
WINDOW = ("2023-07-24", "2024-06-01")  # synth output start .. scoring end
A, B = "M4all", "M5all"
THRESHOLDS = (2, 3, 4, 5, 7)


def clean_scores() -> pd.DataFrame:
    sc = pd.read_csv(os.path.join(REPO, "results", "round2", "summary",
                                  "scores.csv"), dtype={"park_id": str})
    sc = sc[sc.variant == "clean"]
    return sc.pivot_table(index="park_id", columns="experiment",
                          values=["energy_ratio", "r2"])


def threshold_sweep(piv, n_turb) -> pd.DataFrame:
    er, r2 = piv["energy_ratio"], piv["r2"]
    d_abs = (er[B] - 1).abs() - (er[A] - 1).abs()   # <0 = wakes improve
    d_r2 = r2[B] - r2[A]                            # >0 = wakes improve
    rows = []
    for n_min in THRESHOLDS:
        ids = [p for p in er.index if n_turb.get(p, 0) >= n_min]
        n = len(ids)
        st_er = evaluation.compare_pathways((er[B] - 1).abs().loc[ids],
                                            (er[A] - 1).abs().loc[ids])
        st_r2 = evaluation.compare_pathways(r2[B].loc[ids], r2[A].loc[ids])
        rows.append({
            "n_min_turbines": n_min, "N": n,
            "median_dR2": float(d_r2.loc[ids].median()),
            "median_dAbsERdev": float(d_abs.loc[ids].median()),
            "wilcoxon_p_absERdev": st_er.get("wilcoxon_p"),
            "wilcoxon_p_r2": st_r2.get("wilcoxon_p"),
            "rank_biserial_absERdev": st_er.get("rank_biserial"),
            "n_improving_absERdev": int((d_abs.loc[ids] < 0).sum()),
            "n_improving_r2": int((d_r2.loc[ids] > 0).sum()),
            "p_min_attainable": 2.0 / 2 ** n,
        })
    return pd.DataFrame(rows)


def calibration(piv, n_turb) -> pd.DataFrame:
    er = piv["energy_ratio"]
    rows = []
    for pid in er.index:
        if n_turb.get(pid, 0) < 3:
            continue
        w = pd.read_parquet(os.path.join(WAKE_DIR, f"w_{pid}_k{WAKE_K:.3f}.parquet"))
        loss = 1.0 - float(w.loc[WINDOW[0]:WINDOW[1], "w"].mean())
        er_a, er_b = float(er.loc[pid, A]), float(er.loc[pid, B])
        rows.append({
            "park_id": pid, "pseudonym": PSEUDONYM.get(pid, pid),
            "n_turbines": int(n_turb[pid]),
            "wake_loss_modeled": loss,
            "er_m4all": er_a, "er_m5all": er_b,
            "d_er_realized": er_b - er_a,
            "d_er_expected": -loss * er_a,
            "abs_dev_improvement": abs(er_a - 1) - abs(er_b - 1),
        })
    return pd.DataFrame(rows)


# manual label placement for the two crowded clusters (offset pts, ha)
LABEL_POS = {
    "North Rhine-Westphalia North": ((-12, 16), "right"),
    "Hesse South": ((-12, 1), "right"),
    "North Rhine-Westphalia East": ((-12, -16), "right"),
    "Saxony-Anhalt": ((-12, 10), "right"),
    "Thuringia": ((10, -18), "left"),
    "North Rhine-Westphalia West": ((-12, -6), "right"),
}


def fig_calibration(cal: pd.DataFrame, path: str):
    apply_print_style()
    x, y = cal["d_er_expected"].values, cal["d_er_realized"].values
    fig, ax = plt.subplots(figsize=(9, 7.5))
    ax.scatter(x, y, s=140, color="tab:blue", zorder=3)
    for _, r in cal.iterrows():
        (dx, dy), ha = LABEL_POS.get(r["pseudonym"], ((8, 6), "left"))
        ax.annotate(r["pseudonym"], (r["d_er_expected"], r["d_er_realized"]),
                    textcoords="offset points", xytext=(dx, dy), ha=ha,
                    fontsize=13.5)
    lim = min(x.min(), y.min()) * 1.15
    ax.plot([lim, 0], [lim, 0], "k--", lw=1.2, label="identity (Jensen exact)")
    ax.set_xlabel(r"expected $\Delta$ER $= -\ell \cdot$ ER(M4)")
    ax.set_ylabel(r"realized $\Delta$ER $=$ ER(M5) $-$ ER(M4)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    piv = clean_scores()
    n_turb = evaluation.park_turbine_counts()

    sweep = threshold_sweep(piv, n_turb)
    sweep.round(4).to_csv(os.path.join(OUT, "wake_stratified.csv"), index=False)
    print("=== 1. threshold sweep (paired M4all vs M5all, clean) ===")
    print(sweep.round(4).to_string(index=False))

    cal = calibration(piv, n_turb)
    # OLS realized ~ expected (with intercept) + through-origin slope
    lr = stats.linregress(cal["d_er_expected"], cal["d_er_realized"])
    tcrit = stats.t.ppf(0.975, len(cal) - 2)
    slope_ci = (lr.slope - tcrit * lr.stderr, lr.slope + tcrit * lr.stderr)
    slope0 = float(np.sum(cal.d_er_expected * cal.d_er_realized)
                   / np.sum(cal.d_er_expected ** 2))
    rho, rho_p = stats.spearmanr(cal["wake_loss_modeled"],
                                 cal["abs_dev_improvement"])
    cal.round(5).to_csv(os.path.join(OUT, "wake_calibration.csv"), index=False)
    fig_calibration(cal, os.path.join(OUT, "fig_wake_calibration.png"))

    print("\n=== 2. Jensen magnitude calibration (N=11 multi-turbine) ===")
    print(cal[["pseudonym", "n_turbines", "wake_loss_modeled",
               "d_er_expected", "d_er_realized",
               "abs_dev_improvement"]].round(4).to_string(index=False))
    print(f"\nOLS slope = {lr.slope:.3f}  95% CI [{slope_ci[0]:.3f}, "
          f"{slope_ci[1]:.3f}]  intercept = {lr.intercept:+.4f}  "
          f"R2 = {lr.rvalue**2:.3f}")
    print(f"through-origin slope = {slope0:.3f}")
    print(f"Spearman(modeled loss, |ER-1| improvement): rho = {rho:.3f}, "
          f"p = {rho_p:.4f}")

    with open(os.path.join(OUT, "definition_note.md"), "w") as f:
        f.write(
            "# Ask 30 item 0 — definition note\n\n"
            "The '-0.074 / p=0.054' companion row in ablation_summary.md is the\n"
            "RAW-variant server-ladder contrast M4 vs M5 (ladder_metrics.csv;\n"
            "5-park aging), reproduced exactly: all-13 median d|ER-1| -0.0253 /\n"
            "n>=3 -0.0737, exact Wilcoxon p=0.0537 (identical for both sets —\n"
            "the two single-turbine zero diffs never enter the statistic).\n\n"
            "The CANONICAL paper contrast is M4all vs M5all, CLEAN variant\n"
            "(scores.csv): all-13 median d|ER-1| -0.0237 / n>=3 -0.0240,\n"
            "exact Wilcoxon p=0.577 — consistent with the manuscript quote\n"
            "(dR2 +0.019, d|ER-1| -0.024, p >= 0.10). The ablation row differs\n"
            "in BOTH variant (raw vs clean) and rung family (M4/M5 vs\n"
            "M4all/M5all); it stays in ablation_summary.md as the ladder\n"
            "diagnostic but must not be quoted as the paper wake contrast.\n")
    print(f"\nwritten -> {OUT}")


if __name__ == "__main__":
    main()
