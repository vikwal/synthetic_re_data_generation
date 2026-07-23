#!/usr/bin/env python3
"""Placebo permutation-null figure for the manuscript (aging section).

Reads results/round2/ask26/perm_null_alt.csv (1000 age-permutation draws +
true-assignment row, T_mean statistic = fleet-mean Delta|ER-1| improvement
over the no-aging twin), plots the null histogram with the true value marked,
for M2 and M5 (the two rungs quoted in the text, p_perm = 0.003 / 0.009).
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
from round2.paper_style import apply_print_style, rung  # noqa: E402

FIG_DIR = os.path.join(REPO, "figs", "round2", "summary")
os.makedirs(FIG_DIR, exist_ok=True)

df = pd.read_csv(os.path.join(REPO, "results", "round2", "ask26", "perm_null_alt.csv"),
                  comment="#")

apply_print_style()
matplotlib.rcParams.update({
    "font.size": 17, "axes.labelsize": 19,
    "xtick.labelsize": 15, "ytick.labelsize": 15,
    "legend.fontsize": 15,
})

fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)

# renumbering: data keyed by literal "M2"/"M5" (unrelated to PAPER_RUNG),
# "M5" here is the old-numbering full stability chain, now displayed as S3.
DISPLAY = {"M2": "M2", "M5": "S3"}
for ax, model in zip(axes, ["M2", "M5"]):
    sub = df[df["model"] == model]
    null = sub[sub["draw"] != "true"]["T_mean"].astype(float)
    true_val = float(sub[sub["draw"] == "true"]["T_mean"].iloc[0])
    p_perm = ((null >= true_val).sum() + 1) / (len(null) + 1)

    ax.hist(null, bins=30, color="lightsteelblue", edgecolor="white", zorder=2)
    ax.axvline(true_val, color="firebrick", lw=2.4, zorder=3, label="True ages")
    title = DISPLAY[model] + r" ($p_{\mathrm{perm}}=" + f"{p_perm:.3f}" + "$)"
    ax.set_title(title)
    ax.set_xlabel(r"Fleet-mean $T_{\mathrm{mean}}$ ($\Delta|ER-1|$)")
    ax.legend(loc="upper left", frameon=False)

axes[0].set_ylabel("Draws (of 1000)")
fig.tight_layout()
out_path = os.path.join(FIG_DIR, "placebo_hist.png")
fig.savefig(out_path, dpi=600, bbox_inches="tight")
print("wrote", out_path)
