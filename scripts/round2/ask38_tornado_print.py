#!/usr/bin/env python3
"""Print-size regeneration of the two-block tornado (manuscript Fig. 5.18).

Plot-only: reads the ask-29 sweep results from
results/round2/summary/main_effects.csv (NO chain re-runs) and re-renders
figs/round2/summary/tornado_main_effects.png with fonts sized for a
single-column figure (~3x shrink in print), matching the placebo/Morris/ACF
print style. Plotting geometry mirrors wp6_tornado.py — keep the two in sync.
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
from round2.paper_style import SA_NAME  # noqa: E402

SUM_DIR = os.path.join(REPO, "results", "round2", "summary")
FIG_DIR = os.path.join(REPO, "figs", "round2", "summary")

df = pd.read_csv(os.path.join(SUM_DIR, "main_effects.csv"))

NAME = dict(SA_NAME)
# levels x1.1-x1.3 are annotated on the bar; keep the tick label short
NAME["age_scaling"] = "Park age scaling"
NAME.setdefault("aging_model", "Aging model")
BLOCKS = [
    ("Component switches",
     ["correction_mode", "aging_model", "shear_method",
      "density_mode", "wake_k"]),
    ("Parameter uncertainties",
     ["wind_level_factor", "power_curve_scale", "aging_lambda",
      "aging_kappa", "z0_scale", "age_scaling"]),
]
sweep = df[df["parameter"] != "baseline"]
agg = sweep.groupby("parameter")["d_energy_pct"].agg(["min", "max"])
agg["strength"] = np.maximum(agg["min"].abs(), agg["max"].abs())
missing = {p for _, ps in BLOCKS for p in ps} ^ set(agg.index)
assert not missing, f"tornado blocks out of sync with sweeps: {missing}"

GAP = 1.9  # rows between the blocks (room for the larger block headers)
ys, params, block_hdr = [], [], []
y_cur = 0.0
for label, ps in reversed(BLOCKS):
    ordered = agg.loc[ps].sort_values("strength")
    for p in ordered.index:
        ys.append(y_cur)
        params.append(p)
        y_cur += 1.0
    block_hdr.append((label, y_cur - 0.55))
    y_cur += GAP
ys = np.array(ys)
ordered_agg = agg.loc[params]

fig, ax = plt.subplots(figsize=(9, 7))
ax.barh(ys, ordered_agg["max"].clip(lower=0), color="tab:blue",
        alpha=0.85, label="increase")
ax.barh(ys, ordered_agg["min"].clip(upper=0), color="tab:red",
        alpha=0.85, label="decrease")
for yi, (param, r) in zip(ys, ordered_agg.iterrows()):
    lo = sweep[(sweep.parameter == param) &
               (sweep.d_energy_pct == r["min"])]["level"].iloc[0]
    hi = sweep[(sweep.parameter == param) &
               (sweep.d_energy_pct == r["max"])]["level"].iloc[0]
    if r["min"] < -0.05:
        ax.text(r["min"] - 0.25, yi, lo, va="center", ha="right", fontsize=19)
    if r["max"] > 0.05:
        ax.text(r["max"] + 0.25, yi, hi, va="center", ha="left", fontsize=19)
xmax = max(abs(agg["min"].min()), agg["max"].max()) * 1.35
for label, y_hdr in block_hdr:
    ax.text(-xmax * 0.98, y_hdr + 0.45, label, fontsize=22,
            fontweight="bold", va="bottom", ha="left")
ax.set_yticks(ys, [NAME.get(p, p) for p in params], fontsize=24)
ax.tick_params(axis="x", labelsize=22)
ax.axvline(0, color="k", lw=1.2)
# baseline reference lives in the caption; the long form overflows the width
ax.set_xlabel("Fleet energy change [%]", fontsize=26)
ax.grid(alpha=0.3, axis="x")
ax.set_xlim(-xmax, xmax)
fig.tight_layout()
out = os.path.join(FIG_DIR, "tornado_main_effects.png")
fig.savefig(out, dpi=600, bbox_inches="tight")
print("wrote", out)
