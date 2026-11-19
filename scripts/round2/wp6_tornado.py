#!/usr/bin/env python3
"""Ask 20b — main-effects / tornado figure from the fast chain.

For each chain parameter: sweep over its SA range with everything else at
the full-chain baseline (M5: QM height-consistent, MOST, fleet Weibull,
wakes on, density v1_mixed) and record the fleet-level response:
  d_energy_pct = (sum_p E_synth(theta) - sum_p E_synth(base)) / sum_p E_synth(base) * 100
  pooled |ER-1| = mean over parks of |ER-1|
Additionally an explicit age-scaling sweep (all park ages x 1.1/1.2/1.3,
aging input only) — the paper quotes these numbers directly.

Outputs: figs/round2/summary/tornado_main_effects.png (ask-13 fonts)
         results/round2/summary/main_effects.csv (+ copy next to figs)
"""

import os
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
from round2.chain import Chain  # noqa: E402
from round2.paper_style import SA_NAME, apply_print_style  # noqa: E402

SUM_DIR = os.path.join(REPO, "results", "round2", "summary")
FIG_DIR = os.path.join(REPO, "figs", "round2", "summary")

BASELINE = {"wind_level_factor": 1.0, "correction": "height_consistent",
            "shear": "most", "z0_scale": 1.0,
            "aging_model": "weibull", "aging_lambda": 54.5, "aging_kappa": 2.0,
            "power_curve_scale": 1.0, "density": "v1_mixed",
            "wake_k": 0.075, "wake_enabled": True}

# parameter -> list of (level_label, theta_override)
SWEEPS = {
    "wind_level_factor": [(f"{v:.2f}", {"wind_level_factor": v})
                          for v in (0.95, 0.975, 1.025, 1.05)],
    "correction_mode": [("off", {"correction": "off"}),
                        ("10 m only", {"correction": "wind10_only"})],
    # ask 29: the missing component lever — aging fully off (DF = 1,
    # everything else at the S3 baseline == the S3-noage state)
    "aging_model": [("off", {"aging_model": "off"})],
    "shear_method": [("power law", {"shear": "power_law"})],
    "z0_scale": [(f"x{v:g}", {"z0_scale": v}) for v in (0.5, 0.71, 1.41, 2.0)],
    "aging_lambda": [(f"{v:g}", {"aging_lambda": v}) for v in (45, 50, 60, 70)],
    "aging_kappa": [(f"{v:g}", {"aging_kappa": v}) for v in (1.5, 2.0, 2.5)],
    "power_curve_scale": [(f"{v:.2f}", {"power_curve_scale": v})
                          for v in (0.97, 1.03)],
    "density_mode": [("static 1.225", {"density": "static_1225"}),
                     ("dynamic", {"density": "dynamic"})],
    "wake_k": [(f"{v:g}", {"wake_k": v}) for v in (0.05, 0.10)],
}
AGE_FACTORS = (1.1, 1.2, 1.3)


def fleet_stats(chain, theta):
    """(fleet energy [energy-weighted], pooled mean |ER-1|, per-park ERs)."""
    ers, energy = [], 0.0
    for pc in chain.parks:
        er = pc.energy_ratio(theta)
        e_meas = np.nansum(pc.meas)
        energy += er * e_meas
        ers.append(er)
    ers = np.array(ers)
    return energy, float(np.mean(np.abs(ers - 1.0))), ers


def main():
    apply_print_style()
    chain = Chain()
    # SA baseline = paper-S3 (server-internal id M5all) -- ALL parks aged. Without
    # this the chain honours the round-1 apply_ageing flags (5/13 parks) and
    # every aging lever in the tornado is understated vs the headline rungs.
    for pc in chain.parks:
        pc.force_all_aging()
    n_aged = sum(pc.ages is not None for pc in chain.parks)
    print(f"parks cached: {len(chain.parks)} ({n_aged} aged — all-aging baseline)")
    e0, aed0, ers0 = fleet_stats(chain, BASELINE)
    rows = [{"parameter": "baseline", "level": "S3 baseline",
             "d_energy_pct": 0.0, "d_energy_pct_parkmean": 0.0,
             "pooled_abs_er_dev": aed0}]

    for param, levels in SWEEPS.items():
        for label, over in levels:
            e, aed, ers = fleet_stats(chain, {**BASELINE, **over})
            rows.append({"parameter": param, "level": label,
                         "d_energy_pct": (e - e0) / e0 * 100.0,
                         "d_energy_pct_parkmean": float(
                             np.mean(ers / ers0 - 1.0)) * 100.0,
                         "pooled_abs_er_dev": aed})
            print(f"{param:20s} {label:14s} dE={rows[-1]['d_energy_pct']:+.2f} %")

    # explicit age-scaling sweep (aging input only)
    originals = [pc.ages for pc in chain.parks]
    for f in AGE_FACTORS:
        for pc, ages in zip(chain.parks, originals):
            pc.ages = None if ages is None else ages * f
        e, aed, ers = fleet_stats(chain, BASELINE)
        rows.append({"parameter": "age_scaling", "level": f"x{f:g}",
                     "d_energy_pct": (e - e0) / e0 * 100.0,
                     "d_energy_pct_parkmean": float(
                         np.mean(ers / ers0 - 1.0)) * 100.0,
                     "pooled_abs_er_dev": aed})
        print(f"{'age_scaling':20s} x{f:<13g} dE={rows[-1]['d_energy_pct']:+.2f} %")
    for pc, ages in zip(chain.parks, originals):
        pc.ages = ages

    df = pd.DataFrame(rows)
    out_csv = os.path.join(SUM_DIR, "main_effects.csv")
    df.round(4).to_csv(out_csv, index=False)
    shutil.copy(out_csv, os.path.join(FIG_DIR, "main_effects.csv"))
    print("csv:", out_csv)

    # ---- tornado: per parameter min..max d_energy_pct across its levels ----
    # ask 29 (2): two labeled blocks — component on/off switches vs
    # parameter uncertainties — instead of one flat strength sort.
    NAME = dict(SA_NAME)
    NAME["age_scaling"] = "Park age scaling (x1.1-x1.3)"
    NAME.setdefault("aging_model", "Aging model")
    BLOCKS = [  # (block label, parameters) — drawn top block first
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
    assert not missing, f"tornado blocks out of sync with SWEEPS: {missing}"

    GAP = 1.6  # rows between the blocks
    ys, params, block_hdr = [], [], []  # bar positions, order, header positions
    y_cur = 0.0
    for label, ps in reversed(BLOCKS):  # bottom block first (barh grows up)
        ordered = agg.loc[ps].sort_values("strength")  # weakest at bottom
        for p in ordered.index:
            ys.append(y_cur)
            params.append(p)
            y_cur += 1.0
        block_hdr.append((label, y_cur - 0.55))
        y_cur += GAP
    ys = np.array(ys)
    ordered_agg = agg.loc[params]

    fig, ax = plt.subplots(figsize=(10.5, 8.5))
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
            ax.text(r["min"] - 0.25, yi, lo, va="center", ha="right",
                    fontsize=13.5)
        if r["max"] > 0.05:
            ax.text(r["max"] + 0.25, yi, hi, va="center", ha="left",
                    fontsize=13.5)
    xmax = max(abs(agg["min"].min()), agg["max"].max()) * 1.35
    for label, y_hdr in block_hdr:
        ax.text(-xmax * 0.98, y_hdr + 0.45, label, fontsize=15,
                fontweight="bold", va="bottom", ha="left")
    ax.set_yticks(ys, [NAME.get(p, p) for p in params], fontsize=16)
    ax.axvline(0, color="k", lw=1.2)
    ax.set_xlabel("Fleet energy change vs. S3 baseline [%]")
    ax.grid(alpha=0.3, axis="x")
    ax.set_xlim(-xmax, xmax)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "tornado_main_effects.png"), dpi=600,
                bbox_inches="tight")
    print("figure: figs/round2/summary/tornado_main_effects.png")


if __name__ == "__main__":
    main()
