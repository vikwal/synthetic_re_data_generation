#!/usr/bin/env python3
"""WP6.3 — Morris screening over the guide's parameter table (SALib).

Output scalar: mean over parks of |energy_ratio - 1| (round2.chain).
r = 30 trajectories -> (k+1)*r chain runs. Deliverable: mu*-sigma plot +
ranking CSV.
"""

import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
from round2.chain import Chain  # noqa: E402

from SALib.sample.morris import sample as morris_sample
from SALib.analyze.morris import analyze as morris_analyze

# guide 6.2 parameter table; categoricals encoded as levels
# ask 29 (3): aging_model {weibull, none} added as 10th factor.
PROBLEM = {
    "num_vars": 10,
    "names": ["wind_level_factor", "correction_mode", "shear_method",
              "z0_scale", "aging_lambda", "aging_kappa",
              "power_curve_scale", "density_mode", "wake_k", "aging_model"],
    "bounds": [
        [0.95, 1.05],    # wind level factor
        [0.0, 3.0],      # correction: [0,1) off, [1,2) wind10_only, [2,3] height_consistent
        [0.0, 2.0],      # shear: [0,1) power_law, [1,2] most
        [np.log(0.5), np.log(2.0)],  # z0 x[0.5,2] log-uniform
        [45.0, 70.0],    # aging lambda
        [1.5, 2.5],      # aging kappa
        [0.97, 1.03],    # power curve scale
        [0.0, 2.0],      # density: [0,1) static_1225, [1,2] dynamic
        [0.05, 0.10],    # wake k
        [0.0, 2.0],      # aging: [0,1) weibull, [1,2] none (DF=1)
    ],
}

CORR = ["off", "wind10_only", "height_consistent"]
DENS = ["static_1225", "dynamic"]
SHEAR = ["power_law", "most"]
AGING = ["weibull", "none"]

# ask 29 stability guard: mu* of the 9-factor screening quoted in the
# manuscript (results/round2/morris_ranking.csv at the time of ask 29)
MU_STAR_REFERENCE = {
    "wind_level_factor": 0.1311, "correction_mode": 0.0834,
    "power_curve_scale": 0.0344, "aging_kappa": 0.0104,
    "aging_lambda": 0.0082, "density_mode": 0.0082, "wake_k": 0.0079,
    "shear_method": 0.0074, "z0_scale": 0.0036,
}
MU_STAR_TOLERANCE = 0.005


def decode(x):
    return {
        "wind_level_factor": float(x[0]),
        "correction": CORR[min(int(x[1]), 2)],
        "shear": SHEAR[min(int(x[2]), 1)],
        "z0_scale": float(np.exp(x[3])),
        "aging_model": AGING[min(int(x[9]), 1)] if len(x) > 9 else "weibull",
        "aging_lambda": float(x[4]),
        "aging_kappa": float(x[5]),
        "power_curve_scale": float(x[6]),
        "density": DENS[min(int(x[7]), 1)],
        "wake_k": float(x[8]),
        "wake_enabled": True,
    }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--factors", type=int, choices=(9, 10), default=9,
                    help="9 = canonical screening (manuscript); "
                         "10 = ask-29 companion incl. aging_model switch")
    args = ap.parse_args()

    problem = dict(PROBLEM)
    if args.factors == 9:
        problem["num_vars"] = 9
        problem["names"] = PROBLEM["names"][:9]
        problem["bounds"] = PROBLEM["bounds"][:9]
    suffix = "" if args.factors == 9 else "_ask29_10factor"

    chain = Chain()
    # all-aging SA baseline (paper: S3), see wp6_tornado.py
    for pc in chain.parks:
        pc.force_all_aging()
    print(f"parks cached: {len(chain.parks)} (all-aging baseline, "
          f"{args.factors} factors)")
    X = morris_sample(problem, N=30, num_levels=4, seed=42)
    print(f"chain runs: {len(X)}")
    Y = np.array([chain.run_chain(decode(x)) for x in X])
    print(f"output: mean={Y.mean():.4f} min={Y.min():.4f} max={Y.max():.4f}")
    res = morris_analyze(problem, X, Y, num_levels=4, seed=42)
    df = pd.DataFrame({"parameter": problem["names"], "mu_star": res["mu_star"],
                       "mu": res["mu"], "sigma": res["sigma"],
                       "mu_star_conf": res["mu_star_conf"]})
    df = df.sort_values("mu_star", ascending=False)

    # informational: shifts vs the pre-ask-29 (5-park-aging baseline) quotes;
    # after the ask-29(b) baseline switch ALL manuscript quotes are re-drawn
    shifted = []
    for param, ref in MU_STAR_REFERENCE.items():
        new = float(df.loc[df.parameter == param, "mu_star"].iloc[0])
        if abs(new - ref) > MU_STAR_TOLERANCE:
            shifted.append(f"{param}: {ref:.4f} -> {new:.4f}")
    if shifted:
        print("mu* shifts vs pre-ask-29 manuscript values (expected after "
              "the all-aging baseline switch):\n  " + "\n  ".join(shifted))
    if args.factors == 10:
        aging_mu = float(df.loc[df.parameter == "aging_model", "mu_star"].iloc[0])
        aging_rank = int((df.reset_index(drop=True).parameter == "aging_model")
                         .idxmax()) + 1
        print(f"aging_model switch: mu* = {aging_mu:.4f} (rank {aging_rank})")

    out_csv = os.path.join(REPO, "results", "round2",
                           f"morris_ranking{suffix}.csv")
    df.to_csv(out_csv, index=False)
    print(df.round(4).to_string(index=False))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(df["mu_star"], df["sigma"])
    for _, r in df.iterrows():
        ax.annotate(r["parameter"], (r["mu_star"], r["sigma"]),
                    textcoords="offset points", xytext=(5, 3), fontsize=8)
    ax.set_xlabel(r"$\mu^*$ (mean |elementary effect| on mean |ER-1|)")
    ax.set_ylabel(r"$\sigma$ (interaction / nonlinearity)")
    ax.set_title("Morris screening (r=30), output: mean |energy ratio - 1|")
    fig.savefig(os.path.join(REPO, "figs", "round2",
                             f"morris_mu_star{suffix}.png"),
                dpi=200, bbox_inches="tight")
    print(f"figure: figs/round2/morris_mu_star{suffix}.png")


if __name__ == "__main__":
    main()
