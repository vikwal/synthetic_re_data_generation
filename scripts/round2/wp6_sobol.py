#!/usr/bin/env python3
"""WP6.4 — Sobol indices on the top-4 Morris factors (Saltelli, N=512).

Reads results/round2/morris_ranking.csv, keeps the top 4 CONTINUOUS factors
(categoricals are reported via Morris only), fixes the rest at baseline.
"""

import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
from round2.chain import Chain  # noqa: E402

from SALib.sample import saltelli
from SALib.analyze import sobol

CONTINUOUS = {
    "wind_level_factor": [0.95, 1.05],
    "z0_scale_log": [np.log(0.5), np.log(2.0)],
    "aging_lambda": [45.0, 70.0],
    "aging_kappa": [1.5, 2.5],
    "power_curve_scale": [0.97, 1.03],
    "wake_k": [0.05, 0.10],
}
NAME_MAP = {"z0_scale": "z0_scale_log"}


def main():
    ranking = pd.read_csv(os.path.join(REPO, "results", "round2",
                                       "morris_ranking.csv"))
    top = []
    for name in ranking["parameter"]:
        key = NAME_MAP.get(name, name)
        if key in CONTINUOUS and key not in top:
            top.append(key)
        if len(top) == 4:
            break
    print("top-4 continuous factors:", top)
    problem = {"num_vars": 4, "names": top,
               "bounds": [CONTINUOUS[k] for k in top]}
    X = saltelli.sample(problem, 512, calc_second_order=False)
    print(f"chain runs: {len(X)}")

    chain = Chain()
    # ask 29 (b): all-aging SA baseline (paper-M5), see wp6_tornado.py
    for pc in chain.parks:
        pc.force_all_aging()

    def theta_of(x):
        th = {"aging_model": "weibull", "wake_enabled": True,
              "correction": "height_consistent", "shear": "power_law",
              "density": "v1_mixed"}
        for k, v in zip(top, x):
            if k == "z0_scale_log":
                th["z0_scale"] = float(np.exp(v))
            else:
                th[k] = float(v)
        return th

    Y = np.array([chain.run_chain(theta_of(x)) for x in X])
    res = sobol.analyze(problem, Y, calc_second_order=False, seed=42)
    df = pd.DataFrame({"parameter": top, "S1": res["S1"],
                       "S1_conf": res["S1_conf"], "ST": res["ST"],
                       "ST_conf": res["ST_conf"]}).sort_values("ST", ascending=False)
    df.to_csv(os.path.join(REPO, "results", "round2", "sobol_indices.csv"),
              index=False)
    print(df.round(4).to_string(index=False))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(df))
    ax.bar(x - 0.2, df["S1"], 0.4, yerr=df["S1_conf"], label="S1 (first order)")
    ax.bar(x + 0.2, df["ST"], 0.4, yerr=df["ST_conf"], label="ST (total)")
    ax.set_xticks(x, df["parameter"], rotation=20)
    ax.set_ylabel("Sobol index")
    ax.set_title("Sobol indices, output: mean |energy ratio - 1|")
    ax.legend()
    fig.savefig(os.path.join(REPO, "figs", "round2", "sobol_indices.png"),
                dpi=200, bbox_inches="tight")
    print("figure: figs/round2/sobol_indices.png")


if __name__ == "__main__":
    main()
