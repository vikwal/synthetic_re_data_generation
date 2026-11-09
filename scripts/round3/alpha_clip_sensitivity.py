#!/usr/bin/env python3
"""Round 3 (reviewer 1c-2): effect of the clip of the two-level ERA5 shear
exponent alpha = ln(v100/v10)/ln(10) on the farm energy ratio.

The deployed chain clips alpha to [0, 0.4]. The fast chain (round2.chain,
deployed configuration M4: height-consistent correction, power law, IEC
density, Weibull aging on all farms, NOJ wakes k = 0.075) is rerun with
wider bounds, and per farm the energy ratio, |ER-1| and the share of hours
whose alpha lies outside the deployed bounds are reported.

Run on l2: synthre/bin/python scripts/round3/alpha_clip_sensitivity.py
Output: results/round3/alpha_clip_sensitivity.csv
"""
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
from round2.chain import Chain  # noqa: E402

OUT = os.path.join(REPO, "results", "round3")
BASE = {"wake_enabled": True, "wake_k": 0.075, "correction": "height_consistent",
        "shear": "power_law", "density": "dynamic", "power_curve_scale": 1.0,
        "wind_level_factor": 1.0, "z0_scale": 1.0, "aging_model": "weibull",
        "aging_lambda": 54.5, "aging_kappa": 2.0}
BOUNDS = {"[0, 0.4] deployed": (0.0, 0.4), "[0, 0.5]": (0.0, 0.5), "[0, 0.6]": (0.0, 0.6),
          "[-0.1, 0.6]": (-0.1, 0.6), "[-0.2, 1.0]": (-0.2, 1.0)}


def main():
    os.makedirs(OUT, exist_ok=True)
    chain = Chain()
    for pc in chain.parks:
        pc.force_all_aging()
    rows = []
    for name, (lo, hi) in BOUNDS.items():
        th = dict(BASE, alpha_min=lo, alpha_max=hi)
        for pc in chain.parks:
            er = pc.energy_ratio(th)
            rows.append({"bounds": name, "alpha_min": lo, "alpha_max": hi, "park_id": pc.park_id,
                         "energy_ratio": er, "abs_er_dev": abs(er - 1) if np.isfinite(er) else np.nan})
    df = pd.DataFrame(rows)
    base = df[df.bounds == "[0, 0.4] deployed"].set_index("park_id")["energy_ratio"]
    df["d_er_vs_deployed_pct"] = (df.energy_ratio / df.park_id.map(base) - 1) * 100
    df.to_csv(os.path.join(OUT, "alpha_clip_sensitivity.csv"), index=False)
    pd.set_option("display.width", 200)
    print("energy ratio per farm and clip bounds:")
    print(df.pivot(index="park_id", columns="bounds", values="energy_ratio").round(4).to_string())
    print("\nfleet summary:")
    print(df.groupby("bounds", sort=False).agg(median_abs_er_dev=("abs_er_dev", "median"),
                                               mean_abs_er_dev=("abs_er_dev", "mean"),
                                               median_er=("energy_ratio", "median"),
                                               max_abs_d_er_pct=("d_er_vs_deployed_pct", lambda x: x.abs().max()),
                                               median_d_er_pct=("d_er_vs_deployed_pct", "median")).round(4).to_string())
    print("ALPHA_CLIP_DONE")


if __name__ == "__main__":
    main()
