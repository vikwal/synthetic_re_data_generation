#!/usr/bin/env python3
"""Round 3 (reviewer 2e-3): can the 13-farm cross-section discriminate the
functional form of the degradation law? The chain (round2.chain, all-aging
baseline, IEC density) is run with
  none          DF = 1
  linear        constant-rate 0.63 %/a (round-1 law, DF_const)
  weibull       exp(-(a/54.5)^2)  (deployed)
  weibull_step  weibull with an additional step after year 20
  uniform_match a flat derating equal to the fleet-mean Weibull factor
and the per-park energy ratios and |ER-1| are compared.

Run on l2: synthre/bin/python scripts/round3/aging_forms.py
Output: results/round3/aging_forms.csv
"""
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
from round2.chain import Chain  # noqa: E402
from round2 import aging as r2_aging  # noqa: E402

OUT = os.path.join(REPO, "results", "round3")
BASE = {"wake_enabled": True, "wake_k": 0.075, "correction": "height_consistent",
        "shear": "power_law", "density": "dynamic", "power_curve_scale": 1.0,
        "wind_level_factor": 1.0, "z0_scale": 1.0, "aging_lambda": 54.5, "aging_kappa": 2.0}
FORMS = {"none": {"aging_model": "none"},
         "linear_0.63pct": {"aging_model": "const"},
         "weibull": {"aging_model": "weibull"},
         "weibull_step": {"aging_model": "weibull_step"}}


def main():
    os.makedirs(OUT, exist_ok=True)
    chain = Chain()
    for pc in chain.parks:
        pc.force_all_aging()
    rows = []
    for form, extra in FORMS.items():
        th = dict(BASE, **extra)
        for pc in chain.parks:
            er = pc.energy_ratio(th)
            age = float(np.nanmean(pc.ages)) if getattr(pc, "ages", None) is not None else np.nan
            rows.append({"form": form, "park_id": pc.park_id, "mean_age": age, "energy_ratio": er,
                         "abs_er_dev": abs(er - 1) if np.isfinite(er) else np.nan})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "aging_forms.csv"), index=False)
    piv = df.pivot(index="park_id", columns="form", values="abs_er_dev")
    piv["mean_age"] = df[df.form == "weibull"].set_index("park_id")["mean_age"]
    pd.set_option("display.width", 200)
    print("|ER-1| per park and degradation form (all-aging baseline):")
    print(piv.sort_values("mean_age").round(3).to_string())
    print("\nfleet summary:")
    print(df.groupby("form")["abs_er_dev"].agg(["median", "mean"]).round(4).to_string())
    old = piv[piv.mean_age >= 20]
    if len(old):
        print("\nparks >= 20 y:")
        print(old.round(3).to_string())
    print("AGING_FORMS_DONE")


if __name__ == "__main__":
    main()
