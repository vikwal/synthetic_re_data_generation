#!/usr/bin/env python3
"""WP4.3 — cross-sectional fleet recalibration/validation.

Per park: efficiency = sum(P_meas) / sum(P_synth_no_aging) over the validation
window, on the FINAL frozen wind chain (pass the experiment id of the
no-aging variant). Regress ln(efficiency) on park age (N=13), compare
const-ADR vs Weibull vs Weibull+step curves (validate, don't fit: parameters
stay literature-anchored). Curtailment screen applied as companion.

Usage: wp4_fleet_regression.py --experiment M3noage
Outputs: results/round2/wp4_fleet_regression.csv, figs/round2/bias_vs_age.png
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
from round2 import aging, meterdata  # noqa: E402

SYNTH_BASE = "/mnt/nvme2/synthetic"
WINDOW = ("2023-06-01", "2024-06-01")


def park_age(park_id: str, at: str = "2023-12-01") -> float:
    # Ask 12: single authoritative source (config-derived table)
    from round2 import parkinfo
    cd = parkinfo.commissioning_date(park_id)
    return (pd.Timestamp(at) - pd.Timestamp(cd)).days / 365.25


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True,
                    help="ladder experiment WITHOUT aging (P_synth free of DF)")
    args = ap.parse_args()

    prices = pd.read_csv(os.path.join(REPO, "data", "round2", "prices_delu.csv"),
                         index_col=0, parse_dates=True)["price_eur_mwh"]
    rows = []
    import glob
    for f in sorted(glob.glob(os.path.join(SYNTH_BASE, "wind", "round2",
                                           args.experiment, "synth_*.csv"))):
        park_id = os.path.basename(f)[len("synth_"):-len(".csv")]
        synth = pd.read_csv(f, sep=";", index_col=0, parse_dates=True)["power_park"]
        synth = synth.loc[WINDOW[0]:WINDOW[1]]
        meas = meterdata.load_park_power(park_id, WINDOW)
        both = pd.concat([meas.rename("meas"), synth.rename("synth")], axis=1).dropna()
        if len(both) < 1000:
            print(f"skip {park_id}: only {len(both)} h")
            continue
        eff = both["meas"].sum() / both["synth"].sum()
        flag = prices.reindex(both.index).fillna(1.0) <= 0
        eff_nc = both.loc[~flag, "meas"].sum() / both.loc[~flag, "synth"].sum()
        rows.append({"park_id": park_id, "age": park_age(park_id),
                     "efficiency": eff, "efficiency_excl_curt": eff_nc,
                     "n_hours": len(both)})
    df = pd.DataFrame(rows)
    assert len(df) >= 10, f"too few parks: {len(df)}"

    results = {}
    for col in ("efficiency", "efficiency_excl_curt"):
        y = np.log(df[col].values)
        X = sm.add_constant(df["age"].values)
        fit = sm.OLS(y, X).fit()
        slope, ci = fit.params[1], fit.conf_int()[1]
        results[col] = {"slope_pct_per_yr": slope * 100,
                        "ci_lo": ci[0] * 100, "ci_hi": ci[1] * 100,
                        "r2": fit.rsquared, "p": fit.pvalues[1]}
        print(f"{col}: slope {slope*100:.3f} %/yr [{ci[0]*100:.3f}, {ci[1]*100:.3f}] "
              f"(Germer anchor: -0.63 %/yr), R2={fit.rsquared:.3f}")

    # compare the three literature-anchored shapes: SSE of ln(eff) vs ln(DF(age))
    # up to a free level constant (parks differ in absolute bias) -> AIC + LOO
    ages = df["age"].values
    y = np.log(df["efficiency"].values)
    comp = []
    for name, fn in (("const_adr", lambda a: aging.DF_const(a)),
                     ("weibull", lambda a: aging.DF_weibull(a)),
                     ("weibull_step", lambda a: aging.DF_weibull_step(a))):
        x = np.log(fn(ages))
        c = float(np.mean(y - x))       # free level offset
        resid = y - (x + c)
        sse = float((resid ** 2).sum())
        n, k = len(y), 1
        aic = n * np.log(sse / n) + 2 * k
        loo = []
        for i in range(n):
            mask = np.arange(n) != i
            ci_ = float(np.mean(y[mask] - x[mask]))
            loo.append((y[i] - (x[i] + ci_)) ** 2)
        comp.append({"model": name, "sse": sse, "aic": aic,
                     "loo_mse": float(np.mean(loo)), "level_offset": c})
    comp = pd.DataFrame(comp)
    print(comp.round(4).to_string(index=False))

    df.to_csv(os.path.join(REPO, "results", "round2", "wp4_fleet_efficiency.csv"),
              index=False)
    comp.to_csv(os.path.join(REPO, "results", "round2", "wp4_curve_comparison.csv"),
                index=False)
    pd.DataFrame(results).T.to_csv(
        os.path.join(REPO, "results", "round2", "wp4_regression.csv"))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(df["age"], df["efficiency"], label="parks (efficiency)", zorder=3)
    for _, r in df.iterrows():
        ax.annotate(r["park_id"], (r["age"], r["efficiency"]), fontsize=7,
                    textcoords="offset points", xytext=(4, 3))
    a_grid = np.linspace(0, max(ages) + 2, 100)
    for name, fn, style in (("const ADR", aging.DF_const, "--"),
                            ("Weibull", aging.DF_weibull, "-"),
                            ("Weibull+step", aging.DF_weibull_step, ":")):
        c = comp.loc[comp["model"] == name.lower().replace(" ", "_").replace("+", "_"),
                     "level_offset"]
        c = float(c.iloc[0]) if len(c) else 0.0
        ax.plot(a_grid, np.exp(np.log(fn(a_grid)) + c), style, label=name)
    ax.set_xlabel("park age [yr]")
    ax.set_ylabel("efficiency  =  sum P_meas / sum P_synth(no aging)")
    ax.set_title("WP4: bias vs age, three degradation shapes (level-adjusted)")
    ax.legend()
    fig.savefig(os.path.join(REPO, "figs", "round2", "bias_vs_age.png"),
                dpi=200, bbox_inches="tight")
    print("figure: figs/round2/bias_vs_age.png")


if __name__ == "__main__":
    main()
