#!/usr/bin/env python3
"""Round 3 (reviewer 2e): longitudinal (within-asset) aging test.

Uses the chain WITHOUT aging (experiment M2noage_hist: ERA5 power law,
correction gate, dynamic IEC density, no wakes) generated over the full
ERA5 record 2020-01 .. 2026-04 from wind_era5_v3, and the meter data
2020-01 .. 2024-06. For every park with >= 24 months of data the monthly
energy ratio ER_m = sum(meas) / sum(synth_noaging) is regressed on time:
  ER_m = a + b * (t - t0)        (b in 1/yr, relative trend b/a in %/yr)
and compared with the Weibull prediction for the same park and window,
  DF(age) = exp(-(age/54.5)^2), mean annual loss over the window, and the
linear 0.63 %/yr law. Months with fewer than 500 valid hours are dropped;
hours with measured power == 0 above cut-in are excluded (stage-1 filter,
same rule as the cleaning) because outages dominate the monthly ratio.

Run on l2: synthre/bin/python scripts/round3/longitudinal_aging.py [--exp M2noage_hist]
Output: results/round3/longitudinal_monthly.csv, longitudinal_trends.csv
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
import yaml
from round2 import meterdata, parkinfo, outliers  # noqa: E402


def min_cut_in(park_id):
    with open(os.path.join(REPO, "configs", "real_wind_parks_era5", f"config_{park_id}.yaml")) as f:
        turbines = yaml.safe_load(f)["params"]["turbines"]
    specs = pd.read_csv(os.path.join(REPO, "power_curves", "turbine_specs.csv"), sep=";").drop_duplicates(subset="Turbine").set_index("Turbine")
    return float(min(float(specs.loc[t, "Einschaltgeschwindigkeit"]) for t in set(turbines)))


OUT = os.path.join(REPO, "results", "round3")
SYNTH = "/mnt/nvme2/synthetic/wind/round2"
LAMBDA, KAPPA = 54.5, 2.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="M2noage_hist")
    ap.add_argument("--min-hours", type=int, default=500)
    ap.add_argument("--min-months", type=int, default=24)
    ap.add_argument("--clean", choices=["cutin", "dbscan"], default="dbscan",
                    help="cutin: stage-1 outage filter only; dbscan: the validation cleaning (cut-in filter + per-park DBSCAN), applied per calendar year so the point density matches the one-year calibration")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    mapping = meterdata.load_mapping()
    monthly_rows, trend_rows = [], []
    for park_id in sorted(mapping.park_id):
        f = os.path.join(SYNTH, args.exp, f"synth_{park_id}.csv")
        if not os.path.exists(f):
            print(f"{park_id}: no synth file"); continue
        syn = pd.read_csv(f, sep=";", index_col=0, parse_dates=True)
        p_syn = syn["power_park"] if "power_park" in syn else syn[[c for c in syn.columns if c.startswith("power_t")]].sum(axis=1)
        wcols = [c for c in syn.columns if c.startswith("wind_speed_t")]
        v_hub = syn[wcols].mean(axis=1)
        meas = meterdata.load_park_power(park_id, ("2019-12-31", "2024-06-02"))
        df = pd.concat([meas.rename("meas"), p_syn.rename("syn"), v_hub.rename("v")], axis=1).dropna()
        if args.clean == "dbscan":
            keep = pd.Series(False, index=df.index)
            # the per-park (eps, min_samples) were calibrated on one full year (~8760 points);
            # DBSCAN density depends on the point count, so min_samples is scaled with the
            # number of points of the year (2024 has only Jan-May) to keep the threshold equivalent
            p = outliers.get_park_params(park_id)
            cut_in = min_cut_in(park_id)
            for yr, g in df.groupby(df.index.year):
                k = pd.Series(True, index=g.index)
                k[(g.meas == 0) & (g.v >= cut_in)] = False
                sub = g[k]
                ms = max(3, int(round(p["min_samples"] * len(sub) / 8760.0)))
                labels = DBSCAN(eps=p["eps"], min_samples=ms).fit_predict(StandardScaler().fit_transform(sub[["v", "meas"]].values))
                k.loc[sub.index[labels == -1]] = False
                keep.loc[g.index] = k
            share = {int(y): round(1 - k.mean(), 3) for y, k in keep.groupby(keep.index.year)}
            print(f"{park_id}: outlier share per year {share}")
            df = df[keep]
        else:
            df = df[~((df.meas == 0) & (df.v >= 3.0))]            # stage-1 outage filter
        m = df.resample("MS").agg(meas=("meas", "sum"), syn=("syn", "sum"), n=("meas", "count"))
        m = m[(m.n >= args.min_hours) & (m.syn > 0)]
        if len(m) < args.min_months:
            print(f"{park_id}: only {len(m)} usable months"); continue
        m["er"] = m.meas / m.syn
        cd = pd.Timestamp(parkinfo.commissioning_date(park_id), tz="UTC")
        m["age"] = (m.index - cd).days / 365.25
        t = (m.index - m.index[0]).days / 365.25
        res = stats.linregress(t, m.er)
        rel = res.slope / res.intercept * 100.0
        # Weibull prediction over the same window: relative change of DF per year
        df_w = np.exp(-(m.age / LAMBDA) ** KAPPA)
        weib = stats.linregress(t, df_w / df_w.iloc[0]).slope * 100.0
        for ts, r in m.iterrows():
            monthly_rows.append({"park_id": park_id, "month": ts, "n_hours": int(r.n), "er": r.er, "age": r.age})
        trend_rows.append({"park_id": park_id, "months": len(m), "first": m.index[0], "last": m.index[-1],
                           "age_start": float(m.age.iloc[0]), "age_end": float(m.age.iloc[-1]),
                           "er_mean": float(m.er.mean()), "trend_rel_pct_per_yr": rel,
                           "trend_se_pct_per_yr": res.stderr / res.intercept * 100.0,
                           "p_value": res.pvalue, "weibull_pred_pct_per_yr": weib,
                           "linear_pred_pct_per_yr": -0.63})
        print(f"{park_id}: {len(m)} months, age {m.age.iloc[0]:.1f}->{m.age.iloc[-1]:.1f} y, ER mean {m.er.mean():.3f}, "
              f"trend {rel:+.2f} %/yr (se {res.stderr/res.intercept*100:.2f}, p {res.pvalue:.3f}), Weibull {weib:+.2f} %/yr")
    mon = pd.DataFrame(monthly_rows)
    mon.to_csv(os.path.join(OUT, f"longitudinal_monthly_{args.clean}.csv"), index=False)
    # Same-calendar-month design: ER of month m in year y2 against year y1 of the same
    # park, so that the seasonal profile of the chain error cancels. One estimate per
    # (park, calendar month, year pair): d ln(ER) / (y2 - y1) in %/yr; per park the mean
    # and its standard error over the pairs; Weibull prediction from the mean ages.
    same_rows, same_park = [], []
    for park_id, g in mon.groupby("park_id"):
        g = g.assign(y=g.month.dt.year, mo=g.month.dt.month)
        pairs = []
        for mo, gm in g.groupby("mo"):
            gm = gm.sort_values("y")
            for i in range(len(gm)):
                for j in range(i + 1, len(gm)):
                    a, b = gm.iloc[i], gm.iloc[j]
                    dy = b.y - a.y
                    d = np.log(b.er / a.er) / dy * 100.0
                    w = -(2.0 / LAMBDA ** 2) * ((a.age + b.age) / 2.0) * 100.0  # d ln DF / dt for kappa = 2
                    pairs.append(d); same_rows.append({"park_id": park_id, "calendar_month": int(mo), "year_from": int(a.y),
                                                       "year_to": int(b.y), "er_from": a.er, "er_to": b.er,
                                                       "dln_er_pct_per_yr": d, "weibull_pred_pct_per_yr": w})
        if pairs:
            pairs = np.array(pairs)
            same_park.append({"park_id": park_id, "n_pairs": len(pairs), "mean_pct_per_yr": pairs.mean(),
                              "se_pct_per_yr": pairs.std(ddof=1) / np.sqrt(len(pairs)) if len(pairs) > 1 else np.nan,
                              "weibull_pred_pct_per_yr": np.mean([r["weibull_pred_pct_per_yr"] for r in same_rows if r["park_id"] == park_id])})
    pd.DataFrame(same_rows).to_csv(os.path.join(OUT, f"longitudinal_same_month_pairs_{args.clean}.csv"), index=False)
    sp = pd.DataFrame(same_park)
    sp.to_csv(os.path.join(OUT, f"longitudinal_same_month_parks_{args.clean}.csv"), index=False)
    if len(sp):
        print("\nsame-calendar-month design:")
        for _, r in sp.iterrows():
            print(f"  {r.park_id}: {int(r.n_pairs)} pairs, {r.mean_pct_per_yr:+.2f} %/yr (se {r.se_pct_per_yr:.2f}), Weibull {r.weibull_pred_pct_per_yr:+.2f} %/yr")
    tr = pd.DataFrame(trend_rows)
    tr.to_csv(os.path.join(OUT, f"longitudinal_trends_{args.clean}.csv"), index=False)
    if len(tr):
        w = 1.0 / tr.trend_se_pct_per_yr ** 2
        pooled = float((tr.trend_rel_pct_per_yr * w).sum() / w.sum())
        print(f"\nparks {len(tr)}: median trend {tr.trend_rel_pct_per_yr.median():+.2f} %/yr, "
              f"inverse-variance pooled {pooled:+.2f} %/yr, Weibull prediction median {tr.weibull_pred_pct_per_yr.median():+.2f} %/yr, "
              f"negative in {(tr.trend_rel_pct_per_yr < 0).sum()}/{len(tr)} parks")
    print("LONGITUDINAL_DONE")


if __name__ == "__main__":
    main()
