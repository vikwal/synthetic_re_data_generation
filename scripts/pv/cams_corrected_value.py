#!/usr/bin/env python
"""Value of the corrected CAMS product for PV: the repaired PVWatts chain (variant P, fitted
sub-arrays) driven by raw CAMS vs the cams_bc-corrected GHI/DHI at the park point, against the meters.

    python scripts/pv/cams_corrected_value.py --cams-dir <dir> --horizon-json <pvgis.json> --corrected-dir <dir> --tags xgb mlp
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pv import chain, metrics, timebase, drivers  # noqa: E402
from parks import era5_db  # noqa: E402
import trianel_common as tc  # noqa: E402

plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e6e5e1", "grid.linewidth": 0.6, "font.size": 8.5, "legend.frameon": False, "savefig.bbox": "tight"})
COL = {"cams": "#eb6834", "xgb": "#2a78d6", "mlp": "#1baf7a", "station": "#52514e"}


def scale_fit(y, p, m):
    m = m & y.notna() & p.notna() & (p > 0)
    x, t = p[m].values, y[m].values
    return float(np.dot(x, t) / np.dot(x, x)) if np.dot(x, x) > 0 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2023-07-24")
    ap.add_argument("--end", default="2024-06-01")
    ap.add_argument("--fit-end", default="2024-02-01")
    ap.add_argument("--cams-dir", required=True)
    ap.add_argument("--horizon-json", required=True)
    ap.add_argument("--corrected-dir", required=True)
    ap.add_argument("--tags", nargs="+", default=["xgb"])
    ap.add_argument("--mastr-cand", default=None)
    ap.add_argument("--mastr-xml", default=None)
    ap.add_argument("--extras-dir", default=os.path.expanduser("~/Work/FL_Contribution/reports/figs_pv_generator"))
    ap.add_argument("--out", default=os.path.expanduser("~/Work/FL_Contribution/reports/figs_pv_generator"))
    args = ap.parse_args()
    m = tc.load_masterdata()
    parks = tc.parks_table(m)
    meter = tc.load_meter_hourly(parks)
    facts = tc.mastr_facts(parks, args.mastr_cand, args.mastr_xml)
    par = tc.park_parameters(parks, facts, meter)
    conn = era5_db.connect()
    idx = timebase.hourly_index(args.start, args.end)
    rows, monthly = [], []
    for pv in [pv for pv in parks.index if pv in meter and meter[pv].notna().sum() > 1000]:
        p = parks.loc[pv]
        drvs, _ = tc.build_drivers(p, args.start, args.end, conn, args.cams_dir, pv)
        hz = tc.load_horizon(args.horizon_json, pv)
        y = meter[pv].reindex(idx)
        cap_w = float(par.loc[pv, "kwp_dc"]) * 1000.0
        fitted = json.load(open(os.path.join(args.extras_dir, f"extras_{pv}.json")))["fitted_arrays"]
        cfg = tc.base_config(pv, p, par.loc[pv], arrays=fitted, variant="P")
        runs = {"cams": drvs["cams"], "station": drvs["station"]}
        for tag in args.tags:
            f = os.path.join(args.corrected_dir, f"{pv}_cams_corrected_hourly_{tag}.parquet")
            if not os.path.exists(f):
                continue
            c = pd.read_parquet(f).reindex(idx)
            d = drvs["cams"].copy()
            d["ghi"] = c["ghi_corr"].fillna(d["ghi"]).clip(lower=0)
            d["dhi"] = np.minimum(c["dhi_corr"].fillna(d["dhi"]).clip(lower=0), d["ghi"])
            d["bhi"] = d["ghi"] - d["dhi"]
            runs[tag] = d
        sp = timebase.solar_position(idx, p["lat"], p["lon"], p["alt"])
        zen = sp["apparent_zenith"]
        day = zen < 90.0
        ps = {k: chain.run_park(cfg, v, horizon=hz)["p_ac_w"] for k, v in runs.items()}
        cs_g = timebase.clearsky(idx, p["lat"], p["lon"], p["alt"])["ghi"]
        kc = drvs["cams"]["ghi"] / cs_g.replace(0, np.nan)
        suspect = day & (kc > 0.6) & ((90.0 - zen) > 20.0) & (y < 0.5 * ps["cams"]) & y.notna()
        subsets = {"day": day, "day_nosuspect": day & ~suspect, "eval_window": day & ~suspect & (idx >= pd.Timestamp(args.fit_end, tz="UTC")),
                   "cloudy": day & ~suspect & (kc < 0.5), "clear": day & ~suspect & (kc >= 0.8)}
        for k, pser in ps.items():
            s = scale_fit(y, pser, day & ~suspect)
            for tag, mm in subsets.items():
                b = metrics.basic(y[mm], pser[mm], cap_w)
                bs = metrics.basic(y[mm], pser[mm] * s, cap_w)
                rows.append({"pv": pv, "driver": k, "subset": tag, **b, "r2_scaled": bs.get("r2"), "nrmse_scaled": bs.get("nrmse_cap"), "scale": s})
            mm = day & ~suspect
            for mon, g in pser[mm].groupby(pser[mm].index.to_period("M")):
                b = metrics.basic(y[mm].loc[g.index], g, cap_w)
                monthly.append({"pv": pv, "driver": k, "month": str(mon), **b})
        print(pv, {k: round(metrics.basic(y[day & ~suspect], v[day & ~suspect], cap_w)["r2"], 4) for k, v in ps.items()}, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, "cams_corrected_value.csv"), index=False)
    pd.DataFrame(monthly).to_csv(os.path.join(args.out, "cams_corrected_monthly.csv"), index=False)
    for sub in ("day_nosuspect", "eval_window", "cloudy"):
        print(sub); print(df[df.subset == sub].pivot(index="pv", columns="driver", values=["r2", "nrmse_cap"]).round(4).to_string())
    drvs_order = ["station", "cams"] + [t for t in args.tags if t in df.driver.unique()]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    for ax, sub in zip(axes, ["day_nosuspect", "cloudy"]):
        d = df[df.subset == sub]
        pvs = sorted(d.pv.unique())
        x = np.arange(len(pvs))
        wd = 0.8 / len(drvs_order)
        for i, k in enumerate(drvs_order):
            v = [d[(d.pv == pv) & (d.driver == k)]["nrmse_cap"].mean() * 100 for pv in pvs]
            ax.bar(x + (i - len(drvs_order) / 2 + 0.5) * wd, v, wd, color=COL.get(k, "#4a3aa7"), label={"station": "DWD station (imputed)", "cams": "CAMS raw"}.get(k, f"CAMS corrected ({k})"))
        ax.set_xticks(x, pvs)
        ax.set_ylabel("nRMSE % of kWp")
        ax.set_title(f"hours: {sub}", loc="left")
    axes[0].legend(fontsize=7.5)
    fig.suptitle("PV chain error by irradiance driver (variant P, fitted sub-arrays)", x=0.01, ha="left", fontsize=9)
    fig.savefig(os.path.join(args.out, "fig_cams_corrected_value.png"))


if __name__ == "__main__":
    main()
