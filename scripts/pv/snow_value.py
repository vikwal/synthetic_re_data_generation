#!/usr/bin/env python
"""Is a snow layer worth having, and which one? Validated on the Trianel meters, winter 2023/24.

Chain: repaired PVWatts (variant P), CAMS driver, NNLS-fitted sub-arrays (the master data may be
wrong), level calibrated on snow-free months (Apr-Oct, suspect hours excluded) so that the winter
deficit is isolated. Snow driver: ERA5 sf/sd/rsn/2t at the nearest grid point. Variants:

    none            no snow loss
    marion_*        Marion et al. (2013) coverage (pvlib.snow.coverage_nrel), fresh-snow density 100,
                    snowfall threshold {0.2, 0.5, 1.0} cm/h, slide coefficient {-80, -40}
    marion_*_gate   the same, coverage set to 0 when the ERA5 snow pack is < 0.5 cm
    depth_full      covered while ERA5 snow depth >= 2 cm
    depth_linear    coverage = min(depth / 5 cm, 1)

    python scripts/pv/snow_value.py --cams-dir <dir> --horizon-json <pvgis.json> --snow-dir <era5 grib dir> \
        --extras-dir <figs dir with extras_pv*.json> --out <dir>
"""
import argparse
import itertools
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
from pv import chain, metrics, snow, timebase  # noqa: E402
from parks import era5_db  # noqa: E402
import trianel_common as tc  # noqa: E402

plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e6e5e1", "grid.linewidth": 0.6, "font.size": 8.5, "legend.frameon": False, "savefig.bbox": "tight"})
COL = {"meas": "#52514e", "none": "#eb6834", "marion": "#2a78d6", "depth": "#1baf7a", "gate": "#4a3aa7"}


def scale_fit(y, p, m):
    m = m & y.notna() & p.notna() & (p > 0)
    x, t = p[m].values, y[m].values
    return float(np.dot(x, t) / np.dot(x, x)) if np.dot(x, x) > 0 else np.nan


def variants(sn: pd.DataFrame, poa: pd.Series, tilt: float) -> dict:
    sf_cm = snow.snowfall_cm(sn["snowfall_mm_we"]).reindex(poa.index).fillna(0.0)
    depth_cm = (sn["snow_depth_m"] * 100.0).reindex(poa.index).ffill().bfill()
    ta = sn["temp_air"].reindex(poa.index).ffill().bfill()
    out = {"none": pd.Series(0.0, index=poa.index)}
    for thr, slide in itertools.product((0.2, 0.5, 1.0), (-80.0, -40.0)):
        cov = snow.coverage(sf_cm, poa, ta, tilt, threshold_snowfall=thr, can_slide_coefficient=slide)
        out[f"marion_t{thr:g}_c{int(-slide)}"] = cov
        out[f"marion_t{thr:g}_c{int(-slide)}_gate"] = snow.depth_gate(cov, depth_cm, 0.5)
    out["depth_full"] = (depth_cm >= 2.0).astype(float)
    out["depth_linear"] = (depth_cm / 5.0).clip(0.0, 1.0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2023-07-24")
    ap.add_argument("--end", default="2024-06-01")
    ap.add_argument("--cams-dir", required=True)
    ap.add_argument("--horizon-json", required=True)
    ap.add_argument("--snow-dir", required=True)
    ap.add_argument("--mastr-cand", default=None)
    ap.add_argument("--mastr-xml", default=None)
    ap.add_argument("--extras-dir", default=os.path.expanduser("~/Work/FL_Contribution/reports/figs_pv_generator"))
    ap.add_argument("--out", default=os.path.expanduser("~/Work/FL_Contribution/reports/figs_pv_generator"))
    ap.add_argument("--snow-cache", default=None, help="directory for the extracted ERA5 point series (confidential, outside repos)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    m = tc.load_masterdata()
    parks = tc.parks_table(m)
    meter = tc.load_meter_hourly(parks)
    facts = tc.mastr_facts(parks, args.mastr_cand, args.mastr_xml)
    par = tc.park_parameters(parks, facts, meter)
    conn = era5_db.connect()
    pvs = [pv for pv in parks.index if pv in meter and meter[pv].notna().sum() > 1000]
    idx = timebase.hourly_index(args.start, args.end)
    cand = pd.read_csv(os.path.join(args.extras_dir, "snow_candidates.csv"), index_col=0)
    rows, daily_rows, cover_rows = [], [], []
    for pv in pvs:
        p = parks.loc[pv]
        drvs, _ = tc.build_drivers(p, args.start, args.end, conn, args.cams_dir, pv)
        drv = drvs["cams"]
        hz = tc.load_horizon(args.horizon_json, pv)
        y = meter[pv].reindex(idx)
        cap_w = float(par.loc[pv, "kwp_dc"]) * 1000.0
        fitted = json.load(open(os.path.join(args.extras_dir, f"extras_{pv}.json")))["fitted_arrays"]
        tilt = float(max(fitted, key=lambda a: a["share"])["tilt"])
        cfg = tc.base_config(pv, p, par.loc[pv], arrays=fitted, variant="P")
        r = chain.run_park(cfg, drv, horizon=hz)
        ps0, poa = r["p_ac_w"], r["poa_global"]
        cache = os.path.join(args.snow_cache, f"{pv}_era5_snow.parquet") if args.snow_cache else None
        if cache and os.path.exists(cache):
            sn = pd.read_parquet(cache)
        else:
            sn = snow.era5_snow_point(args.snow_dir, p["lat"], p["lon"])
            if cache:
                os.makedirs(args.snow_cache, exist_ok=True)
                sn.to_parquet(cache)
        sp = timebase.solar_position(idx, p["lat"], p["lon"], p["alt"])
        zen = sp["apparent_zenith"]
        day = zen < 90.0
        cs_g = timebase.clearsky(idx, p["lat"], p["lon"], p["alt"])["ghi"]
        kc = drv["ghi"] / cs_g.replace(0, np.nan)
        suspect = day & (kc > 0.6) & ((90.0 - zen) > 20.0) & (y < 0.5 * ps0) & y.notna()
        summer = day & ~suspect & idx.month.isin([4, 5, 6, 7, 8, 9, 10])
        s = scale_fit(y, ps0, summer)
        winter = day & ~suspect & ((idx >= pd.Timestamp("2023-11-15", tz="UTC")) & (idx < pd.Timestamp("2024-03-15", tz="UTC")))
        cdays = pd.to_datetime(str(cand.loc[pv, "dates"]).split(";")).date if pv in cand.index and isinstance(cand.loc[pv, "dates"], str) else []
        cmask = pd.Series(np.isin(idx.date, list(cdays)), index=idx) & day
        snow_days = pd.Series(np.isin(idx.date, list(set((sn.index[sn["snow_depth_m"] >= 0.01]).date))), index=idx) & day & ~suspect
        covs = variants(sn, poa, tilt)
        for name, cov in covs.items():
            ps = ps0 * s * snow.loss_factor(cov)
            fam = "none" if name == "none" else "gate" if name.endswith("gate") else name.split("_")[0]
            for tag, mm in {"winter": winter, "candidate_days": cmask, "era5_snow_days": snow_days, "all_day": day & ~suspect}.items():
                b = metrics.basic(y[mm], ps[mm], cap_w)
                rows.append({"pv": pv, "variant": name, "family": fam, "subset": tag, "scale_summer": s, "covered_hours": float((cov[mm] > 0.5).sum()), **b})
            for mon, g in ps[winter].groupby(ps[winter].index.to_period("M")):
                yy = y[winter].loc[g.index]
                rows.append({"pv": pv, "variant": name, "family": fam, "subset": f"month_{mon}", "scale_summer": s,
                             "covered_hours": float((cov.loc[g.index] > 0.5).sum()), **metrics.basic(yy, g, cap_w)})
            cover_rows.append({"pv": pv, "variant": name, "hours_covered_gt50": float((cov > 0.5).sum()), "mean_coverage_winter": float(cov[winter].mean())})
        # daily energy ratios in winter for the figure (measurement vs none vs best few)
        dd = pd.DataFrame({"y": y[winter], "none": (ps0 * s)[winter]})
        for name in ("marion_t0.5_c80", "marion_t0.5_c80_gate", "depth_full", "depth_linear"):
            dd[name] = (ps0 * s * snow.loss_factor(covs[name]))[winter]
        dd["snow_depth_cm"] = (sn["snow_depth_m"] * 100).reindex(winter[winter].index)
        dsum = dd.groupby(dd.index.date).sum(min_count=1)
        dsum["snow_depth_cm"] = dd["snow_depth_cm"].groupby(dd.index.date).mean()
        dsum["pv"] = pv
        daily_rows.append(dsum)
        print(pv, "tilt", round(tilt, 1), "scale", round(s, 3), {k: round(v, 3) for k, v in pd.DataFrame(rows).query("pv == @pv and subset == 'winter'").set_index("variant")["energy_ratio"].items()}, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, "snow_value_long.csv"), index=False)
    pd.DataFrame(cover_rows).to_csv(os.path.join(args.out, "snow_coverage_hours.csv"), index=False)
    daily = pd.concat(daily_rows)
    daily.index.name = "date"
    daily.to_csv(os.path.join(args.out, "snow_daily_winter.csv"))
    w = df[df.subset == "winter"].pivot(index="pv", columns="variant", values="r2")
    print("winter R2:\n", w.round(3).to_string())
    e = df[df.subset == "winter"].pivot(index="pv", columns="variant", values="energy_ratio")
    print("winter energy ratio:\n", e.round(3).to_string())
    # figure: daily energy in Dec/Jan for each park, measurement vs none vs marion gate vs depth_full
    fig, axes = plt.subplots(len(pvs), 1, figsize=(10, 1.8 * len(pvs)), sharex=True)
    for ax, pv in zip(np.atleast_1d(axes), pvs):
        d = daily[daily.pv == pv]
        d.index = pd.to_datetime(d.index)
        d = d[(d.index >= "2023-11-20") & (d.index < "2024-02-15")]
        ax.plot(d.index, d["y"] / 1e3, color=COL["meas"], lw=1.4, label="meter")
        ax.plot(d.index, d["none"] / 1e3, color=COL["none"], lw=1.0, label="no snow")
        ax.plot(d.index, d["marion_t0.5_c80_gate"] / 1e3, color=COL["gate"], lw=1.0, label="Marion 0.5 cm/h + depth gate")
        ax.plot(d.index, d["depth_full"] / 1e3, color=COL["depth"], lw=1.0, label="depth ≥ 2 cm")
        ax2 = ax.twinx()
        ax2.fill_between(d.index, 0, d["snow_depth_cm"], color="#c3c2b7", alpha=0.35, step="mid")
        ax2.set_ylabel("ERA5 snow cm", color="#8a8982")
        ax2.grid(False)
        ax.set_ylabel(f"{pv}\nkWh/day")
    np.atleast_1d(axes)[0].legend(ncol=4, fontsize=7.5, loc="upper left")
    fig.suptitle("Daily energy in winter 2023/24: meter vs chain without and with snow layers", x=0.01, ha="left", fontsize=9)
    fig.savefig(os.path.join(args.out, "fig_snow_daily.png"))


if __name__ == "__main__":
    main()
