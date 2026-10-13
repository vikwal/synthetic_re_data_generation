#!/usr/bin/env python
"""How much is the terrain layer worth? Validated on the Trianel meters: the repaired PVWatts chain
(variant P, CAMS driver) is run with different sub-array geometries and compared on the same hours.

    master         one plane with the master-data angles (the master data may be wrong)
    fit            NNLS-fitted sub-arrays from the benchmark (clear hours before --fit-end)
    terrain_master terrain-following clusters (OSM footprint + GLO-30) around the master angles
    terrain_fit    terrain-following clusters around the fitted mean angles
    design_mild    statistical stand-in, sigma_tilt 4 / sigma_az 15 (federated_setting P6), 5 seeds
    design_strong  statistical stand-in, sigma_tilt 8 / sigma_az 35 (P10), 5 seeds

    python scripts/pv/terrain_value.py --cams-dir <dir> --horizon-json <pvgis.json> --osm-json <overpass.json> \
        --extras-dir <figs dir with extras_pv*.json> --out <dir>
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
from pv import angles, arrays as arr, chain, metrics, terrain, timebase  # noqa: E402
from parks import era5_db  # noqa: E402
import trianel_common as tc  # noqa: E402
from terrain_trianel import density_for  # noqa: E402

plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e6e5e1", "grid.linewidth": 0.6, "font.size": 8.5, "legend.frameon": False, "savefig.bbox": "tight"})
COL = {"master": "#52514e", "fit": "#4a3aa7", "terrain_master": "#1baf7a", "terrain_fit": "#eb6834", "design_mild": "#2a78d6", "design_strong": "#c53a8f"}
DESIGN = {"design_mild": (4.0, 15.0), "design_strong": (8.0, 35.0)}


def as_dicts(cl) -> list:
    return [{"tilt": float(c.tilt), "azimuth": float(c.azimuth), "share": float(c.share)} for c in cl]


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
    ap.add_argument("--osm-json", required=True)
    ap.add_argument("--mastr-cand", default=None)
    ap.add_argument("--mastr-xml", default=None)
    ap.add_argument("--extras-dir", default=os.path.expanduser("~/Work/FL_Contribution/reports/figs_pv_generator"))
    ap.add_argument("--out", default=os.path.expanduser("~/Work/FL_Contribution/reports/figs_pv_generator"))
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    m = tc.load_masterdata()
    parks = tc.parks_table(m)
    meter = tc.load_meter_hourly(parks)
    facts = tc.mastr_facts(parks, args.mastr_cand, args.mastr_xml)
    par = tc.park_parameters(parks, facts, meter)
    osm = json.load(open(args.osm_json))
    conn = era5_db.connect()
    pvs = [pv for pv in parks.index if pv in meter and meter[pv].notna().sum() > 1000]
    idx = timebase.hourly_index(args.start, args.end)
    rows, geo_rows = [], []
    for pv in pvs:
        p = parks.loc[pv]
        drvs, _ = tc.build_drivers(p, args.start, args.end, conn, args.cams_dir, pv)
        drv = drvs["cams"]
        hz = tc.load_horizon(args.horizon_json, pv)
        y = meter[pv].reindex(idx)
        cap_w = float(par.loc[pv, "kwp_dc"]) * 1000.0
        fitted = json.load(open(os.path.join(args.extras_dir, f"extras_{pv}.json")))["fitted_arrays"]
        te, ae = arr.mean_angles(np.array([a["tilt"] for a in fitted]), np.array([a["azimuth"] for a in fitted]), np.array([a["share"] for a in fitted]))
        els = osm.get(pv, {}).get("elements", [])
        poly = terrain.pick_plant_near(els, p["lat"], p["lon"], 2.5) if els else None
        src = "osm"
        if poly is None:
            ha = float(facts.loc[pv, "area_ha"]) if pd.notna(facts.loc[pv, "area_ha"]) else par.loc[pv, "kwp_dc"] / density_for(int(par.loc[pv, "year"]))
            poly, src = terrain.buffer_polygon(p["lat"], p["lon"], ha), "buffer"

        def clusters(t0, a0):
            t = terrain.terrain_table(poly, t0, a0)
            return as_dicts(arr.cluster_arrays(t["tilt_eff"].values, t["az_eff"].values, t["area_m2"].values, args.k))

        variants = {"master": p["arrays"], "fit": fitted, "terrain_master": clusters(p["tilt_master"], p["az_master"]),
                    "terrain_fit": clusters(te, ae)}
        for name, (st, sa) in DESIGN.items():
            for s in range(args.seeds):
                variants[f"{name}#{s}"] = as_dicts(arr.dirichlet_arrays(np.random.default_rng(100 + s), args.k, p["tilt_master"], p["az_master"], st, sa))
        sp = timebase.solar_position(idx, p["lat"], p["lon"], p["alt"])
        zen = sp["apparent_zenith"]
        day = zen < 90.0
        morning = day & (sp["azimuth"] < 180.0)
        clear = angles.clear_hours(drv["ghi"], p["lat"], p["lon"], p["alt"]) & y.notna()
        out = {}
        for name, arrays in variants.items():
            cfg = tc.base_config(pv, p, par.loc[pv], arrays=arrays, variant="P")
            out[name] = chain.run_park(cfg, drv, horizon=hz)["p_ac_w"]
        cs_g = timebase.clearsky(idx, p["lat"], p["lon"], p["alt"])["ghi"]
        kc = drv["ghi"] / cs_g.replace(0, np.nan)
        suspect = day & (kc > 0.6) & ((90.0 - zen) > 20.0) & (y < 0.5 * out["fit"]) & y.notna()
        subsets = {"day": day, "day_nosuspect": day & ~suspect, "clear": clear & ~suspect, "eval_window": day & ~suspect & (idx >= pd.Timestamp(args.fit_end, tz="UTC"))}
        for name, ps in out.items():
            s = scale_fit(y, ps, day & ~suspect)
            for tag, mm in subsets.items():
                b = metrics.basic(y[mm], ps[mm], cap_w)
                bs = metrics.basic(y[mm], ps[mm] * s, cap_w)
                am = (y[mm & morning].sum() / ps[mm & morning].sum()) if ps[mm & morning].sum() > 0 else np.nan
                pm = (y[mm & ~morning].sum() / ps[mm & ~morning].sum()) if ps[mm & ~morning].sum() > 0 else np.nan
                rows.append({"pv": pv, "variant": name, "family": name.split("#")[0], "subset": tag, **b, "r2_scaled": bs.get("r2"),
                             "nrmse_scaled": bs.get("nrmse_cap"), "scale": s, "asym_morning_over_afternoon": am / pm if pm else np.nan})
        geo_rows.append({"pv": pv, "footprint_src": src, "fit_mean_tilt": te, "fit_mean_az": ae, "n_fitted": len(fitted),
                         **{f"{k}_arrays": "; ".join(f"{a['tilt']:.1f}°/{a['azimuth']:.0f}°×{a['share']:.2f}" for a in variants[k]) for k in ("master", "fit", "terrain_master", "terrain_fit")}})
        print(pv, src, {k: round(v, 4) for k, v in pd.DataFrame(rows).query("pv == @pv and subset == 'day_nosuspect'").groupby("family")["r2"].mean().items()}, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, "terrain_value_long.csv"), index=False)
    pd.DataFrame(geo_rows).set_index("pv").to_csv(os.path.join(args.out, "terrain_value_geometry.csv"))
    # summary: family means (design variants averaged over seeds) and deltas to master
    fam = df.groupby(["pv", "family", "subset"])[["r2", "nrmse_cap", "r2_scaled", "nrmse_scaled", "energy_ratio", "asym_morning_over_afternoon"]].mean().reset_index()
    base = fam[fam.family == "master"].set_index(["pv", "subset"])
    fam["d_r2"] = [r.r2 - base.loc[(r.pv, r.subset), "r2"] for r in fam.itertuples()]
    fam["d_nrmse_pp"] = [100 * (r.nrmse_cap - base.loc[(r.pv, r.subset), "nrmse_cap"]) for r in fam.itertuples()]
    fam["d_r2_scaled"] = [r.r2_scaled - base.loc[(r.pv, r.subset), "r2_scaled"] for r in fam.itertuples()]
    fam.to_csv(os.path.join(args.out, "terrain_value_summary.csv"), index=False)
    piv = fam[fam.subset == "day_nosuspect"].pivot(index="pv", columns="family", values="d_r2")
    print(piv.round(4).to_string())
    # figure: delta R2 vs master per park and family, two subsets
    fams = ["fit", "terrain_master", "terrain_fit", "design_mild", "design_strong"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    for ax, sub in zip(axes, ["day_nosuspect", "clear"]):
        d = fam[fam.subset == sub]
        x = np.arange(len(pvs))
        wd = 0.8 / len(fams)
        for i, f in enumerate(fams):
            v = [d[(d.pv == pv) & (d.family == f)]["d_r2"].mean() for pv in pvs]
            ax.bar(x + (i - len(fams) / 2 + 0.5) * wd, v, wd, color=COL[f], label=f.replace("_", " "))
        ax.axhline(0, color="#52514e", lw=0.8)
        ax.set_xticks(x, pvs)
        ax.set_title(f"ΔR² vs master data plane, hours: {sub}", loc="left")
    axes[0].set_ylabel("ΔR²")
    axes[1].legend(ncol=1, fontsize=7.5)
    fig.savefig(os.path.join(args.out, "fig_terrain_value.png"))


if __name__ == "__main__":
    main()
