#!/usr/bin/env python
"""Ablation ladder of the parks_pv_v1 production chain on the metered Trianel parks (pv1..pv7, anonymised).

Full chain = corrected CAMS (cams_bc xgb_v5) -> terrain sub-arrays around the master-data mount angles ->
PVGIS horizon + inter-row shading -> PVWatts-P (era gamma, IAM, ERA5 albedo, era aging) -> losses E3
(DC 5 %, AC 2 %, availability 0.98) -> ERA5 snow factor (depth_linear). Every rung removes or swaps ONE
component, so the deltas are comparable; soiling (HSU with CAMS PM and ERA5 rain) is added on top with
several initial-state spin-ups. The loss level is additionally calibrated leave-one-park-out.

    python scripts/pv/ablation_trianel.py --cams-dir <dir> --horizon-json <pvgis.json> --corrected-dir <dir> \
        --osm-json <overpass.json> --mastr-cand <csv> --mastr-xml <json> --snow-dir <era5 grib dir> --snow-cache <dir>
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
from pv import arrays as arr, chain, drivers, metrics, snow, soiling, terrain, timebase  # noqa: E402
from parks import era5_db  # noqa: E402
from parks_pv.params import DENSITY_KWP_HA, DENSITY_YEARS, LOSSES  # noqa: E402
import trianel_common as tc  # noqa: E402

plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e6e5e1", "grid.linewidth": 0.6, "font.size": 8.5, "legend.frameon": False, "savefig.bbox": "tight"})
LOSSES_P = {"dc": 0.02, "ac": 0.01, "availability": 1.0}
SNOW_FULL_COVER_CM = 5.0


def scale_fit(y, p, m):
    m = m & y.notna() & p.notna() & (p > 0)
    x, t = p[m].values, y[m].values
    return float(np.dot(x, t) / np.dot(x, x)) if np.dot(x, x) > 0 else np.nan


def geometry(pv, p, facts, par, osm, k=4, terrain_dem="dgm1"):
    """master (single array from the master data), terrain (clusters around the master angles), footprint source.
    terrain_dem 'dgm1' (default since 2026-10-22): state DGM1 on the window-method footprint of the park
    (geometry_trianel_windows.prepare: OSM ways and relations, 10 m cells); 'glo30': the GLO-30 DSM on the OSM plant
    near the master-data point (the ladders up to v4)."""
    master = [{"name": "a0", "tilt": float(p["tilt_master"]), "azimuth": float(p["az_master"]), "share": 1.0}]
    if terrain_dem == "dgm1":
        import geometry_trianel_windows as TW
        from parks_pv.geometry import terrain_arrays_grid
        try:
            P = TW.prepare(pv, with_rgb=False)
        except Exception as ex:                       # no geodata in the cache (pv4): GLO-30 below
            print(pv, "no DGM1 terrain:", type(ex).__name__, str(ex)[:80], flush=True)
            P = {"dgm": np.full(1, np.nan)}
        if np.isfinite(P["dgm"]).any():
            terr, _ = terrain_arrays_grid([dict(master[0], name="a0", source="master")], P["dgm"], P["inside"], P["RES"], P["conv"], k_total=k)
            return master, [{"name": a["name"], "tilt": a["tilt"], "azimuth": a["azimuth"], "share": a["share"]} for a in terr], f"dgm1:{P['fp_src']}"
    els = osm.get(pv, {}).get("elements", [])
    poly = terrain.pick_plant_near(els, p["lat"], p["lon"], 2.5) if els else None
    src = "osm"
    if poly is None:
        ha = float(facts.loc[pv, "area_ha"]) if pd.notna(facts.loc[pv, "area_ha"]) else par.loc[pv, "kwp_dc"] / float(np.interp(par.loc[pv, "year"], DENSITY_YEARS, DENSITY_KWP_HA))
        poly, src = terrain.buffer_polygon(p["lat"], p["lon"], ha), "mastr_area"
    t = terrain.terrain_table(poly, p["tilt_master"], p["az_master"])
    cl = arr.cluster_arrays(t["tilt_eff"].values, t["az_eff"].values, t["area_m2"].values, k)
    terr = [{"name": c.name, "tilt": c.tilt, "azimuth": c.azimuth, "share": c.share} for c in cl]
    return master, terr, src


def soiling_ratio(pm: pd.DataFrame, rain: pd.Series, tilt: float, idx: pd.DatetimeIndex, spinup_start: str, init_dry_days: int,
                  threshold_mm: float) -> pd.Series:
    """HSU ratio on an index extended backwards: real PM/rain from spinup_start, before that init_dry_days of
    mean PM without rain (emulates an unknown initial soiling state)."""
    ext = pd.date_range(pd.Timestamp(spinup_start, tz="UTC") - pd.Timedelta(days=init_dry_days), idx[-1], freq="h")
    pm_e = pm.reindex(ext)
    pm_e = pm_e.fillna(pm.loc[spinup_start:].iloc[:24 * 30].mean())
    rain_e = rain.reindex(ext).fillna(0.0)
    r = soiling.hsu_ratio(rain_e, pm_e["pm2_5"], pm_e["pm10"], tilt, cleaning_threshold_mm=threshold_mm)
    return r.reindex(idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2023-07-24")
    ap.add_argument("--end", default="2024-06-01")
    ap.add_argument("--cams-dir", required=True)
    ap.add_argument("--horizon-json", required=True)
    ap.add_argument("--corrected-dir", required=True)
    ap.add_argument("--corrected-tag", default="xgb_v5")
    ap.add_argument("--osm-json", required=True)
    ap.add_argument("--mastr-cand", default=None)
    ap.add_argument("--mastr-xml", default=None)
    ap.add_argument("--snow-dir", required=True)
    ap.add_argument("--snow-cache", required=True)
    ap.add_argument("--extras-dir", default=os.path.expanduser("~/Work/FL_Contribution/reports/figs_pv_generator"))
    ap.add_argument("--geodata-arrays", default=None, help="CSV park_id,array_id,tilt,azimuth,share; since 2026-10-22 from geometry_trianel_windows.py (geodata_arrays_trianel_windows_v6.csv)")
    ap.add_argument("--terrain-dem", choices=["dgm1", "glo30"], default="dgm1", help="terrain of the 'full' rung: state DGM1 (default since 2026-10-22) or GLO-30 (ladders up to v4)")
    ap.add_argument("--base-arrays", default=None, help="CSV park_id,array_id,tilt,azimuth,share: replaces the master-data (+terrain) geometry as the base of every ladder rung")
    ap.add_argument("--estimate-windows", nargs="*", type=int, default=[30, 90, 365], help="meter history (days from start) for the power-based angle estimate")
    ap.add_argument("--geometry-only", action="store_true", help="only the geometry rungs (faster)")
    ap.add_argument("--estimate-bank", choices=["poa", "chain"], default="poa", help="basis series for the power-based estimate")
    ap.add_argument("--estimate-prior", default=None, help="CSV pv,row_axis_deg: fix the azimuth from the mapped rows, estimate only the tilt")
    ap.add_argument("--rung-suffix", default="", help="suffix for the power-based rung names and output files")
    ap.add_argument("--out", default=os.path.expanduser("~/Work/FL_Contribution/reports/figs_pv_generator"))
    args = ap.parse_args()
    m = tc.load_masterdata()
    parks = tc.parks_table(m)
    meter = tc.load_meter_hourly(parks)
    facts = tc.mastr_facts(parks, args.mastr_cand, args.mastr_xml)
    par = tc.park_parameters(parks, facts, meter)
    osm = json.load(open(args.osm_json))
    conn = era5_db.connect()
    idx = timebase.hourly_index(args.start, args.end)
    master_st = pd.read_csv(tc.STATIONS_MASTER, dtype={"station_id": str})
    pm_ids = sorted(os.path.basename(f)[5:-4] for f in __import__("glob").glob(os.path.join(drivers.CAMS_PM_DIR, "cams_*.csv")))
    rows, geo_rows, scales = [], [], {}
    results = {}
    geo_arr = pd.read_csv(args.geodata_arrays) if args.geodata_arrays and os.path.exists(args.geodata_arrays) else pd.DataFrame()
    from pv import plausibility as PL
    pvs = [pv for pv in parks.index if pv in meter and meter[pv].notna().sum() > 1000]
    for pv in pvs:
        p = parks.loc[pv]
        drvs, _ = tc.build_drivers(p, args.start, args.end, conn, args.cams_dir, pv)
        raw = drvs["cams"]
        c = pd.read_parquet(os.path.join(args.corrected_dir, f"{pv}_cams_corrected_hourly_{args.corrected_tag}.parquet")).reindex(idx)
        corr = raw.copy()
        corr["ghi"] = c["ghi_corr"].fillna(raw["ghi"]).clip(lower=0)
        corr["dhi"] = np.minimum(c["dhi_corr"].fillna(raw["dhi"]).clip(lower=0), corr["ghi"])
        corr["bhi"] = corr["ghi"] - corr["dhi"]
        hz = tc.load_horizon(args.horizon_json, pv)
        y = meter[pv].reindex(idx)
        cap_w = float(par.loc[pv, "kwp_dc"]) * 1000.0
        master, terr, fp_src = geometry(pv, p, facts, par, osm, terrain_dem=args.terrain_dem)
        terr_master = terr
        if args.base_arrays and os.path.exists(args.base_arrays):
            ba = pd.read_csv(args.base_arrays); ba = ba[ba.park_id == pv]
            if len(ba):
                master = terr = [{"name": str(a.array_id), "tilt": float(a.tilt), "azimuth": float(a.azimuth), "share": float(a.share)} for a in ba.itertuples()]
                fp_src = f"base:{os.path.basename(args.base_arrays)}"
        fitted = json.load(open(os.path.join(args.extras_dir, f"extras_{pv}.json")))["fitted_arrays"]
        geo_rows.append({"pv": pv, "footprint_src": fp_src, "n_terrain_arrays": len(terr), "tilt_master": p["tilt_master"], "az_master": p["az_master"],
                         "terrain_arrays": "; ".join(f"{a['tilt']:.1f}/{a['azimuth']:.0f}x{a['share']:.2f}" for a in terr)})
        # snow + rain at the point (START labels)
        cache = os.path.join(args.snow_cache, f"{pv}_era5_snow.parquet")
        if os.path.exists(cache):
            sn = pd.read_parquet(cache)
        else:
            sn = snow.era5_snow_point(args.snow_dir, p["lat"], p["lon"])
            os.makedirs(args.snow_cache, exist_ok=True)
            sn.to_parquet(cache)
        sn_s = sn.copy()
        sn_s.index = sn_s.index - pd.Timedelta("1h")
        depth_cm = (sn_s["snow_depth_m"] * 100.0).reindex(idx).ffill(limit=6).fillna(0.0)
        snow_factor = snow.loss_factor((depth_cm / SNOW_FULL_COVER_CM).clip(0.0, 1.0))
        rain = sn_s["precip_mm"]
        sid, pm_dist = drivers.nearest_station(p["lat"], p["lon"], master_st, pm_ids)
        pm = drivers.cams_pm(sid, "2022-10-01", args.end)

        def cfg_for(arrays, interrow=True, horizon=True, losses=LOSSES, aging=True):
            cfg = tc.base_config(pv, p, par.loc[pv], arrays=arrays, variant="PR" if interrow else "P")
            cfg["shading"]["horizon"] = horizon
            cfg["losses"] = dict(losses)
            cfg["aging"]["enabled"] = aging
            return cfg

        def run(drv, cfg, with_snow=True):
            res = chain.run_park(cfg, drv, horizon=hz if cfg["shading"]["horizon"] else None)
            pac = res["p_ac_w"] * (snow_factor if with_snow else 1.0)
            return pac, res

        rungs = {
            "full": (corr, cfg_for(terr), True),
            **({"reference_master": (corr, cfg_for(terr_master), True)} if terr_master is not terr else {}),
            "geometry_master_only": (corr, cfg_for(master), True),
            "geometry_fitted_step1": (corr, cfg_for(fitted), True),
        }
        if len(geo_arr) and (geo_arr.park_id == pv).any():
            g = geo_arr[geo_arr.park_id == pv]
            rungs["geometry_geodata"] = (corr, cfg_for([{"name": a.array_id, "tilt": float(a.tilt), "azimuth": float(a.azimuth), "share": float(a.share)} for a in g.itertuples()]), True)
        # power-based estimate from the first N days of meter data (grid best pair, corrected driver), then applied to the whole window
        ac_lim = float(par.loc[pv, "ac_cap_meter_kw"]) * 1000.0 if np.isfinite(par.loc[pv, "ac_cap_meter_kw"]) else None
        prior = None
        if args.estimate_prior:
            pr = pd.read_csv(args.estimate_prior).set_index("pv")
            if pv in pr.index and np.isfinite(pr.loc[pv, "row_axis_deg"]) and bool(pr.loc[pv, "prior_ok"]) if "prior_ok" in pr else np.isfinite(pr.loc[pv, "row_axis_deg"]):
                prior = PL.azimuth_prior_from_rows(float(pr.loc[pv, "row_axis_deg"]))
        az_prior, prior_ew = (None, False) if prior is None else ((prior["azimuths"][0], True) if prior["east_west"] else (prior["azimuth"], False))
        est_cfg = cfg_for(terr) if args.estimate_bank == "chain" else None
        for nd in args.estimate_windows:
            nd = min(nd, int((idx[-1] - idx[0]).days))          # the meter window ends 2024-05-31: "365" is 313 days
            sl = slice(idx[0], idx[0] + pd.Timedelta(days=nd))
            try:
                est = PL.assess(y.loc[sl], corr.loc[sl], p["lat"], p["lon"], p["alt"], exclude=(depth_cm > 0.5).loc[sl], ac_limit_w=ac_lim, n_boot=0,
                                cfg=est_cfg, horizon=hz if est_cfg is not None else None, azimuth_prior=az_prior, prior_east_west=prior_ew)
                gr, mx = est["grid"], est["mixture"]
                ewm = est["east_west_model"]
                if ewm.get("preferred"):
                    arrays_est = [{"name": "east", "tilt": float(ewm["tilt"]), "azimuth": float(ewm["azimuth_east"]), "share": float(ewm["share_east"])},
                                  {"name": "west", "tilt": float(ewm["tilt"]), "azimuth": float(ewm["azimuth_west"]), "share": 1.0 - float(ewm["share_east"])}]
                elif mx["east_west"]:
                    comps = [c for c in mx["components"]]
                    tot = sum(c[2] for c in comps)
                    arrays_est = [{"name": f"m{i}", "tilt": c[0], "azimuth": c[1], "share": c[2] / tot} for i, c in enumerate(comps)]
                else:
                    arrays_est = [{"name": "est", "tilt": float(gr["tilt"]), "azimuth": float(gr["azimuth"]), "share": 1.0}]
                rungs[f"geometry_estimated_{nd}d{args.rung_suffix}"] = (corr, cfg_for(arrays_est), True)
                geo_rows.append({"pv": pv, "estimate_days": nd, "bank": est["bank"], "azimuth_prior": az_prior, "prior_east_west": prior_ew,
                                 "tilt_est": gr["tilt"], "az_est": gr["azimuth"], "east_west": mx["east_west"], "ew_model": ewm.get("preferred"),
                                 "ew_model_tilt": ewm.get("tilt"), "ew_model_share_east": ewm.get("share_east"), "n_clear": est["n_clear_hours"],
                                 "n_suspect": est["n_suspect"], "n_clipped_clear": est["n_clipped_clear"], "zenith_spread": round(est["zenith_spread"], 1), "tilt_identifiable": est["tilt_identifiable"]})
            except Exception as e:  # noqa: BLE001
                print(pv, "estimate failed", nd, str(e)[:80], flush=True)
        if not args.geometry_only:
            rungs.update({
            "no_snow": (corr, cfg_for(terr), False),
            "no_terrain_master_angles": (corr, cfg_for(master), True),
            "no_horizon": (corr, cfg_for(terr, horizon=False), True),
            "no_interrow": (corr, cfg_for(terr, interrow=False), True),
            "raw_cams": (raw, cfg_for(terr), True),
            "losses_P_default": (corr, cfg_for(terr, losses=LOSSES_P), True),
            "no_aging": (corr, cfg_for(terr, aging=False), True),
            "fitted_geometry": (corr, cfg_for(fitted), True),
            "register_only_raw_cams_no_snow": (raw, cfg_for(master, horizon=False, interrow=False, losses=LOSSES_P), False),
            })
        ps = {}
        tilt_mean = float(sum(a["tilt"] * a["share"] for a in terr))
        for name, (drv, cfg, ws) in rungs.items():
            ps[name], _ = run(drv, cfg, ws)
        for thr, init in (() if args.geometry_only else ((1.0, 0), (1.0, 30), (1.0, 90), (0.5, 30), (2.0, 30))):
            r = soiling_ratio(pm, rain, tilt_mean, idx, "2023-07-01", init, thr)
            ps[f"full_soiling_thr{thr:g}_init{init}"] = ps["full"] * r.fillna(1.0)
            rows.append({"pv": pv, "rung": f"full_soiling_thr{thr:g}_init{init}", "subset": "soiling_ratio_stats", "mean_ratio": float(r.mean()),
                         "min_ratio": float(r.min()), "pm_station_dist_km": pm_dist})
        sp = timebase.solar_position(idx, p["lat"], p["lon"], p["alt"])
        zen = sp["apparent_zenith"]
        day = zen < 90.0
        cs_g = timebase.clearsky(idx, p["lat"], p["lon"], p["alt"])["ghi"]
        kc = raw["ghi"] / cs_g.replace(0, np.nan)
        suspect = day & (kc > 0.6) & ((90.0 - zen) > 20.0) & (y < 0.5 * ps.get("reference_master", ps["full"])) & y.notna()   # mask from the master-data chain, comparable across runs
        winter = day & ~suspect & (idx >= pd.Timestamp("2023-11-15", tz="UTC")) & (idx < pd.Timestamp("2024-03-15", tz="UTC"))
        snow_days = pd.Series(np.isin(idx.date, list(set(depth_cm.index[depth_cm >= 1.0].date))), index=idx) & day & ~suspect
        subsets = {"day": day, "day_nosuspect": day & ~suspect, "cloudy": day & ~suspect & (kc < 0.5), "clear": day & ~suspect & (kc >= 0.8),
                   "winter": winter, "snow_days": snow_days, "summer": day & ~suspect & idx.month.isin([4, 5, 6, 7, 8, 9])}
        scales[pv] = scale_fit(y, ps["full"], day & ~suspect)
        for name, pser in ps.items():
            for tag, mm in subsets.items():
                b = metrics.basic(y[mm], pser[mm], cap_w)
                rows.append({"pv": pv, "rung": name, "subset": tag, "n": int((mm & y.notna()).sum()), **b})
        results[pv] = (y, ps, subsets, cap_w)
        print(pv, {k: round(metrics.basic(y[day & ~suspect], v[day & ~suspect], cap_w)["r2"], 4) for k, v in ps.items() if not k.startswith("full_soiling")}, flush=True)
    # leave-one-park-out loss calibration on the full chain
    for pv, (y, ps, subsets, cap_w) in results.items():
        loo = float(np.mean([s for k, s in scales.items() if k != pv]))
        for tag, mm in subsets.items():
            b = metrics.basic(y[mm], ps["full"][mm] * loo, cap_w)
            rows.append({"pv": pv, "rung": "full_loo_scaled", "subset": tag, "n": int((mm & y.notna()).sum()), "scale_loo": loo, "scale_own": scales[pv], **b})
    df = pd.DataFrame(rows)
    tag = ("geometry" if args.geometry_only else "long") + args.rung_suffix
    df.to_csv(os.path.join(args.out, f"ablation_trianel_{tag}.csv"), index=False)
    pd.DataFrame(geo_rows).to_csv(os.path.join(args.out, f"ablation_trianel_{tag}_geometry.csv"), index=False)
    print("\nscales (full chain, day_nosuspect):", {k: round(v, 3) for k, v in scales.items()})
    main_rungs = [r for r in df.rung.unique() if r != "full_loo_scaled"]
    for sub in ("day_nosuspect", "cloudy", "winter", "snow_days"):
        d = df[(df.subset == sub)].pivot(index="rung", columns="pv", values="r2").reindex(main_rungs + ["full_loo_scaled"])
        d["mean"] = d.mean(axis=1)
        d["delta_vs_full"] = d["mean"] - d.loc["full", "mean"]
        print(f"\nR2 {sub}\n", d.round(4).to_string())
    e = df[df.subset == "day_nosuspect"].pivot(index="rung", columns="pv", values="energy_ratio").reindex(main_rungs + ["full_loo_scaled"])
    e["mean"] = e.mean(axis=1)
    print("\nenergy ratio day_nosuspect\n", e.round(3).to_string())
    # figure: delta R2 vs full per rung (day_nosuspect, winter)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=False)
    for ax, sub in zip(axes, ("day_nosuspect", "winter")):
        d = df[(df.subset == sub) & (df.rung != "full")].pivot(index="rung", columns="pv", values="r2")
        full = df[(df.subset == sub) & (df.rung == "full")].set_index("pv")["r2"]
        delta = (d - full).reindex([r for r in main_rungs if r != "full"] + ["full_loo_scaled"])
        xs = np.arange(len(delta))
        for j, pv in enumerate(delta.columns):
            ax.scatter(xs + (j - len(delta.columns) / 2) * 0.08, delta[pv], s=14, label=pv)
        ax.plot(xs, delta.mean(axis=1), color="k", marker="_", ms=14, lw=0, label="mean")
        ax.axhline(0, color="#999", lw=0.8)
        ax.set_xticks(xs, [r.replace("full_soiling_", "soil_") for r in delta.index], rotation=60, ha="right", fontsize=7)
        ax.set_ylabel("ΔR² vs full chain")
        ax.set_title(f"hours: {sub}", loc="left")
    axes[0].legend(fontsize=7, ncol=2)
    fig.suptitle("Ablation of the parks_pv_v1 chain on the metered parks: removing one component at a time", x=0.01, ha="left", fontsize=9)
    fig.savefig(os.path.join(args.out, "fig_ablation_trianel.png"))


if __name__ == "__main__":
    main()
