#!/usr/bin/env python
"""Stages of the parks_pv_v1 release (PV counterpart of scripts/parks/run_parks_v1.py):

    sites      footprints (OSM/MaStR), terrain, true sub-arrays, horizons, register view
               -> data/parks_pv_v1/{pv_arrays.csv, sites.csv, pv_masterdata_view.csv, park_params.csv}, horizons JSON
    configs    one chain config YAML per park -> configs/pv/_generated/parks_pv_v1/
    points     points.json of the park centres for the CAMS correction (apply_parks)
    correct    run scripts/cams_bc/apply_parks.py with the cams_bc model (needs the ERA5 venv, GPU)
    generate   drivers (corrected CAMS, ERA5, snow) -> chain -> work/chain/<park>.parquet + meta
    assemble   release frames + manifests -> ${DATA_ROOT}/synthetic/solar/parks_pv_v1/
    tables     parks.csv, clients.csv, sites.csv, pv_arrays.csv, pv_masterdata_view.csv, README.md

    synthre/bin/python scripts/pv/run_parks_pv_v1.py sites [--parks ID,ID] [--seed 42]
    synthre/bin/python scripts/pv/run_parks_pv_v1.py generate --model-tag xgb_v5 --workers 8
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from parks_pv import assemble as A, geometry as G, metadata as M, params as P, paths  # noqa: E402
from pv import terrain  # noqa: E402

START, END = "2023-07-24", "2026-08-01"          # release window [start, end)
CORR_DIR = os.path.join(paths.WORK_DIR, "cams_corrected")
CHAIN_DIR = os.path.join(paths.WORK_DIR, "chain")
SNOW_CACHE = os.path.join(paths.WORK_DIR, "era5_snow")

OSM_JSON = os.path.join(paths.INPUT_DIR, "osm", "parks_osm.json")
PVGIS_JSON = os.path.join(paths.INPUT_DIR, "osm", "pvgis_horizon.json")
HORIZON_OUT = os.path.join(paths.INPUT_DIR, "horizons_used.json")


def load_selection(only=None) -> pd.DataFrame:
    sel = pd.read_csv(paths.SELECTION).dropna(subset=["park_id"])
    if only:
        sel = sel[sel.park_id.isin(only)]
    return sel.reset_index(drop=True)


def stage_sites(args):
    sel = load_selection(args.parks)
    osm = json.load(open(OSM_JSON)) if os.path.exists(OSM_JSON) else {}
    pvg = json.load(open(PVGIS_JSON)) if os.path.exists(PVGIS_JSON) else {}
    terr = pd.read_parquet(os.path.join(paths.MASTR_DIR, "pool_terrain_20250625.parquet")).set_index("park_id")
    arrays_rows, site_rows, view_rows, par_rows, horizons = [], [], [], [], {}
    for r in sel.itertuples():
        rng = G.park_rng(r.park_id, args.seed)
        par = P.park_parameters(r)
        nominal, deviation = G.nominal_arrays(r, rng)
        area_fb = float(r.area_ha) if np.isfinite(r.area_ha) and r.area_ha > 0 else par["kwp_dc"] / par["kwp_per_ha"]
        poly, fp_src = G.footprint(r, osm.get(r.park_id, {}).get("elements", []), area_fb)
        try:
            arrays, tstats = G.terrain_arrays(poly, nominal, G.MAX_ARRAYS, seed=0)
        except Exception as e:  # noqa: BLE001  (DEM tile boundary etc.)
            print(r.park_id, "terrain failed, nominal arrays kept:", str(e)[:80], flush=True)
            arrays, tstats = [dict(a, mount_tilt=a["tilt"], mount_azimuth=a["azimuth"]) for a in nominal], {}
        hz = pvg.get(r.park_id) or {}
        if len(hz) >= 24:
            hz_src = "pvgis"
        else:
            hz = {str(k): float(v) for k, v in terrain.dem_horizon(r.lat, r.lon).items()}
            hz_src = "dem"
        horizons[r.park_id] = hz
        for a in arrays:
            arrays_rows.append({"park_id": r.park_id, "array_id": a["name"], "tilt": round(a["tilt"], 2), "azimuth": round(a["azimuth"], 2),
                                "share": round(a["share"], 5), "capacity_kwp": round(a["share"] * par["kwp_dc"], 1),
                                "mount_tilt": round(a.get("mount_tilt", a["tilt"]), 2), "mount_azimuth": round(a.get("mount_azimuth", a["azimuth"]), 2),
                                "source": a["source"], "module_era": par["era"], "gamma_pdc": par["gamma_pdc"], "dc_ac": par["dc_ac"],
                                "gcr": par["gcr"], "commissioning_year": par["year"]})
        t_mean = sum(a["tilt"] * a["share"] for a in arrays)
        tilts, azs = np.array([a["tilt"] for a in arrays]), np.array([a["azimuth"] for a in arrays])
        from pv import arrays as ARR
        tm, am = ARR.mean_angles(tilts, azs, np.array([a["share"] for a in arrays]))
        site_rows.append({"site_id": r.park_id, "lat": r.lat, "lon": r.lon, "altitude": terr.elevation.get(r.park_id, np.nan),
                          "slope": terr.slope.get(r.park_id, np.nan), "aspect": terr.aspect.get(r.park_id, np.nan),
                          "tpi5": terr.tpi5.get(r.park_id, np.nan), "tpi75": terr.tpi75.get(r.park_id, np.nan), "elev_std": terr.elev_std.get(r.park_id, np.nan),
                          "terrain_cls": r.terrain_cls, "footprint_src": fp_src, "footprint_ha": round(terrain.area_ha(poly), 2),
                          **{k: round(v, 3) for k, v in tstats.items()}, "n_arrays": len(arrays), "tilt_mean_true": round(float(tm), 2),
                          "azimuth_mean_true": round(float(am), 2), "tilt_capw_true": round(t_mean, 2),
                          "register_deviation": deviation or "", "horizon_src": hz_src, "horizon_max_deg": round(max(hz.values()), 2) if hz else np.nan,
                          "horizon_south_deg": round(float(hz.get("180.0", np.nan)), 2) if hz else np.nan})
        view_rows.append(G.register_view(r, arrays))
        par_rows.append({"park_id": r.park_id, **par})
    pd.DataFrame(arrays_rows).to_csv(os.path.join(paths.DATA_DIR, "pv_arrays.csv"), index=False)
    pd.DataFrame(site_rows).to_csv(os.path.join(paths.DATA_DIR, "sites.csv"), index=False)
    pd.DataFrame(view_rows).to_csv(os.path.join(paths.DATA_DIR, "pv_masterdata_view.csv"), index=False)
    pd.DataFrame(par_rows).to_csv(os.path.join(paths.DATA_DIR, "park_params.csv"), index=False)
    os.makedirs(paths.INPUT_DIR, exist_ok=True)
    json.dump(horizons, open(HORIZON_OUT, "w"))
    s = pd.DataFrame(site_rows)
    print(f"{len(s)} parks: footprint {s.footprint_src.value_counts().to_dict()}, horizon {s.horizon_src.value_counts().to_dict()}, "
          f"arrays/park {s.n_arrays.mean():.2f}, deviations {(s.register_deviation != '').sum()}, slope_mean median {s.slope_mean.median():.2f}")


def stage_configs(args):
    sel = load_selection(args.parks)
    arrays = pd.read_csv(os.path.join(paths.DATA_DIR, "pv_arrays.csv"))
    par = pd.read_csv(os.path.join(paths.DATA_DIR, "park_params.csv")).set_index("park_id")
    sites = pd.read_csv(os.path.join(paths.DATA_DIR, "sites.csv")).set_index("site_id")
    os.makedirs(paths.CONFIG_DIR, exist_ok=True)
    for r in sel.itertuples():
        a = arrays[arrays.park_id == r.park_id]
        cfg = P.park_config(r.park_id, r.lat, r.lon, float(sites.altitude.get(r.park_id, 0.0) or 0.0), par.loc[r.park_id].to_dict(),
                            [{"name": x.array_id, "tilt": x.tilt, "azimuth": x.azimuth, "share": x.share} for x in a.itertuples()], seed=args.seed)
        with open(os.path.join(paths.CONFIG_DIR, f"{r.park_id}.yaml"), "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    print(f"{len(sel)} configs -> {paths.CONFIG_DIR}")


def stage_points(args):
    """points.json for apply_parks: pv = park_id, lat/lon as requested from ADS, alt = DEM altitude."""
    pts = pd.read_csv(os.path.join(paths.INPUT_DIR, "cams_points.csv"))
    c = pts[pts.kind == "centre"]
    out = [{"pv": r.park_id, "lat": float(r.lat), "lon": float(r.lon), "alt": float(r.altitude_m)} for r in c.itertuples()]
    json.dump(out, open(os.path.join(paths.INPUT_DIR, "points.json"), "w"), indent=1)
    print(len(out), "points ->", os.path.join(paths.INPUT_DIR, "points.json"))


def stage_correct(args):
    import subprocess
    os.makedirs(CORR_DIR, exist_ok=True)
    cmd = [os.path.expanduser("~/Work/NWP/ERA5/.venv/bin/python"), os.path.join(REPO, "scripts", "cams_bc", "apply_parks.py"),
           "--points-json", os.path.join(paths.INPUT_DIR, "points.json"), "--cams-dir", os.path.join(paths.INPUT_DIR, "cams"),
           "--ring-dir", os.path.join(paths.INPUT_DIR, "cams_ring"), "--models-dir", os.path.join(paths.CAMS_MODELS, args.model_tag),
           "--out-dir", CORR_DIR, "--start", START, "--end", END, "--gpu", str(args.gpu)]
    if args.parks:
        cmd += ["--parks", ",".join(args.parks)]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def _one_generate(task):
    park_id, lat, lon, cfg_path, hz, model_tag = task
    from parks import era5_db
    from parks_pv import drivers as DRV, generate as GEN
    conn = era5_db.connect()
    try:
        drv, meta = DRV.build(park_id, lat, lon, START, END, conn, model_tag, snow_cache=os.path.join(SNOW_CACHE, f"{park_id}.parquet"))
    finally:
        conn.close()
    cfg = yaml.safe_load(open(cfg_path))
    horizon = pd.Series({float(k): float(v) for k, v in hz.items()}).sort_index() if hz else None
    res = GEN.run(cfg, drv, horizon=horizon)
    os.makedirs(CHAIN_DIR, exist_ok=True)
    drv.to_parquet(os.path.join(CHAIN_DIR, f"{park_id}_drivers.parquet"))
    res.to_parquet(os.path.join(CHAIN_DIR, f"{park_id}_chain.parquet"))
    json.dump(meta, open(os.path.join(CHAIN_DIR, f"{park_id}_meta.json"), "w"), indent=1, default=str)
    day = drv["ghi_clear_cams"] > 10
    cf = res["power_park"].mean() / (cfg["kwp_dc"] * 1000.0 / cfg["dc_ac"])
    return {"park_id": park_id, "cf_ac": float(cf), "cf_dc": float(res["power_park"].mean() / (cfg["kwp_dc"] * 1000.0)),
            "clipped_share_day": float(res.loc[day, "clipped"].mean()), "snow_loss_share": float(1 - res["power_park"].sum() / res["power_park_free"].sum()),
            "cams_fallback_share": meta["share_hours_cams_fallback"], "ring_missing_share": meta["share_hours_ring_missing"],
            "era5_solar_cell": meta["era5_solar_cell"], "era5_met_cell": meta["era5_met_cell"], "n_hours": int(len(res))}


def stage_generate(args):
    from concurrent.futures import ProcessPoolExecutor
    sel = load_selection(args.parks)
    hz = json.load(open(HORIZON_OUT))
    tasks = [(r.park_id, r.lat, r.lon, os.path.join(paths.CONFIG_DIR, f"{r.park_id}.yaml"), hz.get(r.park_id, {}), args.model_tag) for r in sel.itertuples()]
    rows = []
    with ProcessPoolExecutor(args.workers) as ex:
        for out in ex.map(_one_generate, tasks):
            rows.append(out)
            print(out["park_id"], f"CF_ac {out['cf_ac']:.3f} clipped {out['clipped_share_day']:.3f} snow {out['snow_loss_share']:.3f} "
                  f"fallback {out['cams_fallback_share']:.3f}", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(paths.DATA_DIR, "run_summary.csv"), index=False)


def stage_assemble(args):
    import hashlib
    sel = load_selection(args.parks)
    par = pd.read_csv(os.path.join(paths.DATA_DIR, "park_params.csv")).set_index("park_id")
    sites = pd.read_csv(os.path.join(paths.DATA_DIR, "sites.csv")).set_index("site_id")
    arrays = pd.read_csv(os.path.join(paths.DATA_DIR, "pv_arrays.csv"))
    hz = json.load(open(HORIZON_OUT))
    for r in sel.itertuples():
        drv = pd.read_parquet(os.path.join(CHAIN_DIR, f"{r.park_id}_drivers.parquet"))
        res = pd.read_parquet(os.path.join(CHAIN_DIR, f"{r.park_id}_chain.parquet"))
        meta = json.load(open(os.path.join(CHAIN_DIR, f"{r.park_id}_meta.json")))
        cfg_path = os.path.join(paths.CONFIG_DIR, f"{r.park_id}.yaml")
        cfg = yaml.safe_load(open(cfg_path))
        rel = A.release_frame(drv, res)
        man = A.manifest(r, cfg, par.loc[r.park_id].to_dict(), meta, sites.loc[r.park_id].to_dict(),
                         arrays[arrays.park_id == r.park_id].to_dict("records"), sites.loc[r.park_id, "horizon_src"], rel,
                         hashlib.sha256(open(cfg_path, "rb").read()).hexdigest(),
                         hashlib.sha256(json.dumps(hz.get(r.park_id, {}), sort_keys=True).encode()).hexdigest())
        A.write_release(r.park_id, rel, man)
    print(f"{len(sel)} parks -> {paths.RELEASE_DIR}")


def stage_tables(args):
    import shutil
    sel = load_selection()
    par = pd.read_csv(os.path.join(paths.DATA_DIR, "park_params.csv"))
    sites = pd.read_csv(os.path.join(paths.DATA_DIR, "sites.csv"))
    rs = pd.read_csv(os.path.join(paths.DATA_DIR, "run_summary.csv"))
    parks = M.parks_table(sel, par, sites, rs)
    parks.to_csv(os.path.join(paths.RELEASE_DIR, "parks.csv"), index=False)
    M.clients_table(parks).to_csv(os.path.join(paths.RELEASE_DIR, "clients.csv"), index=False)
    for f in ("sites.csv", "pv_arrays.csv", "pv_masterdata_view.csv"):
        shutil.copy(os.path.join(paths.DATA_DIR, f), os.path.join(paths.RELEASE_DIR, f))
    shutil.copy(os.path.join(REPO, "parks_pv", "README_release.md"), os.path.join(paths.RELEASE_DIR, "README.md"))
    print(parks[["client_id", "capacity_kw"]].groupby("client_id").sum().round(0).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["sites", "configs", "points", "correct", "generate", "assemble", "tables"])
    ap.add_argument("--parks", default=None, help="comma-separated park ids")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model-tag", default="xgb_v5")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    a.parks = a.parks.split(",") if a.parks else None
    {"sites": stage_sites, "configs": stage_configs, "points": stage_points, "correct": stage_correct, "generate": stage_generate,
     "assemble": stage_assemble, "tables": stage_tables}[a.stage](a)


if __name__ == "__main__":
    main()
