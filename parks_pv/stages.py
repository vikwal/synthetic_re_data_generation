"""Stages of a PV park release, driven by a run config (parks_pv.run_config). Entry point: generate_pv.py.

    prepare    inputs of the selection that lack them: OSM/PVGIS, CAMS centre + ring (ADS), corrected CAMS, ICON-D2 SL
    baseline   register geometry per park (MaStR classes + footprint + terrain), park parameters, horizon
               -> data/<release>/baseline_{arrays,sites,params}.csv; append-only, existing parks are never recomputed
    geometry   sub-arrays per park by geometry.mode: estimate (angle estimation store, estimated where missing; register
               fallback), register (baseline) or manual (CSV) -> data/<release>/geometry_arrays.csv
    sites      final sub-arrays (geometry_arrays.csv), site table, register view
               -> data/<release>/{pv_arrays,sites,pv_masterdata_view,park_params}.csv
    configs    one chain config YAML per park -> configs/pv/_generated/<release>/
    generate   drivers (corrected CAMS, ERA5, snow) -> chain -> <release dir>/work/chain/<park>_{drivers,chain}.parquet + meta
    assemble   release frames + manifests -> <release dir>/synth_<park>.parquet, manifest_<park>.json
    tables     parks.csv, clients.csv, sites.csv, pv_arrays.csv, pv_masterdata_view.csv, README.md
generate, assemble and tables refuse a release directory that holds a FROZEN file (assert_not_frozen).
"""
import hashlib
import io
import json
import os
import shutil
import subprocess

import numpy as np
import pandas as pd
import yaml

import settings
from parks_pv import assemble as A, geometry as G, metadata as M, params as P, paths
from pv import arrays as ARR, terrain
from pv.plausibility import orientation_class

BASELINE = ("arrays", "sites", "params")


def osm_json() -> str:
    return os.path.join(paths.INPUT_DIR, "osm", "parks_osm.json")


def pvgis_json() -> str:
    return os.path.join(paths.INPUT_DIR, "osm", "pvgis_horizon.json")


def horizons_json() -> str:
    return os.path.join(paths.INPUT_DIR, "horizons_used.json")


def assert_not_frozen():
    """A release directory with a FROZEN file is never written (stages generate, assemble, tables); a test run of the
    same config takes another variant."""
    f = os.path.join(paths.RELEASE_DIR, "FROZEN")
    if os.path.exists(f):
        raise SystemExit(f"{paths.RELEASE_DIR} is frozen ({f}); run with another variant to test")


def chain_dir() -> str:
    return os.path.join(paths.WORK_DIR, "chain")


def baseline_path(kind: str) -> str:
    return os.path.join(paths.DATA_DIR, f"baseline_{kind}.csv")


def load_selection(rc: dict, only=None) -> pd.DataFrame:
    sel = pd.read_csv(paths.SELECTION, dtype={"park_id": str, "uniform_flag": str}).dropna(subset=["park_id"])   # flag "1", not 1.0
    if rc.get("overrides"):
        from utils import overrides as OVR
        sel = OVR.apply(sel, rc["overrides"], "pv")[0]
    if only:
        sel = sel[sel.park_id.isin(only)]
    return sel.reset_index(drop=True)


# ---------------------------------------------------------------- prepare

def stage_prepare(rc: dict, parks=None):
    from parks_pv import prepare
    ic = rc["prepare"]["icon"]
    prepare.run(rc["prepare"]["steps"], rc["irradiance"]["correction"], rc["gpu"], ic["name"], ic["previous"], ic["sources"],
                ic["hours"], period=(rc["period"]["start"], rc["period"]["end"]))


def write_points():
    """points.json for apply_parks: pv = park_id, lat/lon as requested from ADS, alt = DEM altitude."""
    pts = pd.read_csv(os.path.join(paths.INPUT_DIR, "cams_points.csv"))
    c = pts[pts.kind == "centre"]
    out = [{"pv": r.park_id, "lat": float(r.lat), "lon": float(r.lon), "alt": float(r.altitude_m)} for r in c.itertuples()]
    json.dump(out, open(os.path.join(paths.INPUT_DIR, "points.json"), "w"), indent=1)
    print(len(out), "points ->", os.path.join(paths.INPUT_DIR, "points.json"))


def run_correction(model_tag: str, gpu: int, start: str, end: str, parks=None):
    """Corrected CAMS (30 min and hourly) of the parks by scripts/cams_bc/apply_parks.py."""
    os.makedirs(paths.CORR_DIR, exist_ok=True)
    cmd = [settings.get("correction_python"), os.path.join(paths.REPO, "scripts", "cams_bc", "apply_parks.py"),
           "--points-json", os.path.join(paths.INPUT_DIR, "points.json"), "--cams-dir", os.path.join(paths.INPUT_DIR, "cams"),
           "--ring-dir", os.path.join(paths.INPUT_DIR, "cams_ring"), "--models-dir", os.path.join(paths.CAMS_MODELS, model_tag),
           "--out-dir", paths.CORR_DIR, "--start", start, "--end", end, "--gpu", str(gpu)]
    if parks:
        cmd += ["--parks", ",".join(parks)]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------- baseline

def site_tables(r, osm: dict, pvg: dict, terr: pd.DataFrame, row_axis: pd.DataFrame, seed: int) -> tuple:
    """Register geometry of one park: (array rows, site row, register view row, parameter row, horizon dict).
    Nominal mounts from the MaStR classes and era, footprint from OSM (else MaStR area), terrain-following arrays on
    GLO-30, horizon from PVGIS (else DEM)."""
    rng = G.park_rng(r.park_id, seed)
    par = P.park_parameters(r)
    ax = float(row_axis.loc[r.park_id, "row_axis_deg"]) if r.park_id in row_axis.index and row_axis.loc[r.park_id, "n_elongated"] >= 3 else None
    nominal, deviation = G.nominal_arrays(r, rng, ax)
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
    arrays_rows = []
    for a in arrays:
        arrays_rows.append({"park_id": r.park_id, "array_id": a["name"], "tilt": round(a["tilt"], 2), "azimuth": round(a["azimuth"], 2),
                            "share": round(a["share"], 5), "capacity_kwp": round(a["share"] * par["kwp_dc"], 1),
                            "mount_tilt": round(a.get("mount_tilt", a["tilt"]), 2), "mount_azimuth": round(a.get("mount_azimuth", a["azimuth"]), 2),
                            "source": a["source"], "module_era": par["era"], "gamma_pdc": par["gamma_pdc"], "dc_ac": par["dc_ac"],
                            "gcr": par["gcr"], "commissioning_year": par["year"]})
    t_mean = sum(a["tilt"] * a["share"] for a in arrays)
    tilts, azs = np.array([a["tilt"] for a in arrays]), np.array([a["azimuth"] for a in arrays])
    tm, am = ARR.mean_angles(tilts, azs, np.array([a["share"] for a in arrays]))
    site = {"site_id": r.park_id, "lat": r.lat, "lon": r.lon, "altitude": terr.elevation.get(r.park_id, np.nan),
            "slope": terr.slope.get(r.park_id, np.nan), "aspect": terr.aspect.get(r.park_id, np.nan),
            "tpi5": terr.tpi5.get(r.park_id, np.nan), "tpi75": terr.tpi75.get(r.park_id, np.nan), "elev_std": terr.elev_std.get(r.park_id, np.nan),
            "terrain_cls": r.terrain_cls, "footprint_src": fp_src, "footprint_ha": round(terrain.area_ha(poly), 2),
            **{k: round(v, 3) for k, v in tstats.items()}, "n_arrays": len(arrays), "tilt_mean_true": round(float(tm), 2),
            "azimuth_mean_true": round(float(am), 2), "tilt_capw_true": round(t_mean, 2),
            "register_deviation": deviation or "", "row_axis_osm_deg": ax if ax is not None else np.nan, "horizon_src": hz_src,
            "horizon_max_deg": round(max(hz.values()), 2) if hz else np.nan,
            "horizon_south_deg": round(float(hz.get("180.0", np.nan)), 2) if hz else np.nan}
    return arrays_rows, site, G.register_view(r, arrays), {"park_id": r.park_id, **par}, hz


def load_baseline(kind: str) -> pd.DataFrame:
    p = baseline_path(kind)
    key = "site_id" if kind == "sites" else "park_id"
    return pd.read_csv(p, dtype={key: str}) if os.path.exists(p) else pd.DataFrame(columns=[key])


def stage_baseline(rc: dict, parks=None):
    """Baseline tables of the selection parks that have none yet (appended, existing rows are kept). Derived only from
    the selection row (MaStR), the shared inputs (OSM, PVGIS) and the DEM: deleting the tables and rerunning gives the
    same tables. Horizons of parks new to the shared horizons_used.json are added there."""
    from pv import osm_rows
    sel = load_selection(rc, parks)
    have = set(load_baseline("params").park_id)
    todo = sel[~sel.park_id.isin(have)]
    if todo.empty:
        print(f"baseline: all {len(sel)} parks present in {paths.DATA_DIR}")
        return
    osm = json.load(open(osm_json()))
    pvg = json.load(open(pvgis_json())) if os.path.exists(pvgis_json()) else {}
    terr = pd.read_parquet(os.path.join(paths.MASTR_DIR, "pool_terrain_20250625.parquet")).set_index("park_id")
    new_t = todo[~todo.park_id.isin(terr.index)]
    if len(new_t):                                                # parks outside the MaStR pool (park specs): same metrics
        from parks_pv import park_spec
        terr = pd.concat([terr, pd.DataFrame([dict(park_id=r.park_id, **park_spec.terrain_metrics(float(r.lat), float(r.lon)))
                                              for r in new_t.itertuples()]).set_index("park_id")])
    ax = []
    for r in todo.itertuples():
        res = osm_rows.row_axis_park(osm.get(r.park_id, {}).get("elements", []), r.lat, r.lon, r.dc_kwp)
        ax.append(dict(park_id=r.park_id, row_axis_deg=res["row_axis_deg"], n_elongated=res["n_elongated"]))
    row_axis = pd.DataFrame(ax).set_index("park_id")
    hz_all = json.load(open(horizons_json())) if os.path.exists(horizons_json()) else {}
    hz_new = False
    new = {k: [] for k in BASELINE}
    for r in todo.itertuples():
        a_rows, site, _, p, hz = site_tables(r, osm, pvg, terr, row_axis, rc["seed"])
        new["arrays"] += a_rows
        new["sites"].append(site)
        new["params"].append(p)
        if r.park_id not in hz_all:
            hz_all[r.park_id], hz_new = hz, True
        print(r.park_id, "baseline:", site["footprint_src"], site["horizon_src"], f"{site['n_arrays']} arrays, row axis {row_axis.loc[r.park_id].to_dict()}", flush=True)
    if hz_new:                                                    # shared file: only touched for new parks
        os.makedirs(paths.INPUT_DIR, exist_ok=True)
        json.dump(hz_all, open(horizons_json(), "w"))
    for k in BASELINE:
        p = baseline_path(k)
        df = pd.DataFrame(new[k])
        if os.path.exists(p):                                     # existing rows as text: a CSV float round trip is not exact
            old = pd.read_csv(p, dtype=str, keep_default_na=False)
            df = pd.concat([old, pd.read_csv(io.StringIO(df.to_csv(index=False)), dtype=str, keep_default_na=False)],
                           ignore_index=True).fillna("")
        df.to_csv(p, index=False)
    print(f"baseline: {len(todo)} parks added -> {paths.DATA_DIR}")


# ---------------------------------------------------------------- geometry

GEO_COLS = ["park_id", "array_id", "tilt", "azimuth", "share", "source", "az_source", "note", "mount_tilt", "mount_azimuth", "geometry_source"]


def geometry_path() -> str:
    return os.path.join(paths.DATA_DIR, "geometry_arrays.csv")


def _baseline_rows(base_a: pd.DataFrame, pid: str, geometry_source: str) -> pd.DataFrame:
    a = base_a[base_a.park_id == pid]
    if not len(a):
        raise SystemExit(f"{pid}: no baseline arrays, run stage baseline first")
    return a.assign(source="register", az_source="register", note="", geometry_source=geometry_source).reindex(columns=GEO_COLS)


def _spec_mode(r, default: str) -> str:
    """Geometry mode of a park: the park spec's (column geometry_mode) or the run config's."""
    m = getattr(r, "geometry_mode", None)
    return m if isinstance(m, str) and m else default


def _spec_arrays(r) -> pd.DataFrame:
    """Manual sub-arrays of a park spec (column geometry_arrays, JSON) as geometry rows, or None."""
    a = getattr(r, "geometry_arrays", None)
    if not isinstance(a, str) or not a:
        return None
    a = json.loads(a)
    return pd.DataFrame(dict(park_id=r.park_id, array_id=[f"m{k}" for k in range(len(a))], tilt=[float(x["tilt"]) for x in a],
                             azimuth=[float(x["azimuth"]) for x in a], share=[float(x["share"]) for x in a], source="manual", az_source="manual",
                             note="", mount_tilt=np.nan, mount_azimuth=np.nan, geometry_source="manual"))


def _manual_rows(path: str, sel: pd.DataFrame) -> dict:
    """Sub-arrays of mode manual per park: the park spec's arrays, else the CSV of geometry.manual (park_id, tilt
    against the horizontal, azimuth true with 180 = south, share [, array_id]); shares are normalised in stage sites."""
    out = {r.park_id: _spec_arrays(r) for r in sel.itertuples() if _spec_arrays(r) is not None}
    if len(out) == len(sel):
        return out
    if not path:
        raise SystemExit(f"no manual arrays for {sorted(set(sel.park_id) - set(out))} (park spec geometry.arrays or run config geometry.manual)")
    m = pd.read_csv(path, dtype={"park_id": str})
    need = {"park_id", "tilt", "azimuth", "share"} - set(m.columns)
    if need:
        raise SystemExit(f"{path}: columns {sorted(need)} missing")
    for pid, g in m.groupby("park_id", sort=False):
        if pid in out:
            continue
        ids = g.array_id.astype(str) if "array_id" in g else [f"m{k}" for k in range(len(g))]
        out[pid] = pd.DataFrame(dict(park_id=pid, array_id=list(ids), tilt=g.tilt.values, azimuth=g.azimuth.values, share=g.share.values,
                                     source="manual", az_source="manual", note="", mount_tilt=np.nan, mount_azimuth=np.nan, geometry_source="manual"))
    miss = sorted(set(sel.park_id) - set(out))
    if miss:
        raise SystemExit(f"{path}: no manual arrays for {miss}")
    return out


def _estimated_rows(r, method: str, base_a: pd.DataFrame, base_p: pd.DataFrame, osm: dict) -> pd.DataFrame:
    """Measured arrays of the store (as text, so a stored estimate passes unchanged), else the register fallback of
    this release: baseline re-terrained on the state DGM1 (geometry_estimate.fallback_arrays) or the baseline."""
    from parks_pv import geometry_estimate as GE, geometry_store as GSt
    d = GSt.park_dir(method, r.park_id)
    m = pd.read_csv(os.path.join(d, "arrays.csv"), dtype=str, keep_default_na=False)
    if len(m):
        return m.assign(geometry_source=m.source.iloc[0]).reindex(columns=GEO_COLS)
    st = GSt.sources(method, r.park_id).status
    if (st == "no_fetcher").all():
        return _baseline_rows(base_a, r.park_id, "register (no geodata fetcher)")
    try:
        tiles, zone = GE.park_tiles(r, osm)
    except Exception as e:                                        # tiles unreachable: the baseline as it is
        print(r.park_id, "geometry fallback without tiles:", type(e).__name__, str(e)[:80], flush=True)
        tiles, zone = {}, None
    arr, src = GE.fallback_arrays(r, tiles, zone, base_a, base_p, osm)
    rows = pd.DataFrame(GE.register_rows(r, arr, src))
    return rows.assign(geometry_source=f"{src} (no accepted measurement)").reindex(columns=GEO_COLS)


def stage_geometry(rc: dict, parks=None):
    """Sub-arrays of the selection parks (all, or the given ones: their rows are replaced) by geometry.mode, or per park
    by the mode of its park spec."""
    g = rc["geometry"]
    sel = load_selection(rc, parks)
    base_a, base_p = load_baseline("arrays"), load_baseline("params")
    missing = set(sel.park_id) - set(base_p.park_id)
    assert not missing, f"run stage baseline first: {sorted(missing)}"
    mode = pd.Series([_spec_mode(r, g["mode"]) for r in sel.itertuples()], index=sel.index)
    rows = {pid: _baseline_rows(base_a, pid, "register") for pid in sel.park_id[mode == "register"]}
    if (mode == "manual").any():
        rows.update(_manual_rows(g["manual"], sel[mode == "manual"]))
    est = sel[mode == "estimate"]
    if len(est):
        from parks_pv import geometry_estimate as GE, geometry_store as GSt
        new = GE.estimate_parks(est, g["method"])
        print(f"geometry: {len(new)} parks estimated, {len(est) - len(new)} from the store {GSt.root(g['method'])}", flush=True)
        osm = {}
        if any(not len(GSt.arrays(g["method"], pid)) for pid in est.park_id):
            osm = json.load(open(osm_json()))
        rows.update({r.park_id: _estimated_rows(r, g["method"], base_a, base_p, osm) for r in est.itertuples()})
    out = pd.concat([rows[pid] for pid in sel.park_id], ignore_index=True)
    p = geometry_path()
    if parks and os.path.exists(p):                               # partial run: other parks keep their rows, selection order
        old = pd.read_csv(p, dtype=str, keep_default_na=False)
        order = list(load_selection(rc).park_id)
        out = pd.concat([old[~old.park_id.isin(out.park_id)], out.astype(str).replace("nan", "")], ignore_index=True)
        out = out.iloc[out.park_id.map({k: i for i, k in enumerate(order)}).argsort(kind="stable")]
    out.to_csv(p, index=False)
    print(f"geometry ({dict(mode.value_counts())}): {out.park_id.nunique()} parks -> {p}; "
          f"{out.groupby('park_id').geometry_source.first().str.split(':').str[0].value_counts().to_dict()}")


# ---------------------------------------------------------------- sites

def stage_sites(rc: dict, parks=None):
    """Final sub-arrays of stage geometry per park; site rows from the baseline with the measured (or manual) angles
    and the deviation from the registered orientation class."""
    sel = load_selection(rc)
    if not os.path.exists(geometry_path()):
        raise SystemExit(f"sites: {geometry_path()} missing, run stage geometry first")
    geo = pd.read_csv(geometry_path(), dtype={"park_id": str})
    s1 = load_baseline("sites").set_index("site_id")
    par = load_baseline("params").set_index("park_id")
    missing = set(sel.park_id) - set(par.index)
    assert not missing, f"run stage baseline first: {sorted(missing)}"
    arrays_rows, site_rows, view_rows = [], [], []
    for r in sel.itertuples():
        g = geo[geo.park_id == r.park_id]
        p = par.loc[r.park_id]
        if not len(g):
            raise SystemExit(f"{r.park_id}: no rows in {geometry_path()}, run stage geometry")
        src = str(g.geometry_source.iloc[0])
        arrays = []
        for x in g.itertuples():
            measured = str(x.source).startswith("windows:") or x.source == "manual"   # angles against the horizontal
            arrays.append(dict(name=str(x.array_id), tilt=float(x.tilt), azimuth=float(x.azimuth) % 360.0, share=float(x.share),
                               mount_tilt=float(x.tilt) if measured or pd.isna(x.mount_tilt) else float(x.mount_tilt),
                               mount_azimuth=float(x.azimuth) if measured or pd.isna(x.mount_azimuth) else float(x.mount_azimuth),
                               source=str(x.source), az_source=str(getattr(x, "az_source", "")), note=str(getattr(x, "note", "") if pd.notna(getattr(x, "note", "")) else "")))
        tot = sum(a["share"] for a in arrays)
        for a in arrays:
            a["share"] /= tot
            arrays_rows.append({"park_id": r.park_id, "array_id": a["name"], "tilt": round(a["tilt"], 2), "azimuth": round(a["azimuth"], 2),
                                "share": round(a["share"], 5), "capacity_kwp": round(a["share"] * p.kwp_dc, 1), "mount_tilt": round(a["mount_tilt"], 2),
                                "mount_azimuth": round(a["mount_azimuth"], 2), "source": a["source"], "az_source": a["az_source"], "note": a["note"],
                                "geometry_source": src, "module_era": p.era, "gamma_pdc": p.gamma_pdc, "dc_ac": p.dc_ac, "gcr": p.gcr,
                                "commissioning_year": p.year})
        tm, am = ARR.mean_angles(np.array([a["tilt"] for a in arrays]), np.array([a["azimuth"] for a in arrays]), np.array([a["share"] for a in arrays]))
        s = s1.loc[r.park_id].to_dict()
        if src.startswith("windows:") or src == "manual":         # deviation from the register as measured / set
            ew = any("Flanke" in a["note"] for a in arrays)
            if src == "manual":                                   # no flank notes: an opposite pair facing east/west
                from parks_pv.geometry_estimate import measured_classes
                ew = "Ost-West" in measured_classes([a["azimuth"] for a in arrays])
            cls = "Ost-West" if ew else orientation_class(arrays[int(np.argmax([a["share"] for a in arrays]))]["azimuth"])
            how = "measured" if src != "manual" else "manual"
            s["register_deviation"] = "" if cls == r.orientation_class else f"registered {r.orientation_class}, {how} {cls} ({'geodata' if how == 'measured' else 'manual'})"
        s.update(site_id=r.park_id, n_arrays=len(arrays), tilt_mean_true=round(float(tm), 2), azimuth_mean_true=round(float(am), 2),
                 tilt_capw_true=round(sum(a["tilt"] * a["share"] for a in arrays), 2), geometry_source=src)
        site_rows.append(s)
        view_rows.append(G.register_view(r, arrays))
    pd.DataFrame(arrays_rows).to_csv(os.path.join(paths.DATA_DIR, "pv_arrays.csv"), index=False)
    pd.DataFrame(site_rows).to_csv(os.path.join(paths.DATA_DIR, "sites.csv"), index=False)
    pd.DataFrame(view_rows).to_csv(os.path.join(paths.DATA_DIR, "pv_masterdata_view.csv"), index=False)
    par.reset_index().to_csv(os.path.join(paths.DATA_DIR, "park_params.csv"), index=False)
    s = pd.DataFrame(site_rows)
    print(f"{len(s)} parks -> {paths.DATA_DIR}; geometry {s.geometry_source.str.split(':').str[0].value_counts().to_dict()}; "
          f"arrays/park {s.n_arrays.mean():.2f}; register deviations measured/manual {s.register_deviation.astype(str).str.contains('(geodata|manual)').sum()}")


# ---------------------------------------------------------------- configs

def stage_configs(rc: dict, parks=None):
    sel = load_selection(rc, parks)
    arrays = pd.read_csv(os.path.join(paths.DATA_DIR, "pv_arrays.csv"))
    par = pd.read_csv(os.path.join(paths.DATA_DIR, "park_params.csv")).set_index("park_id")
    sites = pd.read_csv(os.path.join(paths.DATA_DIR, "sites.csv")).set_index("site_id")
    dates = P.commissioning_dates(sel)
    os.makedirs(paths.CONFIG_DIR, exist_ok=True)
    for r in sel.itertuples():
        a = arrays[arrays.park_id == r.park_id]
        pr = dict(par.loc[r.park_id].to_dict(), commissioning_first=f"{dates.loc[r.park_id, 'commissioning_first']:%Y-%m-%d}")
        cfg = P.park_config(r.park_id, r.lat, r.lon, float(sites.altitude.get(r.park_id, 0.0) or 0.0), pr,
                            [{"name": x.array_id, "tilt": x.tilt, "azimuth": x.azimuth, "share": x.share} for x in a.itertuples()], seed=rc["seed"],
                            cohorts=P.age_cohorts(r) if rc["aging"]["cohorts"] else None)
        with open(os.path.join(paths.CONFIG_DIR, f"{r.park_id}.yaml"), "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    print(f"{len(sel)} configs -> {paths.CONFIG_DIR}")


# ---------------------------------------------------------------- generate

def _one_generate(task):
    park_id, lat, lon, cfg_path, hz, model_tag, step, snow_model, start, period, out_dir = task
    from parks import era5_db
    from parks_pv import drivers as DRV, generate as GEN
    conn = era5_db.connect()
    try:
        drv, meta = DRV.build(park_id, lat, lon, period[0], period[1], conn, model_tag,
                              snow_cache=os.path.join(paths.SNOW_CACHE_DIR, f"{park_id}.parquet"), step=step)
    finally:
        conn.close()
    drv = drv[drv.index >= pd.Timestamp(start, tz="UTC")]       # no series before the park has its registered capacity
    cfg = yaml.safe_load(open(cfg_path))
    snow_on = snow_model != "off"
    if snow_model == "depth_slide":
        drv["icon_snow_depth_cm"], meta["icon_snow"] = DRV.icon_snow_series(park_id, drv.index)
    if snow_model != "depth_linear":
        meta["snow_model"] = snow_model
        meta["snow_params"] = GEN.snow_params(snow_model) if snow_on else None
    horizon = pd.Series({float(k): float(v) for k, v in hz.items()}).sort_index() if hz else None
    meta["age_cohorts"] = sum("era" in a for a in cfg["arrays"]) > 0     # cohort arrays in the config
    res = GEN.run(cfg, drv, horizon=horizon, snow_on=snow_on, snow_model=snow_model if snow_on else None)
    os.makedirs(out_dir, exist_ok=True)
    drv.to_parquet(os.path.join(out_dir, f"{park_id}_drivers.parquet"))
    res.to_parquet(os.path.join(out_dir, f"{park_id}_chain.parquet"))
    json.dump(meta, open(os.path.join(out_dir, f"{park_id}_meta.json"), "w"), indent=1, default=str)
    day = drv["ghi_clear_cams"] > 10
    cf = res["power_park"].mean() / (cfg["kwp_dc"] * 1000.0 / cfg["dc_ac"])
    return {"park_id": park_id, "cf_ac": float(cf), "cf_dc": float(res["power_park"].mean() / (cfg["kwp_dc"] * 1000.0)),
            "clipped_share_day": float(res.loc[day, "clipped"].mean()), "snow_loss_share": float(1 - res["power_park"].sum() / res["power_park_free"].sum()),
            "cams_fallback_share": meta["share_hours_cams_fallback"], "ring_missing_share": meta["share_hours_ring_missing"],
            "era5_solar_cell": meta["era5_solar_cell"], "era5_met_cell": meta["era5_met_cell"], "n_steps": int(len(res)), "step": step,
            "series_start": str(drv.index[0])}


def stage_generate(rc: dict, parks=None):
    assert_not_frozen()
    from concurrent.futures import ProcessPoolExecutor
    sel = load_selection(rc, parks)
    hz = json.load(open(horizons_json())) if rc["horizon"] else {}
    start, end = rc["period"]["start"], rc["period"]["end"]
    full = P.commissioning_dates(sel)["commissioning_full"]
    tasks = [(r.park_id, r.lat, r.lon, os.path.join(paths.CONFIG_DIR, f"{r.park_id}.yaml"), hz.get(r.park_id, {}), rc["irradiance"]["correction"],
              rc["step"], rc["snow"]["model"], str(max(pd.Timestamp(start), full[r.park_id]).date()), (start, end), chain_dir())
             for r in sel.itertuples()]
    rows = []
    with ProcessPoolExecutor(rc["workers"]) as ex:
        for out in ex.map(_one_generate, tasks):
            rows.append(out)
            print(out["park_id"], f"CF_ac {out['cf_ac']:.3f} clipped {out['clipped_share_day']:.3f} snow {out['snow_loss_share']:.3f} "
                  f"fallback {out['cams_fallback_share']:.3f}", flush=True)
    summary = os.path.join(paths.DATA_DIR, f"run_summary{paths.VARIANT}.csv")
    if parks and os.path.exists(summary):                         # partial run: replace the rows of these parks
        old = pd.read_csv(summary, dtype={"park_id": str})
        rows = old[~old.park_id.isin(parks)].to_dict("records") + rows
    pd.DataFrame(rows).to_csv(summary, index=False)


# ---------------------------------------------------------------- assemble, tables

def stage_assemble(rc: dict, parks=None):
    assert_not_frozen()
    sel = load_selection(rc, parks)
    par = pd.read_csv(os.path.join(paths.DATA_DIR, "park_params.csv")).set_index("park_id")
    sites = pd.read_csv(os.path.join(paths.DATA_DIR, "sites.csv")).set_index("site_id")
    arrays = pd.read_csv(os.path.join(paths.DATA_DIR, "pv_arrays.csv"))
    hz = json.load(open(horizons_json())) if rc["horizon"] else {}
    dates = P.commissioning_dates(sel)
    cdir = chain_dir()
    for r in sel.itertuples():
        drv = pd.read_parquet(os.path.join(cdir, f"{r.park_id}_drivers.parquet"))
        res = pd.read_parquet(os.path.join(cdir, f"{r.park_id}_chain.parquet"))
        meta = json.load(open(os.path.join(cdir, f"{r.park_id}_meta.json")))
        cfg_path = os.path.join(paths.CONFIG_DIR, f"{r.park_id}.yaml")
        cfg = yaml.safe_load(open(cfg_path))
        rel = A.release_frame(drv, res)
        pr = dict(par.loc[r.park_id].to_dict(), **{k: f"{v:%Y-%m-%d}" for k, v in dates.loc[r.park_id].items()})
        man = A.manifest(r, cfg, pr, meta, sites.loc[r.park_id].to_dict(),
                         arrays[arrays.park_id == r.park_id].to_dict("records"), sites.loc[r.park_id, "horizon_src"] if rc["horizon"] else "off", rel,
                         hashlib.sha256(open(cfg_path, "rb").read()).hexdigest(),
                         hashlib.sha256(json.dumps(hz.get(r.park_id, {}), sort_keys=True).encode()).hexdigest())
        man["run_config"] = {k: v for k, v in rc.items() if k != "config_path"}
        A.write_release(r.park_id, rel, man)
    print(f"{len(sel)} parks -> {paths.RELEASE_DIR}")


SNOW_TEXT = {
    "off": "off (no snow losses).",
    "depth_linear": "`depth_linear`: coverage = clip(ERA5 snow depth / 5 cm, 0, 1), factor = 1 - dc_loss_nrel(coverage, 100).",
    "depth_slide": ("`depth_slide`: ramp = clip(max(ERA5 snow depth, ICON-D2 09 UTC h_snow of the UTC day) / 5 cm, 0, 1); "
                    "coverage_t = max(ramp_t, coverage_(t-1) - 0.1 sin(tilt) dt_h) while T_air > -POA/80, else max(ramp_t, "
                    "coverage_(t-1)); tilt = share-weighted park tilt; factor = 1 - dc_loss_nrel(coverage, 100). Parks without "
                    "ICON-D2 data fall back to ERA5 alone (manifest `drivers.icon_snow.icon_status`). Driver column "
                    "`icon_snow_depth_cm` in work/chain/<park>_drivers.parquet."),
}


def write_readme(rc: dict, n_parks: int):
    src = pd.read_csv(os.path.join(paths.RELEASE_DIR, "pv_arrays.csv"), dtype={"park_id": str}).groupby("park_id").geometry_source.first()
    kind = src.str.split(":").str[0]
    fill = {"release": paths.NAME, "variant": paths.VARIANT or "-", "config": os.path.relpath(rc["config_path"], paths.REPO),
            "date": f"{pd.Timestamp.now():%Y-%m-%d}", "n_parks": n_parks, "start": rc["period"]["start"], "end": rc["period"]["end"],
            "step": rc["step"], "correction": rc["irradiance"]["correction"], "snow": SNOW_TEXT[rc["snow"]["model"]],
            "cohorts": "on" if rc["aging"]["cohorts"] else "off", "horizon": "PVGIS (DEM where PVGIS has none)" if rc["horizon"] else "off",
            "n_windows": int((kind == "windows").sum()), "n_fallback": int((kind != "windows").sum())}
    txt = open(os.path.join(paths.REPO, "parks_pv", "README_release.md")).read()
    for k, v in fill.items():
        txt = txt.replace("{" + k + "}", str(v))
    open(os.path.join(paths.RELEASE_DIR, "README.md"), "w").write(txt)


def stage_tables(rc: dict, parks=None):
    assert_not_frozen()
    sel = load_selection(rc)
    par = pd.read_csv(os.path.join(paths.DATA_DIR, "park_params.csv"))
    sites = pd.read_csv(os.path.join(paths.DATA_DIR, "sites.csv"))
    rs = pd.read_csv(os.path.join(paths.DATA_DIR, f"run_summary{paths.VARIANT}.csv"))
    parks_t = M.parks_table(sel, par, sites, rs, P.commissioning_dates(sel))
    parks_t.to_csv(os.path.join(paths.RELEASE_DIR, "parks.csv"), index=False)
    M.clients_table(parks_t).to_csv(os.path.join(paths.RELEASE_DIR, "clients.csv"), index=False)
    for f in ("sites.csv", "pv_arrays.csv", "pv_masterdata_view.csv"):
        shutil.copy(os.path.join(paths.DATA_DIR, f), os.path.join(paths.RELEASE_DIR, f))
    write_readme(rc, len(sel))
    print(parks_t[["client_id", "capacity_kw"]].groupby("client_id").sum().round(0).to_string())


STAGE_FUNCS = {"prepare": stage_prepare, "baseline": stage_baseline, "geometry": stage_geometry, "sites": stage_sites, "configs": stage_configs,
               "generate": stage_generate, "assemble": stage_assemble, "tables": stage_tables}
