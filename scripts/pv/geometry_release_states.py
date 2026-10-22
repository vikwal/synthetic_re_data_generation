#!/usr/bin/env python
"""Measured module geometry of the release parks (parks_pv_v1 selection) from open state geodata by the window
method (pv.geodata_windows, 2026-10-22; the block method of 2026-10-19 is geometry_release_states_legacy.py).

Per park: tiles of the state (DOP20, DGM1, surface sources: LAZ point cloud, laser DOM1, image bDOM / DOM20) ->
footprint = OSM solar plants (ways and relations) within 400 m of the register point, else a 300-m box grown while
accepted windows touch its edge (E4b) -> per surface source: height above ground -> windows -> groups -> arrays and
park acceptance (E4d) -> comparison of the main group with the MaStR orientation and tilt classes (or_agree,
tilt_agree, tilt_agree_tol as before). Arrays for the generator per park: the measured arrays of the best accepted
source (laser first, survey after commissioning or undated), else the nominal mounts of parks_pv_v1 (pv_arrays.csv)
re-terrained on the state DGM1 (parks_pv.geometry.terrain_arrays_grid), else the v1 arrays as they are.

    synthre/bin/python scripts/pv/geometry_release_states.py --state Bayern [--parks ID,ID] [--fetch-only] [--suffix _v2]
    synthre/bin/python scripts/pv/geometry_release_states.py --state Bayern --reevaluate   # classes/gates from the CSV
Outputs (never overwritten; a different --suffix makes a new version):
    data/parks_pv_v1/geodata_release_<state><suffix>.csv          one row per park and surface source
    data/parks_pv_v1/geodata_release_<state><suffix>_arrays.csv   generator arrays per park (measured or fallback)
    results/pv/geodata_release<suffix>/<park>/{windows,groups,arrays}_<source>.csv, park_<source>.json, windows_<source>.png
Tiles are cached under ${DATA_ROOT}/synthetic/solar/parks_pv_v1/inputs/geodata/."""
import argparse
import json
import os
import sys
import traceback

import numpy as np
import pandas as pd
from shapely.geometry import Point

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from parks_pv import geometry as PG, paths  # noqa: E402
from pv import geodata_bayern as GB, geodata_states as GS, geodata_tiles as GT, geodata_windows as GW  # noqa: E402
from pv.plausibility import orientation_class, tilt_class  # noqa: E402

GEO = os.path.join(paths.INPUT_DIR, "geodata")
STATE = {"Niedersachsen": ("ni", 32), "Brandenburg": ("bb", 33), "Mecklenburg-Vorpommern": ("mv", 33), "Bayern": ("by", 32),
         "Nordrhein-Westfalen": ("nw", 32), "Hessen": ("he", 32), "Sachsen": ("sn", 33), "Thüringen": ("th", 32), "Sachsen-Anhalt": ("st", 32)}
SOURCE_RANK = {"laz": 0, "dom1_laser": 1, "bdom_image": 2, "dom20_image": 3}     # point cloud before DOM1 before image models (2026-10-22)
LASER = ("laz", "dom1_laser")
FENCE_MAX_M = 400.0
TILT_TOL_DEG = 2.0
CLASS_BOUNDS = {"< 20 Grad": (-90.0, 20.0), "20 - 40 Grad": (20.0, 40.0), "40 - 60 Grad": (40.0, 60.0), "> 60 Grad": (60.0, 90.0)}
DERIVED = ("ew_measured", "tilt_measured", "az_measured", "or_cls_measured", "tilt_cls_measured", "gate_after_build", "survey_date_known",
           "accept", "laser", "source_rank", "or_agree", "tilt_agree", "tilt_agree_tol", "best")


def fetch_tiles(state: str, lat: float, lon: float, year: int, landkreis: str = None, gemeinde: str = None, attempts: int = 3,
                margin_m: float = 300.0) -> dict:
    """{product: [paths]} with products dop20, dgm1 and the surface sources. NI takes the newest survey after the
    commissioning year and returns its dates; BB adds the image-based bDOM where ALS tiles are missing; Hessen needs the
    district and municipality (packages per municipality); Thüringen reads the hand-filled cache only."""
    last = None
    for k in range(attempts):
        try:
            if state == "Bayern":
                out = {"dop20": [], "dgm1": [], "dom20": []}
                for e, n in GS.tiles_for(lat, lon, 32, margin_m):        # full tile range (GB.tiles_for skips middle tiles)
                    for prod in ("dop20", "dgm1", "dom20"):
                        try:
                            out[prod].append(GB.fetch(prod, e, n, os.path.join(GEO, "bayern")))
                        except Exception as ex:
                            print(f"  {prod} {e}_{n}: {type(ex).__name__}", flush=True)
                out.update(GS.fetch_by_laser(lat, lon, GEO, margin_m))
                return out
            if state == "Niedersachsen":
                return GS.fetch_ni(lat, lon, GEO, after=f"{year}-01-01", margin_m=margin_m)
            if state == "Brandenburg":
                return GS.fetch_bb(lat, lon, GEO, products=("dop20", "als", "bdom", "dgm1"), margin_m=margin_m)
            if state == "Nordrhein-Westfalen":
                return GS.fetch_nw(lat, lon, GEO, margin_m=margin_m)
            if state == "Hessen":
                return GS.fetch_he(lat, lon, GEO, landkreis, gemeinde, margin_m=margin_m)
            if state == "Sachsen":
                return GS.fetch_sn(lat, lon, GEO, margin_m=margin_m)
            if state == "Thüringen":
                return GS.fetch_th(lat, lon, GEO, margin_m=margin_m)
            if state == "Sachsen-Anhalt":
                return GS.fetch_st(lat, lon, GEO, margin_m=margin_m)
            return GS.FETCHERS[state](lat, lon, GEO, margin_m=margin_m)
        except RuntimeError as ex:                  # truncated transfers (curl 18) are retried
            last = ex
            print(f"  fetch attempt {k + 1} failed: {ex}", flush=True)
    raise last


def fetch_margin(r, osm: dict, zone: int) -> float:
    """Half side (m) of the square around the register point that the tiles must cover: the OSM outline (+ 25 m, the
    analysis grid adds 20 m) or, without an outline, the largest box of measure_box; at least 300 m."""
    from pyproj import Transformer
    to_m = Transformer.from_crs("EPSG:4326", GT.CRS[zone], always_xy=True).transform
    x, y = to_m(float(r.lon), float(r.lat))
    poly, _ = GW.osm_footprint((osm.get(r.park_id) or {}).get("elements", []), [Point(x, y)], to_m, FENCE_MAX_M)
    if poly is not None:
        minx, miny, maxx, maxy = poly.buffer(25).bounds
        m = max(x - minx, maxx - x, y - miny, maxy - y)
    else:
        m = max(np.sqrt(GW.BOX["area_factor"] * float(r.dc_kwp) * 5.0 * 2.2), GW.BOX["size_m"]) / 2 + GW.BOX["step_m"] + 25
    return float(max(m, 300.0))


def municipality(unit_ids: str) -> str:
    """Municipality of the park's MaStR units (most frequent), for the Hessen packages."""
    global _UNITS
    if "_UNITS" not in globals():
        _UNITS = pd.read_parquet(os.path.join(paths.MASTR_DIR, "solar_units_20250625.parquet"), columns=["EinheitMastrNummer", "Gemeinde"]).set_index("EinheitMastrNummer")
    g = _UNITS.reindex(str(unit_ids).split(";")).Gemeinde.dropna()
    return g.mode().iloc[0] if len(g) else None


def sources_from(tiles: dict) -> list:
    src = []
    if tiles.get("dom1"):
        src.append(("dom1_laser", tiles["dom1"]))
    if tiles.get("laser") or tiles.get("als"):
        src.append(("laz", (tiles.get("laser") or []) + (tiles.get("als") or [])))
    if tiles.get("bdom"):
        src.append(("bdom_image", tiles["bdom"]))
    if tiles.get("dom20"):
        src.append(("dom20_image", tiles["dom20"]))
    return src


def rasters(tiles: dict, name: str, paths_: list, bounds: tuple, zone: int, lat: float, lon: float) -> dict:
    """Height above ground and the terrain of one surface source on the grid of bounds (0.5 m)."""
    S = GW.surface_from(name, paths_, bounds, zone, GW.RES, dates=tiles.get("dom1_dates") if name == "dom1_laser" else None)
    dgm = GT.mosaic(tiles["dgm1"], bounds, GW.RES, zone)[0].astype(np.float32) if tiles.get("dgm1") else None
    if dgm is None or not np.isfinite(dgm).any():                 # no DGM1: laser ground (LAZ) or a surface percentile (rasters)
        ground = S["ground"] if S["ground"] is not None else GT.ground_from_surface(S["surf"], GW.RES)
        dgm, dgm_src = np.asarray(ground, np.float32), "laser_ground" if S["ground"] is not None else "surface_percentile"
    else:
        dgm_src = "dgm1"
    hag = GW.height_above_ground(S["surf"], dgm, S["ground"])
    return dict(hag=hag, dgm=dgm, dgm_src=dgm_src, surf=S["surf"], source=name, survey_year=S["survey_year"], survey_dates=S["survey_dates"],
                conv=GT.meridian_convergence(lat, lon, zone))


def ortho_pitch(tiles: dict, bounds: tuple, inside: np.ndarray, zone: int) -> float:
    if not tiles.get("dop20"):
        return np.nan
    rgb = GT.mosaic(tiles["dop20"], bounds, GW.RES, zone, bands=[1, 2, 3])
    return GW.orthophoto_axis(rgb, inside, GW.RES)["pitch0"]


def measure(r, tiles: dict, osm: dict, zone: int, outd: str) -> tuple:
    """(rows per surface source, {source: window result}, footprint info) for one park."""
    from pyproj import Transformer
    to_m = Transformer.from_crs("EPSG:4326", GT.CRS[zone], always_xy=True).transform
    pt = Point(*to_m(float(r.lon), float(r.lat)))
    poly, fences = GW.osm_footprint((osm.get(r.park_id) or {}).get("elements", []), [pt], to_m, FENCE_MAX_M)
    rows, outs = [], {}
    os.makedirs(outd, exist_ok=True)
    srcs = sources_from(tiles)
    if not srcs:
        return [dict(pv=r.park_id, source=None, status="failed", status_reason="keine Oberflächenquelle")], {}, dict(footprint="none")
    for name, paths_ in srcs:
        try:
            if poly is not None:
                g = GW.analysis_grid(poly, GW.RES)
                R = rasters(tiles, name, paths_, g["bounds"], zone, float(r.lat), float(r.lon))
                pitch0 = ortho_pitch(tiles, g["bounds"], g["inside"], zone)
                o = GW.measure_park(R["hag"], R["dgm"], g["inside"], R["conv"], name, pitch0, GW.RES, r.orientation_class)
                fp = dict(footprint="osm", footprint_ha=round(poly.area / 1e4, 2), n_fences=int((fences.dist_m <= FENCE_MAX_M).sum()))
                inside = g["inside"]
            else:
                def load(bounds, name=name, paths_=paths_):
                    R_ = rasters(tiles, name, paths_, bounds, zone, float(r.lat), float(r.lon))
                    R_["pitch0"] = ortho_pitch(tiles, bounds, np.ones(R_["hag"].shape, bool), zone)
                    return R_
                o = GW.measure_box(load, (pt.x, pt.y), float(r.dc_kwp), register_orientation=r.orientation_class)
                R = o["rasters"]
                inside = np.ones(R["hag"].shape, bool)
                fp = dict(footprint="box", footprint_ha=round((o["park"]["box_side_m"] ** 2) / 1e4, 2), n_fences=0)
            o["dgm"], o["inside"], o["conv"] = R["dgm"], inside, R["conv"]
            outs[name] = o
            o["win"].to_csv(os.path.join(outd, f"windows_{name}.csv"), index=False)
            o["G"].to_csv(os.path.join(outd, f"groups_{name}.csv"), index=False)
            o["arrays"].to_csv(os.path.join(outd, f"arrays_{name}.csv"), index=False)
            park = dict(o["park"], survey_year=R["survey_year"], survey_dates=R["survey_dates"], dgm_src=R["dgm_src"], **fp)
            json.dump(park, open(os.path.join(outd, f"park_{name}.json"), "w"), indent=1, default=str)
            try:
                GW.plot_park(o, R["hag"], os.path.join(outd, f"windows_{name}.png"), f"{r.park_id}: {name}, Befliegung {R['survey_year']}, {park['status']}")
            except Exception as ex:                                    # a figure must never fail the measurement
                print("  fig error", r.park_id, name, ex, flush=True)
            G0 = o["G"].iloc[0] if len(o["G"]) else None
            A0 = o["arrays"][o["arrays"].group == 0] if len(o["arrays"]) else o["arrays"]
            rows.append(dict(pv=r.park_id, source=name, survey_year=R["survey_year"], survey_dates=str(R["survey_dates"])[:200], dgm_src=R["dgm_src"],
                             **fp, **{k: v for k, v in park.items() if k not in ("source", "survey_year", "survey_dates", "dgm_src", "footprint", "footprint_ha", "n_fences", "box_steps")},
                             g0_tilt=G0.tilt_horiz if G0 is not None else np.nan, g0_tilt_p10=G0.tilt_p10 if G0 is not None else np.nan,
                             g0_tilt_p90=G0.tilt_p90 if G0 is not None else np.nan, g0_azimuth=float(A0.azimuth.iloc[0]) if len(A0) else np.nan,
                             g0_az_source=A0.az_source.iloc[0] if len(A0) else "", g0_ew=bool(G0.ew) if G0 is not None else False,
                             g0_pitch=G0.pitch if G0 is not None else np.nan, g0_share=G0.share if G0 is not None else np.nan,
                             g0_n_windows=int(G0.n_windows) if G0 is not None else 0, g0_n_sure=int(G0.n_sure) if G0 is not None else 0,
                             arrays=str([(round(a.tilt, 1), round(a.azimuth), round(a.share, 2), a.az_source) for a in o["arrays"].itertuples()])))
        except Exception as e:
            traceback.print_exc()
            rows.append(dict(pv=r.park_id, source=name, status="failed", status_reason=f"{type(e).__name__}: {str(e)[:100]}"))
    return rows, outs, dict(poly=poly)


def evaluate(df: pd.DataFrame) -> pd.DataFrame:
    """Register comparison of the main group and the choice of the best source per park. A measured park needs
    status accepted and a survey after the commissioning year (unknown survey date: flagged, not rejected).
    or_agree is only defined where the azimuth was measured (an azimuth taken from the register class agrees by
    construction)."""
    df = df.copy()
    col = lambda c, d=np.nan: df[c] if c in df.columns else pd.Series(d, index=df.index)
    df["ew_measured"] = col("g0_ew", False).fillna(False).astype(bool)
    df["tilt_measured"] = col("g0_tilt").astype(float)
    df["az_measured"] = col("g0_azimuth").astype(float)
    df["or_cls_measured"] = [("Ost-West" if e else (orientation_class(a) if np.isfinite(a) else "")) for e, a in zip(df.ew_measured, df.az_measured)]
    df["tilt_cls_measured"] = [tilt_class(t) if np.isfinite(t) else "" for t in df.tilt_measured]
    sy = col("survey_year").astype(float)
    df["survey_date_known"] = sy.notna()
    df["gate_after_build"] = (sy >= col("year_built").astype(float)) | sy.isna()
    df["accept"] = (col("status", "") == "accepted") & df.gate_after_build
    df["laser"] = col("source", "").isin(LASER)
    df["source_rank"] = col("source", "").map(SOURCE_RANK).fillna(9)
    az_measured = col("g0_az_source", "").isin(["measured", "axis"])
    df["or_agree"] = np.where(az_measured, df.or_cls_measured == col("orientation_class_reg", ""), np.nan)
    reg_t = col("tilt_class_reg", "")
    df["tilt_agree"] = df.tilt_cls_measured == reg_t
    df["tilt_agree_tol"] = [bool(np.isfinite(t) and c in CLASS_BOUNDS and CLASS_BOUNDS[c][0] - TILT_TOL_DEG <= t < CLASS_BOUNDS[c][1] + TILT_TOL_DEG)
                            for t, c in zip(df.tilt_measured, reg_t)]
    best = df[df.accept].sort_values(["pv", "source_rank"]).drop_duplicates("pv")
    df["best"] = df.index.isin(best.index)
    return df


def v1_nominal(arrays_v1: pd.DataFrame, park_id: str) -> list:
    """Nominal mounts of parks_pv_v1 (pv_arrays.csv): one entry per nominal array (array_id prefix before the
    terrain cluster suffix), share summed, mount angles relative to the ground."""
    a = arrays_v1[arrays_v1.park_id == park_id]
    out = []
    for nom, g in a.groupby(a.array_id.str.rsplit("_", n=1).str[0], sort=False):
        out.append(dict(name=nom, tilt=float(g.mount_tilt.iloc[0]), azimuth=float(g.mount_azimuth.iloc[0]), share=float(g.share.sum()),
                        source=str(g.source.iloc[0])))
    return out


def fallback_arrays(r, tiles: dict, zone: int, arrays_v1: pd.DataFrame, osm: dict) -> tuple:
    """(arrays, source) of a park without an accepted measurement: the nominal mounts of parks_pv_v1 re-terrained on the
    state DGM1 over the v1 footprint (parks_pv.geometry.footprint: OSM plant within 2.5 km, else the MaStR area as a
    disc), so that v1 and v2 differ only in the terrain model (GLO-30 -> DGM1); without DGM1 the v1 arrays unchanged."""
    nominal = v1_nominal(arrays_v1, r.park_id)
    if nominal and tiles.get("dgm1"):
        par = pd.read_csv(os.path.join(paths.DATA_DIR, "park_params.csv"), dtype={"park_id": str}).set_index("park_id").loc[r.park_id]
        area_fb = float(r.area_ha) if pd.notna(r.area_ha) and r.area_ha > 0 else float(par.kwp_dc) / float(par.kwp_per_ha)
        poly_ll, _ = PG.footprint(r, (osm.get(r.park_id) or {}).get("elements", []), area_fb)
        g = GW.analysis_grid(GT.to_utm(poly_ll, zone), GW.RES, inner_m=0.0)
        m_ = GT.mosaic(tiles["dgm1"], g["bounds"], GW.RES, zone)           # None when no DGM1 tile touches the v1 footprint
        dgm = m_[0].astype(np.float32) if m_ is not None else None
        if dgm is not None and np.isfinite(dgm[g["inside"]]).mean() > 0.5:
            arr_, _ = PG.terrain_arrays_grid(nominal, dgm, g["inside"], GW.RES, GT.meridian_convergence(float(r.lat), float(r.lon), zone))
            return arr_, "register_v1+dgm1"
    a = arrays_v1[arrays_v1.park_id == r.park_id]
    return [dict(name=x.array_id, tilt=x.tilt, azimuth=x.azimuth, share=x.share, mount_tilt=x.mount_tilt, mount_azimuth=x.mount_azimuth)
            for x in a.itertuples()], "register_v1"


def park_arrays(r, df_park: pd.DataFrame, outs: dict, arrays_v1: pd.DataFrame, tiles: dict = None, zone: int = None, osm: dict = None) -> list:
    """Generator arrays of one park: measured (best accepted source, read from the stored arrays CSV of the park) or
    the fallback above."""
    best = df_park[df_park.best]
    if len(best):
        src = best.source.iloc[0]
        A = outs[src]["arrays"] if src in outs else pd.read_csv(os.path.join(REPO, "results", "pv", f"geodata_release{SUFFIX[0]}", r.park_id, f"arrays_{src}.csv"))
        return [dict(park_id=r.park_id, array_id=f"g{int(a.group)}_{k}", tilt=round(a.tilt, 2), azimuth=round(a.azimuth, 2), share=round(a.share, 5),
                     source=f"windows:{src}", az_source=a.az_source, note=a.note if pd.notna(a.note) else "", mount_tilt=np.nan, mount_azimuth=np.nan)
                for k, a in enumerate(A.itertuples())]
    arr_, src = fallback_arrays(r, tiles or {}, zone, arrays_v1, osm or {})
    return [dict(park_id=r.park_id, array_id=a["name"], tilt=round(a["tilt"], 2), azimuth=round(a["azimuth"], 2), share=round(a["share"], 5),
                 source=src, az_source="register", note="", mount_tilt=round(a.get("mount_tilt", np.nan), 2), mount_azimuth=round(a.get("mount_azimuth", np.nan), 2))
            for a in arr_]


SUFFIX = ["_v2"]


def summary(df: pd.DataFrame, label: str) -> None:
    pd.set_option("display.width", 250)
    print(f"\n{label}: {df.pv.nunique()} parks; accepted at {df[df.accept].pv.nunique()} (laser {df[df.accept & df.laser].pv.nunique()}); "
          f"status per source {df.status.value_counts().to_dict()}")
    b = df[df.best]
    if len(b):
        om = b[pd.notna(b.or_agree)]
        print(f"orientation class agrees {int(om.or_agree.astype(bool).sum())}/{len(om)} (measured azimuth); tilt class {int(b.tilt_agree.sum())}/{len(b)} "
              f"(+-{TILT_TOL_DEG:.0f} deg: {int(b.tilt_agree_tol.sum())}/{len(b)})")
        cols = ["pv", "source", "footprint", "n_ok", "tab_share_accepted", "orientation_class_reg", "or_cls_measured", "tilt_class_reg", "tilt_measured",
                "tilt_ends_med", "az_measured", "g0_az_source", "ew_measured", "survey_year", "year_built"]
        print(b[[c for c in cols if c in b.columns]].round(2).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default="Bayern", choices=list(STATE))
    ap.add_argument("--parks", default=None, help="comma-separated park ids (default: all of the state)")
    ap.add_argument("--fetch-only", action="store_true")
    ap.add_argument("--suffix", default="_v2")
    ap.add_argument("--reevaluate", action="store_true", help="recompute classes and the best source from the stored CSV only")
    ap.add_argument("--arrays-only", action="store_true", help="recompute <state><suffix>_arrays.csv from the stored run (overwrites it)")
    a = ap.parse_args()
    code, zone = STATE[a.state]
    out_csv = os.path.join(paths.DATA_DIR, f"geodata_release_{a.state.lower()}{a.suffix}.csv")
    arr_csv = out_csv.replace(".csv", "_arrays.csv")
    if a.reevaluate:
        raw = pd.read_csv(out_csv)
        df = evaluate(raw[[c for c in raw.columns if c not in DERIVED]])
        df.to_csv(out_csv, index=False)
        summary(df, a.state)
        return
    SUFFIX[0] = a.suffix
    if a.arrays_only:                                         # recompute the generator arrays from the stored run (cached tiles), measurement unchanged
        df = pd.read_csv(out_csv, dtype={"pv": str})
        sel = pd.read_csv(paths.SELECTION, dtype={"park_id": str}); sel = sel[sel.bundesland == a.state]
        osm = json.load(open(os.path.join(paths.INPUT_DIR, "osm", "parks_osm.json")))
        arrays_v1 = pd.read_csv(os.path.join(paths.DATA_DIR, "pv_arrays.csv"), dtype={"park_id": str})
        rows_a = []
        for r in sel.itertuples():
            try:
                tiles = fetch_tiles(a.state, float(r.lat), float(r.lon), int(r.year), r.landkreis, municipality(r.unit_ids) if a.state == "Hessen" else None,
                                    margin_m=fetch_margin(r, osm, zone))
            except Exception as ex:
                print("  tiles", r.park_id, type(ex).__name__, flush=True)
                tiles = {}
            rows_a += park_arrays(r, df[df.pv == r.park_id], {}, arrays_v1, tiles, zone, osm)
        pd.DataFrame(rows_a).to_csv(arr_csv, index=False)
        print("written", arr_csv, pd.DataFrame(rows_a).groupby("park_id").source.first().value_counts().to_dict())
        return
    if not a.fetch_only and (os.path.exists(out_csv) or os.path.exists(arr_csv)):
        raise SystemExit(f"{out_csv} exists - results are versioned, choose another --suffix")
    sel = pd.read_csv(paths.SELECTION, dtype={"park_id": str})
    sel = sel[sel.bundesland == a.state]
    if a.parks:
        sel = sel[sel.park_id.isin(a.parks.split(","))]
    osm = json.load(open(os.path.join(paths.INPUT_DIR, "osm", "parks_osm.json")))
    arrays_v1 = pd.read_csv(os.path.join(paths.DATA_DIR, "pv_arrays.csv"), dtype={"park_id": str})
    outd_root = os.path.join(REPO, "results", "pv", f"geodata_release{a.suffix}")
    rows, arr_rows = [], []
    for r in sel.itertuples():
        print(f"== {r.park_id} {a.state} {r.dc_kwp:.0f} kWp {r.year} {r.orientation_class} / {r.tilt_class}", flush=True)
        tiles = {}
        try:
            margin = fetch_margin(r, osm, zone)
            tiles = fetch_tiles(a.state, float(r.lat), float(r.lon), int(r.year), r.landkreis, municipality(r.unit_ids) if a.state == "Hessen" else None,
                                margin_m=margin)
            if a.fetch_only:
                print("  tiles:", {k: len(v) for k, v in tiles.items()}, flush=True)
                continue
            o, outs, _ = measure(r, tiles, osm, zone, os.path.join(outd_root, r.park_id))
        except Exception as e:
            traceback.print_exc()
            o, outs = [dict(pv=r.park_id, source=None, status="failed", status_reason=f"{type(e).__name__}: {str(e)[:100]}")], {}
        for x in o:
            x.update(state_code=code, orientation_class_reg=r.orientation_class, tilt_class_reg=r.tilt_class, dc_kwp=float(r.dc_kwp), year_built=int(r.year))
        dfp = evaluate(pd.DataFrame(o))
        arr_rows += park_arrays(r, dfp, outs, arrays_v1, tiles, zone, osm)
        rows += o
        print("  ", [(x.get("source"), x.get("status"), x.get("n_ok"), round(x.get("g0_tilt", np.nan), 1), x.get("arrays"), x.get("status_reason")) for x in o], flush=True)
    if a.fetch_only:
        return
    df = evaluate(pd.DataFrame(rows))
    os.makedirs(paths.DATA_DIR, exist_ok=True)
    df.to_csv(out_csv, index=False)
    pd.DataFrame(arr_rows).to_csv(arr_csv, index=False)
    summary(df, a.state)
    print("written", out_csv, arr_csv)


if __name__ == "__main__":
    main()
