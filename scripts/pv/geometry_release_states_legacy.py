#!/usr/bin/env python
"""LEGACY (2026-10-22): block method, replaced by scripts/pv/geometry_release_states.py (window method,
pv.geodata_windows). Kept to reproduce the release CSVs of 2026-10-19 (geodata_release_<state>.csv).

Measured module geometry of the release parks (parks_pv_v1 selection) from open state geodata, laser first:
Bavaria = LAZ point cloud (geodaten.bayern.de) + DOP20 + DGM1 (+ DOM20 image for comparison); Niedersachsen
(DOM1 laser via STAC, newest survey after commissioning), Brandenburg (ALS LAZ where covered, bDOM image otherwise),
Mecklenburg-Vorpommern (DOM1 laser XYZ) via pv.geodata_states.
Per park: pv.geodata_park.measure_park (same pipeline that resolved the Trianel parks, incl. the east-west pair of
pv7), then quality gates and the comparison with the MaStR orientation and tilt classes (evaluate()).

    synthre/bin/python scripts/pv/geometry_release_states_legacy.py --state Bayern [--parks ID,ID] [--fetch-only]
    synthre/bin/python scripts/pv/geometry_release_states_legacy.py --state Bayern --reevaluate   # gates only, from the CSV
Outputs: data/parks_pv_v1/geodata_release_<state>.csv (one row per park and surface source, with gates and class
         agreement), geodata_release_<state>_rows.csv (row-axis peaks and blocks),
         results/pv/geodata_release/<park>_hag_<source>.png (height above ground + folded profile, no imagery).
Tiles are cached under ${DATA_ROOT}/synthetic/solar/parks_pv_v1/inputs/geodata/."""
import argparse
import json
import os
import re
import sys
import traceback

import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from parks_pv import paths  # noqa: E402
from pv import geodata_bayern as GB, geodata_park as GP, geodata_states as GS, geodata_tiles as GT  # noqa: E402
from pv.plausibility import orientation_class, tilt_class  # noqa: E402

GEO = os.path.join(paths.INPUT_DIR, "geodata")
OUTD = os.path.join(REPO, "results", "pv", "geodata_release")
STATE = {"Niedersachsen": ("ni", 32), "Brandenburg": ("bb", 33), "Mecklenburg-Vorpommern": ("mv", 33), "Bayern": ("by", 32)}
SOURCE_RANK = {"laz": 0, "dom1_laser": 0, "bdom_image": 1, "dom20_image": 2}
LASER = ("laz", "dom1_laser")
# gates, from the Trianel runs (pv1/pv3/pv5/pv7 laser ok; pv6 with 3 tables and pv7 DOM20 with contrast 0.10 wrong)
GATES = {"fold_contrast": 0.5, "n_tables": 5, "flank_m": 1.5, "h_ridge": (0.8, 4.5), "estimator_agree_deg": 3.0}
EW_AZ_TOL_DEG = 35.0    # flank direction of an east-west pair within 35 deg of east or west
TILT_TOL_DEG = 2.0      # laser vs power-based tilt at the Trianel parks: within 2 deg
CLASS_BOUNDS = {"< 20 Grad": (-90.0, 20.0), "20 - 40 Grad": (20.0, 40.0), "40 - 60 Grad": (40.0, 60.0), "> 60 Grad": (60.0, 90.0)}
DERIVED = ("ew_measured", "tilt_measured", "az_measured", "estimator_diff_deg", "or_cls_measured", "tilt_cls_measured",
           "surface_year", "survey_date_known", "gates_passed", "accept", "laser", "source_rank", "or_agree", "tilt_agree",
           "tilt_agree_tol", "best")


def footprint(park_id: str, lat: float, lon: float, kw: float, zone: int, osm: dict) -> tuple:
    """(polygon in CRS[zone], source): OSM plant polygons within 400 m of the register point, else a disc."""
    p = Point(*GT.Transformer.from_crs("EPSG:4326", GT.CRS[zone], always_xy=True).transform(lon, lat))
    plants = []
    for e in (osm.get(park_id) or {}).get("elements", []):
        if e.get("type") == "way" and (e.get("tags") or {}).get("power") == "plant" and len(e.get("geometry", [])) >= 4:
            poly = GT.to_utm(Polygon([(g["lon"], g["lat"]) for g in e["geometry"]]), zone).buffer(0)
            if poly.distance(p) <= 400:
                plants.append(poly)
    if plants:
        return unary_union(plants), "osm"
    radius = np.sqrt(max(kw, 300.0) / 600.0 * 1e4 / np.pi) * 1.3 + 60.0
    return p.buffer(radius), "disc"


def fetch_tiles(state: str, lat: float, lon: float, year: int, attempts: int = 3) -> dict:
    """{product: [paths]} with products dop20, dgm1 and the surface sources. NI takes the newest survey after the
    commissioning year and returns its dates; BB adds the image-based bDOM where ALS tiles are missing."""
    last = None
    for k in range(attempts):
        try:
            if state == "Bayern":
                out = {"dop20": [], "dgm1": [], "dom20": []}
                for e, n in GB.tiles_for(lat, lon):
                    for prod in ("dop20", "dgm1", "dom20"):
                        try:
                            out[prod].append(GB.fetch(prod, e, n, os.path.join(GEO, "bayern")))
                        except Exception as ex:
                            print(f"  {prod} {e}_{n}: {type(ex).__name__}", flush=True)
                out.update(GS.fetch_by_laser(lat, lon, GEO))
                return out
            if state == "Niedersachsen":
                return GS.fetch_ni(lat, lon, GEO, after=f"{year}-01-01")
            if state == "Brandenburg":
                return GS.fetch_bb(lat, lon, GEO, products=("dop20", "als", "bdom", "dgm1"))
            return GS.FETCHERS[state](lat, lon, GEO)
        except RuntimeError as ex:                  # truncated transfers (curl 18) are retried
            last = ex
            print(f"  fetch attempt {k + 1} failed: {ex}", flush=True)
    raise last


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


def surface_year(row: pd.Series) -> float:
    """Acquisition year of the surface source (LAZ header creation dates or STAC datetimes), else NaN."""
    ys = [int(y) for y in re.findall(r"(20\d\d)-\d\d-\d\d", str(row.get("dates", "")))]
    return float(min(ys)) if ys else np.nan


def _col(df: pd.DataFrame, name: str, default=np.nan) -> pd.Series:
    return df[name] if name in df.columns else pd.Series(default, index=df.index)


def evaluate(df: pd.DataFrame) -> pd.DataFrame:
    """Measured geometry per source row, quality gates and the comparison with the register classes.
    East-west pair (two flanks, symmetric ridge): tilt = mean of the two flank tilts of the folded profile (the plane
    fit sees one flank only), orientation class Ost-West. Single face: tilt = median table plane, cross-checked against
    the profile tilt (two independent estimators within 3 deg)."""
    df = df.copy()
    ok = _col(df, "ok", False).fillna(False).astype(bool)
    az = _col(df, "azimuth_true").astype(float)
    # an east-west pair has symmetric flanks that face east and west; symmetric north/south flanks are not one
    ew_dir = ((az % 180.0) - 90.0).abs() <= EW_AZ_TOL_DEG
    two = (_col(df, "two_flank", False).fillna(False).astype(bool) & (_col(df, "ridge_sym", 0.0).fillna(0.0) >= 0.6) & ew_dir)
    tl, tr, tp, tpl = (_col(df, c).astype(float) for c in ("tilt_left", "tilt_right", "tilt_profile", "tilt_plane_median"))
    df["ew_measured"] = two
    df["tilt_measured"] = np.where(two, 0.5 * (tl + tr), tpl)
    df["az_measured"] = az
    # tilt against the horizontal (2026-10-21): hag-based tilt + terrain slope in the facing direction
    tlh, trh, tph = (_col(df, c).astype(float) for c in ("tilt_left_horiz", "tilt_right_horiz", "tilt_plane_horiz"))
    df["tilt_measured_horiz"] = np.where(two, 0.5 * (tlh + trh), tph)
    df["slope_facing_deg"] = np.degrees(np.arctan(_col(df, "slope_facing").astype(float)))
    df["estimator_diff_deg"] = np.where(two, (tl - tr).abs(), (tp - tpl).abs())
    df["or_cls_measured"] = [("Ost-West" if e else (orientation_class(a) if np.isfinite(a) else ""))
                             for e, a in zip(two, df["az_measured"])]
    df["tilt_cls_measured"] = [tilt_class(t) if np.isfinite(t) else "" for t in df["tilt_measured"]]
    sy = df.apply(surface_year, axis=1)
    df["surface_year"] = sy
    df["survey_date_known"] = sy.notna()
    lo, hi = GATES["h_ridge"]
    gates = {"gate_ok": ok,
             "gate_contrast": _col(df, "fold_contrast").astype(float) >= GATES["fold_contrast"],
             "gate_tables": _col(df, "n_tables", 0).fillna(0) >= GATES["n_tables"],
             "gate_flank": _col(df, "flank_m").astype(float) >= GATES["flank_m"],
             "gate_ridge": _col(df, "h_ridge").astype(float).between(lo, hi),
             "gate_estimators": pd.Series(df["estimator_diff_deg"] <= GATES["estimator_agree_deg"], index=df.index),
             # unknown survey date: flagged (survey_date_known), not rejected
             "gate_after_build": (sy >= _col(df, "year_built").astype(float)) | sy.isna()}
    for k, v in gates.items():
        df[k] = v.fillna(False).astype(bool)
    df["gates_passed"] = df[list(gates)].sum(axis=1)
    df["accept"] = df[list(gates)].all(axis=1)
    df["laser"] = _col(df, "source", "").isin(LASER)
    df["source_rank"] = _col(df, "source", "").map(SOURCE_RANK).fillna(9)
    reg_t = _col(df, "tilt_class_reg", "")
    df["or_agree"] = df["or_cls_measured"] == _col(df, "orientation_class_reg", "")
    df["tilt_agree"] = df["tilt_cls_measured"] == reg_t
    # tolerant tilt agreement: a measurement within TILT_TOL_DEG of the registered class counts as agreeing
    df["tilt_agree_tol"] = [bool(np.isfinite(t) and c in CLASS_BOUNDS and CLASS_BOUNDS[c][0] - TILT_TOL_DEG <= t < CLASS_BOUNDS[c][1] + TILT_TOL_DEG)
                            for t, c in zip(df["tilt_measured"], reg_t)]
    best = df[df.accept & df.laser].sort_values(["pv", "source_rank"]).drop_duplicates("pv")
    df["best"] = df.index.isin(best.index)
    return df


def summary(df: pd.DataFrame, label: str) -> None:
    pd.set_option("display.width", 250)
    laser = df[df.laser]
    after = laser[laser.survey_date_known & laser.gate_after_build].pv.nunique()
    unknown = laser[~laser.survey_date_known].pv.nunique()
    print(f"\n{label}: {df.pv.nunique()} parks; laser source at {laser.pv.nunique()}; survey after build at {after} "
          f"(survey date unknown at {unknown}); accepted laser measurement at {int(df.best.sum())}")
    b = df[df.best]
    if len(b):
        print(f"orientation class agrees {int(b.or_agree.sum())}/{len(b)}; tilt class agrees {int(b.tilt_agree.sum())}/{len(b)} "
              f"(with +-{TILT_TOL_DEG:.0f} deg tolerance {int(b.tilt_agree_tol.sum())}/{len(b)})")
        cols = ["pv", "source", "orientation_class_reg", "or_cls_measured", "tilt_class_reg", "tilt_measured", "az_measured",
                "ew_measured", "estimator_diff_deg", "fold_contrast", "n_tables", "surface_year", "year_built"]
        print(b[cols].round(2).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default="Bayern", choices=list(STATE))
    ap.add_argument("--parks", default=None, help="comma-separated park ids (default: all of the state)")
    ap.add_argument("--fetch-only", action="store_true")
    ap.add_argument("--surface-only", action="store_true")
    ap.add_argument("--suffix", default="")
    ap.add_argument("--reevaluate", action="store_true", help="recompute gates and classes from the stored CSV only")
    a = ap.parse_args()
    code, zone = STATE[a.state]
    out_csv = os.path.join(paths.DATA_DIR, f"geodata_release_{a.state.lower()}{a.suffix}.csv")
    if a.reevaluate:
        raw = pd.read_csv(out_csv)
        raw = raw[[c for c in raw.columns if not c.startswith("gate_") and c not in DERIVED]]
        df = evaluate(raw)
        df.to_csv(out_csv, index=False)
        summary(df, a.state)
        return
    sel = pd.read_csv(paths.SELECTION, dtype={"park_id": str})
    sel = sel[sel.bundesland == a.state]
    if a.parks:
        sel = sel[sel.park_id.isin(a.parks.split(","))]
    osm = json.load(open(os.path.join(paths.INPUT_DIR, "osm", "parks_osm.json")))
    out, rows = [], []
    for r in sel.itertuples():
        print(f"== {r.park_id} {a.state} {r.dc_kwp:.0f} kWp {r.year} {r.orientation_class} / {r.tilt_class}", flush=True)
        try:
            tiles = fetch_tiles(a.state, float(r.lat), float(r.lon), int(r.year))
            if a.fetch_only:
                print("  tiles:", {k: len(v) for k, v in tiles.items()}, flush=True)
                continue
            poly, fp_src = footprint(r.park_id, float(r.lat), float(r.lon), float(r.dc_kwp), zone, osm)
            o, rr = GP.measure_park(r.park_id, poly, fp_src, zone, float(r.lat), float(r.lon), float(r.dc_kwp), int(r.year),
                                    tiles.get("dop20", []), tiles.get("dgm1", []), sources_from(tiles), a.surface_only, OUTD, code)
            for x in o:
                x.update(orientation_class_reg=r.orientation_class, tilt_class_reg=r.tilt_class, dc_kwp=float(r.dc_kwp))
                if x.get("source") == "dom1_laser" and tiles.get("dom1_dates"):
                    x["dates"] = str(sorted(tiles["dom1_dates"].values()))
            out += o
            rows += rr
            print("  ", [(x.get("source"), x.get("ok"), x.get("tilt_plane_median"), x.get("azimuth_true"), x.get("two_flank"),
                          x.get("fold_contrast"), x.get("n_tables"), x.get("reason")) for x in o], flush=True)
        except Exception as e:
            traceback.print_exc()
            out.append(dict(pv=r.park_id, state_code=code, source=None, ok=False, reason=f"{type(e).__name__}: {str(e)[:100]}",
                            orientation_class_reg=r.orientation_class, tilt_class_reg=r.tilt_class, dc_kwp=float(r.dc_kwp),
                            year_built=int(r.year)))
    if a.fetch_only:
        return
    df = evaluate(pd.DataFrame(out))
    os.makedirs(paths.DATA_DIR, exist_ok=True)
    df.to_csv(out_csv, index=False)
    pd.DataFrame(rows).to_csv(os.path.join(paths.DATA_DIR, f"geodata_release_{a.state.lower()}{a.suffix}_rows.csv"), index=False)
    summary(df, a.state)


if __name__ == "__main__":
    main()
