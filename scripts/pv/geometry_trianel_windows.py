#!/usr/bin/env python
"""Module geometry of the metered Trianel parks (pv1..pv7) by the window method (pv.geodata_windows, 2026-10-22).

prepare(pv) gathers the inputs exactly as cells 0-2b of the window notebook did (pv_sim2real/notebooks/
trianel_windows_<pv>.ipynb, which now imports this function): master data (ranking by capacity -> pv1..pv7),
MaStR candidates (state, UTM zone) and the capacity match (register point, commissioning year, register class), UTM
point and meridian convergence, OSM footprint (plants as ways and relations within 600 m of the Trianel or the
register point; else the bounds of the earlier Bavarian run; else a disc from the capacity), analysis grid, DOP20
orthophoto with the row axis and pitch of the orthophoto spectrum, surface (LAZ point cloud, else laser DOM1, else
image bDOM), DGM1 and the height above ground. run(pv) adds the window method and writes the outputs.

    synthre/bin/python scripts/pv/geometry_trianel_windows.py [--pv pv1 --pv pv5] [--table F/geodata_arrays_trianel_windows_v6.csv]
Outputs (confidential, stay on l2; no coordinates in the CSVs):
    results/pv/geodata_trianel_windows/<pv>_windows.csv, <pv>_groups.csv, <pv>_arrays.csv, <pv>_park.json, <pv>_windows.png
    reports/figs_pv_generator/geodata_arrays_trianel_windows_v6.csv (park_id, array_id, tilt, azimuth, share, source; the
    format of v5 and of ablation_trianel.py --geodata-arrays / trianel_twins.py --laser-csv)"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pv import geodata_tiles as GT, geodata_windows as GW  # noqa: E402
import trianel_common as TC  # noqa: E402

ROOT = os.environ.get("DATA_ROOT", "/mnt/lambda1/nvme1")
B = os.path.join(ROOT, "synthetic/solar/trianel_benchmark/inputs")
GEO = os.path.join(B, "geodata")
FIGS = os.path.expanduser("~/Work/FL_Contribution/reports/figs_pv_generator")
OUTD = os.path.join(REPO, "results", "pv", "geodata_trianel_windows")
STATE = {"Niedersachsen": ("ni", 32), "Brandenburg": ("bb", 33), "Mecklenburg-Vorpommern": ("mv", 33), "Sachsen-Anhalt": ("st", 32),
         "Bayern": ("by", 32), "Rheinland-Pfalz": ("rp", 32)}
FENCE_MAX_M = 600.0


def _tiles(sub: str, prod: str) -> list:
    return sorted(glob.glob(os.path.join(GEO, sub, prod, "*")))


def prepare(pv: str, res: float = GW.RES, with_rgb: bool = True) -> dict:
    """All inputs of one park up to the height above ground (cells 0-2b of the window notebook)."""
    from pyproj import Transformer
    from shapely.geometry import Point, Polygon
    import rasterio
    P = dict(PV=pv, RES=res)
    # 0a master data
    master = TC.load_masterdata()
    rows = master[master.pv == pv]
    P.update(master=master, rows=rows, kw_master=float(rows.kw.sum()), lat=float(rows.lat.iloc[0]), lon=float(rows.lon.iloc[0]))
    # 0b MaStR: candidates around the Trianel point (state, zone), capacity match (register point, year, class)
    mc = pd.read_csv(os.path.join(B, "mastr", "mastr_candidates.csv"))
    units = pd.read_parquet(os.path.join(ROOT, "synthetic/solar/mastr/solar_units_20250625.parquet"),
                            columns=["EinheitMastrNummer", "Bundesland", "Hauptausrichtung", "HauptausrichtungNeigungswinkel", "Nettonennleistung", "Inbetriebnahmedatum"])
    cand_all = mc[mc.pv == pv].merge(units, on="EinheitMastrNummer", how="left", suffixes=("", "_units"))
    cand_all["hit"] = (cand_all.dist_km <= 0.5) & (pd.to_numeric(cand_all.Nettonennleistung, errors="coerce") >= 1000)
    cand = cand_all[cand_all.hit] if cand_all.hit.any() else cand_all
    state = cand.Bundesland.dropna().mode().iloc[0]
    sub, zone = STATE[state]
    match = pd.read_csv(os.path.join(B, "mastr", "mastr_match.csv"))
    mm = match[(match.pv == pv) & (match.d_km <= 50)].sort_values("d_km")
    year_built = int(mm.iloc[0].year) if len(mm) and pd.notna(mm.iloc[0].year) else int(str(cand.Inbetriebnahmedatum.min())[:4])
    reg_lat, reg_lon = (float(mm.iloc[0].lat), float(mm.iloc[0].lon)) if len(mm) else (None, None)
    reg_orient = str(mm.iloc[0].orient).split("/")[0] if len(mm) and pd.notna(mm.iloc[0].orient) else None
    P.update(cand_all=cand_all, cand=cand, state=state, sub=sub, zone=zone, mm=mm, year_built=year_built, reg_lat=reg_lat, reg_lon=reg_lon,
             register_orientation=reg_orient, register_tilt_class=str(mm.iloc[0].tilt).split("/")[0] if len(mm) else None)
    # 0c UTM and meridian convergence
    to_m = Transformer.from_crs("EPSG:4326", GT.CRS[zone], always_xy=True).transform
    pt_tri = Point(*to_m(P["lon"], P["lat"]))
    pt_reg = Point(*to_m(reg_lon, reg_lat)) if reg_lat is not None else None
    conv = GT.meridian_convergence(P["lat"], P["lon"], zone)
    P.update(to_m=to_m, pt_tri=pt_tri, pt_reg=pt_reg, conv=conv)
    # 0d OSM footprint
    osm = json.load(open(os.path.join(B, "osm", "trianel_pv_osm.json")))[pv]
    _, fences = GW.osm_footprint(osm.get("elements", []), [pt_tri, pt_reg], to_m, FENCE_MAX_M)
    from parks_pv.params import DENSITY_KWP_HA, DENSITY_YEARS       # main-park separation (a), as in geometry_release_states.footprint
    poly, _, _ = GW.select_fences(fences, P["kw_master"] / float(np.interp(year_built, DENSITY_YEARS, DENSITY_KWP_HA)) * 1e4, FENCE_MAX_M)
    far = pt_reg is not None and len(mm) and float(mm.iloc[0].d_km) > 0.5           # the Trianel point is far off -> register point
    old_raster = os.path.join(GEO, f"{pv}_rasters.tif")
    if poly is not None:
        fp_src = "osm"
    elif os.path.exists(old_raster):
        with rasterio.open(old_raster) as r:
            bb = r.bounds
        poly, fp_src = Polygon([(bb.left, bb.bottom), (bb.right, bb.bottom), (bb.right, bb.top), (bb.left, bb.top)]), "bayern_run"
    else:
        centre = pt_reg if far else pt_tri
        radius = np.sqrt(max(P["kw_master"], 300.0) / 600.0 * 1e4 / np.pi) * 1.3 + 60.0
        poly, fp_src = centre.buffer(radius), "disc"
    park_pt = poly.centroid if fp_src == "osm" else (pt_reg if far else pt_tri)
    P.update(osm=osm, fences=fences, poly=poly, fp_src=fp_src, park_pt=park_pt)
    # 1a analysis grid, 1b orthophoto + row axis / pitch of the orthophoto spectrum
    g = GW.analysis_grid(poly, res)
    P.update(g)
    minx, miny, maxx, maxy = g["bounds"]
    P.update(minx=minx, miny=miny, maxx=maxx, maxy=maxy, extent=[0, g["W"] * res, 0, g["H"] * res])
    dop_files = _tiles(sub, "dop20") + (sorted(glob.glob(os.path.join(GEO, "dop20_*.tif"))) if sub == "by" else [])
    rgb = GT.mosaic(dop_files, g["bounds"], res, zone, bands=[1, 2, 3])
    ax = GW.orthophoto_axis(rgb, g["inside"], res)
    P.update(dop_files=dop_files, axis0=ax["axis0"], pitch0=ax["pitch0"], ortho_rows=ax["rows"])
    if with_rgb:
        P["rgb"] = rgb
    # 2a surface: point cloud, else laser DOM1, else image bDOM
    laz = _tiles(sub, "laser") + _tiles(sub, "als")
    laz_hit = [f for f in laz if f.lower().endswith((".laz", ".las")) and _laz_touches(f, g["bounds"])]
    dom, bdom = _tiles(sub, "dom1"), _tiles(sub, "bdom")
    if laz_hit:
        S = GW.surface_from("laz", laz_hit, g["bounds"], zone, res)
    elif dom:
        S = GW.surface_from("dom1_laser", dom, g["bounds"], zone, res)
    elif bdom:
        S = GW.surface_from("bdom_image", bdom, g["bounds"], zone, res)
    else:
        raise RuntimeError("no surface source in the cache for " + pv)
    P.update(surf=S["surf"], ground=S["ground"], source=S["source"], survey_year=S["survey_year"], survey_dates=S["survey_dates"])
    # 2b DGM1 and height above ground
    dgm_files = _tiles(sub, "dgm1") or (sorted(glob.glob(os.path.join(GEO, "dgm1_*.tif"))) if sub == "by" else [])
    dgm = GT.mosaic(dgm_files, g["bounds"], res, zone)[0].astype(np.float32)
    P.update(dgm=dgm, hag=GW.height_above_ground(S["surf"], dgm, S["ground"]))
    return P


def _laz_touches(path: str, bounds: tuple) -> bool:
    import laspy
    minx, miny, maxx, maxy = bounds
    with laspy.open(path) as lf:
        h_ = lf.header
        return not (h_.maxs[0] <= minx or h_.mins[0] >= maxx or h_.maxs[1] <= miny or h_.mins[1] >= maxy)


def run(pv: str, P: dict = None, outd: str = OUTD, figure: bool = True) -> dict:
    """Window method on the prepared inputs; writes the per-park outputs (no coordinates in the CSVs)."""
    P = P or prepare(pv, with_rgb=False)
    out = GW.measure_park(P["hag"], P["dgm"], P["inside"], P["conv"], P["source"], P["pitch0"], P["RES"], P["register_orientation"],
                          kwp=P["kw_master"], centre_xy=(P["park_pt"].x - P["minx"], P["park_pt"].y - P["miny"]))
    os.makedirs(outd, exist_ok=True)
    out["win"].to_csv(os.path.join(outd, f"{pv}_windows.csv"), index=False)
    out["G"].to_csv(os.path.join(outd, f"{pv}_groups.csv"), index=False)
    out["arrays"].to_csv(os.path.join(outd, f"{pv}_arrays.csv"), index=False)
    park = dict(out["park"], pv=pv, footprint=P["fp_src"], survey_year=P["survey_year"], survey_dates=P["survey_dates"], year_built=P["year_built"],
                state_code=P["sub"], register_orientation=P["register_orientation"], register_tilt_class=P["register_tilt_class"],
                operator=[(float(r.tilt0), float(r.tilt1), float(r.az0), float(r.az1), float(r.kw)) for r in P["rows"].itertuples()])
    json.dump(park, open(os.path.join(outd, f"{pv}_park.json"), "w"), indent=1, default=str)
    if figure:
        GW.plot_park(out, P["hag"], os.path.join(outd, f"{pv}_windows.png"), f"{pv}: {P['source']}, Befliegung {P['survey_year']}, {park['status']}", P["RES"])
    out["park"] = park
    return out


def table_rows(pv: str, out: dict) -> list:
    """Rows of the anonymous arrays table (format of geodata_arrays_trianel_windows_v5.csv)."""
    rows, G = [], out["G"].set_index("group") if len(out["G"]) else None
    for k, a in enumerate(out["arrays"].itertuples()):
        g = G.loc[a.group]
        kind = "E-W, " if g.ew else ""
        note = f", {a.note}" if a.note and not g.ew else ""
        rows.append(dict(park_id=pv, array_id=f"a{k}", tilt=round(float(a.tilt), 2), azimuth=round(float(a.azimuth), 1), share=round(float(a.share), 4),
                         source=f"windows module 2026-10-22 ({int(g.n_windows)} Fenster, {kind}horizontal, az {a.az_source}{note}); park {out['park']['status']}"))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pv", action="append", default=None)
    ap.add_argument("--table", default=os.path.join(FIGS, "geodata_arrays_trianel_windows_v6.csv"))
    ap.add_argument("--no-figure", action="store_true")
    a = ap.parse_args()
    rows, summ = [], []
    for pv in a.pv or ["pv1", "pv2", "pv3", "pv5", "pv6", "pv7"]:
        out = run(pv, figure=not a.no_figure)
        rows += table_rows(pv, out)
        p = out["park"]
        summ.append(dict(pv=pv, source=p["source"], survey=p["survey_year"], status=p["status"], n_windows=p["n_windows"], n_ok=p["n_ok"],
                         tab_share=round(p["tab_share_accepted"], 2), tilt=round(p["tilt_med"], 2), tilt_ends=round(p["tilt_ends_med"], 2),
                         tilt_rows=round(p["tilt_rows_med"], 2), groups=p["n_groups"], warn_rejected=p["warn_rejected"],
                         arrays=str([(round(x.tilt, 1), round(x.azimuth), round(x.share, 2), x.az_source) for x in out["arrays"].itertuples()])))
        print(summ[-1], flush=True)
    if os.path.exists(a.table):
        raise SystemExit(f"{a.table} exists - results are versioned, never overwritten")
    pd.DataFrame(rows).to_csv(a.table, index=False)
    print(pd.DataFrame(summ).to_string(index=False))
    print("written", a.table)


if __name__ == "__main__":
    main()
