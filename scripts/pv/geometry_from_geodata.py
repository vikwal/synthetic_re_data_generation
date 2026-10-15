#!/usr/bin/env python
"""Measured module geometry for the selected parks from open state geodata (Bavaria first): fetch DOP20/DOM20/DGM1
tiles, row direction + pitch from the orthophoto, table planes from DOM/DGM (OSM table polygons, else DOM
segmentation), 1 m rasters, and a per-park summary with <= 4 arrays from k-means on the table normals.

    synthre/bin/python scripts/pv/geometry_from_geodata.py --state Bayern [--parks ID,ID]
Outputs: ${DATA_ROOT}/synthetic/solar/parks_pv_v1/inputs/geodata/<state>/<park>_{tables.csv,rasters.tif},
         data/parks_pv_v1/geodata_geometry.csv (park summary), geodata_arrays.csv (measured arrays)."""
import argparse
import json
import os
import sys
import tempfile

import numpy as np
import pandas as pd
import rasterio
from rasterio.merge import merge
from shapely.geometry import Polygon
from shapely.ops import transform as stransform, unary_union
from pyproj import Transformer

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from parks_pv import paths  # noqa: E402
from pv import arrays as arr, geodata_bayern as GB, geodata_geometry as GG, imagery, terrain  # noqa: E402

TO_UTM = Transformer.from_crs("EPSG:4326", "EPSG:25832", always_xy=True)


def mosaic(product, tiles, out_dir):
    paths_ = [GB.fetch(product, e, n, out_dir) for e, n in tiles]
    if len(paths_) == 1:
        return paths_[0]
    key = "_".join(f"{e}_{n}" for e, n in tiles)
    out = os.path.join(out_dir, f"{product}_mosaic_{key}.tif")
    if not os.path.exists(out):
        srcs = [rasterio.open(p) for p in paths_]
        data, tf = merge(srcs)
        meta = srcs[0].meta.copy()
        meta.update(height=data.shape[1], width=data.shape[2], transform=tf)
        with rasterio.open(out, "w", **meta) as dst:
            dst.write(data)
        for s in srcs:
            s.close()
    return out


def osm_tables(elements, footprint_utm):
    polys = []
    for e in elements:
        if e.get("type") != "way" or e.get("tags", {}).get("power") != "generator" or not e.get("geometry"):
            continue
        coords = [(c["lon"], c["lat"]) for c in e["geometry"]]
        if len(coords) < 4:
            continue
        p = stransform(lambda x, y: TO_UTM.transform(x, y), Polygon(coords))
        if p.is_valid and p.area >= 20 and p.intersects(footprint_utm.buffer(50)):
            polys.append(p)
    return polys


def one(r, osm, out_dir, area_fallback_ha):
    els = osm.get(r.park_id, {}).get("elements", [])
    fp = terrain.pick_plant_near(els, r.lat, r.lon, 2.5) if els else None
    fp_src = "osm"
    if fp is None:
        fp, fp_src = terrain.buffer_polygon(r.lat, r.lon, area_fallback_ha), "mastr_area"
    fp_utm = stransform(lambda x, y: TO_UTM.transform(x, y), fp)
    minx, miny, maxx, maxy = fp_utm.buffer(60).bounds
    tiles = [(e, n) for e in range(int(minx // 1000), int(maxx // 1000) + 1) for n in range(int(miny // 1000), int(maxy // 1000) + 1)]
    dop, dom, dgm = (mosaic(p, tiles, out_dir) for p in ("dop20", "dom20", "dgm1"))
    # footprint check against the DOM: if the mapped/registered footprint holds < 0.3 ha of module cells, or the
    # point lies outside it, take the module field nearest to the coordinate from the DOM itself
    m0, _, _, res0 = GG.module_mask(dop, dom, dgm, fp_utm, min_height=0.3)
    x0, y0 = TO_UTM.transform(r.lon, r.lat)
    mod0_ha = m0.sum() * res0 * res0 / 1e4
    # the module field is always located from the DOM (periodic stripes near the point); a mapped OSM polygon is
    # kept only when it agrees with that field (contains its centre and is not more than twice its hull)
    osm_ok = False
    if True:
        R = float(getattr(r, "search_radius_m", 600.0))          # MaStR coordinates are precise; rounded ones need ~1500 m
        tiles_s = [(e, n) for e in range(int((x0 - R) // 1000), int((x0 + R) // 1000) + 1) for n in range(int((y0 - R) // 1000), int((y0 + R) // 1000) + 1)]
        dom_s, dgm_s, dop_s = mosaic("dom20", tiles_s, out_dir), mosaic("dgm1", tiles_s, out_dir), mosaic("dop20", tiles_s, out_dir)
        exp_ha = float(getattr(r, "expected_module_ha", np.nan)) if np.isfinite(getattr(r, "expected_module_ha", np.nan)) else None
        hull, mod_ha = GG.footprint_from_dom(dom_s, dgm_s, x0, y0, radius_m=R, min_height=0.3, expected_module_ha=exp_ha, dop_path=dop_s)
        if hull is not None:
            if fp_src == "osm" and fp_utm.contains(hull.centroid) and fp_utm.area <= 2.0 * hull.area:
                pass                                          # mapped polygon agrees with the DOM field: keep it
            else:
                fp_utm, fp_src = hull, "dom"
            minx, miny, maxx, maxy = fp_utm.buffer(60).bounds
            tiles2 = [(e, n) for e in range(int(minx // 1000), int(maxx // 1000) + 1) for n in range(int(miny // 1000), int(maxy // 1000) + 1)]
            if tiles2 != tiles:
                dop, dom, dgm = (mosaic(p, tiles2, out_dir) for p in ("dop20", "dom20", "dgm1"))
    rows = imagery.row_geometry(dop, fp_utm)
    m, h, tfb, res = GG.module_mask(dop, dom, dgm, fp_utm, min_height=0.3)
    ax_dom, pitch_dom, share_dom = GG.row_axis_from_mask(m, res)
    rows.update({"row_axis_dom": ax_dom, "pitch_dom": pitch_dom, "share_dom": share_dom, "module_cells_ha": float(m.sum() * res * res / 1e4)})
    polys = osm_tables(els, fp_utm)
    tab_src = "osm"
    if len(polys) < 3:
        polys, tab_src = GG.row_bands(m, tfb, res, ax_dom, pitch_dom), "dom_row_bands"
    t = GG.table_planes(dom, dgm, polys, min_height=0.3, row_axis_deg=ax_dom)
    prof = GG.profile_geometry(dom, dgm, fp_utm, ax_dom, pitch_dom)
    dom_date = ""
    for e_, n_ in tiles:
        tp = os.path.join(out_dir, f"dom20_{e_}_{n_}.tif")
        if os.path.exists(tp):
            with rasterio.open(tp) as d:
                dom_date = d.tags().get("TIFFTAG_DATETIME", "") or dom_date
    if t.empty:
        return {"park_id": r.park_id, "status": "no_tables", "footprint_src": fp_src, "table_src": tab_src, "dom_date": dom_date, **{f"row_{k}": v for k, v in rows.items() if k != "peaks"}}, None
    t["geometry"] = [polys[i] for i in t["table"]]
    t = t[(t.plane_rms < 1.5) & (t.height_above_ground.between(0.3, 5.0))].copy()
    w = t.area_m2 / t.area_m2.sum()
    tm, am = arr.mean_angles(t.tilt.values, t.azimuth.values, w.values)
    cl = arr.cluster_arrays(t.tilt.values, t.azimuth.values, t.area_m2.values, 4)
    t.drop(columns="geometry").to_csv(os.path.join(out_dir, f"{r.park_id}_tables.csv"), index=False)
    ras = GG.rasters_1m(t, fp_utm, dgm)
    with rasterio.open(os.path.join(out_dir, f"{r.park_id}_rasters.tif"), "w", driver="GTiff", height=ras["slope"].shape[0], width=ras["slope"].shape[1],
                       count=4, dtype="float32", crs="EPSG:25832", transform=ras["transform"], nodata=np.nan) as dst:
        for i, k in enumerate(("slope", "aspect", "tilt", "azimuth")):
            dst.write(ras[k].astype(np.float32), i + 1)
            dst.set_band_description(i + 1, k)
    summ = {"park_id": r.park_id, "status": "ok", "footprint_src": fp_src, "table_src": tab_src, "dom_date": dom_date, "n_tables": int(len(t)),
            "table_area_ha": float(t.area_m2.sum() / 1e4), "tilt_capw": float((t.tilt * w).sum()), "tilt_mean_normal": float(tm), "azimuth_mean_normal": float(am),
            "tilt_p10": float(t.tilt.quantile(0.1)), "tilt_p90": float(t.tilt.quantile(0.9)), "azimuth_std": float(t.azimuth.std()),
            "two_face_share": float(t[t.n_faces == 2].area_m2.sum() / t.area_m2.sum()),
            "height_median": float(t.height_above_ground.median()), "plane_rms_median": float(t.plane_rms.median()),
            "ground_slope_capw": float((t.ground_slope * w).sum()), "terrain_following": float(np.mean(np.abs(t.tilt - t.ground_slope) < 1.0)),
            **{f"row_{k}": v for k, v in rows.items() if k != "peaks"},
            "profile_type": prof.get("type"), "profile_faces": "; ".join(f"{f['tilt']:.1f}/{f['azimuth']:.0f}x{f['share']:.2f}" for f in prof.get("faces", [])),
            "profile_ridge_m": prof.get("height_ridge"), "profile_min_m": prof.get("height_min"),
            "arrays": "; ".join(f"{c.tilt:.1f}/{c.azimuth:.0f}x{c.share:.2f}" for c in cl)}
    exp_ha = float(getattr(r, "expected_module_ha", np.nan))
    prof_faces = prof.get("faces", [])
    if prof.get("type") == "single_face" and prof_faces and len(t) >= 3:
        # side of the single face: the planes know which way the tables face
        az_pl = float(am)
        f = prof_faces[0]
        if abs(((f["azimuth"] - az_pl) + 180.0) % 360.0 - 180.0) > 90.0:
            f["azimuth"] = (f["azimuth"] + 180.0) % 360.0
            # the flank tilt stays; mark the flip
            prof["side_flipped_to_planes"] = True
    prof_tilt = float(sum(f["tilt"] * f["share"] for f in prof_faces)) if prof_faces else np.nan
    flank_w = min(prof.get("flank_left_width", 0.0) or 0.0, prof.get("flank_right_width", 0.0) or 0.0) if prof.get("type") == "two_faces" else max(prof.get("flank_left_width", 0.0) or 0.0, prof.get("flank_right_width", 0.0) or 0.0)
    year = int(getattr(r, "year", 0) or 0)
    dom_year = int(dom_date[:4]) if dom_date[:4].isdigit() else 0
    gates = {"gate_dom_after_build": bool(dom_year and year and dom_year > year),
             "gate_stripes": bool(share_dom >= 0.15),
             "gate_area": bool(np.isfinite(exp_ha) and exp_ha > 0 and 0.5 <= (m.sum() * res * res / 1e4) / exp_ha <= 2.0),
             "gate_profile_planes_agree": bool(np.isfinite(prof_tilt) and abs(prof_tilt - float((t.tilt * w).sum())) <= 4.0),
             "gate_flank_width": bool(flank_w >= 3.0)}
    summ["profile_faces"] = "; ".join(f"{f['tilt']:.1f}/{f['azimuth']:.0f}x{f['share']:.2f}" for f in prof_faces)
    gates["gates_passed"] = int(sum(gates.values()))
    gates["accept"] = bool(gates["gate_dom_after_build"] and gates["gate_stripes"] and gates["gate_area"] and gates["gate_flank_width"])
    summ.update(gates)
    summ.update({"expected_module_ha": exp_ha, "measured_module_ha": float(m.sum() * res * res / 1e4), "profile_tilt": prof_tilt, "flank_width_m": flank_w})
    arrays_rows = [{"park_id": r.park_id, "array_id": c.name, "tilt": c.tilt, "azimuth": c.azimuth, "share": c.share, "source": "geodata_planes"} for c in cl]
    arrays_rows += [{"park_id": r.park_id, "array_id": f"p{i}", "tilt": f["tilt"], "azimuth": f["azimuth"], "share": f["share"], "source": "geodata_profile"} for i, f in enumerate(prof.get("faces", []))]
    return summ, arrays_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default="Bayern")
    ap.add_argument("--parks", default=None)
    a = ap.parse_args()
    sel = pd.read_csv(paths.SELECTION).dropna(subset=["park_id"])
    sel = sel[sel.bundesland == a.state]
    if a.parks:
        sel = sel[sel.park_id.isin(a.parks.split(","))]
    osm = json.load(open(os.path.join(paths.INPUT_DIR, "osm", "parks_osm.json")))
    out_dir = os.path.join(paths.INPUT_DIR, "geodata", a.state.lower())
    os.makedirs(out_dir, exist_ok=True)
    summ, arrays_rows = [], []
    from parks_pv.params import WP_M2
    from pv import modules
    sel = sel.copy()
    sel["expected_module_ha"] = [float(x.dc_kwp) * 1000.0 / WP_M2[modules.era_of(int(x.year))] / 1e4 for x in sel.itertuples()]
    for r in sel.itertuples():
        try:
            s, ar = one(r, osm, out_dir, float(r.area_ha) if np.isfinite(r.area_ha) and r.area_ha > 0 else 5.0)
        except Exception as e:  # noqa: BLE001
            s, ar = {"park_id": r.park_id, "status": f"error: {str(e)[:100]}"}, None
        summ.append(s)
        if ar:
            arrays_rows += ar
        print(r.park_id, s["status"], {k: (round(v, 1) if isinstance(v, float) else v) for k, v in s.items() if k in ("n_tables", "tilt_capw", "azimuth_mean_normal", "row_row_axis_deg", "row_pitch_m", "table_src")}, "| register:", r.orientation_class, r.tilt_class, flush=True)
    pd.DataFrame(summ).to_csv(os.path.join(paths.DATA_DIR, f"geodata_geometry_{a.state.lower()}.csv"), index=False)
    pd.DataFrame(arrays_rows).to_csv(os.path.join(paths.DATA_DIR, f"geodata_arrays_{a.state.lower()}.csv"), index=False)


if __name__ == "__main__":
    main()
