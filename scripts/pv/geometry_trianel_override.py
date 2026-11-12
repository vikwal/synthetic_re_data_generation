#!/usr/bin/env python
"""LEGACY (2026-10-22): block method; the Trianel parks are measured by scripts/pv/geometry_trianel_windows.py.

Laser geometry of Trianel parks at CORRECTED coordinates (2026-10-20): pv6 = two MaStR locations whose DC capacity
matches the master data (see inputs/mastr/mastr_match.csv; names and coordinates are confidential and stay out of the
code), whose Trianel coordinate lies 2.6-3.4 km away on an empty field; pv1 at its (rounded) Trianel coordinate with a
larger footprint. Uses the release
pipeline pv.geodata_park.measure_park (same calls as the batch) with the footprint from an Overpass result where one
exists, else a disc. Writes reports/figs_pv_generator/geodata_states_trianel_override.csv and PNGs under
results/pv/geodata_states_override/. No coordinates in the CSV.

    synthre/bin/python scripts/pv/geometry_trianel_override.py --sites sites.csv [--osm-dir /tmp]
sites.csv columns: pv, part, lat, lon, kw, year, state, osm_json (optional path to an Overpass JSON for this part)
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts", "pv"))
from pv import geodata_park as GP, geodata_tiles as GT  # noqa: E402
from parks_pv import geometry_estimate as R  # noqa: E402

ROOT = os.environ.get("DATA_ROOT", "/mnt/lambda1/nvme1")
GEO = os.path.join(ROOT, "synthetic/solar/trianel_benchmark/inputs/geodata")
FIGS = os.path.join(REPO, "..", "FL_Contribution", "reports", "figs_pv_generator")
OUTD = os.path.join(REPO, "results", "pv", "geodata_states_override")
STATE = {"Niedersachsen": ("ni", 32), "Brandenburg": ("bb", 33), "Mecklenburg-Vorpommern": ("mv", 33), "Sachsen-Anhalt": ("st", 32), "Bayern": ("by", 32)}


def tiles(sub, prod):
    return sorted(glob.glob(os.path.join(GEO, sub, prod, "*")))


def osm_footprint(path, lat, lon, zone, max_dist=600.0):
    if not path or not os.path.exists(path):
        return None
    els = json.load(open(path)).get("elements", [])
    p = Point(*GT.Transformer.from_crs("EPSG:4326", GT.CRS[zone], always_xy=True).transform(lon, lat))
    polys = []
    for e in els:
        if e.get("type") == "way" and (e.get("tags") or {}).get("power") == "plant" and len(e.get("geometry", [])) >= 4:
            poly = GT.to_utm(Polygon([(g["lon"], g["lat"]) for g in e["geometry"]]), zone).buffer(0)
            if poly.distance(p) <= max_dist:
                polys.append(poly)
        if e.get("type") == "relation" and (e.get("tags") or {}).get("power") == "plant" and (e.get("tags") or {}).get("plant:source", "solar") == "solar":
            for mem in e.get("members", []):
                if mem.get("role") in ("outer", "") and len(mem.get("geometry", [])) >= 4:
                    poly = GT.to_utm(Polygon([(g["lon"], g["lat"]) for g in mem["geometry"]]), zone).buffer(0)
                    if poly.distance(p) <= max_dist:
                        polys.append(poly)
    return unary_union(polys) if polys else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", required=True)
    a = ap.parse_args()
    sites = pd.read_csv(a.sites)
    out, rows = [], []
    for s in sites.itertuples():
        sub, zone = STATE[s.state]
        fp = osm_footprint(getattr(s, "osm_json", None), s.lat, s.lon, zone)
        if fp is not None and fp.area > 2000:
            poly, fp_src = fp, "osm"
        else:
            radius = np.sqrt(max(s.kw, 300.0) / 600.0 * 1e4 / np.pi) * 1.3 + 60.0
            poly, fp_src = Point(*GT.Transformer.from_crs("EPSG:4326", GT.CRS[zone], always_xy=True).transform(s.lon, s.lat)).buffer(radius), "disc"
        dop = tiles(sub, "dop20") + (sorted(glob.glob(os.path.join(GEO, "dop20_*.tif"))) if sub == "by" else [])
        dgm = tiles(sub, "dgm1") or (sorted(glob.glob(os.path.join(GEO, "dgm1_*.tif"))) if sub == "by" else [])
        sources = []
        if tiles(sub, "dom1"):
            sources.append(("dom1_laser", tiles(sub, "dom1")))
        if tiles(sub, "als") or tiles(sub, "laser"):
            sources.append(("laz", tiles(sub, "als") + tiles(sub, "laser")))
        label = f"{s.pv}{s.part}"
        print(f"== {label} {s.state} {s.kw:.0f} kW {s.year} footprint {fp_src} {poly.area / 1e4:.1f} ha; sources {[(n, len(p_)) for n, p_ in sources]}", flush=True)
        o, rr = GP.measure_park(label, poly, fp_src, zone, float(s.lat), float(s.lon), float(s.kw), int(s.year), dop, dgm, sources, False, OUTD, sub)
        for x in o:
            x.update(pv=s.pv, part=s.part, kw=float(s.kw))
            print("  ", {k: x.get(k) for k in ("source", "ok", "tilt_profile", "tilt_plane_median", "tilt_plane_p10", "tilt_plane_p90", "azimuth_true", "two_flank", "fold_contrast", "n_tables", "h_ridge", "flank_m", "block", "reason")}, flush=True)
        out += o
        rows += rr
    df = R.evaluate(pd.DataFrame(out))
    df = df[[c for c in df.columns if c not in ("lat", "lon")]]
    os.makedirs(FIGS, exist_ok=True)
    df.to_csv(os.path.join(FIGS, "geodata_states_trianel_override.csv"), index=False)
    pd.DataFrame(rows).to_csv(os.path.join(FIGS, "geodata_states_trianel_override_rows.csv"), index=False)
    print(df[["pv", "part", "source", "ok", "tilt_measured", "az_measured", "ew_measured", "fold_contrast", "n_tables", "flank_m", "h_ridge", "estimator_diff_deg", "gates_passed", "accept"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
