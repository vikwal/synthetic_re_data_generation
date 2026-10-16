"""Row axis per portfolio park from the park's own OSM table polygons (fix of 2026-10-16: the Overpass response
covers 3 km and used to be averaged over neighbouring installations). Writes data/parks_pv_v1/osm_row_geometry.csv
with the schema the release pipeline reads (park_id, n_generator_ways, n_elongated, row_axis_deg, n_plant, tags,
register, row_axis_class) plus selection diagnostics, and keeps the previous file as *_before_parkfix.csv.
Usage: synthre/bin/python scripts/pv/osm_row_geometry.py"""
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from parks_pv import paths  # noqa: E402
from pv import osm_rows  # noqa: E402


def row_axis_class(axis: float) -> str:
    if not np.isfinite(axis):
        return "n/a"
    if abs(axis - 90.0) <= 30.0:
        return "E-W rows (south-facing)"
    if axis < 30.0 or axis > 150.0:
        return "N-S rows (east-west park)"
    return "oblique"


def main():
    osm = json.load(open(os.path.join(paths.INPUT_DIR, "osm", "parks_osm.json")))
    pts = {p["pv"]: p for p in json.load(open(os.path.join(paths.INPUT_DIR, "points.json")))}
    sel = pd.read_csv(paths.SELECTION).dropna(subset=["park_id"]).set_index("park_id")
    out = os.path.join(paths.DATA_DIR, "osm_row_geometry.csv")
    if os.path.exists(out) and not os.path.exists(out.replace(".csv", "_before_parkfix.csv")):
        shutil.copy(out, out.replace(".csv", "_before_parkfix.csv"))
    rows = []
    for pid, resp in osm.items():
        els = resp.get("elements", [])
        p = pts.get(pid); r = sel.loc[pid] if pid in sel.index else None
        lat, lon = (p["lat"], p["lon"]) if p else (float(r.lat), float(r.lon))
        kwp = float(r.dc_kwp) if r is not None and "dc_kwp" in r.index and np.isfinite(r.dc_kwp) else 1000.0
        res = osm_rows.row_axis_park(els, lat, lon, kwp)
        tags = sorted({f"{k}={v}" for e in els if (e.get("tags") or {}).get("power") == "generator" for k, v in (e.get("tags") or {}).items() if k.startswith("generator:")})
        rows.append({"park_id": pid, "n_generator_ways": res["n_generators"], "n_elongated": res["n_elongated"], "row_axis_deg": res["row_axis_deg"],
                     "n_plant": sum(1 for e in els if (e.get("tags") or {}).get("power") == "plant"), "tags": ";".join(tags[:3]),
                     "register": r.orientation_class if r is not None and "orientation_class" in r.index else "", "row_axis_class": row_axis_class(res["row_axis_deg"]),
                     "axis_spread_deg": res["axis_spread_deg"], "prior_ok": res["prior_ok"], "selection": res["selection"], "n_generators_response": res["n_generators_response"]})
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    print(f"{out}: {len(df)} parks, with axis {df.row_axis_deg.notna().sum()}, prior_ok {int(df.prior_ok.sum())}, by selection {df.selection.value_counts().to_dict()}")


if __name__ == "__main__":
    main()
