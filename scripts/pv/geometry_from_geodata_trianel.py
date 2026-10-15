#!/usr/bin/env python
"""Measured module geometry for the metered Trianel parks from Bavarian open geodata (parks outside Bavaria are
reported as unavailable). Confidential inputs; outputs anonymised (pv1..pv7) under
${DATA_ROOT}/synthetic/solar/trianel_benchmark/inputs/geodata/ and reports/figs_pv_generator/geodata_geometry_trianel.csv."""
import json
import os
import sys

import numpy as np
import pandas as pd
import rasterio

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts", "pv"))
from pv import arrays as arr, geodata_bayern as GB  # noqa: E402
import geometry_from_geodata as GFG  # noqa: E402
import trianel_common as tc  # noqa: E402

B = f"{os.environ.get('DATA_ROOT', '/mnt/lambda1/nvme1')}/synthetic/solar/trianel_benchmark/inputs"
OUT = os.path.join(B, "geodata")
FIGS = os.path.expanduser("~/Work/FL_Contribution/reports/figs_pv_generator")


def main():
    m = tc.load_masterdata()
    parks = tc.parks_table(m)
    osm = json.load(open(os.path.join(B, "osm", "trianel_pv_osm.json")))
    os.makedirs(OUT, exist_ok=True)
    summ, arrays_rows = [], []
    for pv, p in parks.iterrows():
        tiles = GB.tiles_for(p["lat"], p["lon"], 400.0)
        try:
            GB.fetch("dgm1", *tiles[0], OUT)
        except Exception:  # noqa: BLE001
            summ.append({"park_id": pv, "status": "outside Bavaria (no tile)"})
            print(pv, "no Bavarian tile", flush=True)
            continue
        # expected module area: kWp / (typical kWp per ha of modules ~ 2 x ground density) - rough, only ranks candidates
        r = pd.Series({"park_id": pv, "lat": p["lat"], "lon": p["lon"], "area_ha": np.nan, "expected_module_ha": float(p["kw_master"]) / 1500.0, "year": int(p.get("year_hint", 0) or 0), "search_radius_m": 1500.0})
        try:
            s, ar = GFG.one(r, osm, OUT, 5.0)
        except Exception as e:  # noqa: BLE001
            s, ar = {"park_id": pv, "status": f"error: {str(e)[:120]}"}, None
        summ.append(s)
        if ar:
            arrays_rows += ar
        print(pv, s["status"], {k: (round(v, 1) if isinstance(v, float) else v) for k, v in s.items() if k in ("n_tables", "tilt_capw", "azimuth_mean_normal", "row_row_axis_deg", "row_pitch_m", "table_src", "terrain_following", "arrays")},
              "| master:", p["tilt_master"], p["az_master"], flush=True)
    pd.DataFrame(summ).to_csv(os.path.join(FIGS, "geodata_geometry_trianel.csv"), index=False)
    pd.DataFrame(arrays_rows).to_csv(os.path.join(FIGS, "geodata_arrays_trianel.csv"), index=False)


if __name__ == "__main__":
    main()
