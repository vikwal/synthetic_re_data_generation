#!/usr/bin/env python
"""Fetch OSM footprints and PVGIS horizons for the selected PV parks (resumable JSON caches).

    synthre/bin/python scripts/pv/fetch_site_inputs.py [--selection ...] [--out ${DATA_ROOT}/synthetic/solar/parks_pv_v1/inputs/osm]
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pv import site_inputs  # noqa: E402

ROOT = os.environ.get("DATA_ROOT", "/mnt/lambda1/nvme1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", default=os.path.expanduser("~/Work/FL_Contribution/data/mastr/pv_park_selection_v1.csv"))
    ap.add_argument("--out", default=f"{ROOT}/synthetic/solar/parks_pv_v1/inputs/osm")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    sel = pd.read_csv(a.selection).dropna(subset=["park_id"])
    site_inputs.fetch_all(sel[["park_id", "lat", "lon"]], os.path.join(a.out, "parks_osm.json"), os.path.join(a.out, "pvgis_horizon.json"))
    print("done", flush=True)


if __name__ == "__main__":
    main()
