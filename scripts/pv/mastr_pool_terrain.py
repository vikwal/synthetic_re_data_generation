#!/usr/bin/env python
"""Point terrain metrics (GLO-30 mosaics, same as the wind sites table) for every open-ground MaStR park
of the candidate pool: elevation, slope, aspect, TPI 5/75 km, elevation std 5 km. Used to stratify the
park selection by hilliness; the true sub-array geometry is derived later from footprints.

    synthre/bin/python scripts/pv/mastr_pool_terrain.py [--min-kwp 750]
"""
import argparse
import os
import sys
import time

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts", "round2"))
from round2.topo import location_metrics  # noqa: E402
from wp0_topo_features import MOSAIC_COARSE, MOSAIC_FINE, load  # noqa: E402

ROOT = os.environ.get("DATA_ROOT", "/mnt/lambda1/nvme1")


def pool_filter(parks: pd.DataFrame, min_kwp: float) -> pd.DataFrame:
    """Open-ground parks in operation with coordinates and DC capacity >= min_kwp."""
    m = parks["mounting"].str.contains("Freifl", na=False) & (parks["status"] == "In Betrieb")
    m &= (parks["dc_kwp"] >= min_kwp) & parks["lat"].notna() & parks["lon"].notna()
    return parks[m]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parks", default=f"{ROOT}/synthetic/solar/mastr/solar_parks_20250625.parquet")
    ap.add_argument("--out", default=f"{ROOT}/synthetic/solar/mastr/pool_terrain_20250625.parquet")
    ap.add_argument("--min-kwp", type=float, default=750.0)
    a = ap.parse_args()
    pool = pool_filter(pd.read_parquet(a.parks), a.min_kwp)
    t0 = time.time()
    fine, coarse = load(MOSAIC_FINE), load(MOSAIC_COARSE)
    rows = [{"park_id": r.park_id, **location_metrics(fine, coarse, r.lat, r.lon)} for r in pool.itertuples()]
    out = pd.DataFrame(rows)
    out.to_parquet(a.out, index=False)
    print(f"{len(out)} parks in {time.time() - t0:.0f} s -> {a.out}")
    print(out[["elevation", "slope", "tpi5", "tpi75", "elev_std"]].describe().round(2).to_string())


if __name__ == "__main__":
    main()
