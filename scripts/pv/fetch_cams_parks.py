#!/usr/bin/env python
"""Build the CAMS point list for the selected PV parks and fetch the series from ADS, one multi-year request
per point, resumable, capped per run to respect the daily quota.

    synthre/bin/python scripts/pv/fetch_cams_parks.py --selection ~/Work/FL_Contribution/data/mastr/pv_park_selection_v1.csv \
        --out ${DATA_ROOT}/synthetic/solar/parks_pv_v1/inputs --max-requests 450 [--threads 3]
"""
import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pv import cams_points as CP  # noqa: E402

ROOT = os.environ.get("DATA_ROOT", "/mnt/lambda1/nvme1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", default=os.path.expanduser("~/Work/FL_Contribution/data/mastr/pv_park_selection_v1.csv"))
    ap.add_argument("--out", default=f"{ROOT}/synthetic/solar/parks_pv_v1/inputs")
    ap.add_argument("--start", default="2023-07-01")
    ap.add_argument("--end", default="2026-09-30")
    ap.add_argument("--tag", default="2023_2026")
    ap.add_argument("--max-requests", type=int, default=450)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--points-only", action="store_true")
    a = ap.parse_args()
    cams_dir, ring_dir = os.path.join(a.out, "cams"), os.path.join(a.out, "cams_ring")
    for d in (cams_dir, ring_dir):
        os.makedirs(d, exist_ok=True)
    sel = pd.read_csv(a.selection).dropna(subset=["park_id"])
    pts_path = os.path.join(a.out, "cams_points.csv")
    if os.path.exists(pts_path):
        pts = pd.read_csv(pts_path)
        new = sel[~sel.park_id.isin(pts.park_id)]                 # parks added to a later selection (v2): append their points
        if len(new):
            pts = pd.concat([pts, CP.point_table(new[["park_id", "lat", "lon"]])], ignore_index=True)
            pts.to_csv(pts_path, index=False)
            print("added points of", len(new), "parks", flush=True)
    else:
        pts = CP.point_table(sel[["park_id", "lat", "lon"]])
        pts.to_csv(pts_path, index=False)
    pts["target"] = [CP.target_path(cams_dir, ring_dir, p, a.tag) for p in pts.itertuples()]
    pts["done"] = [os.path.exists(t) and os.path.getsize(t) > 1000 for t in pts.target]
    todo = pts[~pts.done]
    print(f"{len(pts)} points, {pts.done.sum()} done, {len(todo)} to fetch; this run max {a.max_requests}", flush=True)
    if a.points_only or todo.empty:
        return
    batch = todo.head(a.max_requests)
    with ThreadPoolExecutor(a.threads) as ex:
        res = list(ex.map(lambda p: CP.fetch_point(p, a.start, a.end, p.target), batch.itertuples()))
    print("result:", pd.Series(res).value_counts().to_dict(), flush=True)
    if "quota" in res:
        print("ADS daily quota reached - rerun tomorrow", flush=True)


if __name__ == "__main__":
    main()
