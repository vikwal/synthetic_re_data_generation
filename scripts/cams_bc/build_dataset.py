#!/usr/bin/env python
"""Build the 30-min feature frames for all stations (daytime rows) and the station splits.

    python scripts/cams_bc/build_dataset.py --workers 24
"""
import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
from cams_bc import data, splits  # noqa: E402
from cams_bc.config import DATASET, SPLITS  # noqa: E402


def one(sid):
    t0 = time.time()
    df = data.build_station(sid)
    os.makedirs(DATASET, exist_ok=True)
    df.to_parquet(data.dataset_path(sid))
    return sid, len(df), round(time.time() - t0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    ids = args.only or data.station_ids()
    if not os.path.exists(SPLITS):
        os.makedirs(os.path.dirname(SPLITS), exist_ok=True)
        sp = splits.make_splits(data.station_ids())
        sp.to_csv(SPLITS, index=False)
        print(sp.groupby(["role", "fold"]).size().to_string())
    with ProcessPoolExecutor(args.workers) as ex:
        for sid, n, s in ex.map(one, ids):
            print(f"{sid} rows {n} {s}s", flush=True)


if __name__ == "__main__":
    main()
