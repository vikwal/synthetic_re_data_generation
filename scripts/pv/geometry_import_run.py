#!/usr/bin/env python
"""One-off migration of a geometry run of the former per-state layout into the angle estimation store
(parks_pv.geometry_store): data/parks_pv_v1/geodata_release_<state><suffix>.csv (rows per source),
results/pv/geodata_release<suffix>/<park>/ (window outputs) -> <geometry_dir>/<method>/<park>/.
Measured arrays are taken as text from --arrays (the generator arrays a release used, so the release stays
bit-identical), else from the per-state *_arrays.csv. Existing park directories are left alone.

    synthre/bin/python scripts/pv/geometry_import_run.py --suffix _v4 --method v4 \
        --arrays ~/Work/pv_sim2real/results/geodata_release_2026-11-07/pv_arrays_geodata_v2.csv"""
import argparse
import glob
import json
import os
import shutil
import sys

import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
import settings  # noqa: E402
from parks_pv import geometry_store as GSt  # noqa: E402

WORK_ROOT = settings.get("work_root")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--arrays", default=None, help="generator arrays of a release (preferred source of the measured arrays)")
    a = ap.parse_args()
    data_dir = os.path.join(WORK_ROOT, "data", "parks_pv_v1")
    res_dir = os.path.join(WORK_ROOT, "results", "pv", f"geodata_release{a.suffix}")
    rd = lambda f: pd.read_csv(f, dtype=str, keep_default_na=False)
    files = [f for f in sorted(glob.glob(os.path.join(data_dir, f"geodata_release_*{a.suffix}.csv"))) if not f.endswith("_arrays.csv")]
    rel = rd(os.path.expanduser(a.arrays)) if a.arrays else None
    n = dict(imported=0, skipped=0, measured=0)
    for f in files:
        src, arr = rd(f), rd(f.replace(".csv", "_arrays.csv"))
        for pid, s in src.groupby("pv", sort=False):
            if GSt.has(a.method, pid):
                n["skipped"] += 1
                continue
            d = GSt.park_dir(a.method, pid)
            if os.path.isdir(os.path.join(res_dir, pid)):
                shutil.copytree(os.path.join(res_dir, pid), d, dirs_exist_ok=True)
            os.makedirs(d, exist_ok=True)
            s.to_csv(os.path.join(d, "sources.csv"), index=False)
            m = rel[rel.park_id == pid] if rel is not None and pid in set(rel.park_id) else arr[arr.park_id == pid]
            m = m[m.source.str.startswith("windows:")]
            if len(m):
                n["measured"] += 1
            json.dump(dict(park_id=pid, method=a.method, imported_from=dict(run=a.suffix, state_table=os.path.basename(f), arrays=a.arrays if rel is not None and pid in set(rel.park_id) else os.path.basename(f.replace(".csv", "_arrays.csv"))),
                           rules=s.rules.iloc[0] if "rules" in s else ""), open(os.path.join(d, "meta.json"), "w"), indent=1)
            m.reindex(columns=GSt.ARRAY_COLS).fillna("").to_csv(os.path.join(d, "arrays.csv"), index=False)
            n["imported"] += 1
    if n["imported"] and GSt.method_spec(a.method) is None:      # rule set of the run = rule set of the method
        rules = pd.concat([rd(f) for f in files]).rules.replace("", pd.NA).dropna().unique()
        if len(rules) != 1:
            raise SystemExit(f"run {a.suffix} mixes rule sets: {list(rules)}")
        GSt.write_method_spec(a.method, rules[0], False)
    print(n, "->", GSt.root(a.method))


if __name__ == "__main__":
    main()
