#!/usr/bin/env python
"""Angle estimation (parks_pv.geometry_estimate) on its own, outside a release run: estimates the parks of a selection
that have no estimate of the method in the store yet and stores them (<geometry_dir>/<method>/<park_id>/). Stage
`geometry` of generate_pv.py reads the same store, so parks estimated here are not estimated again.

    synthre/bin/python scripts/pv/estimate_geometry.py --config configs/runs/parks_pv_v4.yaml [--parks ID,ID] [--state Bayern]
    synthre/bin/python scripts/pv/estimate_geometry.py --selection sel.csv --method v5 [--rules -az_known] [--force]
    synthre/bin/python scripts/pv/estimate_geometry.py --config ... --status          # what is in the store, nothing estimated
Rules: a method keeps the rule set it was created with (<geometry_dir>/<method>/method.json); a new method starts from
geometry_estimate.PRODUCTION_RULES with --rules / --extend applied. Other rules for an existing method are refused.
The selection rows need park_id, lat, lon, dc_kwp, year, area_ha, bundesland, landkreis, unit_ids, orientation_class,
tilt_class (MaStR selection, pv.selection)."""
import argparse
import os
import sys

import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
from parks_pv import geometry_estimate as GE, geometry_store as GSt, run_config  # noqa: E402


def selection(a) -> tuple:
    """(selection rows, method) from --config or --selection."""
    if a.config:
        rc = run_config.load(a.config)
        run_config.apply(rc)
        from parks_pv import stages
        return stages.load_selection(rc), a.method or rc["geometry"]["method"]
    if not a.method:
        raise SystemExit("--selection needs --method")
    return pd.read_csv(os.path.expanduser(a.selection), dtype={"park_id": str}).dropna(subset=["park_id"]), a.method


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--config", help="run config: selection (+ overrides) and geometry.method of the release")
    src.add_argument("--selection", help="selection CSV (MaStR rows)")
    ap.add_argument("--method", default=None, help="store version (default: geometry.method of the run config)")
    ap.add_argument("--parks", default=None, help="comma-separated park ids")
    ap.add_argument("--state", default=None, help="only parks of this Bundesland")
    ap.add_argument("--force", action="store_true", help="estimate again although the store has the park")
    ap.add_argument("--rules", default="", help="new method only: rule switches on top of PRODUCTION_RULES (name = on, -name = off)")
    ap.add_argument("--extend", action="store_true", help="variant c (probe): extend low-coverage OSM outlines")
    ap.add_argument("--status", action="store_true", help="only report the store state of the selection")
    a = ap.parse_args()
    sel, method = selection(a)
    if a.parks:
        sel = sel[sel.park_id.isin(a.parks.split(","))]
    if a.state:
        sel = sel[sel.bundesland == a.state]
    miss = GSt.missing(method, sel.park_id)
    print(f"{len(sel)} parks; store {GSt.root(method)}: {len(sel) - len(miss)} estimated, {len(miss)} missing")
    if a.status:
        return
    print("rules:", GE.method_rules(method, a.rules, a.extend))
    done = GE.estimate_parks(sel, method, force=a.force)
    if done:
        GE.summary(GSt.all_sources(method, done), f"{method}: {len(done)} parks estimated")


if __name__ == "__main__":
    main()
