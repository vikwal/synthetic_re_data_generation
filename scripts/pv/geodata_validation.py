#!/usr/bin/env python
"""Validate the geodata geometry batch against the register: agreement of tilt class and orientation class,
overall and for parks that pass the quality gates; list the deviators and render their four-panel figures.

    synthre/bin/python scripts/pv/geodata_validation.py --state Bayern
"""
import argparse
import glob
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import rasterio
from shapely.geometry import box

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from parks_pv import paths  # noqa: E402
from parks_pv.geometry import AZ_CENTRE, orientation_class, tilt_class  # noqa: E402


def pick(gdir, product, bounds):
    best = None
    for f in glob.glob(f"{gdir}/{product}*.tif"):
        with rasterio.open(f) as d:
            if box(*d.bounds).contains(box(*bounds)) and (best is None or os.path.getsize(f) < os.path.getsize(best)):
                best = f
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default="Bayern")
    ap.add_argument("--figs", default=os.path.expanduser("~/Work/FL_Contribution/reports/figs_pv_park_synthesis_v1/geodata"))
    ap.add_argument("--render", type=int, default=6, help="number of deviator figures to render")
    a = ap.parse_args()
    g = pd.read_csv(os.path.join(paths.DATA_DIR, f"geodata_geometry_{a.state.lower()}.csv"))
    sel = pd.read_csv(paths.SELECTION).set_index("park_id")
    g = g.join(sel[["orientation_class", "tilt_class", "year", "dc_kwp"]], on="park_id")
    ok = g[g.status == "ok"].copy()
    # measured classes from the profile (preferred) and the planes
    ok["faces"] = ok.profile_faces.fillna("")
    ok["ew_measured"] = ok.profile_type == "two_faces"
    ok["tilt_measured"] = ok.profile_tilt.where(ok.profile_tilt.notna(), ok.tilt_capw)
    ok["az_measured"] = ok.faces.str.extract(r"/(\d+)x")[0].astype(float)
    ok["tilt_cls_measured"] = ok.tilt_measured.map(tilt_class)
    ok["or_cls_measured"] = np.where(ok.ew_measured, "Ost-West", ok.az_measured.map(lambda a: orientation_class(a) if np.isfinite(a) else None))
    ok["tilt_agree"] = ok.tilt_cls_measured == ok.tilt_class
    ok["or_agree"] = (ok.or_cls_measured == ok.orientation_class) | ((ok.orientation_class != "Ost-West") & ~ok.ew_measured
                     & ok.az_measured.notna() & (ok.orientation_class.map(AZ_CENTRE).notna())
                     & ((((ok.az_measured - ok.orientation_class.map(AZ_CENTRE).fillna(0)) + 180) % 360 - 180).abs() <= 45))
    acc = ok[ok.accept.astype(str).eq("True")]
    print(f"{a.state}: {len(g)} parks, status {g.status.value_counts().to_dict()}")
    print("gates passed:", ok.gates_passed.value_counts().sort_index().to_dict(), "| accepted:", len(acc))
    for name, d in (("all ok", ok), ("accepted", acc)):
        if len(d):
            print(f"  {name:9s} n={len(d):2d}  tilt class agree {d.tilt_agree.mean():.2f}  orientation agree (<=45 deg or EW) {d.or_agree.mean():.2f}  "
                  f"median |tilt - class centre| n/a  EW register {int((d.orientation_class == 'Ost-West').sum())} measured {int(d.ew_measured.sum())}")
    cols = ["park_id", "year", "dom_date", "orientation_class", "tilt_class", "or_cls_measured", "tilt_measured", "faces", "gates_passed", "accept",
            "gate_dom_after_build", "gate_stripes", "gate_area", "gate_flank_width", "measured_module_ha", "expected_module_ha", "row_pitch_dom"]
    print(ok[cols].round(1).to_string(index=False))
    dev = acc[~(acc.tilt_agree & acc.or_agree)]
    print("\ndeviators among accepted:", dev.park_id.tolist())
    ok.to_csv(os.path.join(paths.DATA_DIR, f"geodata_validation_{a.state.lower()}.csv"), index=False)
    os.makedirs(a.figs, exist_ok=True)
    gdir = os.path.join(paths.INPUT_DIR, "geodata", a.state.lower())
    for pid in dev.park_id.tolist()[: a.render]:
        try:
            with rasterio.open(f"{gdir}/{pid}_rasters.tif") as r:
                b = r.bounds
            files = {p: pick(gdir, p, b) for p in ("dop20", "dom20", "dgm1")}
            if all(files.values()):
                subprocess.run(["synthre/bin/python", os.path.join(REPO, "scripts", "pv", "plot_geodata_park.py"), "--park", pid, "--geodata-dir", gdir,
                                "--dop", files["dop20"], "--dom", files["dom20"], "--dgm", files["dgm1"], "--out", os.path.join(a.figs, f"{pid}.png")],
                               check=False, capture_output=True, cwd=REPO)
                print("rendered", pid)
        except Exception as e:  # noqa: BLE001
            print(pid, "render failed", str(e)[:80])


if __name__ == "__main__":
    main()
