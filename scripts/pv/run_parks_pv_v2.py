#!/usr/bin/env python
"""parks_pv_v2 (2026-10-22): the parks_pv_v1 release with the module geometry from open geodata (window method).

Same 90 parks, park parameters, drivers (corrected CAMS, ERA5, snow), horizons and chain as parks_pv_v1; only the
sub-arrays change: parks with an accepted geodata measurement get the measured arrays (tilt against the horizontal,
true azimuth, share = table area; east-west pairs as two arrays), all others keep the nominal mounts of v1, re-terrained
on the state DGM1 where it was fetched (register_v1+dgm1), else the v1 arrays unchanged (register_v1). The arrays come
from scripts/pv/fleet_geo_status.py (pv_arrays_geodata_v2.csv). Stage `sites` writes data/parks_pv_v2/; the other
stages are those of run_parks_pv_v1.py, run with PV_RELEASE_NAME=parks_pv_v2 (configs, generated chain and release in
their own directories, inputs and corrected CAMS shared with v1). v1 outputs are not touched.

    export PV_RELEASE_NAME=parks_pv_v2 PV_RELEASE_VARIANT=_30min
    synthre/bin/python scripts/pv/run_parks_pv_v2.py sites --geodata ~/Work/pv_sim2real/results/geodata_release_<date>/pv_arrays_geodata_v2.csv
    synthre/bin/python scripts/pv/run_parks_pv_v2.py configs
    synthre/bin/python scripts/pv/run_parks_pv_v2.py generate --step 30min --workers 8
    synthre/bin/python scripts/pv/run_parks_pv_v2.py assemble
    synthre/bin/python scripts/pv/run_parks_pv_v2.py tables
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parks_pv import geometry as G, paths  # noqa: E402
from pv import arrays as ARR  # noqa: E402
from pv.plausibility import orientation_class  # noqa: E402

V1_DATA = os.path.join(REPO, "data", "parks_pv_v1")


def stage_sites(args):
    assert paths.NAME == "parks_pv_v2", "set PV_RELEASE_NAME=parks_pv_v2"
    sel = pd.read_csv(paths.SELECTION, dtype={"park_id": str}).dropna(subset=["park_id"])
    geo = pd.read_csv(args.geodata, dtype={"park_id": str})
    a1 = pd.read_csv(os.path.join(V1_DATA, "pv_arrays.csv"), dtype={"park_id": str})
    s1 = pd.read_csv(os.path.join(V1_DATA, "sites.csv"), dtype={"site_id": str}).set_index("site_id")
    par = pd.read_csv(os.path.join(V1_DATA, "park_params.csv"), dtype={"park_id": str}).set_index("park_id")
    arrays_rows, site_rows, view_rows = [], [], []
    for r in sel.itertuples():
        g = geo[geo.park_id == r.park_id]
        p = par.loc[r.park_id]
        if not len(g):
            raise SystemExit(f"{r.park_id}: no rows in {args.geodata}")
        src = str(g.geometry_source.iloc[0])
        arrays = []
        for k, x in enumerate(g.itertuples()):
            measured = str(x.source).startswith("windows:")
            arrays.append(dict(name=str(x.array_id), tilt=float(x.tilt), azimuth=float(x.azimuth) % 360.0, share=float(x.share),
                               mount_tilt=float(x.tilt) if measured or pd.isna(x.mount_tilt) else float(x.mount_tilt),
                               mount_azimuth=float(x.azimuth) if measured or pd.isna(x.mount_azimuth) else float(x.mount_azimuth),
                               source=str(x.source), az_source=str(getattr(x, "az_source", "")), note=str(getattr(x, "note", "") if pd.notna(getattr(x, "note", "")) else "")))
        tot = sum(a["share"] for a in arrays)
        for a in arrays:
            a["share"] /= tot
            arrays_rows.append({"park_id": r.park_id, "array_id": a["name"], "tilt": round(a["tilt"], 2), "azimuth": round(a["azimuth"], 2),
                                "share": round(a["share"], 5), "capacity_kwp": round(a["share"] * p.kwp_dc, 1), "mount_tilt": round(a["mount_tilt"], 2),
                                "mount_azimuth": round(a["mount_azimuth"], 2), "source": a["source"], "az_source": a["az_source"], "note": a["note"],
                                "geometry_source": src, "module_era": p.era, "gamma_pdc": p.gamma_pdc, "dc_ac": p.dc_ac, "gcr": p.gcr,
                                "commissioning_year": p.year})
        tm, am = ARR.mean_angles(np.array([a["tilt"] for a in arrays]), np.array([a["azimuth"] for a in arrays]), np.array([a["share"] for a in arrays]))
        s = s1.loc[r.park_id].to_dict()
        measured = src.startswith("windows:")
        if measured:                                              # deviation from the register as measured (the v1 deviation was simulated)
            ew = any("Flanke" in a["note"] for a in arrays)
            cls = "Ost-West" if ew else orientation_class(arrays[int(np.argmax([a["share"] for a in arrays]))]["azimuth"])
            s["register_deviation"] = "" if cls == r.orientation_class else f"registered {r.orientation_class}, measured {cls} (geodata)"
        s.update(site_id=r.park_id, n_arrays=len(arrays), tilt_mean_true=round(float(tm), 2), azimuth_mean_true=round(float(am), 2),
                 tilt_capw_true=round(sum(a["tilt"] * a["share"] for a in arrays), 2), geometry_source=src)
        site_rows.append(s)
        view_rows.append(G.register_view(r, arrays))
    os.makedirs(paths.DATA_DIR, exist_ok=True)
    pd.DataFrame(arrays_rows).to_csv(os.path.join(paths.DATA_DIR, "pv_arrays.csv"), index=False)
    pd.DataFrame(site_rows).to_csv(os.path.join(paths.DATA_DIR, "sites.csv"), index=False)
    pd.DataFrame(view_rows).to_csv(os.path.join(paths.DATA_DIR, "pv_masterdata_view.csv"), index=False)
    par.reset_index().to_csv(os.path.join(paths.DATA_DIR, "park_params.csv"), index=False)
    s = pd.DataFrame(site_rows)
    print(f"{len(s)} parks -> {paths.DATA_DIR}; geometry {s.geometry_source.str.split(':').str[0].value_counts().to_dict()}; "
          f"arrays/park {s.n_arrays.mean():.2f}; measured register deviations {(s.register_deviation.str.contains('geodata', na=False)).sum()}")


def main():
    import run_parks_pv_v1 as V1
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["sites", "configs", "generate", "assemble", "tables"])
    ap.add_argument("--geodata", default=None, help="pv_arrays_geodata_v2.csv of scripts/pv/fleet_geo_status.py (stage sites)")
    ap.add_argument("--parks", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model-tag", default="xgb_v5")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--step", default="30min")
    ap.add_argument("--snow-model", default="depth_linear", choices=["depth_linear", "depth_slide"])
    a = ap.parse_args()
    assert paths.NAME == "parks_pv_v2", "set PV_RELEASE_NAME=parks_pv_v2"
    if a.parks:
        a.parks = a.parks.split(",")
    if a.stage == "sites":
        stage_sites(a)
    else:
        getattr(V1, f"stage_{a.stage}")(a)


if __name__ == "__main__":
    main()
