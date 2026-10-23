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


def stage_added(args):
    """Parks of selection v2 that are not in parks_pv_v1: the v1 tables built by run_parks_pv_v1.site_tables (same
    seed, OSM row axis of pv.osm_rows) -> data/parks_pv_v2/added_v1_{arrays,sites,params}.csv; their horizons are
    added to the shared horizons_used.json (new keys only). geometry_release_states.py reads added_v1_arrays.csv for
    the fallback arrays of these parks."""
    import json
    import run_parks_pv_v1 as V1
    from pv import osm_rows
    sel = pd.read_csv(paths.SELECTION, dtype={"park_id": str}).dropna(subset=["park_id"])
    v1 = pd.read_csv(os.path.join(V1_DATA, "park_params.csv"), dtype={"park_id": str})
    added = sel[~sel.park_id.isin(v1.park_id)]
    osm = json.load(open(V1.OSM_JSON)); pvg = json.load(open(V1.PVGIS_JSON)) if os.path.exists(V1.PVGIS_JSON) else {}
    terr = pd.read_parquet(os.path.join(paths.MASTR_DIR, "pool_terrain_20250625.parquet")).set_index("park_id")
    ax = []
    for r in added.itertuples():
        res = osm_rows.row_axis_park(osm.get(r.park_id, {}).get("elements", []), r.lat, r.lon, r.dc_kwp)
        ax.append(dict(park_id=r.park_id, row_axis_deg=res["row_axis_deg"], n_elongated=res["n_elongated"]))
    row_axis = pd.DataFrame(ax).set_index("park_id")
    hz_all = json.load(open(V1.HORIZON_OUT))
    arrays, sites, pars = [], [], []
    for r in added.itertuples():
        a_rows, site, view, p, hz = V1.site_tables(r, osm, pvg, terr, row_axis, args.seed)
        arrays += a_rows; sites.append(site); pars.append(p)
        hz_all.setdefault(r.park_id, hz)
        print(r.park_id, "added:", site["footprint_src"], site["horizon_src"], f"{site['n_arrays']} v1 arrays, row axis {row_axis.loc[r.park_id].to_dict()}", flush=True)
    json.dump(hz_all, open(V1.HORIZON_OUT, "w"))
    os.makedirs(paths.DATA_DIR, exist_ok=True)
    pd.DataFrame(arrays).to_csv(os.path.join(paths.DATA_DIR, "added_v1_arrays.csv"), index=False)
    pd.DataFrame(sites).to_csv(os.path.join(paths.DATA_DIR, "added_v1_sites.csv"), index=False)
    pd.DataFrame(pars).to_csv(os.path.join(paths.DATA_DIR, "added_v1_params.csv"), index=False)


def stage_sites(args):
    assert paths.NAME == "parks_pv_v2", "set PV_RELEASE_NAME=parks_pv_v2"
    sel = pd.read_csv(paths.SELECTION, dtype={"park_id": str}).dropna(subset=["park_id"])
    geo = pd.read_csv(args.geodata, dtype={"park_id": str})
    a1 = pd.read_csv(os.path.join(V1_DATA, "pv_arrays.csv"), dtype={"park_id": str})
    s1 = pd.read_csv(os.path.join(V1_DATA, "sites.csv"), dtype={"site_id": str}).set_index("site_id")
    par = pd.read_csv(os.path.join(V1_DATA, "park_params.csv"), dtype={"park_id": str}).set_index("park_id")
    added = sel[~sel.park_id.isin(par.index)]
    if len(added):                                                # parks of selection v2 that replace v1 parks (stage `added`)
        s1 = pd.concat([s1, pd.read_csv(os.path.join(paths.DATA_DIR, "added_v1_sites.csv"), dtype={"site_id": str}).set_index("site_id")])
        par = pd.concat([par, pd.read_csv(os.path.join(paths.DATA_DIR, "added_v1_params.csv"), dtype={"park_id": str}).set_index("park_id")])
        missing = set(added.park_id) - set(par.index)
        assert not missing, f"run stage added first: {missing}"
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


def write_readme():
    """Release README of v2 from the template; the counts per geometry source come from the written pv_arrays.csv."""
    src = pd.read_csv(os.path.join(paths.RELEASE_DIR, "pv_arrays.csv"), dtype={"park_id": str}).groupby("park_id").geometry_source.first()
    kind = src.str.split(":").str[0]
    txt = open(os.path.join(REPO, "parks_pv", "README_release_v2.md")).read()
    txt = txt.replace("{date}", f"{pd.Timestamp(os.path.getmtime(os.path.join(paths.RELEASE_DIR, 'pv_arrays.csv')), unit='s'):%Y-%m-%d}").replace("{n_windows}", str((kind == "windows").sum()))
    txt = txt.replace("{n_dgm1}", str((kind == "register_v1+dgm1").sum())).replace("{n_v1}", str(kind.str.startswith("register_v1").sum() - (kind == "register_v1+dgm1").sum()))
    open(os.path.join(paths.RELEASE_DIR, "README.md"), "w").write(txt)


def main():
    import run_parks_pv_v1 as V1
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["added", "sites", "configs", "generate", "assemble", "tables"])
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
    if a.stage == "added":
        stage_added(a)
    elif a.stage == "sites":
        stage_sites(a)
    else:
        getattr(V1, f"stage_{a.stage}")(a)
        if a.stage == "tables":                                   # the release README of v2 (v1 text below the v2 header)
            write_readme()


if __name__ == "__main__":
    main()
