#!/usr/bin/env python3
"""parks_v1 driver: 90 real MaStR wind parks, chain M4-noQM, ERA5 from Postgres.

Stages (run in order; each re-runnable, later stages read earlier outputs):
  layouts   per-turbine layout (OSM-corrected coordinates, ERA5 cell, group id)
            + group table + spec overrides           -> data/parks_v1/
  configs   one round-2 config per park              -> configs/round2/_generated/parks_v1/
  terrain   DEM/CORINE metrics at the park centroids + hybrid-gate info table
  generate  free-stream synthesis per park (parallel) -> RUN_DIR/free/
  wakes     PyWake NOJ w(t) from each park's own free-stream frame -> RUN_DIR/wakes/
  assemble  release parquet + manifest per park       -> RELEASE_DIR (l1, nasuser)
  tables    parks.csv, wind_groups.csv, clients.csv, sites.csv, park_layouts.csv

Usage: run_parks_v1.py <stage> [--parks SEL... ] [--workers 12]
Needs WEATHER_DB_URL and DATA_ROOT (from ~/.bashrc).
"""

import argparse
import json
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
os.chdir(REPO)  # generate_wind resolves power_curves/ relative to the repo

from parks import assemble, config, era5_db, gate, layout, library, metadata, paths, synth, wakes  # noqa: E402

FREE_DIR = os.path.join(paths.RUN_DIR, "free")
WAKE_DIR = os.path.join(paths.RUN_DIR, "wakes")


def load_inputs():
    sel = pd.read_csv(paths.SELECTION_CSV)
    tur = pd.read_csv(paths.TURBINES_CSV, low_memory=False)
    osm = pd.read_csv(paths.OSM_CORRECTIONS_CSV)
    return sel, tur, osm


def selected(sel: pd.DataFrame, parks) -> pd.DataFrame:
    return sel[sel.lokation.isin(parks)] if parks else sel


# ---------------------------------------------------------------- layouts

def stage_layouts(_args) -> None:
    sel, tur, osm = load_inputs()
    pc = library.load_power_curves()
    lay = layout.build_turbine_table(sel, tur, osm, library.curve_max_kw(set(tur.lib_name.dropna()), pc))
    conn = era5_db.connect()
    points = era5_db.load_grid_points(conn)
    lay = layout.assign_groups(layout.assign_cells(lay, points))
    groups = layout.group_table(lay)
    os.makedirs(paths.DATA_DIR, exist_ok=True)
    points.to_csv(os.path.join(paths.DATA_DIR, "era5_grid_points.csv"), index=False)
    lay.to_csv(paths.LAYOUT_CSV, index=False)
    groups.to_csv(paths.GROUPS_CSV, index=False)
    ov = library.build_spec_overrides(lay.lib_name.unique(), pc)
    ov.to_csv(paths.SPEC_OVERRIDES_CSV, index=False)

    spacing = lay.groupby("park_id").apply(layout.min_spacing_d, include_groups=False)
    tight = spacing[spacing < 1.0]
    print(f"layout: {len(lay)} turbines, {lay.park_id.nunique()} parks, {len(groups)} groups, "
          f"{lay.era5_cell_id.nunique()} ERA5 cells, {(groups.groupby('park_id').era5_cell_id.nunique() > 1).sum()} "
          f"parks over >1 cell, max turbine-cell distance {lay.era5_cell_dist_km.max():.1f} km")
    print(f"coordinates: {lay.osm_status.value_counts().to_dict()}")
    print(f"spec overrides:\n{ov.to_string(index=False)}")
    print(f"min spacing [D]: median {spacing.median():.2f}, parks < 1 D: {tight.round(2).to_dict()}")
    if len(tight):
        raise SystemExit("pairs closer than one rotor diameter remain - resolve before wakes")


# ---------------------------------------------------------------- configs

def stage_configs(_args) -> None:
    sel, _, _ = load_inputs()
    lay, groups = pd.read_csv(paths.LAYOUT_CSV), pd.read_csv(paths.GROUPS_CSV)
    written = config.write_configs(sel, groups, lay)
    print(f"{len(written)} configs -> {paths.CONFIG_DIR}")


# ---------------------------------------------------------------- terrain

def stage_terrain(_args) -> None:
    import importlib.util
    from round2 import topo
    lay = pd.read_csv(paths.LAYOUT_CSV)
    cen = lay.groupby("park_id")[["latitude", "longitude"]].mean().reset_index()
    spec = importlib.util.spec_from_file_location(
        "wp0_topo", os.path.join(REPO, "scripts", "round2", "wp0_topo_features.py"))
    wp0 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wp0)
    fine, coarse = wp0.load(wp0.MOSAIC_FINE), wp0.load(wp0.MOSAIC_COARSE)
    corine = topo.CorineSampler(wp0.CORINE_TIF)
    rows = []
    for r in cen.itertuples(index=False):
        m = topo.location_metrics(fine, coarse, r.latitude, r.longitude)
        m["z0"], m["clc_class"] = corine.z0(r.latitude, r.longitude), corine.clc_class(r.latitude, r.longitude)
        rows.append({"location_id": r.park_id, "kind": "park", "latitude": r.latitude,
                     "longitude": r.longitude, **m})
    topo_df = pd.DataFrame(rows)
    topo_df.to_csv(paths.TOPO_CSV, index=False)
    print(topo_df[["elevation", "slope", "tpi5", "tpi75", "elev_std", "z0"]].describe().round(3).to_string())
    g = gate.hybrid_gate(cen)
    g.to_csv(os.path.join(paths.DATA_DIR, "gate_hybrid_info.csv"), index=False)
    print(g.branch.value_counts().to_dict(), g.reason.value_counts().to_dict())


# ---------------------------------------------------------------- generate

def _generate_one(lokation: str) -> dict:
    t0 = time.time()
    cfg = config.load_yaml(paths.config_path(lokation))
    r2 = cfg["round2"]
    conn = era5_db.connect()
    try:
        frames = synth.fetch_frames(conn, cfg["params"]["era5_cells"], r2["output_start"], r2["output_end"])
    finally:
        conn.close()
    free = synth.run_park(cfg, frames, library.load_overrides())
    os.makedirs(FREE_DIR, exist_ok=True)
    free.to_parquet(os.path.join(FREE_DIR, f"free_{lokation}.parquet"))
    meta = {"park_id": lokation, "table": era5_db.TABLE, "window": [r2["output_start"], r2["output_end"]],
            "cells": {str(c): {"sha256": era5_db.frame_sha256(f), "n_hours": len(f)} for c, f in frames.items()},
            "primary_cell": cfg["park"]["primary_era5_cell"], "free_sha256": assemble.frame_sha256(free),
            "seconds": round(time.time() - t0, 1)}
    with open(os.path.join(FREE_DIR, f"era5_{lokation}.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def stage_generate(args) -> None:
    sel, _, _ = load_inputs()
    ids = selected(sel, args.parks).lokation.tolist()
    failed = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_generate_one, lk): lk for lk in ids}
        for i, fu in enumerate(as_completed(futs), 1):
            lk = futs[fu]
            try:
                m = fu.result()
                print(f"[{i}/{len(ids)}] {lk}: {len(m['cells'])} cell(s), {m['seconds']} s", flush=True)
            except Exception as e:  # noqa: BLE001 - report and continue with the other parks
                failed.append((lk, repr(e)))
                print(f"[{i}/{len(ids)}] {lk}: FAILED {e!r}", flush=True)
    if failed:
        raise SystemExit(f"{len(failed)} parks failed: {failed}")


# ---------------------------------------------------------------- wakes

def _wake_one(lokation: str) -> dict:
    cfg = config.load_yaml(paths.config_path(lokation))
    free = pd.read_parquet(os.path.join(FREE_DIR, f"free_{lokation}.parquet"))
    lay = pd.read_csv(paths.LAYOUT_CSV)
    lay = lay[lay.park_id == lokation]
    pc, ct = library.load_power_curves(), library.load_ct_curves()
    groups = synth.groups_of(cfg)
    k = float(cfg["round2"]["wake"]["k"])
    w, info = wakes.wake_factor(lay, wakes.inflow(free, groups), k, pc, ct)
    os.makedirs(WAKE_DIR, exist_ok=True)
    w.to_frame().to_parquet(os.path.join(WAKE_DIR, f"w_{lokation}_k{k:.3f}.parquet"))
    info.assign(park_id=lokation).to_csv(os.path.join(WAKE_DIR, f"types_{lokation}.csv"), index=False)
    return {"park_id": lokation, "n_turbines": len(lay), "mean_w": float(w.mean()),
            "wake_efficiency": wakes.park_efficiency(free["power_park_free"], w),
            "p05_w": float(w.quantile(0.05)), "min_w": float(w.min()),
            "generic_ct_types": int(info.generic_ct.sum()), "n_wake_types": len(info),
            "free_sha256": assemble.frame_sha256(free)}


def stage_wakes(args) -> None:
    sel, _, _ = load_inputs()
    ids = selected(sel, args.parks).lokation.tolist()
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_wake_one, lk): lk for lk in ids}
        for i, fu in enumerate(as_completed(futs), 1):
            r = fu.result()
            rows.append(r)
            print(f"[{i}/{len(ids)}] {r['park_id']}: n={r['n_turbines']} efficiency "
                  f"{r['wake_efficiency']:.3f} (mean w {r['mean_w']:.3f})", flush=True)
    summ = pd.DataFrame(rows)
    path = os.path.join(paths.DATA_DIR, "wake_summary.csv")
    if args.parks and os.path.exists(path):
        old = pd.read_csv(path)
        summ = pd.concat([old[~old.park_id.isin(summ.park_id)], summ], ignore_index=True)
    summ.sort_values("park_id").to_csv(path, index=False)


# ---------------------------------------------------------------- assemble

def _assemble_one(lokation: str) -> dict:
    cfg = config.load_yaml(paths.config_path(lokation))
    k = float(cfg["round2"]["wake"]["k"])
    free = pd.read_parquet(os.path.join(FREE_DIR, f"free_{lokation}.parquet"))
    w = pd.read_parquet(os.path.join(WAKE_DIR, f"w_{lokation}_k{k:.3f}.parquet"))["w"]
    with open(os.path.join(FREE_DIR, f"era5_{lokation}.json")) as f:
        era5_meta = json.load(f)
    summ = pd.read_csv(os.path.join(paths.DATA_DIR, "wake_summary.csv")).set_index("park_id").loc[lokation]
    free_sha = assemble.frame_sha256(free)
    if summ["free_sha256"] != free_sha or era5_meta["free_sha256"] != free_sha:
        raise ValueError(f"{lokation}: wake factor was computed on a different free-stream frame "
                         "(stale wakes) - rerun the wakes stage")
    rel = assemble.release_frame(free, w, cfg["params"]["group_ids"])
    wake_meta = {"wake_efficiency": float(summ["wake_efficiency"]), "mean_w": float(summ["mean_w"]),
                 "generic_ct_types": int(summ["generic_ct_types"])}
    era5_meta.pop("free_sha256", None)
    era5_meta.pop("seconds", None)
    man = assemble.manifest(cfg, era5_meta, wake_meta, rel, free_sha, library.load_overrides())
    assemble.write_release(lokation, rel, man)
    cap = cfg["park"]["capacity_kw"] * 1000.0
    return {"park_id": lokation, "cf_free": float(rel.power_park_free.mean() / cap),
            "cf": float(rel.power_park.mean() / cap)}


def stage_assemble(args) -> None:
    sel, _, _ = load_inputs()
    ids = selected(sel, args.parks).lokation.tolist()
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for r in ex.map(_assemble_one, ids):
            rows.append(r)
    cf = pd.DataFrame(rows)
    path = os.path.join(paths.DATA_DIR, "cf_summary.csv")
    if args.parks and os.path.exists(path):
        old = pd.read_csv(path)
        cf = pd.concat([old[~old.park_id.isin(cf.park_id)], cf], ignore_index=True)
    cf.sort_values("park_id").to_csv(path, index=False)
    fix_permissions(paths.RELEASE_DIR)
    print(cf.describe().round(3).to_string())


def fix_permissions(root: str) -> None:
    """Group nasuser (setgid dir) needs rw for the l1 account (other uid)."""
    for d, _, files in os.walk(root):
        os.chmod(d, 0o2775)
        for f in files:
            os.chmod(os.path.join(d, f), 0o664)


# ---------------------------------------------------------------- tables

def stage_tables(_args) -> None:
    sel, _, _ = load_inputs()
    lay, groups = pd.read_csv(paths.LAYOUT_CSV), pd.read_csv(paths.GROUPS_CSV)
    gate_info = pd.read_csv(os.path.join(paths.DATA_DIR, "gate_hybrid_info.csv"))
    run = pd.read_csv(os.path.join(paths.DATA_DIR, "wake_summary.csv"))[["park_id", "wake_efficiency"]] \
        .merge(pd.read_csv(os.path.join(paths.DATA_DIR, "cf_summary.csv")), on="park_id")
    parks = metadata.parks_table(sel, lay, groups, gate_info, run)
    ct_info = pd.concat([pd.read_csv(os.path.join(WAKE_DIR, f"types_{lk}.csv")) for lk in sel.lokation])
    aging = []
    for lk in sel.lokation:
        rel = pd.read_parquet(os.path.join(paths.RELEASE_DIR, f"synth_{lk}.parquet"))
        for g in groups[groups.park_id == lk].group_id:
            a = rel[f"aging_factor_{g}"]
            aging.append({"park_id": lk, "group_id": g, "aging_factor_start": float(a.iloc[0]),
                          "aging_factor_end": float(a.iloc[-1]), "aging_factor_mean": float(a.mean())})
    wg = metadata.wind_groups_table(groups, ct_info, pd.DataFrame(aging))
    clients = metadata.clients_table(parks)
    topo_df = pd.read_csv(paths.TOPO_CSV)
    cell_dist = groups.groupby("park_id").apply(
        lambda g: float(np.average(g.era5_cell_dist_km, weights=g.n_turbines * g.rated_kw)), include_groups=False)
    sites = metadata.sites_table(parks, topo_df, cell_dist)
    out = paths.RELEASE_DIR
    os.makedirs(out, exist_ok=True)
    parks.to_csv(os.path.join(out, "parks.csv"), index=False)
    wg.to_csv(os.path.join(out, "wind_groups.csv"), index=False)
    clients.to_csv(os.path.join(out, "clients.csv"), index=False)
    sites.to_csv(os.path.join(out, "sites.csv"), index=False)
    lay.to_csv(os.path.join(out, "park_layouts.csv"), index=False)
    library.load_overrides().to_csv(os.path.join(out, "turbine_specs_overrides.csv"), index=False)
    shutil.copy(os.path.join(REPO, "parks", "README_release.md"), os.path.join(out, "README.md"))
    fix_permissions(out)
    print(clients.to_string(index=False))
    print(f"parks {len(parks)}, groups {len(wg)}, capacity {parks.capacity_kw.sum() / 1e3:.0f} MW")


STAGES = {"layouts": stage_layouts, "configs": stage_configs, "terrain": stage_terrain,
          "generate": stage_generate, "wakes": stage_wakes, "assemble": stage_assemble,
          "tables": stage_tables}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=list(STAGES) + ["all"])
    ap.add_argument("--parks", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    for name in (STAGES if args.stage == "all" else [args.stage]):
        t0 = time.time()
        print(f"== {name}", flush=True)
        STAGES[name](args)
        print(f"== {name} done in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
