"""Inputs of the selected parks on demand (2026-11-03, Viktor: new sites should get their inputs without hand work).

What a park needs before stage `generate`, and the step that makes it (each step only for the parks that lack it,
all steps resumable, nothing existing is rewritten):
    site     OSM answer (parks_osm.json) and PVGIS horizon (pvgis_horizon.json)   scripts/pv/fetch_site_inputs.py
    cams     CAMS centre + 8 ring points from ADS (daily quota, waits)            scripts/pv/fetch_cams_parks.py
    correct  corrected CAMS of the model tag (30 min and hourly)                  scripts/cams_bc/apply_parks.py (ERA5 venv, GPU)
    icon     ICON-D2 SL 09 UTC grid points (snow driver of depth_slide)           NWP-points/points/run_pv_add.sh (NAS, hours)
ERA5 (WeatherDB) and the ERA5 snow GRIBs cover Germany and need no step; the snow point cache is filled by `generate`.
Angle estimation: stage geometry (parks_pv.geometry_estimate, store parks_pv.geometry_store)."""
import json
import os
import subprocess
import sys
import time

import pandas as pd

import settings
from parks_pv import paths

REPO = paths.REPO
PY = sys.executable
TAG = "2023_2026"                                                  # CAMS request tag of fetch_cams_parks.py
NWP_POINTS = settings.get("nwp_points_dir")
STEPS = ("site", "cams", "correct", "icon")


def _json_keys(path: str) -> set:
    return set(json.load(open(path))) if os.path.exists(path) else set()


def inventory(sel: pd.DataFrame, model_tag: str) -> pd.DataFrame:
    """One row per park: which inputs exist."""
    osm = _json_keys(os.path.join(paths.INPUT_DIR, "osm", "parks_osm.json"))
    pvg = _json_keys(os.path.join(paths.INPUT_DIR, "osm", "pvgis_horizon.json"))
    ring = os.listdir(os.path.join(paths.INPUT_DIR, "cams_ring"))
    ok = lambda p: os.path.exists(p) and os.path.getsize(p) > 1000  # noqa: E731
    rows = []
    for pid in sel.park_id:
        icon = os.path.join(paths.ICON_SL09_DIR, f"park_{pid}")
        rows.append(dict(park_id=pid, osm=pid in osm, pvgis=pid in pvg,
                         cams=ok(os.path.join(paths.INPUT_DIR, "cams", f"CAMS_rad_{pid}_{TAG}.csv")),
                         ring=sum(f.startswith(f"CAMS_rad_{pid}_r") and f.endswith(f"_{TAG}.csv") for f in ring),
                         correct=all(os.path.exists(os.path.join(paths.CORR_DIR, f"{pid}_cams_corrected_{s}_{model_tag}.parquet")) for s in ("30min", "hourly")),
                         icon=os.path.isdir(icon) and any(f.endswith("_SL.parquet") for f in os.listdir(icon))))
    inv = pd.DataFrame(rows)
    inv["complete"] = inv.osm & inv.pvgis & inv.cams & (inv.ring == 8) & inv.correct & inv.icon
    return inv


def step_site(sel: pd.DataFrame, inv: pd.DataFrame):
    if (inv.osm & inv.pvgis).all():
        return
    subprocess.run([PY, os.path.join(REPO, "scripts", "pv", "fetch_site_inputs.py"), "--selection", paths.SELECTION,
                    "--out", os.path.join(paths.INPUT_DIR, "osm")], check=True, cwd=REPO)


def step_cams(sel: pd.DataFrame, inv: pd.DataFrame, batch: int = 60, sleep_s: int = 1800, max_wait_h: float = 72.0):
    """fetch_cams_parks.py in batches until every point of the selection is there; sleeps on the ADS daily quota."""
    t0 = time.time()
    while not (inv.cams & (inv.ring == 8)).all():
        out = subprocess.run([PY, os.path.join(REPO, "scripts", "pv", "fetch_cams_parks.py"), "--selection", paths.SELECTION,
                              "--out", paths.INPUT_DIR, "--max-requests", str(batch), "--threads", "2"],
                             cwd=REPO, capture_output=True, text=True)
        log = out.stdout + out.stderr
        print(time.strftime("%F %T"), " | ".join(l for l in log.splitlines() if l.startswith(("result", "ADS")) or "to fetch" in l), flush=True)
        if out.returncode:
            raise RuntimeError(f"fetch_cams_parks.py failed:\n{log[-2000:]}")
        inv = inventory(sel, "-")
        if (inv.cams & (inv.ring == 8)).all():
            break
        if (time.time() - t0) / 3600 > max_wait_h:
            raise RuntimeError(f"CAMS still missing after {max_wait_h} h: {inv[~(inv.cams & (inv.ring == 8))].park_id.tolist()}")
        time.sleep(sleep_s if "quota" in log.lower() else 30)


def step_correct(sel: pd.DataFrame, inv: pd.DataFrame, model_tag: str, gpu: int, period: tuple):
    todo = inv[~inv.correct].park_id.tolist()
    if not todo:
        return
    lack = inv[~inv.correct & ~(inv.cams & (inv.ring == 8))].park_id.tolist()
    if lack:
        raise RuntimeError(f"correct: CAMS centre/ring missing for {lack} - run step cams first")
    from parks_pv import stages
    stages.write_points()                                          # points.json from cams_points.csv (derived, all centres)
    stages.run_correction(model_tag, gpu, period[0], period[1], todo)


def step_icon(sel: pd.DataFrame, inv: pd.DataFrame, name: str, previous: str, sources: str, hours: str = "09"):
    """Site lists for NWP-points (confidential coordinates, under nwp_ready_pv/sites_input, never in a repo), then
    run_pv_add.sh: select, stage the added parks, assemble with --only-missing."""
    todo = sel[sel.park_id.isin(inv[~inv.icon].park_id)]
    if todo.empty:
        return
    sdir = os.path.join(paths.ROOT, "nwp_ready_pv", "sites_input")
    add_path, all_path = os.path.join(sdir, f"{name}_add_sites.csv"), os.path.join(sdir, f"{name}_sites.csv")
    for p in (add_path, all_path):
        if os.path.exists(p):
            raise SystemExit(f"{p} exists - choose another --icon-name (never overwritten)")
    add = pd.DataFrame(dict(site_id=todo.park_id, lat=todo.lat, lon=todo.lon, altitude=todo.elevation, group="portfolio"))
    prev = pd.read_csv(os.path.join(sdir, f"{previous}_sites.csv"), dtype={"site_id": str})
    add.to_csv(add_path, index=False)
    pd.concat([prev, add[~add.site_id.isin(prev.site_id)]], ignore_index=True).to_csv(all_path, index=False)
    subprocess.run(["bash", os.path.join(NWP_POINTS, "points", "run_pv_add.sh"), name, previous, sources, hours], check=True, cwd=NWP_POINTS)


def run(steps, model_tag: str = "xgb_v5", gpu: int = 0, icon_name: str = None, icon_previous: str = "pv_v2",
        icon_sources: str = "pv_v1 pv_v2_add", icon_hours: str = "09", dry_run: bool = False,
        period: tuple = ("2023-07-24", "2026-08-01")) -> pd.DataFrame:
    sel = pd.read_csv(paths.SELECTION, dtype={"park_id": str}).dropna(subset=["park_id"])
    inv = inventory(sel, model_tag)
    print(f"{paths.SELECTION}: {len(sel)} parks, {int(inv.complete.sum())} complete", flush=True)
    print(inv[~inv.complete].to_string(index=False), flush=True)
    if dry_run:
        return inv
    for s in steps:
        print(time.strftime("%F %T"), "step", s, flush=True)
        if s == "site":
            step_site(sel, inv)
        elif s == "cams":
            step_cams(sel, inv)
        elif s == "correct":
            step_correct(sel, inv, model_tag, gpu, period)
        elif s == "icon":
            step_icon(sel, inv, icon_name or "pv_" + paths.NAME.rsplit("_", 1)[-1], icon_previous, icon_sources, icon_hours)
        inv = inventory(sel, model_tag)
    print(time.strftime("%F %T"), f"done: {int(inv.complete.sum())}/{len(inv)} complete", flush=True)
    print(inv[~inv.complete].to_string(index=False), flush=True)
    return inv
