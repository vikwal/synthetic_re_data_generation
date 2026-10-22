#!/usr/bin/env python
"""Experiment (2026-10-22, Viktor: test (b) with (a) as the first stage): separate the park at the register point from
neighbouring plants without per-park exceptions. Not part of the production chain.

(a) outline: OSM plants within 400 m, sorted by distance to the register point; the nearest one always, further ones
    only while the union stays <= A_FACTOR x the fenced area expected from the capacity and the era density.
(b) groups: when the measured module area exceeds M_HIGH m2/kWp, whole groups are dropped (farthest group centroid
    from the register point first) until it fits; if one group is left and still too large, its farthest accepted
    windows are dropped down to M_TRIM m2/kWp.
Runs on every fleet park with a fetcher (selection v2, tiles from the cache) whose outline changes under (a), plus (b)
on all of them, without the per-park exceptions; and on the Trianel parks. Compares the arrays with the production
run (_v2b with exceptions; Trianel v6).

    synthre/bin/python scripts/pv/test_main_park.py [--parks ID,ID] [--trianel]
Output: ~/Work/pv_sim2real/results/test_main_park_2026-10-22.csv (+ printed summary)."""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shapely.ops import unary_union  # noqa: E402

from parks_pv import paths  # noqa: E402
from parks_pv.params import DENSITY_KWP_HA, DENSITY_YEARS  # noqa: E402
from pv import geodata_tiles as GT, geodata_windows as GW  # noqa: E402

A_FACTOR, M_HIGH, M_TRIM = 1.5, 12.0, 8.0


def expected_fenced_m2(kwp: float, year: int) -> float:
    return float(kwp) / float(np.interp(int(year), DENSITY_YEARS, DENSITY_KWP_HA)) * 1e4


def select_fences(fences: pd.DataFrame, kwp: float, year: int, max_dist: float = 400.0) -> tuple:
    """(a) (polygon, n_used, n_near): nearest plant always, further plants by distance while the union <= A_FACTOR x expected."""
    near = fences[fences.dist_m <= max_dist].sort_values(["dist_m", "area_ha"])
    if not len(near):
        return None, 0, 0
    lim = A_FACTOR * expected_fenced_m2(kwp, year)
    poly, used = near.geom.iloc[0], 1
    for g in near.geom.iloc[1:]:
        u = unary_union([poly, g])
        if u.area > lim:
            continue
        poly, used = u, used + 1
    return poly, used, len(near)


def plan_m2(win: pd.DataFrame, tab_share: float) -> float:
    ok = win[win.ok.astype(bool)]
    return float(ok.area_tab.sum()) / max(float(tab_share), 1e-6)


def trim_groups(win: pd.DataFrame, nw: int, kwp: float, centre_xy: tuple, tab_share: float) -> tuple:
    """(b) (window table, action): drop farthest groups, then trim the last group, while the area exceeds the band."""
    if plan_m2(win, tab_share) / kwp <= M_HIGH:
        return win, ""
    win = win.copy()
    cx, cy = centre_xy
    G = GW.group_windows(win)
    gws = {int(g.group): [int(x) for x in str(g.windows).split()] for g in G.itertuples()}
    dist = {k: float(np.mean([np.hypot(win.loc[win.w == w_, "x"].iloc[0] - cx, win.loc[win.w == w_, "y"].iloc[0] - cy) for w_ in ws])) for k, ws in gws.items()}
    action = []
    for k in sorted(dist, key=dist.get, reverse=True):
        if plan_m2(win, tab_share) / kwp <= M_HIGH or len([g for g in gws if win[win.w.isin(gws[g])].ok.any()]) <= 1:
            break
        m = win.w.isin(gws[k])
        win.loc[m, "ok"] = False
        win.loc[m, "reason"] = "anderer Park (Fläche)"
        action.append(f"drop group {k}")
    if plan_m2(win, tab_share) / kwp > M_HIGH:                     # one group left, still too large: trim its farthest windows
        acc = win[win.ok.astype(bool)].assign(d=lambda t: np.hypot(t.x - cx, t.y - cy)).sort_values("d", ascending=False)
        for k, r in acc.iterrows():
            if plan_m2(win, tab_share) / kwp <= M_TRIM or win.ok.astype(bool).sum() <= 2:
                break
            win.at[k, "ok"] = False
            win.at[k, "reason"] = "anderer Park (Fläche)"
            action.append(f"trim w{int(r.w)}")
    return win, ";".join(action)


def summarise(out: dict) -> str:
    return "; ".join(f"{a.tilt:.1f}/{a.azimuth:.0f}x{a.share:.2f}" for a in out["arrays"].itertuples())


def regroup(win: pd.DataFrame, nw: int, register_orientation: str, box: bool) -> dict:
    flag = GW.rejected_flag(win, nw, box)
    G = GW.group_windows(win)
    return dict(win=win, G=G, arrays=GW.arrays_from_groups(G, register_orientation), park=GW.park_status(win, G, flag, box))


def fleet(parks: list) -> list:
    import geometry_release_states as R
    from pyproj import Transformer
    from shapely.geometry import Point
    sel = pd.read_csv(paths.SELECTION, dtype={"park_id": str})
    osm = json.load(open(os.path.join(paths.INPUT_DIR, "osm", "parks_osm.json")))
    prod = pd.concat([pd.read_csv(f, dtype={"pv": str}) for f in glob.glob(os.path.join(paths.DATA_DIR, "geodata_release_*_v2b.csv")) if not f.endswith("_arrays.csv")])
    prod_a = pd.concat([pd.read_csv(f, dtype={"park_id": str}) for f in glob.glob(os.path.join(paths.DATA_DIR, "geodata_release_*_v2b_arrays.csv"))])
    rows = []
    for r in sel[sel.bundesland.isin(R.STATE) & (sel.bundesland != "Thüringen")].itertuples():
        if parks and r.park_id not in parks:
            continue
        code, zone = R.STATE[r.bundesland]
        to_m = Transformer.from_crs("EPSG:4326", GT.CRS[zone], always_xy=True).transform
        pt = Point(*to_m(float(r.lon), float(r.lat)))
        union, fences = GW.osm_footprint((osm.get(r.park_id) or {}).get("elements", []), [pt], to_m, R.FENCE_MAX_M)
        p_a, used, n_near = select_fences(fences, r.dc_kwp, r.year) if union is not None else (None, 0, 0)
        best = prod[(prod.pv == r.park_id) & prod.best.astype(bool)]
        row = dict(park=r.park_id, state=code, kwp=r.dc_kwp, n_fences=n_near, fences_used=used, prod_status="accepted" if len(best) else "not accepted",
                   prod_arrays="; ".join(f"{a.tilt:.1f}/{a.azimuth:.0f}x{a.share:.2f}" for a in prod_a[prod_a.park_id == r.park_id].itertuples()))
        src = best.source.iloc[0] if len(best) else None
        changed_outline = union is not None and used < n_near
        rd = os.path.join(REPO, "results", "pv", "geodata_release_v2b", r.park_id)
        try:
            if changed_outline or r.park_id in ("SEL978764131202", "SEL954870838563"):
                tiles = R.fetch_tiles(r.bundesland, float(r.lat), float(r.lon), int(r.year), r.landkreis, R.municipality(r.unit_ids) if r.bundesland == "Hessen" else None,
                                      margin_m=R.fetch_margin(r, osm, zone))
                srcs = dict(R.sources_from(tiles))
                src = src or next((s for s in ("laz", "dom1_laser", "bdom_image", "dom20_image") if s in srcs), None)
                g = GW.analysis_grid(p_a, GW.RES)
                ras = R.rasters(tiles, src, srcs[src], g["bounds"], zone, float(r.lat), float(r.lon))
                pitch0 = R.ortho_pitch(tiles, g["bounds"], g["inside"], zone)
                o = GW.measure_park(ras["hag"], ras["dgm"], g["inside"], ras["conv"], src, pitch0, GW.RES, r.orientation_class)
                centre = (pt.x - g["bounds"][0], pt.y - g["bounds"][1]); box = False
            elif src and os.path.exists(os.path.join(rd, f"windows_{src}.csv")):
                win = pd.read_csv(os.path.join(rd, f"windows_{src}.csv"))
                pj = json.load(open(os.path.join(rd, f"park_{src}.json")))
                box = bool(pj.get("box"))
                if box:
                    centre = (pj["box_side_m"] / 2, pj["box_side_m"] / 2)
                else:
                    g = GW.analysis_grid(union, GW.RES); centre = (pt.x - g["bounds"][0], pt.y - g["bounds"][1])
                nw = int(round(pj["win_m"] / GW.RES))
                o = dict(win=win, nw=nw, park=pj, arrays=pd.read_csv(os.path.join(rd, f"arrays_{src}.csv")))
            else:
                rows.append(dict(row, test_status="not measured", test_arrays="", m2_before=np.nan, m2_after=np.nan, action="")); continue
            nw = o["nw"]
            m2_0 = plan_m2(o["win"], o["park"]["tab_share_accepted"]) / r.dc_kwp if o["win"].ok.any() else np.nan
            win_b, action = trim_groups(o["win"], nw, r.dc_kwp, centre, o["park"]["tab_share_accepted"])
            ob = regroup(win_b, nw, r.orientation_class, box) if (action or changed_outline or r.park_id in ("SEL978764131202", "SEL954870838563")) else o
            row.update(test_status=ob["park"]["status"], test_arrays=summarise(ob) if ob["park"]["status"] == "accepted" else "", m2_before=round(m2_0, 1),
                       m2_after=round(plan_m2(ob["win"], ob["park"]["tab_share_accepted"]) / r.dc_kwp, 1) if ob["win"].ok.any() else np.nan, action=action,
                       outline_changed=changed_outline)
        except Exception as ex:
            row.update(test_status=f"error {type(ex).__name__}: {str(ex)[:80]}")
        rows.append(row)
        print({k: v for k, v in row.items() if k not in ("prod_arrays",)}, flush=True)
    return rows


def trianel() -> list:
    import geometry_trianel_windows as TW
    v6 = pd.read_csv(os.path.join(TW.FIGS, "geodata_arrays_trianel_windows_v6.csv"))
    rows = []
    for pv in ["pv1", "pv2", "pv3", "pv5", "pv6", "pv7"]:
        P = TW.prepare(pv, with_rgb=False)
        kwp, year = P["kw_master"], P["year_built"]
        changed = False
        if P["fp_src"] == "osm":
            pa, used, n_near = select_fences(P["fences"].assign(dist_m=P["fences"].dist_m), kwp, year, TW.FENCE_MAX_M)
            changed = used < n_near
        o = GW.measure_park(P["hag"], P["dgm"], P["inside"], P["conv"], P["source"], P["pitch0"], P["RES"], P["register_orientation"])
        centre = (P["park_pt"].x - P["minx"], P["park_pt"].y - P["miny"])
        win_b, action = trim_groups(o["win"], o["nw"], kwp, centre, o["park"]["tab_share_accepted"])
        rows.append(dict(park=pv, state=P["sub"], kwp=kwp, n_fences=int((P["fences"].dist_m <= TW.FENCE_MAX_M).sum()) if len(P["fences"]) else 0,
                         outline_would_change=changed, m2_before=round(plan_m2(o["win"], o["park"]["tab_share_accepted"]) / kwp, 1), action=action,
                         prod_arrays="; ".join(f"{a.tilt:.1f}/{a.azimuth:.0f}x{a.share:.2f}" for a in v6[v6.park_id == pv].itertuples()),
                         test_arrays=summarise(o) if not action else summarise(regroup(win_b, o["nw"], P["register_orientation"], False))))
        print(rows[-1], flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parks", default=None)
    ap.add_argument("--trianel", action="store_true")
    a = ap.parse_args()
    if a.trianel:
        rows = trianel()
        out = os.path.expanduser("~/Work/pv_sim2real/results/test_main_park_trianel_2026-10-22.csv")    # anonymous pv ids, no coordinates
    else:
        rows = fleet(a.parks.split(",") if a.parks else [])
        out = os.path.expanduser("~/Work/pv_sim2real/results/test_main_park_2026-10-22.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print("written", out)


if __name__ == "__main__":
    main()
