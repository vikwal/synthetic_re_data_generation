"""Candidates for the fleet expansion (2026-11-05, Viktor: 200 good parks, reviewed by him, then forecast data).
Pool parks that passed the pre-screen (prescreen_pool.py + recheck_municipality.py, prescreen_v2.csv) are picked per class
until the expected number of good parks covers the gap of the class to --per-class (current fleet counted), times
--margin. Pick order within a class: greedy maximum of p_good x spread, spread = min(distance to the fleet and the
picked parks of the same class and cohort, CAP_KM) blended with the distance within the class, so new cohorts and empty
regions come first and a second park of the same type next to an existing one comes last.
p_good (pool_vorpruefung_2026-11-05.md, mid estimate): acceptance rate of the cohort x 0.85 (review), x 0.5 without OSM
plant (box), x 0.5 with an area or year flag, x 0.3 for C5 (both directions measured), 0 for BW 2021+ (no survey yet).
Output: a selection CSV (format of pv_park_selection_v3, client X, kind expand) for PV_SELECTION of
geometry_release_states.py, with the pick rank and p_good."""
import argparse
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [REPO, os.path.join(REPO, "scripts", "pv")]
import replace_parks as RP  # noqa: E402

ACC = {"<=2012": 7 / 9, "2013-20": 11 / 14, "2021+": 24 / 46}
CAP_KM = 150.0
R_KM = 6371.0


def p_good(p: pd.DataFrame) -> pd.Series:
    clean = (p.flag_area.fillna("") == "") & (p.flag_years.fillna("") == "")
    pg = p.cohort.map(ACC) * 0.85 * np.where(p.n_osm > 0, 1.0, 0.5) * np.where(clean, 1.0, 0.5) * np.where(p.cls == "C5_multi", 0.3, 1.0)
    return pd.Series(np.where((p.state == "Baden-Württemberg") & (p.cohort == "2021+"), 0.0, pg), index=p.index)


def km(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Haversine distances (km) of the points a (n x 2, lat lon degrees) to b (m x 2) as n x m."""
    a, b = np.radians(a)[:, None, :], np.radians(b)[None, :, :]
    h = np.sin((b[..., 0] - a[..., 0]) / 2) ** 2 + np.cos(a[..., 0]) * np.cos(b[..., 0]) * np.sin((b[..., 1] - a[..., 1]) / 2) ** 2
    return 2 * R_KM * np.arcsin(np.sqrt(h))


def pick_class(cand: pd.DataFrame, fleet: pd.DataFrame, need: float, w_cohort: float = 0.7) -> pd.DataFrame:
    """Greedy picks of one class until the summed p_good reaches need."""
    cand = cand[cand.p_good > 0].copy()
    pts = cand[["lat", "lon"]].values
    d_cls = np.full(len(cand), CAP_KM)
    d_coh = np.full(len(cand), CAP_KM)
    for coh in cand.cohort.unique():
        m = (cand.cohort == coh).values
        f = fleet[fleet.cohort == coh]
        if len(f):
            d_coh[m] = np.minimum(d_coh[m], km(pts[m], f[["lat", "lon"]].values).min(axis=1))
    if len(fleet):
        d_cls = np.minimum(d_cls, km(pts, fleet[["lat", "lon"]].values).min(axis=1))
    out, got, free = [], 0.0, np.ones(len(cand), bool)
    while got < need and free.any():
        score = cand.p_good.values * (w_cohort * d_coh + (1 - w_cohort) * d_cls)
        score[~free] = -1
        i = int(np.argmax(score))
        free[i] = False
        out.append(i)
        got += cand.p_good.values[i]
        d = km(pts, pts[i:i + 1])[:, 0]
        d_cls = np.minimum(d_cls, d)
        same = (cand.cohort == cand.cohort.iloc[i]).values
        d_coh[same] = np.minimum(d_coh[same], d[same])
    return cand.iloc[out].assign(pick=np.arange(1, len(out) + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prescreen", default=os.path.join(REPO, "results", "pv", "pool_prescreen_2026-11-05", "prescreen_v2.csv"))
    ap.add_argument("--fleet", default=os.path.expanduser("~/Work/pv_sim2real/results/geo_review_2026-11-05/flotte/flotte_status_2026-11-05.csv"))
    ap.add_argument("--per-class", type=int, default=40)
    ap.add_argument("--margin", type=float, default=1.3, help="expected good parks per class = gap x margin")
    ap.add_argument("--max-per-class", type=int, default=90, help="at most this many candidates per class in one wave")
    ap.add_argument("--exclude", default="", help="file with park_id column (parks already measured or rejected)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    pool = pd.read_csv(os.path.join(RP.FL, "mastr", "pv_park_pool_v1.csv")).set_index("park_id")
    pre = pd.read_csv(a.prescreen, index_col=0)
    cand = pre[pre.stage.str.startswith("passed")].join(pool[["lat", "lon"]])
    cand["p_good"] = p_good(cand)
    fl = pd.read_csv(a.fleet)
    fleet = pool.loc[fl.park, ["lat", "lon", "cls", "cohort"]].assign(status=fl.status.values)
    if a.exclude:
        cand = cand[~cand.index.isin(pd.read_csv(a.exclude).park_id)]
    picks = []
    for c, g in cand.groupby("cls"):
        f = fleet[fleet.cls == c]
        need = max(a.per_class - len(f), 0) * a.margin
        s = pick_class(g, f, need)
        s = s[s.pick <= a.max_per_class]
        print(f"{c}: fleet {len(f)}, need {need:.0f} expected good, {len(s)} candidates, expected {s.p_good.sum():.1f} of {len(g)} passed", flush=True)
        picks.append(s)
    sel = pd.concat(picks)
    rows = []
    for pid, r in sel.iterrows():
        pk = pool.loc[pid].copy()
        pk["park_id"] = pid
        old = pd.Series({"client": "X", "kind": "expand", "cls": r.cls})
        rows.append(dict(RP.selection_row(old, pk), pick=int(r.pick), p_good=round(float(r.p_good), 3), n_osm=int(r.n_osm),
                         coord_rule=r.coord_rule, flag_area=r.flag_area, flag_years=r.flag_years))
    out = pd.DataFrame(rows)
    out.to_csv(a.out, index=False)
    print(len(out), "candidates ->", a.out)
    print(pd.crosstab(out.cls, out.bundesland).to_string())
    print(pd.crosstab(out.cls, out.cohort).to_string())


if __name__ == "__main__":
    main()
