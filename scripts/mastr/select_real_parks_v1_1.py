#!/usr/bin/env python3
"""Selection v1.1: repair of selection v1 (decisions 2026-10-01).

1. Class k is recomputed from the simulated power-curve types (lib_name) instead
   of the manufacturer|rotor|rating key of v1, which counted manufacturer
   renames (REpower/Senvion, GE Wind/GE Deutschland) and rating modes of one
   curve (E-101 3000/3050 kW, E-66 1800/2000 kW) as different types.
2. Parks whose class changes under 1., and parks whose layout cannot be
   resolved against OpenStreetMap (FORCE_REPLACE), are replaced in their slot:
   same client, same class, same region as the replaced park (regional clients:
   the client region; nationwide clients: keeps the v1 region rotation), cohort
   different from the partner park of the same client+class if possible,
   candidate must be strictly OSM-consistent (osm_match.park_consistent).
3. Every other v1 park is kept unchanged (its row gets the recomputed k).

Usage: select_real_parks_v1_1.py <wind_turbines_matched.csv> <data/mastr dir> <osm.json> [--seed 42]
Writes wind_park_selection_v1_1.csv and selection_v1_1_changes.csv; v1 files stay untouched.
"""
import argparse
import os

import numpy as np
import pandas as pd

from osm_match import OsmIndex, load_osm, park_consistent
from select_real_parks_v1 import CLASSES, COHORT

# layouts not resolvable against OSM (see reports/park_synthesis_v1.md)
FORCE_REPLACE = {
    "SEL923430089738": "layout unresolvable: both units on one MaStR point, 4 candidate OSM masts",
    "SEL922558743003": "layout unresolvable: Koenigsmoor I/II/IV cluster, several E-66 400-900 m off OSM",
}


def classify(n: int, k: int) -> str:
    return next(c for c, f in CLASSES if f(n, k))


def lib_classes(turbines: pd.DataFrame) -> pd.DataFrame:
    g = turbines.groupby("lokation")
    p = g.agg(n=("EinheitMastrNummer", "size"), k_lib=("lib_name", "nunique"))
    p["cls_lib"] = [classify(r.n, r.k_lib) for r in p.itertuples()]
    return p


def types_str(units: pd.DataFrame) -> str:
    return " | ".join(f"{v}x {n}" for n, v in units["lib_name"].value_counts().items())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("matched"); ap.add_argument("mastr_dir"); ap.add_argument("osm")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    t = pd.read_csv(a.matched, low_memory=False)
    sel = pd.read_csv(os.path.join(a.mastr_dir, "wind_park_selection_v1.csv"))
    pool = pd.read_csv(os.path.join(a.mastr_dir, "wind_park_pool_v1.csv")).set_index("lokation")
    lc = lib_classes(t[t.lokation.isin(pool.index)])
    pool = pool.join(lc[["k_lib", "cls_lib"]])
    osm = OsmIndex(load_osm(a.osm))
    units = {lk: g for lk, g in t[t.lokation.isin(pool.index)].groupby("lokation")}

    def consistent(lk: str) -> bool:
        u = units[lk]
        return park_consistent(osm.match_park(u), u["Rotordurchmesser"].values)

    used = set(sel["lokation"])
    rows, changes = [], []
    for i, r in sel.iterrows():
        lk = r["lokation"]
        cls_new = pool.loc[lk, "cls_lib"]
        reason = FORCE_REPLACE.get(lk)
        if reason is None and cls_new != r["cls"]:
            reason = f"class changes {r['cls']} -> {cls_new} when k counts power-curve types"
        if reason is None:
            rows.append({**r.to_dict(), "k": int(pool.loc[lk, "k_lib"]), "replaced": ""})
            continue
        partner = sel[(sel["client"] == r["client"]) & (sel["cls"] == r["cls"]) & (sel.index != i)]
        have = set(partner["cohort"])
        cand = pool[(pool["cls_lib"] == r["cls"]) & (pool["region"] == r["region"]) & ~pool.index.isin(used)]
        cand = cand.assign(score=cand["cohort"].isin(have).astype(int) + rng.random(len(cand)) * 0.5).sort_values("score")
        pick, n_tested = None, 0
        for c in cand.index:
            n_tested += 1
            if consistent(c):
                pick = c
                break
        if pick is None:
            raise SystemExit(f"no OSM-consistent replacement for {lk} ({r['client']}/{r['cls']}/region {r['region']})")
        used.add(pick)
        p = pool.loc[pick]
        u = units[pick]
        rows.append({"client": r["client"], "kind": r["kind"], "cls": r["cls"], "lokation": pick,
                     "region": int(p["region"]), "n": int(p["n"]), "k": int(p["k_lib"]), "n_operators": int(p["ops"]),
                     "cap_mw": round(p["cap_kw"] / 1e3, 2), "y_min": int(p["y_min"]), "y_max": int(p["y_max"]),
                     "cohort": p["cohort"], "lat": round(p["lat"], 4), "lon": round(p["lon"], 4),
                     "hubs": p["hubs"], "types": types_str(u), "name": p["name"], "replaced": lk})
        changes.append({"client": r["client"], "cls": r["cls"], "old": lk, "old_name": r["name"],
                        "new": pick, "new_name": p["name"], "reason": reason,
                        "candidates_tested": n_tested, "old_cohort": r["cohort"], "new_cohort": p["cohort"]})

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(a.mastr_dir, "wind_park_selection_v1_1.csv"), index=False)
    ch = pd.DataFrame(changes)
    ch.to_csv(os.path.join(a.mastr_dir, "selection_v1_1_changes.csv"), index=False)
    print(ch.to_string(index=False))
    print(f"\nselected {len(out)} | turbines {int(out.n.sum())} | capacity {out.cap_mw.sum():.0f} MW | "
          f"classes {out.cls.value_counts().to_dict()}")


if __name__ == "__main__":
    main()
