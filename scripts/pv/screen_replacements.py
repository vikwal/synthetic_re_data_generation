#!/usr/bin/env python
"""Screen replacement candidates for release parks whose newest surface survey predates the plant (2026-10-29,
Viktor: replace them by parks of the same stratum that the survey covers, so that the whole chain can be run).

For every park to replace, the candidates come from the same stratum of the pool as in replace_parks.py (class,
region, cohort, size bin; parks of the selection excluded) in a state with a geodata fetcher (Thüringen excluded:
hand-filled cache). They are tried in order of preference (earlier commissioning first, since the surveys are older
than the newest parks; then terrain class; then a seeded random key) through the checks of replace_parks.py
(coordinate in the registered municipality, OSM solar plant within 400 m) and then the full geometry measurement of
parks_pv.geometry_estimate (tiles, windows, park acceptance, survey after commissioning). The first candidate with an
accepted source is the proposal; the others tried are listed. Nothing in the selection or the release is changed.

    synthre/bin/python scripts/pv/screen_replacements.py --replace SEL983846005258,SEL931358182929 [--max-tries 6]
--relax overrides one stratum field when the stratum is exhausted (2026-11-03, NRW slot), e.g. --relax size_bin=2-5MW.
Outputs (never overwritten): results/pv/replacement_screen_<date>/{screen.csv,proposal.csv,<candidate>/...}."""
import argparse
import json
import os
import sys
import traceback

import numpy as np
import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parks_pv import geometry_estimate as GRS  # noqa: E402
import replace_parks as RP  # noqa: E402

STATES = RP.FETCH_STATES - {"Thüringen"}


def screen_candidate(old: pd.Series, pk: pd.Series, cache: dict, outd: str) -> dict:
    """Checks and measurement of one pool park in the slot of old; returns one result row."""
    res = dict(old=old.park_id, cand=pk.park_id, state=pk.bundesland, gemeinde=pk.gemeinde, year=int(pk.year), dc_kwp=round(float(pk.dc_kwp), 1),
               terrain=pk.terrain_cls, orientation_class=pk.orientation_class, tilt_class=pk.tilt_class)
    ok, txt = RP.in_municipality(pk, RP.reverse(pk.lat, pk.lon, cache))
    res.update(coord_ok=ok, coord_txt=txt)
    if not ok:
        return dict(res, verdict="coordinate outside municipality")
    n_osm, ans = RP.osm_plant_near(pk.lat, pk.lon)
    if n_osm < 0:
        raise SystemExit(f"{pk.park_id}: Overpass unavailable - stopped instead of counting the candidate as without OSM plant")
    res.update(n_osm=n_osm)
    if n_osm == 0:
        return dict(res, verdict="no OSM plant within 400 m")
    r = pd.Series(RP.selection_row(old, pk))
    code, zone = GRS.STATE[pk.bundesland]
    osm = {pk.park_id: ans}
    try:
        tiles = GRS.fetch_tiles(pk.bundesland, float(r.lat), float(r.lon), int(r.year), r.landkreis,
                                GRS.municipality(r.unit_ids) if pk.bundesland == "Hessen" else None, margin_m=GRS.fetch_margin(r, osm, zone))
        rows, _, _ = GRS.measure(r, tiles, osm, zone, os.path.join(outd, pk.park_id))
    except Exception as e:
        traceback.print_exc()
        return dict(res, verdict=f"measurement error {type(e).__name__}: {str(e)[:80]}")
    for x in rows:
        x.update(state_code=code, orientation_class_reg=r.orientation_class, tilt_class_reg=r.tilt_class, dc_kwp=float(r.dc_kwp), year_built=int(r.year))
    df = GRS.evaluate(pd.DataFrame(rows))
    df.to_csv(os.path.join(outd, pk.park_id, "sources.csv"), index=False)
    srcs = "; ".join(f"{s.source}:{s.status}:{int(s.n_ok) if pd.notna(s.get('n_ok')) else 0}/{int(s.n_windows) if pd.notna(s.get('n_windows')) else 0}:"
                     f"{int(s.survey_year) if pd.notna(s.get('survey_year')) else '?'}" for _, s in df.iterrows())
    best = df[df.best]
    if not len(best):
        return dict(res, sources=srcs, verdict="no accepted source with survey after commissioning")
    b = best.iloc[0]
    return dict(res, sources=srcs, verdict="accepted", best_source=b.source, survey_year=b.get("survey_year"), footprint=b.get("footprint"),
                n_ok=int(b.n_ok), tilt=round(float(b.tilt_measured), 1) if np.isfinite(b.tilt_measured) else np.nan,
                azimuth=round(float(b.az_measured)) if np.isfinite(b.az_measured) else np.nan, ew=bool(b.ew_measured),
                or_agree=b.or_agree, tilt_agree_tol=bool(b.tilt_agree_tol))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replace", required=True)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-tries", type=int, default=6, help="measured candidates per park (coordinate/OSM rejects not counted)")
    ap.add_argument("--exclude", default="", help="candidates not to try (e.g. a proposal rejected after review)")
    ap.add_argument("--relax", default="", help="stratum field overridden for the candidate search, e.g. size_bin=2-5MW or cohort=2013-20")
    ap.add_argument("--extend", action="store_true", help="measure with the outline extension (variant c)")
    ap.add_argument("--selection", default="pv_park_selection_v2.csv", help="selection file in FL/mastr whose parks are replaced")
    ap.add_argument("--rules", default="", help="rule switches as in scripts/pv/estimate_geometry.py, e.g. box_cap,box_tab_near")
    ap.add_argument("--out", default=os.path.join(REPO, "results", "pv", "replacement_screen_" + pd.Timestamp.now().strftime("%Y-%m-%d")))
    a = ap.parse_args()
    if os.path.exists(os.path.join(a.out, "screen.csv")):
        raise SystemExit(f"{a.out}/screen.csv exists - choose another --out")
    os.makedirs(a.out, exist_ok=True)
    GRS.EXTEND_ON = a.extend
    print("rules:", GRS.set_rules(a.rules), flush=True)
    relax = dict([a.relax.split("=", 1)]) if a.relax else {}
    if set(relax) - {"cls", "region", "cohort", "size_bin"}:
        raise SystemExit(f"--relax: unknown stratum field {a.relax}")
    sel = pd.read_csv(os.path.join(RP.FL, "mastr", a.selection))
    pool = pd.read_csv(os.path.join(RP.FL, "mastr", "pv_park_pool_v1.csv"))
    pool = pool[pool.reject.isna()]
    sel["gemeinde"] = sel.park_id.map(pool.set_index("park_id").gemeinde)
    cache = json.load(open(RP.CACHE)) if os.path.exists(RP.CACHE) else {}
    rng = np.random.default_rng(a.seed)
    used, out, proposal = set(sel.park_id) | set(filter(None, a.exclude.split(","))), [], []
    for pid in a.replace.split(","):
        old = sel[sel.park_id == pid].iloc[0]
        st = {k: type(old[k])(relax[k]) if k in relax else old[k] for k in ("cls", "region", "cohort", "size_bin")}
        c = pool[(pool.cls == st["cls"]) & (pool.region == st["region"]) & (pool.cohort == st["cohort"]) & (pool.size_bin == st["size_bin"])
                 & pool.bundesland.isin(STATES) & ~pool.park_id.isin(used)].copy()
        c["key"] = (c.year - c.year.min()) + (c.terrain_cls != old.terrain_cls).astype(int) * 0.5 + rng.random(len(c)) * 0.5
        print(f"== {pid} {old.bundesland} {st['cls']} region {st['region']} {st['cohort']} {st['size_bin']} (relaxed: {a.relax or '-'}): {len(c)} candidates", flush=True)
        n_meas, found = 0, None
        for pk in c.sort_values("key").itertuples():
            if n_meas >= a.max_tries:
                break
            pk = pool[pool.park_id == pk.park_id].iloc[0]
            row = screen_candidate(old, pk, cache, os.path.join(a.out, pid))
            n_meas += row["verdict"] not in ("coordinate outside municipality", "no OSM plant within 400 m")
            out.append(row)
            print("  ", {k: row.get(k) for k in ("cand", "state", "year", "verdict", "sources", "tilt", "azimuth")}, flush=True)
            pd.DataFrame(out).to_csv(os.path.join(a.out, "screen.csv"), index=False)
            if row["verdict"] == "accepted":
                found = row
                break
        if found:
            used.add(found["cand"])
        proposal.append(dict(old=pid, old_state=old.bundesland, old_year=int(old.year), client=old.client, cls=old.cls, region=old.region,
                             cohort=old.cohort, size_bin=old.size_bin, relaxed=a.relax or None, new=found["cand"] if found else None,
                             new_state=found["state"] if found else None, new_year=found["year"] if found else None,
                             source=found.get("best_source") if found else None, survey_year=found.get("survey_year") if found else None,
                             tried=sum(r["old"] == pid for r in out), candidates=len(c)))
        pd.DataFrame(proposal).to_csv(os.path.join(a.out, "proposal.csv"), index=False)
    print(pd.DataFrame(proposal).to_string(index=False))


if __name__ == "__main__":
    main()
