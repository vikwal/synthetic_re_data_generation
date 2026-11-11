#!/usr/bin/env python
"""Replace release parks whose MaStR coordinate does not lie in the registered municipality (2026-10-22, Viktor:
replace them in the fleet, test parks included). For every park to replace, the candidates come from the same
stratum of the pool (pv.selection: class, region, cohort, size bin; terrain class preferred, as the second park of a
pair in select()) and are tried in order of preference (state with a geodata fetcher, then terrain class, then a
seeded random key) until one passes both checks: the coordinate lies in the registered municipality (Nominatim
reverse geocoding, municipality or county name) and an OSM solar plant lies within 400 m (Overpass). The replacement
takes over client, role, fold and cell of the replaced park. v1 files are not touched.

--assign (2026-11-03, selection v3) skips the search: a CSV with old, new, reason (proposals of screen_replacements.py
confirmed by Viktor) is applied to the --src version and written as --dst; the stratum may differ (relaxed slot).

    synthre/bin/python scripts/pv/replace_parks.py --replace SEL948807179706,SEL991206873119,SEL946560347417
    synthre/bin/python scripts/pv/replace_parks.py --assign data/parks_pv_v3/selection_v3_assign.csv --src v2 --dst v3
Outputs: FL_Contribution/data/mastr/pv_park_selection_<dst>.csv, FL_Contribution/data/pv_transfer_split_<dst>.csv,
         FL_Contribution/data/mastr/selection_<dst>_changes.csv (old, new, reason, candidates tested, checks)."""
import argparse
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
FL = os.path.expanduser("~/Work/FL_Contribution/data")
UA = "pv-sim2real-geocheck/1.0 (research; HKA)"
FETCH_STATES = {"Bayern", "Niedersachsen", "Mecklenburg-Vorpommern", "Brandenburg", "Nordrhein-Westfalen", "Hessen", "Sachsen", "Sachsen-Anhalt", "Baden-Württemberg", "Rheinland-Pfalz"}
CACHE = os.path.join(REPO, "data", "parks_pv_v1", "_nominatim_cache.json")


ABBREV = [(r"\ba\.\s*d\.\s*", "an der "), (r"\bi\.\s*d\.\s*", "in der "), (r"\bb\.\s*", "bei "), (r"\ba\.\s*", "am "),
          (r"\bi\.\s*", "in "), (r"\bst\.\s*", "sankt "), (r"\bopf\.", "oberpfalz"), (r"\bbay\.", "bayern"),
          (r"\bndb\.", "niederbayern"), (r"\bufr\.", "unterfranken"), (r"\bofr\.", "oberfranken"), (r"\bmfr\.", "mittelfranken")]


def _norm(s) -> str:
    """Comparable municipality or county name: MaStR abbreviations expanded (2026-11-05: "Neustadt a.d. Donau" is
    "Neustadt an der Donau" in OSM), parentheses and administrative words (whole words only: "Neustadt" stays) removed, letters only."""
    s = str(s or "").lower().replace("ß", "ss")
    for pat, rep in ABBREV:
        s = re.sub(pat, rep, s)
    s = re.sub(r"\(.*?\)|\b(landkreis|kreisfreie stadt|stadt|kreis|städteregion|region|verwaltungsgemeinschaft)\b", "", s)
    return re.sub(r"[^a-zäöü]", "", s)


def reverse(lat: float, lon: float, cache: dict) -> dict:
    key = f"{lat:.5f},{lon:.5f}"
    if key not in cache:
        q = urllib.parse.urlencode(dict(lat=lat, lon=lon, format="jsonv2", zoom=12, addressdetails=1))
        with urllib.request.urlopen(urllib.request.Request("https://nominatim.openstreetmap.org/reverse?" + q, headers={"User-Agent": UA}), timeout=60) as f:
            cache[key] = json.load(f).get("address", {})
        json.dump(cache, open(CACHE, "w"))
        time.sleep(1.1)                                           # Nominatim usage policy: at most 1 request per second
    return cache[key]


def in_municipality(r, addr: dict) -> tuple:
    """(ok, text): the reverse-geocoded municipality or (fallback) the county matches the MaStR entry."""
    names = [addr.get(k) for k in ("municipality", "city", "town", "village", "suburb")]
    g, k = _norm(r.gemeinde), _norm(r.landkreis)
    hit_g = any(n and (_norm(n) == g or g in _norm(n) or _norm(n) in g) for n in names if _norm(n))
    hit_k = bool(k) and (k in _norm(addr.get("county")) or _norm(addr.get("county")) in k) if addr.get("county") else False
    return bool(hit_g or (not g and hit_k)), f"{[n for n in names if n]} / {addr.get('county')} / {addr.get('state')}"


def osm_plant_near(lat: float, lon: float, radius_m: int = 400) -> tuple:
    """(number of OSM solar plants within radius_m, Overpass answer of 3 km): the query of the release inputs
    (pv.site_inputs.fetch_overpass, mirrors and retries), plants counted as in pv.geodata_windows.osm_footprint."""
    from pyproj import Transformer
    from shapely.geometry import Point
    from pv import geodata_windows as GW, site_inputs
    ans = site_inputs.fetch_overpass(lat, lon, 3000)
    if ans.get("error"):                                          # no Overpass answer is not "no plant" (2026-10-29)
        return -1, ans
    zone = 32 if lon < 12.0 else 33
    to_m = Transformer.from_crs("EPSG:4326", f"EPSG:258{zone}", always_xy=True).transform
    _, fences = GW.osm_footprint(ans.get("elements", []), [Point(*to_m(lon, lat))], to_m, radius_m)
    return int((fences.dist_m <= radius_m).sum()) if len(fences) else 0, ans


def selection_row(old: pd.Series, pk: pd.Series) -> dict:
    """A selection row for pool park pk in the slot (client, kind, cls) of the replaced park (fields as pv.selection.select)."""
    last = pd.to_datetime(pk.commissioned_last, errors="coerce")
    return {"client": old.client, "kind": old.kind, "cls": old.cls, "park_id": pk.park_id, "region": int(pk.region), "cohort": pk.cohort,
            "size_bin": pk.size_bin, "terrain_cls": pk.terrain_cls, "dc_kwp": round(float(pk.dc_kwp), 1), "ac_kw": round(float(pk.ac_kw), 1),
            "inv_kw": pk.inv_kw, "dc_ac": round(float(pk.dc_ac), 3), "year": int(pk.year), "year_last": int(last.year) if pd.notna(last) else int(pk.year),
            "n_units": int(pk.n_units), "orientation_class": pk.orientation_class, "tilt_class": pk.tilt_class,
            "secondary_orientation": pk.secondary_orientation, "secondary_tilt_class": pk.secondary_tilt_class, "uniform_flag": pk.uniform_flag,
            "power_limit": pk.power_limit, "area_ha": pk.area_ha, "n_modules": pk.n_modules, "lat": round(float(pk.lat), 5), "lon": round(float(pk.lon), 5),
            "elevation": round(float(pk.elevation), 1), "slope": round(float(pk.slope), 2), "elev_std": round(float(pk.elev_std), 1),
            "bundesland": pk.bundesland, "landkreis": pk.landkreis, "operator": pk.operator, "n_operators": int(pk.n_operators),
            "eeg_tender": pk.get("eeg_tender"), "n_lokationen": int(pk.n_lokationen), "merged_ids": pk.merged_ids,
            "foreign_within_km": pk.foreign_within_km, "unit_ids": unit_ids(pk), "gemeinde": pk.gemeinde}


def unit_ids(pk: pd.Series) -> str:
    """MaStR unit ids of the park's Lokationen (park_id and merged_ids are Lokation numbers)."""
    global _U
    if "_U" not in globals():
        _U = pd.read_parquet(os.path.join(os.environ.get("DATA_ROOT", "/mnt/lambda1/nvme1"), "synthetic/solar/mastr/solar_units_20250625.parquet"),
                             columns=["EinheitMastrNummer", "LokationMaStRNummer"])
    loks = {pk.park_id} | set(str(pk.merged_ids).split(";")) if pd.notna(pk.merged_ids) else {pk.park_id}
    return ";".join(sorted(_U[_U.LokationMaStRNummer.isin(loks)].EinheitMastrNummer))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replace", default="")
    ap.add_argument("--assign", default=None, help="CSV old,new,reason: replacements given, no search")
    ap.add_argument("--src", default="v1")
    ap.add_argument("--dst", default="v2")
    ap.add_argument("--reason", default="MaStR coordinate outside the registered municipality (Nominatim check 2026-10-22)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-tries", type=int, default=25)
    a = ap.parse_args()
    sel = pd.read_csv(os.path.join(FL, "mastr", f"pv_park_selection_{a.src}.csv"))
    split = pd.read_csv(os.path.join(FL, f"pv_transfer_split_{a.src}.csv"))
    pool = pd.read_csv(os.path.join(FL, "mastr", "pv_park_pool_v1.csv"))
    pool = pool[pool.reject.isna()]
    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    rng = np.random.default_rng(a.seed)
    gem = pool.set_index("park_id").gemeinde
    sel["gemeinde"] = sel.park_id.map(gem)
    changes, used, osm_new = [], set(sel.park_id), {}
    given = pd.read_csv(a.assign).set_index("old") if a.assign else None
    for pid in (given.index if given is not None else a.replace.split(",")):
        old = sel[sel.park_id == pid].iloc[0]
        if given is not None:
            new = pool[pool.park_id == given.loc[pid, "new"]].iloc[0]
            assert new.park_id not in used, f"{new.park_id} already in the selection"
            diff = [k for k in ("cls", "region", "cohort", "size_bin") if new[k] != old[k]]
            tried, reason = [f"assigned; stratum differs in {diff}" if diff else "assigned"], given.loc[pid, "reason"]
        else:
            new, tried, reason = search(old, pool, used, cache, rng, a, osm_new)
        used.add(new.park_id)
        row = selection_row(old, new)
        sel = pd.concat([sel[sel.park_id != pid], pd.DataFrame([row])], ignore_index=True)
        so = split[split.park_id == pid].iloc[0].to_dict()
        so.update(park_id=new.park_id, lat=row["lat"], lon=row["lon"], terrain_cls=new.terrain_cls, size_bin=new.size_bin, cohort=new.cohort)
        split = pd.concat([split[split.park_id != pid], pd.DataFrame([so])], ignore_index=True)
        changes.append(dict(client=old.client, cls=old.cls, role=so["role"], fold=so["fold"], old=pid, old_gemeinde=old.gemeinde, new=new.park_id,
                            new_gemeinde=new.gemeinde, new_state=new.bundesland, reason=reason, candidates_tested=len(tried), tried=" ".join(tried),
                            old_terrain=old.terrain_cls, new_terrain=new.terrain_cls, old_cohort=old.cohort, new_cohort=new.cohort,
                            old_size_bin=old.size_bin, new_size_bin=new.size_bin))
    order = pd.read_csv(os.path.join(FL, "mastr", f"pv_park_selection_{a.src}.csv")).park_id.tolist()
    rep = {c["old"]: c["new"] for c in changes}
    key = {(rep.get(p, p)): k for k, p in enumerate(order)}
    sel = sel.sort_values("park_id", key=lambda s: s.map(key)).reset_index(drop=True)
    split = split.sort_values("park_id", key=lambda s: s.map(key)).reset_index(drop=True)
    for path, df in ((os.path.join(FL, "mastr", f"pv_park_selection_{a.dst}.csv"), sel), (os.path.join(FL, f"pv_transfer_split_{a.dst}.csv"), split),
                     (os.path.join(FL, "mastr", f"selection_{a.dst}_changes.csv"), pd.DataFrame(changes))):
        if os.path.exists(path):
            raise SystemExit(f"{path} exists - never overwritten")
        df.to_csv(path, index=False)
        print("written", path)
    if osm_new:
        from parks_pv import paths                                # Overpass answers of the new parks into the shared OSM cache (keys added only)
        osm_path = os.path.join(paths.INPUT_DIR, "osm", "parks_osm.json")
        osm = json.load(open(osm_path))
        osm.update({k: v for k, v in osm_new.items() if k not in osm})
        json.dump(osm, open(osm_path, "w"))
    print(pd.DataFrame(changes)[["client", "cls", "role", "old", "old_gemeinde", "new", "new_gemeinde", "new_state", "candidates_tested"]].to_string(index=False))


def search(old: pd.Series, pool: pd.DataFrame, used: set, cache: dict, rng, a, osm_new: dict) -> tuple:
    """(pool row, tried list, reason) of the first candidate of old's stratum passing the coordinate and OSM checks."""
    pid = old.park_id
    c = pool[(pool.cls == old.cls) & (pool.region == old.region) & (pool.cohort == old.cohort) & (pool.size_bin == old.size_bin) & ~pool.park_id.isin(used)].copy()
    c["key"] = (~c.bundesland.isin(FETCH_STATES)).astype(int) * 2 + (c.terrain_cls != old.terrain_cls).astype(int) + rng.random(len(c)) * 0.5
    tried, new = [], None
    for pk in c.sort_values("key").head(a.max_tries).itertuples():
        ok, txt = in_municipality(pk, reverse(pk.lat, pk.lon, cache))
        n_osm, ans = osm_plant_near(pk.lat, pk.lon) if ok else (-1, None)
        tried.append(f"{pk.park_id}:{'ok' if ok else 'coord'}:{n_osm}")
        print(pid, "candidate", pk.park_id, pk.bundesland, pk.gemeinde, "| coordinate:", txt, "| OSM plants:", n_osm, flush=True)
        if ok and n_osm > 0:
            new = pool[pool.park_id == pk.park_id].iloc[0]
            osm_new[pk.park_id] = ans
            break
    if new is None:
        raise SystemExit(f"{pid}: no candidate passed ({len(c)} in the stratum, tried {tried})")
    return new, tried, a.reason


if __name__ == "__main__":
    main()
