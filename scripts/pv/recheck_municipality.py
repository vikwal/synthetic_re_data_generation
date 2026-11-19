"""Second look at the municipality check of prescreen_pool.py (2026-11-05, Viktor: do not drop parks with potential).
1. Name match again on the cached reverse answers with the tolerant replace_parks._norm (MaStR abbreviations
   "a.d.", "b.", "i.d. OPf." expanded, administrative words only as whole words).
2. Parks still without a match: the registered municipality is looked up by name (Nominatim search, boundary polygon,
   1 request/s, own cache) and the distance of the register point to its polygon is measured. Inside or within
   BORDER_M the park passes with the flag "border", farther away the coordinate stays rejected.
Writes prescreen_v2.csv and summary_v2.csv next to prescreen.csv (the first version stays)."""
import argparse
import json
import os
import sys
import time
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [REPO, os.path.join(REPO, "scripts", "pv")]
import replace_parks as RP  # noqa: E402

BORDER_M = 1000.0
SEARCH_CACHE = os.path.join(os.path.dirname(RP.CACHE), "_nominatim_search_cache.json")
ADMIN = {"municipality", "city", "town", "village", "suburb", "borough", "hamlet"}


def search(gemeinde: str, landkreis: str, state: str, cache: dict) -> list:
    """Nominatim search hits (with GeoJSON polygon) for a municipality name, cached by query."""
    q = f"{gemeinde}, {landkreis}, {state}"
    if q not in cache:
        p = urllib.parse.urlencode(dict(q=q, countrycodes="de", format="jsonv2", polygon_geojson=1, polygon_threshold=0.0005, limit=5))
        with urllib.request.urlopen(urllib.request.Request("https://nominatim.openstreetmap.org/search?" + p, headers={"User-Agent": RP.UA}), timeout=60) as f:
            cache[q] = json.load(f)
        time.sleep(1.1)
    return cache[q]


def border_distance(r, hits: list) -> tuple:
    """(metres from the register point to the nearest matching municipality polygon (0 inside), name of the hit)."""
    from pyproj import Transformer
    from shapely.geometry import Point, shape
    from shapely.ops import transform
    zone = 32 if r.lon < 12.0 else 33
    to_m = Transformer.from_crs("EPSG:4326", f"EPSG:258{zone}", always_xy=True).transform
    p = Point(*to_m(r.lon, r.lat))
    best = (np.nan, "")
    g = RP._norm(r.gemeinde)
    for h in hits:
        gj = h.get("geojson", {})
        if gj.get("type") not in ("Polygon", "MultiPolygon") or h.get("addresstype") not in ADMIN:
            continue
        n = RP._norm(h.get("name"))
        if not n or not (n == g or n in g or g in n):
            continue
        d = transform(to_m, shape(gj)).distance(p)
        if not np.isfinite(best[0]) or d < best[0]:
            best = (d, h.get("display_name", "")[:80])
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(REPO, "results", "pv", "pool_prescreen_2026-11-05"))
    a = ap.parse_args()
    df = pd.read_csv(os.path.join(a.dir, "prescreen.csv"), index_col=0)
    pool = pd.read_csv(os.path.join(RP.FL, "mastr", "pv_park_pool_v1.csv")).set_index("park_id")
    rev = json.load(open(RP.CACHE))
    ok2 = {pid: RP.in_municipality(pool.loc[pid], rev[f"{pool.loc[pid].lat:.5f},{pool.loc[pid].lon:.5f}"])[0] for pid in df.index}
    df["coord_ok_v1"] = df.coord_ok
    df["coord_ok"] = pd.Series(ok2)
    df["coord_rule"] = np.where(df.coord_ok_v1, "name", np.where(df.coord_ok, "name (tolerant)", ""))
    df["border_m"], df["border_hit"] = np.nan, ""
    cache = json.load(open(SEARCH_CACHE)) if os.path.exists(SEARCH_CACHE) else {}
    todo = df.index[~df.coord_ok]
    print(len(df), "parks;", int((df.coord_ok & ~df.coord_ok_v1).sum()), "pass with the tolerant name match;", len(todo), "to search", flush=True)
    for i, pid in enumerate(todo):
        r = pool.loc[pid]
        for k in range(5):
            try:
                d, hit = border_distance(r, search(r.gemeinde, r.landkreis, r.bundesland, cache))
                break
            except Exception as e:  # noqa: BLE001
                print(pid, "nominatim", str(e)[:80], flush=True)
                time.sleep(30)
        else:
            d, hit = np.nan, "nominatim unavailable"
        df.loc[pid, ["border_m", "border_hit"]] = [round(d, 0) if np.isfinite(d) else np.nan, hit]
        if d <= BORDER_M:
            df.loc[pid, ["coord_ok", "coord_rule"]] = [True, "inside" if d == 0 else "border"]
        if i % 50 == 0:
            json.dump(cache, open(SEARCH_CACHE, "w"))
            print("searched", i, "/", len(todo), flush=True)
    json.dump(cache, open(SEARCH_CACHE, "w"))
    df["stage"] = np.where(~df.coord_ok, np.where(df.border_m.notna(), "coordinate outside municipality", "municipality not found"),
                           np.where(df.n_osm > 0, "passed (OSM outline)", "passed (box, no OSM plant)"))
    df.to_csv(os.path.join(a.dir, "prescreen_v2.csv"))
    passed = df[df.stage.str.startswith("passed")]
    keys = ["cls", "cohort", "size_bin", "region"]
    summ = pd.concat([df.groupby(k).agg(pool=("stage", "size"), coord_ok=("coord_ok", "sum"),
                                        osm_ok=("stage", lambda s: int((s == "passed (OSM outline)").sum())),
                                        box_ok=("stage", lambda s: int((s == "passed (box, no OSM plant)").sum())))
                      .assign(clean=passed.groupby(k)[["flag_area", "flag_years"]].apply(lambda t: int((t.fillna("") == "").all(axis=1).sum())))
                      .fillna(0).astype(int).rename_axis("value").reset_index().assign(field=k) for k in keys])
    summ.to_csv(os.path.join(a.dir, "summary_v2.csv"), index=False)
    print(df.stage.value_counts().to_string())
    print(df.coord_rule.value_counts().to_string())


if __name__ == "__main__":
    main()
