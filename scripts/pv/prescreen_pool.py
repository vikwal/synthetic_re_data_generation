"""Pre-screen of the whole park pool without geodata downloads (2026-11-05): the first two checks of
screen_replacements.py for every usable pool park in a state with a fetcher, plus two cheap flags.

  1. Coordinate in the registered municipality (Nominatim reverse, cache of replace_parks.py, 1 request/s).
  2. OSM solar plant (power=plant) within 400 m of the park point, counted as in screen_replacements.py
     (pv.geodata_windows.osm_footprint). The plants come from one Overpass query per state (cached), not one per park.
     Not a filter (Viktor 2026-11-05): parks without a plant are kept as "box" candidates, the window method measures
     them in the box around the register point (E4b); the OSM plant only confirms that a park stands there.
  Flags (not filters):
  - area_ratio: area of the OSM plants within 400 m over the fenced area expected from capacity and year
    (parks_pv.geometry_estimate.expected_fenced_m2); < 0.5 hints at a partial outline, > 2 at neighbouring plants.
  - year_spread: years between the first and the last MaStR unit of the park (merged Lokationen, SEL961128163827:
    2011 + 2023); > 3 marks a park of several construction phases.
The survey-date check and the window measurement need the tiles and stay with screen_replacements.py.
Output: results/pv/pool_prescreen_<date>/prescreen.csv and summary.csv."""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import requests

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [REPO, os.path.join(REPO, "scripts", "pv")]
from parks_pv import geometry_estimate as GRS  # noqa: E402
import replace_parks as RP  # noqa: E402
from parks_pv import paths  # noqa: E402
from pv import geodata_windows as GW, site_inputs  # noqa: E402

ISO = {"Bayern": "DE-BY", "Niedersachsen": "DE-NI", "Mecklenburg-Vorpommern": "DE-MV", "Brandenburg": "DE-BB", "Nordrhein-Westfalen": "DE-NW",
       "Hessen": "DE-HE", "Sachsen": "DE-SN", "Sachsen-Anhalt": "DE-ST", "Baden-Württemberg": "DE-BW", "Rheinland-Pfalz": "DE-RP"}
RADIUS_M = 400.0


def _overpass(q: str, label: str, attempts: int):
    """Elements of one Overpass query (mirrors and retries), None when no mirror answers."""
    for i in range(attempts):
        url = site_inputs.OVERPASS_URLS[i % len(site_inputs.OVERPASS_URLS)]
        try:
            r = requests.post(url, data={"data": q}, headers=site_inputs.UA, timeout=900)
            if r.status_code == 200 and "elements" in r.json():
                return r.json()["elements"]
            print(label, "overpass status", r.status_code, url, flush=True)
        except Exception as e:  # noqa: BLE001
            print(label, "overpass error", str(e)[:100], flush=True)
        time.sleep(30 + 15 * i)
    return None


def _plants_query(scope: str, timeout: int = 600) -> str:
    return (f'[out:json][timeout:{timeout}][maxsize:1073741824];{scope[0]}\n(\n  way["power"="plant"]["plant:source"="solar"]{scope[1]};\n'
            f'  relation["power"="plant"]["plant:source"="solar"]{scope[1]};\n);\nout tags geom;')


def state_plants(state: str, cache_dir: str, pts: pd.DataFrame, attempts: int = 5, cell: float = 0.5) -> list:
    """All OSM solar plants (ways and relations with geometry) of one state, cached as JSON. When the state query
    times out (Nordrhein-Westfalen 2026-11-05: 504 on all mirrors), one bbox query per `cell`-degree cell with pool parks."""
    f = os.path.join(cache_dir, f"overpass_plants_{ISO[state]}.json")
    if os.path.exists(f):
        return json.load(open(f))["elements"]
    el = _overpass(_plants_query((f'\narea["ISO3166-2"="{ISO[state]}"]->.a;', "(area.a)")), state, attempts)
    if el is None:
        el, seen = [], set()
        cells = sorted({(np.floor(la / cell) * cell, np.floor(lo / cell) * cell) for la, lo in zip(pts.lat, pts.lon)})
        for la, lo in cells:
            bb = f"({la - 0.01:.3f},{lo - 0.01:.3f},{la + cell + 0.01:.3f},{lo + cell + 0.01:.3f})"
            part = _overpass(_plants_query(("", bb), 300), f"{state} cell {la},{lo}", 8)
            if part is None:
                raise SystemExit(f"{state}: Overpass unavailable - stopped instead of counting parks as without OSM plant")
            el += [e for e in part if (e["type"], e["id"]) not in seen]
            seen |= {(e["type"], e["id"]) for e in part}
        print(state, len(cells), "cell queries", flush=True)
    json.dump({"elements": el}, open(f, "w"))
    return el


def year_spread(pool: pd.DataFrame) -> pd.Series:
    """Years between first and last commissioning of the MaStR units of each pool park (Lokationen park_id + merged_ids)."""
    u = pd.read_parquet(os.path.join(paths.MASTR_DIR, "solar_units_20250625.parquet"), columns=["LokationMaStRNummer", "Inbetriebnahmedatum"])
    u["y"] = pd.to_datetime(u.Inbetriebnahmedatum, errors="coerce").dt.year
    y = u.groupby("LokationMaStRNummer").y.agg(["min", "max"])
    out = {}
    for r in pool.itertuples():
        loks = [r.park_id] + (str(r.merged_ids).split(";") if pd.notna(r.merged_ids) else [])
        t = y.reindex(loks).dropna()
        out[r.park_id] = int(t["max"].max() - t["min"].min()) if len(t) else np.nan
    return pd.Series(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", default="pv_park_selection_v3.csv")
    ap.add_argument("--out", default=os.path.join(REPO, "results", "pv", f"pool_prescreen_{time.strftime('%Y-%m-%d')}"))
    ap.add_argument("--no-geocode", action="store_true", help="skip the Nominatim step (OSM check only)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    pool = pd.read_csv(os.path.join(RP.FL, "mastr", "pv_park_pool_v1.csv"))
    sel = pd.read_csv(os.path.join(RP.FL, "mastr", a.selection))
    pool = pool[pool.reject.isna() & pool.bundesland.isin(ISO) & ~pool.park_id.isin(sel.park_id)].copy()
    print(len(pool), "pool parks in states with a fetcher, not in", a.selection, flush=True)
    from pyproj import Transformer
    from shapely.geometry import Point
    rows = []
    for state, g in pool.groupby("bundesland"):
        el = state_plants(state, a.out, g)
        print(state, len(g), "parks,", len(el), "OSM plants", flush=True)
        zone = 32 if g.lon.median() < 12.0 else 33                     # one zone per state is enough for distances of 400 m
        to_m = Transformer.from_crs("EPSG:4326", f"EPSG:258{zone}", always_xy=True).transform
        _, fences = GW.osm_footprint(el, [Point(0, 0)], to_m, 0.0)     # all plants of the state in metres (distances recomputed below)
        from shapely.strtree import STRtree
        tree = STRtree(list(fences.geom)) if len(fences) else None
        for r in g.itertuples():
            p = Point(*to_m(r.lon, r.lat))
            idx = tree.query(p.buffer(RADIUS_M)) if tree is not None else []
            near = [fences.geom.iloc[i] for i in idx if fences.geom.iloc[i].distance(p) <= RADIUS_M]
            area = sum(x.area for x in near) / 1e4
            dmin = min((fences.geom.iloc[i].distance(p) for i in idx), default=np.nan)
            exp = GRS.expected_fenced_m2(r.dc_kwp, r.year) / 1e4
            rows.append(dict(park_id=r.park_id, state=state, cls=r.cls, region=r.region, cohort=r.cohort, size_bin=r.size_bin, year=int(r.year),
                             dc_kwp=round(float(r.dc_kwp), 1), area_ha_mastr=r.area_ha, n_osm=len(near), dist_osm_m=round(dmin, 0) if np.isfinite(dmin) else np.nan,
                             osm_ha=round(area, 2), expected_ha=round(exp, 2), area_ratio=round(area / exp, 2) if near else np.nan))
    df = pd.DataFrame(rows).set_index("park_id")
    df["year_spread"] = year_spread(pool.set_index("park_id").loc[df.index].reset_index())
    df["coord_ok"], df["coord_txt"] = pd.NA, ""
    if not a.no_geocode:
        cache = json.load(open(RP.CACHE)) if os.path.exists(RP.CACHE) else {}
        pk = pool.set_index("park_id")
        todo = df.index
        for i, pid in enumerate(todo):
            r = pk.loc[pid]
            for k in range(5):
                try:
                    ok, txt = RP.in_municipality(r, RP.reverse(r.lat, r.lon, cache))
                    break
                except Exception as e:  # noqa: BLE001
                    print(pid, "nominatim", str(e)[:80], flush=True)
                    time.sleep(30)
            else:
                ok, txt = pd.NA, "nominatim unavailable"
            df.loc[pid, ["coord_ok", "coord_txt"]] = [ok, txt]
            if i % 200 == 0:
                print("geocoded", i, "/", len(todo), flush=True)
                df.to_csv(os.path.join(a.out, "prescreen.csv"))
    ok = df.coord_ok.astype("boolean")
    df["stage"] = np.where(ok.isna(), "geocode missing", np.where(~ok.fillna(False), "coordinate outside municipality",
                           np.where(df.n_osm > 0, "passed (OSM outline)", "passed (box, no OSM plant)")))
    df["flag_area"] = np.where(df.area_ratio < 0.5, "outline small", np.where(df.area_ratio > 2.0, "outline large", ""))
    df["flag_years"] = np.where(df.year_spread > 3, "phases > 3 a", "")
    df.to_csv(os.path.join(a.out, "prescreen.csv"))
    passed = df[df.stage.str.startswith("passed")]
    keys = ["cls", "cohort", "size_bin", "region"]
    summ = pd.concat([df.groupby(k).agg(pool=("stage", "size"), osm=("n_osm", lambda s: int((s > 0).sum())),
                                        coord_ok=("stage", lambda s: int(s.str.startswith("passed").sum())),
                                        osm_ok=("stage", lambda s: int((s == "passed (OSM outline)").sum())),
                                        box_ok=("stage", lambda s: int((s == "passed (box, no OSM plant)").sum())))
                      .assign(clean=passed.groupby(k).apply(lambda t: int(((t.flag_area == "") & (t.flag_years == "")).sum())))
                      .fillna(0).astype(int).rename_axis("value").reset_index().assign(field=k) for k in keys])
    summ.to_csv(os.path.join(a.out, "summary.csv"), index=False)
    print(df.stage.value_counts().to_string())
    print(summ.to_string(index=False))


if __name__ == "__main__":
    main()
