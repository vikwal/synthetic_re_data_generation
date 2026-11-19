#!/usr/bin/env python
"""Match the Trianel parks to MaStR locations by CAPACITY, not by coordinate (Viktor 2026-10-20: both coordinate
sources can be wrong by kilometres, pv1 register point 27 km off, pv6 Trianel point 3 km off; names carry the
town and recur at neighbours). Candidates = locations (LokationMaStRNummer) of ground-mounted units in operation
whose DC sum or AC sum lies within +-tol of the Trianel 'Installierte Leistung'; also pairs of such locations
within pair_km of each other (two meters = two plants, pv6). Distance and name are reported, not used to filter.

    synthre/bin/python scripts/pv/trianel_mastr_match.py [--tol 0.03] [--pair-km 3]
Output: ${DATA_ROOT}/synthetic/solar/trianel_benchmark/inputs/mastr/mastr_match.csv (one row per candidate, best first)
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts", "pv"))
import trianel_common as TC  # noqa: E402

ROOT = os.environ.get("DATA_ROOT", "/mnt/lambda1/nvme1")
OUT = os.path.join(ROOT, "synthetic/solar/trianel_benchmark/inputs/mastr/mastr_match.csv")


def hav(lat1, lon1, lat2, lon2):
    p = np.pi / 180
    a = np.sin((lat2 - lat1) * p / 2) ** 2 + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tol", type=float, default=0.03, help="relative tolerance of the capacity match")
    ap.add_argument("--pair-km", type=float, default=3.0, help="max distance between two locations forming one park")
    ap.add_argument("--all-types", action="store_true", help="do not restrict to Freiflaeche (pv5 is registered as Bauliche Anlagen (Sonstige))")
    ap.add_argument("--near-km", type=float, default=50.0, help="report the candidates within this distance separately")
    a = ap.parse_args()
    parks = TC.parks_table(TC.load_masterdata())
    u = pd.read_parquet(f"{ROOT}/synthetic/solar/mastr/solar_units_20250625.parquet")
    u = u[u.EinheitBetriebsstatus == "In Betrieb"]
    if not a.all_types:
        u = u[u.ArtDerSolaranlage == "Freifläche"]
    loc = u.groupby("LokationMaStRNummer").agg(dc=("Bruttoleistung", "sum"), ac=("Nettonennleistung", "sum"), n_units=("EinheitMastrNummer", "size"),
                                              lat=("Breitengrad", "median"), lon=("Laengengrad", "median"), gemeinde=("Gemeinde", "first"),
                                              name=("NameStromerzeugungseinheit", "first"), year=("Inbetriebnahmedatum", "min"),
                                              orient=("Hauptausrichtung", "first"), tilt=("HauptausrichtungNeigungswinkel", "first"),
                                              modules=("AnzahlModule", "sum")).reset_index()
    print(f"{len(loc)} locations in operation ({'all types' if a.all_types else 'ground-mounted'}), {len(u)} units; tol {a.tol}")
    rows = []
    for pv, p in parks.iterrows():
        kw = float(p.kw_master)
        loc["d_km"] = hav(p.lat, p.lon, loc.lat.values, loc.lon.values)
        # single locations
        m = (np.abs(loc.dc - kw) <= a.tol * kw) | (np.abs(loc.ac - kw) <= a.tol * kw)
        for r in loc[m].itertuples():
            rows.append(dict(pv=pv, kind="single", locations=r.LokationMaStRNummer, dc=r.dc, ac=r.ac, n_units=r.n_units,
                             match=("dc" if abs(r.dc - kw) <= a.tol * kw else "ac"), err_pct=100 * min(abs(r.dc - kw), abs(r.ac - kw)) / kw,
                             d_km=r.d_km, lat=r.lat, lon=r.lon, gemeinde=r.gemeinde, name=r.name, year=str(r.year)[:4], orient=r.orient, tilt=r.tilt, modules=r.modules))
        # pairs of locations close to each other (two plants, two meters)
        big = loc[(loc.dc >= 0.15 * kw) & (loc.dc <= kw * (1 + a.tol)) & loc.lat.notna() & loc.lon.notna()].reset_index(drop=True)
        from scipy.spatial import cKDTree
        xy = np.column_stack([big.lat.values * 111.0, big.lon.values * 111.0 * np.cos(np.radians(big.lat.values))])   # km, lokal eben
        pairs = cKDTree(xy).query_pairs(a.pair_km)
        for i, j in sorted(pairs):
            ri, rj = big.iloc[i], big.iloc[j]
            if True:
                dc, ac = ri.dc + rj.dc, ri.ac + rj.ac
                if abs(dc - kw) <= a.tol * kw or abs(ac - kw) <= a.tol * kw:
                    rows.append(dict(pv=pv, kind="pair", locations=f"{ri.LokationMaStRNummer}+{rj.LokationMaStRNummer}", dc=dc, ac=ac, n_units=ri.n_units + rj.n_units,
                                     match=("dc" if abs(dc - kw) <= a.tol * kw else "ac"), err_pct=100 * min(abs(dc - kw), abs(ac - kw)) / kw,
                                     d_km=min(ri.d_km, rj.d_km), lat=(ri.lat + rj.lat) / 2, lon=(ri.lon + rj.lon) / 2, gemeinde=ri.gemeinde,
                                     name=f"{ri.name} + {rj.name}", year=str(min(ri.year, rj.year))[:4], orient=f"{ri.orient}/{rj.orient}", tilt=f"{ri.tilt}/{rj.tilt}", modules=ri.modules + rj.modules))
    df = pd.DataFrame(rows).sort_values(["pv", "err_pct", "d_km"]).reset_index(drop=True)
    df["kw_trianel"] = df.pv.map(parks.kw_master)
    df.to_csv(OUT, index=False)
    pd.set_option("display.width", 260)
    pd.set_option("display.max_colwidth", 38)
    for pv in parks.index:
        s = df[df.pv == pv]
        print(f"\n== {pv}: Trianel {parks.loc[pv].kw_master:.0f} kW, {len(s)} candidate(s) within +-{100 * a.tol:.0f} %")
        near = s[s.d_km <= a.near_km].sort_values("d_km")
        print(f"   within {a.near_km:.0f} km: {len(near)}  (next candidate beyond: {s[s.d_km > a.near_km].d_km.min():.0f} km)")
        print(near[["kind", "match", "err_pct", "dc", "ac", "n_units", "d_km", "gemeinde", "name", "year", "orient", "tilt"]].head(5).round(2).to_string(index=False))
    print("\nwritten", OUT)


if __name__ == "__main__":
    main()
