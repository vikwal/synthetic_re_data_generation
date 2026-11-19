#!/usr/bin/env python
"""Terrain analysis of the Trianel PV parks: footprint (OSM polygons or an area buffer), GLO-30
slope/aspect, effective tilt/azimuth distribution of terrain-following rows, k-cluster sub-arrays,
DEM vs PVGIS horizon. Anonymised outputs (pv1..pv7) only.

    python scripts/pv/terrain_trianel.py --osm-json <overpass.json> --horizon-json <pvgis.json> --out <dir>
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pv import arrays as arr, terrain  # noqa: E402
import trianel_common as tc  # noqa: E402

plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e6e5e1", "grid.linewidth": 0.6, "font.size": 8.5, "legend.frameon": False, "savefig.bbox": "tight"})
DENSITY_KWP_HA = {2010: 450, 2012: 500, 2014: 600, 2016: 700, 2018: 850, 2020: 1000, 2022: 1150, 2024: 1250}


def density_for(year: int) -> float:
    ys = sorted(DENSITY_KWP_HA)
    return float(np.interp(year, ys, [DENSITY_KWP_HA[y] for y in ys]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--osm-json", required=True)
    ap.add_argument("--horizon-json", required=True)
    ap.add_argument("--mastr-cand", default=None)
    ap.add_argument("--mastr-xml", default=None)
    ap.add_argument("--out", default=os.path.expanduser("~/Work/FL_Contribution/reports/figs_pv_generator"))
    ap.add_argument("--k", type=int, default=4)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    m = tc.load_masterdata()
    parks = tc.parks_table(m)
    meter = tc.load_meter_hourly(parks)
    facts = tc.mastr_facts(parks, args.mastr_cand, args.mastr_xml)
    par = tc.park_parameters(parks, facts, meter)
    osm = json.load(open(args.osm_json))
    pvg = json.load(open(args.horizon_json))
    rows, tables, clusters, horizons = [], {}, {}, {}
    for pv, p in parks.iterrows():
        els = osm.get(pv, {}).get("elements", [])
        poly = terrain.pick_plant_near(els, p["lat"], p["lon"], 2.5) if els else None
        src = "osm"
        if poly is None:
            ha = float(facts.loc[pv, "area_ha"]) if pd.notna(facts.loc[pv, "area_ha"]) else par.loc[pv, "kwp_dc"] / density_for(int(par.loc[pv, "year"]))
            poly = terrain.buffer_polygon(p["lat"], p["lon"], ha)
            src = "mastr_area" if pd.notna(facts.loc[pv, "area_ha"]) else "density_buffer"
        t = terrain.terrain_table(poly, p["tilt_master"], p["az_master"])
        tables[pv] = t
        cl = arr.cluster_arrays(t["tilt_eff"].values, t["az_eff"].values, t["area_m2"].values, args.k)
        clusters[pv] = cl
        te, ae = arr.mean_angles(t["tilt_eff"].values, t["az_eff"].values, t["area_m2"].values)
        hz_dem = terrain.dem_horizon(p["lat"], p["lon"])
        hz_pv = pd.Series({float(k): float(v) for k, v in pvg[pv].items()}).sort_index()
        horizons[pv] = (hz_dem, hz_pv)
        rows.append({"pv": pv, "footprint_src": src, "footprint_ha": terrain.area_ha(poly), "n_cells": len(t),
                     "kwp_per_ha": par.loc[pv, "kwp_dc"] / terrain.area_ha(poly),
                     "elev_min": t["elev"].min(), "elev_max": t["elev"].max(), "slope_mean": t["slope"].mean(), "slope_p90": t["slope"].quantile(0.9),
                     "slope_max": t["slope"].max(), "tilt_eff_mean": te, "az_eff_mean": ae, "tilt_eff_std": t["tilt_eff"].std(),
                     "az_eff_std": t["az_eff"].std(), "tilt_master": p["tilt_master"], "az_master": p["az_master"],
                     "cluster_arrays": "; ".join(f"{c.tilt:.1f}°/{c.azimuth:.0f}°×{c.share:.2f}" for c in cl),
                     "horizon_dem_max": hz_dem.max(), "horizon_dem_mean": hz_dem.mean(), "horizon_pvgis_max": hz_pv.max(), "horizon_pvgis_mean": hz_pv.mean(),
                     "horizon_dem_S": float(hz_dem.loc[180.0]) if 180.0 in hz_dem.index else np.nan, "horizon_pvgis_S": float(hz_pv.loc[180.0]) if 180.0 in hz_pv.index else np.nan,
                     "sky_view_dem": 1 - np.mean(np.sin(np.radians(hz_dem.values)) ** 2), "sky_view_pvgis": 1 - np.mean(np.sin(np.radians(hz_pv.values)) ** 2)})
    df = pd.DataFrame(rows).set_index("pv")
    df.to_csv(os.path.join(args.out, "terrain_parks.csv"))

    pvs = list(df.index)
    fig, axes = plt.subplots(2, len(pvs), figsize=(2.2 * len(pvs), 5.2))
    for j, pv in enumerate(pvs):
        t = tables[pv]
        ax = axes[0, j]
        ax.hist(t["slope"], bins=np.linspace(0, 15, 31), weights=t["area_m2"] / t["area_m2"].sum(), color="#2a78d6")
        ax.set_title(f"{pv} ({df.loc[pv, 'footprint_src']}, {df.loc[pv, 'footprint_ha']:.0f} ha)", loc="left", fontsize=7.5)
        ax.set_xlabel("Hangneigung (°)")
        ax = axes[1, j]
        ax.scatter(t["az_eff"], t["tilt_eff"], s=3, alpha=0.3, color="#1baf7a")
        for c in clusters[pv]:
            ax.scatter(c.azimuth, c.tilt, s=200 * c.share + 10, color="#4a3aa7", marker="o", edgecolor="white")
        ax.plot(df.loc[pv, "az_master"], df.loc[pv, "tilt_master"], marker="x", color="#e34948", ms=8, mew=2)
        ax.set_xlabel("effektiver Azimut (°)")
        if j == 0:
            axes[0, j].set_ylabel("Flächenanteil")
            ax.set_ylabel("effektive Neigung (°)")
    fig.suptitle("Gelände je Park: Hangneigung (oben) und effektive Reihenwinkel bei Stammdaten-Montagewinkel (unten; Kreise = k-Cluster, × = Stammdaten)", x=0.01, ha="left", fontsize=9)
    fig.savefig(os.path.join(args.out, "fig_terrain_angles.png"))
    plt.close(fig)

    fig, axes = plt.subplots(1, len(pvs), figsize=(2.2 * len(pvs), 2.4), sharey=True)
    for ax, pv in zip(axes, pvs):
        d, pg = horizons[pv]
        ax.plot(d.index, d.values, color="#2a78d6", label="DEM 90 m")
        ax.plot(pg.index, pg.values, color="#eb6834", label="PVGIS")
        ax.set_title(pv, loc="left")
        ax.set_xlabel("Azimut (°)")
    axes[0].set_ylabel("Horizonthöhe (°)")
    axes[0].legend(fontsize=7)
    fig.suptitle("Horizontprofile: eigenes DEM-Raymarching vs. PVGIS", x=0.01, ha="left", fontsize=10)
    fig.savefig(os.path.join(args.out, "fig_horizon.png"))
    plt.close(fig)
    print(df[["footprint_src", "footprint_ha", "kwp_per_ha", "slope_mean", "slope_p90", "tilt_eff_std", "az_eff_std", "horizon_dem_max", "horizon_pvgis_max"]].round(2).to_string())


if __name__ == "__main__":
    main()
