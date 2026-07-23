#!/usr/bin/env python3
"""WP0 — distance to coast for all DWD stations.

Natural Earth 10m coastline, clipped to a North/Baltic Sea bbox, reprojected
to EPSG:3035 (LAEA Europe, metric). Distance = min distance station -> any
coastline geometry, in km. 'Dcoast' is part of the candidate static-feature
pool in the reference implementation (TRWindBC computeGBFeatureImp.py).

Run with the synthre venv:
    synthre/bin/python scripts/era5_bc/wp0_distance_to_coast.py

Output: data/era5_bc/dist_coast.csv  (station_id, dist_coast_km)
"""

import io
import os
import sys
import urllib.request
import zipfile

import geopandas as gpd
import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
from era5_bc.config import load_config  # noqa: E402

NE_URL = ("https://naciscdn.org/naturalearth/10m/physical/"
          "ne_10m_coastline.zip")
# generous bbox around Germany incl. North Sea, Baltic incl. Danish straits
BBOX = (-5.0, 45.0, 25.0, 62.0)  # lon_min, lat_min, lon_max, lat_max
CRS_METRIC = "EPSG:3035"


def fetch_coastline(cache_dir: str) -> gpd.GeoDataFrame:
    shp = os.path.join(cache_dir, "ne_10m_coastline.shp")
    if not os.path.exists(shp):
        os.makedirs(cache_dir, exist_ok=True)
        print(f"downloading {NE_URL}")
        with urllib.request.urlopen(NE_URL) as resp:
            zipfile.ZipFile(io.BytesIO(resp.read())).extractall(cache_dir)
    coast = gpd.read_file(shp)
    coast = coast.cx[BBOX[0]:BBOX[2], BBOX[1]:BBOX[3]]
    assert len(coast) > 0, "empty coastline after bbox clip"
    return coast.to_crs(CRS_METRIC)


def main():
    cfg = load_config()
    coast = fetch_coastline(cfg["paths"]["coastline_dir"])
    coast_union = coast.geometry.union_all()

    master = pd.read_csv(cfg["paths"]["stations_master"],
                         dtype={"station_id": str})
    master["station_id"] = master["station_id"].str.zfill(5)
    locs = master[["station_id", "longitude", "latitude"]]
    # park centroids (wp9 downscaling) — ids 'park_*', no station-id clash
    parks = (pd.read_csv(cfg["paths"]["topo_features"],
                         dtype={"location_id": str})
             .query("kind == 'park'")
             .rename(columns={"location_id": "station_id"})
             [["station_id", "longitude", "latitude"]])
    locs = pd.concat([locs, parks], ignore_index=True)
    pts = gpd.GeoDataFrame(
        locs[["station_id"]],
        geometry=gpd.points_from_xy(locs["longitude"], locs["latitude"]),
        crs="EPSG:4326",
    ).to_crs(CRS_METRIC)

    pts["dist_coast_km"] = pts.geometry.distance(coast_union) / 1000.0

    out = cfg["paths"]["dist_coast"]
    os.makedirs(os.path.dirname(out), exist_ok=True)
    pts[["station_id", "dist_coast_km"]].to_csv(out, index=False)

    # sanity: northern island/coastal stations near zero, southern inland far
    s = pts.set_index("station_id")["dist_coast_km"]
    print(f"wrote {out}: n={len(s)}, min={s.min():.1f} km, max={s.max():.1f} km")
    print("10 closest:\n", s.nsmallest(10).to_string())
    print("10 farthest:\n", s.nlargest(10).to_string())
    assert s.min() < 10, "expected at least one station within 10 km of the coast"
    assert s.max() > 300, "expected inland stations > 300 km from the coast"


if __name__ == "__main__":
    main()
