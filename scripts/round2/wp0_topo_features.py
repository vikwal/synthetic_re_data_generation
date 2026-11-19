#!/usr/bin/env python3
"""WP0.4/0.5 — topo features + z0 for all locations.

Builds two lat/lon DEM mosaics from the GLO-30 tiles (90 m and ~1 km) and
computes per-location metrics + CORINE z0. Locations: 203 DWD stations
(stations_master.csv), 13 park centroids (park_layouts.csv), and the ML sites
(data/ML, ids are DWD station ids -> same coordinates, flagged as 'site').

Output: data/round2/topo_features.csv with location_id unique per row.
"""

import glob
import os
import sys

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.merge import merge as rio_merge

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
from round2 import topo  # noqa: E402

DEM_DIR = "/mnt/nvme2/synthetic/raw/round2/dem/glo30"
MOSAIC_FINE = "/mnt/nvme2/synthetic/raw/round2/dem/mosaic_90m.tif"
MOSAIC_COARSE = "/mnt/nvme2/synthetic/raw/round2/dem/mosaic_1km.tif"
CORINE_TIF = ("/mnt/nvme2/synthetic/raw/round2/corine/Results/"
              "u2018_clc2018_v2020_20u1_raster100m/DATA/U2018_CLC2018_V2020_20u1.tif")
OUT = os.path.join(REPO, "data", "round2", "topo_features.csv")


def build_mosaic(out_path: str, decimate: int):
    if os.path.exists(out_path):
        return
    files = sorted(glob.glob(os.path.join(DEM_DIR, "*.tif")))
    assert files, "no DEM tiles found"
    srcs = [rasterio.open(f) for f in files]
    # GLO-30 native ~1 arcsec; decimate via res parameter
    res0 = srcs[0].res
    res = (res0[0] * decimate, res0[1] * decimate)
    arr, transform = rio_merge(srcs, res=res, resampling=Resampling.average,
                               nodata=srcs[0].nodata)
    meta = srcs[0].meta.copy()
    meta.update(height=arr.shape[1], width=arr.shape[2], transform=transform,
                compress="deflate")
    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(arr)
    for s in srcs:
        s.close()
    print(f"mosaic {out_path}: {arr.shape} @ {res}")


def load(path):
    with rasterio.open(path) as ds:
        arr = ds.read(1).astype(float)
        if ds.nodata is not None:
            arr[arr == ds.nodata] = np.nan
        return {"arr": arr, "transform": ds.transform}


def main():
    build_mosaic(MOSAIC_FINE, decimate=3)     # ~90 m
    build_mosaic(MOSAIC_COARSE, decimate=33)  # ~1 km
    fine, coarse = load(MOSAIC_FINE), load(MOSAIC_COARSE)
    corine = topo.CorineSampler(CORINE_TIF) if os.path.exists(CORINE_TIF) else None
    if corine is None:
        print("WARNING: CORINE tif missing, z0 will be default")

    stations = pd.read_csv(os.path.join(REPO, "..", "stations_master.csv"),
                           dtype={"station_id": str})
    locations = [{"location_id": r.station_id, "kind": "station",
                  "latitude": float(r.latitude), "longitude": float(r.longitude)}
                 for r in stations.itertuples()]
    lay = pd.read_csv(os.path.join(REPO, "data", "round2", "park_layouts.csv"),
                      dtype={"park_id": str})
    for pid, grp in lay.dropna(subset=["park_id"]).groupby("park_id"):
        locations.append({"location_id": f"park_{pid}", "kind": "park",
                          "latitude": grp["latitude"].mean(),
                          "longitude": grp["longitude"].mean()})
    site_ids = sorted({os.path.basename(p)[len("nwp_"):-len(".csv")]
                       for p in glob.glob(os.path.join(REPO, "data", "ML", "nwp_*.csv"))})
    st_coords = stations.set_index("station_id")
    n_site_miss = 0
    for sid in site_ids:
        if sid in st_coords.index:
            locations.append({"location_id": f"site_{sid}", "kind": "site",
                              "latitude": float(st_coords.loc[sid, "latitude"]),
                              "longitude": float(st_coords.loc[sid, "longitude"])})
        else:
            n_site_miss += 1

    rows = []
    for loc in locations:
        m = topo.location_metrics(fine, coarse, loc["latitude"], loc["longitude"])
        if corine is not None:
            m["z0"] = corine.z0(loc["latitude"], loc["longitude"])
            m["clc_class"] = corine.clc_class(loc["latitude"], loc["longitude"])
        rows.append({**loc, **m})
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"locations: {len(df)} (sites without coords: {n_site_miss})")
    print(df.groupby("kind").size().to_string())
    print(df[["elevation", "slope", "tpi5", "tpi75", "tdi", "elev_std", "z0"]]
          .describe().round(3).to_string())


if __name__ == "__main__":
    main()
