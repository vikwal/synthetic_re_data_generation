#!/usr/bin/env python3
"""WP2-B — random-forest ERA5 quality classifier + applicability-domain map.

Train: topo features (+ mean gwd if available) -> era5_class from WP2-A,
80/20 split on the ~200 stations. Predict: all sites + parks + a Germany map.

Outputs: data/round2/rf_class.pkl, data/round2/predicted_classes.csv,
figs/round2/era5_class_map.png
"""

import os
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)

TOPO = os.path.join(REPO, "data", "round2", "topo_features.csv")
TABLE = os.path.join(REPO, "data", "round2", "station_correction_table.parquet")
FEATURES = ["elevation", "slope", "aspect", "tpi5", "tpi75", "tdi", "elev_std", "z0"]


def main():
    topo = pd.read_csv(TOPO, dtype={"location_id": str})
    table = pd.read_parquet(TABLE)
    st = topo[topo["kind"] == "station"].merge(
        table[["station_id", "era5_class", "rmse"]],
        left_on="location_id", right_on="station_id")
    st = st.dropna(subset=FEATURES + ["era5_class"])
    X, y = st[FEATURES].values, st["era5_class"].values

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42,
                                              stratify=y)
    rf = RandomForestClassifier(n_estimators=500, random_state=42,
                                class_weight="balanced", min_samples_leaf=2)
    rf.fit(X_tr, y_tr)
    print("test accuracy:", rf.score(X_te, y_te))
    print(confusion_matrix(y_te, rf.predict(X_te)))
    print(classification_report(y_te, rf.predict(X_te)))
    cv = cross_val_score(rf, X, y, cv=5)
    print("5-fold CV acc: %.3f +- %.3f" % (cv.mean(), cv.std()))
    imp = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=False)
    print("importances:\n", imp.round(3).to_string())

    rf.fit(X, y)  # final fit on all stations
    with open(os.path.join(REPO, "data", "round2", "rf_class.pkl"), "wb") as f:
        pickle.dump({"model": rf, "features": FEATURES}, f)

    # predictions for every location (stations keep their observed class)
    topo_ok = topo.dropna(subset=FEATURES).copy()
    topo_ok["predicted_class"] = rf.predict(topo_ok[FEATURES].values)
    obs = table.set_index("station_id")["era5_class"]
    topo_ok["observed_class"] = topo_ok["location_id"].map(obs)
    topo_ok.to_csv(os.path.join(REPO, "data", "round2", "predicted_classes.csv"),
                   index=False)
    print(topo_ok.groupby("kind")["predicted_class"].value_counts().to_string())

    # Germany map on the 1 km DEM grid
    make_map(rf)


def make_map(rf):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import rasterio
    from round2 import topo as r2_topo

    fine = _load("/mnt/nvme2/synthetic/raw/round2/dem/mosaic_90m.tif")
    coarse = _load("/mnt/nvme2/synthetic/raw/round2/dem/mosaic_1km.tif")
    corine = r2_topo.CorineSampler(
        "/mnt/nvme2/synthetic/raw/round2/corine/Results/"
        "u2018_clc2018_v2020_20u1_raster100m/DATA/U2018_CLC2018_V2020_20u1.tif")

    lats = np.arange(47.3, 55.05, 0.1)
    lons = np.arange(5.9, 15.05, 0.1)
    grid = np.full((len(lats), len(lons)), np.nan)
    import geopandas as gpd
    de = gpd.read_file(os.path.join(REPO, "misc", "geoBoundaries-DE.geojson"))
    from shapely.geometry import Point
    de_geom = de.union_all()
    for i, la in enumerate(lats):
        rows = []
        for lo in lons:
            m = r2_topo.location_metrics(fine, coarse, la, lo)
            m["z0"] = corine.z0(la, lo)
            rows.append([m.get(k, np.nan) for k in
                         ["elevation", "slope", "aspect", "tpi5", "tpi75",
                          "tdi", "elev_std", "z0"]])
        Xrow = np.array(rows, dtype=float)
        ok = ~np.isnan(Xrow).any(axis=1)
        ok &= np.array([de_geom.contains(Point(lo, la)) for lo in lons])
        if ok.any():
            grid[i, ok] = rf.predict(Xrow[ok])

    fig, ax = plt.subplots(figsize=(7, 9))
    im = ax.pcolormesh(lons, lats, grid, cmap=plt.get_cmap("RdYlGn_r", 3),
                       vmin=0.5, vmax=3.5)
    pred = pd.read_csv(os.path.join(REPO, "data", "round2", "predicted_classes.csv"),
                       dtype={"location_id": str})
    parks = pred[pred["kind"] == "park"]
    ax.scatter(parks["longitude"], parks["latitude"], marker="^", s=60,
               edgecolor="k", facecolor="none", label="parks (13)")
    # Ask 34: the published dataset switched 160 -> 200 sites = the 200 DWD
    # stations with an observed class (site_v2_20260714 / site_branch_manifest).
    sites = pred[(pred["kind"] == "station") & pred["observed_class"].notna()]
    ax.scatter(sites["longitude"], sites["latitude"], marker=".", s=8,
               color="k", alpha=0.5, label=f"sites ({len(sites)})")
    cbar = fig.colorbar(im, ticks=[1, 2, 3], shrink=0.6)
    cbar.set_label("predicted ERA5 quality class (RMSE-based)")
    ax.set_title("ERA5 applicability domain (RF classifier)")
    ax.legend(loc="lower right")
    out = os.path.join(REPO, "figs", "round2", "era5_class_map.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print("map:", out)


def _load(path):
    import rasterio
    with rasterio.open(path) as ds:
        arr = ds.read(1).astype(float)
        if ds.nodata is not None:
            arr[arr == ds.nodata] = np.nan
        return {"arr": arr, "transform": ds.transform}


if __name__ == "__main__":
    main()
