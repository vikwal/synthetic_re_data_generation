#!/usr/bin/env python3
"""HANDOFF ask 21a — similarity gate (brainstorm D3), replaces the distance gate.

A-PRIORI thresholds (fixed before looking at any park result, NOT tuned):
  - footprint radius for sector roughness/land cover: 7.5 km (guide: 5-10 km)
  - 12 sectors of 30 degrees; sector z0 = geometric mean of CORINE z0
  - energy-dominant sectors: smallest set carrying >= 60 % of ERA5 100-m
    wind energy (v^3), one full year (2023-07-01 .. 2024-07-01), at the
    driving DWD station
  - branch A requires:
      same CORINE macro-class (CLC level-1: artificial / agricultural /
      forest & semi-natural / wetlands / water) as the MODAL class over the
      union of the energy-dominant-sector footprints (per-sector modal is
      brittle: a mixed agri/forest sector flips the modal class while the
      z0 geo-means are nearly identical)
      AND z0 ratio (park vs station) <= 2 in EVERY energy-dominant sector
    AND no coastline / large water body within 20 km upwind in those
    sectors for either the park or the station. The IBL argument (Garratt
    1990, growth ~1:100) is about marine/lake air masses with fetch, so:
      marine water (CLC 39 intertidal, 42 lagoons, 43 estuaries, 44 sea)
      >= 2 % of the 20-km sector wedge (~2 km^2), OR
      inland water bodies (CLC 41) >= 10 % (~10 km^2 — genuinely large
      lakes; gravel pits and reservoirs of a few km^2 do not build a
      hub-height IBL). Rivers (CLC 40) excluded.
  - the same coastal criterion guards branch B: coastal locations go to
    branch C regardless of the classifier output (gate5 finding: the LGBM
    overcorrects the coast too)
  - branch B otherwise as before: predicted class >= 2

Stations/sites are their own nearest station (self-similarity trivially
holds), so for them the gate reduces to the coastal criterion.

Outputs (data/round2_simgate/): branch_assignment.csv, correction/*.json,
sector_features.csv, topo_features.csv (copy), lab_notebook.md.
"""

import json
import math
import os
import shutil
import sys

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
from round2.correction import QUANTILES  # noqa: E402
from round2.topo import CLC_Z0, Z0_DEFAULT  # noqa: E402

CORINE_TIF = ("/mnt/nvme2/synthetic/raw/round2/corine/Results/"
              "u2018_clc2018_v2020_20u1_raster100m/DATA/U2018_CLC2018_V2020_20u1.tif")
ERA5_DIR = "/mnt/nvme2/synthetic/raw/wind_era5_v2"
OUT_BASE = os.path.join(REPO, "data", "round2_simgate")
OUT_DIR = os.path.join(OUT_BASE, "correction")

N_SECTORS = 12
SECTOR_DEG = 360.0 / N_SECTORS
FOOTPRINT_M = 7_500.0
COAST_RADIUS_M = 20_000.0
ENERGY_SHARE = 0.60
Z0_RATIO_MAX = 2.0
MARINE_SHARE_MIN = 0.02
LAKE_SHARE_MIN = 0.10
MARINE_CLASSES = {39, 42, 43, 44}   # intertidal, lagoons, estuaries, sea
LAKE_CLASSES = {41}                 # inland water bodies (rivers=40 excluded)
ERA5_WINDOW = ("2023-07-01", "2024-07-01")


def macro_class(clc: np.ndarray) -> np.ndarray:
    """CLC grid code (1..44) -> level-1 macro class (1..5), 0 = nodata."""
    out = np.zeros_like(clc, dtype=np.int8)
    out[(clc >= 1) & (clc <= 11)] = 1
    out[(clc >= 12) & (clc <= 22)] = 2
    out[(clc >= 23) & (clc <= 34)] = 3
    out[(clc >= 35) & (clc <= 39)] = 4
    out[(clc >= 40) & (clc <= 44)] = 5
    return out


class SectorSampler:
    def __init__(self, tif_path: str):
        self.ds = rasterio.open(tif_path)
        self.tf = Transformer.from_crs("EPSG:4326", self.ds.crs, always_xy=True)
        z0_lut = np.full(256, Z0_DEFAULT, dtype=float)
        for k, v in CLC_Z0.items():
            z0_lut[k] = v
        self.z0_lut = z0_lut

    def features(self, lat: float, lon: float) -> dict | None:
        """Per-sector footprint z0 (geo-mean), modal macro class, and
        20-km water share."""
        x0, y0 = self.tf.transform(lon, lat)
        margin = COAST_RADIUS_M + 200.0
        win = rasterio.windows.from_bounds(
            x0 - margin, y0 - margin, x0 + margin, y0 + margin,
            transform=self.ds.transform).round_offsets().round_lengths()
        arr = self.ds.read(1, window=win)
        if arr.size == 0:
            return None
        wt = self.ds.window_transform(win)
        rows, cols = np.mgrid[0:arr.shape[0], 0:arr.shape[1]]
        xs = wt.c + wt.a * (cols + 0.5)
        ys = wt.f + wt.e * (rows + 0.5)
        dx, dy = xs - x0, ys - y0
        dist = np.hypot(dx, dy)
        bearing = (np.degrees(np.arctan2(dx, dy))) % 360.0
        sector = (bearing // SECTOR_DEG).astype(int) % N_SECTORS
        valid = (arr >= 1) & (arr <= 44)
        z0 = self.z0_lut[arr]
        macro = macro_class(arr)
        marine = np.isin(arr, list(MARINE_CLASSES))
        lake = np.isin(arr, list(LAKE_CLASSES))

        out = {}
        for s in range(N_SECTORS):
            foot = (sector == s) & (dist <= FOOTPRINT_M) & (dist > 0) & valid
            ring = (sector == s) & (dist <= COAST_RADIUS_M) & (dist > 0) & valid
            if foot.sum() == 0 or ring.sum() == 0:
                return None
            out[s] = {
                "z0": float(np.exp(np.log(z0[foot]).mean())),
                "macro_counts": np.bincount(macro[foot], minlength=6),
                "marine_share_20km": float(marine[ring].mean()),
                "lake_share_20km": float(lake[ring].mean()),
            }
        return out


def dominant_sectors(sid: str) -> list | None:
    """Smallest sector set carrying >= ENERGY_SHARE of v100^3, one year."""
    path = os.path.join(ERA5_DIR, f"Station_{sid}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, usecols=["timestamp", "u_wind_100m", "v_wind_100m"],
                     parse_dates=["timestamp"], index_col="timestamp")
    df = df.loc[ERA5_WINDOW[0]:ERA5_WINDOW[1]]
    u, v = df["u_wind_100m"].values, df["v_wind_100m"].values
    speed = np.hypot(u, v)
    dir_from = (180.0 + np.degrees(np.arctan2(u, v))) % 360.0
    sec = (dir_from // SECTOR_DEG).astype(int) % N_SECTORS
    energy = np.bincount(sec, weights=speed ** 3, minlength=N_SECTORS)
    order = np.argsort(energy)[::-1]
    cum = np.cumsum(energy[order]) / energy.sum()
    k = int(np.searchsorted(cum, ENERGY_SHARE) + 1)
    return sorted(int(s) for s in order[:k])


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    pred = pd.read_csv(os.path.join(REPO, "data", "round2",
                                    "predicted_classes.csv"),
                       dtype={"location_id": str})
    table = pd.read_parquet(os.path.join(
        REPO, "data", "round2", "station_correction_table.parquet")) \
        .set_index("station_id")
    model_q = pd.read_csv(os.path.join(REPO, "data", "round2",
                                       "model_quantiles.csv"),
                          dtype={"location_id": str}).set_index("location_id")
    q_cols_era5 = [f"q_era5_{q:.3f}" for q in QUANTILES]
    q_cols_dwd = [f"q_dwd_{q:.3f}" for q in QUANTILES]

    sampler = SectorSampler(CORINE_TIF)
    st = pred[pred["kind"] == "station"].set_index("location_id")

    feat_cache, dom_cache = {}, {}

    def feats(loc_id, lat, lon):
        if loc_id not in feat_cache:
            feat_cache[loc_id] = sampler.features(lat, lon)
        return feat_cache[loc_id]

    def doms(sid):
        if sid not in dom_cache:
            dom_cache[sid] = dominant_sectors(sid)
        return dom_cache[sid]

    rows, feat_rows = [], []
    for loc_id, row in pred.set_index("location_id").iterrows():
        kind = row["kind"]
        if kind == "station":
            sid = loc_id
        elif kind == "site":
            sid = loc_id.split("_")[1]
        else:
            sid = loc_id.split("_")[1][:5]
        if sid not in table.index or sid not in st.index:
            continue
        pred_class = int(row["predicted_class"])

        dom = doms(sid)
        f_loc = feats(loc_id, row["latitude"], row["longitude"])
        f_st = feats(sid, st.loc[sid, "latitude"], st.loc[sid, "longitude"])
        if dom is None or f_loc is None or f_st is None:
            dom_used = list(range(N_SECTORS)) if dom is None else dom
        else:
            dom_used = dom

        if f_loc is None or f_st is None:
            # cannot evaluate the gate -> conservative: no correction
            branch, reason = "C", "no CORINE coverage"
            macro_ok = None
            z0_ratio_max = np.nan
            marine_max = lake_max = np.nan
        else:
            def union_macro(f):
                counts = sum(f[s]["macro_counts"] for s in dom_used)
                return int(counts[1:].argmax() + 1)
            macro_ok = union_macro(f_loc) == union_macro(f_st)
            ratios = [max(f_loc[s]["z0"], f_st[s]["z0"])
                      / min(f_loc[s]["z0"], f_st[s]["z0"]) for s in dom_used]
            z0_ratio_max = float(max(ratios))
            z0_ok = z0_ratio_max <= Z0_RATIO_MAX
            marine_max = float(max(max(f[s]["marine_share_20km"]
                                       for s in dom_used)
                                   for f in (f_loc, f_st)))
            lake_max = float(max(max(f[s]["lake_share_20km"]
                                     for s in dom_used)
                                 for f in (f_loc, f_st)))
            coastal = (marine_max >= MARINE_SHARE_MIN
                       or lake_max >= LAKE_SHARE_MIN)
            if coastal:
                branch = "C"
                reason = "coastal guard"
            elif macro_ok and z0_ok:
                branch, reason = "A", "similar"
            elif pred_class >= 2:
                branch, reason = "B", ("macro mismatch" if not macro_ok
                                       else "z0 ratio > 2")
            else:
                branch, reason = "C", ("macro mismatch" if not macro_ok
                                       else "z0 ratio > 2")
            for s in dom_used:
                feat_rows.append({
                    "location_id": loc_id, "sector": s,
                    "z0_loc": f_loc[s]["z0"], "z0_station": f_st[s]["z0"],
                    "macro_loc": int(f_loc[s]["macro_counts"][1:].argmax() + 1),
                    "macro_station": int(f_st[s]["macro_counts"][1:].argmax() + 1),
                    "marine_share_loc": f_loc[s]["marine_share_20km"],
                    "marine_share_station": f_st[s]["marine_share_20km"],
                    "lake_share_loc": f_loc[s]["lake_share_20km"],
                    "lake_share_station": f_st[s]["lake_share_20km"]})

        rows.append({"location_id": loc_id, "kind": kind, "station_id": sid,
                     "predicted_class": pred_class,
                     "observed_class": table.loc[sid, "era5_class"],
                     "dominant_sectors": ";".join(map(str, dom_used)),
                     "macro_match": macro_ok, "z0_ratio_max":
                     round(z0_ratio_max, 3) if np.isfinite(z0_ratio_max) else np.nan,
                     "marine_share_max": marine_max, "lake_share_max": lake_max,
                     "branch": branch, "gate_reason": reason})

        if kind in ("park", "station"):
            q_era5 = table.loc[sid, q_cols_era5].values.astype(float).tolist()
            q_station = table.loc[sid, q_cols_dwd].values.astype(float).tolist()
            q_model = None
            if loc_id in model_q.index:
                q_model = model_q.loc[
                    loc_id, [f"q_model_{q:.3f}" for q in QUANTILES]] \
                    .values.astype(float).tolist()
            payload = {"station_id": sid, "location_id": loc_id,
                       "branch": branch, "q_era5": q_era5,
                       "q_target_station": q_station, "q_target_model": q_model}
            name = sid if kind == "station" else loc_id.split("park_")[1]
            with open(os.path.join(OUT_DIR, f"{name}.json"), "w") as f:
                json.dump(payload, f)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_BASE, "branch_assignment.csv"), index=False)
    pd.DataFrame(feat_rows).round(5).to_csv(
        os.path.join(OUT_BASE, "sector_features.csv"), index=False)
    shutil.copy(os.path.join(REPO, "data", "round2", "topo_features.csv"),
                os.path.join(OUT_BASE, "topo_features.csv"))
    print(df.groupby(["kind", "branch"]).size().to_string())
    print()
    parks = df[df["kind"] == "park"]
    print(parks[["location_id", "station_id", "dominant_sectors", "macro_match",
                 "z0_ratio_max", "marine_share_max", "lake_share_max",
                 "branch", "gate_reason"]].to_string(index=False))
    write_lab_notebook(df)


def write_lab_notebook(df: pd.DataFrame):
    parks = df[df["kind"] == "park"]
    nb = os.path.join(OUT_BASE, "lab_notebook.md")
    lines = [
        "# WP2-E Lab Notebook — similarity gate (ask 21a), written BEFORE the runs",
        "",
        "Gate: A iff same CORINE macro-class AND z0 ratio <= 2 in every "
        "energy-dominant sector (>=60 % of v100^3) AND no large water body "
        f"(marine >= {MARINE_SHARE_MIN:.0%} or lakes >= {LAKE_SHARE_MIN:.0%} of the 20-km sector wedge) upwind at "
        "either site or station; coastal locations -> C regardless of the "
        "classifier (guards branch B); else B iff predicted class >= 2.",
        f"Footprint {FOOTPRINT_M/1000:g} km, {N_SECTORS} sectors, thresholds "
        "fixed a priori (see script header).",
        "",
        "## Pre-registered success criterion (from the brainstorm)",
        "1. Lower Saxony repair retained (R2 0.08 -> ~0.58 at the M1->M2 step)",
        "2. NO corrected park degrades vs its uncorrected twin",
        "3. net QM contribution >= 0",
        "If it fails, report honestly.",
        "",
        "## Park branch assignment (before running the chain)",
        "",
        parks[["location_id", "station_id", "dominant_sectors", "macro_match",
               "z0_ratio_max", "marine_share_max", "lake_share_max",
               "branch", "gate_reason"]].to_markdown(index=False),
    ]
    with open(nb, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("lab notebook:", nb)


if __name__ == "__main__":
    main()
