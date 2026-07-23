"""Static covariate table + min-max scaling.

Table = round2 topo features (elevation, slope, aspect->sin/cos, tpi5, tpi75,
tdi, elev_std, z0) + distance to coast + ERA5 ws10 quantiles (5/50/95 %)
computed over the TRAINING period from the processed parquets.

Min-max scaling to [0,1] is fitted on a caller-supplied station set (paper:
train stations only; final models: train+val) and persisted as JSON so both
venvs (synthre / frcst) can read it without sharing sklearn pickles.
"""

import json
import os

import numpy as np
import pandas as pd

from .data import load_processed

ERA5_QUANTILES = {"q05_era5": 0.05, "q50_era5": 0.50, "q95_era5": 0.95}
TOPO_COLS = ["elevation", "slope", "aspect", "tpi5", "tpi75", "tdi",
             "elev_std", "z0"]


def assemble_static_table(cfg: dict, station_ids: list[str]) -> pd.DataFrame:
    """One row per station, unscaled static covariates (config order)."""
    topo = (pd.read_csv(cfg["paths"]["topo_features"],
                        dtype={"location_id": str})
            .query("kind == 'station'")
            .drop_duplicates(subset="location_id")
            .set_index("location_id"))
    dist = (pd.read_csv(cfg["paths"]["dist_coast"], dtype={"station_id": str})
            .set_index("station_id"))

    per = cfg["periods"]
    rows = {}
    for sid in station_ids:
        ws = load_processed(cfg, sid).loc[per["train_start"]:per["train_end"], "ws10"]
        assert ws.index.min() >= per["train_start"] and ws.index.max() <= per["train_end"]
        row = {name: float(ws.quantile(q)) for name, q in ERA5_QUANTILES.items()}
        row.update(topo.loc[sid, TOPO_COLS].to_dict())
        aspect_rad = np.deg2rad(row.pop("aspect"))
        row["aspect_sin"] = float(np.sin(aspect_rad))
        row["aspect_cos"] = float(np.cos(aspect_rad))
        row["dist_coast"] = float(dist.loc[sid, "dist_coast_km"])
        rows[sid] = row

    table = pd.DataFrame.from_dict(rows, orient="index")[cfg["static_covariates"]]
    table.index.name = "station_id"
    # tdi is 0/0-undefined on perfectly flat terrain (offshore lighthouse
    # 02961: elevation range 0 in the window) — no dissection -> 0
    if "tdi" in table:
        flat = table["tdi"].isna() & (table["elev_std"] == 0)
        table.loc[flat, "tdi"] = 0.0
    assert not table.isna().any().any(), \
        f"NaNs in static table: {table.columns[table.isna().any()].tolist()}"
    q = table[list(ERA5_QUANTILES)]
    assert ((q["q05_era5"] < q["q50_era5"]) & (q["q50_era5"] < q["q95_era5"])).all()
    return table


def assemble_park_static_table(cfg: dict, park_station: dict) -> pd.DataFrame:
    """Unscaled static covariates for park centroids (wp9 downscaling).

    park_station: park location_id -> name-giving DWD station id. Topo and
    dist_coast come from the PARK location; the ERA5 ws10 quantiles from the
    station's processed frame (the park uses that grid point's ERA5, round2
    chain convention)."""
    topo = (pd.read_csv(cfg["paths"]["topo_features"],
                        dtype={"location_id": str})
            .query("kind == 'park'")
            .set_index("location_id"))
    dist = (pd.read_csv(cfg["paths"]["dist_coast"], dtype={"station_id": str})
            .set_index("station_id"))

    per = cfg["periods"]
    rows = {}
    for park_id, sid in park_station.items():
        ws = load_processed(cfg, sid).loc[per["train_start"]:per["train_end"], "ws10"]
        row = {name: float(ws.quantile(q)) for name, q in ERA5_QUANTILES.items()}
        row.update(topo.loc[park_id, TOPO_COLS].to_dict())
        aspect_rad = np.deg2rad(row.pop("aspect"))
        row["aspect_sin"] = float(np.sin(aspect_rad))
        row["aspect_cos"] = float(np.cos(aspect_rad))
        row["dist_coast"] = float(dist.loc[park_id, "dist_coast_km"])
        rows[park_id] = row

    table = pd.DataFrame.from_dict(rows, orient="index")[cfg["static_covariates"]]
    table.index.name = "station_id"
    if "tdi" in table:
        flat = table["tdi"].isna() & (table["elev_std"] == 0)
        table.loc[flat, "tdi"] = 0.0
    assert not table.isna().any().any(), \
        f"NaNs in park static table: {table.columns[table.isna().any()].tolist()}"
    return table


def load_static_table(cfg: dict) -> pd.DataFrame:
    return pd.read_csv(cfg["paths"]["static_features"],
                       dtype={"station_id": str}).set_index("station_id")


def minmax_fit(table: pd.DataFrame, fit_ids: list[str]) -> dict:
    """Min/max per column over the fit stations only (paper: train set)."""
    sub = table.loc[fit_ids]
    return {c: {"min": float(sub[c].min()), "max": float(sub[c].max())}
            for c in table.columns}


def minmax_apply(table: pd.DataFrame, scaler: dict) -> pd.DataFrame:
    out = table.copy()
    for c, s in scaler.items():
        out[c] = (out[c] - s["min"]) / (s["max"] - s["min"])
    return out


def save_scaler(scaler: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(scaler, f, indent=1)


def load_scaler(path: str) -> dict:
    with open(path) as f:
        return json.load(f)
