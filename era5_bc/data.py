"""Data pipeline: raw ERA5 CSVs + DWD hourly parquets -> tidy per-station
frames with scaling-factor target, plus per-gridpoint normalization stats.

Follows the reference implementation (TRWindBC databuilder/dataloader.py):
- target y_t = ws_obs / ws_era5 (raw ratio; validity handled via mask)
- u/v enter as unit direction vectors u/(ws+1e-5)
- remaining dynamic covariates standardized per grid point with mean/std
  computed over the TRAINING period only (ERA5-only, so this is leakage-free
  for val/test stations as well).
"""

import json
import os

import numpy as np
import pandas as pd

from .config import station_dwd_path, station_era5_path, station_processed_path

# processed-frame columns holding raw (unnormalized) dynamic covariates
DYN_RAW = ["ws10", "blh", "sp", "t2m"]  # standardized per grid point
EPS_DIR = 1e-5  # authors' epsilon for the u/v unit vectors


def load_era5(cfg: dict, station_id: str) -> pd.DataFrame:
    """Hourly UTC ERA5 frame with u10, v10, ws10, t2m, blh, sp."""
    df = pd.read_csv(
        station_era5_path(cfg, station_id),
        usecols=["timestamp", "u_wind_10m", "v_wind_10m", "temp_2m",
                 "pressure", "blh"],
        parse_dates=["timestamp"],
        index_col="timestamp",
    ).rename(columns={"u_wind_10m": "u10", "v_wind_10m": "v10",
                      "temp_2m": "t2m", "pressure": "sp"})
    df["ws10"] = np.hypot(df["u10"], df["v10"])
    assert df.index.is_monotonic_increasing and df.index.tz is not None
    return df


def load_obs(cfg: dict, station_id: str) -> pd.Series:
    """Hourly UTC observed 10 m wind speed."""
    obs = pd.read_parquet(station_dwd_path(cfg, station_id),
                          columns=["wind_speed"])["wind_speed"]
    return obs[obs.notna()]


def build_station_frame(cfg: dict, station_id: str) -> pd.DataFrame:
    """Tidy hourly frame over the full ERA5 span.

    Columns: u10, v10, ws10, t2m, blh, sp (raw), u10_dir, v10_dir,
    ws_obs, y (raw ratio, NaN where invalid), y_valid.
    """
    era = load_era5(cfg, station_id)
    # ERA5 is gap-free; enforce a continuous hourly index anyway
    full = pd.date_range(era.index[0], era.index[-1], freq="1h", tz="UTC")
    assert len(full) == len(era), f"{station_id}: ERA5 gaps ({len(era)} vs {len(full)})"

    df = era.reindex(full)
    df.index.name = "timestamp"
    df["u10_dir"] = df["u10"] / (df["ws10"] + EPS_DIR)
    df["v10_dir"] = df["v10"] / (df["ws10"] + EPS_DIR)

    df["ws_obs"] = load_obs(cfg, station_id).reindex(full)
    min_era5 = cfg["target"]["min_era5_wind"] or 0.0
    df["y_valid"] = df["ws_obs"].notna() & (df["ws10"] >= min_era5)
    df["y"] = np.where(df["y_valid"], df["ws_obs"] / df["ws10"], np.nan)
    return df


def compute_norm_stats(frame: pd.DataFrame, cfg: dict) -> dict:
    """Per-gridpoint mean/std of the standardized dynamic covariates,
    over the TRAINING period only (hard-asserted)."""
    per = cfg["periods"]
    sl = frame.loc[per["train_start"]:per["train_end"], DYN_RAW]
    assert len(sl) > 0, "empty training slice"
    assert sl.index.min() >= per["train_start"] and sl.index.max() <= per["train_end"]
    return {c: {"mean": float(sl[c].mean()), "std": float(sl[c].std())}
            for c in DYN_RAW}


def normalized_dynamics(frame: pd.DataFrame, stats: dict, cfg: dict) -> np.ndarray:
    """[T, n_dyn] float32 array in config order (dynamic_covariates)."""
    cols = []
    for name in cfg["dynamic_covariates"]:
        if name in ("u10_dir", "v10_dir"):
            cols.append(frame[name].to_numpy(np.float32))
        else:
            s = stats[name]
            cols.append(((frame[name] - s["mean"]) / s["std"]).to_numpy(np.float32))
    return np.stack(cols, axis=-1)


def load_norm_stats(cfg: dict) -> dict:
    with open(cfg["paths"]["norm_stats"]) as f:
        return json.load(f)


def load_processed(cfg: dict, station_id: str) -> pd.DataFrame:
    return pd.read_parquet(station_processed_path(cfg, station_id))


def obs_coverage(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> float:
    """Fraction of hours in [start, end] with a valid observation."""
    sl = frame.loc[start:end, "ws_obs"]
    n_hours = int((end - start) / pd.Timedelta("1h")) + 1
    return float(sl.notna().sum() / n_hours)
