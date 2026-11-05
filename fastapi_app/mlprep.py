"""Data preparation for the pre-trained model tab: weather features -> windows -> scaled arrays.

The learning task is a sequence-to-one surrogate: a window of the last `lookback` hours of
weather/time features predicts the plant's output at the window's final hour, expressed as a
fraction of rated capacity (so models transfer between plants of different sizes).
Pure numpy/pandas; no ML framework required.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

WIND_FEATURES = ["wind_speed_hub", "u_wind_100m", "v_wind_100m", "density_hub", "temperature_hub"]
SOLAR_FEATURES = ["Total", "Direct", "Diffuse", "temperature_2m", "wind_speed"]
TIME_FEATURES = ["hour_sin", "hour_cos", "doy_sin", "doy_cos"]

# What the user can choose from. Weather columns are used as they are; the two time options each expand into
# a sine/cosine pair (so 23:00 and 00:00 are neighbours for the model).
FEATURE_CATALOG = {
    "wind": [
        ("wind_speed_hub", "Wind speed", "at hub height"),
        ("u_wind_100m", "Wind direction (east-west)", "100 m wind component"),
        ("v_wind_100m", "Wind direction (north-south)", "100 m wind component"),
        ("density_hub", "Air density", "at hub height"),
        ("temperature_hub", "Temperature", "at hub height"),
    ],
    "solar": [
        ("Total", "Total irradiance", "plane of array"),
        ("Direct", "Direct irradiance", "plane of array"),
        ("Diffuse", "Diffuse irradiance", "plane of array"),
        ("temperature_2m", "Air temperature", "2 m above ground"),
        ("wind_speed", "Wind speed", "cools the panels"),
    ],
    # Archived NWP forecasts (ICON-D2 / ECMWF); ids are columns derived in nwp.load_weather()
    "nwp_wind": [
        ("wind_speed_100m", "Wind speed", "100 m above ground"),
        ("u_wind_100m", "Wind direction (east-west)", "100 m wind component"),
        ("v_wind_100m", "Wind direction (north-south)", "100 m wind component"),
        ("wind_speed_10m", "Wind speed", "10 m above ground"),
        ("u_wind_10m", "Wind direction (east-west)", "10 m wind component"),
        ("v_wind_10m", "Wind direction (north-south)", "10 m wind component"),
        ("temp_2m", "Temperature", "2 m above ground"),
        ("pressure", "Surface pressure", "air pressure"),
        ("dew_point_2m", "Dew point", "2 m above ground (humidity)"),
    ],
    "nwp_solar": [
        ("shortwave_radiation", "Total irradiance", "global horizontal"),
        ("direct_radiation", "Direct irradiance", "horizontal"),
        ("diffuse_radiation", "Diffuse irradiance", "horizontal"),
        ("temp_2m", "Air temperature", "2 m above ground"),
        ("wind_speed_10m", "Wind speed", "10 m above ground"),
    ],
    "time": [
        ("time_of_day", "Time of day", "hour of the day"),
        ("season", "Time of year", "day of the year"),
    ],
}
TIME_GROUPS = {"time_of_day": ["hour_sin", "hour_cos"], "season": ["doy_sin", "doy_cos"]}


# Height (m above ground) each forecast input is measured at; 0 = surface value. The Pre-train form lets you choose
# which heights the inputs may come from.
FEATURE_LAYER = {
    "wind_speed_100m": 100, "u_wind_100m": 100, "v_wind_100m": 100,
    "wind_speed_10m": 10, "u_wind_10m": 10, "v_wind_10m": 10,
    "temp_2m": 2, "dew_point_2m": 2, "pressure": 0,
    "shortwave_radiation": 0, "direct_radiation": 0, "diffuse_radiation": 0,
}


def catalog() -> dict:
    return {k: [{"id": i, "label": l, "hint": h, "layer": FEATURE_LAYER.get(i) if k.startswith("nwp_") else None}
                for i, l, h in v] for k, v in FEATURE_CATALOG.items()}


def catalog_key(kind: str, source: Optional[str]) -> str:
    return kind if source in (None, "era5") else f"nwp_{kind}"


def resolve_features_multi(selected: Optional[List[str]], kind: str, sources: List[str],
                           allow_empty: bool = False) -> Tuple[List[str], List[str]]:
    """Like `resolve_features` for several weather models in one run. Weather columns are tagged with their model
    (`<source>:<column>`, e.g. `srv_icon_d2:wind_speed_100m`); selected=None means every input of every model."""
    ids = [i for i, _, _ in FEATURE_CATALOG[f"nwp_{kind}"]]
    valid = [f"{s}:{i}" for s in sources for i in ids]
    if selected is None:
        return valid, list(TIME_FEATURES)
    unknown = [f for f in selected if f not in valid and f not in TIME_GROUPS]
    if unknown:
        raise ValueError(f"Unknown feature(s) for {kind} data: {', '.join(unknown)}")
    weather = [f for f in selected if f in valid]
    if not weather and not allow_empty:
        raise ValueError("Select at least one weather input (or use recent power).")
    return weather, list(TIME_FEATURES)


def resolve_features(selected: Optional[List[str]], kind: str,
                     source: Optional[str] = None, allow_empty: bool = False) -> Tuple[Optional[List[str]], List[str]]:
    """User choice -> (weather columns or None for 'all available', time columns).
    The time encodings (hour of day, day of year) are always part of the inputs, so only weather is chosen.
    selected=None means every weather column the source offers."""
    if selected is None:
        return (None if source in (None, "era5") else [i for i, _, _ in FEATURE_CATALOG[catalog_key(kind, source)]],
                list(TIME_FEATURES))
    valid = {i for i, _, _ in FEATURE_CATALOG[catalog_key(kind, source)]}
    unknown = [f for f in selected if f not in valid and f not in TIME_GROUPS]
    if unknown:
        raise ValueError(f"Unknown feature(s) for {kind} data: {', '.join(unknown)}")
    weather = [f for f in selected if f in valid]
    if not weather and not allow_empty:
        raise ValueError("Select at least one weather input (or use recent power).")
    return weather, list(TIME_FEATURES)


def pick_features(df: pd.DataFrame, kind: str) -> List[str]:
    """Weather columns available in a generated result (time features are added separately)."""
    wanted = WIND_FEATURES if kind == "wind" else SOLAR_FEATURES
    return [c for c in wanted if c in df.columns and df[c].notna().any()]


def build_features(hourly: pd.DataFrame, weather: List[str], time_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """Chosen weather columns plus the chosen cyclic time encodings (UTC). Default: all four time columns."""
    time_cols = list(TIME_FEATURES) if time_cols is None else time_cols
    missing = [c for c in weather if c not in hourly.columns]
    if missing:
        raise ValueError(f"The dataset lacks the feature column(s): {', '.join(missing)}")
    X = hourly[weather].astype(float).copy()
    empty = [c for c in weather if X[c].isna().all()]
    if empty:
        raise ValueError(f"The weather source has no data for: {', '.join(empty)}. Untick them and try again.")
    idx = hourly.index
    all_time = {
        "hour_sin": np.sin(2 * np.pi * idx.hour / 24.0), "hour_cos": np.cos(2 * np.pi * idx.hour / 24.0),
        "doy_sin": np.sin(2 * np.pi * idx.dayofyear / 365.25), "doy_cos": np.cos(2 * np.pi * idx.dayofyear / 365.25),
    }
    for c in time_cols:
        X[c] = all_time[c]
    return X


def to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    h = df.resample("h").mean(numeric_only=True)
    if h.index.tz is None:
        h.index = h.index.tz_localize("UTC")
    return h


def make_windows(X: np.ndarray, y: np.ndarray, times: pd.DatetimeIndex,
                 lookback: int, horizon: int = 1, future: bool = False, power_col: Optional[int] = None,
                 power_lag: int = 1) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """All complete windows ending at each hour s: X (N, lookback, F), end times (N,) and the targets:
    y (N,) for horizon 1 (power at s), or y (N, horizon) with the power at s, s+1, ..., s+horizon-1.

    future=True (multi-step only) extends every window by the horizon - 1 hours after s, so the model also sees the
    weather *forecast* for the hours it predicts: X (N, lookback + horizon - 1, F + 1). The power column
    `power_col` (the plant's own power, shifted by `power_lag` hours) is unknown there and is set to NaN; a final
    0/1 channel marks those future steps. Windows with any other NaN are dropped."""
    if lookback < 1 or horizon < 1:
        raise ValueError("lookback and horizon must be at least 1")
    if future and horizon > 1:
        steps = lookback + horizon - 1
        if len(X) < steps:
            raise ValueError("The dataset is shorter than the lookback window plus the forecast horizon.")
        win = np.transpose(sliding_window_view(X, steps, axis=0), (0, 2, 1)).astype(np.float32)   # (N0, steps, F)
        tgt = sliding_window_view(y, horizon)[lookback - 1:]                                        # first predicted hour s
        bad = np.isnan(win)
        if power_col is not None:
            unknown = lookback - 1 + int(power_lag)          # first step whose lagged power would be after s - 1
            if unknown < steps:
                win[:, unknown:, power_col] = np.nan
                bad[:, unknown:, power_col] = False
        ok = ~bad.any(axis=(1, 2)) & ~np.isnan(tgt).any(axis=1)
        flag = np.broadcast_to((np.arange(steps) >= lookback).astype(np.float32)[None, :, None], (len(win), steps, 1))
        out = np.concatenate([win, flag], axis=2)[ok]
        return out, tgt[ok].astype(np.float32), times[lookback - 1: lookback - 1 + len(ok)][ok]
    if len(X) < lookback + horizon - 1:
        raise ValueError("The dataset is shorter than the lookback window plus the forecast horizon.")
    win = np.transpose(sliding_window_view(X, lookback, axis=0), (0, 2, 1))       # (T-L+1, L, F)
    tgt = sliding_window_view(y, horizon)[lookback - 1:]                          # (T-L-H+2, H): power at s .. s+H-1
    win = win[: len(tgt)]
    ok = ~np.isnan(win).any(axis=(1, 2)) & ~np.isnan(tgt).any(axis=1)
    tgt = tgt[ok].astype(np.float32)
    return win[ok].astype(np.float32), (tgt[:, 0] if horizon == 1 else tgt), times[lookback - 1: lookback - 1 + len(ok)][ok]


def prepare(df: pd.DataFrame, kind: str, capacity_w: float, lookback: int,
            weather: Optional[List[str]] = None, target: Optional[pd.Series] = None,
            time_cols: Optional[List[str]] = None, weather_df: Optional[pd.DataFrame] = None,
            power_lag: Optional[int] = None, horizon: int = 1,
            future_weather: bool = False) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex, List[str]]:
    """One dataset -> (windows, targets, end times, weather feature names).

    `target` (hourly, fraction of capacity, UTC index) overrides the dataset's own power,
    which is how real measurements are paired with the synthetic weather at the same site.
    """
    if capacity_w is None or capacity_w <= 0:
        raise ValueError("The dataset has no rated capacity to normalise power with.")
    hourly = to_hourly(df)                                   # the run: power (and its own ERA5-style weather)
    wx = hourly if weather_df is None else weather_df        # where the input features come from
    feats = pick_features(wx, kind) if weather is None else weather
    if not feats and not (time_cols if time_cols is not None else TIME_FEATURES):
        raise ValueError("No input features found in this dataset.")
    if weather_df is not None:                               # NWP weather: restrict to hours the run covers
        wx = wx.reindex(wx.index.intersection(hourly.index))
    X = build_features(wx, feats, time_cols)
    if len(X) > 1:                                           # make hours without weather explicit gaps (NaN), not skipped rows
        X = X.reindex(pd.date_range(X.index.min(), X.index.max(), freq="h"))
    if target is None:
        if "power" not in hourly.columns:
            raise ValueError("The dataset has no `power` column.")
        y = (hourly["power"] / capacity_w).reindex(X.index).to_numpy(dtype=float)
    else:
        y = target.reindex(X.index).to_numpy(dtype=float)
    if power_lag:
        # Recent power as an input: the value at hour s is the power at hour s - lag, so a window ending at t
        # only sees power up to t - lag (a lag-hour-ahead forecast) and never the value being predicted.
        X = X.copy()
        X["power_lag"] = pd.Series(y, index=X.index).shift(freq=pd.Timedelta(hours=int(power_lag))).reindex(X.index).to_numpy()
    power_col = list(X.columns).index("power_lag") if power_lag else None
    Xw, yw, tw = make_windows(X.to_numpy(dtype=float), y, X.index, lookback, horizon,
                              future=future_weather, power_col=power_col, power_lag=power_lag or 1)
    if len(yw) < 50:
        raise ValueError(f"Only {len(yw)} usable windows after alignment; need at least 50. "
                         "Check that the date ranges overlap and the data has few gaps.")
    return Xw, yw, tw, feats


def per_horizon_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> List[float]:
    """RMSE (% of capacity) for each forecast step; one value for single-step models."""
    yt, yp = np.asarray(y_true, float), np.asarray(y_pred, float)
    if yt.ndim == 1:
        yt, yp = yt[:, None], yp[:, None]
    return [round(100 * float(np.sqrt(((yp[:, k] - yt[:, k]) ** 2).mean())), 3) for k in range(yt.shape[1])]


def split_by_date(times, train_until: str, horizon: int = 1, train_from: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Boolean masks (train, validation) over window end times for a split by calendar date.

    Training uses every window whose last target hour is on or before `train_until` (end of that day, UTC);
    validation uses windows ending on the next day or later. Windows whose targets straddle the cut are left out
    of both sets, so no validation hour is ever a training target. `train_from` (optional) drops everything before that day."""
    t = pd.DatetimeIndex(times)
    if t.tz is None:
        t = t.tz_localize("UTC")
    cut = pd.Timestamp(train_until)
    cut = (cut.tz_localize("UTC") if cut.tzinfo is None else cut.tz_convert("UTC")) + pd.Timedelta(days=1)   # first validation hour
    last_target = t + pd.Timedelta(hours=max(int(horizon), 1) - 1)
    train = np.asarray(last_target < cut)
    if train_from:                                            # data before this day is simply not used
        start = pd.Timestamp(train_from)
        start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
        train &= np.asarray(t >= start)
    return train, np.asarray(t >= cut)


def chrono_split(n: int, val_frac: float) -> int:
    """Index where validation starts (last `val_frac` of the series, never shuffled)."""
    cut = int(round(n * (1.0 - val_frac)))
    return min(max(cut, 1), n - 1)


SCALINGS = ("standard", "minmax", "none")


def fit_scaler(Xtrain: np.ndarray, kind: str = "standard") -> Dict[str, object]:
    """Per-feature (x - mean) / std. `minmax` maps the training range to [0, 1]; `none` leaves values as they are."""
    if kind not in SCALINGS:
        raise ValueError(f"scaling must be one of {', '.join(SCALINGS)}")
    flat = Xtrain.reshape(-1, Xtrain.shape[-1])
    if kind == "none":
        return {"type": kind, "mean": [0.0] * flat.shape[1], "std": [1.0] * flat.shape[1]}
    if kind == "minmax":
        lo, hi = np.nanmin(flat, axis=0), np.nanmax(flat, axis=0)
        span = np.where(hi - lo < 1e-6, 1.0, hi - lo)
        return {"type": kind, "mean": lo.tolist(), "std": span.tolist()}
    mean = np.nanmean(flat, axis=0)                          # NaN only marks unknown future power: ignored here
    std = np.nanstd(flat, axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return {"type": kind, "mean": mean.tolist(), "std": std.tolist()}


def apply_scaler(X: np.ndarray, scaler: Dict[str, List[float]]) -> np.ndarray:
    out = ((X - np.asarray(scaler["mean"], np.float32)) / np.asarray(scaler["std"], np.float32)).astype(np.float32)
    return np.nan_to_num(out, nan=0.0)                       # unknown future power -> 0 (the training mean after scaling)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Optional[float]]:
    """Errors in % of rated capacity (inputs are capacity fractions)."""
    y_true, y_pred = np.asarray(y_true, float).ravel(), np.asarray(y_pred, float).ravel()
    err = y_pred - y_true
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    r2 = 1.0 - float((err ** 2).sum()) / ss_tot if ss_tot > 0 else None
    return {
        "mae_pct": round(100 * float(np.abs(err).mean()), 3),
        "rmse_pct": round(100 * float(np.sqrt((err ** 2).mean())), 3),
        "bias_pct": round(100 * float(err.mean()), 3),
        "r2": None if r2 is None else round(r2, 4),
        "n": int(len(y_true)),
    }
