"""
Pure analytics over a generated power time series (pandas/numpy only).
All functions return plain Python structures so they are easy to test and serialise.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
SECTORS = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
SPEED_BINS = [(0, 4), (4, 8), (8, 12), (12, 16), (16, float("inf"))]


def _num(x) -> Optional[float]:
    x = float(x)
    return x if np.isfinite(x) else None


def step_hours(df: pd.DataFrame) -> float:
    """Typical time step of the index in hours (1.0 if it cannot be inferred)."""
    if len(df.index) < 2:
        return 1.0
    step = pd.Series(df.index[1:] - df.index[:-1]).median()
    hours = step / pd.Timedelta(hours=1)
    return float(hours) if np.isfinite(hours) and hours > 0 else 1.0


def span_years(df: pd.DataFrame) -> float:
    dt = pd.Timedelta(hours=step_hours(df))
    return max((df.index.max() - df.index.min() + dt) / pd.Timedelta(days=365.25), 1e-9)


def kpis(df: pd.DataFrame, kind: str, capacity_w: float) -> Dict[str, Optional[float]]:
    """Headline numbers for a scenario run."""
    p = df["power"].astype(float)
    dt = step_hours(df)
    energy_mwh = float(np.nansum(p) * dt / 1e6)
    years = span_years(df)
    annual_mwh = energy_mwh / years
    out: Dict[str, Optional[float]] = {
        "capacity_mw": _num(capacity_w / 1e6),
        "energy_mwh": _num(energy_mwh),
        "annual_energy_mwh": _num(annual_mwh),
        "capacity_factor_pct": _num(100.0 * np.nanmean(p) / capacity_w) if capacity_w > 0 else None,
        "full_load_hours": _num(annual_mwh * 1e6 / capacity_w) if capacity_w > 0 else None,
        "peak_power_mw": _num(np.nanmax(p) / 1e6),
        "mean_power_mw": _num(np.nanmean(p) / 1e6),
        "zero_output_pct": _num(100.0 * np.mean(p.fillna(0) <= 0)),
        "years": _num(years),
    }
    if kind == "wind" and "wind_speed_hub" in df.columns:
        out["mean_wind_speed"] = _num(np.nanmean(df["wind_speed_hub"]))
    if kind == "solar" and capacity_w > 0:
        out["specific_yield_kwh_kwp"] = _num(annual_mwh * 1e3 / (capacity_w / 1e3))
    return out


def monthly_energy(df: pd.DataFrame) -> Dict[str, List]:
    """Energy (MWh) per calendar month for every year in the data."""
    dt = step_hours(df)
    s = (df["power"].astype(float) * dt / 1e6).groupby([df.index.year, df.index.month]).sum()
    labels = [f"{y}-{m:02d}" for y, m in s.index]
    return {"labels": labels, "values": [_num(v) for v in s.values]}


def seasonality(df: pd.DataFrame) -> Dict[str, List]:
    """Mean monthly energy (MWh) for each calendar month, averaged over years."""
    dt = step_hours(df)
    per = (df["power"].astype(float) * dt / 1e6).groupby([df.index.year, df.index.month]).sum()
    per.index.names = ["year", "month"]
    mean = per.groupby(level="month").mean()
    return {"months": MONTHS, "values": [_num(mean.get(m, np.nan)) for m in range(1, 13)]}


def hour_month_heatmap(df: pd.DataFrame, capacity_w: float) -> Dict[str, List]:
    """Mean output as % of capacity by hour of day (rows) x month (columns)."""
    cf = df["power"].astype(float) / capacity_w * 100.0
    grid = cf.groupby([df.index.hour, df.index.month]).mean().unstack()
    grid = grid.reindex(index=range(24), columns=range(1, 13))
    z = [[_num(v) for v in row] for row in grid.values]
    return {"x": MONTHS, "y": list(range(24)), "z": z}


def duration_curve(df: pd.DataFrame, capacity_w: float, points: int = 200) -> Dict[str, List]:
    """Power duration curve: output (% of capacity) vs. share of time it is exceeded."""
    p = np.sort(df["power"].astype(float).dropna().to_numpy())[::-1]
    if len(p) == 0:
        return {"x": [], "y": []}
    n = len(p)
    idx = np.unique(np.linspace(0, n - 1, min(points, n)).astype(int))
    return {"x": [_num(100.0 * (i + 1) / n) for i in idx], "y": [_num(100.0 * p[i] / capacity_w) for i in idx]}


def wind_rose(df: pd.DataFrame) -> Optional[Dict]:
    """Frequency (%) per 22.5° sector and speed class, using the 100 m wind components."""
    if not {"u_wind_100m", "v_wind_100m"}.issubset(df.columns):
        return None
    u = df["u_wind_100m"].astype(float).to_numpy()
    v = df["v_wind_100m"].astype(float).to_numpy()
    ok = np.isfinite(u) & np.isfinite(v)
    u, v = u[ok], v[ok]
    if len(u) == 0:
        return None
    speed = np.hypot(u, v)
    # meteorological convention: direction the wind blows FROM, clockwise from north
    direction = (270.0 - np.degrees(np.arctan2(v, u))) % 360.0
    sector = (np.floor((direction + 11.25) / 22.5).astype(int)) % 16
    total = len(u)
    series = []
    for lo, hi in SPEED_BINS:
        in_bin = (speed >= lo) & (speed < hi)
        counts = np.bincount(sector[in_bin], minlength=16)
        label = f"{lo:g}–{hi:g} m/s" if np.isfinite(hi) else f"≥{lo:g} m/s"
        series.append({"label": label, "values": [_num(100.0 * c / total) for c in counts]})
    return {"sectors": SECTORS, "series": series}


def speed_histogram(df: pd.DataFrame, column: str = "wind_speed_hub", bin_width: float = 1.0) -> Optional[Dict]:
    if column not in df.columns:
        return None
    x = df[column].astype(float).dropna().to_numpy()
    if len(x) == 0:
        return None
    edges = np.arange(0, np.ceil(x.max() / bin_width + 1) * bin_width + 1e-9, bin_width)
    counts, edges = np.histogram(x, bins=edges)
    return {"x": [_num((a + b) / 2) for a, b in zip(edges[:-1], edges[1:])],
            "y": [_num(100.0 * c / len(x)) for c in counts]}


def energy_waterfall(ideal_mwh: float, after_ageing_mwh: float, net_mwh: float) -> Dict[str, float]:
    """Breakdown from ideal energy to net energy: ageing loss, then wake loss."""
    return {
        "ideal_mwh": ideal_mwh,
        "ageing_loss_mwh": max(0.0, ideal_mwh - after_ageing_mwh),
        "wake_loss_mwh": max(0.0, after_ageing_mwh - net_mwh),
        "net_mwh": net_mwh,
    }
