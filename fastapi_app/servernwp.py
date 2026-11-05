"""NWP forecasts stored on the server, read per grid point from Parquet files.

Active only when the environment variable SYNTH_RE_NWP_ROOT points to the folder that holds the model folders, e.g.
    SYNTH_RE_NWP_ROOT=/mnt/nvme1
    ECMWF:   <root>/ecmwf/parquet/SL/47_2600_10_1100_wind_sl.parquet                  (all runs in one file)
    ICON-D2: <root>/icon-d2/parquet/ML/<run hour>/<station>/47_3796_10_2951_ML.parquet (one file per run hour)
File names carry the grid point: `47_2600_10_1100` = lat 47.2600, lon 10.1100. Each file holds many forecast runs
(starttime, forecasttime in hours). For a run's location the nearest points are weighted by inverse distance; for every
hour the latest forecast whose lead time lies in the chosen window is used.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import parkdata

# id -> (label, description, model folder below the root, lead-time limit in hours)
SOURCE_INFO = {
    "srv_icon_d2": ("ICON-D2", "DWD high-resolution model (2 km), forecast runs stored on the server (06, 09, 12 and 15 UTC).",
                    "icon-d2/parquet/ML", 48),
    "srv_ecmwf": ("ECMWF IFS", "ECMWF global model (0.25°), forecast runs stored on the server (00 and 12 UTC).",
                  "ecmwf/parquet/SL", 57),
}
RADIATION = ["shortwave_radiation", "direct_radiation", "diffuse_radiation"]
FEATURES = {
    "srv_icon_d2": parkdata.FEATURES["pkg_icon_d2"] + RADIATION,
    "srv_ecmwf": parkdata.FEATURES["pkg_ecmwf"] + RADIATION,
}
N_POINTS = 3                                                   # nearest grid points that are blended
MAX_KM = float(os.environ.get("SYNTH_RE_NWP_MAX_KM", "50"))   # farther than this from the nearest point -> error


def root() -> Optional[Path]:
    env = os.environ.get("SYNTH_RE_NWP_ROOT")
    return Path(env) if env and Path(env).is_dir() else None


def is_server(source: str) -> bool:
    return source in SOURCE_INFO


def folder(source: str) -> Path:
    r = root()
    if r is None:
        raise ValueError("The server NWP folder is not set. Set SYNTH_RE_NWP_ROOT.")
    return r / SOURCE_INFO[source][2]


def _parse(name: str) -> Optional[Tuple[float, float]]:
    """'47_2600_10_1100_wind_sl.parquet' or '47_3796_10_2951_ML.parquet' -> (47.26, 10.11)."""
    p = name.split(".")[0].split("_")
    try:
        return float(f"{p[0]}.{p[1]}"), float(f"{p[2]}.{p[3]}")
    except (IndexError, ValueError):
        return None


@lru_cache(maxsize=4)
def index(source: str, root_str: str) -> List[Tuple[float, float, Tuple[str, ...]]]:
    """All grid points of a source as (lat, lon, files). ICON-D2 has one file per run hour (and station folder) for a
    point; ECMWF has a single file. The folders are listed once per process."""
    base = Path(root_str) / SOURCE_INFO[source][2]
    found: Dict[Tuple[float, float], List[str]] = {}
    files = base.glob("*/*/*.parquet") if source == "srv_icon_d2" else base.glob("*.parquet")
    for f in files:
        ll = _parse(f.name)
        if ll:
            found.setdefault(ll, []).append(str(f))
    return [(la, lo, tuple(sorted(v))) for (la, lo), v in found.items()]


def status(source: str) -> dict:
    """Point count for the source list (no file is opened)."""
    pts = index(source, str(root())) if root() else []
    return {"points": len(pts)}


def nearest(source: str, lat: float, lon: float, k: int = N_POINTS) -> List[Tuple[Tuple[str, ...], float]]:
    """The k grid points nearest to (lat, lon) as (files, distance in km)."""
    pts = index(source, str(root()))
    if not pts:
        raise ValueError(f"No {SOURCE_INFO[source][0]} files found in {folder(source)}.")
    arr = np.array([(p[0], p[1]) for p in pts])
    dy = (arr[:, 0] - lat) * 111.0
    dx = (arr[:, 1] - lon) * 111.0 * np.cos(np.deg2rad(lat))
    d = np.hypot(dx, dy)
    order = np.argsort(d)[:k]
    return [(pts[i][2], float(d[i])) for i in order]


def weights_of(dists: List[float]) -> List[float]:
    """Inverse-distance weights; a point within 1 km takes everything."""
    if dists[0] < 1.0:
        return [1.0] + [0.0] * (len(dists) - 1)
    w = np.array([1.0 / d for d in dists])
    return list(w / w.sum())


def icon_radiation(ml_path: str, lead_from: int, lead_to: int) -> Optional[pd.DataFrame]:
    """Surface radiation of an ICON-D2 point from its single-level (SL) file, which sits next to the ML tree:
    .../ML/<run>/<station>/<point>_ML.parquet -> .../SL/<run>/<point>_SL.parquet. The hourly means (`_avg`) are used.
    Returns starttime, lead and the three radiation columns, or None when the SL file is missing."""
    p = Path(ml_path)
    sl = p.parents[3] / "SL" / p.parents[1].name / p.name.replace("_ML.parquet", "_SL.parquet")
    if not sl.exists():
        return None
    d = pd.read_parquet(sl, columns=["starttime", "forecasttime", "aswdir_s_avg", "aswdifd_s_avg"])
    d = d[(d["forecasttime"] >= lead_from) & (d["forecasttime"] <= lead_to) & (d["forecasttime"] % 1 == 0)]
    d = d.drop_duplicates(["starttime", "forecasttime"])
    direct = d["aswdir_s_avg"].clip(lower=0).astype(float)
    diffuse = d["aswdifd_s_avg"].clip(lower=0).astype(float)
    return pd.DataFrame({"starttime": pd.to_datetime(d["starttime"], utc=True),
                         "lead": d["forecasttime"].to_numpy(dtype=float),
                         "direct_radiation": direct.to_numpy(), "diffuse_radiation": diffuse.to_numpy(),
                         "shortwave_radiation": (direct + diffuse).to_numpy()})


def _icon_run_frame(path: str, lead_from: int, lead_to: int) -> pd.DataFrame:
    f = parkdata.icon_frame(pd.read_parquet(path), lead_from, lead_to)
    if f.empty:
        return f
    rad = icon_radiation(path, lead_from, lead_to)
    if rad is not None:
        f = f.merge(rad, on=["starttime", "lead"], how="left")
    return f


def ecmwf_radiation(wind_path: str, lead_from: int, lead_to: int) -> Optional[pd.DataFrame]:
    """Surface radiation of an ECMWF point from the `solar` folder next to the wind folders:
    .../parquet/SL/<point>_wind_sl.parquet -> .../parquet/solar/<point>_solar_sl.parquet.
    `ssrd` (global) and `fdir` (direct on the horizontal) are hourly means in W/m²; diffuse = global - direct.
    Returns None when the solar file is missing."""
    p = Path(wind_path)
    sol = p.parents[1] / "solar" / p.name.replace("_wind_sl.parquet", "_solar_sl.parquet")
    if not sol.exists():
        return None
    d = pd.read_parquet(sol, columns=["starttime", "forecasttime", "ssrd", "fdir"])
    d = d[(d["forecasttime"] >= lead_from) & (d["forecasttime"] <= lead_to)].dropna(subset=["ssrd", "fdir"])
    total = d["ssrd"].clip(lower=0).astype(float)
    direct = d["fdir"].clip(lower=0).astype(float).clip(upper=total)
    return pd.DataFrame({"starttime": pd.to_datetime(d["starttime"], utc=True),
                         "lead": d["forecasttime"].to_numpy(dtype=float),
                         "shortwave_radiation": total.to_numpy(), "direct_radiation": direct.to_numpy(),
                         "diffuse_radiation": (total - direct).to_numpy()})


def _ecmwf_run_frame(path: str, lead_from: int, lead_to: int) -> pd.DataFrame:
    f = parkdata.ecmwf_frame(pd.read_parquet(path), lead_from, lead_to)
    if f.empty:
        return f
    rad = ecmwf_radiation(path, lead_from, lead_to)
    if rad is not None:                                         # lead 0 has no radiation: that hour falls back to the next run
        f = f.merge(rad, on=["starttime", "lead"], how="inner")
    return f


@lru_cache(maxsize=8)
def _load(source: str, root_str: str, lat: float, lon: float, lead_from: int, lead_to: int) -> Tuple[pd.DataFrame, dict]:
    near = nearest(source, lat, lon)
    if near[0][1] > MAX_KM:
        raise ValueError(f"The nearest {SOURCE_INFO[source][0]} grid point is {near[0][1]:.0f} km from the park "
                         f"(limit {MAX_KM:.0f} km). Choose another weather model.")
    w = weights_of([d for _, d in near])
    series: Dict[str, pd.DataFrame] = {}
    weights: Dict[str, float] = {}
    for (paths, _), wi in zip(near, w):
        if wi <= 0:
            continue
        if source == "srv_icon_d2":
            frames = [_icon_run_frame(p, lead_from, lead_to) for p in paths]
        else:
            frames = [_ecmwf_run_frame(p, lead_from, lead_to) for p in paths]
        frames = [f for f in frames if not f.empty]
        if not frames:
            continue
        series[paths[0]] = parkdata.pick_latest(pd.concat(frames, ignore_index=True))
        weights[paths[0]] = wi
    if not series:
        raise ValueError(f"No forecasts with a lead time of {lead_from} to {lead_to} h at this location.")
    wx = parkdata.combine_points(series, weights)
    if wx.empty:
        raise ValueError("The grid points share no forecast hours in this lead-time window. Try a wider window.")
    note = {"source": source, "grid_points": len(series), "distance_km": round(near[0][1], 1),
            "lead_from": lead_from, "lead_to": lead_to,
            "start": str(wx.index.min()), "end": str(wx.index.max())}
    return wx, note


def load_weather(source: str, lat: float, lon: float, lead_from: int, lead_to: int) -> Tuple[pd.DataFrame, dict]:
    """Hourly weather at (lat, lon) from the forecasts with a lead time in [lead_from, lead_to] hours."""
    if source not in SOURCE_INFO:
        raise ValueError(f"Unknown weather source '{source}'")
    if lead_from > lead_to:
        raise ValueError("The lead-time window must start before it ends.")
    return _load(source, str(root()), round(float(lat), 4), round(float(lon), 4), int(lead_from), int(lead_to))
