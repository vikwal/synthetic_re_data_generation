"""Archived NWP forecasts (ICON-D2, ECMWF IFS) as an alternative weather source for the pre-trained model tab.

Optional local archive, one folder per model (not offered in the form; the package and server sources are used instead):
    streamlit/data/parquet/grid_points_icon_d2/grid_<LAT>_<LON>.parquet
    streamlit/data/parquet/grid_points_ecmwf/grid_<LAT>_<LON>.parquet
Each file is hourly UTC with u/v wind (10 m, 100 m), temp_2m, dew_point_2m, pressure and irradiance columns.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import settings

LABELS = {
    "icon_d2": ("ICON-D2", "DWD high-resolution model (2 km), Central Europe."),
    "ecmwf": ("ECMWF IFS", "ECMWF global model (0.25°)."),
}


def _files(model: str) -> List[Tuple[float, float, Path]]:
    folder = settings.NWP_DIRS[model]
    out = []
    if folder.exists():
        for f in folder.iterdir():
            if f.suffix in (".parquet", ".csv") and f.name.startswith("grid_"):
                parts = f.stem.split("_")
                try:
                    out.append((float(parts[1]), float(parts[2]), f))
                except (IndexError, ValueError):
                    continue
    return out


def coverage(model: str) -> Optional[Tuple[str, str]]:
    """First and last day available across the model's downloaded files."""
    lo = hi = None
    for _, _, path in _files(model):
        try:
            idx = _read(str(path)).index
        except Exception:  # noqa: BLE001 - unreadable file: ignore for the date hint
            continue
        lo = idx.min() if lo is None else min(lo, idx.min())
        hi = idx.max() if hi is None else max(hi, idx.max())
    return None if lo is None else (str(lo.date()), str(hi.date()))


def sources() -> List[dict]:
    """Weather sources the pre-train tab can offer: the generated run's own (ERA5) plus every NWP model with data."""
    from . import parkdata, servernwp                          # sample-package runs; forecasts stored on the server
    if servernwp.root() is not None:                           # on the server only its own ICON-D2 and ECMWF are offered
        out = []
        for sid, (label, desc, _folder, lead_max) in servernwp.SOURCE_INFO.items():
            n = servernwp.status(sid)["points"]
            out.append({"id": sid, "label": label, "description": desc, "available": n > 0, "points": n,
                        "start": None, "end": None, "package": True, "server": True, "features": servernwp.FEATURES[sid],
                        "lead": {"min": 0, "max": lead_max, "default": list(parkdata.DEFAULT_LEAD)},
                        "reason": None if n else f"No files found in {servernwp.folder(sid)}."})
        return out
    res = [{"id": "era5", "label": "Generated run (ERA5)", "description": "The weather the run was generated from.",
            "available": True, "reason": None, "points": None, "start": None, "end": None}]
    for key, (label, desc) in LABELS.items():
        n = len(_files(key))
        cov = coverage(key) if n else None
        res.append({"id": key, "label": label, "description": desc, "available": n > 0, "points": n, "hidden": True,   # not offered in the form for now
                    "start": cov[0] if cov else None, "end": cov[1] if cov else None,
                    "reason": None if n else "No data found."})
    found = parkdata.package_dir() is not None
    for sid, (label, desc, _model, lead_max) in parkdata.SOURCE_INFO.items():
        res.append({"id": sid, "label": label, "description": desc, "available": found, "points": None,
                    "start": None, "end": None, "package": True, "features": parkdata.FEATURES[sid],
                    "lead": {"min": 0, "max": lead_max, "default": list(parkdata.DEFAULT_LEAD)},
                    "reason": None if found else "Sample-data package not found. Put it in data/ or set SYNTH_RE_PACKAGE_DIR."})
    return res


def is_package(source: str) -> bool:
    """Sources that come from the sample package (they need parks imported from it)."""
    return source in ("pkg_icon_d2", "pkg_ecmwf")


def has_lead(source: str) -> bool:
    """Sources made of forecast runs, so a lead-time window applies."""
    from . import servernwp
    return is_package(source) or servernwp.is_server(source)


def weather_for(source: str, meta: Dict, lead: Optional[Tuple[int, int]] = None) -> Tuple[pd.DataFrame, dict]:
    """Weather of a generated run from the chosen source. Package sources need a run imported from the package."""
    if is_package(source):
        from . import parkdata
        park = meta.get("package_park_id")
        if not park:
            raise ValueError(f"{parkdata.SOURCE_INFO[source][0]} forecasts only exist for the imported wind parks. "
                             "Choose one of those parks, or another weather model.")
        lo, hi = lead or parkdata.DEFAULT_LEAD
        return parkdata.load_weather(source, park, int(lo), int(hi))
    from . import servernwp
    if servernwp.is_server(source):
        lo, hi = lead or parkdata.DEFAULT_LEAD
        return servernwp.load_weather(source, *site_of(meta), int(lo), int(hi))
    return load_weather(source, *site_of(meta))


def weather_multi(sources: List[str], meta: Dict, lead: Optional[Tuple[int, int]] = None) -> Tuple[pd.DataFrame, List[dict]]:
    """Weather of a run from several models at once. Every column is prefixed with its model (`srv_icon_d2:temp_2m`), so
    the same input from two models stays two separate columns. Only hours present in every model are kept."""
    frames, notes = [], []
    for s in sources:
        wx, note = weather_for(s, meta, lead if has_lead(s) else None)
        frames.append(wx.add_prefix(f"{s}:"))
        notes.append(note)
    idx = frames[0].index
    for f in frames[1:]:
        idx = idx.intersection(f.index)
    if len(idx) == 0:
        raise ValueError("The chosen weather models have no hours in common. Pick models that cover the same period.")
    return pd.concat([f.loc[idx] for f in frames], axis=1).sort_index(), notes


def _nearest(model: str, lat: float, lon: float) -> Tuple[float, float, Path, float]:
    files = _files(model)
    if not files:
        raise ValueError(f"No {LABELS[model][0]} data found in {settings.NWP_DIRS[model]}.")
    la, lo, path = min(files, key=lambda t: (t[0] - lat) ** 2 + ((t[1] - lon) * np.cos(np.deg2rad(lat))) ** 2)
    dist_km = float(np.hypot((la - lat) * 111.0, (lo - lon) * 111.0 * np.cos(np.deg2rad(lat))))
    return la, lo, path, dist_km


@lru_cache(maxsize=16)
def _read(path: str) -> pd.DataFrame:
    p = Path(path)
    df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p, index_col=0)
    if not isinstance(df.index, pd.DatetimeIndex):
        for col in ("time", "timestamp"):
            if col in df.columns:
                df = df.set_index(col)
                break
    df.index = pd.to_datetime(df.index)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    out = df.sort_index().copy()
    for h in ("10", "100"):                                   # speed from the components
        if f"u_wind_{h}m" in out.columns and f"v_wind_{h}m" in out.columns:
            out[f"wind_speed_{h}m"] = np.hypot(out[f"u_wind_{h}m"], out[f"v_wind_{h}m"])
    return out


def load_weather(model: str, lat: float, lon: float) -> Tuple[pd.DataFrame, dict]:
    """Hourly weather of the NWP grid point nearest to (lat, lon) plus a note on which point was used."""
    if model not in LABELS:
        raise ValueError(f"Unknown weather source '{model}'")
    la, lo, path, dist = _nearest(model, lat, lon)
    return _read(str(path)), {"source": model, "grid_lat": la, "grid_lon": lo, "distance_km": round(dist, 1)}


def site_of(meta: Dict) -> Tuple[float, float]:
    """Representative location of a generated run (park centroid or solar site)."""
    params = meta.get("params") or {}
    turbines = params.get("turbines") or []
    if turbines:
        return float(np.mean([t["lat"] for t in turbines])), float(np.mean([t["lon"] for t in turbines]))
    if params.get("lat") is not None:
        return float(params["lat"]), float(params["lon"])
    sites = meta.get("sites") or []
    if sites:
        return float(np.mean([s["lat"] for s in sites])), float(np.mean([s["lon"] for s in sites]))
    raise ValueError("The run has no location to look up NWP weather for.")
