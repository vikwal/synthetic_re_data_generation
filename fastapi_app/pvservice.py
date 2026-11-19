"""Weather data for solar scenarios from an external service, cached per user and location.

Flow for a location:
  1. look in the user's cache (<user storage>/pvdata/<location>.csv.gz) -> use it if it covers the configured period
  2. otherwise request the service once, check the answer, store it, use it

The service is a placeholder for now. Set SYNTH_RE_PV_SERVICE_URL (and SYNTH_RE_PV_SERVICE_TOKEN) to call a real one; with no URL a
built-in stand-in answers with synthetic hourly weather, so the whole request-and-cache flow can be tried without any service.

What the service must answer (JSON, one request per location, hourly, UTC):
  GET <URL>?lat=..&lon=..&start=YYYY-MM-DD&end=YYYY-MM-DD        (header  Authorization: Bearer <token>  if a token is set)
  {"timestamps": ["2023-01-01T00:00:00Z", ...],
   "ghi": [W/m2], "dhi": [W/m2], "temperature_2m": [degC], "wind_speed": [m/s],
   "altitude": 123.0}                                             (altitude in m is optional)
"""
from __future__ import annotations

import hashlib
import json
import threading
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import requests

from . import settings, storage

COLUMNS = ("ghi", "dhi", "temperature_2m", "wind_speed")
LOCATION_DECIMALS = 2                       # 0.01 degrees, about 1 km: nearby clicks share one cached location

_locks: Dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


# ---------------------------------------------------------------- location + cache
def snap(lat: float, lon: float) -> Tuple[float, float]:
    """The location that is requested and cached (rounded, so the same place always gives the same cache entry)."""
    return round(float(lat), LOCATION_DECIMALS), round(float(lon), LOCATION_DECIMALS)


def location_key(lat: float, lon: float) -> str:
    la, lo = snap(lat, lon)
    return f"lat{la:.{LOCATION_DECIMALS}f}_lon{lo:.{LOCATION_DECIMALS}f}".replace("-", "m")


def _cache_dir():
    d = storage._root() / "pvdata"                  # per user when logins are on
    d.mkdir(parents=True, exist_ok=True)
    return d


def _lock_for(path: str) -> threading.Lock:
    with _locks_guard:
        return _locks.setdefault(path, threading.Lock())


def _period() -> Tuple[str, str]:
    return settings.PV_SERVICE_START, settings.PV_SERVICE_END


def _read_cache(data_path, meta_path) -> Optional[Tuple[pd.DataFrame, dict]]:
    if not (data_path.exists() and meta_path.exists()):
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if (meta.get("start"), meta.get("end")) != _period():
            return None                                 # the configured period changed: ask the service again
        df = pd.read_csv(data_path, index_col=0, parse_dates=True)
        if df.empty or not set(COLUMNS).issubset(df.columns):
            return None
        return df, meta
    except (OSError, ValueError, KeyError):
        return None                                     # damaged cache entry: simply request again


def _write_cache(data_path, meta_path, df: pd.DataFrame, meta: dict) -> None:
    tmp = data_path.with_name(data_path.name + ".tmp")
    df.to_csv(tmp, compression="gzip")
    tmp.replace(data_path)
    meta_path.write_text(json.dumps(meta, indent=1), encoding="utf-8")


# ---------------------------------------------------------------- the service (placeholder)
def _mock_payload(lat: float, lon: float, start: str, end: str) -> dict:
    """Stand-in for the external service: plausible hourly weather for any place, built from simple solar geometry and a
    seeded random cloud cover (the same location always gives the same series). Replace by a real service via the URL setting."""
    idx = pd.date_range(start, pd.Timestamp(end) + pd.Timedelta(hours=23), freq="h")
    n = len(idx)
    doy = idx.dayofyear.values.astype(float)
    hour = idx.hour.values + 0.5
    decl = np.radians(23.44) * np.sin(2 * np.pi * (doy - 81) / 365.0)
    ha = np.radians((hour - 12.0) * 15.0 + lon)                          # hour angle, UTC time -> local solar time via longitude
    phi = np.radians(lat)
    sin_el = np.sin(phi) * np.sin(decl) + np.cos(phi) * np.cos(decl) * np.cos(ha)
    sin_el = np.clip(sin_el, 0.0, None)
    clear = 1050.0 * sin_el ** 1.15

    seed = int(hashlib.sha256(location_key(lat, lon).encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    cloud = np.empty(n)                                                  # slowly varying cloud cover 0..1
    x = rng.uniform(0.2, 0.8)
    for i in range(n):
        x = float(np.clip(0.97 * x + 0.03 * 0.55 + rng.normal(0, 0.09), 0.0, 1.0))
        cloud[i] = x
    ghi = clear * (1.0 - 0.75 * cloud ** 1.4)
    dhi = np.minimum(ghi, ghi * (0.15 + 0.75 * cloud))
    season = np.sin(2 * np.pi * (doy - 110) / 365.0)
    temp = 9.0 + 10.0 * season + 4.0 * np.sin(2 * np.pi * (hour - 9.0) / 24.0) + rng.normal(0, 1.0, n)
    wind = np.clip(3.5 + 1.6 * np.abs(rng.normal(0, 1, n)) + 1.0 * cloud, 0.2, 25.0)
    return {
        "timestamps": [t.strftime("%Y-%m-%dT%H:%M:%SZ") for t in idx],
        "ghi": np.round(ghi, 2).tolist(), "dhi": np.round(dhi, 2).tolist(),
        "temperature_2m": np.round(temp, 2).tolist(), "wind_speed": np.round(wind, 2).tolist(),
        "altitude": None, "source": "placeholder",
    }


def request_service(lat: float, lon: float, start: str, end: str) -> dict:
    """One request to the service (or the built-in stand-in when no URL is set). Returns the raw JSON answer."""
    if not settings.PV_SERVICE_URL:
        return _mock_payload(lat, lon, start, end)
    headers = {"Authorization": f"Bearer {settings.PV_SERVICE_TOKEN}"} if settings.PV_SERVICE_TOKEN else {}
    try:
        resp = requests.get(settings.PV_SERVICE_URL, params={"lat": lat, "lon": lon, "start": start, "end": end},
                            headers=headers, timeout=settings.PV_SERVICE_TIMEOUT)
    except requests.RequestException as exc:
        raise RuntimeError(f"The weather data service could not be reached: {exc.__class__.__name__}") from exc
    if resp.status_code != 200:
        raise RuntimeError(f"The weather data service answered with status {resp.status_code}.")
    try:
        return resp.json()
    except ValueError as exc:
        raise ValueError("The weather data service did not return valid JSON.") from exc


def parse_payload(payload: dict) -> Tuple[pd.DataFrame, dict]:
    """Check the service answer and turn it into an hourly frame (UTC, no time zone) plus a small meta dict."""
    try:
        ts = pd.to_datetime(payload["timestamps"], utc=True).tz_convert(None)
        cols = {c: pd.to_numeric(pd.Series(payload[c]), errors="coerce") for c in COLUMNS}
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("The weather data service answer is incomplete (needs timestamps, ghi, dhi, temperature_2m, wind_speed).") from exc
    if len(ts) == 0 or any(len(s) != len(ts) for s in cols.values()):
        raise ValueError("The weather data service answer has columns of different length or is empty.")
    df = pd.DataFrame({c: s.values for c, s in cols.items()}, index=pd.DatetimeIndex(ts, name="timestamp")).sort_index()
    df = df[~df.index.duplicated()]
    if df.isna().all().any() or not np.isfinite(df.to_numpy(dtype=float, na_value=0.0)).all():
        raise ValueError("The weather data service answer contains unusable values.")
    df["ghi"] = df["ghi"].clip(lower=0)
    df["dhi"] = df["dhi"].clip(lower=0).clip(upper=df["ghi"])
    alt = payload.get("altitude")
    meta = {"altitude": float(alt) if isinstance(alt, (int, float)) and np.isfinite(alt) else None,
            "service": payload.get("source") or ("external" if settings.PV_SERVICE_URL else "placeholder")}
    return df, meta


# ---------------------------------------------------------------- public entry point
def get_location_data(lat: float, lon: float, progress: Optional[Callable] = None) -> Tuple[pd.DataFrame, dict]:
    """Hourly weather for a location: from this user's cache when present, else requested once and cached.
    Returns (frame with ghi, dhi [W/m2], temperature_2m [degC], wind_speed [m/s]; info dict)."""
    la, lo = snap(lat, lon)
    key = location_key(la, lo)
    folder = _cache_dir()
    data_path, meta_path = folder / f"{key}.csv.gz", folder / f"{key}.json"
    with _lock_for(str(data_path)):                     # two runs for the same place must not both call the service
        hit = _read_cache(data_path, meta_path)
        if hit is not None:
            _say(progress, 0.2, "Using stored weather data for the location")
            df, meta = hit
            return df, {**meta, "cached": True}
        start, end = _period()
        _say(progress, 0.2, "Requesting weather data for the location")
        df, info = parse_payload(request_service(la, lo, start, end))
        meta = {"lat": la, "lon": lo, "start": start, "end": end, "rows": int(len(df)),
                "fetched_at": storage.now_iso(), **info}
        _write_cache(data_path, meta_path, df, meta)
        return df, {**meta, "cached": False}


def _say(progress, fraction: float, message: str) -> None:
    if progress is not None:
        progress(fraction, message)
