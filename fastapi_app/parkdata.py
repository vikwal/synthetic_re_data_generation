"""Sample-data package: synthetic wind parks plus archived NWP forecast runs.

Folder layout (see the package's own README):
    <package>/wind/parks.csv, park_layouts.csv
    <package>/wind/<park_id>/synth.parquet, technik.json
    <package>/wind/<park_id>/nwp/icon-d2/<HH>/<point_key>_ML.parquet            (+ turbine_points.csv)
    <package>/wind/<park_id>/nwp/ecmwf/SL/<HH>/<point_key>_wind_sl.parquet       (+ turbine_points.csv)

* `import_wind_parks` stores every park's power as an ordinary result, so it shows up next to your own runs.
* `load_weather` turns the forecast runs of a park into one hourly weather series: for each hour it uses the latest
  run whose lead time lies in the chosen window, and averages the park's grid points with the turbine weights.
"""
from __future__ import annotations

import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import analytics, settings, storage

# id -> (label, description, model folder, lead-time limit in hours)
SOURCE_INFO = {
    "pkg_icon_d2": ("ICON-D2",
                    "DWD high-resolution model (2 km): forecast runs at the park's own grid points.", "icon-d2", 48),
    "pkg_ecmwf": ("ECMWF IFS",
                  "ECMWF global model: forecast runs at the park's own grid points.", "ecmwf", 57),
}
FEATURES = {
    "pkg_icon_d2": ["wind_speed_100m", "u_wind_100m", "v_wind_100m", "wind_speed_10m", "u_wind_10m", "v_wind_10m",
                    "temp_2m", "pressure", "dew_point_2m"],
    "pkg_ecmwf": ["wind_speed_100m", "u_wind_100m", "v_wind_100m", "wind_speed_10m", "u_wind_10m", "v_wind_10m",
                  "temp_2m", "dew_point_2m"],
}
DEFAULT_LEAD = (12, 36)                  # roughly a day-ahead forecast


# ------------------------------------------------------------------ package discovery
def package_dir() -> Optional[Path]:
    """Folder of the sample package, or None. SYNTH_RE_PACKAGE_DIR overrides the default `data/beispieldaten_*`."""
    cands: List[Path] = []
    env = os.environ.get("SYNTH_RE_PACKAGE_DIR")
    if env:
        cands.append(Path(env))
    base = settings.PROJECT_ROOT / "data"
    if base.exists():
        cands += sorted(base.glob("beispieldaten_*"), reverse=True)
    for c in cands:
        if (c / "wind" / "parks.csv").exists():
            return c
    return None


def result_id(park_id: str) -> str:
    """Stable 12-hex id, so importing the same park twice replaces it instead of adding a copy."""
    return hashlib.md5(f"pkg:{park_id}".encode()).hexdigest()[:12]


def wind_parks() -> List[dict]:
    root = package_dir()
    if root is None:
        return []
    df = pd.read_csv(root / "wind" / "parks.csv")
    if "technology" in df.columns:
        df = df[df["technology"] == "wind"]
    out = []
    for r in df.itertuples():
        if (root / "wind" / r.park_id / "synth.parquet").exists():
            out.append({"park_id": r.park_id, "name": getattr(r, "name", None) or r.park_id,
                        "capacity_kw": float(r.capacity_kw), "n_turbines": int(getattr(r, "n_turbines", 0) or 0),
                        "wake_efficiency": getattr(r, "wake_efficiency", None)})
    return out


def status() -> dict:
    root = package_dir()
    parks = wind_parks() if root else []
    imported = 0
    for p in parks:
        try:
            storage.get_result_meta(result_id(p["park_id"]))
            imported += 1
        except storage.NotFound:
            pass
    return {"found": root is not None, "n_wind": len(parks), "n_imported": imported,
            "folder": root.name if root else None}


# ------------------------------------------------------------------ importing the parks' power
def _layout(root: Path, park_id: str) -> pd.DataFrame:
    df = pd.read_csv(root / "wind" / "park_layouts.csv")
    return df[df["park_id"] == park_id].reset_index(drop=True)


def build_result(park_id: str) -> Tuple[dict, pd.DataFrame]:
    """(meta, dataframe) of one park in the format of a generated result (`power` in W)."""
    root = package_dir()
    if root is None:
        raise ValueError("The sample-data package was not found. Put it in data/ or set SYNTH_RE_PACKAGE_DIR.")
    pdir = root / "wind" / park_id
    parks = {p["park_id"]: p for p in wind_parks()}
    if park_id not in parks:
        raise ValueError(f"Unknown park '{park_id}'.")
    info = parks[park_id]
    df = pd.read_parquet(pdir / "synth.parquet")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    tech = json.loads((pdir / "technik.json").read_text())

    out = df.copy()
    out["power"] = df["power_park"].astype(float)
    groups = tech.get("groups", [])
    wts = {g["group_id"]: float(g.get("n", 1)) * float(g.get("rated_cap_kw", 1.0)) for g in groups}
    hub = [(g["group_id"], wts[g["group_id"]]) for g in groups if f"wind_speed_hub_{g['group_id']}" in df.columns]
    if hub:                                                   # capacity-weighted hub-height wind, for the dashboard
        tot = sum(w for _, w in hub)
        out["wind_speed_hub"] = sum(df[f"wind_speed_hub_{g}"] * w for g, w in hub) / tot
    if "density" in df.columns:
        out["density_hub"] = df["density"]
    out["temperature_hub"] = df["temp_2m"] if "temp_2m" in df.columns else np.nan

    dt = analytics.step_hours(out)
    net = float(np.nansum(out["power"]) * dt / 1e6)
    free = float(np.nansum(df["power_park_free"]) * dt / 1e6) if "power_park_free" in df.columns else net
    ideal = 0.0
    for g in groups:
        gid = g["group_id"]
        if f"power_{gid}" in df.columns and f"aging_factor_{gid}" in df.columns:
            ideal += float((df[f"power_{gid}"] / df[f"aging_factor_{gid}"].where(df[f"aging_factor_{gid}"] > 0)).sum(skipna=True) * dt / 1e6)
    ideal = max(ideal, free)
    capacity_w = info["capacity_kw"] * 1000.0
    eff = info.get("wake_efficiency")
    summary = {
        "n_turbines": info["n_turbines"], "capacity_w": capacity_w, "wake_applied": True, "wake_decay": 0.075,
        "park_wake_loss_pct": None if eff is None or pd.isna(eff) else float(100.0 * (1.0 - float(eff))),
        "ageing_applied": True, "energies": analytics.energy_waterfall(ideal, free, net), "warnings": [],
        "start": str(out.index.min().date()), "end": str(out.index.max().date()),
    }
    lay = _layout(root, park_id)
    turbines = [{"lat": float(r.latitude), "lon": float(r.longitude), "turbine": r.lib_name, "hub_height": float(r.hub_height),
                 "commissioning_date": str(r.commissioning_date)} for r in lay.itertuples()]
    sites = [{"slot": i + 1, **t, "rated_mw": float(r.rated_cap_kw) / 1000.0}
             for i, (t, r) in enumerate(zip(turbines, lay.itertuples()))]
    meta = {
        "scenario_id": None, "scenario_name": info["name"], "kind": "wind",
        "params": {"turbines": turbines, "apply_ageing": True, "apply_wake": True, "wake_decay": 0.075},
        "summary": summary, "kpis": analytics.kpis(out, "wind", capacity_w), "sites": sites, "package_park_id": park_id,
    }
    return meta, out


def import_wind_parks(park_ids: Optional[List[str]] = None, progress=None) -> List[str]:
    ids = park_ids or [p["park_id"] for p in wind_parks()]
    rids = []
    for i, pid in enumerate(ids):
        if progress:
            progress(i / max(len(ids), 1), f"Importing {pid}")
        meta, df = build_result(pid)
        rids.append(storage.save_result(meta, df, rid=result_id(pid)))
    ensure_scenarios(force=True)
    if progress:
        progress(1.0, "Done")
    return rids


_SEEN: set = set()          # results already looked at in this process
_SUFFIX = " (package)"      # old name ending, removed from existing imports


def ensure_scenarios(force: bool = False) -> int:
    """Give every imported package park a scenario of its own, so it shows up in the Workspace list and opens on its
    results. The scenario id equals the result id, so this is safe to repeat; a park whose scenario was deleted is gone
    (deleting a scenario deletes its results). Returns the number of scenarios created."""
    if force:
        _SEEN.clear()
    existing = {s["id"]: s for s in storage.list_scenarios()}
    made = 0
    for meta in storage.result_metas():
        rid = meta.get("id")
        if not meta.get("package_park_id") or not rid or rid in _SEEN:
            continue
        _SEEN.add(rid)
        name = (meta.get("scenario_name") or rid)
        clean = name[: -len(_SUFFIX)] if name.endswith(_SUFFIX) else name       # older imports were named "<park> (package)"
        if clean != meta.get("scenario_name"):
            storage.update_result_meta(rid, {"scenario_name": clean})
        if rid in existing:
            if existing[rid].get("name", "").endswith(_SUFFIX):
                storage.update_scenario(rid, {"name": existing[rid]["name"][: -len(_SUFFIX)]})
            continue
        if meta.get("scenario_id") not in (None, rid):
            continue
        storage.create_scenario(clean, "wind", meta.get("params") or {}, sid=rid,
                                last_result_id=rid, when=meta.get("created"))
        storage.set_result_scenario(rid, rid)
        made += 1
    return made


# ------------------------------------------------------------------ forecast runs -> hourly weather
def _to_utc(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s)
    return s.dt.tz_localize("UTC") if s.dt.tz is None else s.dt.tz_convert("UTC")


def _dew_point_k(q: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Dew point (K) from specific humidity (kg/kg) and pressure (Pa), Magnus formula."""
    e = np.clip(q * p / (0.622 + 0.378 * q), 1e-3, None)
    a = np.log(e / 611.2)
    return 243.5 * a / (17.67 - a) + 273.15


def icon_frame(df: pd.DataFrame, lo: int, hi: int) -> pd.DataFrame:
    """ICON-D2 model-level rows of one grid point -> one row per (run, lead): winds at 10/100 m, T, p, dew point.
    The model levels are layers (`bottomlevel`..`toplevel`, in m); each layer's mid height carries its values and
    winds are interpolated linearly in height."""
    d = df[(df["forecasttime"] >= lo) & (df["forecasttime"] <= hi)].copy()
    cols = ["valid", "starttime", "lead", "u_wind_10m", "v_wind_10m", "u_wind_100m", "v_wind_100m",
            "temp_2m", "pressure", "dew_point_2m"]
    if d.empty:
        return pd.DataFrame(columns=cols)
    d["starttime"] = _to_utc(d["starttime"])
    d["z"] = (d["toplevel"] + d["bottomlevel"]) / 2.0
    key = ["starttime", "forecasttime"]
    d = d.sort_values(key + ["z"])
    cnt = d.groupby(key)["z"].transform("size")
    n_lev = int(cnt.mode().iloc[0])
    d = d[cnt == n_lev]                                       # drop runs with an incomplete set of levels
    z = d["z"].to_numpy().reshape(-1, n_lev)
    ok = np.all(np.abs(z - z[0]) < 0.5, axis=1)               # heights must be the same in every run
    z0 = z[0]

    def grid(c: str) -> np.ndarray:
        return d[c].to_numpy(dtype=float).reshape(-1, n_lev)[ok]

    def at(a: np.ndarray, h: float) -> np.ndarray:
        if h <= z0[0]:
            return a[:, 0]
        if h >= z0[-1]:
            return a[:, -1]
        j = int(np.searchsorted(z0, h))
        f = (h - z0[j - 1]) / (z0[j] - z0[j - 1])
        return (1 - f) * a[:, j - 1] + f * a[:, j]

    u, v, t, p, q = grid("u_wind"), grid("v_wind"), grid("temperature"), grid("pressure"), grid("qs")
    keys = d[key].iloc[::n_lev].reset_index(drop=True)[ok]
    out = pd.DataFrame({
        "starttime": keys["starttime"].to_numpy(), "lead": keys["forecasttime"].to_numpy(dtype=float),
        "u_wind_10m": at(u, 10.0), "v_wind_10m": at(v, 10.0), "u_wind_100m": at(u, 100.0), "v_wind_100m": at(v, 100.0),
        "temp_2m": t[:, 0], "pressure": p[:, 0], "dew_point_2m": _dew_point_k(q[:, 0], p[:, 0]),
    })
    out["starttime"] = pd.to_datetime(out["starttime"], utc=True)
    out["valid"] = out["starttime"] + pd.to_timedelta(out["lead"], unit="h")
    return out[cols]


def ecmwf_frame(df: pd.DataFrame, lo: int, hi: int) -> pd.DataFrame:
    """ECMWF single-level rows of one grid point -> the same columns as `icon_frame` (no surface pressure)."""
    d = df[(df["forecasttime"] >= lo) & (df["forecasttime"] <= hi)]
    cols = ["valid", "starttime", "lead", "u_wind_10m", "v_wind_10m", "u_wind_100m", "v_wind_100m", "temp_2m", "dew_point_2m"]
    if d.empty:
        return pd.DataFrame(columns=cols)
    out = pd.DataFrame({
        "starttime": _to_utc(d["starttime"]).to_numpy(), "lead": d["forecasttime"].to_numpy(dtype=float),
        "u_wind_10m": d["u_wind10m"].to_numpy(float), "v_wind_10m": d["v_wind10m"].to_numpy(float),
        "u_wind_100m": d["u_wind100m"].to_numpy(float), "v_wind_100m": d["v_wind100m"].to_numpy(float),
        "temp_2m": d["temp2m"].to_numpy(float), "dew_point_2m": d["dew_point2m"].to_numpy(float),
    })
    out["starttime"] = pd.to_datetime(out["starttime"], utc=True)
    out["valid"] = out["starttime"] + pd.to_timedelta(out["lead"], unit="h")
    return out[cols]


def pick_latest(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per valid hour: the most recent run (shortest lead) among those already in the lead window."""
    if frame.empty:
        return frame.set_index("valid") if "valid" in frame.columns else frame
    f = frame.sort_values(["valid", "starttime"]).drop_duplicates("valid", keep="last")
    return f.set_index("valid").drop(columns=["starttime", "lead"])


def combine_points(frames: Dict[str, pd.DataFrame], weights: Dict[str, float]) -> pd.DataFrame:
    """Weighted mean of the points' series. An hour is kept only when every point has a value, so the weights always
    add up to one. Wind speeds are computed from the averaged components."""
    keys = [k for k in frames if k in weights and weights[k] > 0]
    if not keys:
        raise ValueError("No grid-point files matched the turbine weights.")
    w = np.array([weights[k] for k in keys], dtype=float)
    w = w / w.sum()
    cols = sorted(set.intersection(*[set(frames[k].columns) for k in keys]))
    idx = frames[keys[0]].index
    for k in keys[1:]:
        idx = idx.intersection(frames[k].index)
    out = pd.DataFrame(index=idx)
    for c in cols:
        out[c] = sum(frames[k].loc[idx, c].to_numpy(float) * wi for k, wi in zip(keys, w))
    for h in ("10", "100"):
        if f"u_wind_{h}m" in out.columns:
            out[f"wind_speed_{h}m"] = np.hypot(out[f"u_wind_{h}m"], out[f"v_wind_{h}m"])
    out.index.name = None
    return out.sort_index()


def _weights(run_dir: Path) -> Dict[str, float]:
    tp = pd.read_csv(run_dir / "turbine_points.csv")
    w = tp.groupby("point_key")["weight"].sum()
    return (w / w.sum()).to_dict()


def _run_dirs(park_id: str, source: str) -> List[Path]:
    root = package_dir()
    if root is None:
        raise ValueError("The sample-data package was not found.")
    model = SOURCE_INFO[source][2]
    base = root / "wind" / park_id / "nwp" / model / ("SL" if model == "ecmwf" else "")
    dirs = sorted(p for p in base.iterdir() if p.is_dir()) if base.exists() else []
    if not dirs:
        raise ValueError(f"No {SOURCE_INFO[source][0]} files found for park {park_id}.")
    return dirs


@lru_cache(maxsize=6)
def load_weather(source: str, park_id: str, lead_from: int, lead_to: int) -> Tuple[pd.DataFrame, dict]:
    """Hourly weather for a park from the forecast runs with lead time in [lead_from, lead_to] hours."""
    if source not in SOURCE_INFO:
        raise ValueError(f"Unknown weather source '{source}'")
    if lead_from > lead_to:
        raise ValueError("The lead-time window must start before it ends.")
    suffix = "_ML.parquet" if source == "pkg_icon_d2" else "_wind_sl.parquet"
    frame_fn = icon_frame if source == "pkg_icon_d2" else ecmwf_frame
    per_point: Dict[str, List[pd.DataFrame]] = {}
    weights: Dict[str, float] = {}
    for run_dir in _run_dirs(park_id, source):
        w = _weights(run_dir)
        weights = weights or w
        for key in w:
            path = run_dir / f"{key}{suffix}"
            if path.exists():
                per_point.setdefault(key, []).append(frame_fn(pd.read_parquet(path), lead_from, lead_to))
    if not per_point:
        raise ValueError(f"No forecast files found for park {park_id}.")
    series = {k: pick_latest(pd.concat([f for f in v if not f.empty], ignore_index=True))
              for k, v in per_point.items() if any(not f.empty for f in v)}
    if not series:
        raise ValueError(f"No forecasts with a lead time of {lead_from} to {lead_to} h for park {park_id}.")
    wx = combine_points(series, weights)
    if wx.empty:
        raise ValueError("The grid points share no forecast hours in this lead-time window. Try a wider window.")
    note = {"source": source, "park": park_id, "grid_points": len(series), "lead_from": lead_from, "lead_to": lead_to,
            "hours": int(len(wx))}
    return wx, note
