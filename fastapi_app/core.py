"""
Generation + lookup logic (Streamlit-free). Reuses the project's physics modules
(generate_wind_era5.py, generate_pv.py) unchanged.
"""
from __future__ import annotations

import logging
import os
import sys
from functools import lru_cache
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import requests
from ruamel.yaml import YAML
from scipy.spatial import KDTree

from . import analytics, settings
from .wake import apply_wake_losses

if str(settings.PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(settings.PROJECT_ROOT))

from generate_wind_era5 import (  # noqa: E402
    get_turbines, gen_full_dataframe, get_ageing_degradation, interpolate,
)
from generate_pv import get_features as get_pv_features, generate_pv_power  # noqa: E402

ProgressFn = Optional[Callable[[float, str], None]]


_log = logging.getLogger("synth_re.core")


def _report(progress: ProgressFn, fraction: float, message: str) -> None:
    _log.info("progress %3.0f%% %s", fraction * 100, message)   # visible in the uvicorn terminal
    if progress:
        progress(fraction, message)


# --------------------------------------------------------------------------
# Config / static data
# --------------------------------------------------------------------------
def load_config() -> dict:
    """Fresh config each call (callers mutate it per run), with absolute paths."""
    with open(settings.CONFIG_PATH, "r") as fh:
        config = YAML().load(fh)
    config["data"]["wind_ages"] = str(settings.DATA_DIR / "wind_ages.npy")
    return config


def _turbine_files(config: dict) -> dict:
    tdir = settings.PROJECT_ROOT / config["data"]["turbine_dir"]
    return dict(
        turbine_path=str(tdir / config["data"]["turbine_power"]),
        cp_path=str(tdir / config["data"]["turbine_cp"]),
        specs_path=str(tdir / config["data"]["turbine_specs"]),
    )


@lru_cache(maxsize=1)
def turbine_types() -> List[str]:
    path = _turbine_files(load_config())["turbine_path"]
    cols = pd.read_csv(path, sep=";", nrows=0).columns.tolist()
    return [c for c in cols if c != "wind_speed"]


def turbine_specs(turbine: str) -> Dict[str, float]:
    """Rotor diameter, cut-in/out and rated wind speed for one turbine model."""
    config = load_config()
    try:
        _, _, specs = get_turbines(params={"turbines": [turbine]}, **_turbine_files(config))
        return specs[turbine]
    except (ValueError, IndexError):
        # The specs table has a "-" (or no row) for some value of this model. Return what is known (None for the rest)
        # instead of failing: the dashboard only needs e.g. the rotor diameter to draw the turbine.
        table = pd.read_csv(_turbine_files(config)["specs_path"], sep=";")
        row = table[table["Turbine"] == turbine]
        if row.empty:
            raise ValueError(f"No specifications found for turbine '{turbine}'.")

        def num(col: str):
            v = pd.to_numeric(row[col].iloc[0], errors="coerce")
            return None if pd.isna(v) else float(v)

        return {"diameter": num("Rotordurchmesser"), "cut_in": num("Einschaltgeschwindigkeit"),
                "cut_out": num("Abschaltgeschwindigkeit"), "rated": num("Nennwindgeschwindigkeit")}


@lru_cache(maxsize=1)
def grid_index():
    """KDTree over all NWP grid points (parsed from grid_LAT_LON.parquet names)."""
    points, names = [], []
    coords = None                                              # station id -> (lat, lon), for Station_<id>.parquet files
    if settings.GRID_DIR.exists():
        for f in os.listdir(settings.GRID_DIR):
            if f.startswith("grid_") and f.endswith(".parquet") and f != "grid_index.parquet":
                parts = f.replace(".parquet", "").split("_")
                if len(parts) == 3:
                    try:
                        points.append([float(parts[1]), float(parts[2])])
                        names.append(f)
                    except ValueError:
                        continue
            elif f.startswith("Station_") and f.endswith(".parquet"):
                if coords is None:
                    master = pd.read_csv(settings.DATA_DIR / "masterdata.csv", dtype={"station_id": str})
                    coords = {str(r.station_id).zfill(5): (float(r.latitude), float(r.longitude))
                              for r in master.itertuples()}
                ll = coords.get(f[len("Station_"):-len(".parquet")])
                if ll:
                    points.append([ll[0], ll[1]])
                    names.append(f)
    if not points:
        return np.array([]), None, []
    arr = np.array(points)
    return arr, KDTree(arr), names


@lru_cache(maxsize=1)
def solar_stations() -> pd.DataFrame:
    """Masterdata rows for stations that actually have a solar csv."""
    master = pd.read_csv(settings.DATA_DIR / "masterdata.csv", sep=",", dtype={"station_id": str})
    if settings.SOLAR_ERA5_DIR is not None:                     # ERA5 mode: stations that have irradiance (and met) files
        have = _era5_station_ids(settings.SOLAR_ERA5_DIR)
        if settings.MET_ERA5_DIR is not None:
            have &= _era5_station_ids(settings.MET_ERA5_DIR)
    else:
        have = {int(f.split("_")[1].split(".")[0]) for f in os.listdir(settings.SOLAR_DIR)}
    return master[master["station_id"].astype(int).isin(have)].reset_index(drop=True)


def _era5_station_ids(folder) -> set:
    """Station ids of the Station_<id>.parquet files in a folder."""
    out = set()
    if folder.exists():
        for f in os.listdir(folder):
            if f.startswith("Station_") and f.endswith(".parquet"):
                try:
                    out.add(int(f[len("Station_"):-len(".parquet")]))
                except ValueError:
                    continue
    return out


def _era5_solar_frame(station_id: str) -> pd.DataFrame:
    """Hourly ERA5 inputs of a station in the layout the PV code expects (ghi/dhi as DWD J/cm² per 10 min, °C, m/s).
    The PV code multiplies irradiance by 1e4/600 to get W/m²; ERA5 is already W/m², so it is scaled down first."""
    name = f"Station_{station_id}.parquet"
    sol = pd.read_parquet(settings.SOLAR_ERA5_DIR / name, columns=["ghi", "bhi"])
    sol.index = pd.to_datetime(sol.index)
    if settings.MET_ERA5_DIR is None:
        raise ValueError("Set SYNTH_RE_ERA5_MET_DIR (temperature and wind) next to SYNTH_RE_ERA5_SOLAR_DIR.")
    met = pd.read_parquet(settings.MET_ERA5_DIR / name, columns=["temp_2m", "u_wind_10m", "v_wind_10m"])
    met.index = pd.to_datetime(met.index)
    df = sol.join(met, how="inner").dropna()                   # period where both inputs exist
    if df.empty:
        raise ValueError(f"No common hours of irradiance and temperature/wind for station {station_id}.")
    ghi = df["ghi"].clip(lower=0).astype(float)
    dhi = (df["ghi"] - df["bhi"]).clip(lower=0).clip(upper=ghi).astype(float)
    temp = df["temp_2m"].astype(float)
    out = pd.DataFrame({
        "ghi": ghi * 600.0 / 1e4, "dhi": dhi * 600.0 / 1e4,
        "temperature_2m": temp - 273.15 if temp.mean() > 100 else temp,
        "wind_speed": np.hypot(df["u_wind_10m"].astype(float), df["v_wind_10m"].astype(float)),
    })
    out.index.name = "timestamp"
    return out


def grid_points_in_bounds(south: float, west: float, north: float, east: float, limit: int = 3000) -> List[List[float]]:
    points, _, _ = grid_index()
    if len(points) == 0:
        return []
    m = (points[:, 0] >= south) & (points[:, 0] <= north) & (points[:, 1] >= west) & (points[:, 1] <= east)
    return points[m][:limit].tolist()


# --------------------------------------------------------------------------
# Lookups
# --------------------------------------------------------------------------
def search_location(query: str, limit: int = 5) -> List[dict]:
    """Nominatim forward geocoding, restricted to real places/boundaries (OSM can contain
    vandalised POI names, so shops/offices/etc. are dropped)."""
    if not query or not query.strip():
        return []
    try:
        resp = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": query.strip(), "format": "json", "addressdetails": 1,
                    "limit": limit, "countrycodes": "de"},
            headers={"User-Agent": "SyntheticRenewables/1.0"}, timeout=5,
        )
        if resp.status_code != 200:
            return []
        out = []
        for r in resp.json():
            name = (r.get("display_name") or "").strip()
            if r.get("class") not in {"place", "boundary"} or not name or len(name) > 120:
                continue
            try:
                out.append({"display_name": name, "lat": float(r["lat"]), "lon": float(r["lon"])})
            except (KeyError, ValueError, TypeError):
                continue
        return out
    except Exception:
        return []


def find_nearest_grid_point(lat: float, lon: float) -> Optional[dict]:
    points, tree, names = grid_index()
    if tree is None or len(points) == 0:
        return None
    dist, i = tree.query([lat, lon])
    return {
        "grid_lat": float(points[i][0]), "grid_lon": float(points[i][1]),
        "filename": names[i], "distance_km": float(dist * 111.0),
    }


def _haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0088 * 2 * np.arcsin(np.sqrt(a))


def find_nearest_station(lat: float, lon: float) -> dict:
    st_df = solar_stations()
    d = _haversine_km(lat, lon, st_df["latitude"].astype(float).values, st_df["longitude"].astype(float).values)
    row = st_df.iloc[int(np.argmin(d))]
    return {
        "station_id": row["station_id"], "lat": float(row["latitude"]), "lon": float(row["longitude"]),
        "altitude": float(row["station_height"]), "distance_km": float(d.min()),
    }


def load_grid_point_data(filename: str) -> Optional[pd.DataFrame]:
    path = settings.GRID_DIR / filename
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        for col in ("time", "timestamp"):
            if col in df.columns:
                df = df.set_index(col)
                break
    df.index = pd.to_datetime(df.index)
    return df


def clean_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=[c for c in df.columns if "Unnamed:" in str(c)], errors="ignore")
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df


# --------------------------------------------------------------------------
# Wind generation
# --------------------------------------------------------------------------
def _to_utc(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if index.tz is None:
        return index.tz_localize("UTC")
    return index.tz_convert("UTC") if str(index.tz) != "UTC" else index


def _degradation_rate(config: dict, override=None) -> float:
    """Annual ageing loss rate: user value, else config `annual_degradation`, else the pipeline's 0.0063."""
    if override is not None:
        return float(override)
    return float(config.get("wind_params", {}).get("annual_degradation", 0.0063))


def generate_single_turbine(grid_data, grid_lat, grid_lon, config, turbine, hub_height,
                            commissioning_date=None, apply_ageing=False, annual_degradation=None):
    """One turbine at one NWP grid point. Returns (df, degradation, power_curves, specs)."""
    wind_params = dict(config["wind_params"])
    wind_params["turbines"] = [turbine]
    wind_params["hub_heights"] = [hub_height]
    wind_params["apply_ageing"] = bool(apply_ageing)

    df = grid_data.copy()
    required = ["u_wind_10m", "v_wind_10m", "u_wind_100m", "v_wind_100m", "temp_2m", "pressure", "dew_point_2m"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Grid data is missing required ERA5 columns: {missing}")
    if df["temp_2m"].mean() < 100:
        df["temp_2m"] = df["temp_2m"] + 273.15
    if df["pressure"].mean() < 10000:
        df["pressure"] = df["pressure"] * 100
    if df["dew_point_2m"].mean() < 100:
        df["dew_point_2m"] = df["dew_point_2m"] + 273.15

    wind_params["latitude"] = grid_lat
    wind_params["longitude"] = grid_lon
    wind_params.setdefault("altitude", 100)

    degradation = None
    if apply_ageing and commissioning_date:
        ages = np.load(config["data"]["wind_ages"])
        degradation, _ = get_ageing_degradation(
            time_vector=_to_utc(df.index), real_ages=ages,
            commissioning_date=str(commissioning_date),
            random_seed=wind_params.get("random_seed", 42),
            annual_load_factor_loss_rate=_degradation_rate(config, annual_degradation),
        )

    power_curves, _, specs = get_turbines(params=wind_params, **_turbine_files(config))
    res = gen_full_dataframe(
        power_curves=power_curves, turbine=turbine, params=wind_params, df=df,
        hub_height=hub_height, specs=specs, rated_power=None,
        degradation_vector=degradation, suffix_for_turbine_cols="_t1",
    )
    out = df.copy()
    out["power"] = res["power_t1"]
    speed = res["wind_speed_t1"] if "wind_speed_t1" in res.columns else res.get("wind_speed")
    out["wind_speed_mean"] = speed
    out["wind_speed_hub"] = speed
    for extra in ("density_hub", "temperature_hub"):
        if extra in res.columns:
            out[extra] = res[extra]
    return out, degradation, power_curves, specs


def generate_wind_park(turbines, apply_ageing=False, apply_wake=True, wake_decay=0.075,
                       default_commissioning_date=None, thrust_coefficient=0.8,
                       annual_degradation=None, progress: ProgressFn = None):
    """
    Park of independently-sited turbines: each gets its own nearest NWP grid point and its
    own commissioning date; wake losses couple them afterwards.

    Returns (combined_df, site_infos, summary). `summary["energies"]` holds the ideal →
    ageing → wake → net breakdown used for the loss waterfall.
    """
    config = load_config()
    if not turbines:
        raise ValueError("Add at least one turbine.")
    use_ageing = bool(apply_ageing)

    steps_total = len(turbines) * (2 if use_ageing else 1) + 2
    done = 0

    def tick(msg: str) -> None:
        nonlocal done
        done += 1
        _report(progress, done / steps_total, msg)

    dfs, metas, sites, warnings = [], [], [], []
    baseline_energy_w = 0.0                      # Σ power with ageing OFF (ideal), in W·steps
    for n, t in enumerate(turbines, start=1):
        label = f"Turbine {n}/{len(turbines)}"
        near = find_nearest_grid_point(t["lat"], t["lon"])
        if near is None:
            warnings.append(f"Turbine {n}: no grid data available; skipped.")
            continue
        gdata = load_grid_point_data(near["filename"])
        if gdata is None or not all(c in gdata.columns for c in ("u_wind_10m", "v_wind_10m", "temp_2m", "pressure")):
            warnings.append(f"Turbine {n}: nearest grid point has no wind data; skipped.")
            continue

        commissioning = t.get("commissioning_date") or default_commissioning_date
        try:
            tdf, degr, pcurves, specs = generate_single_turbine(
                gdata, near["grid_lat"], near["grid_lon"], config, t["turbine"], t["hub_height"],
                commissioning_date=commissioning, apply_ageing=use_ageing,
                annual_degradation=annual_degradation)
        except Exception as exc:  # keep going with the other turbines
            warnings.append(f"Turbine {n}: generation failed ({exc}); skipped.")
            continue
        tick(f"{label}: generated")

        # Ideal baseline (no ageing) for the loss waterfall.
        if use_ageing and degr is not None:
            try:
                base_df, _, _, _ = generate_single_turbine(
                    gdata, near["grid_lat"], near["grid_lon"], config, t["turbine"], t["hub_height"],
                    commissioning_date=None, apply_ageing=False)
                baseline_energy_w += float(np.nansum(pd.to_numeric(base_df["power"], errors="coerce")))
            except Exception:
                baseline_energy_w += float(np.nansum(pd.to_numeric(tdf["power"], errors="coerce")))
            tick(f"{label}: baseline computed")
        else:
            baseline_energy_w += float(np.nansum(pd.to_numeric(tdf["power"], errors="coerce")))

        slot = len(dfs) + 1
        tdf[f"power_t{slot}"] = pd.to_numeric(tdf["power"], errors="coerce")
        tdf[f"wind_speed_t{slot}"] = pd.to_numeric(tdf["wind_speed_mean"], errors="coerce")
        dfs.append(tdf)

        try:
            diameter = specs[t["turbine"]]["diameter"]
            dense = interpolate(pcurves[t["turbine"]], specs[t["turbine"]]["cut_out"])
        except Exception:
            diameter, dense = None, None
        metas.append({"lat": t["lat"], "lon": t["lon"], "diameter": diameter, "power_curve": dense})

        sites.append({
            "slot": slot, **{k: t[k] for k in ("lat", "lon", "turbine", "hub_height")},
            "commissioning_date": str(commissioning) if (use_ageing and commissioning) else None,
            "grid_lat": near["grid_lat"], "grid_lon": near["grid_lon"], "distance_km": near["distance_km"],
            "ageing_loss_pct": None if degr is None else 100.0 * (1 - float(np.mean(degr))),
            "ageing_efficiency": None if degr is None else np.asarray(degr, float),
            "ageing_index": tdf.index if degr is not None else None,
            "wake_loss_pct": None,
        })

    if not dfs:
        raise ValueError("Could not generate wind power for any turbine. " + " ".join(warnings))

    dt = analytics.step_hours(dfs[0])
    pre_wake_w = float(sum(np.nansum(d[f"power_t{i}"]) for i, d in enumerate(dfs, start=1)))

    wake_applied = False
    if apply_wake and len(dfs) >= 2 and all(m["diameter"] and m["power_curve"] is not None for m in metas):
        losses = apply_wake_losses(dfs, metas, wake_decay=wake_decay, thrust_coefficient=thrust_coefficient)
        for s, loss in zip(sites, losses):
            s["wake_loss_pct"] = loss
        wake_applied = True
    tick("Wake losses applied" if wake_applied else "Combining turbines")

    combined = dfs[0].copy()
    for slot, d in enumerate(dfs, start=1):
        combined[f"power_t{slot}"] = d[f"power_t{slot}"].reindex(combined.index)
        combined[f"wind_speed_t{slot}"] = d[f"wind_speed_t{slot}"].reindex(combined.index)
    power_cols = [f"power_t{i}" for i in range(1, len(dfs) + 1)]
    speed_cols = [f"wind_speed_t{i}" for i in range(1, len(dfs) + 1)]
    combined["power"] = combined[power_cols].sum(axis=1, skipna=True)
    combined["wind_speed_mean"] = combined[speed_cols].mean(axis=1)
    combined["wind_speed_hub"] = combined["wind_speed_mean"]
    combined = clean_index(combined)

    # per-turbine statistics (after wake)
    for s, meta in zip(sites, metas):
        col = combined[f"power_t{s['slot']}"]
        rated = float(meta["power_curve"].max()) if meta["power_curve"] is not None else float(col.max())
        s["rated_mw"] = rated / 1e6
        s["energy_mwh"] = float(np.nansum(col) * dt / 1e6)
        s["capacity_factor_pct"] = float(100.0 * np.nanmean(col) / rated) if rated > 0 else None
        s["mean_wind_speed"] = float(np.nanmean(combined[f"wind_speed_t{s['slot']}"]))

    if all(m["power_curve"] is not None for m in metas):
        capacity_w = float(sum(m["power_curve"].max() for m in metas))
    else:
        capacity_w = float(combined[power_cols].max().sum())

    net_mwh = float(np.nansum(combined["power"]) * dt / 1e6)
    after_ageing_mwh = pre_wake_w * dt / 1e6
    ideal_mwh = max(baseline_energy_w * dt / 1e6, after_ageing_mwh)
    energies = analytics.energy_waterfall(ideal_mwh, after_ageing_mwh, net_mwh)

    summary = {
        "n_turbines": len(dfs),
        "capacity_w": capacity_w,
        "wake_applied": wake_applied,
        "wake_decay": wake_decay,
        "park_wake_loss_pct": float(np.nanmean([s["wake_loss_pct"] for s in sites])) if wake_applied else None,
        "ageing_applied": any(s["ageing_loss_pct"] is not None for s in sites),
        "energies": energies,
        "warnings": warnings,
        "start": str(combined.index.min().date()),
        "end": str(combined.index.max().date()),
    }
    _report(progress, 1.0, "Finished")
    return combined, sites, summary


# --------------------------------------------------------------------------
# Solar generation
# --------------------------------------------------------------------------
def generate_solar(lat: float, lon: float, pv_overrides: Optional[dict] = None, progress: ProgressFn = None):
    """PV power for a site, driven by the nearest DWD station's irradiance/weather."""
    config = load_config()
    params = dict(config["pv_params"])
    params.update({k: v for k, v in (pv_overrides or {}).items() if v is not None})
    features = config["features"]

    _report(progress, 0.1, "Finding nearest weather station")
    station = find_nearest_station(lat, lon)
    era5 = settings.SOLAR_ERA5_DIR is not None
    path = settings.SOLAR_DIR / f"Station_{station['station_id']}.csv"
    if not era5 and not path.exists():
        raise FileNotFoundError(f"No solar data for station {station['station_id']}")

    params["latitude"] = lat
    params["longitude"] = lon
    params["altitude"] = station["altitude"]

    _report(progress, 0.3, "Reading station data")
    if era5:
        data = _era5_solar_frame(station["station_id"])
    else:
        data = pd.read_csv(path, sep=",")
        data["timestamp"] = pd.to_datetime(data["timestamp"])
        data = data.set_index("timestamp")

    _report(progress, 0.5, "Computing irradiance on the panel")
    irr, cell_t = get_pv_features(data=data, features=features, params=params)
    _report(progress, 0.8, "Computing PV power")
    data["power"] = generate_pv_power(total_irradiance=irr["poa_global"], cell_temperature=cell_t, params=params)
    data["Total"] = irr["poa_global"]
    data["Direct"] = irr["poa_direct"]
    data["Diffuse"] = irr["poa_diffuse"]

    data = data.resample("h").mean(numeric_only=True)
    data = data.drop(columns=["station_id"], errors="ignore")
    data = clean_index(data)

    summary = {
        "capacity_w": float(params["dc_rating_watts"]),
        "station": station,
        "start": str(data.index.min().date()),
        "end": str(data.index.max().date()),
        "warnings": [],
    }
    _report(progress, 1.0, "Finished")
    return data, summary


def turbine_power_curve_points(turbine: str) -> dict:
    """Raw (original) + linearly interpolated power-curve points, in MW."""
    path = _turbine_files(load_config())["turbine_path"]
    df = pd.read_csv(path, sep=";", decimal=",", usecols=["wind_speed", turbine])
    was_na = df[turbine].isna()
    df[turbine] = pd.to_numeric(df[turbine], errors="coerce").interpolate(method="linear")
    df = df.dropna(subset=[turbine])
    was_na = was_na.loc[df.index]
    df["mw"] = df[turbine] / 1000.0
    return {
        "original": {"x": df.loc[~was_na, "wind_speed"].tolist(), "y": df.loc[~was_na, "mw"].tolist()},
        "interpolated": {"x": df.loc[was_na, "wind_speed"].tolist(), "y": df.loc[was_na, "mw"].tolist()},
    }
