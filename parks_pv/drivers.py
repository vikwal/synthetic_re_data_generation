"""Driver frame per park for the PV chain (START labels, UTC, hourly or 30 min): corrected CAMS at the park
(output of scripts/cams_bc/apply_parks.py, which writes both steps), ERA5 met (temp, wind, pressure, dew point) and ERA5 albedo from the
WeatherDB, ERA5 snow depth from the monthly GRIBs. Every input gets a SHA256 for the manifest."""
import glob
import hashlib
import os
import warnings

import numpy as np
import pandas as pd

from pv import drivers as D, snow, timebase
from parks_pv import paths

CAMS_RAW_COLS = ["cams_ghi", "cams_dhi", "cams_ghi_clear"]


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_frame(df: pd.DataFrame) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values.tobytes()).hexdigest()


def cams_files(park_id: str, tag: str = "2023_2026") -> dict:
    """Centre and ring CSV paths of a park (raw ADS files)."""
    c = sorted(glob.glob(os.path.join(paths.INPUT_DIR, "cams", f"CAMS_rad_{park_id}_{tag}.csv")))
    r = sorted(glob.glob(os.path.join(paths.INPUT_DIR, "cams_ring", f"CAMS_rad_{park_id}_r*_{tag}.csv")))
    return {"centre": c, "ring": r}


def corrected_cams(park_id: str, model_tag: str, step="1h", corr_dir: str = None) -> pd.DataFrame:
    """Corrected CAMS written by apply_parks (cams_ghi, cams_dhi, cams_ghi_clear, ghi_corr, dhi_corr,
    ring_missing) on START labels: the hourly file is START-labelled, the 30-min file END-labelled."""
    corr_dir = corr_dir or paths.CORR_DIR
    if pd.Timedelta(step) == pd.Timedelta("1h"):
        df = pd.read_parquet(os.path.join(corr_dir, f"{park_id}_cams_corrected_hourly_{model_tag}.parquet"))
        df.index = pd.DatetimeIndex(df.index, tz="UTC") if df.index.tz is None else df.index.tz_convert("UTC")
        return df
    if pd.Timedelta(step) != pd.Timedelta("30min"):
        raise ValueError(f"no corrected CAMS product for step {step}")
    df = pd.read_parquet(os.path.join(corr_dir, f"{park_id}_cams_corrected_30min_{model_tag}.parquet"))
    df.index = pd.DatetimeIndex(df.index, tz="UTC") if df.index.tz is None else df.index.tz_convert("UTC")
    df.index = df.index - pd.Timedelta("30min")                    # END -> START
    return df


def snow_series(lat: float, lon: float, grib_dir: str = paths.SNOW_DIR, cache: str = None, step="1h") -> pd.DataFrame:
    """Snow depth (cm, physical) and snowfall (cm/h) at the nearest ERA5 point, START labels, on the chain
    step: the depth (instantaneous at the hour end) is interpolated in time, the snowfall rate repeated."""
    if cache and os.path.exists(cache):
        sn = pd.read_parquet(cache)
    else:
        sn = snow.era5_snow_point(grib_dir, lat, lon)
        if cache:
            os.makedirs(os.path.dirname(cache), exist_ok=True)
            sn.to_parquet(cache)
    out = pd.DataFrame(index=sn.index - pd.Timedelta("1h"))        # END -> START (instant depth at the hour end)
    out["snow_depth_cm"] = (sn["snow_depth_m"].values * 100.0)
    out["snowfall_cm"] = snow.snowfall_cm(sn["snowfall_mm_we"]).values
    if pd.Timedelta(step) != pd.Timedelta("1h"):
        out = _refine_snow(out, step)
    return out


def _refine_snow(hourly: pd.DataFrame, step) -> pd.DataFrame:
    """Hourly snow frame (START labels) onto a finer step: the depth is interpolated linearly inside hours
    whose both ends are known (gaps stay NaN, the caller's ffill/fillna rule applies as on the hourly path),
    the snowfall rate (cm/h) is repeated over the sub-intervals."""
    step = pd.Timedelta(step)
    n = int(pd.Timedelta("1h") / step)
    grid = pd.date_range(hourly.index[0], hourly.index[-1] + pd.Timedelta("1h") - step, freq=step)
    fine = pd.DataFrame(index=grid)
    d = hourly["snow_depth_cm"]
    inside = (d.notna() & d.shift(-1).notna()).reindex(grid, method="ffill")           # hour with both ends known
    fine["snow_depth_cm"] = d.reindex(d.index.union(grid)).interpolate(method="time", limit_area="inside").reindex(grid).where(inside | d.reindex(grid).notna())
    fine["snowfall_cm"] = hourly["snowfall_cm"].reindex(grid, method="ffill", limit=n - 1)
    return fine


ICON_LEAD_MAX_H = 3.0            # leads 0..3 h of the 09 UTC run = "analysis" value of the day
ICON_MIN_COVERAGE = 0.9          # warn below this share of days with an ICON run


class IconSnowMissing(UserWarning):
    """ICON-D2 snow depth missing or incomplete for a park; the snow driver degrades to ERA5 only."""


def icon_snow_daily(park_dir: str) -> pd.Series:
    """Daily ICON-D2 snow depth (cm) at the nearest grid point of a park (points.csv center_rank 1): mean h_snow over
    the leads 0..ICON_LEAD_MAX_H of the 09 UTC run. Index: UTC run date (tz-naive). Same reduction as the snow
    calibration (FL_Contribution/reports/pv_snow_fix_2026-10-19/snow_common.icon_daily)."""
    pts = pd.read_csv(os.path.join(park_dir, "points.csv"))
    key = pts.sort_values("center_rank").iloc[0]["point_key"]
    df = pd.read_parquet(os.path.join(park_dir, f"{key}_SL.parquet"), columns=["starttime", "forecasttime", "h_snow"])
    df["date"] = df["starttime"].dt.tz_convert("UTC").dt.normalize().dt.tz_localize(None)
    s = df[df["forecasttime"] <= ICON_LEAD_MAX_H].groupby("date")["h_snow"].mean() * 100.0
    s.name = "icon_snow_depth_cm"
    return s


def icon_snow_series(park_id: str, idx: pd.DatetimeIndex, icon_dir: str = None, ffill_days: int = 2) -> tuple:
    """(ICON daily snow depth on the step index idx, meta). Each step gets the value of its UTC date; missing run days
    are carried forward for up to ffill_days days, then 0 (as in the calibration)."""
    park_dir = os.path.join(icon_dir or paths.ICON_SL09_DIR, f"park_{park_id}")
    day_key = idx.tz_convert("UTC").normalize().tz_localize(None)
    days = pd.DatetimeIndex(sorted(set(day_key)))
    try:
        daily = icon_snow_daily(park_dir)
    except (FileNotFoundError, KeyError, IndexError, ValueError) as e:
        # no ICON-D2 snow for this park: depth_slide runs on ERA5 snow depth alone
        warnings.warn(f"park {park_id}: ICON-D2 h_snow not available ({type(e).__name__}: {e}); "
                      f"snow driver falls back to ERA5 snow depth only", IconSnowMissing, stacklevel=2)
        out = pd.Series(0.0, index=idx, name="icon_snow_depth_cm")
        meta = {"icon_dir": park_dir, "icon_days": 0, "release_days": int(len(days)), "icon_coverage": 0.0,
                "icon_daily_sha256": None, "icon_status": "unavailable", "icon_error": f"{type(e).__name__}: {e}"}
        return out, meta
    have = daily.reindex(days)
    cov = float(have.notna().mean())
    if cov < ICON_MIN_COVERAGE:
        warnings.warn(f"park {park_id}: ICON-D2 h_snow covers only {cov:.1%} of the {len(days)} days; "
                      f"missing days count as no ICON snow (ERA5 only)", IconSnowMissing, stacklevel=2)
    filled = have.ffill(limit=ffill_days).fillna(0.0)
    out = pd.Series(filled.reindex(day_key).to_numpy(), index=idx, name="icon_snow_depth_cm")
    meta = {"icon_dir": park_dir, "icon_days": int(have.notna().sum()), "release_days": int(len(days)),
            "icon_coverage": float(have.notna().mean()), "icon_daily_sha256": sha256_frame(daily.to_frame())}
    return out, meta


def build(park_id: str, lat: float, lon: float, start: str, end: str, conn, model_tag: str, snow_cache: str = None,
          step="1h", corr_dir: str = None, cams_dir: str = None) -> tuple:
    """(driver frame on the chain index of the given step, meta with hashes and cell ids). corr_dir / cams_dir
    override the release locations (Trianel twins keep their confidential inputs elsewhere)."""
    idx = timebase.step_index(start, end, step)
    cc = corrected_cams(park_id, model_tag, step, corr_dir)
    irr = pd.DataFrame({"ghi": cc["ghi_corr"], "dhi": cc["dhi_corr"], "ghi_cams_raw": cc["cams_ghi"], "dhi_cams_raw": cc["cams_dhi"],
                        "ghi_clear_cams": cc["cams_ghi_clear"], "ring_missing": cc["ring_missing"].astype(float)})
    irr["imputed_share"] = irr["ghi"].isna().astype(float)
    # gaps in the corrected product (missing ring / ERA5 hours) fall back to raw CAMS, flagged
    irr["ghi"] = irr["ghi"].fillna(irr["ghi_cams_raw"])
    irr["dhi"] = irr["dhi"].fillna(irr["dhi_cams_raw"])
    met, m_met = D.era5_met(conn, lat, lon, start, end, step=step)
    e5, m_sol = D.era5_solar(conn, lat, lon, start, end, step=step)
    drv = D.merge_driver(irr, met, idx)
    drv["albedo"] = e5["albedo"].reindex(idx).ffill().bfill().clip(0.05, 0.9)
    drv["ghi_era5"] = e5["ghi"].reindex(idx)
    sn = snow_series(lat, lon, cache=snow_cache, step=step).reindex(idx)
    drv["snow_depth_cm"] = sn["snow_depth_cm"].ffill(limit=int(pd.Timedelta("6h") / pd.Timedelta(step))).fillna(0.0)
    drv["snowfall_cm"] = sn["snowfall_cm"].fillna(0.0)
    files = cams_files(park_id) if cams_dir is None else {"centre": sorted(glob.glob(os.path.join(cams_dir, f"CAMS_rad_{park_id}_*.csv"))), "ring": []}
    meta = {**m_met, **m_sol, "model_tag": model_tag, "step": str(pd.Timedelta(step)),
            "cams_sha256": {os.path.basename(f): sha256_file(f) for f in files["centre"] + files["ring"]},
            "cams_corrected_sha256": sha256_frame(cc),
            "era5_met_sha256": sha256_frame(met), "era5_solar_sha256": sha256_frame(e5),
            "snow_gribs": sorted(os.path.basename(f) for f in glob.glob(os.path.join(paths.SNOW_DIR, "*.grib"))),
            "snow_point_sha256": sha256_frame(sn.dropna(how="all")),
            "share_hours_cams_fallback": float(irr["imputed_share"].reindex(idx).mean()),
            "share_hours_ring_missing": float(irr["ring_missing"].reindex(idx).fillna(1.0).mean())}
    return drv, meta
