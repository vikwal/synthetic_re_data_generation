"""Hourly driver frame per park for the PV chain (START labels, UTC): corrected CAMS at the park (output of
scripts/cams_bc/apply_parks.py), ERA5 met (temp, wind, pressure, dew point) and ERA5 albedo from the
WeatherDB, ERA5 snow depth from the monthly GRIBs. Every input gets a SHA256 for the manifest."""
import glob
import hashlib
import os

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


def corrected_cams(park_id: str, model_tag: str) -> pd.DataFrame:
    """Hourly corrected CAMS (START labels) written by apply_parks: cams_ghi, cams_dhi, cams_ghi_clear,
    ghi_corr, dhi_corr, ring_missing."""
    p = os.path.join(paths.WORK_DIR, "cams_corrected", f"{park_id}_cams_corrected_hourly_{model_tag}.parquet")
    df = pd.read_parquet(p)
    df.index = pd.DatetimeIndex(df.index, tz="UTC") if df.index.tz is None else df.index.tz_convert("UTC")
    return df


def snow_series(lat: float, lon: float, grib_dir: str = paths.SNOW_DIR, cache: str = None) -> pd.DataFrame:
    """Hourly snow depth (cm, physical) and snowfall (cm/h) at the nearest ERA5 point, START labels."""
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
    return out


def build(park_id: str, lat: float, lon: float, start: str, end: str, conn, model_tag: str, snow_cache: str = None) -> tuple:
    """(driver frame on the hourly chain index, meta with hashes and cell ids)."""
    idx = timebase.hourly_index(start, end)
    cc = corrected_cams(park_id, model_tag)
    irr = pd.DataFrame({"ghi": cc["ghi_corr"], "dhi": cc["dhi_corr"], "ghi_cams_raw": cc["cams_ghi"], "dhi_cams_raw": cc["cams_dhi"],
                        "ghi_clear_cams": cc["cams_ghi_clear"], "ring_missing": cc["ring_missing"].astype(float)})
    irr["imputed_share"] = irr["ghi"].isna().astype(float)
    # gaps in the corrected product (missing ring / ERA5 hours) fall back to raw CAMS, flagged
    irr["ghi"] = irr["ghi"].fillna(irr["ghi_cams_raw"])
    irr["dhi"] = irr["dhi"].fillna(irr["dhi_cams_raw"])
    met, m_met = D.era5_met(conn, lat, lon, start, end)
    e5, m_sol = D.era5_solar(conn, lat, lon, start, end)
    drv = D.merge_driver(irr, met, idx)
    drv["albedo"] = e5["albedo"].reindex(idx).ffill().bfill().clip(0.05, 0.9)
    drv["ghi_era5"] = e5["ghi"].reindex(idx)
    sn = snow_series(lat, lon, cache=snow_cache).reindex(idx)
    drv["snow_depth_cm"] = sn["snow_depth_cm"].ffill(limit=6).fillna(0.0)
    drv["snowfall_cm"] = sn["snowfall_cm"].fillna(0.0)
    files = cams_files(park_id)
    meta = {**m_met, **m_sol, "model_tag": model_tag,
            "cams_sha256": {os.path.basename(f): sha256_file(f) for f in files["centre"] + files["ring"]},
            "cams_corrected_sha256": sha256_frame(cc),
            "era5_met_sha256": sha256_frame(met), "era5_solar_sha256": sha256_frame(e5),
            "snow_gribs": sorted(os.path.basename(f) for f in glob.glob(os.path.join(paths.SNOW_DIR, "*.grib"))),
            "snow_point_sha256": sha256_frame(sn.dropna(how="all")),
            "share_hours_cams_fallback": float(irr["imputed_share"].reindex(idx).mean()),
            "share_hours_ring_missing": float(irr["ring_missing"].reindex(idx).fillna(1.0).mean())}
    return drv, meta
