"""Run the PV chain for one park on its driver frame and add the snow loss as a multiplicative factor on the AC power.

Snow models (coverage -> factor = 1 - pvlib dc_loss_nrel(coverage, 100 strings), pv.snow.loss_factor):
  depth_linear  (default, E8 / release v1): coverage = clip(ERA5 snow depth / 5 cm, 0, 1)
  depth_slide   (model (e), calibrated on the six Trianel meters, FL_Contribution/reports/pv_snow_fix_2026-10-19):
                ramp = clip(max(ERA5 depth, ICON-D2 09 UTC daily h_snow) / 5 cm, 0, 1);
                cov_t = max(ramp_t, cov_{t-1} - slide * sin(tilt) * dt_h) while T_air > -POA_global / 80 (pvlib
                can-slide rule), else cov_t = max(ramp_t, cov_{t-1}); slide = 0.1 /h, tilt = share-weighted park tilt.
                Needs the driver column icon_snow_depth_cm (parks_pv.drivers.icon_snow_series).
"""
import numpy as np
import pandas as pd

from pv import chain, snow

SNOW_FULL_COVER_CM = 5.0
SNOW_MODELS = ("depth_linear", "depth_slide")
SNOW_DEFAULTS = {"depth_linear": {"d_full_cm": SNOW_FULL_COVER_CM},
                 "depth_slide": {"d_full_cm": SNOW_FULL_COVER_CM, "slide_per_h": 0.1, "can_slide_coefficient": -80.0}}
SNOW_DRIVER = {"depth_linear": "ERA5 snow depth (sd/rsn, nearest cell)",
               "depth_slide": "max(ERA5 snow depth, ICON-D2 09 UTC run h_snow, mean of leads 0-3 h, nearest point, per UTC day)"}
OUT_EXTRA = ["snow_depth_cm", "snow_coverage", "snow_factor", "power_park_free", "power_park"]


def park_tilt(arrays: list) -> float:
    """Share-weighted mean tilt (deg) of the sub-arrays (dicts with tilt, share)."""
    return float(np.average([float(a["tilt"]) for a in arrays], weights=[float(a.get("share", 1.0)) for a in arrays]))


def snow_params(model: str, params: dict = None) -> dict:
    if model not in SNOW_MODELS:
        raise ValueError(f"unknown snow model {model!r}, expected one of {SNOW_MODELS}")
    return {**SNOW_DEFAULTS[model], **(params or {})}


def slide_off(ramp_cov: pd.Series, temp_air: pd.Series, poa_global: pd.Series, tilt: float, slide_per_h: float,
              step_h: float, can_slide_coefficient: float = -80.0) -> pd.Series:
    """Depth-initiated coverage with Marion-type slide-off: the coverage rises to the ramp at once but never drops
    faster than slide_per_h * sin(tilt) per hour, and only while T_air > POA / can_slide_coefficient."""
    r = ramp_cov.fillna(0.0).to_numpy(dtype=float)
    can = temp_air.ffill().bfill().to_numpy(dtype=float) > poa_global.fillna(0.0).to_numpy(dtype=float) / can_slide_coefficient
    rate = slide_per_h * np.sin(np.radians(tilt)) * step_h
    out = np.empty_like(r)
    prev = 0.0
    for i in range(len(r)):
        prev = max(r[i], prev - (rate if can[i] else 0.0))
        out[i] = prev
    return pd.Series(out, index=ramp_cov.index)


def snow_coverage(frame: pd.DataFrame, model: str = "depth_linear", tilt: float = None, params: dict = None) -> pd.Series:
    """Coverage on the frame index. frame: snow_depth_cm (ERA5); depth_slide also icon_snow_depth_cm, temp_air, poa_global."""
    par = snow_params(model, params)
    era5 = frame["snow_depth_cm"].fillna(0.0)
    if model == "depth_linear":
        return (era5 / par["d_full_cm"]).clip(0.0, 1.0)
    if "icon_snow_depth_cm" not in frame:
        raise KeyError("snow model depth_slide needs the driver column icon_snow_depth_cm")
    if tilt is None:
        raise ValueError("snow model depth_slide needs the park tilt")
    depth = np.maximum(era5, frame["icon_snow_depth_cm"].fillna(0.0))
    ramp = (depth / par["d_full_cm"]).clip(0.0, 1.0)
    step_h = (frame.index[1] - frame.index[0]) / pd.Timedelta("1h")
    return slide_off(ramp, frame["temp_air"], frame["poa_global"], tilt, par["slide_per_h"], step_h, par["can_slide_coefficient"])


def run(cfg: dict, drv: pd.DataFrame, horizon: pd.Series = None, snow_on: bool = True, snow_model: str = None,
        snow_par: dict = None) -> pd.DataFrame:
    """snow_model: None -> cfg['snow']['model'] if the config has one, else depth_linear (release v1 behaviour)."""
    res = chain.run_park(cfg, drv, horizon=horizon)
    res["snow_depth_cm"] = drv["snow_depth_cm"].reindex(res.index).fillna(0.0)
    model = snow_model or (cfg.get("snow") or {}).get("model", "depth_linear")
    par = snow_par if snow_par is not None else (cfg.get("snow") or {}).get("params")
    if not snow_on:
        cov = pd.Series(0.0, index=res.index)
    elif model == "depth_slide":
        frame = pd.DataFrame({"snow_depth_cm": res["snow_depth_cm"], "icon_snow_depth_cm": drv["icon_snow_depth_cm"].reindex(res.index),
                              "temp_air": drv["temp_air"].reindex(res.index), "poa_global": res["poa_global"]}, index=res.index)
        cov = snow_coverage(frame, model, park_tilt(cfg["arrays"]), par)
    else:
        cov = snow_coverage(res, model, None, par)
    res["snow_coverage"] = cov
    res["snow_factor"] = snow.loss_factor(cov).reindex(res.index).fillna(1.0) if snow_on else 1.0
    res["power_park_free"] = res["p_ac_w"]
    res["power_park"] = res["p_ac_w"] * res["snow_factor"]
    return res


def resnow_frame(frame: pd.DataFrame, model: str, tilt: float = None, params: dict = None,
                 icon_depth_cm: pd.Series = None) -> pd.DataFrame:
    """Recompute snow_coverage, snow_factor and power_park of a stored release frame (power_park_free, snow_depth_cm,
    temp_air, poa_global present) with another snow model; every other column stays bit-identical."""
    src = frame[["snow_depth_cm", "temp_air", "poa_global"]].copy()
    if icon_depth_cm is not None:
        src["icon_snow_depth_cm"] = icon_depth_cm.reindex(frame.index)
    cov = snow_coverage(src, model, tilt, params)
    out = frame.copy()
    out["snow_coverage"] = cov.astype(frame["snow_coverage"].dtype)
    out["snow_factor"] = snow.loss_factor(cov).reindex(out.index).fillna(1.0).astype(frame["snow_factor"].dtype)
    out["power_park"] = out["power_park_free"] * out["snow_factor"]
    return out
