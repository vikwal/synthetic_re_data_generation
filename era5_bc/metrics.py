"""Evaluation metrics (paper Table 3 + Perkins skill score).

All time-series metrics operate on paired hourly (obs, pred) arrays of one
station. Aggregation across stations (median, as in paper Tables 4-6) happens
in the evaluation script.
"""

import numpy as np
import pandas as pd


def mae(obs, pred):
    return float(np.mean(np.abs(obs - pred)))


def mbe(obs, pred):
    """Mean bias error, positive = overestimation (paper: pred - obs)."""
    return float(np.mean(pred - obs))


def rmse(obs, pred):
    return float(np.sqrt(np.mean((obs - pred) ** 2)))


def pcc(obs, pred):
    return float(np.corrcoef(obs, pred)[0, 1])


def r2(obs, pred):
    ss_res = np.sum((obs - pred) ** 2)
    ss_tot = np.sum((obs - np.mean(obs)) ** 2)
    return float(1.0 - ss_res / ss_tot)


def pss(obs, pred, bin_width: float, max_ws: float) -> float:
    """Perkins skill score in % — overlap of the normalized histograms."""
    bins = np.arange(0.0, max_ws + bin_width, bin_width)
    z_obs, _ = np.histogram(np.clip(obs, 0, max_ws - 1e-9), bins=bins)
    z_pred, _ = np.histogram(np.clip(pred, 0, max_ws - 1e-9), bins=bins)
    z_obs = z_obs / max(len(obs), 1)
    z_pred = z_pred / max(len(pred), 1)
    return float(np.minimum(z_obs, z_pred).sum() * 100.0)


def station_metrics(obs: np.ndarray, pred: np.ndarray, ev: dict) -> dict:
    """All per-station metrics. ev = cfg['evaluation']."""
    lwt = obs < np.percentile(obs, ev["pss_tail_pcts"][0])
    upt = obs > np.percentile(obs, ev["pss_tail_pcts"][1])
    out = {
        "mae": mae(obs, pred), "mbe": mbe(obs, pred),
        "rmse": rmse(obs, pred), "pcc": pcc(obs, pred), "r2": r2(obs, pred),
        "median_obs": float(np.median(obs)), "median_pred": float(np.median(pred)),
        "pss_all": pss(obs, pred, ev["pss_bin_width"], ev["pss_max_ws"]),
        "pss_lwt": pss(obs[lwt], pred[lwt], ev["pss_bin_width"], ev["pss_max_ws"]),
        "pss_upt": pss(obs[upt], pred[upt], ev["pss_bin_width"], ev["pss_max_ws"]),
        "n_hours": int(len(obs)),
    }
    for q in ev["quantiles"]:
        out[f"q{int(q * 100)}_obs"] = float(np.quantile(obs, q))
        out[f"q{int(q * 100)}_pred"] = float(np.quantile(pred, q))
    return out


def median_ws_metrics(per_station: pd.DataFrame) -> dict:
    """Across-station metrics on the median wind speed (paper Table 4)."""
    o = per_station["median_obs"].to_numpy()
    p = per_station["median_pred"].to_numpy()
    return {"mae": mae(o, p), "mbe": mbe(o, p), "r2": r2(o, p),
            "rmse": rmse(o, p)}


def pct_improvement(sm_model: float, sm_era5: float, sm_perf: float) -> float:
    """Percentage improvement over uncorrected ERA5 (paper eq. 20)."""
    denom = sm_perf - sm_era5
    if denom == 0:
        return np.nan
    return float((sm_model - sm_era5) / denom * 100.0)
