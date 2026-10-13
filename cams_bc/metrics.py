"""Pooled and per-station error metrics in W/m², plus the skill against raw CAMS on identical rows."""
import numpy as np
import pandas as pd


def basic(y: np.ndarray, p: np.ndarray) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    m = np.isfinite(y) & np.isfinite(p)
    y, p = y[m], p[m]
    if len(y) < 2:
        return {"n": int(len(y)), "rmse": np.nan, "mae": np.nan, "bias": np.nan, "r2": np.nan}
    r = p - y
    ss = np.sum((y - y.mean()) ** 2)
    return {"n": int(len(y)), "rmse": float(np.sqrt(np.mean(r ** 2))), "mae": float(np.mean(np.abs(r))),
            "bias": float(r.mean()), "r2": float(1 - np.sum(r ** 2) / ss) if ss > 0 else np.nan}


def skill(rmse_model: float, rmse_ref: float) -> float:
    return float(1.0 - rmse_model / rmse_ref) if rmse_ref > 0 else np.nan


def evaluate(df: pd.DataFrame, pred: dict, var: str, valid: np.ndarray) -> dict:
    """pred: {name: array}. Returns pooled metrics per name plus skill vs cams_<var>, on `valid` rows."""
    y = df.loc[valid, f"{var}_obs"].to_numpy()
    ref = df.loc[valid, f"cams_{var}"].to_numpy()
    out = {"cams": basic(y, ref)}
    for name, p in pred.items():
        b = basic(y, np.asarray(p)[valid])
        b["skill"] = skill(b["rmse"], out["cams"]["rmse"])
        out[name] = b
    return out


def per_station(df: pd.DataFrame, pred: np.ndarray, var: str, valid: np.ndarray) -> pd.DataFrame:
    rows = []
    sub = df.loc[valid, ["station_id", f"{var}_obs", f"cams_{var}"]].copy()
    sub["pred"] = np.asarray(pred)[valid]
    for sid, g in sub.groupby("station_id"):
        c = basic(g[f"{var}_obs"], g[f"cams_{var}"])
        m = basic(g[f"{var}_obs"], g["pred"])
        rows.append({"station_id": sid, "n": m["n"], "rmse_cams": c["rmse"], "rmse_model": m["rmse"], "r2_cams": c["r2"],
                     "r2_model": m["r2"], "bias_cams": c["bias"], "bias_model": m["bias"], "skill": skill(m["rmse"], c["rmse"])})
    return pd.DataFrame(rows)


def hourly(df: pd.DataFrame, pred: np.ndarray, var: str, valid: np.ndarray) -> pd.DataFrame:
    """Hourly means of observation, CAMS and prediction over valid 30-min rows (the PV chain is hourly).
    Rows carry interval-END labels, so hour HH = rows HH:30 and HH+1:00; the 1-minute shift before
    flooring puts both on the START label HH:00 (same rule as apply_parks)."""
    sub = df.loc[valid, ["station_id", f"{var}_obs", f"cams_{var}"]].copy()
    sub["pred"] = np.asarray(pred)[valid]
    sub["hour"] = (sub.index - pd.Timedelta("1min")).floor("h")
    g = sub.groupby(["station_id", "hour"])
    h = g.mean(numeric_only=True)
    h["n"] = g.size()
    return h[h["n"] == 2]
