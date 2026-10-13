"""Estimate the effective orientation of a park from measured power and an irradiance driver.

Two estimators:
  grid_fit     single (tilt, azimuth) maximising R2 of normalised POA-driven power on clear hours
  nnls_mixture non-negative weights over a bank of (tilt, azimuth) POA basis series; the bank
               weights give the effective sub-array mixture, their normal-weighted mean the
               effective orientation
Both work on a normalised target (power / capacity) and compare shapes, so the DC rating and
constant losses drop out; a free scale is fitted per candidate.
"""

import numpy as np
import pandas as pd
import pvlib
from scipy.optimize import nnls

from pv import arrays as arr, timebase, transposition


def clear_hours(ghi: pd.Series, lat: float, lon: float, alt: float = 0.0, kc_min: float = 0.8,
                elev_min: float = 10.0) -> pd.Series:
    """Boolean mask of hours with clear-sky index >= kc_min and sun elevation >= elev_min."""
    cs = timebase.clearsky(ghi.index, lat, lon, alt)
    sp = timebase.solar_position(ghi.index, lat, lon, alt)
    kc = ghi / cs["ghi"].replace(0, np.nan)
    return (kc >= kc_min) & ((90.0 - sp["apparent_zenith"]) >= elev_min) & ghi.notna()


def poa_bank(drv: pd.DataFrame, lat: float, lon: float, alt: float, tilts, azimuths, albedo: float = 0.2,
             horizon: pd.Series = None, dni: pd.Series = None) -> dict:
    """{(tilt, az): poa_eff} for all pairs on the driver frame."""
    sp = timebase.solar_position(drv.index, lat, lon, alt)
    if dni is None:
        cs = timebase.clearsky(drv.index, lat, lon, alt)
        dni = transposition.dni_from_closure(drv["ghi"], drv["dhi"], sp["apparent_zenith"], cs["dni"])
    dni_extra = pvlib.irradiance.get_extra_radiation(drv.index + timebase.CENTRE)
    dni_extra.index = drv.index
    bank = {}
    for t in tilts:
        for a in azimuths:
            bank[(float(t), float(a))] = transposition.poa(drv["ghi"], drv["dhi"], dni, sp, t, a, albedo, dni_extra, horizon=horizon)["poa_eff"]
    return bank


def _r2(y, yhat):
    ss = np.sum((y - y.mean()) ** 2)
    return 1.0 - np.sum((y - yhat) ** 2) / ss if ss > 0 else np.nan


def grid_fit(target: pd.Series, bank: dict, mask: pd.Series, t_cell: pd.Series = None, gamma: float = -0.0035) -> pd.DataFrame:
    """R2 of a scaled POA (optionally temperature-corrected) against the target per bank pair."""
    rows = []
    m = mask & target.notna()
    y = target[m].values
    for (t, a), poa in bank.items():
        x = poa[m].values
        if t_cell is not None:
            x = x * (1.0 + gamma * (t_cell[m].values - 25.0))
        if x.sum() <= 0:
            continue
        scale = np.dot(x, y) / np.dot(x, x)
        rows.append({"tilt": t, "azimuth": a, "scale": scale, "r2": _r2(y, scale * x),
                     "rmse": float(np.sqrt(np.mean((y - scale * x) ** 2)))})
    out = pd.DataFrame(rows).sort_values("r2", ascending=False).reset_index(drop=True)
    return out


def nnls_mixture(target: pd.Series, bank: dict, mask: pd.Series, t_cell: pd.Series = None, gamma: float = -0.0035,
                 min_weight: float = 0.02) -> dict:
    """Non-negative mixture over the bank. Returns weights (normalised), effective tilt/azimuth,
    R2 and the fitted scale (sum of raw weights)."""
    m = mask & target.notna()
    keys = list(bank.keys())
    X = np.column_stack([bank[k][m].values for k in keys])
    if t_cell is not None:
        X = X * (1.0 + gamma * (t_cell[m].values[:, None] - 25.0))
    y = target[m].values
    w, _ = nnls(X, y)
    yhat = X @ w
    scale = w.sum()
    wn = w / scale if scale > 0 else w
    keep = wn >= min_weight
    tilts = np.array([k[0] for k in keys])[keep]
    azs = np.array([k[1] for k in keys])[keep]
    ws = wn[keep] / wn[keep].sum() if keep.any() else wn[keep]
    te, ae = arr.mean_angles(tilts, azs, ws) if keep.any() else (np.nan, np.nan)
    return {"weights": pd.Series(wn, index=pd.MultiIndex.from_tuples(keys, names=["tilt", "azimuth"])),
            "tilt_eff": float(te), "azimuth_eff": float(ae), "r2": _r2(y, yhat), "scale": float(scale),
            "n_hours": int(m.sum()), "n_components": int(keep.sum())}


def morning_evening_asymmetry(power: pd.Series, solpos: pd.DataFrame, elev_bins=(10, 20, 30, 40)) -> pd.DataFrame:
    """Mean normalised power east (sun azimuth < 180) vs west at matched elevation bands; a
    west surplus points to a west-facing orientation (or a morning horizon/soiling effect)."""
    elev = 90.0 - solpos["apparent_zenith"]
    east = solpos["azimuth"] < 180.0
    rows = []
    for lo, hi in zip(elev_bins[:-1], elev_bins[1:]):
        b = (elev >= lo) & (elev < hi) & power.notna()
        pe, pw = power[b & east].mean(), power[b & ~east].mean()
        rows.append({"elev_lo": lo, "elev_hi": hi, "east": pe, "west": pw, "west_minus_east": pw - pe,
                     "n_east": int((b & east).sum()), "n_west": int((b & ~east).sum())})
    return pd.DataFrame(rows)
