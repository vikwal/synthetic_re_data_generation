"""Cross-check estimators for the orientation of a park. None of them enters a decision in pv.plausibility;
they are reported next to the grid estimate in the calibration and in the step-1 benchmark.

  half_energy_azimuth / azimuth_from_daily_centroid   per clear day the sun azimuth at the half-energy time,
                                                       inverted through the bank statistic at a guessed tilt
  tilt_from_noon_ratio                                 P_noon / BHI_noon = s (cos b + sin b tan z) on clear noons
                                                       (south-facing assumed); needs a spread of noon zenith angles
  morning_evening_asymmetry                            mean normalised power east vs west of noon per elevation band
"""
import numpy as np
import pandas as pd

from pv import timebase, transposition


def half_energy_azimuth(power: pd.Series, sun_az: pd.Series, day_mask: pd.Series, min_hours: int = 6) -> pd.Series:
    """Per day with >= min_hours valid clear hours: sun azimuth (deg) at the time the cumulative energy reaches
    50 % (linear interpolation between hours)."""
    out = {}
    p = power.where(day_mask)
    for d, g in p.groupby(p.index.date):
        g = g.dropna()
        if len(g) < min_hours or g.sum() <= 0:
            continue
        c = g.cumsum() / g.sum()
        az = sun_az.reindex(g.index).values
        j = int(np.searchsorted(c.values, 0.5))
        if j == 0:
            out[d] = float(az[0])
            continue
        f = (0.5 - c.values[j - 1]) / max(c.values[j] - c.values[j - 1], 1e-9)
        out[d] = float(az[j - 1] + f * (az[j] - az[j - 1]))
    return pd.Series(out, dtype=float)


def azimuth_from_daily_centroid(power: pd.Series, bank: dict, sun_az: pd.Series, mask: pd.Series, tilt_guess: float = 25.0) -> dict:
    """Daily half-energy azimuths of the measurement, mapped to module azimuth through the same statistic of the
    bank series at tilt_guess (lookup, monotone in azimuth). Returns median, IQR and the daily series."""
    obs = half_energy_azimuth(power, sun_az, mask)
    if obs.empty:
        return {"azimuth": np.nan, "iqr": np.nan, "n_days": 0}
    cal = {a: half_energy_azimuth(poa, sun_az, mask) for (t, a), poa in bank.items() if abs(t - tilt_guess) < 1e-6}
    cal = pd.DataFrame(cal).reindex(obs.index)                    # days x module azimuths
    est = []
    for d in obs.index:
        row = cal.loc[d].dropna()
        if len(row) < 3:
            continue
        xs, ys = row.values, row.index.values.astype(float)      # bank statistic -> module azimuth
        o = np.argsort(xs)
        est.append(float(np.interp(obs[d], xs[o], ys[o])))
    est = pd.Series(est)
    return {"azimuth": float(est.median()), "iqr": float(est.quantile(0.75) - est.quantile(0.25)), "n_days": int(len(est)), "daily": est}


def tilt_from_noon_ratio(power: pd.Series, drv: pd.DataFrame, lat: float, lon: float, alt: float, mask: pd.Series,
                         albedo: float = 0.2, iterations: int = 2, gamma: float = None, max_zenith: float = 65.0) -> dict:
    """Clear-noon regression P/BHI = a + b tan(z) + c diff/BHI (south-facing plane) after removing the diffuse and
    ground-reflected share for the current tilt guess -> tilt = atan(b / a). Needs noon zenith spread (>= 10 deg).
    Low-sun noons (zenith > max_zenith) are dropped: inter-row shading, IAM and horizon losses at low sun flatten
    the seasonal slope and bias the tilt low (30 deg true -> 19 deg with all noons on synthetic parks).
    Optional temperature correction (gamma, T_cell ~ T_air + 0.03 GHI) when drv has temp_air."""
    if gamma is not None and "temp_air" in drv:
        t_cell = drv["temp_air"].fillna(12.0) + 0.03 * drv["ghi"].fillna(0.0)
        power = power / (1.0 + gamma * (t_cell.reindex(power.index) - 25.0))
    zen = timebase.solar_position(drv.index, lat, lon, alt)["apparent_zenith"]
    cs = timebase.clearsky(drv.index, lat, lon, alt)
    dni = transposition.dni_from_closure(drv["ghi"], drv["dhi"], zen, cs["dni"])
    bhi = (dni * np.cos(np.radians(zen))).clip(lower=0)
    dayz = zen.where(mask & (zen <= max_zenith)).groupby(zen.index.date).idxmin().dropna()   # clear hour nearest noon per day
    idx = pd.DatetimeIndex(dayz)
    if len(idx) < 5:
        return {"tilt": np.nan, "n_days": int(len(idx)), "zenith_spread": 0.0}
    y, z, b = power.reindex(idx), zen.reindex(idx), bhi.reindex(idx)
    tilt = 25.0
    out = {}
    for _ in range(iterations):
        diff = drv["dhi"].reindex(idx) * (1 + np.cos(np.radians(tilt))) / 2 + albedo * drv["ghi"].reindex(idx) * (1 - np.cos(np.radians(tilt))) / 2
        X = np.column_stack([np.ones(len(idx)), np.tan(np.radians(z)).values, (diff / b).values])
        ok = np.isfinite(X).all(1) & y.notna().values & (b.values > 50)
        if ok.sum() < 5:
            return {"tilt": np.nan, "n_days": int(ok.sum()), "zenith_spread": float(np.ptp(z[ok])) if ok.any() else 0.0}
        target = (y / b).values[ok]
        coef, *_ = np.linalg.lstsq(X[ok], target, rcond=None)
        a, bb = coef[0], coef[1]
        tilt = float(np.degrees(np.arctan2(bb, a))) if a > 0 else np.nan
        out = {"tilt": tilt, "scale": float(np.hypot(a, bb)), "n_days": int(ok.sum()), "zenith_spread": float(np.ptp(z[ok])),
               "r2_fit": float(1 - np.sum((X[ok] @ coef - target) ** 2) / max(np.var(target) * ok.sum(), 1e-12))}
        if not np.isfinite(tilt):
            break
    return out


def morning_evening_asymmetry(power: pd.Series, solpos: pd.DataFrame, elev_bins=(10, 20, 30, 40)) -> pd.DataFrame:
    """Mean normalised power east (sun azimuth < 180) vs west at matched elevation bands; a west surplus points
    to a west-facing orientation (or a morning horizon/soiling effect)."""
    elev = 90.0 - solpos["apparent_zenith"]
    east = solpos["azimuth"] < 180.0
    rows = []
    for lo, hi in zip(elev_bins[:-1], elev_bins[1:]):
        b = (elev >= lo) & (elev < hi) & power.notna()
        pe, pw = power[b & east].mean(), power[b & ~east].mean()
        rows.append({"elev_lo": lo, "elev_hi": hi, "east": pe, "west": pw, "west_minus_east": pw - pe,
                     "n_east": int((b & east).sum()), "n_west": int((b & ~east).sum())})
    return pd.DataFrame(rows)


def run(power: pd.Series, drv: pd.DataFrame, lat: float, lon: float, alt: float, bank: dict, mask: pd.Series, tilt_guess: float = 25.0) -> dict:
    """Both cross-checks on the final mask of pv.plausibility.assess (returned there as 'mask')."""
    sun_az = timebase.solar_position(drv.index, lat, lon, alt)["azimuth"]
    cen = azimuth_from_daily_centroid(power, bank, sun_az, mask, tilt_guess=tilt_guess)
    return {"centroid_azimuth": {k: v for k, v in cen.items() if k != "daily"},
            "noon_tilt": tilt_from_noon_ratio(power, drv, lat, lon, alt, mask)}
