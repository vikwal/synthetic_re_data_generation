"""Benchmark metrics: synthetic vs measured park power (both normalised to a capacity)."""

import numpy as np
import pandas as pd


def basic(y: pd.Series, p: pd.Series, cap: float) -> dict:
    m = y.notna() & p.notna()
    y, p = y[m].values, p[m].values
    if len(y) < 10:
        return {"n": int(len(y))}
    err = p - y
    ss = np.sum((y - y.mean()) ** 2)
    return {"n": int(len(y)), "r2": float(1 - np.sum(err ** 2) / ss) if ss > 0 else np.nan,
            "rmse": float(np.sqrt(np.mean(err ** 2))), "nrmse_cap": float(np.sqrt(np.mean(err ** 2)) / cap),
            "mae": float(np.mean(np.abs(err))), "bias": float(err.mean()), "nbias_cap": float(err.mean() / cap),
            "energy_ratio": float(p.sum() / y.sum()) if y.sum() > 0 else np.nan,
            "corr": float(np.corrcoef(y, p)[0, 1])}


def daytime(y: pd.Series, p: pd.Series, zenith: pd.Series, cap: float, zmax: float = 90.0) -> dict:
    d = zenith < zmax
    return basic(y[d], p[d], cap)


def by_group(y: pd.Series, p: pd.Series, groups: pd.Series, cap: float) -> pd.DataFrame:
    rows = []
    for g, idx in groups.groupby(groups).groups.items():
        r = basic(y.loc[idx], p.loc[idx], cap)
        r["group"] = g
        rows.append(r)
    return pd.DataFrame(rows).set_index("group")


def diurnal(y: pd.Series, p: pd.Series, by_season: bool = True) -> pd.DataFrame:
    df = pd.DataFrame({"meas": y, "synth": p})
    df["hour"] = df.index.hour
    df["season"] = df.index.month.map(lambda m: "DJF" if m in (12, 1, 2) else "MAM" if m in (3, 4, 5) else "JJA" if m in (6, 7, 8) else "SON")
    keys = ["season", "hour"] if by_season else ["hour"]
    return df.groupby(keys)[["meas", "synth"]].mean()


def monthly_energy(y: pd.Series, p: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"meas": y, "synth": p}).dropna()
    out = df.resample("MS").sum()
    out["ratio"] = out["synth"] / out["meas"].replace(0, np.nan)
    out["n_hours"] = df["meas"].resample("MS").count()
    return out


def clearsky_index(power: pd.Series, p_clear: pd.Series, zenith: pd.Series, zmax: float = 80.0) -> pd.Series:
    """Power clear-sky index against the synthetic clear-sky power of the same park."""
    d = (zenith < zmax) & (p_clear > 0.05 * p_clear.max())
    return (power / p_clear.replace(0, np.nan))[d].clip(0, 1.5)


def histogram(values: pd.Series, bins) -> pd.Series:
    h, e = np.histogram(values.dropna(), bins=bins)
    return pd.Series(h / max(h.sum(), 1), index=pd.IntervalIndex.from_breaks(e))


def ramps(power_norm: pd.Series) -> pd.Series:
    return power_norm.diff().dropna()


def plateaus(power: pd.Series, cap: float, min_len: int = 3, tol: float = 0.01, lo: float = 0.2, hi: float = 1.0) -> pd.DataFrame:
    """Runs of >= min_len consecutive hours where power stays within +-tol*cap of its run mean,
    between lo and hi of the series maximum: candidates for inverter clipping or curtailment."""
    pmax = power.max()
    x = power.values
    rows = []
    i, n = 0, len(x)
    while i < n:
        j = i
        while j + 1 < n and np.isfinite(x[j + 1]) and np.isfinite(x[i]) and abs(x[j + 1] - x[i:j + 2].mean()) <= tol * cap:
            j += 1
        if j - i + 1 >= min_len and np.isfinite(x[i]):
            level = x[i:j + 1].mean()
            if lo * pmax <= level <= hi * pmax:
                rows.append({"start": power.index[i], "end": power.index[j], "hours": j - i + 1, "level": level, "level_norm": level / cap})
        i = j + 1
    return pd.DataFrame(rows)


def plateau_summary(pl: pd.DataFrame, cap: float, clip_band: float = 0.03) -> dict:
    """Separate a recurring level (inverter/grid limit) from day-specific levels (curtailment)."""
    if pl.empty:
        return {"n_plateaus": 0, "plateau_hours": 0}
    lev = pl["level_norm"]
    mode = lev.round(2).mode().iloc[0]
    at_mode = (lev - mode).abs() <= clip_band
    return {"n_plateaus": int(len(pl)), "plateau_hours": int(pl["hours"].sum()), "mode_level_norm": float(mode),
            "hours_at_mode": int(pl.loc[at_mode, "hours"].sum()), "hours_off_mode": int(pl.loc[~at_mode, "hours"].sum()),
            "n_days_with_plateau": int(pl["start"].dt.date.nunique())}


def residual_by_sun(y: pd.Series, p: pd.Series, zenith: pd.Series, azimuth: pd.Series, cap: float,
                    elev_edges=(0, 10, 20, 30, 40, 50, 65), az_edges=(45, 90, 120, 150, 180, 210, 240, 270, 315)) -> pd.DataFrame:
    """Mean normalised residual (synth - meas) / cap in (elevation, azimuth) cells."""
    df = pd.DataFrame({"res": (p - y) / cap, "elev": 90.0 - zenith, "az": azimuth}).dropna()
    df["elev_bin"] = pd.cut(df["elev"], elev_edges)
    df["az_bin"] = pd.cut(df["az"], az_edges)
    return df.pivot_table(index="elev_bin", columns="az_bin", values="res", aggfunc="mean", observed=False)


def snow_candidates(y: pd.Series, p: pd.Series, temp_air: pd.Series, cap: float, ratio_max: float = 0.3, t_max: float = 1.0) -> pd.Series:
    """Days where the measured daily energy is below ratio_max of the synthetic one while the daily
    mean temperature is below t_max degC: snow cover (or an outage) candidates."""
    d = pd.DataFrame({"y": y, "p": p, "t": temp_air}).resample("D").agg({"y": "sum", "p": "sum", "t": "mean"})
    d = d[d["p"] > 0.05 * cap * 24 * 0.1]
    return ((d["y"] / d["p"]) < ratio_max) & (d["t"] < t_max)
