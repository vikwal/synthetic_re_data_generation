"""WP2 runtime — quantile-mapping application and gated branch logic.

The correction is fitted offline (scripts/round2/wp2*.py) and stored as
per-location quantile tables. At runtime a 13-point quantile map
v_corr = F_loc^-1(F_ERA5(v)) is applied per hour with linear interpolation
between quantiles and linear tail extension (guide WP2-D).

Height-consistent rule (guide WP2): the relative correction factor
c(v10) = QM(v10)/v10 multiplies BOTH v10 and v100, so alpha is unchanged and
v(hub) scales exactly by (1+c). 'wind10_only' (correction on v10 only) is kept
as the WP6 lower bound.
"""

import numpy as np
import pandas as pd

QUANTILES = np.array([0.05, 0.125, 0.20, 0.275, 0.35, 0.425, 0.50,
                      0.575, 0.65, 0.725, 0.80, 0.875, 0.95])


def quantile_map(v, q_src, q_dst):
    """Map values v from the source distribution to the target one via the
    13-point empirical quantile tables. Linear interpolation inside, linear
    tail extension outside (slope of the outermost segment)."""
    v = np.asarray(v, dtype=float)
    q_src = np.asarray(q_src, dtype=float)
    q_dst = np.asarray(q_dst, dtype=float)
    if not (np.all(np.diff(q_src) >= 0) and np.all(np.diff(q_dst) >= 0)):
        raise ValueError("quantile tables must be non-decreasing")
    out = np.interp(v, q_src, q_dst)
    # linear tail extension
    lo_slope = ((q_dst[1] - q_dst[0]) / (q_src[1] - q_src[0])
                if q_src[1] > q_src[0] else 1.0)
    hi_slope = ((q_dst[-1] - q_dst[-2]) / (q_src[-1] - q_src[-2])
                if q_src[-1] > q_src[-2] else 1.0)
    below = v < q_src[0]
    above = v > q_src[-1]
    out[below] = q_dst[0] + (v[below] - q_src[0]) * lo_slope
    out[above] = q_dst[-1] + (v[above] - q_src[-1]) * hi_slope
    return np.clip(out, 0.0, None)


def correction_factor(v10, q_era5, q_target):
    """Relative per-hour factor c so that v_corr = c * v10 (c>0)."""
    v10 = np.asarray(v10, dtype=float)
    corrected = quantile_map(v10, q_era5, q_target)
    with np.errstate(divide="ignore", invalid="ignore"):
        c = np.where(v10 > 0, corrected / v10, 1.0)
    return np.clip(c, 0.2, 5.0)  # sanity clamp against tail blow-ups


def assign_branch(dist_km, delta_elev_std, predicted_class,
                  dist_max_km: float = 15.0, elev_std_max_m: float = 20.0):
    """Guide WP2-D gate: A (nearest-station QM), B (model quantiles), C (none)."""
    if dist_km <= dist_max_km and abs(delta_elev_std) <= elev_std_max_m:
        return "A"
    if predicted_class >= 2:
        return "B"
    return "C"


def apply_bc_factor(df: pd.DataFrame, sf: pd.Series,
                    mode: str = "height_consistent") -> pd.DataFrame:
    """Apply an hourly era5_bc scaling factor (TR model) in place.

    mode 'height_consistent': factor on wind_speed_10m AND _100m (same
    algebra as the height-consistent QM, shear exponent preserved).
    mode 'wind10_only': factor on the 10 m level only — with the per-hour
    two-point power law (alpha from v10/v100) the corrected 10 m wind then
    re-anchors the vertical profile while ERA5's 100 m level stays untouched,
    so the correction fades with height (surface-bias hypothesis).

    sf: hourly factors, tz-aware UTC index (results/era5_bc/park_sf_*.parquet).
    Hours without a factor (before the BC context start) stay uncorrected.
    The stored column keeps the qm_factor name for downstream compatibility."""
    aligned = sf.reindex(df.index)
    matched = float(aligned.notna().mean())
    assert matched > 0.5, (
        f"bc factor alignment failed ({matched:.0%} matched) — "
        "timestamp/timezone mismatch between synth index and factor parquet?")
    c = aligned.fillna(1.0).clip(0.2, 5.0).values
    df["qm_factor"] = c
    df["wind_speed_10m"] = df["wind_speed_10m"] * c
    if mode == "height_consistent":
        df["wind_speed_100m"] = df["wind_speed_100m"] * c
    elif mode != "wind10_only":
        raise ValueError(f"unknown bc mode: {mode}")
    return df


def apply_gated_correction(df: pd.DataFrame, q_era5, q_target,
                           mode: str = "height_consistent") -> pd.DataFrame:
    """Apply the QM correction in place on wind_speed_10m/_100m.

    mode: 'off' | 'wind10_only' | 'height_consistent'
    q_era5/q_target: 13-point quantile tables of ERA5-10m and the target
    distribution at this location (branch A: nearest-station empirical;
    branch B: model-predicted). Pass q_target=None for branch C (no-op).
    """
    if mode == "off" or q_target is None:
        df["qm_factor"] = 1.0
        return df
    c = correction_factor(df["wind_speed_10m"].values, q_era5, q_target)
    df["qm_factor"] = c
    df["wind_speed_10m"] = df["wind_speed_10m"] * c
    if mode == "height_consistent":
        df["wind_speed_100m"] = df["wind_speed_100m"] * c
    elif mode != "wind10_only":
        raise ValueError(f"unknown correction mode: {mode}")
    return df
