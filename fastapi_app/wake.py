"""
Simplified Jensen/Park wake model (pure numpy/pandas — no I/O, no framework).

Assumptions (standard engineering simplifications, not CFD):
  * constant thrust coefficient Ct (default 0.8)
  * wake expands linearly with downwind distance: r(x) = r0 + k * x
  * several wakes are combined by root-sum-square (Katic et al., 1986)
  * partial rotor overlap via circle-circle intersection area
  * wind direction at each time step comes from the upstream turbine's own u/v
The wake effect is applied as a *ratio* of power-curve values, so it can never
increase output.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def offset_m(lat1, lon1, lat2, lon2):
    """Local tangent-plane (east, north) offset in metres from point 1 to point 2."""
    return (lon2 - lon1) * 111320.0 * np.cos(np.radians(lat1)), (lat2 - lat1) * 110540.0


def offset_to_latlon(lat, lon, east_m, north_m):
    """Inverse of offset_m for small distances."""
    return lat + north_m / 110540.0, lon + east_m / (111320.0 * np.cos(np.radians(lat)))


def circle_overlap_area(d, r1, r2):
    """Intersection area of a circle (radius r1, centre distance d) with a fixed circle r2."""
    d = np.asarray(d, float)
    r1 = np.asarray(r1, float)
    r2 = float(r2)
    out = np.zeros_like(d)
    full = d <= np.abs(r1 - r2)
    partial = ~full & ~(d >= r1 + r2)
    out[full] = np.pi * np.minimum(r1[full], r2) ** 2
    dp = np.maximum(d[partial], 1e-6)
    r1p = r1[partial]
    t1 = r1p ** 2 * np.arccos(np.clip((dp ** 2 + r1p ** 2 - r2 ** 2) / (2 * dp * r1p), -1, 1))
    t2 = r2 ** 2 * np.arccos(np.clip((dp ** 2 + r2 ** 2 - r1p ** 2) / (2 * dp * r2), -1, 1))
    t3 = 0.5 * np.sqrt(np.clip((-dp + r1p + r2) * (dp + r1p - r2) * (dp - r1p + r2) * (dp + r1p + r2), 0, None))
    out[partial] = t1 + t2 - t3
    return out


def apply_wake_losses(dfs, meta, wake_decay=0.075, thrust_coefficient=0.8):
    """
    Reduce each turbine's wind speed / power for the wakes of all others, in place.

    dfs[j]  : DataFrame with wind_speed_t{j+1}, power_t{j+1}, u_wind_100m, v_wind_100m
    meta[j] : dict(lat, lon, diameter [m], power_curve [Series speed -> W, dense])
    Returns : list of per-turbine energy loss in percent (>= 0).
    """
    n = len(dfs)
    if n < 2:
        return [0.0] * n
    idx = dfs[0].index
    for d in dfs[1:]:
        idx = idx.intersection(d.index)
    if len(idx) == 0:
        return [0.0] * n

    radii = [m["diameter"] / 2.0 for m in meta]
    dx = np.zeros((n, n))
    dy = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                dx[i, j], dy[i, j] = offset_m(meta[i]["lat"], meta[i]["lon"], meta[j]["lat"], meta[j]["lon"])

    sq = [np.zeros(len(idx)) for _ in range(n)]
    one_minus = 1 - np.sqrt(max(0.0, 1 - thrust_coefficient))
    for i in range(n):
        u = dfs[i].loc[idx, "u_wind_100m"].to_numpy(float)
        v = dfs[i].loc[idx, "v_wind_100m"].to_numpy(float)
        spd = np.sqrt(u ** 2 + v ** 2)
        safe = np.where(spd < 0.1, np.nan, spd)
        uh, vh = u / safe, v / safe
        for j in range(n):
            if i == j:
                continue
            down = dx[i, j] * uh + dy[i, j] * vh          # distance along the flow
            cross = -dx[i, j] * vh + dy[i, j] * uh        # lateral offset
            wr = radii[i] + wake_decay * down
            valid = (down > 1.0) & (wr > 0) & ~np.isnan(down)
            deficit = np.zeros(len(idx))
            if valid.any():
                area = np.zeros(len(idx))
                area[valid] = circle_overlap_area(np.abs(cross[valid]), wr[valid], radii[j])
                frac = area / (np.pi * radii[j] ** 2)
                base = one_minus * (radii[i] / np.where(wr == 0, np.nan, wr)) ** 2
                deficit = np.where(valid, np.nan_to_num(frac * base), 0.0)
            sq[j] = sq[j] + deficit ** 2

    losses = []
    for j in range(n):
        total = np.clip(np.sqrt(sq[j]), 0.0, 0.9)
        sc, pc = f"wind_speed_t{j + 1}", f"power_t{j + 1}"
        v0 = dfs[j].loc[idx, sc].to_numpy(float)
        v1 = v0 * (1 - total)
        curve = meta[j]["power_curve"]
        c0 = np.interp(v0, curve.index.values, curve.values)
        c1 = np.interp(v1, curve.index.values, curve.values)
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = np.where(c0 > 1.0, c1 / c0, 1.0)
        ratio = np.clip(np.nan_to_num(ratio, nan=1.0), 0.0, 1.0)
        p0 = dfs[j].loc[idx, pc].to_numpy(float)
        p1 = p0 * ratio
        dfs[j].loc[idx, sc] = v1
        dfs[j].loc[idx, pc] = p1
        s0 = np.nansum(p0)
        loss = 0.0 if (not np.isfinite(s0) or s0 <= 0) else 100.0 * (1 - np.nansum(p1) / s0)
        losses.append(max(0.0, float(loss)))
    return losses
