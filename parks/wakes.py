"""PyWake NOJ wake factor w(t) for one real park (round-2 WP5 method).

As in scripts/round2/wp5_precompute_wakes.py: Jensen_1983 on a UniformSite
(ti 0.1), real layout in UTM32, heterogeneous types, w(t) = P_waked / P_free
from the same PyWake run, clipped to (0, 1], applied to the chain's
free-stream park sum. Differences for real parks:
  - wake types are (lib_name, hub height, MaStR rotor) via the automatic
    library match (round2.wake.resolve_columns), not MODEL_MAP;
  - inflow speed = capacity-weighted mean of the groups' hub-height winds of
    THIS run's free-stream output (the round-2 wake staleness bug came from
    computing w on the wind of an older chain state; here w is derived from
    the very frame it multiplies, and the manifest stores that frame's hash);
  - inflow direction = capacity-weighted vector mean of the groups' 100 m
    ERA5 directions (parks spanning several cells).
"""

import warnings

import numpy as np
import pandas as pd

from round2 import wake as r2_wake

TI = 0.1


def inflow(free: pd.DataFrame, groups: pd.DataFrame) -> pd.DataFrame:
    """Capacity-weighted park inflow (ws, wd) from the free-stream frame."""
    w = (groups["n"] * groups["rated_cap_kw"]).values.astype(float)
    w = w / w.sum()
    ws = sum(wi * free[f"wind_speed_hub_{g}"].values for wi, g in zip(w, groups["group_id"]))
    rad = [np.radians(free[f"wind_direction_100m_{g}"].values) for g in groups["group_id"]]
    sx = sum(wi * np.sin(r) for wi, r in zip(w, rad))
    cx = sum(wi * np.cos(r) for wi, r in zip(w, rad))
    wd = np.degrees(np.arctan2(sx, cx)) % 360.0
    return pd.DataFrame({"ws": ws, "wd": wd}, index=free.index)


def wake_types(layout: pd.DataFrame) -> pd.DataFrame:
    """Unique (lib_name, hub, rotor) combinations with a stable PyWake name."""
    t = layout[["lib_name", "hub_height", "rotor_diameter"]].drop_duplicates().sort_values(
        ["lib_name", "hub_height", "rotor_diameter"]).reset_index(drop=True)
    t["name"] = [f"{r.lib_name}|{r.hub_height:g}m|D{r.rotor_diameter:g}" for r in t.itertuples()]
    return t


def wake_factor(layout: pd.DataFrame, flow: pd.DataFrame, k: float,
                pc: pd.DataFrame, ct: pd.DataFrame) -> tuple:
    """w(t) series and a per-type info table (generic Ct used or not)."""
    from py_wake.literature.noj import Jensen_1983
    from py_wake.site import UniformSite
    from py_wake.wind_turbines import WindTurbines

    types = wake_types(layout)
    wts, info = [], []
    for r in types.itertuples():
        wt, generic = r2_wake.build_windturbine(r.lib_name, float(r.hub_height), float(r.rotor_diameter),
                                                np.nan, pc, ct, name=r.name)
        wts.append(wt)
        info.append({"wake_type": r.name, "lib_name": r.lib_name, "hub_height": r.hub_height,
                     "rotor_diameter": r.rotor_diameter, "generic_ct": generic,
                     "n_turbines": int(((layout.lib_name == r.lib_name) & (layout.hub_height == r.hub_height)
                                        & (layout.rotor_diameter == r.rotor_diameter)).sum())})
    key = list(zip(types.lib_name, types.hub_height, types.rotor_diameter))
    type_idx = np.array([key.index(k_) for k_ in zip(layout.lib_name, layout.hub_height, layout.rotor_diameter)])
    x, y = layout["x_utm32"].values, layout["y_utm32"].values
    counts = np.bincount(type_idx, minlength=len(wts))

    if len(layout) == 1:
        return pd.Series(1.0, index=flow.index, name="w"), pd.DataFrame(info)

    wfm = Jensen_1983(UniformSite(p_wd=[1.0], ti=TI), WindTurbines.from_WindTurbine_lst(wts), k=k)
    parts = []
    for _, chunk in flow.groupby(pd.Grouper(freq="MS")):
        if chunk.empty:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sim = wfm(x, y, type=type_idx, wd=chunk["wd"].values, ws=chunk["ws"].values, time=True)
            p_waked = sim.Power.sum("wt").values
            p_free = sum(wts[i].power(chunk["ws"].values) * counts[i] for i in range(len(wts)))
        w = np.divide(p_waked, p_free, out=np.ones_like(p_free, dtype=float), where=p_free > 0)
        parts.append(pd.Series(np.clip(w, 1e-6, 1.0), index=chunk.index))
    return pd.concat(parts).sort_index().rename("w"), pd.DataFrame(info)


def park_efficiency(power_free: pd.Series, w: pd.Series) -> float:
    """Energy-weighted wake efficiency sum(P_free * w) / sum(P_free)."""
    pf = power_free.values
    return float((pf * w.reindex(power_free.index).values).sum() / pf.sum()) if pf.sum() > 0 else 1.0
