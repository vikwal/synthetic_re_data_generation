"""Inverter level: PVWatts or Sandia AC model, library selection by AC rating and era."""

import os
import re

import numpy as np
import pandas as pd
import pvlib

CEC_INV_2026 = os.path.expanduser("~/Work/FL_Contribution/data/mastr/CEC_Inverters.csv")
_cache = {}


def _norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def bundled_inverters() -> pd.DataFrame:
    if "bundled" not in _cache:
        _cache["bundled"] = pvlib.pvsystem.retrieve_sam("CECInverter").T
    return _cache["bundled"]


def inverters_2026(path: str = CEC_INV_2026) -> pd.DataFrame:
    if "inv2026" not in _cache:
        df = pd.read_csv(path, skiprows=[1, 2], low_memory=False).set_index("Name")
        for c in ["Vac", "Pso", "Paco", "Pdco", "Vdco", "C0", "C1", "C2", "C3", "Pnt", "Vdcmax", "Idcmax", "Mppt_low", "Mppt_high"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        _cache["inv2026"] = df
    return _cache["inv2026"]


def inverter_params(name: str) -> pd.Series:
    for lib in (bundled_inverters(), inverters_2026()):
        hit = lib[lib.index.map(_norm).str.contains(_norm(name))]
        if not hit.empty:
            row = hit.iloc[0].copy()
            row["name"] = row.name
            return row
    raise KeyError(f"inverter '{name}' not found")


def typical_unit_kw(year: int, pac_total_kw: float) -> float:
    """Typical single-inverter AC size for a ground-mounted park of that era."""
    if pac_total_kw < 300:
        return min(pac_total_kw, 100.0)
    if year <= 2014:
        return 500.0 if pac_total_kw < 3000 else 1000.0
    if year <= 2019:
        return 1500.0 if pac_total_kw >= 2000 else 100.0
    return 185.0 if pac_total_kw < 10000 else 3000.0


def pick_inverter(pac_total_w: float, year: int = 2015, unit_kw: float = None) -> tuple:
    """(params, n_units): the library inverter with Paco closest to the typical unit size,
    replicated n_units times to reach the park AC rating."""
    unit = (unit_kw or typical_unit_kw(year, pac_total_w / 1000.0)) * 1000.0
    lib = pd.concat([bundled_inverters().assign(library="bundled"), inverters_2026().assign(library="2026")])
    lib = lib.dropna(subset=["Paco", "Pdco", "Vdco", "C0", "C1", "C2", "C3", "Pso"])
    lib = lib[(lib["Paco"] >= 0.5 * unit) & (lib["Paco"] <= 2.0 * unit)] if ((lib["Paco"] >= 0.5 * unit) & (lib["Paco"] <= 2.0 * unit)).any() else lib
    row = lib.iloc[(lib["Paco"] - unit).abs().argsort().iloc[0]].copy()
    row["name"] = row.name
    n = max(1, int(round(pac_total_w / float(row["Paco"]))))
    return row, n


def ac_pvwatts(p_dc: pd.Series, pac0_w: float, eta_inv_nom: float = 0.96, eta_inv_ref: float = 0.9637) -> pd.Series:
    return pvlib.inverter.pvwatts(p_dc, pac0_w, eta_inv_nom=eta_inv_nom, eta_inv_ref=eta_inv_ref).clip(lower=0.0)


def ac_sandia(p_dc: pd.Series, v_dc: pd.Series, inv: pd.Series, n_units: int, scale_dc: float = 1.0) -> pd.Series:
    """Sandia inverter model for n identical units sharing the DC power equally. v_dc is the string
    voltage (V); scale_dc = n_units * Paco / pac0 maps the park onto n library units so that the
    AC rating is exactly pac0 while the library efficiency curve keeps its shape."""
    per_unit = p_dc / n_units * scale_dc
    ac = pvlib.inverter.sandia(v_dc, per_unit, inv)
    return (ac * n_units / scale_dc).clip(lower=0.0)


def strings_for(inv: pd.Series, v_mp_module: float) -> int:
    """Modules per string so that the string MPP voltage sits at the inverter's nominal DC voltage."""
    return max(1, int(round(float(inv["Vdco"]) / float(v_mp_module))))
