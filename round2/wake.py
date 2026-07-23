"""WP5 — PyWake NOJ wake losses for the 13 validation parks.

Publishes w(t) = P_waked / P_free in (0,1] per park, precomputed by
scripts/round2/wp5_precompute_wakes.py and applied to the park sum in
generate_wind_era5_v2 (Hook E). py_wake version is pinned in
requirements_round2.txt and recorded in the run manifests.
"""

import os

import numpy as np
import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# layout model name -> (power-curve column, CT column or None)
MODEL_MAP = {
    "E-82 E2 2300": ("Enercon E-82 E2 2.300", None),
    "V126 3300": ("Vestas V126-3.3", "Vestas V126-3.3"),
    "N117 2400": ("Nordex N117 2400", "Nordex N117 Gamma"),
    "MM100 2000": ("Senvion MM100", "Senvion MM100"),
    "M104 3400": ("Senvion 3.4M104", None),
    "M114 3200": ("Senvion 3.2M114", "Senvion 3.4M114"),  # 3.2 CT proxied by 3.4
    "E-66/18.70 1800": ("Enercon E-66/18.70", None),
    "E-58/10.58 1000": ("Enercon E-58/10.58", None),
    "E-53 800": ("Enercon E-53", None),
    "N131 3300": ("Nordex N131/3300 Delta", "Nordex N131/3300 Delta"),
    "E-101 3000": ("Enercon E-101", "Enercon E-101"),
    "E-115 3000": ("Enercon E-115 3.000", "Enercon E-115 3.000"),
    "V126 3450": ("Vestas V126-3.45", "Vestas V126-3.45"),
    "V150 4200": ("Vestas V150-4.2", None),
}


def load_curves():
    pc = pd.read_csv(os.path.join(REPO, "power_curves", "turbine_power.csv"),
                     sep=";", decimal=",", index_col=0)
    pc = pc.loc[:, ~pc.columns.duplicated()]
    ct = pd.read_csv(os.path.join(REPO, "power_curves", "turbine_ct.csv"),
                     sep=";", index_col=0)
    ct = ct.loc[:, ~ct.columns.duplicated()]
    return pc, ct


def build_windturbine(model: str, hub_height: float, rotor_diameter: float,
                      rated_kw: float, pc: pd.DataFrame, ct: pd.DataFrame):
    """PyWake WindTurbine for one layout model. Missing CT curves fall back to
    the GenericWindTurbine CT (guide-sanctioned; logged by the caller)."""
    from py_wake.wind_turbines import WindTurbine
    from py_wake.wind_turbines.power_ct_functions import PowerCtTabular
    from py_wake.wind_turbines.generic_wind_turbines import GenericWindTurbine

    pcol, ctcol = MODEL_MAP[model]
    curve = (pc[pcol] * 1000.0).dropna()  # kW -> W
    ws = curve.index.values.astype(float)
    power_w = curve.values.astype(float)
    generic_ct_used = False
    if ctcol is not None and ctcol in ct.columns:
        ct_curve = ct[ctcol].dropna()
        ct_vals = np.interp(ws, ct_curve.index.values.astype(float),
                            ct_curve.values.astype(float))
    else:
        # power_norm from the actual curve max (layout kW fields are noisy)
        gen = GenericWindTurbine(name=model, diameter=rotor_diameter,
                                 hub_height=hub_height,
                                 power_norm=float(power_w.max()) / 1000.0)
        ct_vals = gen.ct(ws)
        generic_ct_used = True
    ct_vals = np.clip(ct_vals, 1e-3, 1.0)
    wt = WindTurbine(name=model, diameter=rotor_diameter, hub_height=hub_height,
                     powerCtFunction=PowerCtTabular(ws, power_w, "w", ct_vals))
    return wt, generic_ct_used


def wind_direction_met(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Meteorological direction (deg, wind FROM) out of u/v components."""
    return (np.degrees(np.arctan2(-u, -v))) % 360.0
