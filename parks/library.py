"""Power-curve library access for real parks: curves, Ct, specs + spec overrides.

Only real library curves are used (no generic or substituted power curves).
Five library types carry '-' instead of numbers in turbine_specs.csv; the
missing fields are derived from the type's own power curve (rule calibrated
on the 347 complete library rows, median error 0 for every field):
  cut_in    first wind speed with P > 0
  rated     first wind speed with P >= 0.99 * Pmax
  cut_out   last wind speed with P > 0
Existing numeric fields are never overwritten. The derived values live in a
separate override table (paths.SPEC_OVERRIDES_CSV); turbine_specs.csv stays
untouched.
"""

import os

import numpy as np
import pandas as pd

from parks import paths

POWER_CSV = os.path.join(paths.REPO, "power_curves", "turbine_power.csv")
CT_CSV = os.path.join(paths.REPO, "power_curves", "turbine_ct.csv")
SPECS_CSV = os.path.join(paths.REPO, "power_curves", "turbine_specs.csv")

SPEC_FIELDS = {"cut_in": "Einschaltgeschwindigkeit", "cut_out": "Abschaltgeschwindigkeit",
               "rated_ws": "Nennwindgeschwindigkeit"}
RATED_SHARE = 0.99


def load_power_curves() -> pd.DataFrame:
    """kW per wind speed, one column per library type (duplicates dropped)."""
    pc = pd.read_csv(POWER_CSV, sep=";", decimal=",", index_col=0)
    return pc.loc[:, ~pc.columns.duplicated()]


def load_ct_curves() -> pd.DataFrame:
    ct = pd.read_csv(CT_CSV, sep=";", index_col=0)
    return ct.loc[:, ~ct.columns.duplicated()]


def load_specs() -> pd.DataFrame:
    """First library row per type (the generator uses .iloc[0] the same way)."""
    return pd.read_csv(SPECS_CSV, sep=";").drop_duplicates("Turbine").set_index("Turbine")


def derive_specs_from_curve(curve_kw: pd.Series) -> dict:
    c = curve_kw.dropna()
    ws = c.index.values.astype(float)
    p = c.values.astype(float)
    if not (p > 0).any():
        raise ValueError(f"curve {curve_kw.name!r} has no positive power")
    pos = ws[p > 0]
    return {"cut_in": float(pos[0]), "rated_ws": float(ws[p >= RATED_SHARE * p.max()][0]),
            "cut_out": float(pos[-1])}


def build_spec_overrides(types, pc: pd.DataFrame = None, specs: pd.DataFrame = None) -> pd.DataFrame:
    """Rows (turbine, field, value, source) for every non-numeric spec field."""
    pc = load_power_curves() if pc is None else pc
    specs = load_specs() if specs is None else specs
    rows = []
    for t in sorted(set(types)):
        derived = None
        for field, col in SPEC_FIELDS.items():
            if pd.notna(pd.to_numeric(specs.loc[t, col], errors="coerce")):
                continue
            derived = derived or derive_specs_from_curve(pc[t])
            rows.append({"turbine": t, "field": field, "value": derived[field],
                         "library_value": specs.loc[t, col],
                         "source": "derived from library power curve"})
    return pd.DataFrame(rows, columns=["turbine", "field", "value", "library_value", "source"])


def turbine_specs(types, overrides: pd.DataFrame = None, specs: pd.DataFrame = None) -> dict:
    """{type: {diameter, cut_in, cut_out, rated}} in the format of
    generate_wind.get_turbines(), with overrides applied."""
    specs = load_specs() if specs is None else specs
    ov = {} if overrides is None or overrides.empty else \
        {(r.turbine, r.field): float(r.value) for r in overrides.itertuples()}
    out = {}
    for t in sorted(set(types)):
        vals = {}
        for field, col in SPEC_FIELDS.items():
            v = ov.get((t, field), pd.to_numeric(specs.loc[t, col], errors="coerce"))
            if pd.isna(v):
                raise ValueError(f"spec {field} missing for {t!r} and no override")
            vals[field] = float(v)
        out[t] = {"diameter": float(specs.loc[t, "Rotordurchmesser"]), "cut_in": vals["cut_in"],
                  "cut_out": vals["cut_out"], "rated": vals["rated_ws"]}
    return out


def power_curves_w(types, pc: pd.DataFrame = None) -> pd.DataFrame:
    """Curves in W for the given types, exactly like generate_wind.get_turbines()."""
    pc = load_power_curves() if pc is None else pc
    out = pc[sorted(set(types))] * 1000.0
    out.index.name = "wind_speed"
    return out


def curve_max_kw(types, pc: pd.DataFrame = None) -> dict:
    pc = load_power_curves() if pc is None else pc
    return {t: float(pc[t].max()) for t in set(types)}


def load_overrides() -> pd.DataFrame:
    if os.path.exists(paths.SPEC_OVERRIDES_CSV):
        return pd.read_csv(paths.SPEC_OVERRIDES_CSV)
    return pd.DataFrame(columns=["turbine", "field", "value", "library_value", "source"])


def ct_available(lib_name: str, ct: pd.DataFrame = None) -> bool:
    ct = load_ct_curves() if ct is None else ct
    return lib_name in ct.columns and ct[lib_name].notna().sum() >= 3
