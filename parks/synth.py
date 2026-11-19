"""Free-stream synthesis of one real park, group by group.

Every turbine group runs through the unchanged chain functions of
generate_wind.py (get_features -> interpolate -> Cp -> generate_wind_power via
gen_full_dataframe), with
  - the ERA5 frame of the group's own grid cell (read from Postgres) after
    the chain's own input step utils.tools.knn_imputer (as in read_dfs; on
    gap-free input it only zeroes |x| < 0.01),
  - the group's own degradation vector (round2.aging.degradation_by_group),
  - the group's rating cap as rated_power (params.rated, kW),
  - no correction (M4-noQM: q_target None -> branch C).
generate_wind.main() itself is not used: it drives one DWD station per run
(park_id[:5] lookup, DB station masterdata, one commissioning date per park).
The per-group physics is identical; tests/test_parks.py checks single-group
equivalence against main() on a station config.

Output columns (per group t<i>; power is the GROUP TOTAL = n x one turbine):
  wind_speed_hub_t<i>  density_hub_t<i>  wind_direction_100m_t<i>
  aging_factor_t<i>    power_t<i> [W]
plus the primary cell's ERA5 base fields, derived 2 m fields and
power_park_free = sum of power_t<i>.
"""

import numpy as np
import pandas as pd

import generate_wind as gw
from parks import era5_db, library
from round2 import aging as r2_aging
from round2 import wake as r2_wake

DERIVED_COLS = ["wind_speed_100m", "relhum", "sat_vap_pressure", "density"]


def groups_of(cfg: dict) -> pd.DataFrame:
    p = cfg["params"]
    return pd.DataFrame({"group_id": p["group_ids"], "lib_name": p["turbines"],
                         "hub_height": p["hub_heights"], "rated_cap_kw": p["rated"],
                         "n": p["group_sizes"], "commissioning_date": p["commissioning_dates"],
                         "cell": p["era5_cells"]})


def check_config(cfg: dict) -> None:
    r2 = cfg["round2"]
    p = cfg["params"]
    lens = {k: len(p[k]) for k in ("turbines", "hub_heights", "rated", "group_ids",
                                   "group_sizes", "commissioning_dates", "era5_cells")}
    if len(set(lens.values())) != 1:
        raise ValueError(f"group arrays differ in length: {lens}")
    if r2.get("era5_source") != "db" or r2.get("commissioning_mode") != "group":
        raise ValueError("parks.synth needs round2.era5_source='db' and commissioning_mode='group'")
    if r2.get("correction", "off") != "off":
        raise ValueError("parks_v1 is M4-noQM: round2.correction must be 'off'")
    if r2.get("shear") != "power_law":
        raise ValueError("parks_v1 runs the power-law chain (no sshf/blh in the DB)")
    if p.get("noise", 0.0) != 0.0:
        raise ValueError("params.noise must be 0 in this stage")


def fetch_frames(conn, cells, start: str, end: str) -> dict:
    return {int(c): era5_db.fetch_cell(conn, int(c), start, end) for c in sorted(set(cells))}


def prepare_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """The chain's input preprocessing (generate_wind.read_dfs)."""
    return gw.tools.knn_imputer(data=frame.copy(), n_neighbors=5)


def run_park(cfg: dict, frames: dict, overrides: pd.DataFrame = None) -> pd.DataFrame:
    """Free-stream park frame on the output window [output_start, output_end]."""
    check_config(cfg)
    frames = {c: prepare_frame(f) for c, f in frames.items()}
    r2 = gw.get_round2_params(cfg)
    params = dict(cfg["params"])
    groups = groups_of(cfg)
    types = set(groups["lib_name"])
    specs = library.turbine_specs(types, overrides)
    curves = library.power_curves_w(types)
    ctx = {"q_era5": None, "q_target": None, "branch": "C", "z0": r2["z0_default"]}

    primary = int(cfg["park"]["primary_era5_cell"])
    index = frames[primary].index
    for c, f in frames.items():
        if not f.index.equals(index):
            raise ValueError(f"cell {c}: time index differs from the primary cell")

    if params.get("apply_ageing", True) and r2["aging_model"] != "none":
        dvs = r2_aging.degradation_by_group(
            index, dict(zip(groups["group_id"], groups["commissioning_date"])),
            model=r2["aging_model"], lam=r2["aging"]["lambda"], kappa=r2["aging"]["kappa"],
            step_delta=r2["aging"]["step_delta"])
    else:
        dvs = {g: None for g in groups["group_id"]}

    out = frames[primary][era5_db.RAW_COLS].copy()
    derived = gw.get_features(data=frames[primary].copy(), params=params, hub_height=100.0,
                              suffix="_derived", r2=r2, station_ctx=ctx)
    out[DERIVED_COLS] = derived[DERIVED_COLS]

    cols = {}
    for g in groups.itertuples(index=False):
        sfx = f"_{g.group_id}"
        frame = frames[int(g.cell)]
        df = gw.gen_full_dataframe(power_curves=curves, turbine=g.lib_name, params=params,
                                   df=frame.copy(), hub_height=float(g.hub_height), specs=specs,
                                   rated_power=float(g.rated_cap_kw) * 1000.0,
                                   degradation_vector=dvs[g.group_id], suffix_for_turbine_cols=sfx,
                                   r2=r2, station_ctx=ctx)
        dv = dvs[g.group_id]
        cols[f"wind_speed_hub{sfx}"] = df[f"wind_speed{sfx}"].values
        cols[f"density_hub{sfx}"] = df[f"density{sfx}"].values
        cols[f"wind_direction_100m{sfx}"] = r2_wake.wind_direction_met(
            frame["u_wind_100m"].values, frame["v_wind_100m"].values)
        cols[f"aging_factor{sfx}"] = np.ones(len(index)) if dv is None else np.asarray(dv, dtype=float)
        cols[f"power{sfx}"] = np.asarray(df[f"power{sfx}"], dtype=float) * int(g.n)
    out = pd.concat([out, pd.DataFrame(cols, index=index)], axis=1)
    out["power_park_free"] = out[[f"power_{g}" for g in groups["group_id"]]].sum(axis=1)
    out.index.name = "timestamp"
    return out.loc[r2["output_start"]:r2["output_end"]]
