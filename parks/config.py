"""Per-park configs in the round-2 format (config_<lokation>.yaml).

Same construction as scripts/round2/run_ladder.py / ask33: the base
configs/config_wind.yaml, the experiment's round2 block (PARKS_v1.yaml) and
the park's params. Chain extensions for real parks live in params and are
read only by parks.synth:
  turbines / hub_heights / rated   one entry per turbine GROUP (rated = cap, kW)
  group_ids, group_sizes           t<i> and n turbines per group
  commissioning_dates              per group (round2.commissioning_mode: group)
  era5_cells                       ERA5 grid cell per group (round2.era5_source: db)
Park metadata goes into a separate top-level 'park' block.
"""

import os

import pandas as pd
import yaml

from parks import paths
from parks.layout import primary_cell


def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_config(park: pd.Series, groups: pd.DataFrame, layout: pd.DataFrame,
                 base: dict, chain: dict, synth_base: str = "/mnt/nvme2/synthetic") -> dict:
    """park: selection row; groups/layout: rows of this park only."""
    cfg = {k: (dict(v) if isinstance(v, dict) else v) for k, v in base.items()}
    cfg["data"] = dict(base["data"])
    cfg["data"]["synth_dir"] = synth_base
    cfg["round2"] = dict(chain["round2"])
    params = dict(base["params"])
    params.update({
        "turbines": groups["lib_name"].tolist(),
        "hub_heights": [float(h) for h in groups["hub_height_m"]],
        "rated": [float(r) for r in groups["rated_cap_kw"]],
        "group_ids": groups["group_id"].tolist(),
        "group_sizes": [int(n) for n in groups["n_turbines"]],
        "commissioning_dates": groups["commissioning_date"].tolist(),
        "era5_cells": [int(c) for c in groups["era5_cell_id"]],
        "apply_ageing": True,
        "noise": 0.0,
        "random_seed": 42,
    })
    params.pop("commissioning_date", None)
    cfg["params"] = params
    cfg["park"] = {
        "lokation": park["lokation"], "name": str(park["name"]), "client": park["client"],
        "cls": park["cls"], "n_turbines": int(groups["n_turbines"].sum()),
        "capacity_kw": float(layout["rated_kw"].sum()),
        "capacity_cap_kw": float(layout["rated_cap_kw"].sum()),
        "latitude": float(layout["latitude"].mean()), "longitude": float(layout["longitude"].mean()),
        "primary_era5_cell": primary_cell(groups),
        "replaced_v1_park": (str(park["replaced"]) if pd.notna(park.get("replaced")) and park.get("replaced")
                             else None),
    }
    return cfg


def write_configs(selection: pd.DataFrame, groups: pd.DataFrame, layout: pd.DataFrame,
                  out_dir: str = paths.CONFIG_DIR) -> list:
    base, chain = load_yaml(paths.BASE_YAML), load_yaml(paths.CHAIN_YAML)
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for _, park in selection.iterrows():
        lk = park["lokation"]
        cfg = build_config(park, groups[groups["park_id"] == lk], layout[layout["park_id"] == lk], base, chain)
        path = os.path.join(out_dir, f"config_{lk}.yaml")
        with open(path, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
        written.append(path)
    return written
