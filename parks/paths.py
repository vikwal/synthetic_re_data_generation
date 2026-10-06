"""File locations of the parks_v1 product (inputs, configs, scratch, release)."""

import os

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
VERSION = "parks_v1"
EXPERIMENT_ID = "PARKS_v1"

FL_DIR = os.path.expanduser("~/Work/FL_Contribution")
MASTR_DIR = os.path.join(FL_DIR, "data", "mastr")
SELECTION_CSV = os.path.join(MASTR_DIR, "wind_park_selection_v1_1.csv")
TURBINES_CSV = os.path.join(MASTR_DIR, "wind_turbines_matched.csv")
OSM_CORRECTIONS_CSV = os.path.join(MASTR_DIR, "osm_corrections_v1_1.csv")

# generator-side artefacts (data/ is gitignored, configs/_generated/parks_v1 too)
DATA_DIR = os.path.join(REPO, "data", VERSION)
LAYOUT_CSV = os.path.join(DATA_DIR, "park_layouts.csv")
GROUPS_CSV = os.path.join(DATA_DIR, "park_groups.csv")
SPEC_OVERRIDES_CSV = os.path.join(DATA_DIR, "turbine_specs_overrides.csv")
TOPO_CSV = os.path.join(DATA_DIR, "topo_features.csv")
CONFIG_DIR = os.path.join(REPO, "configs", "round2", "_generated", VERSION)
CHAIN_YAML = os.path.join(REPO, "configs", "round2", "PARKS_v1.yaml")
BASE_YAML = os.path.join(REPO, "configs", "config_wind.yaml")

# scratch run dir on the local NVMe of l2, release dir on l1 (nasuser group)
RUN_DIR = os.path.join("/mnt/nvme2/synthetic/wind/round2", EXPERIMENT_ID)
DATA_ROOT = os.environ.get("DATA_ROOT", "/mnt/lambda1/nvme1")
RELEASE_DIR = os.path.join(DATA_ROOT, "synthetic", "wind", VERSION)


def config_path(lokation: str) -> str:
    return os.path.join(CONFIG_DIR, f"config_{lokation}.yaml")
