"""Locations. DATA_ROOT from the environment (l2 /mnt/lambda1/nvme1, l1 /mnt/nvme1); nothing under $HOME."""
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.environ.get("DATA_ROOT", "/mnt/lambda1/nvme1")
NAME = "parks_pv_v1"
SELECTION = os.path.expanduser("~/Work/FL_Contribution/data/mastr/pv_park_selection_v1.csv")
RELEASE_DIR = f"{ROOT}/synthetic/solar/{NAME}"
INPUT_DIR = f"{RELEASE_DIR}/inputs"              # cams/, cams_ring/, osm/, cams_points.csv
WORK_DIR = f"{RELEASE_DIR}/work"                 # drivers, chain outputs, corrected CAMS
MASTR_DIR = f"{ROOT}/synthetic/solar/mastr"
SNOW_DIR = f"{ROOT}/synthetic/era5_snow/raw"
CAMS_MODELS = f"{ROOT}/synthetic/cams_bc/models"
DATA_DIR = os.path.join(REPO, "data", NAME)      # small tables kept with the code (public register data)
CONFIG_DIR = os.path.join(REPO, "configs", "pv", "_generated", NAME)
for d in (DATA_DIR,):
    os.makedirs(d, exist_ok=True)
