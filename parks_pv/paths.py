"""Locations. DATA_ROOT from the environment (l2 /mnt/lambda1/nvme1, l1 /mnt/nvme1); nothing under $HOME."""
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.environ.get("DATA_ROOT", "/mnt/lambda1/nvme1")
NAME = os.environ.get("PV_RELEASE_NAME", "parks_pv_v1")   # parks_pv_v2 (2026-10-22): same parks and drivers, geometry from geodata
VARIANT = os.environ.get("PV_RELEASE_VARIANT", "")      # e.g. "_30min": own release and chain work dir, shared inputs
SELECTION = os.path.expanduser("~/Work/FL_Contribution/data/mastr/pv_park_selection_v1.csv")
SHARED_DIR = f"{ROOT}/synthetic/solar/parks_pv_v1"      # inputs, corrected CAMS and snow cache are shared by all releases
BASE_DIR = f"{ROOT}/synthetic/solar/{NAME}"
RELEASE_DIR = BASE_DIR + VARIANT
INPUT_DIR = f"{SHARED_DIR}/inputs"               # cams/, cams_ring/, osm/, geodata/, cams_points.csv (shared by all variants)
WORK_DIR = f"{RELEASE_DIR}/work"                 # drivers and chain outputs of this variant
CORR_DIR = f"{SHARED_DIR}/work/cams_corrected"   # corrected CAMS (30 min and hourly), shared
SNOW_CACHE_DIR = f"{SHARED_DIR}/work/era5_snow"  # ERA5 snow point cache, shared
MASTR_DIR = f"{ROOT}/synthetic/solar/mastr"
SNOW_DIR = f"{ROOT}/synthetic/era5_snow/raw"
ICON_SL09_DIR = f"{ROOT}/nwp_ready_pv/icon-d2/SL/09"   # ICON-D2 single level, 09 UTC run, park_<id>/ (snow driver of depth_slide)
CAMS_MODELS = f"{ROOT}/synthetic/cams_bc/models"
DATA_DIR = os.path.join(REPO, "data", NAME)      # small tables kept with the code (public register data)
CONFIG_DIR = os.path.join(REPO, "configs", "pv", "_generated", NAME)
for d in (DATA_DIR,):
    os.makedirs(d, exist_ok=True)
