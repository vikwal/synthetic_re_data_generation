"""Paths and constants. Data root via DATA_ROOT from ~/.bashrc (l2: /mnt/lambda1/nvme1, l1: /mnt/nvme1),
SOLAR_DATEN_WURZEL (NWP/ERA5 convention) as fallback. The station frames and CAMS caches are products
of the NWP/ERA5 and NWP/CAMS pipelines; this package only reads them."""
import os

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ROOT = os.environ.get("DATA_ROOT", os.environ.get("SOLAR_DATEN_WURZEL", "/mnt/lambda1/nvme1"))
HOME = os.path.expanduser("~")

PROCESSED = f"{ROOT}/synthetic/solar_bc/processed_30min"       # solar_bc station frames, 30 min
RING_DIR = f"{ROOT}/synthetic/cams_rad_cache/ring15min"        # CAMS ring points, 15 min
RING_POINTS = f"{ROOT}/synthetic/cams_rad_cache/ringpunkte_alle.csv"
IMPUTED = f"{ROOT}/synthetic/interpol/solar"                   # imputed station product (observed + gaps filled)
DATASET = f"{ROOT}/synthetic/cams_bc/dataset_30min"            # feature frames built here
STATIONS_LIST = f"{HOME}/Work/NWP/DWD/data/stations_list.csv"
RESULTS = os.path.join(REPO, "results", "cams_bc")
SPLITS = os.path.join(REPO, "data", "cams_bc", "splits.csv")

STEP = "30min"
RADII = (3.0, 8.0)
AZIMUTHS = ("N", "O", "S", "W")        # as in ringpunkte_alle.csv (O = east)
SOLAR_CONSTANT = 1361.0
TEST_FRACTION = 0.2
N_FOLDS = 3
SEED = 20261013
