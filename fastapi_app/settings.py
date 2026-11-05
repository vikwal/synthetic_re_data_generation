"""Central configuration (paths, limits). Override with environment variables."""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STREAMLIT_DIR = PROJECT_ROOT / "streamlit"            # config.yaml + data/ live here
DATA_DIR = STREAMLIT_DIR / "data"
CONFIG_PATH = STREAMLIT_DIR / "config.yaml"
GRID_DIR = Path(os.environ.get("SYNTH_RE_ERA5_DIR") or DATA_DIR / "parquet" / "grid_points")   # ERA5 weather files
# Optional local NWP archive, one folder per model (not offered in the form)
NWP_DIRS = {
    "icon_d2": DATA_DIR / "parquet" / "grid_points_icon_d2",
    "ecmwf": DATA_DIR / "parquet" / "grid_points_ecmwf",
}
SOLAR_DIR = DATA_DIR / "solar"
# Optional ERA5 solar inputs per station (Station_<id>.parquet): irradiance folder + a folder with temperature and wind.
# When SYNTH_RE_ERA5_SOLAR_DIR is set, PV power is generated from these instead of the measured station csv files.
# SYNTH_RE_SOLAR_SOURCE=measured switches back to the measured station csv files even when the ERA5 folders are set.
_ERA5_SOLAR_ON = os.environ.get("SYNTH_RE_SOLAR_SOURCE", "era5").strip().lower() != "measured"
SOLAR_ERA5_DIR = Path(os.environ["SYNTH_RE_ERA5_SOLAR_DIR"]) if os.environ.get("SYNTH_RE_ERA5_SOLAR_DIR") and _ERA5_SOLAR_ON else None
MET_ERA5_DIR = Path(os.environ["SYNTH_RE_ERA5_MET_DIR"]) if os.environ.get("SYNTH_RE_ERA5_MET_DIR") else None

# Where scenarios and generated results are persisted (survives restarts).
STORAGE_DIR = Path(os.environ.get("SYNTH_RE_STORAGE_DIR", PROJECT_ROOT / "fastapi_app" / "storage"))

MAX_RESULTS_IN_MEMORY = int(os.environ.get("SYNTH_RE_MAX_RESULTS_IN_MEMORY", "8"))
JOB_WORKERS = int(os.environ.get("SYNTH_RE_JOB_WORKERS", "2"))
MAX_TURBINES = 200
