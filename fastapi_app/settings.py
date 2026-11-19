"""Central configuration (paths, limits). Override with environment variables."""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STREAMLIT_DIR = PROJECT_ROOT / "streamlit"            # config.yaml + data/ live here
DATA_DIR = STREAMLIT_DIR / "data"
CONFIG_PATH = Path(os.environ.get("SYNTH_RE_CONFIG") or STREAMLIT_DIR / "config.yaml")
GRID_DIR = Path(os.environ.get("SYNTH_RE_ERA5_DIR") or DATA_DIR / "parquet" / "grid_points")   # ERA5 weather files
# Optional: read the ERA5 grid points from PostgreSQL instead of GRID_DIR (see utils/import_era5_to_postgres.py).
# Example: postgresql://era5_user:PASSWORD@localhost:5432/era5. Unset = parquet files, as before.
ERA5_DB_URL = os.environ.get("SYNTH_RE_ERA5_DB_URL") or None
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

# Login is always on: every request needs a signed-in user and each user gets a private storage folder.
# (SYNTH_RE_AUTH=0 switches it off, e.g. for quick experiments; no variable is needed otherwise.)
AUTH_ENABLED = os.environ.get("SYNTH_RE_AUTH", "1").strip().lower() in ("1", "true", "yes", "on")
SESSION_HOURS = float(os.environ.get("SYNTH_RE_SESSION_HOURS", "12"))

# Sub-path the app is served under behind a reverse proxy, e.g. SYNTH_RE_BASE_PATH=/synthetic-dashboard (the proxy strips it before
# forwarding). Empty = served at the root, as locally.
_base = os.environ.get("SYNTH_RE_BASE_PATH", "").strip().strip("/")
BASE_PATH = f"/{_base}" if _base else ""

# External weather service for solar (placeholder until a real one is connected). Empty URL = built-in stand-in.
# Every location is requested once for this fixed period and then cached per user.
PV_SERVICE_URL = os.environ.get("SYNTH_RE_PV_SERVICE_URL", "").strip()
PV_SERVICE_TOKEN = os.environ.get("SYNTH_RE_PV_SERVICE_TOKEN", "").strip()
PV_SERVICE_START = os.environ.get("SYNTH_RE_PV_SERVICE_START", "2023-01-01")
PV_SERVICE_END = os.environ.get("SYNTH_RE_PV_SERVICE_END", "2024-12-31")
PV_SERVICE_TIMEOUT = float(os.environ.get("SYNTH_RE_PV_SERVICE_TIMEOUT", "120"))

MAX_RESULTS_IN_MEMORY = int(os.environ.get("SYNTH_RE_MAX_RESULTS_IN_MEMORY", "8"))
JOB_WORKERS = int(os.environ.get("SYNTH_RE_JOB_WORKERS", "2"))
MAX_TURBINES = 200
