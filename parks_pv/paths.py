"""Locations of a release, derived from the central settings (settings.py: environment > settings file > defaults)
and the run config (parks_pv.run_config -> configure()). Nothing is created on import; writers call ensure_dirs().

Without a run config the release comes from PV_RELEASE_NAME / PV_RELEASE_VARIANT / PV_SELECTION (analysis scripts)."""
import os

import settings

REPO = settings.REPO
ROOT = settings.get("data_root")
WORK_ROOT = settings.get("work_root")            # data/, configs/pv/_generated/, results/pv/ (default: the repo)
SHARED_DIR = settings.get("pv_shared_dir")       # inputs, corrected CAMS and snow cache, shared by all releases
INPUT_DIR = f"{SHARED_DIR}/inputs"               # cams/, cams_ring/, osm/, geodata/, cams_points.csv
CORR_DIR = f"{SHARED_DIR}/work/cams_corrected"   # corrected CAMS (30 min and hourly)
SNOW_CACHE_DIR = f"{SHARED_DIR}/work/era5_snow"  # ERA5 snow point cache
MASTR_DIR = settings.get("mastr_dir")
SNOW_DIR = settings.get("era5_snow_dir")
ICON_SL09_DIR = settings.get("icon_sl09_dir")    # ICON-D2 single level, 09 UTC run, park_<id>/ (snow driver of depth_slide)
CAMS_MODELS = settings.get("cams_models_dir")

# release-specific, set by configure()
NAME = VARIANT = SELECTION = BASE_DIR = RELEASE_DIR = WORK_DIR = DATA_DIR = CONFIG_DIR = None


def configure(name: str, variant: str = "", selection: str = None):
    """Point the release locations at one release: name (dataset name, e.g. parks_pv_v4), variant (suffix of the
    release and chain work dir, e.g. _30min), selection CSV (default <selection_dir>/pv_park_selection_<vN>.csv)."""
    global NAME, VARIANT, SELECTION, BASE_DIR, RELEASE_DIR, WORK_DIR, DATA_DIR, CONFIG_DIR
    NAME, VARIANT = name, variant or ""
    SELECTION = selection or os.path.join(settings.get("selection_dir"), f"pv_park_selection_{name.rsplit('_', 1)[-1]}.csv")
    BASE_DIR = f"{ROOT}/synthetic/solar/{NAME}"
    RELEASE_DIR = BASE_DIR + VARIANT
    WORK_DIR = f"{RELEASE_DIR}/work"             # drivers and chain outputs of this variant
    DATA_DIR = os.path.join(WORK_ROOT, "data", NAME)   # park tables of the release (baseline, sites, arrays, params)
    CONFIG_DIR = os.path.join(WORK_ROOT, "configs", "pv", "_generated", NAME)


# default parks_pv_v1: the analysis scripts of the former geometry runs read their tables from data/parks_pv_v1
configure(os.environ.get("PV_RELEASE_NAME", "parks_pv_v1"), os.environ.get("PV_RELEASE_VARIANT", ""),
          os.environ.get("PV_SELECTION") or None)


def mastr_file(kind: str = "units") -> str:
    """Parsed MaStR parquet (units | eeg | parks) of the configured date; parses or downloads the register if missing
    (pv.mastr_source.ensure)."""
    from pv import mastr_source
    return mastr_source.parquet_paths(mastr_source.ensure(mastr_dir=MASTR_DIR), MASTR_DIR)[kind]


def ensure_dirs():
    """Create the write targets of this release (call from entry points that write, not on import)."""
    for d in (DATA_DIR,):
        os.makedirs(d, exist_ok=True)
