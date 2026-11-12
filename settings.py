"""Central settings of the generator: every location outside the code, in one place.

Precedence per key: environment variable > settings file > default.
  settings file: $SYNTHRE_SETTINGS, else configs/settings.yaml in the repo if it exists (gitignored;
                 template configs/settings.example.yaml)
  defaults:      the layout of w-lambdablade2 that produced release parks_pv_v4
Values may use {repo}, {data_root} and {work_root}; a leading ~ is expanded. Nothing is created on import.

    import settings
    settings.get("mastr_dir")
"""
import os
import sys

import yaml

REPO = os.path.dirname(os.path.abspath(__file__))

# key -> (environment variable, default)
SPEC = {
    "data_root":         ("DATA_ROOT",            "/mnt/lambda1/nvme1"),              # shared data (inputs, releases, caches)
    "work_root":         ("PV_WORK_ROOT",         "{repo}"),                          # small tables, generated configs, geometry runs (data/, configs/pv/_generated/, results/pv/)
    "selection_dir":     ("PV_SELECTION_DIR",     "~/Work/FL_Contribution/data/mastr"),   # pv_park_selection_<ver>.csv
    "pv_shared_dir":     ("PV_SHARED_DIR",        "{data_root}/synthetic/solar/parks_pv"),  # PV inputs, corrected CAMS, snow cache, angle estimates (all releases)
    "geometry_dir":      ("PV_GEOMETRY_DIR",      "{pv_shared_dir}/geometry"),   # angle estimates per park: <method>/<park_id>/
    "mastr_dir":         ("PV_MASTR_DIR",         "{data_root}/synthetic/solar/mastr"),  # parsed MaStR parquets
    "mastr_date":        ("PV_MASTR_DATE",        "20250625"),                        # date tag of the parsed export (parks_pv_v4); auto = newest present
    "mastr_export_dir":  ("PV_MASTR_EXPORT_DIR",  "{repo}/data/Marktstammdatenregister_20250625_25"),  # unpacked full export (XML)
    "cams_models_dir":   ("PV_CAMS_MODELS_DIR",   "{data_root}/synthetic/cams_bc/models"),  # <tag>/ e.g. xgb_v5
    "icon_sl09_dir":     ("PV_ICON_SL09_DIR",     "{data_root}/nwp_ready_pv/icon-d2/SL/09"),  # park_<id>/ (snow driver of depth_slide)
    "era5_snow_dir":     ("PV_ERA5_SNOW_DIR",     "{data_root}/synthetic/era5_snow/raw"),     # monthly ERA5 snow GRIBs
    "dem_glo30_dir":     ("PV_DEM_GLO30_DIR",     "/mnt/nvme2/synthetic/raw/round2/dem/glo30"),  # Copernicus GLO-30 tiles
    "dem_mosaic_90m":    ("PV_DEM_MOSAIC_90M",    "/mnt/nvme2/synthetic/raw/round2/dem/mosaic_90m.tif"),
    "dem_mosaic_1km":    ("PV_DEM_MOSAIC_1KM",    "/mnt/nvme2/synthetic/raw/round2/dem/mosaic_1km.tif"),   # TPI 75 km of park specs
    "cec_modules_csv":   ("PV_CEC_MODULES_CSV",   "~/Work/FL_Contribution/data/mastr/CEC_Modules.csv"),
    "cec_inverters_csv": ("PV_CEC_INVERTERS_CSV", "~/Work/FL_Contribution/data/mastr/CEC_Inverters.csv"),
    "cams_pm_dir":       ("PV_CAMS_PM_DIR",       "/mnt/nas/cams/sqltocsv"),          # CAMS at DWD stations (benchmarks only)
    "nwp_points_dir":    ("PV_NWP_POINTS_DIR",    "~/Work/NWP-points"),               # ICON-D2 point extraction (stage prepare icon)
    "correction_python": ("PV_CORRECTION_PYTHON", ""),                                # interpreter of the CAMS correction; empty = this one
}
_ORDER = ("data_root", "work_root")           # resolved first, other keys may reference them


def settings_file() -> str:
    """Path of the settings file in use, or '' if none."""
    p = os.environ.get("SYNTHRE_SETTINGS")
    if p:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"SYNTHRE_SETTINGS={p} does not exist")
        return p
    p = os.path.join(REPO, "configs", "settings.yaml")
    return p if os.path.isfile(p) else ""


def load(path: str = None) -> dict:
    """All settings resolved (environment > file > default)."""
    path = settings_file() if path is None else path
    file_vals = {}
    if path:
        file_vals = yaml.safe_load(open(path)) or {}
        unknown = set(file_vals) - set(SPEC)
        if unknown:
            raise KeyError(f"unknown keys in {path}: {sorted(unknown)}")
    out = {"repo": REPO}
    for key in list(_ORDER) + [k for k in SPEC if k not in _ORDER]:
        env, default = SPEC[key]
        raw = os.environ.get(env) or file_vals.get(key) or default
        val = os.path.expanduser(str(raw).format(**out))
        out[key] = val
    if not out["correction_python"]:
        out["correction_python"] = sys.executable
    return out


_S = load()


def get(key: str) -> str:
    return _S[key]


def all_settings() -> dict:
    return dict(_S)


if __name__ == "__main__":
    print(f"# settings file: {settings_file() or '(none, defaults)'}")
    for k, v in _S.items():
        print(f"{k:18s} {v}")
