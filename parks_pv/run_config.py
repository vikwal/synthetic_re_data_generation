"""Run config of generate_pv.py: one YAML per release. Every key has a default; unknown keys are an error.

    release:   name of the dataset (data/<name>/, ${DATA_ROOT}/synthetic/solar/<name><variant>/)
    variant:   suffix of the release and chain work dir (e.g. _30min)
    selection: park list CSV (MaStR-derived selection rows); placeholders {selection_dir}, {work_root}, {repo}
    parks:     park spec file instead of selection (parks_pv.park_spec: hand-typed, fictitious or MaStR parks); its
               selection rows are written to data/<release>/selection_spec.csv on every run
    overrides: optional dashboard overrides JSON applied to the selection (utils/overrides.py)
    period:    {start, end} release window [start, end); a park starts when its last MaStR unit is in operation
    step:      chain step, 30min or 1h
    seed:      seed of the park geometry and inverter draws
    irradiance.correction: cams_bc model tag of the corrected CAMS (e.g. xgb_v5)
    geometry.mode: estimate | register | manual
                 estimate: measured angles of the angle estimation (parks_pv.geometry_estimate), taken from the store
                           <geometry_dir>/<method>/<park_id>/ and estimated first where missing; register fallback
                 register: register geometry of the baseline (MaStR classes and era, terrain-following), no estimation
                 manual:   sub-arrays from geometry.manual (CSV park_id, tilt, azimuth, share [, array_id]; tilt against
                           the horizontal, true azimuth), e.g. for a fictitious park
    geometry.method: version of the angle estimation in the store (e.g. v4)
    geometry.manual: CSV of mode manual
    snow.model: off | depth_linear | depth_slide
    aging.cohorts: arrays split into age cohorts of the MaStR units (else park-level era and aging)
    horizon:   true = PVGIS horizon (DEM where PVGIS has none); false = no horizon shading
    prepare:   steps of stage prepare (site, cams, correct, icon) and the ICON settings of NWP-points
    stages:    default stage list of a run
    workers, gpu: parallel parks in stage generate, GPU of the CAMS correction
"""
import copy
import os

import yaml

import settings

STAGES = ("prepare", "baseline", "geometry", "sites", "configs", "generate", "assemble", "tables")
SNOW_MODELS = ("off", "depth_linear", "depth_slide")
GEOMETRY_MODES = ("estimate", "register", "manual")

DEFAULTS = {
    "release": None,
    "variant": "",
    "selection": None,
    "parks": None,
    "overrides": None,
    "period": {"start": "2023-07-24", "end": "2026-08-01"},
    "step": "30min",
    "seed": 42,
    "irradiance": {"correction": "xgb_v5"},
    "geometry": {"mode": "estimate", "method": "v4", "manual": None},
    "snow": {"model": "depth_slide"},
    "aging": {"cohorts": True},
    "horizon": True,
    "prepare": {"steps": ["site", "cams", "correct", "icon"],
                "icon": {"name": None, "previous": "pv_v2", "sources": "pv_v1 pv_v2_add", "hours": "09"}},
    "stages": list(STAGES),
    "workers": 8,
    "gpu": 0,
}


def _merge(base: dict, over: dict, where: str = "") -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if k not in base:
            raise KeyError(f"unknown run config key {where}{k}")
        out[k] = _merge(base[k], v, f"{where}{k}.") if isinstance(base[k], dict) and isinstance(v, dict) and base[k] else v
    return out


def _path(p: str):
    if not p:
        return None
    s = settings.all_settings()
    return os.path.expanduser(str(p).format(selection_dir=s["selection_dir"], work_root=s["work_root"], repo=s["repo"],
                                            data_root=s["data_root"]))


def load(path: str) -> dict:
    """Run config with defaults filled in and paths resolved."""
    raw = yaml.safe_load(open(path)) or {}
    rc = _merge(DEFAULTS, raw)
    if not rc["release"]:
        raise ValueError(f"{path}: release name missing")
    if rc["snow"]["model"] not in SNOW_MODELS:
        raise ValueError(f"{path}: snow.model must be one of {SNOW_MODELS}")
    if rc["geometry"]["mode"] not in GEOMETRY_MODES:
        raise ValueError(f"{path}: geometry.mode must be one of {GEOMETRY_MODES}")
    if rc["geometry"]["mode"] == "manual" and not rc["geometry"]["manual"]:
        raise ValueError(f"{path}: geometry.mode manual needs geometry.manual (CSV)")
    if rc["step"] not in ("30min", "1h"):
        raise ValueError(f"{path}: step must be 30min or 1h")
    bad = set(rc["stages"]) - set(STAGES)
    if bad:
        raise ValueError(f"{path}: unknown stages {sorted(bad)}")
    if rc["selection"] and rc["parks"]:
        raise ValueError(f"{path}: selection and parks exclude each other")
    rc["selection"], rc["overrides"], rc["parks"] = _path(rc["selection"]), _path(rc["overrides"]), _path(rc["parks"])
    rc["geometry"]["manual"] = _path(rc["geometry"]["manual"])
    rc["config_path"] = os.path.abspath(path)
    return rc


def apply(rc: dict):
    """Point parks_pv.paths at the release of this run config."""
    from parks_pv import paths
    paths.configure(rc["release"], rc["variant"], rc["selection"])
    paths.ensure_dirs()
    if rc["parks"]:                                               # park spec -> selection rows of this release
        from parks_pv import park_spec
        sel = park_spec.write_selection(rc["parks"], os.path.join(paths.DATA_DIR, "selection_spec.csv"))
        paths.configure(rc["release"], rc["variant"], sel)
