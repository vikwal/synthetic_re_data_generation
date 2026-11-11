"""Park configuration schema and loaders.

A park config (YAML or dict):
  park_id: str
  latitude, longitude, altitude: float
  kwp_dc: float                 DC rating of all arrays, kWp
  dc_ac: float                  DC/AC ratio -> AC rating = kwp_dc / dc_ac
  arrays: [{name, tilt, azimuth, share, gcr}]      shares sum to 1; optional per array (age cohorts, 2026-11-05):
           gamma_pdc, era, commissioning_date - override module.gamma_pdc / aging.era / aging.commissioning_date
  module:  {model: pvwatts|cec, name: <library name or null>, gamma_pdc: -0.0035, year: int}
  inverter: {model: pvwatts|sandia, name: <library name or null>, eta_inv_nom: 0.96, unit_kw: null}
  temperature: {model: faiman|pvsyst, u0: 25, u1: 6.84}
  losses: {dc: 0.03, ac: 0.01, availability: 0.99}     constant PVsyst-style factors
  aging: {enabled: bool, commissioning_date: 'YYYY-MM-DD', era: <pv.aging.ERAS key or null>}
  soiling: {enabled: bool, model: hsu, cleaning_threshold_mm: 1.0, pm_station: <id>}
  shading: {horizon: bool, inter_row: bool}
  transposition: {model: haydavies, iam: physical, albedo: 0.2 | 'era5'}
  random_seed: int
"""

import copy
import os

import yaml

from pv.aging import ERAS

DEFAULTS = {
    "altitude": 0.0,
    "dc_ac": 1.0,
    "module": {"model": "pvwatts", "name": None, "gamma_pdc": -0.0035, "year": 2015},
    "inverter": {"model": "pvwatts", "name": None, "eta_inv_nom": 0.96, "eta_inv_ref": 0.9637, "unit_kw": None},
    "temperature": {"model": "faiman", "u0": 25.0, "u1": 6.84},
    "losses": {"dc": 0.0, "ac": 0.0, "availability": 1.0},
    "aging": {"enabled": False, "commissioning_date": None, "era": None},
    "soiling": {"enabled": False, "model": "hsu", "cleaning_threshold_mm": 1.0, "pm_station": None},
    "shading": {"horizon": False, "inter_row": False},
    "transposition": {"model": "haydavies", "iam": "physical", "albedo": 0.2},
    "random_seed": 42,
}


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def normalise(cfg: dict) -> dict:
    """Fill defaults, validate shares and required keys."""
    c = _merge(DEFAULTS, cfg)
    for k in ("park_id", "latitude", "longitude", "kwp_dc", "arrays"):
        if k not in c or c[k] is None:
            raise ValueError(f"park config lacks '{k}'")
    if not c["arrays"]:
        raise ValueError("park config has no arrays")
    tot = sum(float(a.get("share", 1.0)) for a in c["arrays"])
    for j, a in enumerate(c["arrays"]):
        a.setdefault("name", f"a{j}")
        a["share"] = float(a.get("share", 1.0)) / tot
        a.setdefault("gcr", None)
        for k in ("tilt", "azimuth"):
            if k not in a:
                raise ValueError(f"array {a['name']} lacks '{k}'")
        if a.get("era") is not None and a["era"] not in ERAS:
            raise ValueError(f"array {a['name']}: unknown era {a['era']!r}")
        if "gamma_pdc" in a and not -0.01 < float(a["gamma_pdc"]) < 0.0:
            raise ValueError(f"array {a['name']}: gamma_pdc {a['gamma_pdc']} outside (-0.01, 0)")
    if c["aging"]["enabled"] and not c["aging"]["commissioning_date"]:
        raise ValueError("aging enabled but no commissioning_date")
    return c


def load_park(path: str) -> dict:
    with open(path) as f:
        return normalise(yaml.safe_load(f))


def from_legacy(old: dict, park_id: str, latitude: float, longitude: float, altitude: float = 0.0) -> dict:
    """Convert a configs/real_pv_parks/*.yaml or configs/config_pv.yaml 'pv_params' block.
    The legacy files nest tilt_az_list one level too deep ([[[25, 180]]]) and carry the DC
    rating in watts; both are repaired here."""
    p = old.get("pv_params", old)
    pairs = p["tilt_az_list"]
    while pairs and isinstance(pairs[0], list) and pairs[0] and isinstance(pairs[0][0], list):
        pairs = [x for sub in pairs for x in sub]
    arrays = [{"name": f"a{j}", "tilt": float(t), "azimuth": float(a), "share": 1.0 / len(pairs)} for j, (t, a) in enumerate(pairs)]
    cfg = {
        "park_id": park_id, "latitude": latitude, "longitude": longitude, "altitude": altitude,
        "kwp_dc": float(p["dc_rating_watts"]) / 1000.0, "dc_ac": float(p.get("dc_ac_ratio", 1.0)),
        "arrays": arrays,
        "module": {"model": "pvwatts", "gamma_pdc": float(p.get("gamma_pdc", -0.0035))},
        "inverter": {"model": "pvwatts", "eta_inv_nom": float(p.get("mean_eta_inv_nom", 0.96)), "eta_inv_ref": float(p.get("eta_inv_ref", 0.9637))},
        "temperature": {"model": "faiman", "u0": float(p.get("faiman_u0", 25.0)), "u1": float(p.get("faiman_u1", 6.84))},
        "aging": {"enabled": bool(p.get("apply_ageing", False)), "commissioning_date": p.get("commissioning_date")},
        "soiling": {"enabled": bool(p.get("apply_soiling", False))},
        "transposition": {"albedo": float(p.get("albedo", 0.2))},
        "random_seed": int(p.get("random_seed", 42)),
    }
    return normalise(cfg)


def station_grid(station_id: str, latitude: float, longitude: float, altitude: float, tilt: float, azimuth: float,
                 kwp_dc: float = 10.0, **over) -> dict:
    """One 10-kW reference array (the legacy station product, one config per (tilt, azimuth))."""
    cfg = {"park_id": f"{station_id}_t{int(tilt)}_a{int(azimuth)}", "latitude": latitude, "longitude": longitude,
           "altitude": altitude, "kwp_dc": kwp_dc, "arrays": [{"name": "a0", "tilt": tilt, "azimuth": azimuth, "share": 1.0}]}
    return normalise(_merge(cfg, over))


def dump(cfg: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)
