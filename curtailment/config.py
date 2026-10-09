"""Curtailment config block: defaults, deep merge and validation.

All hyperparameters live in config['curtailment']; get_curtailment_params()
merges the user block over CURTAILMENT_DEFAULTS (dicts recursively, lists and
scalars replaced) and validates the result. enabled=False (default) leaves
every output untouched.

Sources of the defaults (rows of FL_Contribution/literature/curtailment/
calibration_numbers.md): disturbance [C36-C38], first setpoint [C32], event
process a/b [C27], participation beta [C31], durations [C29], release segment
[C34a], area targets [C53], cohorts [D7, D7a], exposed response [D4], price
bands [D3], bat rules [E4, E15].
"""

import copy
import datetime as dt
import os
import re

AREAS = ("A1_SH", "A2_NI_NW", "A3_NI_O_ST", "A4_NO", "A5_MITTE_W", "A6_SUED")
LAYERS = ("environment", "market", "grid")
BANDS = ("-10..0", "-20..-10", "-50..-20", "-100..-50", "<-100")


def _area(p0, a=None, b=None, beta=None, dur=None, rel=None, target=None) -> dict:
    if a is None:
        return {"p0": p0}
    return {"p0": p0, "a": a, "b": b, "beta": list(beta), "dur": list(dur),
            "release": {"p": rel[0], "p30": rel[1], "dur": list(rel[2])}, "target_pct": dict(target)}


CURTAILMENT_DEFAULTS = {
    "enabled": False,
    "dataset": "parks_v1_curt",            # release sub-directory
    "seed": 20261009,
    "step_minutes": 15,                    # internal grid; output hourly (mean of 4 quarter-hours)
    "layers": {"environment": True, "market": True, "grid": True},
    "inputs": {
        "area_table": "~/Work/FL_Contribution/pipeline/curtailment/park_grid_areas.csv",
        "states_geojson": "~/Work/FL_Contribution/data/curtailment/raw/geo/bundeslaender_mittel.geo.json",
        "mastr_fleet": "~/Work/FL_Contribution/data/mastr/wind_turbines_matched.csv",
        "driver_cache": "data/curtailment/drivers",          # gitignored
        "calibration_file": "data/curtailment/calibration_{dataset}.json",
    },
    "grid": {
        "target_scale": 1.0,               # parks_v1_curt_x4: 4.0
        "node_radius_km": 10.0,
        "disturbance": {"phi": 0.4, "sigma_z": 1.0, "rho_z": 0.4},
        "setpoint_first": {0.0: 0.78, 0.3: 0.12, 0.6: 0.10},
        "max_duration_h": 168.0,           # cap of one segment (lognormal tail; documented deviation)
        "areas": {
            # a, b: start rate log lambda = a + b * CF_DA [1/h, unit]; beta: participation;
            # dur: LogNormal(mu, sigma) in ln h; release: appended 30/60 % segment after s0 = 0
            "A1_SH": _area(0.1, -0.38, 3.43, (1.22, 270), (1.63, 1.48), (0.087, 0.65, (-1.15, 1.26)),
                           {2023: 5.5, 2024: 4.3, 2025: 4.2, 2026: 4.2}),
            "A2_NI_NW": _area(0.2, -1.63, 4.28, (0.71, 143), (1.20, 1.46), (0.104, 0.50, (-1.26, 0.90)),
                              {2023: 4.1, 2024: 3.7, 2025: 3.3, 2026: 3.3}),
            "A3_NI_O_ST": _area(0.2, -2.30, 5.19, (0.53, 86), (1.43, 1.11), (0.110, 0.39, (-1.07, 1.60)),
                                {2023: 4.1, 2024: 3.7, 2025: 3.3, 2026: 3.3}),
            "A4_NO": _area(0.2, -2.30, 5.19, (0.53, 86), (1.43, 1.11), (0.110, 0.39, (-1.07, 1.60)),
                           {2023: 4.8, 2024: 4.4, 2025: 4.2, 2026: 4.2}),
            "A5_MITTE_W": _area(0.5, -5.06, 6.65, (0.32, 28), (0.99, 1.43), (0.048, 0.58, (-1.03, 1.28)),
                                {2023: 1.3, 2024: 1.2, 2025: 1.1, 2026: 1.1}),
            "A6_SUED": _area(1.0),         # no grid layer
        },
        "calibration": {"n_virtual_nodes": 2000, "rel_tol": 0.02, "max_iter": 40},
    },
    "market": {
        "post_eeg": "unsubsidized",        # p* = 0 from 1 Jan of (commissioning year + 21); 'feed_in': never
        "cohorts": [                       # strike = anzulegender Wert [ct/kWh]; rule_h: rule length [h]
            {"from": "2000-01-01", "to": "2011-12-31", "strike_ct": 8.5, "rule_h": None},
            {"from": "2012-01-01", "to": "2015-12-31", "strike_ct": 8.9, "rule_h": None},
            {"from": "2016-01-01", "to": "2020-12-31", "strike_ct": 7.6, "rule_h": 6, "min_unit_kw": 3000},
            {"from": "2021-01-01", "to": "2022-12-31", "strike_ct": 7.55, "rule_h": 4, "min_unit_kw": 500},
            {"from": "2023-01-01", "to": "2025-02-24", "strike_ct": 7.22, "rule_h": 3, "min_unit_kw": 400},
            {"from": "2025-02-25", "to": "2100-01-01", "strike_ct": 8.93, "rule_h": 0.25, "min_unit_kw": 0},
        ],
        "response": {
            "theta0": -1.5, "theta1": 0.8, "tau": 1.5, "calibrate": True,
            "target_exposed_response": {2023: 0.32, 2024: 0.26, 2025: 0.27, 2026: 0.32},
            "target_band_rate": {"-10..0": 0.14, "-20..-10": 0.22, "-50..-20": 0.36,
                                 "-100..-50": 0.52, "<-100": 0.58},
        },
    },
    "environment": {
        "bat": {
            "min_commissioning_year": 2010,
            "levels": {                    # v_max at hub wind, t_min_c at hub temperature
                "standard": {"p": 0.6, "start": "04-01", "end": "10-31",
                             "before_sunset_h": {"04-08": 1, "09-10": 3}, "v_max": 6.0, "t_min_c": 10.0},
                "mild": {"p": 0.2, "start": "06-01", "end": "09-15",
                         "before_sunset_h": {"all": 0}, "v_max": 5.5, "t_min_c": 16.0},
                "strict": {"p": 0.2, "start": "04-12", "end": "10-13",
                           "before_sunset_h": {"all": 0}, "v_max": 10.2, "t_min_c": 13.0},
            },
        },
    },
}


def area_key(name: str) -> str:
    """'A3 NI-O/ST' -> 'A3_NI_O_ST' (area names of park_grid_areas.csv -> config keys)."""
    return re.sub(r"[^A-Z0-9]+", "_", str(name).upper()).strip("_")


def deep_merge(base: dict, user: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (user or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _as_date(x) -> dt.date:
    if isinstance(x, dt.datetime):
        return x.date()
    if isinstance(x, dt.date):
        return x
    return dt.date.fromisoformat(str(x))


def _normalise(c: dict) -> dict:
    """Canonical key types (YAML may give str/float keys) and parsed dates."""
    g = c["grid"]
    g["setpoint_first"] = {float(k): float(v) for k, v in g["setpoint_first"].items()}
    g["areas"] = {area_key(k): v for k, v in g["areas"].items()}
    for a in g["areas"].values():
        if "target_pct" in a:
            a["target_pct"] = {int(y): float(v) for y, v in a["target_pct"].items()}
    m = c["market"]
    m["cohorts"] = [{**co, "from": _as_date(co["from"]), "to": _as_date(co["to"])} for co in m["cohorts"]]
    r = m["response"]
    r["target_exposed_response"] = {int(y): float(v) for y, v in r["target_exposed_response"].items()}
    r["target_band_rate"] = {str(k): float(v) for k, v in r["target_band_rate"].items()}
    return c


def validate(c: dict) -> None:
    def need(cond, msg):
        if not cond:
            raise ValueError(f"curtailment config: {msg}")

    need(isinstance(c["enabled"], bool), "enabled must be true/false")
    need(int(c["step_minutes"]) == 15, "only step_minutes = 15 is implemented")
    need(set(c["layers"]) <= set(LAYERS), f"layers must be a subset of {LAYERS}")
    g = c["grid"]
    need(float(g["target_scale"]) > 0, "grid.target_scale must be > 0")
    need(float(g["node_radius_km"]) >= 0, "grid.node_radius_km must be >= 0")
    need(float(g["max_duration_h"]) >= 0.25, "grid.max_duration_h must be >= 0.25")
    d = g["disturbance"]
    need(0 <= d["phi"] < 1 and d["sigma_z"] >= 0 and 0 <= d["rho_z"] <= 1, "grid.disturbance out of range")
    sp = g["setpoint_first"]
    need(abs(sum(sp.values()) - 1) < 1e-9 and all(0 <= s < 1 for s in sp), "setpoint_first: levels in [0,1), p sum 1")
    for name, a in g["areas"].items():
        need(name in AREAS, f"unknown area {name} (expected {AREAS})")
        need(0 <= a["p0"] <= 1, f"{name}: p0 in [0, 1]")
        if a["p0"] < 1:
            for k in ("a", "b", "beta", "dur", "release", "target_pct"):
                need(k in a, f"{name}: '{k}' missing (needed unless p0 = 1)")
            need(len(a["beta"]) == 2 and min(a["beta"]) > 0, f"{name}: beta = [alpha, beta] > 0")
            need(len(a["dur"]) == 2 and a["dur"][1] > 0, f"{name}: dur = [mu, sigma > 0]")
            rel = a["release"]
            need(0 <= rel["p"] <= 1 and 0 <= rel["p30"] <= 1 and rel["dur"][1] > 0, f"{name}: release out of range")
            need(all(v >= 0 for v in a["target_pct"].values()), f"{name}: target_pct >= 0")
    cal = g["calibration"]
    need(cal["n_virtual_nodes"] > 0 and cal["rel_tol"] > 0 and cal["max_iter"] > 0, "grid.calibration out of range")
    m = c["market"]
    need(m["post_eeg"] in ("unsubsidized", "feed_in"), "market.post_eeg: unsubsidized | feed_in")
    cos = m["cohorts"]
    for prev, nxt in zip(cos, cos[1:]):
        need(prev["to"] < nxt["from"], f"cohorts overlap or are unsorted at {nxt['from']}")
    for co in cos:
        need(co["from"] <= co["to"] and co["strike_ct"] > 0, f"cohort {co['from']}: bad range or strike")
        need(co.get("rule_h") is None or co["rule_h"] > 0, f"cohort {co['from']}: rule_h > 0 or null")
    r = m["response"]
    need(r["tau"] >= 0, "market.response.tau >= 0")
    need(set(r["target_band_rate"]) <= set(BANDS), f"target_band_rate keys must be in {BANDS}")
    bat = c["environment"]["bat"]
    lv = bat["levels"]
    need(abs(sum(v["p"] for v in lv.values()) - 1) < 1e-9, "bat level probabilities must sum to 1")
    for name, v in lv.items():
        for k in ("start", "end"):
            need(re.fullmatch(r"\d\d-\d\d", v[k]) is not None, f"bat {name}.{k} must be 'MM-DD'")
        for k in v["before_sunset_h"]:
            need(k == "all" or re.fullmatch(r"\d\d-\d\d", k) is not None,
                 f"bat {name}.before_sunset_h key '{k}' must be 'all' or 'MM-MM'")


def get_curtailment_params(config: dict) -> dict:
    """Resolved and validated curtailment block of a generator config."""
    user = copy.deepcopy((config or {}).get("curtailment") or {})
    areas = (user.get("grid") or {}).get("areas")
    if areas:                                  # area names as in park_grid_areas.csv are accepted
        user["grid"]["areas"] = {area_key(k): v for k, v in areas.items()}
    c = _normalise(deep_merge(CURTAILMENT_DEFAULTS, user))
    validate(c)
    return c


def resolve_path(path: str, repo: str, dataset: str = None) -> str:
    """'~' expanded; relative paths are relative to the generator repo."""
    p = os.path.expanduser(path.format(dataset=dataset) if dataset else path)
    return p if os.path.isabs(p) else os.path.join(repo, p)
