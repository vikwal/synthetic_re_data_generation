"""Curtailed releases of the parks (parks_v1_curt, parks_v1_curt_x4): I/O around
the curtailment package.

Inputs: the parks_v1 release (frames, parks.csv, wind_groups.csv), a chain
config with a 'curtailment' block, the energy-charts driver cache and the
calibration file. Outputs: one synth_<park_id>.parquet + manifest per park in
${DATA_ROOT}/synthetic/wind/<dataset>/, the event list, the copied and
extended metadata tables and the release README.
"""

import json
import os
import shutil

import numpy as np
import pandas as pd

from curtailment import apply as curt_apply
from curtailment import areas, calibrate, drivers, grid, timegrid
from curtailment.config import get_curtailment_params, resolve_path
from parks import assemble, config, paths

SOURCE_DIR = paths.RELEASE_DIR
TABLES = ["parks.csv", "wind_groups.csv", "clients.csv", "sites.csv", "park_layouts.csv",
          "turbine_specs_overrides.csv"]
LAYER_LOSS = ["loss_env", "loss_mkt", "loss_grid"]


def load_chain(path: str) -> tuple:
    chain = config.load_yaml(path)
    cfg = get_curtailment_params(chain)
    if not cfg["enabled"]:
        raise ValueError(f"{path}: curtailment.enabled is false - nothing to do")
    return chain, cfg


def release_dir(cfg: dict) -> str:
    return os.path.join(paths.DATA_ROOT, "synthetic", "wind", cfg["dataset"])


def calibration_path(cfg: dict) -> str:
    return resolve_path(cfg["inputs"]["calibration_file"], paths.REPO, cfg["dataset"])


def driver_cache(cfg: dict) -> str:
    return resolve_path(cfg["inputs"]["driver_cache"], paths.REPO)


def hourly_index(chain: dict) -> pd.DatetimeIndex:
    r2 = chain["round2"]
    return pd.date_range(pd.Timestamp(r2["output_start"], tz="UTC"), pd.Timestamp(r2["output_end"], tz="UTC"),
                         freq="h", name="timestamp")


def source_tables() -> tuple:
    return (pd.read_csv(os.path.join(SOURCE_DIR, "parks.csv")),
            pd.read_csv(os.path.join(SOURCE_DIR, "wind_groups.csv")))


def node_table(cfg: dict, parks: pd.DataFrame) -> pd.DataFrame:
    """Nodes of ALL release parks (independent of a --parks subset)."""
    area = areas.load_area_table(resolve_path(cfg["inputs"]["area_table"], paths.REPO))
    return areas.node_table(parks, area, cfg["grid"]["node_radius_km"])


def read_source(park_id: str, columns=None) -> pd.DataFrame:
    return pd.read_parquet(os.path.join(SOURCE_DIR, f"synth_{park_id}.parquet"), columns=columns)


# ---------------------------------------------------------------- drivers

def fetch_drivers(chain: dict, cfg: dict, local_dir: str, log=print) -> dict:
    idx = hourly_index(chain)
    start = idx[0].strftime("%Y-%m-01")                                   # full first month (r_m)
    end = (idx[-1] + pd.Timedelta(days=1)).strftime("%Y-%m-%d 23:45")
    cache = driver_cache(cfg)
    man = drivers.fetch(cache, start, end, log=log)
    drv, gaps = drivers.load(cache, timegrid.quarter_index(idx))
    cmp = drivers.compare_local(drv, os.path.join(local_dir, "prices_delu_2023_2025.csv"),
                                os.path.join(local_dir, "wind_onshore_de_actual_vs_da_forecast.csv"))
    rm = drivers.market_value(drv["price"], drv["actual_mw"])
    check = {"gaps": gaps, "compare_local": cmp, "r_m": {str(k): float(v) for k, v in rm.items()},
             "neg_slots": int((drv["price"] < 0).sum()), "cf_da_mean": float(drv["cf_da"].mean()),
             "installed_mw_first_last": [float(drv["installed_mw"].iloc[0]), float(drv["installed_mw"].iloc[-1])]}
    with open(os.path.join(cache, "drivers_check.json"), "w") as f:
        json.dump(check, f, indent=1)
    return {"manifest": man, "check": check}


def driver_hashes(cfg: dict) -> dict:
    with open(os.path.join(driver_cache(cfg), "manifest.json")) as f:
        return {k: v["sha256"] for k, v in json.load(f)["files"].items()}


# ---------------------------------------------------------------- calibration

def area_profiles(parks: pd.DataFrame, nodes: pd.DataFrame, n_hours: int) -> dict:
    """{area: (park ids, (n_parks, n_slots) available power / capacity)} on the 15-min grid."""
    cap = parks.set_index("park_id")["capacity_kw"] * 1000.0
    out = {}
    for area, g in nodes.groupby("area"):
        ids = sorted(g["park_id"])
        prof = np.stack([read_source(p, ["power_park"])["power_park"].to_numpy() / cap[p] for p in ids])
        if prof.shape[1] != n_hours:
            raise ValueError(f"{area}: release frames have {prof.shape[1]} hours, chain period {n_hours}")
        out[area] = (ids, timegrid.to_quarter(prof.T).T)
    return out


def _calibrate_area_job(args):
    cfg, area, qidx, cf_da, u, prof = args
    lines = []
    res = calibrate.calibrate_area(cfg, area, qidx, cf_da, u, prof, log=lines.append)
    return area, res, lines


def run_calibration(chain: dict, cfg: dict, workers: int, log=print) -> dict:
    from concurrent.futures import ProcessPoolExecutor
    idx = hourly_index(chain)
    qidx = timegrid.quarter_index(idx)
    drv, gaps = drivers.load(driver_cache(cfg), qidx)
    parks, _ = source_tables()
    nodes = node_table(cfg, parks)
    prof = area_profiles(parks, nodes, len(idx))
    g_areas = [a for a, v in cfg["grid"]["areas"].items() if v["p0"] < 1]
    u = grid.u_on_slots(qidx, g_areas, cfg["grid"]["disturbance"], cfg["seed"])
    jobs = []
    for a in g_areas:
        src = a if a in prof else _neighbour(a, prof)
        if src != a:
            log(f"{a}: no own parks, using the profiles of {src}")
        jobs.append((cfg, a, qidx, drv["cf_da"].to_numpy(), u[a], prof[src][1]))
    out = {"dataset": cfg["dataset"], "seed": cfg["seed"], "target_scale": cfg["grid"]["target_scale"],
           "period": [str(idx[0]), str(idx[-1])], "grid": {}, "grid_fleet": {},
           "created": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
           "git": assemble.git_state(), "drivers_sha256": driver_hashes(cfg)}
    for a, v in cfg["grid"]["areas"].items():
        if v["p0"] >= 1:
            out["grid"][a] = {int(y): {"c": 0.0, "status": "no grid layer (p0 = 1)"}
                              for y in np.unique(timegrid.utc_years(qidx))}
    with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as ex:
        for area, res, lines in ex.map(_calibrate_area_job, jobs):
            for line in lines:
                log(line)
            out["grid"][area] = res["years"]
            out["grid_fleet"][area] = res["fleet"]
    out["market"] = run_market_calibration(cfg, qidx, drv, log)
    path = calibration_path(cfg)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=1, default=str)
    log(f"-> {path}")
    return out


def run_market_calibration(cfg: dict, qidx, drv, log=print) -> dict:
    r = cfg["market"]["response"]
    mastr = pd.read_csv(resolve_path(cfg["inputs"]["mastr_fleet"], paths.REPO), low_memory=False,
                        usecols=["Inbetriebnahmedatum", "Nettonennleistung"])
    classes = calibrate.fleet_classes(mastr, cfg["market"]["cohorts"], "2025-06-25", qidx[-1].tz_convert(None))
    prob = calibrate.MarketProblem(cfg, qidx, drv, classes)
    if r["calibrate"]:
        res = prob.fit((r["theta0"], r["theta1"]), r["tau"])
    else:
        res = {"theta0": r["theta0"], "theta1": r["theta1"], "tau": r["tau"],
               "achieved": calibrate._jsonable(prob.rates(r["theta0"], r["theta1"], r["tau"])),
               "status": "not calibrated (response.calibrate = false)"}
    res["start"] = {"theta0": r["theta0"], "theta1": r["theta1"]}
    res["fleet_gw_by_cohort_end2025"] = _cohort_capacity(classes, "2025-12-31")
    log(f"market: theta0 = {res['theta0']:.3f}, theta1 = {res['theta1']:.3f}, tau = {res['tau']}, "
        f"rmse = {res.get('rmse', float('nan')):.4f}")
    return res


def _cohort_capacity(classes: pd.DataFrame, when: str) -> dict:
    c = classes[classes["date"] <= pd.Timestamp(when)].copy()
    post = c["post_eeg_from"] <= pd.Timestamp(when)
    c["label"] = np.where(post, "post-EEG", c["cohort"].astype(str) + "|rule " + c["rule_h"].astype(str))
    return (c.groupby("label")["mw"].sum() / 1000.0).round(2).to_dict()


def _neighbour(area: str, prof: dict) -> str:
    order = ["A1_SH", "A2_NI_NW", "A3_NI_O_ST", "A4_NO", "A5_MITTE_W", "A6_SUED"]
    i = order.index(area)
    for d in range(1, len(order)):
        for j in (i - d, i + d):
            if 0 <= j < len(order) and order[j] in prof:
                return order[j]
    raise ValueError("no area with parks")


# ---------------------------------------------------------------- apply

_PLAN = {}


def build_plan(chain: dict, cfg: dict) -> curt_apply.Plan:
    parks, _ = source_tables()
    calib = curt_apply.load_calibration(cfg, paths.REPO)
    return curt_apply.build_plan(cfg, hourly_index(chain), node_table(cfg, parks), calib, paths.REPO)


def apply_one(park_id: str) -> dict:
    """Curtail one park with the plan of this process (set by set_plan before forking)."""
    plan, chain, cfg = _PLAN["plan"], _PLAN["chain"], _PLAN["cfg"]
    parks, groups = source_tables()
    src = read_source(park_id)
    meta = curt_apply.park_meta_from_tables(park_id, parks, groups)
    params = {"temp_gradient": config.load_yaml(paths.BASE_YAML)["params"]["temp_gradient"]}
    out, info = curt_apply.curtail_park(src, meta, plan, cfg, params=params, return_info=True)
    with open(os.path.join(SOURCE_DIR, f"manifest_{park_id}.json")) as f:
        man = json.load(f)
    area = info["node_area"]
    man.update({
        "experiment_id": f"{man['experiment_id']}+curtailment",
        "dataset": cfg["dataset"],
        "curtailment": _jsonable_cfg(cfg),
        "curtailment_park": info,
        "curtailment_calibration": {"file": os.path.basename(calibration_path(cfg)),
                                    "c_by_year": {y: v["c"] for y, v in plan.calib["grid"][area].items()},
                                    "market": {k: plan.calib["market"][k] for k in ("theta0", "theta1", "tau")}},
        "drivers_sha256": driver_hashes(cfg),
        "input_release": {"dir": SOURCE_DIR, "release_sha256": man["release_sha256"],
                          "frame_sha256": assemble.frame_sha256(src)},
        "release_sha256": assemble.frame_sha256(out),
        "git_curtailment": assemble.git_state(),
        "created": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
    })
    out_dir = release_dir(cfg)
    os.makedirs(out_dir, exist_ok=True)
    out.to_parquet(os.path.join(out_dir, f"synth_{park_id}.parquet"))
    with open(os.path.join(out_dir, f"manifest_{park_id}.json"), "w") as f:
        json.dump(man, f, indent=2, default=str)
    ev = plan.node_grid(info["node"], area)["events"]
    tot = out["power_park_avail"].sum()
    return {"park_id": park_id, "node": info["node"], "n_events": len(ev),
            **{f"share_{k[5:]}": float(out[k].sum() / tot) if tot > 0 else 0.0 for k in LAYER_LOSS}}


def set_plan(plan, chain, cfg) -> None:
    _PLAN.update(plan=plan, chain=chain, cfg=cfg)


def write_events(plan: curt_apply.Plan, cfg: dict) -> pd.DataFrame:
    """All grid events of all nodes (also of nodes without parks in a --parks subset)."""
    rows = []
    q = plan.qidx
    for node, g in plan.nodes.groupby("node"):
        ng = plan.node_grid(node, g["node_area"].iloc[0])
        ev = ng["events"]
        if not len(ev):
            continue
        end_main = np.minimum(ev["start"] + ev["n_main"], len(q))
        rows.append(pd.DataFrame({
            "event_id": ev["event_id"], "node": node, "area": g["node_area"].iloc[0],
            "start": q[ev["start"]], "duration_h": (end_main - ev["start"]) * timegrid.STEP_H, "setpoint": ev["s0"],
            "release_setpoint": np.where(ev["n_rel"] > 0, ev["s_rel"], np.nan),
            "release_duration_h": ev["n_rel"] * timegrid.STEP_H}))
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    out.to_csv(os.path.join(release_dir(cfg), "grid_events.csv"), index=False)
    return out


def _jsonable_cfg(cfg: dict):
    if isinstance(cfg, dict):
        return {str(k): _jsonable_cfg(v) for k, v in cfg.items()}
    if isinstance(cfg, list):
        return [_jsonable_cfg(v) for v in cfg]
    if hasattr(cfg, "isoformat"):
        return cfg.isoformat()
    return cfg


# ---------------------------------------------------------------- tables

def summary_tables(cfg: dict, plan_nodes: pd.DataFrame) -> tuple:
    """Per park and year, per park, per client: energy shares of the layers and flagged hours."""
    out_dir = release_dir(cfg)
    parks, _ = source_tables()
    rows = []
    for p in parks["park_id"]:
        df = pd.read_parquet(os.path.join(out_dir, f"synth_{p}.parquet"),
                             columns=["power_park", "power_park_avail", "curt_flag", "grid_setpoint"] + LAYER_LOSS)
        for y, g in df.groupby(df.index.year):
            av = g["power_park_avail"].sum()
            rows.append({"park_id": p, "year": int(y), "hours": len(g), "energy_avail_mwh": av / 1e6,
                         "energy_obs_mwh": g["power_park"].sum() / 1e6,
                         **{f"share_{k[5:]}": (g[k].sum() / av if av > 0 else 0.0) for k in LAYER_LOSS},
                         "share_total": ((av - g["power_park"].sum()) / av if av > 0 else 0.0),
                         "hours_flag": int(g["curt_flag"].sum()),
                         "hours_grid_setpoint": int((g["grid_setpoint"] < 1).sum())})
    per_year = pd.DataFrame(rows)
    e = per_year.assign(**{k: per_year[k] * per_year["energy_avail_mwh"] for k in
                           ("share_env", "share_mkt", "share_grid", "share_total")})
    per_park = e.groupby("park_id")[["energy_avail_mwh", "energy_obs_mwh", "share_env", "share_mkt", "share_grid",
                                     "share_total", "hours_flag", "hours_grid_setpoint", "hours"]].sum()
    for k in ("share_env", "share_mkt", "share_grid", "share_total"):
        per_park[k] = per_park[k] / per_park["energy_avail_mwh"]
    per_park["flag_hours_share"] = per_park["hours_flag"] / per_park["hours"]
    return per_year, per_park.reset_index()


def write_tables(cfg: dict, plan_nodes: pd.DataFrame, readme_src: str) -> dict:
    out_dir = release_dir(cfg)
    os.makedirs(out_dir, exist_ok=True)
    for t in TABLES:
        shutil.copy(os.path.join(SOURCE_DIR, t), os.path.join(out_dir, t))
    per_year, per_park = summary_tables(cfg, plan_nodes)
    per_year.to_csv(os.path.join(out_dir, "curtailment_summary.csv"), index=False)
    info = []
    for p in per_park["park_id"]:
        with open(os.path.join(out_dir, f"manifest_{p}.json")) as f:
            cp = json.load(f)["curtailment_park"]
        info.append({"park_id": p, "curt_area": cp["node_area"], "curt_node": cp["node"],
                     "curt_node_free_p0": cp["node_free_p0"], "curt_B_n": cp["B_n"],
                     "curt_bat_level": cp["bat_level"] or "", "curt_n_grid_events": cp["n_grid_events"]})
    ext = per_park.drop(columns=["hours"]).merge(pd.DataFrame(info), on="park_id")
    ext = ext.rename(columns={c: f"curt_{c}" for c in ext.columns if c.startswith(("share_", "hours_", "flag_"))})
    ext = ext.rename(columns={"energy_avail_mwh": "curt_energy_avail_mwh", "energy_obs_mwh": "curt_energy_obs_mwh"})
    parks = pd.read_csv(os.path.join(out_dir, "parks.csv")).merge(ext, on="park_id", how="left")
    parks.to_csv(os.path.join(out_dir, "parks.csv"), index=False)
    cl = parks.groupby("client_id", sort=False).apply(_client_row, include_groups=False).reset_index()
    clients = pd.read_csv(os.path.join(out_dir, "clients.csv")).merge(cl, on="client_id", how="left")
    clients.to_csv(os.path.join(out_dir, "clients.csv"), index=False)
    shutil.copy(readme_src, os.path.join(out_dir, "README.md"))
    return {"parks": parks, "clients": clients, "per_year": per_year}


def _client_row(g: pd.DataFrame) -> pd.Series:
    e = g["curt_energy_avail_mwh"]
    return pd.Series({f"curt_share_{k}": float((g[f"curt_share_{k}"] * e).sum() / e.sum())
                      for k in ("env", "mkt", "grid", "total")} |
                     {"curt_flag_hours_share_mean": float(g["curt_flag_hours_share"].mean()),
                      "curt_parks_with_grid_events": int((g["curt_n_grid_events"] > 0).sum())})
