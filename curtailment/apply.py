"""Apply the curtailment layers to one park frame.

Order as in settlement (report 6.1), per quarter-hour t, groups g with
share_g = power_t<g> / power_park_free (0 where the denominator is 0):

  P_avail = power_park                         (free x wake, aging included)
  P_env   = P_avail * sum_g share_g (1 - m_env,g)
  P_mkt   = P_avail * sum_g share_g (1 - m_env,g)(1 - m_mkt,g)
  P_obs   = min(P_mkt, s * P_inst)             (P_inst = capacity_kw, W)

P_mkt uses the product of both masks, so a group switched off by both layers
is counted once (in the environment layer). Hourly outputs are means of the
four quarter-hours; power_park becomes P_obs, the input power_park is kept as
power_park_avail.

A Plan holds everything shared by the parks of one run: 15-min grid, drivers,
negative blocks, disturbance u_g, calibration and the grid nodes (setpoint
series are computed once per node and cached).
"""

import json
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from curtailment import areas as areas_mod
from curtailment import drivers, environment, grid, market, timegrid

NEW_COLS = ["power_park_avail", "env_factor", "mkt_factor", "grid_setpoint", "loss_env", "loss_mkt",
            "loss_grid", "curt_flag", "grid_event_id", "neg_block_len_h"]


@dataclass
class Plan:
    cfg: dict
    qidx: pd.DatetimeIndex
    drv: pd.DataFrame
    blocks: tuple
    u: dict
    calib: dict
    nodes: pd.DataFrame                    # park_id -> node, node_area
    node_cache: dict = field(default_factory=dict)

    @property
    def theta(self) -> tuple:
        m = self.calib["market"]
        return m["theta0"], m["theta1"], m["tau"]

    def c_slot(self, area: str) -> np.ndarray:
        """c_{g,y} on the slots (UTC calendar year)."""
        cy = self.calib["grid"][area]
        years = timegrid.utc_years(self.qidx)
        missing = sorted(set(np.unique(years)) - {int(y) for y in cy})
        if missing:
            raise KeyError(f"calibration has no c for area {area}, years {missing}")
        lut = {int(y): float(v["c"]) for y, v in cy.items()}
        return np.array([lut[y] for y in years]) if len(set(lut)) > 1 else np.full(len(years), lut[years[0]])

    def node_grid(self, node: str, area: str) -> dict:
        """Setpoint series, event ids and events of a node (cached)."""
        if node in self.node_cache:
            return self.node_cache[node]
        n = len(self.qidx)
        acfg = self.cfg["grid"]["areas"][area]
        st = grid.node_state(self.cfg["seed"], node, acfg)
        if st["free"] or not self.cfg["layers"].get("grid", True):
            res = {"s": np.ones(n), "ids": np.full(n, "", dtype=object), "events": pd.DataFrame(), **st}
        else:
            cf = self.drv["cf_da"].to_numpy()
            shape0 = grid.base_rate(cf, self.u[area], acfg, 1.0)
            ev = grid.node_events(self.cfg["seed"], node, self.qidx, shape0, st["B"], self.c_slot(area), area, acfg,
                                  self.cfg["grid"], cf)
            s, ids = grid.setpoint_series(n, ev)
            res = {"s": s, "ids": ids, "events": ev, **st}
        res["area"] = area
        self.node_cache[node] = res
        return res


def load_calibration(cfg: dict, repo: str) -> dict:
    from curtailment.config import resolve_path
    path = resolve_path(cfg["inputs"]["calibration_file"], repo, cfg["dataset"])
    if not os.path.exists(path):
        raise FileNotFoundError(f"curtailment calibration file {path} missing - run "
                                f"'scripts/parks/run_parks_v1.py curt_calibrate --chain <yaml>' first")
    with open(path) as f:
        cal = json.load(f)
    if float(cal["target_scale"]) != float(cfg["grid"]["target_scale"]) or int(cal["seed"]) != int(cfg["seed"]):
        raise ValueError(f"{path} was calibrated for target_scale {cal['target_scale']} / seed {cal['seed']}, "
                         f"config has {cfg['grid']['target_scale']} / {cfg['seed']}")
    return cal


def build_plan(cfg: dict, hourly_index: pd.DatetimeIndex, nodes: pd.DataFrame, calib: dict, repo: str) -> Plan:
    from curtailment.config import resolve_path
    qidx = timegrid.quarter_index(hourly_index)
    drv, _ = drivers.load(resolve_path(cfg["inputs"]["driver_cache"], repo), qidx)
    grid_areas = [a for a, v in cfg["grid"]["areas"].items() if v["p0"] < 1]
    u = grid.u_on_slots(qidx, grid_areas, cfg["grid"]["disturbance"], cfg["seed"])
    return Plan(cfg=cfg, qidx=qidx, drv=drv, blocks=market.neg_blocks(drv["price"].to_numpy()), u=u,
                calib=calib, nodes=nodes.set_index("park_id"))


# ---------------------------------------------------------------- park meta

def park_meta_from_tables(park_id: str, parks: pd.DataFrame, groups: pd.DataFrame) -> dict:
    """Park meta from the release tables parks.csv / wind_groups.csv."""
    p = parks.set_index("park_id").loc[park_id]
    g = groups[groups["park_id"] == park_id]
    return {"park_id": park_id, "latitude": float(p["latitude"]), "longitude": float(p["longitude"]),
            "capacity_kw": float(p["capacity_kw"]),
            "groups": [{"group_id": r.group_id, "commissioning_date": str(r.commissioning_date),
                        "unit_kw": float(r.rated_kw), "n": int(r.n_turbines), "hub_height": float(r.hub_height_m)}
                       for r in g.itertuples(index=False)]}


def park_meta_from_station(park_id: str, lat: float, lon: float, params: dict, commissioning_date,
                           unit_kw: list) -> dict:
    """Park meta of a DWD station run: one group t<i> per configured turbine."""
    groups = [{"group_id": f"t{i}", "commissioning_date": str(commissioning_date), "unit_kw": float(kw), "n": 1,
               "hub_height": float(h), "wind_col": f"wind_speed_t{i}"}
              for i, (kw, h) in enumerate(zip(unit_kw, params["hub_heights"]), start=1)]
    return {"park_id": str(park_id), "latitude": float(lat), "longitude": float(lon),
            "capacity_kw": float(sum(unit_kw)), "groups": groups}


def station_plan(cfg: dict, df: pd.DataFrame, park: dict, repo: str) -> Plan:
    """Plan of a single DWD station run: node = park, area by the station coordinate."""
    from curtailment.config import resolve_path
    calib = load_calibration(cfg, repo)
    area = areas_mod.area_of_point(park["longitude"], park["latitude"],
                                   resolve_path(cfg["inputs"]["states_geojson"], repo))[0]
    nodes = pd.DataFrame({"park_id": [park["park_id"]], "node": [park["park_id"]], "area": [area],
                          "node_area": [area]})
    return build_plan(cfg, df.index, nodes, calib, repo)


# ---------------------------------------------------------------- apply

def cohorts_of(park: dict, cfg: dict) -> list:
    return [market.cohort_of(g["commissioning_date"], g["unit_kw"], cfg["market"]["cohorts"]) for g in park["groups"]]


def curtail_park(release_df: pd.DataFrame, park_meta: dict, plan: Plan, cfg: dict, params: dict = None,
                 return_info: bool = False):
    """Curtailed release frame: input columns unchanged except power_park
    (= P_obs), new columns NEW_COLS appended."""
    df = release_df
    if df.index.tz is None:                    # station frames may carry naive UTC stamps
        out = curtail_park(df.set_axis(df.index.tz_localize("UTC")), park_meta, plan, cfg, params, return_info)
        if return_info:
            return out[0].set_axis(df.index), out[1]
        return out.set_axis(df.index)
    qidx = timegrid.quarter_index(df.index)
    if not qidx.equals(plan.qidx):
        raise ValueError("release frame and plan cover different periods")
    gids = [g["group_id"] for g in park_meta["groups"]]
    p_inst = park_meta["capacity_kw"] * 1000.0
    free = df["power_park_free"].to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        share_h = np.where(free[:, None] > 0, df[[f"power_{g}" for g in gids]].to_numpy() / free[:, None], 0.0)
    P = timegrid.to_quarter(df["power_park"].to_numpy())
    S = timegrid.to_quarter(share_h)
    n, G = S.shape
    layers = cfg["layers"]

    m_env, level = (environment.bat_masks(qidx, df, park_meta, cfg, params) if layers.get("environment", True)
                    else (np.zeros((n, G), bool), None))
    keep_env = S * (1 - m_env)
    # factors are exactly 1 where no mask is active (no rounding of the share sum)
    f_env = np.where(m_env.any(axis=1), np.minimum(keep_env.sum(axis=1), 1.0), 1.0)
    p_env = P * f_env

    coh = cohorts_of(park_meta, cfg)
    if layers.get("market", True):
        m_mkt, rec = market.park_masks(qidx, plan.drv, plan.blocks, coh, P[:, None] * keep_env,
                                       park_meta["park_id"], cfg, plan.theta)
    else:
        m_mkt, rec = np.zeros((n, G), bool), pd.DataFrame()
    any_mkt = m_mkt.any(axis=1)
    f_mkt = np.where(any_mkt, np.minimum((S * (1 - m_mkt)).sum(axis=1), 1.0), 1.0)
    f_both = np.where(any_mkt, np.minimum((keep_env * (1 - m_mkt)).sum(axis=1), f_env), f_env)
    p_mkt = P * f_both

    nrow = plan.nodes.loc[park_meta["park_id"]]
    ng = plan.node_grid(nrow["node"], nrow["node_area"])
    p_obs = grid.apply_setpoint(p_mkt, ng["s"], p_inst)

    loss_q = P - p_obs
    loss_h = timegrid.to_hour(loss_q)
    bid, _, lens = plan.blocks
    blen = np.where(bid >= 0, lens[np.maximum(bid, 0)], 0.0)
    out = df.copy()
    out["power_park"] = np.clip(df["power_park"].to_numpy() - loss_h, 0.0, None)
    out["power_park_avail"] = df["power_park"].to_numpy()
    out["env_factor"] = timegrid.to_hour(f_env)
    out["mkt_factor"] = timegrid.to_hour(f_mkt)
    out["grid_setpoint"] = timegrid.to_hour(ng["s"])
    out["loss_env"] = timegrid.to_hour(P - p_env)
    out["loss_mkt"] = timegrid.to_hour(p_env - p_mkt)
    out["loss_grid"] = timegrid.to_hour(p_mkt - p_obs)
    out["curt_flag"] = (loss_h > 0).astype(np.int8)
    ids = ng["ids"].reshape(-1, timegrid.PER_HOUR)
    first = np.argmax(ids != "", axis=1)
    out["grid_event_id"] = ids[np.arange(len(ids)), first].astype(str)
    out["neg_block_len_h"] = timegrid.hour_max(blen)
    if not return_info:
        return out
    info = {"area": nrow["area"], "node": nrow["node"], "node_area": nrow["node_area"],
            "node_free_p0": bool(ng["free"]), "B_n": ng["B"], "bat_level": level,
            "eta_park": market.eta_park(cfg["seed"], park_meta["park_id"], plan.theta[2]),
            "cohorts": [{"group_id": g, **{k: (str(v) if k == "post_eeg_from" else v) for k, v in c.items()}}
                        for g, c in zip(gids, coh)],
            "n_grid_events": int(len(ng["events"])),
            "n_neg_blocks_exposed": int(len(rec)), "n_neg_blocks_reacted": int(rec["react"].sum()) if len(rec) else 0}
    return out, info
