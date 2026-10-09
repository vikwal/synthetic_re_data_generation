"""Grid layer: redispatch setpoints per grid node (report 6.2, steps 0-8).

  0 nodes          single linkage of the parks (areas.py)
  1 never affected node free with probability p0          rng(seed, "p0", node)
  2 participation  B_n ~ Beta(alpha, beta), fixed          rng(seed, "beta", node)
  3 disturbance    z_g(d) = phi z_g(d-1) + sqrt(1-phi^2) sigma_z eps_g(d),
                   eps_g = sqrt(rho_z) eta(d) + sqrt(1-rho_z) xi_g(d),
                   u_g(d) = exp(z_g(d) - sigma_z^2/2), days Europe/Berlin
  4 start rate     lambda_n(t) = c_{g,y} B_n exp(a_g + b_g CF_DA(t)) u_g(d(t))  [1/h]
  5 starts         common random numbers: one candidate per quarter-hour with
                   uniform U_t; a start iff U_t < 1 - exp(-lambda_n(t) * 0.25 h)
  6 duration       D ~ LogNormal(mu, sigma) [h], capped at max_duration_h
  7 setpoint path  main segment [start, start + D) at s0 ~ setpoint_first; only
                   if s0 = 0: with prob. release.p an appended segment at 0.3
                   (share p30) or 0.6, LogNormal(release.dur); rounded to the
                   15-min grid (>= 1 quarter-hour); overlaps: lowest setpoint
  8 coupling       common CF_DA and rho_z; area A6 has no grid layer

Step 5 is thinning with a degenerate candidate process (every quarter-hour is a
candidate): U_t, the duration and the setpoint path of each candidate are
counter-based draws keyed by (seed, node, absolute slot), independent of c and
of target_scale. A start happens iff c_{g,y} > c*_t with the critical value
c*_t = -ln(1 - U_t) / (0.25 h * B_n exp(a + b CF) u). Hence a run with larger
c contains every event of a run with smaller c (parks_v1_curt_x4 contains
parks_v1_curt), and no fixed upper rate c_max has to be chosen.
"""

import numpy as np
import pandas as pd
from scipy.special import ndtri

from curtailment import streams, timegrid

AR1_ORIGIN = pd.Timestamp("2000-01-01")   # the AR(1) path is anchored here, so u_g(d) depends on d only
LEVELS = (0.0, 0.3, 0.6)


# ---------------------------------------------------------------- step 3

def disturbance(days: pd.DatetimeIndex, areas, dist: dict, seed: int) -> dict:
    """u_g(d) for the given local days (tz-naive dates) and areas: {area: array}."""
    if days.min() < AR1_ORIGIN:
        raise ValueError(f"days before the AR(1) origin {AR1_ORIGIN.date()}")
    n = int((days.max() - AR1_ORIGIN).days) + 1
    pos = ((days - AR1_ORIGIN).days).to_numpy()
    phi, sig, rho = dist["phi"], dist["sigma_z"], dist["rho_z"]
    eta = streams.rng(seed, "ar1_common").standard_normal(n)
    out = {}
    for a in areas:
        r = streams.rng(seed, "ar1_area", a)
        z0 = sig * r.standard_normal()
        xi = r.standard_normal(n)
        eps = np.sqrt(rho) * eta + np.sqrt(1 - rho) * xi
        z = np.empty(n)
        z[0] = z0
        k = np.sqrt(1 - phi ** 2) * sig
        for i in range(1, n):
            z[i] = phi * z[i - 1] + k * eps[i]
        out[a] = np.exp(z[pos] - sig ** 2 / 2)
    return out


def u_on_slots(qidx: pd.DatetimeIndex, areas, dist: dict, seed: int) -> dict:
    days, inv = timegrid.day_codes(qidx)
    u = disturbance(days, areas, dist, seed)
    return {a: v[inv] for a, v in u.items()}


# ---------------------------------------------------------------- steps 1-2

def node_state(seed: int, node: str, area_cfg: dict) -> dict:
    """p0 status and participation of a real node (fixed over all years)."""
    free = bool(streams.rng(seed, "p0", node).random() < area_cfg["p0"])
    if area_cfg["p0"] >= 1:
        return {"free": True, "B": 0.0}
    a, b = area_cfg["beta"]
    return {"free": free, "B": float(streams.rng(seed, "beta", node).beta(a, b))}


# ---------------------------------------------------------------- steps 4-7

def base_rate(cf_da, u, area_cfg: dict, B: float) -> np.ndarray:
    """lambda_n(t) / c [1/h]."""
    return B * np.exp(area_cfg["a"] + area_cfg["b"] * np.asarray(cf_da)) * np.asarray(u)


def critical_c(U, rate0) -> np.ndarray:
    """c above which a start happens in the slot (inf where the base rate is 0)."""
    with np.errstate(divide="ignore"):
        return -np.log1p(-np.asarray(U)) / (timegrid.STEP_H * np.asarray(rate0))


def draw_u(seed: int, key: tuple, slots) -> np.ndarray:
    return streams.uniform_at(seed, key, slots)


def event_shapes(seed: int, key: tuple, slots, area_cfg: dict, grid_cfg: dict) -> dict:
    """Duration and setpoint path of the candidates at the absolute slots `slots`
    (independent of c): n_main, s0, n_rel, s_rel (n_rel = 0: no release segment)."""
    slots = np.asarray(slots, dtype=np.int64)
    cap = grid_cfg["max_duration_h"]
    mu, sig = area_cfg["dur"]
    d = np.minimum(np.exp(mu + sig * ndtri(_open(draw_u(seed, key + ("dur",), slots)))), cap)
    n_main = np.maximum(1, np.rint(d / timegrid.STEP_H)).astype(np.int32)
    levels = np.array(list(grid_cfg["setpoint_first"]), float)
    cum = np.cumsum(list(grid_cfg["setpoint_first"].values()))
    idx = np.minimum(np.searchsorted(cum, draw_u(seed, key + ("s0",), slots), side="right"), len(levels) - 1)
    s0 = levels[idx]
    rel = area_cfg["release"]
    has_rel = (s0 == 0.0) & (draw_u(seed, key + ("rel",), slots) < rel["p"])
    s_rel = np.where(draw_u(seed, key + ("rel30",), slots) < rel["p30"], 0.3, 0.6)
    mr, sr = rel["dur"]
    dr = np.minimum(np.exp(mr + sr * ndtri(_open(draw_u(seed, key + ("reldur",), slots)))), cap)
    n_rel = np.where(has_rel, np.maximum(1, np.rint(dr / timegrid.STEP_H)), 0).astype(np.int32)
    return {"n_main": n_main, "s0": s0, "n_rel": n_rel, "s_rel": np.where(has_rel, s_rel, 1.0)}


def _open(u):
    return np.clip(u, 1e-16, 1 - 1e-16)


def node_events(seed: int, node: str, qidx: pd.DatetimeIndex, shape0: np.ndarray, B: float, c_slot: np.ndarray,
                area: str, area_cfg: dict, grid_cfg: dict, cf: np.ndarray = None) -> pd.DataFrame:
    """Accepted events of a real node on the period qidx (shape0 = exp(a + b CF_DA) u):
    one row per event with start index, n_main, s0, n_rel, s_rel, event_id (and the
    area episode in mode 'area')."""
    from curtailment import events
    slots = timegrid.slot_numbers(qidx)
    c_store = float(np.max(c_slot)) if len(c_slot) else 0.0
    eps = (events.area_episodes(seed, area, slots, shape0, area_cfg, grid_cfg, c_store)
           if grid_cfg["events"]["mode"] == "area" else None)
    cd = events.candidates(seed, ("grid", node), B, slots, shape0, area_cfg, grid_cfg, c_store, eps, cf)
    on = c_slot[cd["pos"]] > cd["cstar"]
    ev = pd.DataFrame({"start": cd["pos"][on], **{k: cd[k][on] for k in ("n_main", "s0", "n_rel", "s_rel")}})
    ev["event_id"] = [f"{node}:{t:%Y%m%dT%H%M}" for t in qidx[ev["start"]]]
    if "episode" in cd:
        ev["episode_id"] = [f"{area}:{t:%Y%m%dT%H%M}" for t in qidx[cd["episode"][on]]]
    return ev


def setpoint_series(n: int, ev: pd.DataFrame) -> tuple:
    """Setpoint s(t) (1 = free) and the id of the binding event per slot ('' if free).
    Lowest setpoint wins; among equal setpoints the earliest start."""
    s = np.ones(n)
    eid = np.full(n, -1, dtype=np.int64)
    segs = []
    for k, r in enumerate(ev.itertuples(index=False)):
        a, b = r.start, min(r.start + r.n_main, n)
        segs.append((r.s0, -r.start, a, b, k))
        if r.n_rel > 0 and b < n:
            segs.append((r.s_rel, -r.start, b, min(b + r.n_rel, n), k))
    # paint from the weakest to the binding segment: higher setpoints first, later starts first
    for lev, _, a, b, k in sorted(segs, key=lambda x: (-x[0], x[1])):
        s[a:b] = lev
        eid[a:b] = k
    ids = np.where(eid >= 0, ev["event_id"].to_numpy(dtype=object)[np.maximum(eid, 0)] if len(ev) else "", "")
    return s, ids


def segments_covering(n: int, start, n_main, s0, n_rel, s_rel, row=None, n_rows: int = 1) -> np.ndarray:
    """Vectorised setpoint of many events on an (n_rows, n) grid (no ids):
    coverage per level by difference arrays, then the lowest covering level."""
    row = np.zeros(len(start), np.int64) if row is None else np.asarray(row, np.int64)
    start = np.asarray(start, np.int64)
    out = np.ones((n_rows, n))
    end_main = np.minimum(start + n_main, n)
    seg = [(start, end_main, s0), (end_main, np.minimum(end_main + n_rel, n), s_rel)]
    for lev in sorted(LEVELS, reverse=True):
        diff = np.zeros(n_rows * (n + 1), np.int32)
        for a, b, s in seg:
            m = (s == lev) & (b > a)
            np.add.at(diff, row[m] * (n + 1) + a[m], 1)
            np.add.at(diff, row[m] * (n + 1) + b[m], -1)
        cov = np.cumsum(diff.reshape(n_rows, n + 1)[:, :n], axis=1) > 0
        out[cov] = lev
    return out


def apply_setpoint(p, s, p_inst: float) -> np.ndarray:
    """P_obs = min(P, s * P_inst)."""
    return np.minimum(p, np.asarray(s) * p_inst)
