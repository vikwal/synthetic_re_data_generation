"""Candidate grid events of a node, in two switchable modes, plus the optional
end of an event at low wind.

grid.events.mode
  node   (v1) every node has its own candidate process: one candidate per
         quarter-hour, start iff c > c*_t = -ln(1 - U_t) / (0.25 h * B_n exp(a + b CF) u).
  area   (v1.1) shared area episodes: per quarter-hour the number of episodes is
         N(c) = Poisson quantile of a common uniform U_t at mean c mu_t,
         mu_t = exp(a + b CF) u / kappa * 0.25 h (monotone in c: thinning
         property kept); a node joins each episode with probability
         q_n = min(1, kappa * B_n) and starts if it joins at least one; with one
         uniform V per node and slot this is N(c) >= n_req(V, q_n), i.e.
         c > c* = gammainccinv(n_req, 1 - U_t) / mu_t.
         The start rate of a node is unchanged, B_n lambda (as long as
         kappa B_n <= 1), but nodes now start together, which couples them.
         kappa = 1 is plain Bernoulli participation (correlation ~ B_n, i.e.
         practically none); larger kappa = fewer, larger episodes.
         grid.events.shared_duration: the joining nodes take the duration of
         the episode (else each node draws its own); set-points per node.

grid.termination (v1.1, switchable)
  an event ends at the first quarter-hour after its start with
  CF_DA < areas.<A>.end_cf (ramp threshold c0 of [C20]); at least one
  quarter-hour; an appended release segment follows the truncated end.

All draws are counter-based on (seed, keys, absolute slot) and independent of
c, so a larger c only adds events (x4 contains x1) in both modes.
"""

import numpy as np

from curtailment import grid, timegrid


def low_cf_next(cf: np.ndarray, end_cf: float) -> np.ndarray:
    """For every index i the first index j >= i with cf[j] < end_cf (len(cf) if none)."""
    n = len(cf)
    low = np.flatnonzero(np.asarray(cf) < end_cf)
    j = np.searchsorted(low, np.arange(n))
    return np.where(j < len(low), low[np.minimum(j, len(low) - 1)], n)


def truncate(pos: np.ndarray, n_main: np.ndarray, nxt: np.ndarray) -> np.ndarray:
    """Main segment length after the end at low wind (first low slot after the start)."""
    n = len(nxt)
    after = np.minimum(np.asarray(pos) + 1, n - 1)
    stop = np.where(np.asarray(pos) + 1 < n, nxt[after], n)
    return np.clip(stop - pos, 1, n_main).astype(np.int32)


def area_episodes(seed: int, area: str, slots: np.ndarray, shape0: np.ndarray, acfg: dict, gcfg: dict,
                  c_store: float = np.inf) -> dict:
    """Candidate episode slots of an area on a window (at least one episode for some c < c_store):
    position, common uniform, mean per unit c, maximal episode count at c_store, duration."""
    from scipy.stats import poisson
    kappa = float(gcfg["events"]["concentration"])
    U = grid.draw_u(seed, ("area_ep", area, "U"), slots)
    mu = np.asarray(shape0) / kappa * timegrid.STEP_H
    cs1 = grid.critical_c(U, np.asarray(shape0) / kappa)            # c of the first episode
    k = np.flatnonzero(cs1 < c_store)
    n_max = (poisson.ppf(1.0 - U[k], c_store * mu[k]) if np.isfinite(c_store)
             else np.full(len(k), np.iinfo(np.int32).max)).astype(np.int64)
    n_main = grid.event_shapes(seed, ("area_ep", area), slots[k], acfg, gcfg)["n_main"]
    return {"pos": k, "U": U[k], "mu": mu[k], "n_max": np.maximum(n_max, 1), "n_main": n_main}


def _join_cstar(V: np.ndarray, q: float, U: np.ndarray, mu: np.ndarray, n_max: np.ndarray) -> np.ndarray:
    """Critical c of a node in candidate episode slots (inf where it can not start below c_store)."""
    from scipy.special import gammainccinv
    if q >= 1:
        n_req = np.ones(len(V), np.int64)
    elif q <= 0:
        return np.full(len(V), np.inf)
    else:
        n_req = np.floor(np.log1p(-V) / np.log1p(-q)).astype(np.int64) + 1
    out = np.full(len(V), np.inf)
    ok = n_req <= n_max
    with np.errstate(divide="ignore"):
        out[ok] = gammainccinv(n_req[ok].astype(float), 1.0 - U[ok]) / mu[ok]
    return out


def candidates(seed: int, key: tuple, B: float, slots: np.ndarray, shape0: np.ndarray, acfg: dict, gcfg: dict,
               c_store: float = np.inf, episodes: dict = None, cf: np.ndarray = None) -> dict:
    """Candidate events of one node on a window (slots, shape0 = exp(a + b CF) u on
    the window). Returns pos, cstar, n_main, s0, n_rel, s_rel (and episode pos in mode
    'area'); a candidate becomes an event iff c(t) > cstar."""
    mode = gcfg["events"]["mode"]
    if mode == "node":
        U = grid.draw_u(seed, key + ("U",), slots)
        cs = grid.critical_c(U, B * np.asarray(shape0))
        k = np.flatnonzero(cs < c_store)
        out = {"pos": k, "cstar": cs[k], **grid.event_shapes(seed, key, slots[k], acfg, gcfg)}
    elif mode == "area":
        if episodes is None:
            raise ValueError("mode 'area' needs the area episodes of the window")
        q = min(1.0, float(gcfg["events"]["concentration"]) * B)
        V = grid.draw_u(seed, key + ("join",), slots[episodes["pos"]])
        cs = _join_cstar(V, q, episodes["U"], episodes["mu"], episodes["n_max"])
        m = cs < c_store
        pos = episodes["pos"][m]
        out = {"pos": pos, "cstar": cs[m], **grid.event_shapes(seed, key, slots[pos], acfg, gcfg)}
        if gcfg["events"]["shared_duration"]:
            out["n_main"] = episodes["n_main"][m]
        out["episode"] = pos
    else:
        raise ValueError(f"grid.events.mode '{mode}'")
    if gcfg["termination"]["enabled"]:
        if cf is None:
            raise ValueError("termination needs CF_DA on the window")
        out["n_main"] = truncate(out["pos"], out["n_main"], low_cf_next(cf, acfg["end_cf"]))
    return out
