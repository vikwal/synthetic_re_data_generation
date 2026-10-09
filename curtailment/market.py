"""Market layer: curtailment at negative day-ahead prices (report 6.3).

Cohort per turbine group from its commissioning date and unit rating [D7, D7a];
from 1 Jan of (commissioning year + 21) the group is post-EEG, so the regime is
time dependent. Negative block B = maximal run of DA intervals with price < 0,
L_B its length in hours (known day-ahead). Threshold p*_g(t) [EUR/MWh]:

  post-EEG                       0 ('unsubsidized'; 'feed_in': never exposed)
  rule cohort and L_B >= rule_h  0 for every interval of the block (section 51 EEG)
  otherwise                      -max(10 * strike_ct - r_m, 0)

Exposed iff p_DA(t) < p*_g(t), margin m = p* - p_DA. Response once per park and
block: logit rho = theta0 + theta1 ln(1 + m_bar / 10) + eta_park, m_bar = mean
margin of the exposed intervals weighted with the exposed group power,
eta_park = tau * N(0, 1) fixed per park (rng(seed, "eta", park_id)); the
uniform of the draw is counter-based on (seed, "mkt", park_id) at the
absolute slot of the block start. A reacting park sets m_mkt,g = 1 in all
exposed intervals of the block for all exposed groups.
"""

import datetime as dt

import numpy as np
import pandas as pd

from curtailment import streams, timegrid

BAND_EDGES = {"-10..0": (-10.0, 0.0), "-20..-10": (-20.0, -10.0), "-50..-20": (-50.0, -20.0),
              "-100..-50": (-100.0, -50.0), "<-100": (-np.inf, -100.0)}


def cohort_of(commissioning_date, unit_kw: float, cohorts: list) -> dict:
    """Cohort of a turbine group: strike [ct/kWh], rule length (None if the rule
    does not apply to this unit size), start of post-EEG (1 Jan of year + 21)."""
    d = pd.Timestamp(commissioning_date).date()
    post = dt.date(d.year + 21, 1, 1)
    for k, co in enumerate(cohorts):
        if co["from"] <= d <= co["to"]:
            rule = co.get("rule_h")
            if rule is not None and unit_kw < co.get("min_unit_kw", 0):
                rule = None
            return {"cohort": k, "label": f"{co['from']}..{co['to']}", "strike_ct": float(co["strike_ct"]),
                    "rule_h": rule, "post_eeg_from": post}
    # before the first cohort (pre-2000): post-EEG long before the synthesis period
    return {"cohort": -1, "label": f"<{cohorts[0]['from']}", "strike_ct": float(cohorts[0]["strike_ct"]),
            "rule_h": None, "post_eeg_from": min(post, dt.date(2021, 1, 1))}


def neg_blocks(price: np.ndarray) -> tuple:
    """(block index per slot (-1 outside), block start slot, block length [h])."""
    neg = np.asarray(price) < 0
    m = np.concatenate([[False], neg, [False]])
    d = np.diff(m.astype(np.int8))
    a, b = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    bid = np.full(len(neg), -1, np.int64)
    for k, (s, e) in enumerate(zip(a, b)):
        bid[s:e] = k
    return bid, a, (b - a) * timegrid.STEP_H


def threshold(qidx: pd.DatetimeIndex, price, r_m, block_len_slot, coh: dict, post_eeg: str) -> np.ndarray:
    """p*_g(t) per slot [EUR/MWh] (-inf = never exposed)."""
    premium = np.maximum(10.0 * coh["strike_ct"] - np.asarray(r_m, float), 0.0)
    p = -premium
    if coh["rule_h"] is not None:
        p = np.where(np.asarray(block_len_slot) >= coh["rule_h"] - 1e-9, 0.0, p)
    post = qidx >= pd.Timestamp(coh["post_eeg_from"]).tz_localize(timegrid.TZ_LOCAL)
    p = np.where(post, 0.0 if post_eeg == "unsubsidized" else -np.inf, p)
    return p


def eta_park(seed: int, park_id: str, tau: float) -> float:
    return float(tau * streams.rng(seed, "eta", park_id).standard_normal())


def response_prob(m_bar, theta0: float, theta1: float, eta: float = 0.0) -> np.ndarray:
    x = theta0 + theta1 * np.log1p(np.asarray(m_bar, float) / 10.0) + eta
    return 1.0 / (1.0 + np.exp(-x))


def expected_response(m_bar, theta0: float, theta1: float, tau: float, n_nodes: int = 40) -> np.ndarray:
    """E_eta[rho] with eta ~ N(0, tau^2) (Gauss-Hermite): response of a fleet of
    many parks with the same margin."""
    z, w = np.polynomial.hermite_e.hermegauss(n_nodes)
    w = w / w.sum()
    x = np.asarray(m_bar, float)[..., None]
    return (response_prob(x, theta0, theta1, tau * z) * w).sum(axis=-1)


def park_masks(qidx: pd.DatetimeIndex, drv: pd.DataFrame, blocks: tuple, cohorts: list, p_env_g: np.ndarray,
               park_id: str, cfg: dict, theta: tuple) -> tuple:
    """m_mkt (n_slots, n_groups) bool and per-block records of one park.
    p_env_g: power of each group after the environment layer (weights of m_bar)."""
    mk = cfg["market"]
    bid, starts, lens = blocks
    price = drv["price"].to_numpy()
    blen = np.where(bid >= 0, lens[np.maximum(bid, 0)], 0.0)
    n, G = p_env_g.shape
    pstar = np.column_stack([threshold(qidx, price, drv["r_m"].to_numpy(), blen, c, mk["post_eeg"])
                             for c in cohorts]) if G else np.zeros((n, 0))
    exposed = price[:, None] < pstar
    margin = np.where(exposed, pstar - price[:, None], 0.0)
    m = np.zeros((n, G), bool)
    rows = np.flatnonzero(exposed.any(axis=1))
    if not len(rows):
        return m, pd.DataFrame(columns=["block", "start", "len_h", "m_bar", "rho", "react"])
    th0, th1, tau = theta
    eta = eta_park(cfg["seed"], park_id, tau)
    slots = timegrid.slot_numbers(qidx)
    blk = bid[rows]
    ub = np.unique(blk)
    U = streams.uniform_at(cfg["seed"], ("mkt", park_id), slots[starts[ub]])
    w = np.where(exposed[rows], p_env_g[rows], 0.0)
    num = pd.Series((w * margin[rows]).sum(axis=1)).groupby(blk).sum()
    den = pd.Series(w.sum(axis=1)).groupby(blk).sum()
    raw = pd.Series(margin[rows].sum(axis=1)).groupby(blk).sum() / pd.Series(
        exposed[rows].sum(axis=1)).groupby(blk).sum()
    m_bar = (num / den.where(den > 0)).fillna(raw).reindex(ub).to_numpy()
    rho = response_prob(m_bar, th0, th1, eta)
    react = U < rho
    react_slot = pd.Series(react, index=ub).reindex(blk).to_numpy()
    m[rows] = exposed[rows] & react_slot[:, None]
    rec = pd.DataFrame({"block": ub, "start": qidx[starts[ub]], "len_h": lens[ub], "m_bar": m_bar,
                        "rho": rho, "react": react})
    return m, rec
