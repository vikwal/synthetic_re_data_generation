"""Calibration of the grid scale c_{g,y} and of the market response (theta0, theta1).

Grid (report 6.2 step 7): per area g a virtual fleet of n_virtual_nodes nodes
(p0 status, B_n ~ Beta and the assigned park drawn from rng(seed, "vfleet",
area, j)); each node gets the available power of a randomly chosen park of
the area, normalised by its capacity (every virtual node has the weight of one
equally sized unit). Per calendar year y (in time order, earlier years fixed),
bisection on log c_{g,y} until the curtailed energy share
sum(P_avail - P_obs) / sum(P_avail) over the year (2023 from the period start,
2026 to the period end) is within rel_tol of target_scale * target_pct[y].
The candidates (U, duration, setpoint path per quarter-hour) are the same
common random numbers for every c and every target_scale (grid.py), so a
start happens iff c > c*_t and the share is monotone in c.

Market (report 6.3 step 4): a virtual fleet of all MaStR onshore units
(commissioning date, unit rating -> cohort class, post-EEG year), capacity of
each class over time, availability = national DA forecast CF. Because units of
one class share threshold, exposure and margin, the response is integrated
over eta ~ N(0, tau^2) instead of drawn. theta0, theta1 (tau fixed) are fitted
by least squares to the exposed response per year and to the pooled
curtailment rate per price band.
"""

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from curtailment import grid, market, streams, timegrid

C_FLOOR, C_CEIL = 1e-6, 1e5


# ---------------------------------------------------------------- grid

def virtual_fleet(seed: int, area: str, area_cfg: dict, n_nodes: int, n_parks: int) -> pd.DataFrame:
    rows = []
    a, b = area_cfg["beta"]
    for j in range(n_nodes):
        r = streams.rng(seed, "vfleet", area, j)
        k = int(r.integers(n_parks))
        free = bool(r.random() < area_cfg["p0"])
        rows.append({"j": j, "park": k, "free": free, "B": float(r.beta(a, b))})
    return pd.DataFrame(rows)


class AreaGridProblem:
    """Energy share of one area's virtual fleet as a function of c per year."""

    def __init__(self, cfg: dict, area: str, qidx: pd.DatetimeIndex, cf_da: np.ndarray, u: np.ndarray,
                 profiles: np.ndarray):
        """profiles: (n_parks, n_slots) available power / capacity on the 15-min grid."""
        self.cfg, self.area, self.qidx = cfg, area, qidx
        self.acfg = cfg["grid"]["areas"][area]
        self.gcfg = cfg["grid"]
        self.slots = timegrid.slot_numbers(qidx)
        self.years = timegrid.utc_years(qidx)
        self.profiles = profiles
        n_nodes = cfg["grid"]["calibration"]["n_virtual_nodes"]
        self.fleet = virtual_fleet(cfg["seed"], area, self.acfg, n_nodes, profiles.shape[0])
        self.active = self.fleet[~self.fleet["free"]].reset_index(drop=True)
        self.shape0 = np.exp(self.acfg["a"] + self.acfg["b"] * cf_da) * u        # rate0 / B
        spill_h = 2 * cfg["grid"]["max_duration_h"]
        self.spill = int(np.ceil(spill_h / timegrid.STEP_H))
        self.c = {}                                                              # fixed c per year
        self._cand = None

    def window(self, y: int) -> tuple:
        idx = np.flatnonzero(self.years == y)
        return max(0, idx[0] - self.spill), idx[0], idx[-1] + 1                 # ws, ys, ye

    def candidates(self, y: int, c_store: float) -> dict:
        """Candidate starts in the window of year y with critical c below c_store,
        their (c-independent) shapes, and the availability profiles of the year."""
        ws, ys, ye = self.window(y)
        sl = self.slots[ws:ye]
        parts = {k: [] for k in ("row", "pos", "cstar", "n_main", "s0", "n_rel", "s_rel")}
        for i, r in enumerate(self.active.itertuples(index=False)):
            U = grid.draw_u(self.cfg["seed"], ("vgrid", self.area, r.j, "U"), sl)
            cs = grid.critical_c(U, r.B * self.shape0[ws:ye])
            k = np.flatnonzero(cs < c_store)
            e = grid.event_shapes(self.cfg["seed"], ("vgrid", self.area, r.j), sl[k], self.acfg, self.gcfg)
            parts["row"].append(np.full(len(k), i, np.int64))
            parts["pos"].append(k)
            parts["cstar"].append(cs[k])
            for key in ("n_main", "s0", "n_rel", "s_rel"):
                parts[key].append(e[key])
        cand = {k: np.concatenate(v) for k, v in parts.items()}
        cand["slot_year"] = self.years[ws + cand["pos"]]
        cand.update(c_store=c_store, year=y,
                    prof=self.profiles[self.active["park"].to_numpy()][:, ys:ye],
                    avail=float(self.profiles[self.fleet["park"].to_numpy()][:, ys:ye].sum()))
        return cand

    def share(self, y: int, c_y: float) -> float:
        ws, ys, ye = self.window(y)
        if self._cand is None or self._cand["year"] != y or c_y >= self._cand["c_store"]:
            self._cand = self.candidates(y, max(64.0, 8 * c_y))
        cd = self._cand
        c_slot = np.full(len(cd["pos"]), c_y)
        for yy, cv in self.c.items():
            if yy != y:
                c_slot[cd["slot_year"] == yy] = cv
        on = cd["cstar"] < c_slot
        s = grid.segments_covering(ye - ws, cd["pos"][on], cd["n_main"][on], cd["s0"][on], cd["n_rel"][on],
                                   cd["s_rel"][on], row=cd["row"][on], n_rows=len(self.active))[:, ys - ws:]
        hit = s < 1
        loss = np.clip(cd["prof"][hit] - s[hit], 0, None).sum()
        return float(loss / cd["avail"]) if cd["avail"] > 0 else 0.0

    def solve_year(self, y: int, target: float, rel_tol: float, max_iter: int) -> dict:
        if target <= 0:
            self.c[y] = 0.0
            return {"c": 0.0, "target": target, "achieved": 0.0, "iterations": 0, "status": "zero target"}
        hist = []

        def f(c):
            v = self.share(y, c)
            hist.append((c, v))
            return v

        c = self.c.get(y - 1) or 1.0
        v = f(c)
        lo, hi = ((c, v), None) if v < target else (None, (c, v))
        while hi is None and c < C_CEIL:          # expand the bracket upwards
            c *= 4
            v = f(c)
            if v >= target:
                hi = (c, v)
            else:
                lo = (c, v)
        while lo is None and c > C_FLOOR:         # ... or downwards
            c /= 4
            v = f(c)
            if v < target:
                lo = (c, v)
            else:
                hi = (c, v)
        if hi is None:
            self.c[y] = lo[0]
            return {"c": lo[0], "target": target, "achieved": lo[1], "iterations": len(hist),
                    "status": f"SATURATED: target not reached up to c = {C_CEIL:g}", "history": hist}
        if lo is None:
            self.c[y] = hi[0]
            return {"c": hi[0], "target": target, "achieved": hi[1], "iterations": len(hist),
                    "status": "target exceeded at the floor c", "history": hist}
        best = min((lo, hi), key=lambda t: abs(t[1] / target - 1))
        while abs(best[1] / target - 1) > rel_tol and len(hist) < max_iter:
            c = float(np.sqrt(lo[0] * hi[0]))
            v = f(c)
            if v < target:
                lo = (c, v)
            else:
                hi = (c, v)
            best = min((lo, hi, best), key=lambda t: abs(t[1] / target - 1))
        ok = abs(best[1] / target - 1) <= rel_tol
        self.c[y] = best[0]
        return {"c": best[0], "target": target, "achieved": best[1], "iterations": len(hist),
                "status": "converged" if ok else "NOT CONVERGED (max_iter)", "history": hist}


def calibrate_area(cfg: dict, area: str, qidx, cf_da, u, profiles, log=print) -> dict:
    prob = AreaGridProblem(cfg, area, qidx, cf_da, u, profiles)
    acfg = cfg["grid"]["areas"][area]
    cal = cfg["grid"]["calibration"]
    res = {}
    for y in sorted(np.unique(prob.years)):
        tgt = cfg["grid"]["target_scale"] * acfg["target_pct"][int(y)] / 100.0
        r = prob.solve_year(int(y), tgt, cal["rel_tol"], cal["max_iter"])
        r["history"] = [[float(c), float(v)] for c, v in r.get("history", [])]
        res[int(y)] = r
        log(f"{area} {y}: c = {r['c']:.4g}, share {100 * r['achieved']:.3f} % vs target {100 * tgt:.3f} % "
            f"({r['iterations']} it., {r['status']})")
    meta = {"n_virtual_nodes": len(prob.fleet), "n_free": int(prob.fleet["free"].sum()),
            "B_mean": float(prob.active["B"].mean()) if len(prob.active) else 0.0,
            "n_parks": int(profiles.shape[0])}
    return {"years": res, "fleet": meta}


# ---------------------------------------------------------------- market

def fleet_classes(mastr: pd.DataFrame, cohorts: list, extract_date: str, extrapolate_to) -> pd.DataFrame:
    """Unit capacity per (cohort, rule, post-EEG start) class and commissioning day.
    Units commissioned after the extract date are extrapolated with the mean
    commissioning rate of the 12 months before it (newest cohort)."""
    u = mastr[["Inbetriebnahmedatum", "Nettonennleistung"]].dropna().copy()
    u["date"] = pd.to_datetime(u["Inbetriebnahmedatum"]).dt.normalize()
    u = u[u["date"] <= pd.Timestamp(extract_date)]
    recs = []
    for (d, kw), grp in u.groupby(["date", "Nettonennleistung"]):
        c = market.cohort_of(d, kw, cohorts)
        recs.append({"date": d, "mw": kw * len(grp) / 1000.0, "cohort": c["cohort"], "rule_h": c["rule_h"],
                     "strike_ct": c["strike_ct"], "post_eeg_from": pd.Timestamp(c["post_eeg_from"])})
    df = pd.DataFrame(recs)
    ext0 = pd.Timestamp(extract_date)
    rate = df.loc[df["date"] > ext0 - pd.DateOffset(years=1), "mw"].sum() / 365.0      # MW per day
    days = pd.date_range(ext0 + pd.Timedelta(days=1), pd.Timestamp(extrapolate_to).normalize(), freq="D")
    if len(days):
        c = market.cohort_of(days[0], 1e9, cohorts)
        df = pd.concat([df, pd.DataFrame({"date": days, "mw": rate, "cohort": c["cohort"], "rule_h": c["rule_h"],
                                          "strike_ct": c["strike_ct"],
                                          "post_eeg_from": pd.Timestamp(c["post_eeg_from"])})], ignore_index=True)
    df["rule_h"] = df["rule_h"].astype(object).where(df["rule_h"].notna(), None)
    return df


class MarketProblem:
    """Exposure aggregates of the virtual market fleet (independent of theta)."""

    def __init__(self, cfg: dict, qidx: pd.DatetimeIndex, drv: pd.DataFrame, classes: pd.DataFrame):
        mk = cfg["market"]
        self.cfg = cfg
        price, rm, cf = drv["price"].to_numpy(), drv["r_m"].to_numpy(), drv["cf_da"].to_numpy()
        bid, starts, lens = market.neg_blocks(price)
        blen = np.where(bid >= 0, lens[np.maximum(bid, 0)], 0.0)
        neg = bid >= 0
        years = timegrid.utc_years(qidx)
        band = np.full(len(price), "", dtype=object)
        for name, (lo, hi) in market.BAND_EDGES.items():
            band[(price >= lo) & (price < hi)] = name
        self.years_all = sorted(np.unique(years[neg]))
        self.bands = list(market.BAND_EDGES)
        ni = np.flatnonzero(neg)
        qn = qidx[ni]
        day = qn.tz_convert(timegrid.TZ_LOCAL).tz_localize(None).normalize()
        keys = classes.groupby(["cohort", classes["rule_h"].astype(str), "post_eeg_from", "strike_ct"])
        pot_neg = np.zeros(len(ni))
        X, M, Y, Bd, L = [], [], [], [], []          # exposed potential, margin*pot, year, band, block length
        blk = []
        for (cohort, rule, post, strike), g in keys:
            cap_by_day = g.groupby("date")["mw"].sum().sort_index().cumsum()
            cap = np.interp(day.asi8.astype(float), cap_by_day.index.asi8.astype(float), cap_by_day.values,
                            left=0.0)
            pot = cap * cf[ni]
            pot_neg += pot
            coh = {"strike_ct": strike, "rule_h": None if rule == "None" else float(rule),
                   "post_eeg_from": post.date()}
            pstar = market.threshold(qn, price[ni], rm[ni], blen[ni], coh, mk["post_eeg"])
            ex = price[ni] < pstar
            if not ex.any():
                continue
            X.append(pot[ex])
            M.append(pot[ex] * (pstar[ex] - price[ni][ex]))
            Y.append(years[ni][ex])
            Bd.append(band[ni][ex])
            L.append(blen[ni][ex])
            blk.append(np.char.add(f"{cohort}|{rule}|{post.date()}|", bid[ni][ex].astype(str)))
        self.pot_neg_year = pd.Series(pot_neg).groupby(years[ni]).sum()
        self.pot_band = pd.Series(pot_neg).groupby(band[ni]).sum()
        self.pot_len = pd.Series(pot_neg).groupby([years[ni], _len_cat(blen[ni])]).sum()
        ex = pd.DataFrame({"x": np.concatenate(X), "mx": np.concatenate(M), "year": np.concatenate(Y),
                           "band": np.concatenate(Bd), "len": _len_cat(np.concatenate(L)),
                           "blk": np.concatenate(blk)})
        g = ex.groupby("blk")
        self.mbar = (g["mx"].sum() / g["x"].sum()).where(g["x"].sum() > 0, 0.0)
        ex["code"] = pd.Categorical(ex["blk"], categories=self.mbar.index).codes
        self.ex = ex
        self.x_year = ex.groupby("year")["x"].sum()

    def rates(self, theta0: float, theta1: float, tau: float) -> dict:
        R = market.expected_response(self.mbar.to_numpy(), theta0, theta1, tau)
        cur = self.ex["x"].to_numpy() * R[self.ex["code"].to_numpy()]
        ex = self.ex.assign(cur=cur)
        cy = ex.groupby("year")["cur"].sum()
        cb = ex.groupby("band")["cur"].sum()
        cl = ex.groupby(["year", "len"])["cur"].sum()
        return {
            "exposed_response": (cy / self.x_year).to_dict(),
            "band_rate": (cb / self.pot_band).reindex(self.bands).to_dict(),
            "fleet_rate_neg": (cy / self.pot_neg_year).to_dict(),
            "exposure_share": (self.x_year / self.pot_neg_year).to_dict(),
            "by_block_length": (cl / self.pot_len).dropna().to_dict(),
        }

    def fit(self, theta_start: tuple, tau: float) -> dict:
        r = self.cfg["market"]["response"]
        ty, tb = r["target_exposed_response"], r["target_band_rate"]
        years = [y for y in self.years_all if int(y) in ty]

        def resid(th):
            rt = self.rates(th[0], th[1], tau)
            return np.r_[[rt["exposed_response"][y] - ty[int(y)] for y in years],
                         [rt["band_rate"][b] - tb[b] for b in tb]]

        sol = least_squares(resid, np.asarray(theta_start, float), method="lm")
        rt = self.rates(sol.x[0], sol.x[1], tau)
        res = resid(sol.x)
        return {"theta0": float(sol.x[0]), "theta1": float(sol.x[1]), "tau": float(tau),
                "rmse": float(np.sqrt(np.mean(res ** 2))), "residuals": [float(v) for v in res],
                "targets": {"exposed_response": {int(y): ty[int(y)] for y in years}, "band_rate": tb},
                "achieved": _jsonable(rt), "success": bool(sol.success), "nfev": int(sol.nfev)}


def _len_cat(h) -> np.ndarray:
    h = np.asarray(h, float)
    return np.where(h < 4, "1-3h", np.where(h < 6, "4-5h", ">=6h"))


def _jsonable(d):
    if isinstance(d, dict):
        return {(str(k) if not isinstance(k, tuple) else "|".join(map(str, k))): _jsonable(v) for k, v in d.items()}
    if isinstance(d, (np.floating, np.integer)):
        return d.item()
    return d


# ---------------------------------------------------------------- validation of the virtual grid fleet

def fleet_detail(prob: AreaGridProblem, c_by_year: dict, cf_da: np.ndarray, cf_bins, start_bins) -> dict:
    """Statistics of the calibrated virtual fleet of one area (quarter-hour resolution):
    curtailed hours per affected node and year, depth-weighted share of the ever-affected
    nodes by CF_DA bin, node start rate by CF_DA bin, setpoint shares by curtailed time,
    monthly depth-weighted share."""
    prob.c = {int(y): float(v) for y, v in c_by_year.items()}
    cf_da = np.asarray(cf_da)
    months = prob.qidx.month.to_numpy()
    hours, depth_sum, ever = [], {}, np.zeros(len(prob.active), bool)
    dep_bin = np.zeros(len(cf_bins))
    n_bin = np.zeros(len(cf_bins))
    st_bin = np.zeros(len(start_bins))
    ns_bin = np.zeros(len(start_bins))
    sp = np.zeros(3)
    mon = np.zeros(13)
    mon_n = np.zeros(13)
    dep_parts = []
    for y in sorted(prob.c):
        ws, ys, ye = prob.window(y)
        cd = prob.candidates(y, max(64.0, 8 * prob.c[y]))
        c_slot = np.full(len(cd["pos"]), prob.c[y])
        for yy, cv in prob.c.items():
            c_slot[cd["slot_year"] == yy] = cv
        on = cd["cstar"] < c_slot
        s = grid.segments_covering(ye - ws, cd["pos"][on], cd["n_main"][on], cd["s0"][on], cd["n_rel"][on],
                                   cd["s_rel"][on], row=cd["row"][on], n_rows=len(prob.active))[:, ys - ws:]
        cur = s < 1
        h = cur.sum(axis=1) * timegrid.STEP_H
        hours.append(pd.DataFrame({"year": y, "node": np.arange(len(prob.active)), "hours": h}))
        ever |= cur.any(axis=1)
        d = (1 - s)
        dep_parts.append(d)
        for k, v in enumerate((0.0, 0.3, 0.6)):
            sp[k] += np.isclose(s[cur], v).sum()
        # starts inside the year
        own = on & (cd["slot_year"] == y)
        cf_s = cf_da[ws + cd["pos"][own]]
        cf_y = cf_da[ys:ye]
        for k, (lo, hi) in enumerate(start_bins):
            st_bin[k] += ((cf_s >= lo) & (cf_s < hi)).sum()
            ns_bin[k] += ((cf_y >= lo) & (cf_y < hi)).sum() * timegrid.STEP_H * len(prob.active)
    dep = np.concatenate(dep_parts, axis=1)[ever]
    cf_all = np.concatenate([cf_da[prob.window(y)[1]:prob.window(y)[2]] for y in sorted(prob.c)])
    mo_all = np.concatenate([months[prob.window(y)[1]:prob.window(y)[2]] for y in sorted(prob.c)])
    col = dep.mean(axis=0) if len(dep) else np.zeros(len(cf_all))
    for k, (lo, hi) in enumerate(cf_bins):
        m = (cf_all >= lo) & (cf_all < hi)
        dep_bin[k], n_bin[k] = col[m].sum(), m.sum()
    for mth in range(1, 13):
        m = mo_all == mth
        mon[mth], mon_n[mth] = col[m].sum(), m.sum()
    hrs = pd.concat(hours, ignore_index=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return {"hours": hrs, "n_active": len(prob.active), "n_ever": int(ever.sum()),
                "coupling_pct": list(100 * dep_bin / n_bin),
                "node_start_rate": list(st_bin / ns_bin),
                "setpoint_shares_pct": list(100 * sp / sp.sum()) if sp.sum() else [np.nan] * 3,
                "monthly_depth_pct": {m: float(100 * mon[m] / mon_n[m]) for m in range(1, 13) if mon_n[m]},
                "B_mean_theory": prob.acfg["beta"][0] / sum(prob.acfg["beta"])}
