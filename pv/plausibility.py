"""Estimate a park's effective orientation from measured power and check the master data against it.

Three estimators on clear hours (clear-sky index >= kc_min, sun elevation >= elev_min, no clipping / suspect /
snow hours), all built on the POA bank of pv.angles:
  azimuth_from_daily_centroid   per clear day the sun azimuth at the half-energy time; inverted through the
                                bank (monotone, damped by the diffuse share) -> daily azimuth estimates
  tilt_from_noon_ratio          P_noon / BHI_noon = s (cos b + sin b tan z) on clear noons -> b = atan(slope /
                                intercept), the scale s cancels; needs a spread of noon zenith angles (seasons)
  grid_region                   profile-likelihood region {(b, g): R2 >= R2max - delta} on the bank, plus a
                                bootstrap over clear days for the point estimate
and the decision: master-data classes (8 azimuth octants, tilt classes) are implausible when they lie outside
the confidence region. Everything returns effective angles (terrain-following rows make them a mixture).

Real meters need two more masks (suspect_mask, clipping_mask): hours far below the fitted clear-sky shape
(curtailment, outages, partial availability) and hours at the AC limit distort the grid fit - on a metered
east-west park a year of data drifted to 0 deg / 240 deg without them. The tilt is only identifiable when the
clear noons span a range of zenith angles (seasons); a summer month alone fits 5-10 deg for a 25 deg park with
a deceptively narrow day-bootstrap interval, so the tilt verdict is withheld below TILT_ZENITH_SPREAD."""
import numpy as np
import pandas as pd

from pv import angles, arrays as arr, timebase

TILTS = np.arange(0.0, 61.0, 5.0)
AZIMUTHS = np.arange(60.0, 301.0, 10.0)
EW_AZIMUTHS = [(60.0, 240.0), (70.0, 250.0), (80.0, 260.0), (90.0, 270.0), (100.0, 280.0), (110.0, 290.0), (120.0, 300.0)]
EW_SHARES = (0.3, 0.4, 0.5, 0.6, 0.7)
EW_BIC_MARGIN = 10.0           # the two-plane model must beat the single plane by this much BIC
EW_MIN_TILT = 10.0             # below this tilt an east-west pair is indistinguishable from a flat plane
TILT_ZENITH_SPREAD = 15.0      # deg of clear-noon zenith range needed before a tilt verdict is given
SUSPECT_FRACTION = 0.5         # hours below this fraction of the fitted clear-sky shape are suspect
CLIP_FRACTION = 0.97           # hours at or above this fraction of the AC limit are clipped


def clear_mask(drv: pd.DataFrame, lat: float, lon: float, alt: float, exclude: pd.Series = None, kc_min: float = 0.8,
               elev_min: float = 10.0) -> pd.Series:
    m = angles.clear_hours(drv["ghi"], lat, lon, alt, kc_min, elev_min)
    if exclude is not None:
        m &= ~exclude.reindex(m.index).fillna(False).astype(bool)
    return m


def chain_bank(cfg: dict, drv: pd.DataFrame, tilts, azimuths, horizon: pd.Series = None, gcr: float = None) -> dict:
    """{(tilt, az): p_dc} from the production chain (pv.chain.run_park with one array per pair, return_arrays):
    IAM, cell temperature with the park's module gamma, inter-row shading at the park's GCR, horizon. Aging and
    soiling are off (constant factors drop out of the scale fit). drv needs ghi/dhi; temp_air, wind_speed and
    albedo are filled with defaults when missing. cfg is the park config (module, temperature, shading, ...)."""
    from pv import chain, config as pvconfig
    d = drv.copy()
    for k, v in (("temp_air", 12.0), ("wind_speed", 2.0), ("albedo", 0.2), ("imputed_share", 0.0)):
        if k not in d:
            d[k] = v
    g = gcr if gcr is not None else next((a.get("gcr") for a in cfg.get("arrays", []) if a.get("gcr")), None)
    pairs = [(float(t), float(a)) for t in tilts for a in azimuths]
    c = {k: v for k, v in cfg.items() if k not in ("arrays", "aging", "soiling")}
    c["arrays"] = [{"name": f"b{j}", "tilt": t, "azimuth": a, "share": 1.0 / len(pairs), "gcr": g} for j, (t, a) in enumerate(pairs)]
    c["aging"] = {"enabled": False, "commissioning_date": None, "era": None}
    c["soiling"] = {"enabled": False}
    c.setdefault("transposition", {})
    if not isinstance(c["transposition"].get("albedo", 0.2), (int, float)):
        c["transposition"] = dict(c["transposition"], albedo="era5")
    res = chain.run_park(pvconfig.normalise(c), d, horizon=horizon, return_arrays=True)
    per = res.attrs["arrays"]
    return {(t, a): per[f"b{j}"]["p_dc_w"] for j, (t, a) in enumerate(pairs)}


def azimuth_prior_from_rows(row_axis_deg: float) -> dict:
    """Module azimuth implied by a mapped table-row axis (deg, 0-180): east-west rows carry south-facing modules
    (perpendicular on the south side), north-south rows an east-west pair. Same convention as parks_pv.geometry."""
    ax = float(row_axis_deg) % 180.0
    if ax < 30.0 or ax > 150.0:
        az_e, az_w = (ax + 90.0) % 360.0, (ax + 270.0) % 360.0
        if az_e > 180.0:
            az_e, az_w = az_w, az_e
        return {"east_west": True, "azimuth": None, "azimuths": (az_e, az_w)}
    return {"east_west": False, "azimuth": (ax + 90.0) % 360.0, "azimuths": None}


def _fit(y: np.ndarray, x: np.ndarray) -> tuple:
    s = float(np.dot(x, y) / np.dot(x, x)) if np.dot(x, x) > 0 else 0.0
    rss = float(np.sum((y - s * x) ** 2))
    return s, rss


def east_west_hypothesis(power: pd.Series, bank: dict, mask: pd.Series, pairs=None, shares=EW_SHARES, min_tilt: float = EW_MIN_TILT) -> dict:
    """Explicit two-plane model p = s (w POA(t, az_e) + (1 - w) POA(t, az_w)) over the bank tilts >= min_tilt and the
    azimuth pairs, against the best single plane, compared by BIC (4 vs 3 parameters). preferred is True when the
    pair beats the single plane by EW_BIC_MARGIN. Pairs default to the bank azimuths 180 deg apart (EW_AZIMUTHS)."""
    m = (mask & power.notna()).values
    y = power.values[m]
    n = int(m.sum())
    if n < 24:
        return {"preferred": False, "n": n}
    tilts = sorted({t for (t, a) in bank})
    azs = {a for (t, a) in bank}
    pairs = pairs or [p for p in EW_AZIMUTHS if p[0] in azs and p[1] in azs]
    best1 = None
    for (t, a), s in bank.items():
        sc, rss = _fit(y, s.values[m])
        if best1 is None or rss < best1[2]:
            best1 = (t, a, rss)
    best2 = None
    for t in tilts:
        if t < min_tilt:
            continue
        for ae, aw in pairs:
            if (t, ae) not in bank or (t, aw) not in bank:
                continue
            xe, xw = bank[(t, ae)].values[m], bank[(t, aw)].values[m]
            for w in shares:
                sc, rss = _fit(y, w * xe + (1.0 - w) * xw)
                if best2 is None or rss < best2[4]:
                    best2 = (t, ae, aw, w, rss, sc)
    tss = float(np.sum((y - y.mean()) ** 2))
    if best1 is None or best2 is None or best1[2] <= 0:
        return {"preferred": False, "n": n}
    bic1 = n * np.log(best1[2] / n) + 3 * np.log(n)
    bic2 = n * np.log(best2[4] / n) + 4 * np.log(n)
    return {"preferred": bool(bic2 + EW_BIC_MARGIN < bic1), "n": n, "tilt": best2[0], "azimuth_east": best2[1], "azimuth_west": best2[2],
            "share_east": best2[3], "r2": 1.0 - best2[4] / tss if tss > 0 else np.nan, "r2_single": 1.0 - best1[2] / tss if tss > 0 else np.nan,
            "single": (best1[0], best1[1]), "delta_bic": float(bic2 - bic1)}


def clipping_mask(power: pd.Series, ac_limit_w: float = None, frac: float = CLIP_FRACTION, q: float = 0.999) -> pd.Series:
    """Hours at the AC limit (given, else the q-quantile of the series as a proxy)."""
    lim = float(ac_limit_w) if ac_limit_w is not None and np.isfinite(ac_limit_w) else float(power.quantile(q))
    return (power >= frac * lim).fillna(False)


def suspect_mask(power: pd.Series, bank: dict, mask: pd.Series, frac: float = SUSPECT_FRACTION, iterations: int = 2) -> pd.Series:
    """Clear hours whose power is below frac x the scaled best-pair POA (curtailment, outages, partial
    availability). The best pair is refitted without the suspects once (iterations)."""
    sus = pd.Series(False, index=power.index)
    m = mask & power.notna()
    for _ in range(iterations):
        g = angles.grid_fit(power, bank, m & ~sus)
        if len(g) == 0 or not np.isfinite(g.iloc[0].r2):
            break
        b = g.iloc[0]
        new = (m & (power < frac * b.scale * bank[(b.tilt, b.azimuth)])).fillna(False)
        if new.equals(sus):
            break
        sus = new
    return sus


def noon_zenith_spread(mask: pd.Series, zenith: pd.Series) -> float:
    """Range (deg) of the true noon zenith over the days whose noon hour (or a neighbour) is in the mask: the
    seasonal lever that identifies the tilt. Days with only morning or evening hours in the mask do not count."""
    noon = zenith.groupby(zenith.index.date).idxmin().dropna()
    m = mask.reindex(zenith.index).fillna(False).astype(bool).values
    pos = pd.Series(np.arange(len(zenith)), index=zenith.index).loc[noon.tolist()].values
    keep = [float(zenith.iloc[i]) for i in pos if m[max(i - 1, 0):i + 2].any()]
    return float(np.ptp(keep)) if len(keep) >= 2 else 0.0


def half_energy_azimuth(power: pd.Series, sun_az: pd.Series, day_mask: pd.Series, min_hours: int = 6) -> pd.Series:
    """Per day with >= min_hours valid clear hours: sun azimuth (deg) at the time the cumulative energy reaches
    50 % (linear interpolation between hours)."""
    out = {}
    p = power.where(day_mask)
    for d, g in p.groupby(p.index.date):
        g = g.dropna()
        if len(g) < min_hours or g.sum() <= 0:
            continue
        c = g.cumsum() / g.sum()
        az = sun_az.reindex(g.index).values
        j = int(np.searchsorted(c.values, 0.5))
        if j == 0:
            out[d] = float(az[0])
            continue
        f = (0.5 - c.values[j - 1]) / max(c.values[j] - c.values[j - 1], 1e-9)
        out[d] = float(az[j - 1] + f * (az[j] - az[j - 1]))
    return pd.Series(out, dtype=float)


def azimuth_from_daily_centroid(power: pd.Series, bank: dict, sun_az: pd.Series, mask: pd.Series, tilt_guess: float = 25.0) -> dict:
    """Daily half-energy azimuths of the measurement, mapped to module azimuth through the same statistic of the
    bank POA at tilt_guess (lookup, monotone in azimuth). Returns median, IQR and the daily series."""
    obs = half_energy_azimuth(power, sun_az, mask)
    if obs.empty:
        return {"azimuth": np.nan, "iqr": np.nan, "n_days": 0}
    cal = {}
    for (t, a), poa in bank.items():
        if abs(t - tilt_guess) < 1e-6:
            cal[a] = half_energy_azimuth(poa, sun_az, mask)
    cal = pd.DataFrame(cal).reindex(obs.index)                    # days x module azimuths
    est = []
    for d in obs.index:
        row = cal.loc[d].dropna()
        if len(row) < 3:
            continue
        xs, ys = row.values, row.index.values.astype(float)      # bank statistic -> module azimuth
        o = np.argsort(xs)
        est.append(float(np.interp(obs[d], xs[o], ys[o])))
    est = pd.Series(est)
    return {"azimuth": float(est.median()), "iqr": float(est.quantile(0.75) - est.quantile(0.25)), "n_days": int(len(est)), "daily": est}


def tilt_from_noon_ratio(power: pd.Series, drv: pd.DataFrame, lat: float, lon: float, alt: float, mask: pd.Series,
                         azimuth: float = 180.0, albedo: float = 0.2, iterations: int = 2, gamma: float = None, max_zenith: float = 65.0) -> dict:
    """Clear-noon regression P/BHI = a + b tan(z) after removing the diffuse and ground-reflected POA share for
    the current tilt guess -> tilt = atan(b / a). Needs noon zenith spread (>= 10 deg) to be identifiable.
    Low-sun noons (zenith > max_zenith) are dropped: inter-row shading, IAM and horizon losses at low sun flatten
    the seasonal slope and bias the tilt low (tested on synthetic parks: 30 deg true -> 19 deg with all noons).
    Optional temperature correction (gamma, T_cell ~ T_air + 0.03 GHI) when drv has temp_air."""
    if gamma is not None and "temp_air" in drv:
        t_cell = drv["temp_air"].fillna(12.0) + 0.03 * drv["ghi"].fillna(0.0)
        power = power / (1.0 + gamma * (t_cell.reindex(power.index) - 25.0))
    sp = timebase.solar_position(drv.index, lat, lon, alt)
    zen = sp["apparent_zenith"]
    cs = timebase.clearsky(drv.index, lat, lon, alt)
    from pv import transposition
    dni = transposition.dni_from_closure(drv["ghi"], drv["dhi"], zen, cs["dni"])
    bhi = (dni * np.cos(np.radians(zen))).clip(lower=0)
    dayz = zen.where(mask & (zen <= max_zenith)).groupby(zen.index.date).idxmin().dropna()   # clear hour nearest noon per day
    idx = pd.DatetimeIndex(list(dayz.values)).tz_localize("UTC") if dayz.size and getattr(pd.DatetimeIndex(list(dayz.values)), "tz", None) is None else pd.DatetimeIndex(list(dayz.values))
    if len(idx) < 5:
        return {"tilt": np.nan, "n_days": int(len(idx)), "zenith_spread": 0.0}
    y = power.reindex(idx)
    tilt = 25.0
    out = {}
    for _ in range(iterations):
        # diffuse + ground share at the current tilt, same scale as the beam term
        diff = drv["dhi"].reindex(idx) * (1 + np.cos(np.radians(tilt))) / 2 + albedo * drv["ghi"].reindex(idx) * (1 - np.cos(np.radians(tilt))) / 2
        # P = s * (BHI * cosAOI/cos z + diff) -> P / BHI = s * (cos b + sin b tan z) + s * diff / BHI ; fit a, b, c
        tz = np.tan(np.radians(zen.reindex(idx)))
        X = np.column_stack([np.ones(len(idx)), tz.values, (diff / bhi.reindex(idx)).values])
        ok = np.isfinite(X).all(1) & y.notna().values & (bhi.reindex(idx).values > 50)
        if ok.sum() < 5:
            return {"tilt": np.nan, "n_days": int(ok.sum()), "zenith_spread": float(np.ptp(zen.reindex(idx)[ok])) if ok.any() else 0.0}
        coef, *_ = np.linalg.lstsq(X[ok], (y / bhi.reindex(idx)).values[ok], rcond=None)
        a, b = coef[0], coef[1]
        tilt = float(np.degrees(np.arctan2(b, a))) if a > 0 else np.nan
        out = {"tilt": tilt, "scale": float(np.hypot(a, b)), "n_days": int(ok.sum()), "zenith_spread": float(np.ptp(zen.reindex(idx)[ok])),
               "r2_fit": float(1 - np.sum((X[ok] @ coef - (y / bhi.reindex(idx)).values[ok]) ** 2) / max(np.var((y / bhi.reindex(idx)).values[ok]) * ok.sum(), 1e-12))}
        if not np.isfinite(tilt):
            break
    return out


def grid_region(power: pd.Series, bank: dict, mask: pd.Series, delta: float = 0.005, n_boot: int = 100, seed: int = 0) -> dict:
    """Best (tilt, azimuth) on the bank, the profile-likelihood region R2 >= R2max - delta (as ranges), and a
    bootstrap over clear days (median / 5-95 % of the best pair)."""
    g = angles.grid_fit(power, bank, mask)
    if len(g) == 0 or not np.isfinite(g.iloc[0].r2):
        return {"tilt": np.nan, "azimuth": np.nan, "r2": np.nan, "scale": np.nan, "tilt_range": (np.nan, np.nan), "azimuth_range": (np.nan, np.nan),
                "n_hours": int((mask & power.notna()).sum()), "n_days": 0}
    best = g.iloc[0]
    reg = g[g.r2 >= best.r2 - delta]
    out = {"tilt": float(best.tilt), "azimuth": float(best.azimuth), "r2": float(best.r2), "scale": float(best.scale),
           "tilt_range": (float(reg.tilt.min()), float(reg.tilt.max())), "azimuth_range": (float(reg.azimuth.min()), float(reg.azimuth.max())),
           "n_hours": int((mask & power.notna()).sum())}
    days = np.array(sorted(set(power.index[mask & power.notna()].date)))
    if n_boot and len(days) >= 5:
        rng = np.random.default_rng(seed)
        bt, ba = [], []
        for _ in range(n_boot):
            pick = set(rng.choice(days, len(days), replace=True))
            m = mask & pd.Series(np.isin(power.index.date, list(pick)), index=power.index)
            gb = angles.grid_fit(power, bank, m)
            if len(gb):
                bt.append(gb.iloc[0].tilt); ba.append(gb.iloc[0].azimuth)
        out.update({"tilt_boot": (float(np.percentile(bt, 5)), float(np.median(bt)), float(np.percentile(bt, 95))),
                    "azimuth_boot": (float(np.percentile(ba, 5)), float(np.median(ba)), float(np.percentile(ba, 95))), "n_days": int(len(days))})
    return out


FLAT_TILT = 2.5                # bank components at or below this tilt have no azimuth
FLAT_WEIGHT = 0.5              # mixture weight on flat components that makes the fit "flat"


def mixture(power: pd.Series, bank: dict, mask: pd.Series, min_side: float = 0.25, max_south: float = 0.3) -> dict:
    """NNLS mixture over the bank. east_west is flagged only for a genuine two-sided layout: at least min_side of
    the weight on each of the east (<= 120 deg) and west (>= 240 deg) sides and at most max_south between 135
    and 225 deg - small leaks of weight to the sides (terrain spread, noise) do not count. Components at tilt
    <= FLAT_TILT carry no azimuth (the bank series are identical for every azimuth there) and are counted as
    w_flat, never as south: a flat east-west park is indistinguishable from a flat south park by power alone."""
    nn = angles.nnls_mixture(power, bank, mask)
    wall = nn["weights"]
    az = np.array([a for (t, a) in wall.index])
    flat = np.array([t <= FLAT_TILT for (t, a) in wall.index])
    w_f = float(wall.values[flat].sum())
    w_e, w_w = float(wall.values[(az <= 120) & ~flat].sum()), float(wall.values[(az >= 240) & ~flat].sum())
    w_s = float(wall.values[(az >= 135) & (az <= 225) & ~flat].sum())
    w = wall[wall >= 0.1]
    return {"n_components": int(len(w)), "components": [(float(t), float(a), float(x)) for (t, a), x in w.items()],
            "tilt_eff": nn["tilt_eff"], "azimuth_eff": nn["azimuth_eff"], "r2": nn["r2"], "w_east": w_e, "w_west": w_w, "w_south": w_s,
            "w_flat": w_f, "flat": bool(w_f >= FLAT_WEIGHT),
            "east_west": bool(w_e >= min_side and w_w >= min_side and w_s <= max_south)}


def orientation_class(az: float) -> str:
    names = ["Nord", "Nord-Ost", "Ost", "Süd-Ost", "Süd", "Süd-West", "West", "Nord-West"]
    return names[int(((az + 22.5) % 360.0) // 45.0)]


def tilt_class(t: float) -> str:
    return "< 20 Grad" if t < 20 else "20 - 40 Grad" if t < 40 else "40 - 60 Grad" if t < 60 else "> 60 Grad"


def plausibility(region: dict, mix: dict, register_orientation: str, register_tilt_class: str, use_bootstrap: bool = True,
                 az_margin: float = 45.0, tilt_margin: float = 5.0, tilt_identifiable: bool = True,
                 azimuth_prior: float = None, prior_east_west: bool = False) -> dict:
    """Compare the register classes with the estimate. Azimuth: implausible when the whole bootstrap 5-95 %
    interval of the best azimuth (fallback: profile region) lies more than az_margin (one octant) from the
    registered class centre, or when the mixture says east-west and the register does not (and vice versa).
    A flat fit (mixture weight on tilt <= FLAT_TILT, or the whole tilt interval <= 5 deg) has no azimuth: the
    azimuth verdict is None, except that it is consistent with a registered east-west layout (flat east-west
    rows are flat). Tilt: implausible when the interval, widened by tilt_margin, misses the registered class; with
    tilt_identifiable False (too little noon-zenith spread) the tilt verdict is None and the tilt interval is
    the profile-likelihood region, not the day bootstrap.
    With azimuth_prior (mapped rows / orthophoto) the azimuth verdict is the prior's alone - east-west rows against a
    single-orientation register, or a row azimuth more than az_margin from the registered class centre - and the
    power decides only the tilt (separate methods for orientation and tilt)."""
    from parks_pv.geometry import AZ_CENTRE
    if use_bootstrap and "azimuth_boot" in region:
        az_lo, az_hi = region["azimuth_boot"][0], region["azimuth_boot"][2]
        t_lo, t_hi = region["tilt_boot"][0], region["tilt_boot"][2]
    else:
        (az_lo, az_hi), (t_lo, t_hi) = region["azimuth_range"], region["tilt_range"]
    if not tilt_identifiable:
        t_lo, t_hi = region["tilt_range"]
    if not (np.isfinite(az_lo) and np.isfinite(t_lo)):
        return {"azimuth_ok": None, "tilt_ok": None, "estimated_azimuth_classes": [], "estimated_tilt_classes": []}
    az_classes = {orientation_class(a) for a in np.arange(az_lo, az_hi + 0.1, 5.0)}
    t_classes = {tilt_class(t) for t in np.arange(max(t_lo - tilt_margin, 0.0), t_hi + tilt_margin + 0.1, 2.5)}
    flat = bool(mix.get("flat", False)) or (np.isfinite(t_hi) and t_hi <= 5.0)
    if azimuth_prior is not None:
        flat = False
        if prior_east_west or register_orientation == "Ost-West":
            az_ok = bool(prior_east_west) == (register_orientation == "Ost-West")
        elif register_orientation in AZ_CENTRE:
            az_ok = abs(((float(azimuth_prior) - AZ_CENTRE[register_orientation]) + 180.0) % 360.0 - 180.0) <= az_margin
        else:
            az_ok = True
        az_lo = az_hi = float(azimuth_prior)
    elif mix["east_west"]:
        az_ok = register_orientation == "Ost-West"
    elif flat:
        az_ok = True if register_orientation == "Ost-West" else None
    elif register_orientation == "Ost-West":
        az_ok = False
    elif register_orientation in AZ_CENTRE:
        c = AZ_CENTRE[register_orientation]
        dist = min(abs(((a - c) + 180.0) % 360.0 - 180.0) for a in np.arange(az_lo, az_hi + 0.1, 2.5))
        az_ok = dist <= az_margin
    else:
        az_ok = True
    return {"azimuth_ok": None if az_ok is None else bool(az_ok), "tilt_ok": (register_tilt_class in t_classes) if tilt_identifiable else None,
            "flat": flat, "estimated_azimuth_classes": sorted(({"flach"} if flat else az_classes) | ({"Ost-West"} if mix["east_west"] else set())),
            "estimated_tilt_classes": sorted(t_classes)}


def assess(power: pd.Series, drv: pd.DataFrame, lat: float, lon: float, alt: float, exclude: pd.Series = None,
           register_orientation: str = None, register_tilt_class: str = None, horizon: pd.Series = None, delta: float = 0.005,
           n_boot: int = 50, ac_limit_w: float = None, robust: bool = True, zenith_spread_min: float = TILT_ZENITH_SPREAD,
           bank: dict = None, cfg: dict = None, azimuth_prior: float = None, prior_east_west: bool = False,
           ew_hypothesis: bool = True) -> dict:
    """Full assessment of one park on a driver frame (ghi, dhi) and a power series on the same hourly index.
    Masks, in order: clear hours, caller's exclude (snow, imputed, known outages), clipping at ac_limit_w
    (None: 99.9 % quantile proxy), then the suspect refit (robust). tilt_identifiable requires the clear noons
    to span >= zenith_spread_min deg of zenith.
    Bank: a precomputed {(tilt, az): series} (sliced to power.index), else the production chain when cfg is given
    (chain_bank: IAM, module gamma, inter-row shading), else the plain POA bank. With azimuth_prior (deg, from
    mapped rows / orthophoto) the bank is restricted to that azimuth - or, with prior_east_west, to the pair
    azimuth_prior and azimuth_prior + 180 - and only the tilt (and east/west share) is estimated from power.
    ew_hypothesis adds the explicit two-plane fit with BIC model choice; its verdict ORs into mixture.east_west."""
    clip = clipping_mask(power, ac_limit_w) if (robust or ac_limit_w is not None) else pd.Series(False, index=power.index)
    excl = clip.copy()
    if exclude is not None:
        excl |= exclude.reindex(power.index).fillna(False).astype(bool)
    mask = clear_mask(drv, lat, lon, alt, excl)
    if azimuth_prior is not None:
        azs = [float(azimuth_prior) % 360.0, (float(azimuth_prior) + 180.0) % 360.0] if prior_east_west else [float(azimuth_prior) % 360.0]
    else:
        azs = list(AZIMUTHS)
    if bank is not None:
        bank_src = "precomputed"
        bank = {k: v.reindex(power.index) for k, v in bank.items() if azimuth_prior is None or k[1] in azs}
        if azimuth_prior is not None and not bank:
            raise ValueError("precomputed bank lacks the prior azimuth(s); build it with those azimuths")
    elif cfg is not None:
        bank_src = "chain"
        bank = chain_bank(cfg, drv, TILTS, azs, horizon=horizon)
    else:
        bank_src = "poa"
        bank = angles.poa_bank(drv, lat, lon, alt, TILTS, azs, horizon=horizon)
    sus = suspect_mask(power, bank, mask) if robust else pd.Series(False, index=power.index)
    n_clear_raw = int(mask.sum())
    mask = mask & ~sus
    sp = timebase.solar_position(drv.index, lat, lon, alt)
    spread = noon_zenith_spread(mask & power.notna(), sp["apparent_zenith"])
    identifiable = bool(spread >= zenith_spread_min)
    reg = grid_region(power, bank, mask, delta, n_boot)
    tg = float(round(reg["tilt"] / 5) * 5) if np.isfinite(reg["tilt"]) else 25.0
    cen = azimuth_from_daily_centroid(power, bank, sp["azimuth"], mask, tilt_guess=tg)
    noon = tilt_from_noon_ratio(power, drv, lat, lon, alt, mask, azimuth=reg["azimuth"] if np.isfinite(reg["azimuth"]) else 180.0)
    mix = mixture(power, bank, mask)
    ewh = {"preferred": False, "n": 0}
    if ew_hypothesis:
        pairs = [tuple(azs)] if (azimuth_prior is not None and prior_east_west) else None
        ewh = east_west_hypothesis(power, bank, mask, pairs=pairs) if (azimuth_prior is None or prior_east_west) else ewh
        mix["east_west_nnls"] = mix["east_west"]
        mix["east_west"] = bool(mix["east_west"] or ewh["preferred"])
    out = {"n_clear_hours": int(mask.sum()), "n_clear_before_suspect": n_clear_raw, "n_suspect": int(sus.sum()),
           "n_clipped_clear": int((clip & clear_mask(drv, lat, lon, alt, exclude)).sum()), "zenith_spread": spread,
           "tilt_identifiable": identifiable, "grid": reg, "centroid_azimuth": {k: v for k, v in cen.items() if k != "daily"},
           "noon_tilt": noon, "mixture": mix, "east_west_model": ewh,
           "bank": bank_src,
           "azimuth_prior": azimuth_prior, "prior_east_west": bool(prior_east_west)}
    if register_orientation is not None:
        out["plausibility"] = plausibility(reg, mix, register_orientation, register_tilt_class, tilt_identifiable=identifiable,
                                           azimuth_prior=azimuth_prior, prior_east_west=prior_east_west)
    return out
