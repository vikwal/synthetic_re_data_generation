"""Estimate a park's effective orientation from measured power and check the master data against it.

assess() is the production path:
  masks        clear hours (clear-sky index >= kc_min, sun elevation >= elev_min) minus the caller's exclusions
               (snow, imputed hours), clipped hours (>= CLIP_FRACTION of the AC limit) and suspect hours (below
               SUSPECT_FRACTION of the fitted clear-sky shape: curtailment, outages, partial availability).
               Without the last two a year of a metered east-west park drifted to 0 deg / 240 deg.
  bank         candidate series {(tilt, azimuth): series}: the production chain per pair (chain_bank: IAM, module
               gamma, inter-row shading, horizon) or the plain POA bank of pv.angles
  grid_region  best pair by R2 of a scaled candidate (the free scale makes it a shape comparison, so capacity,
               losses and aging drop out), the profile-likelihood region R2 >= R2max - delta, and a bootstrap over
               clear days
  gate         the tilt is only identifiable when the clear noons span a range of zenith angles (seasons): a
               summer month alone fits 5-10 deg for a 25 deg park with a deceptively narrow day bootstrap, so the
               tilt verdict is withheld below TILT_ZENITH_SPREAD (noon_zenith_spread)
  east-west    NNLS mixture over the bank (mixture) OR the explicit two-plane hypothesis with BIC model choice
               (east_west_hypothesis); flat candidates (tilt <= FLAT_TILT) carry no azimuth
  prior        with a mapped row axis (azimuth_prior_from_rows) the bank is restricted to that azimuth or pair and
               only the tilt (and east share) comes from power; the azimuth verdict is then the map's
  decision     plausibility(): the register classes are implausible when they lie outside the interval
Everything returns effective angles (terrain-following rows make them a mixture). Cross-check estimators (daily
half-energy azimuth, noon regression) live in pv.angle_diagnostics."""
import numpy as np
import pandas as pd

from pv import angles, timebase

TILTS = np.arange(0.0, 61.0, 5.0)
AZIMUTHS = np.arange(60.0, 301.0, 10.0)
EW_AZIMUTHS = [(60.0, 240.0), (70.0, 250.0), (80.0, 260.0), (90.0, 270.0), (100.0, 280.0), (110.0, 290.0), (120.0, 300.0)]
EW_SHARES = (0.3, 0.4, 0.5, 0.6, 0.7)
EW_BIC_MARGIN = 10.0           # the two-plane model must beat the single plane by this much BIC
EW_MIN_TILT = 10.0             # below this tilt an east-west pair is indistinguishable from a flat plane
TILT_ZENITH_SPREAD = 15.0      # deg of clear-noon zenith range needed before a tilt verdict is given
SUSPECT_FRACTION = 0.5         # hours below this fraction of the fitted clear-sky shape are suspect
CLIP_FRACTION = 0.97           # hours at or above this fraction of the AC limit are clipped
FLAT_TILT = 2.5                # bank components at or below this tilt have no azimuth
FLAT_WEIGHT = 0.5              # mixture weight on flat components that makes the fit "flat"
AZ_CENTRE = {"Süd": 180.0, "Süd-West": 225.0, "Süd-Ost": 135.0, "Ost": 90.0, "West": 270.0, "Nord": 0.0,
             "Nord-Ost": 45.0, "Nord-West": 315.0}                 # MaStR orientation classes -> azimuth


# ------------------------------------------------------------------ classes and rows
def orientation_class(az: float) -> str:
    """Azimuth (deg from north) -> MaStR 8-class orientation."""
    names = ["Nord", "Nord-Ost", "Ost", "Süd-Ost", "Süd", "Süd-West", "West", "Nord-West"]
    return names[int(((az + 22.5) % 360.0) // 45.0)]


def tilt_class(t: float) -> str:
    return "< 20 Grad" if t < 20 else "20 - 40 Grad" if t < 40 else "40 - 60 Grad" if t < 60 else "> 60 Grad"


def azimuth_prior_from_rows(row_axis_deg: float) -> dict:
    """Module azimuth implied by a mapped table-row axis (deg, 0-180): east-west rows carry south-facing modules
    (perpendicular on the south side), north-south rows (axis < 30 or > 150) an east-west pair."""
    ax = float(row_axis_deg) % 180.0
    if ax < 30.0 or ax > 150.0:
        az_e, az_w = (ax + 90.0) % 360.0, (ax + 270.0) % 360.0
        if az_e > 180.0:
            az_e, az_w = az_w, az_e
        return {"east_west": True, "azimuth": None, "azimuths": (az_e, az_w)}
    return {"east_west": False, "azimuth": (ax + 90.0) % 360.0, "azimuths": None}


# ------------------------------------------------------------------ masks
def clear_mask(drv: pd.DataFrame, lat: float, lon: float, alt: float, exclude: pd.Series = None, kc_min: float = 0.8,
               elev_min: float = 10.0) -> pd.Series:
    m = angles.clear_hours(drv["ghi"], lat, lon, alt, kc_min, elev_min)
    if exclude is not None:
        m &= ~exclude.reindex(m.index, fill_value=False).astype(bool)
    return m


def clipping_mask(power: pd.Series, ac_limit_w: float = None, frac: float = CLIP_FRACTION, q: float = 0.999) -> pd.Series:
    """Hours at the AC limit. Without a limit the q-quantile of the series stands in, which always removes the
    top hours (about 2 % of the clear hours on parks that never clip); pass the limit where it is known."""
    lim = float(ac_limit_w) if ac_limit_w is not None and np.isfinite(ac_limit_w) else float(power.quantile(q))
    return power >= frac * lim


def suspect_mask(power: pd.Series, bank: dict, mask: pd.Series, frac: float = SUSPECT_FRACTION, iterations: int = 2) -> pd.Series:
    """Clear hours whose power is below frac x the scaled best-pair series (curtailment, outages, partial
    availability). The best pair is refitted without the suspects once (iterations)."""
    sus = pd.Series(False, index=power.index)
    m = mask & power.notna()
    for _ in range(iterations):
        g = angles.grid_fit(power, bank, m & ~sus)
        if len(g) == 0 or not np.isfinite(g.iloc[0].r2):
            break
        b = g.iloc[0]
        new = m & (power < frac * b.scale * bank[(b.tilt, b.azimuth)])
        if new.equals(sus):
            break
        sus = new
    return sus


def noon_zenith_spread(mask: pd.Series, zenith: pd.Series) -> float:
    """Range (deg) of the true noon zenith over the days whose noon hour (or a neighbour) is in the mask: the
    seasonal lever that identifies the tilt. Days with only morning or evening hours in the mask do not count."""
    noon = zenith.groupby(zenith.index.date).idxmin().dropna()
    pos = zenith.index.get_indexer(pd.DatetimeIndex(noon))
    m = mask.reindex(zenith.index, fill_value=False).to_numpy(dtype=bool)
    keep = [float(zenith.iloc[i]) for i in pos if i >= 0 and m[max(i - 1, 0):i + 2].any()]
    return float(np.ptp(keep)) if len(keep) >= 2 else 0.0


# ------------------------------------------------------------------ bank
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
    res = chain.run_park(pvconfig.normalise(c), d, horizon=horizon, return_arrays=True)
    per = res.attrs["arrays"]
    return {(t, a): per[f"b{j}"]["p_dc_w"] for j, (t, a) in enumerate(pairs)}


# ------------------------------------------------------------------ fits
def _fit(y: np.ndarray, x: np.ndarray) -> tuple:
    """Free scale s and residual sum of squares of y ~ s x."""
    s = float(np.dot(x, y) / np.dot(x, x)) if np.dot(x, x) > 0 else 0.0
    return s, float(np.sum((y - s * x) ** 2))


def _best_pair(y: np.ndarray, X: np.ndarray, keys: list) -> tuple:
    """(tilt, azimuth) with the smallest residual sum of squares of y ~ s_k X[:, k]."""
    xx = (X * X).sum(axis=0)
    s = np.where(xx > 0, (X * y[:, None]).sum(axis=0) / np.where(xx > 0, xx, 1.0), 0.0)
    rss = ((y[:, None] - X * s) ** 2).sum(axis=0)
    return keys[int(np.argmin(rss))]


def east_west_hypothesis(power: pd.Series, bank: dict, mask: pd.Series, pairs=None, shares=EW_SHARES, min_tilt: float = EW_MIN_TILT) -> dict:
    """Explicit two-plane model p = s (w B(t, az_e) + (1 - w) B(t, az_w)) over the bank tilts >= min_tilt and the
    azimuth pairs, against the best single plane, compared by BIC (4 vs 3 parameters). preferred is True when the
    pair beats the single plane by EW_BIC_MARGIN. Pairs default to the bank azimuths 180 deg apart (EW_AZIMUTHS)."""
    m = (mask & power.notna()).values
    y = power.values[m]
    n = int(m.sum())
    if n < 24:
        return {"preferred": False, "n": n}
    azs = {a for (t, a) in bank}
    pairs = pairs or [p for p in EW_AZIMUTHS if p[0] in azs and p[1] in azs]
    best1 = min(((t, a, _fit(y, s.values[m])[1]) for (t, a), s in bank.items()), key=lambda r: r[2])
    best2 = None
    for t in sorted({t for (t, a) in bank}):
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
    if best2 is None or best1[2] <= 0:
        return {"preferred": False, "n": n}
    tss = float(np.sum((y - y.mean()) ** 2))
    bic1 = n * np.log(best1[2] / n) + 3 * np.log(n)
    bic2 = n * np.log(best2[4] / n) + 4 * np.log(n)
    return {"preferred": bool(bic2 + EW_BIC_MARGIN < bic1), "n": n, "tilt": best2[0], "azimuth_east": best2[1], "azimuth_west": best2[2],
            "share_east": best2[3], "r2": 1.0 - best2[4] / tss if tss > 0 else np.nan, "r2_single": 1.0 - best1[2] / tss if tss > 0 else np.nan,
            "single": (best1[0], best1[1]), "delta_bic": float(bic2 - bic1)}


def grid_region(power: pd.Series, bank: dict, mask: pd.Series, delta: float = 0.005, n_boot: int = 100, seed: int = 0) -> dict:
    """Best (tilt, azimuth) on the bank, the profile-likelihood region R2 >= R2max - delta (as ranges), and a
    bootstrap over clear days (days redrawn with replacement, their hours repeated by multiplicity; median and
    5-95 % of the best pair)."""
    g = angles.grid_fit(power, bank, mask)
    m = (mask & power.notna()).to_numpy()
    codes, _ = pd.factorize(pd.Index(power.index.date))
    days = np.unique(codes[m])
    out = {"tilt": np.nan, "azimuth": np.nan, "r2": np.nan, "scale": np.nan, "tilt_range": (np.nan, np.nan), "azimuth_range": (np.nan, np.nan),
           "n_hours": int(m.sum()), "n_days": int(len(days))}
    if len(g) == 0 or not np.isfinite(g.iloc[0].r2):
        return out
    best = g.iloc[0]
    reg = g[g.r2 >= best.r2 - delta]
    out.update({"tilt": float(best.tilt), "azimuth": float(best.azimuth), "r2": float(best.r2), "scale": float(best.scale),
                "tilt_range": (float(reg.tilt.min()), float(reg.tilt.max())), "azimuth_range": (float(reg.azimuth.min()), float(reg.azimuth.max()))})
    if n_boot and len(days) >= 5:
        keys = list(bank)
        X = np.column_stack([bank[k].to_numpy() for k in keys])
        y = power.to_numpy()
        hours_of = {d: np.flatnonzero(m & (codes == d)) for d in days}
        rng = np.random.default_rng(seed)
        picks = []
        for _ in range(n_boot):
            h = np.concatenate([hours_of[d] for d in rng.choice(days, len(days), replace=True)])
            picks.append(_best_pair(y[h], X[h], keys))
        bt, ba = np.array([p[0] for p in picks]), np.array([p[1] for p in picks])
        out.update({"tilt_boot": (float(np.percentile(bt, 5)), float(np.median(bt)), float(np.percentile(bt, 95))),
                    "azimuth_boot": (float(np.percentile(ba, 5)), float(np.median(ba)), float(np.percentile(ba, 95)))})
    return out


def mixture(power: pd.Series, bank: dict, mask: pd.Series, min_side: float = 0.25, max_south: float = 0.3) -> dict:
    """NNLS mixture over the bank. east_west is flagged only for a genuine two-sided layout: at least min_side of
    the weight on each of the east (<= 120 deg) and west (>= 240 deg) sides and at most max_south between 135
    and 225 deg - small leaks of weight to the sides (terrain spread, noise) do not count. Components at tilt
    <= FLAT_TILT carry no azimuth (the bank series are identical for every azimuth there) and are counted as
    w_flat, never as south: a flat east-west park is indistinguishable from a flat south park by power alone.
    components lists the pairs with weight >= 0.1."""
    nn = angles.nnls_mixture(power, bank, mask)
    wall = nn["weights"]
    az = np.array([a for (t, a) in wall.index])
    flat = np.array([t <= FLAT_TILT for (t, a) in wall.index])
    w_f = float(wall.values[flat].sum())
    w_e, w_w = float(wall.values[(az <= 120) & ~flat].sum()), float(wall.values[(az >= 240) & ~flat].sum())
    w_s = float(wall.values[(az >= 135) & (az <= 225) & ~flat].sum())
    return {"components": [(float(t), float(a), float(x)) for (t, a), x in wall[wall >= 0.1].items()],
            "tilt_eff": nn["tilt_eff"], "azimuth_eff": nn["azimuth_eff"], "r2": nn["r2"], "w_east": w_e, "w_west": w_w, "w_south": w_s,
            "w_flat": w_f, "flat": bool(w_f >= FLAT_WEIGHT),
            "east_west": bool(w_e >= min_side and w_w >= min_side and w_s <= max_south)}


# ------------------------------------------------------------------ decision
def _interval_distance(lo: float, hi: float, c: float) -> float:
    """Smallest angular distance (deg) from the azimuth c to the interval [lo, hi] (the bank has no north wrap)."""
    if lo <= c <= hi:
        return 0.0
    return min(abs(((lo - c) + 180.0) % 360.0 - 180.0), abs(((hi - c) + 180.0) % 360.0 - 180.0))


def plausibility(region: dict, mix: dict, register_orientation: str, register_tilt_class: str = None, use_bootstrap: bool = True,
                 az_margin: float = 45.0, tilt_margin: float = 5.0, tilt_identifiable: bool = True,
                 azimuth_prior: float = None, prior_east_west: bool = False) -> dict:
    """Compare the register classes with the estimate. Azimuth: implausible when the whole bootstrap 5-95 %
    interval of the best azimuth (fallback: profile region) lies more than az_margin (one octant) from the
    registered class centre, or when the east-west flag and the register disagree. A flat fit (mixture weight on
    tilt <= FLAT_TILT, or the whole tilt interval <= 5 deg) has no azimuth: the verdict is None, except that it is
    consistent with a registered east-west layout. With azimuth_prior (mapped rows) the azimuth verdict is the
    prior's alone and the power decides only the tilt. Tilt: implausible when the interval, widened by
    tilt_margin, misses the registered class; without tilt_identifiable (too little noon-zenith spread) the tilt
    verdict is None and the tilt interval is the profile region, not the day bootstrap."""
    if use_bootstrap and "azimuth_boot" in region:
        az_lo, az_hi = region["azimuth_boot"][0], region["azimuth_boot"][2]
        t_lo, t_hi = region["tilt_boot"][0], region["tilt_boot"][2]
    else:
        (az_lo, az_hi), (t_lo, t_hi) = region["azimuth_range"], region["tilt_range"]
    if not tilt_identifiable:
        t_lo, t_hi = region["tilt_range"]
    if not (np.isfinite(az_lo) and np.isfinite(t_lo)):
        return {"azimuth_ok": None, "tilt_ok": None, "flat": False, "estimated_azimuth_classes": [], "estimated_tilt_classes": []}
    flat = bool(mix.get("flat", False)) or t_hi <= 5.0
    east_west = bool(mix.get("east_west", False))
    if azimuth_prior is not None:
        flat, east_west = False, bool(prior_east_west)
        az_lo = az_hi = float(azimuth_prior) % 360.0
    reg_ew = register_orientation == "Ost-West"
    if east_west:
        az_ok = reg_ew
    elif flat:
        az_ok = True if reg_ew else None
    elif reg_ew:
        az_ok = False
    elif register_orientation in AZ_CENTRE:
        az_ok = _interval_distance(az_lo, az_hi, AZ_CENTRE[register_orientation]) <= az_margin
    else:
        az_ok = True
    t_classes = {tilt_class(t) for t in np.arange(max(t_lo - tilt_margin, 0.0), t_hi + tilt_margin + 0.1, 2.5)}
    az_classes = {"flach"} if flat else {orientation_class(a) for a in np.arange(az_lo, az_hi + 0.1, 5.0)}
    return {"azimuth_ok": None if az_ok is None else bool(az_ok),
            "tilt_ok": None if (not tilt_identifiable or register_tilt_class is None) else bool(register_tilt_class in t_classes),
            "flat": flat, "estimated_azimuth_classes": sorted(az_classes | ({"Ost-West"} if east_west else set())),
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
    (chain_bank), else the plain POA bank. With azimuth_prior (deg, from mapped rows) the bank is restricted to
    that azimuth - or, with prior_east_west, to the pair azimuth_prior and azimuth_prior + 180 - and only the tilt
    (and east/west share) is estimated from power. ew_hypothesis ORs the explicit two-plane verdict into
    mixture.east_west (mixture.east_west_nnls keeps the NNLS flag alone). The final hour mask is returned as
    'mask' for cross-checks (pv.angle_diagnostics)."""
    clear = clear_mask(drv, lat, lon, alt, exclude)
    clip = clipping_mask(power, ac_limit_w) if (robust or ac_limit_w is not None) else pd.Series(False, index=power.index)
    mask = clear & ~clip
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
    spread = noon_zenith_spread(mask & power.notna(), timebase.solar_position(drv.index, lat, lon, alt)["apparent_zenith"])
    identifiable = bool(spread >= zenith_spread_min)
    reg = grid_region(power, bank, mask, delta, n_boot)
    mix = mixture(power, bank, mask)
    ewh = {"preferred": False, "n": 0}
    if ew_hypothesis and (azimuth_prior is None or prior_east_west):
        ewh = east_west_hypothesis(power, bank, mask, pairs=[tuple(azs)] if prior_east_west else None)
    mix["east_west_nnls"] = mix["east_west"]
    mix["east_west"] = bool(mix["east_west"] or ewh["preferred"])
    out = {"n_clear_hours": int(mask.sum()), "n_clear_before_suspect": n_clear_raw, "n_suspect": int(sus.sum()),
           "n_clipped_clear": int((clip & clear).sum()), "zenith_spread": spread, "tilt_identifiable": identifiable,
           "grid": reg, "mixture": mix, "east_west_model": ewh, "bank": bank_src, "mask": mask,
           "azimuth_prior": azimuth_prior, "prior_east_west": bool(prior_east_west)}
    if register_orientation is not None:
        out["plausibility"] = plausibility(reg, mix, register_orientation, register_tilt_class, tilt_identifiable=identifiable,
                                           azimuth_prior=azimuth_prior, prior_east_west=prior_east_west)
    return out
