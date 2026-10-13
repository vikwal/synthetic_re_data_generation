"""PV degradation by technology era: light-induced loss, linear wear-out, late acceleration,
plus inverter outages. Replaces the wind-style Weibull/power-curve-stretch model of
generate_pv.py (which applied 5 %/a to every station).

Sources (rates are medians; the report carries the full table and citations):
  Jordan & Kurtz (2013) Prog. Photovolt.: median 0.5 %/a, mean 0.8 %/a, all technologies
  Jordan et al. (2016) Prog. Photovolt.: 0.5-0.6 %/a median, multi-Si post-2000 ~0.6 %/a
  Jordan et al. (2022) PiP "PV fleet degradation": fleet median 0.75 %/a, ~0.5 %/a for
      well-performing systems; LID 1-3 % in the first year (mono PERC also LeTID 1-2 %)
  Fraunhofer ISE / Kiefer et al. (2019): German fleet 0.15 %/a (DC-side, 2000s systems) vs
      0.7 %/a AC-side including inverter effects
  IEA PVPS T13 (2021) Climatic degradation review: temperate climates at the low end
  NREL ATB / PVsyst defaults: 0.5 %/a (mono PERC), 0.4 %/a (TOPCon/HJT bifacial)
"""

import numpy as np
import pandas as pd

# era -> parameters: lid (fraction, reached within the first year), linear rate (1/a),
# late_start (years), late_rate (extra 1/a after late_start), gamma_drift (fraction per 10 a
# by which |gamma_pdc| grows), r_s_growth (series-resistance growth fraction per 10 a)
ERAS = {
    "multi_si_early":  dict(lid=0.020, rate=0.0070, late_start=15.0, late_rate=0.010, gamma_drift=0.08, r_s_growth=0.15),
    "multi_si":        dict(lid=0.020, rate=0.0060, late_start=15.0, late_rate=0.008, gamma_drift=0.06, r_s_growth=0.12),
    "multi_perc":      dict(lid=0.025, rate=0.0055, late_start=18.0, late_rate=0.007, gamma_drift=0.05, r_s_growth=0.10),
    "mono_perc":       dict(lid=0.020, rate=0.0050, late_start=18.0, late_rate=0.007, gamma_drift=0.05, r_s_growth=0.10),
    "topcon_bifacial": dict(lid=0.010, rate=0.0040, late_start=20.0, late_rate=0.006, gamma_drift=0.04, r_s_growth=0.08),
}


def era_params(era: str, **override) -> dict:
    p = dict(ERAS[era])
    p.update(override)
    return p


def degradation_factor(age_years, era: str = "multi_si", **override) -> np.ndarray:
    """Retained DC power fraction at the given age(s) in years."""
    p = era_params(era, **override)
    age = np.clip(np.asarray(age_years, dtype=float), 0.0, None)
    lid = 1.0 - p["lid"] * np.minimum(age, 1.0)
    linear = (1.0 - p["rate"]) ** age
    late = (1.0 - p["late_rate"]) ** np.maximum(age - p["late_start"], 0.0)
    return lid * linear * late


def age_years(index: pd.DatetimeIndex, commissioning_date) -> np.ndarray:
    c = pd.Timestamp(commissioning_date)
    if c.tzinfo is None:
        c = c.tz_localize("UTC")
    return (index - c).total_seconds().values / (365.25 * 86400.0)


def degradation_series(index: pd.DatetimeIndex, commissioning_date, era: str = "multi_si", **override) -> pd.Series:
    """Exact per-timestamp factor (no linear interpolation between start and end age)."""
    return pd.Series(degradation_factor(age_years(index, commissioning_date), era, **override), index=index, name="aging_factor")


def aged_gamma(gamma_pdc: float, age: float, era: str = "multi_si") -> float:
    """Temperature coefficient drift: |gamma| grows by gamma_drift per decade."""
    return gamma_pdc * (1.0 + ERAS[era]["gamma_drift"] * age / 10.0)


def aged_r_s_growth(age: float, era: str = "multi_si") -> float:
    return ERAS[era]["r_s_growth"] * age / 10.0


def inverter_availability(index: pd.DatetimeIndex, n_units: int, rng: np.random.Generator,
                          failures_per_unit_year: float = 0.10, repair_days_mean: float = 7.0) -> pd.Series:
    """Fraction of inverter units online per hour: Poisson failures, exponential repair times."""
    hours = len(index)
    online = np.full(hours, float(n_units))
    rate_per_hour = failures_per_unit_year / 8766.0
    for _ in range(n_units):
        t = 0
        while t < hours:
            gap = rng.exponential(1.0 / rate_per_hour) if rate_per_hour > 0 else np.inf
            t += int(gap)
            if t >= hours:
                break
            dur = max(1, int(rng.exponential(repair_days_mean * 24.0)))
            online[t:t + dur] -= 1.0
            t += dur
    return pd.Series(np.clip(online / n_units, 0.0, 1.0), index=index, name="availability")
