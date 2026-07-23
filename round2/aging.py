"""WP4 — aging models: constant ADR, Weibull wear-out, Weibull + EEG step.

DF(age) is the retained load-factor fraction (1 = new). Downstream application
is unchanged from round 1: the power curve's wind axis is shifted by
(1/DF)**(1/3) (see the power-curve interpolation in generate_wind.py).
"""

import numpy as np
import pandas as pd

LAMBDA_DEFAULT = 54.5  # yr — anchored: 20-yr loss = 20 x 0.63 % = 12.6 % (Germer-equivalent)
KAPPA_DEFAULT = 2.0
ADR_DEFAULT = 0.0063   # round-1 constant annual degradation rate


def DF_const(age, adr: float = ADR_DEFAULT):
    """Round-1 schedule: DF = (1 - ADR)**age."""
    return (1.0 - adr) ** np.asarray(age, dtype=float)


def DF_weibull(age, lam: float = LAMBDA_DEFAULT, kappa: float = KAPPA_DEFAULT):
    """Weibull wear-out: DF = exp(-(age/lambda)**kappa)."""
    age = np.asarray(age, dtype=float)
    return np.exp(-(age / lam) ** kappa)


def DF_weibull_step(age, lam: float = LAMBDA_DEFAULT, kappa: float = KAPPA_DEFAULT,
                    delta: float = 0.02):
    """Weibull + EEG year-20 step: extra (1-delta)**max(age-20, 0)."""
    age = np.asarray(age, dtype=float)
    return DF_weibull(age, lam, kappa) * (1.0 - delta) ** np.maximum(age - 20.0, 0.0)


MODELS = {
    "const": DF_const,
    "weibull": DF_weibull,
    "weibull_step": DF_weibull_step,
}


def resolve_start_age(time_vector: pd.DatetimeIndex,
                      commissioning_date: str = None,
                      real_ages: np.ndarray = None,
                      mean_age_years: float = 15.0,
                      std_dev_age_years: float = 5.0,
                      random_seed: int = 42):
    """Mirror of the round-1 age logic in get_ageing_degradation: an explicit
    commissioning date wins; otherwise sample from the fleet age distribution
    (or a Normal fallback) and back out a synthetic commissioning date."""
    np.random.seed(random_seed)
    if real_ages is None:
        start_age = np.random.normal(loc=mean_age_years, scale=std_dev_age_years)
    else:
        start_age = float(np.random.choice(real_ages, size=1, replace=False)[0])
    start_age = max(0.0, start_age)
    if commissioning_date is not None:
        start_age = (time_vector[0] - pd.to_datetime(commissioning_date, utc=True)).days / 365.25
    else:
        commissioning_date = str((time_vector[0] - pd.Timedelta(days=start_age * 365.25)).date())
    return start_age, commissioning_date


def get_degradation_vector(time_vector: pd.DatetimeIndex,
                           model: str = "const",
                           commissioning_date: str = None,
                           real_ages: np.ndarray = None,
                           random_seed: int = 42,
                           adr: float = ADR_DEFAULT,
                           lam: float = LAMBDA_DEFAULT,
                           kappa: float = KAPPA_DEFAULT,
                           step_delta: float = 0.02):
    """Drop-in replacement for round-1 get_ageing_degradation.

    Returns (efficiency_vector, commissioning_date). model='const' reproduces
    the round-1 vector exactly (linear interpolation between the start-age and
    end-age efficiency, as in v1); the Weibull variants evaluate DF on the
    per-timestamp age instead (exact, not linearized).
    """
    if model not in MODELS:
        raise ValueError(f"unknown aging model: {model}")
    start_age, commissioning_date = resolve_start_age(
        time_vector, commissioning_date, real_ages, random_seed=random_seed)
    end_age = (time_vector[-1] - pd.to_datetime(commissioning_date, utc=True)).days / 365.25
    if model == "const":
        eff_start = DF_const(start_age, adr)
        eff_end = DF_const(end_age, adr)
        vector = np.linspace(eff_start, eff_end, num=len(time_vector))
    else:
        ages = (time_vector - pd.to_datetime(commissioning_date, utc=True)).days / 365.25
        ages = np.clip(np.asarray(ages, dtype=float), 0.0, None)
        if model == "weibull":
            vector = DF_weibull(ages, lam, kappa)
        else:
            vector = DF_weibull_step(ages, lam, kappa, step_delta)
    return vector, commissioning_date
