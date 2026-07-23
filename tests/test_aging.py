import numpy as np
import pandas as pd
import pytest

from round2 import aging


# Guide WP4.1 reference numbers: loss @ 5/10/20/28 y and annual rates
REF_LOSS = {5: 0.008, 10: 0.033, 20: 0.126, 28: 0.232}
REF_RATE = {5: 0.0034, 10: 0.0067, 20: 0.0135, 28: 0.0189}


@pytest.mark.parametrize("age,loss", REF_LOSS.items())
def test_weibull_reference_losses(age, loss):
    assert 1.0 - aging.DF_weibull(age) == pytest.approx(loss, abs=5e-4)


@pytest.mark.parametrize("age,rate", REF_RATE.items())
def test_weibull_annual_rates(age, rate):
    # the guide's %/yr numbers are the hazard rate -d ln(DF)/dt = kappa*a^(k-1)/lam^k
    df = aging.DF_weibull
    annual = np.log(df(age - 0.5)) - np.log(df(age + 0.5))
    assert annual == pytest.approx(rate, abs=5e-5)


def test_step_only_after_20y():
    ages = np.array([5.0, 19.9, 20.0])
    assert np.allclose(aging.DF_weibull_step(ages), aging.DF_weibull(ages))
    assert aging.DF_weibull_step(25.0) < aging.DF_weibull(25.0)
    # delta=2 %/yr beyond 20
    assert aging.DF_weibull_step(22.0) == pytest.approx(
        aging.DF_weibull(22.0) * 0.98 ** 2, rel=1e-12)


def _reference_const_ageing(
        time_vector: pd.DatetimeIndex,
        mean_age_years: float = 15.0,
        std_dev_age_years: float = 5.0,
        annual_load_factor_loss_rate: float = 0.0063,
        real_ages: np.ndarray = None,
        commissioning_date: str = None,
        random_seed: int = 42):
    """Frozen reference formula for the linear ('const') degradation model.

    Kept as a fixed, independent implementation (not imported from
    round2.aging) so test_const_matches_reference below actually checks
    round2.aging.get_degradation_vector(model="const") against an
    unrelated computation of the same linear formula, rather than against
    itself.
    """
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
    end_age = (time_vector[-1] - pd.to_datetime(commissioning_date, utc=True)).days / 365.25
    base_efficiency = 1.0 - annual_load_factor_loss_rate
    efficiency_factor_start = base_efficiency ** start_age
    efficiency_factor_end = base_efficiency ** end_age
    efficiency_vector = np.linspace(efficiency_factor_start, efficiency_factor_end, num=len(time_vector))
    return efficiency_vector, commissioning_date


def test_const_matches_reference():
    """model='const' must reproduce the linear degradation reference formula."""
    idx = pd.date_range("2023-07-24", "2024-06-30 23:00", freq="1h", tz="UTC")
    ref_vec, ref_date = _reference_const_ageing(
        time_vector=idx, commissioning_date="2015-06-25", random_seed=42)
    vec, date = aging.get_degradation_vector(
        idx, model="const", commissioning_date="2015-06-25", random_seed=42)
    assert ref_date == date
    np.testing.assert_allclose(ref_vec, vec, rtol=0, atol=1e-12)


def test_const_matches_reference_sampled_age():
    idx = pd.date_range("2024-01-01", "2024-03-31 23:00", freq="1h", tz="UTC")
    ages = np.load("data/wind_ages.npy")
    ref_vec, ref_date = _reference_const_ageing(time_vector=idx, real_ages=ages,
                                                random_seed=7)
    vec, date = aging.get_degradation_vector(idx, model="const",
                                             real_ages=ages, random_seed=7)
    assert ref_date == date
    np.testing.assert_allclose(ref_vec, vec, atol=1e-12)


def test_weibull_monotone_decreasing():
    ages = np.linspace(0, 40, 200)
    df = aging.DF_weibull(ages)
    assert np.all(np.diff(df) < 0)
    assert aging.DF_weibull(0.0) == 1.0
