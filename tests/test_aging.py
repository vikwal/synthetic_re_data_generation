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


def test_const_matches_v1():
    """model='const' must reproduce the round-1 get_ageing_degradation vector."""
    import generate_wind_era5 as v1
    idx = pd.date_range("2023-07-24", "2024-06-30 23:00", freq="1h", tz="UTC")
    v1_vec, v1_date = v1.get_ageing_degradation(
        time_vector=idx, commissioning_date="2015-06-25", random_seed=42)
    v2_vec, v2_date = aging.get_degradation_vector(
        idx, model="const", commissioning_date="2015-06-25", random_seed=42)
    assert v1_date == v2_date
    np.testing.assert_allclose(v1_vec, v2_vec, rtol=0, atol=1e-12)


def test_const_matches_v1_sampled_age():
    import generate_wind_era5 as v1
    idx = pd.date_range("2024-01-01", "2024-03-31 23:00", freq="1h", tz="UTC")
    ages = np.load("data/wind_ages.npy")
    v1_vec, v1_date = v1.get_ageing_degradation(time_vector=idx, real_ages=ages,
                                                random_seed=7)
    v2_vec, v2_date = aging.get_degradation_vector(idx, model="const",
                                                   real_ages=ages, random_seed=7)
    assert v1_date == v2_date
    np.testing.assert_allclose(v1_vec, v2_vec, atol=1e-12)


def test_weibull_monotone_decreasing():
    ages = np.linspace(0, 40, 200)
    df = aging.DF_weibull(ages)
    assert np.all(np.diff(df) < 0)
    assert aging.DF_weibull(0.0) == 1.0
