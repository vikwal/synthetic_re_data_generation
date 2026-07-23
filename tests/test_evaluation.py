import numpy as np
import pandas as pd
import pytest
from scipy import stats

from round2 import evaluation


def _series(n=1000, seed=1):
    idx = pd.date_range("2023-07-24", periods=n, freq="1h", tz="UTC")
    rng = np.random.default_rng(seed)
    meas = pd.Series(rng.uniform(0, 2e6, n), index=idx)
    return idx, meas


def test_perfect_prediction():
    _, meas = _series()
    m = evaluation.evaluate(meas, meas.copy(), p_rated=2e6)
    assert m["r2"] == pytest.approx(1.0)
    assert m["energy_ratio"] == pytest.approx(1.0)
    assert m["wasserstein"] == pytest.approx(0.0, abs=1e-12)
    assert m["rmse_n"] == 0.0


def test_energy_ratio_scaling():
    _, meas = _series()
    m = evaluation.evaluate(meas, meas * 1.25, p_rated=2e6)
    assert m["energy_ratio"] == pytest.approx(1.25)


def test_wasserstein_shift():
    idx, meas = _series()
    shift = 0.1 * 2e6
    m = evaluation.evaluate(meas, meas + shift, p_rated=2e6)
    assert m["wasserstein"] == pytest.approx(0.1, rel=1e-6)


def test_curtailment_screen():
    idx, meas = _series(n=200)
    prices = pd.Series(50.0, index=idx)
    prices.iloc[:50] = -5.0  # flagged hours
    synth = meas.copy()
    synth.iloc[:50] *= 3  # corrupt only flagged hours
    m = evaluation.evaluate(meas, synth, p_rated=2e6, prices=prices)
    assert m["curtailed_share"] == pytest.approx(0.25)
    assert m["r2_excl_curt"] == pytest.approx(1.0)
    assert m["r2"] < 1.0


def test_wilcoxon_matches_scipy_and_effect_size():
    a = pd.Series([0.90, 0.85, 0.88, 0.92, 0.80, 0.83, 0.87, 0.91, 0.86, 0.84, 0.89, 0.82, 0.88],
                  index=[f"p{i}" for i in range(13)])
    b = a - pd.Series(np.linspace(0.01, 0.05, 13), index=a.index)  # a strictly better
    res = evaluation.compare_pathways(a, b)
    ref = stats.wilcoxon(a.values, b.values, method="exact")
    assert res["wilcoxon_p"] == pytest.approx(ref.pvalue)
    assert res["rank_biserial"] == pytest.approx(1.0)  # all diffs positive
    assert res["median_diff"] > 0
    assert res["boot_ci_lo"] > 0


def test_rank_biserial_hand_example():
    # diffs: +1 +2 +3 -4 -> ranks 1,2,3,4; W+ = 6, W- = 4, W_min = 4
    # r = 1 - 2*4/10 = 0.2, sign positive (W+ > W-)
    x = np.array([1.0, 2.0, 3.0, 0.0])
    y = np.array([0.0, 0.0, 0.0, 4.0])
    assert evaluation.rank_biserial(x, y) == pytest.approx(0.2)


def test_alignment_drops_nan():
    idx, meas = _series(n=100)
    synth = meas.copy()
    meas.iloc[:10] = np.nan
    m = evaluation.evaluate(meas, synth, p_rated=2e6)
    assert m["n_hours"] == 90
