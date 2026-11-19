import numpy as np
import pandas as pd
import pytest

from round2 import correction


Q = correction.QUANTILES


def _tables(shift=0.0, scale=1.0):
    q_src = np.linspace(1.0, 12.0, 13)
    q_dst = q_src * scale + shift
    return q_src, q_dst


def test_identity_when_tables_equal():
    q, _ = _tables()
    v = np.array([0.5, 3.3, 7.7, 15.0])
    np.testing.assert_allclose(correction.quantile_map(v, q, q), v)


def test_monotonicity():
    q_src, q_dst = _tables(shift=0.4, scale=1.1)
    v = np.linspace(0, 20, 500)
    out = correction.quantile_map(v, q_src, q_dst)
    assert np.all(np.diff(out) >= 0)


def test_linear_tail_extension():
    q_src, q_dst = _tables(scale=1.2)
    # above the last quantile the outermost slope (1.2) continues
    hi = correction.quantile_map(np.array([20.0]), q_src, q_dst)[0]
    assert hi == pytest.approx(q_dst[-1] + (20.0 - q_src[-1]) * 1.2)


def test_height_consistent_scales_hub_wind_exactly():
    """v10 and v100 both scaled by c -> alpha unchanged, v(hub) scaled by c."""
    idx = pd.date_range("2024-01-01", periods=100, freq="1h", tz="UTC")
    rng = np.random.default_rng(0)
    v10 = pd.Series(rng.uniform(2, 10, 100), index=idx)
    v100 = v10 * 1.4
    df = pd.DataFrame({"wind_speed_10m": v10, "wind_speed_100m": v100})
    q_src, q_dst = _tables(scale=1.15)
    out = correction.apply_gated_correction(df.copy(), q_src, q_dst,
                                            mode="height_consistent")
    c = out["qm_factor"].values

    def hub(v10, v100, h=120.0):
        alpha = np.log(v100 / v10) / np.log(10.0)
        return v10 * (h / 10.0) ** alpha

    np.testing.assert_allclose(
        hub(out["wind_speed_10m"].values, out["wind_speed_100m"].values),
        hub(v10.values, v100.values) * c, rtol=1e-12)
    # alpha unchanged
    a_before = np.log(v100.values / v10.values) / np.log(10.0)
    a_after = np.log(out["wind_speed_100m"].values / out["wind_speed_10m"].values) / np.log(10.0)
    np.testing.assert_allclose(a_before, a_after, rtol=1e-12)


def test_wind10_only_nil_at_100m_hub():
    """Guide height rule: correcting v10 only changes v(hub) by (1+c)^(1-beta),
    i.e. ~nil at a 100 m hub (beta ~= 1)."""
    idx = pd.date_range("2024-01-01", periods=50, freq="1h", tz="UTC")
    v10 = pd.Series(np.full(50, 5.0), index=idx)
    v100 = pd.Series(np.full(50, 7.0), index=idx)
    df = pd.DataFrame({"wind_speed_10m": v10, "wind_speed_100m": v100})
    q_src, q_dst = _tables(scale=1.2)
    out = correction.apply_gated_correction(df.copy(), q_src, q_dst, mode="wind10_only")

    def hub(v10, v100, h):
        alpha = np.clip(np.log(v100 / v10) / np.log(10.0), 0, 0.4)
        return v10 * (h / 10.0) ** alpha

    before = hub(5.0, 7.0, 100.0)
    after = hub(out["wind_speed_10m"].iloc[0], out["wind_speed_100m"].iloc[0], 100.0)
    assert after == pytest.approx(before, rel=1e-9)


def test_branch_gate():
    assert correction.assign_branch(10.0, 5.0, 1) == "A"
    assert correction.assign_branch(30.0, 5.0, 2) == "B"
    assert correction.assign_branch(10.0, 50.0, 3) == "B"
    assert correction.assign_branch(30.0, 5.0, 1) == "C"


def test_mode_off_noop():
    df = pd.DataFrame({"wind_speed_10m": [5.0], "wind_speed_100m": [7.0]})
    out = correction.apply_gated_correction(df.copy(), None, None, mode="off")
    assert out["wind_speed_10m"].iloc[0] == 5.0
    assert out["qm_factor"].iloc[0] == 1.0
