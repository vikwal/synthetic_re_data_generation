import numpy as np
import pytest

from round2 import stability


def test_sshf_sign_convention():
    """ERA5 sshf is positive downward; a convective summer day has NEGATIVE
    sshf, which must map to POSITIVE upward H."""
    sshf_summer_noon = -900_000.0  # J/m2 per hour, downward-positive
    H = stability.upward_heat_flux(sshf_summer_noon)
    assert H == pytest.approx(250.0)
    assert H > 0


def test_neutral_limit_recovers_log_profile():
    """|H| < 5 W/m2 -> L = inf -> psi_m = 0 -> pure log-ratio profile."""
    v100, z0, h = 8.0, 0.1, 138.0
    L = stability.obukhov_length(u_star=0.4, rho=1.2, temp_k=288.0, sshf=1000.0)
    assert np.isinf(L)
    v = stability.most_wind_profile(v100, h, z0, L)
    expected = v100 * np.log(h / z0) / np.log(100.0 / z0)
    assert v == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("L", [-50.0, -500.0, 50.0, 500.0, np.inf])
def test_anchor_height_returns_v100(L):
    v = stability.most_wind_profile(8.0, 100.0, 0.05, L)
    assert v == pytest.approx(8.0, rel=1e-12)


def test_psi_m_signs():
    # unstable (L<0): psi_m > 0 (less shear); stable (L>0): psi_m < 0
    assert stability.psi_m(100.0, -100.0) > 0
    assert stability.psi_m(100.0, 100.0) < 0
    assert stability.psi_m(100.0, np.inf) == 0.0


def test_zl_cap():
    # z/L capped at +-2 -> psi_m stable floor = -10
    assert stability.psi_m(1000.0, 1.0) == pytest.approx(-10.0)
    assert stability.psi_m(1000.0, -1.0) == pytest.approx(stability.psi_m(2.0, -1.0))


def test_stable_reduces_and_unstable_increases_hub_wind():
    v100, z0, h = 8.0, 0.1, 150.0
    v_neutral = stability.most_wind_profile(v100, h, z0, np.inf)
    v_stable = stability.most_wind_profile(v100, h, z0, 200.0)
    v_unstable = stability.most_wind_profile(v100, h, z0, -200.0)
    assert v_stable > v_neutral > v_unstable


def test_obukhov_formula_value():
    # hand-computed: u*=0.3, rho=1.2, T=290, sshf=-360000 J/m2 -> H=100 W/m2
    L = stability.obukhov_length(0.3, 1.2, 290.0, -360_000.0)
    expected = -(0.3 ** 3 * 1.2 * 1005.0 * 290.0) / (0.4 * 9.81 * 100.0)
    assert L == pytest.approx(expected)
    assert L < 0  # daytime convective -> unstable


def test_blh_flag():
    flags = stability.blh_flag(138.0, np.array([500.0, 2000.0]))
    assert flags.tolist() == [True, False]
