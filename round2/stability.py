"""WP3 — Monin-Obukhov stability-corrected extrapolation (guide section 3).

All formulas follow Implementation_Guide_Round2.md 3.1-3.4:
  H  = -sshf/3600                          (ERA5 sshf is J/m2 accumulated,
                                            positive DOWNWARD -> upward W/m2)
  L  = -(u*^3 rho c_p T) / (kappa g H)
  unstable: x=(1-16 z/L)^(1/4);
            psi_m = ln[((1+x^2)/2)((1+x)/2)^2] - 2 atan(x) + pi/2
  stable:   psi_m = -5 z/L
  v(h) = v100 * [ln(h/z0) - psi_m(h/L)] / [ln(100/z0) - psi_m(100/L)]
Guards: |H| < 5 W/m2 -> neutral (psi_m = 0); z/L capped at +-2.
"""

import numpy as np

C_P = 1005.0
KARMAN = 0.4
G = 9.81
H_NEUTRAL_W_M2 = 5.0
ZL_CAP = 2.0
ANCHOR_HEIGHT = 100.0


def upward_heat_flux(sshf: np.ndarray) -> np.ndarray:
    """W/m2, positive upward, from hourly-accumulated ERA5 sshf (J/m2, +down)."""
    return -np.asarray(sshf, dtype=float) / 3600.0


def obukhov_length(u_star, rho, temp_k, sshf):
    """Obukhov length L per timestep. Neutral hours (|H| < 5 W/m2) -> L = inf."""
    H = upward_heat_flux(sshf)
    u_star = np.asarray(u_star, dtype=float)
    rho = np.asarray(rho, dtype=float)
    temp_k = np.asarray(temp_k, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        L = -(u_star ** 3 * rho * C_P * temp_k) / (KARMAN * G * H)
    L = np.where(np.abs(H) < H_NEUTRAL_W_M2, np.inf, L)
    return L


def psi_m(z, L):
    """Businger-Dyer integrated stability function; z/L capped at +-2."""
    z = np.asarray(z, dtype=float)
    L = np.asarray(L, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        zeta = np.where(np.isinf(L), 0.0, z / L)
    zeta = np.clip(zeta, -ZL_CAP, ZL_CAP)
    out = np.zeros_like(zeta)
    unstable = zeta < 0
    if np.any(unstable):
        x = (1.0 - 16.0 * zeta[unstable]) ** 0.25
        out[unstable] = (np.log(((1 + x ** 2) / 2.0) * ((1 + x) / 2.0) ** 2)
                         - 2.0 * np.arctan(x) + np.pi / 2.0)
    stable = zeta > 0
    out[stable] = -5.0 * zeta[stable]
    return out


def most_wind_profile(v100, hub_height, z0, L):
    """Stability-corrected wind at hub height, anchored at 100 m (guide 3.3)."""
    v100 = np.asarray(v100, dtype=float)
    num = np.log(hub_height / z0) - psi_m(hub_height, L)
    den = np.log(ANCHOR_HEIGHT / z0) - psi_m(ANCHOR_HEIGHT, L)
    # den <= 0 cannot happen for meaningful z0 << 100 m and capped z/L, but guard
    ratio = np.where(den > 0, num / den, 1.0)
    return v100 * ratio


def stability_class(L, neutral_threshold: float = 500.0):
    """Bin hours by L for the stratified error table: 'unstable' (L<0),
    'stable' (0<L<threshold... actually |L| large -> neutral)."""
    L = np.asarray(L, dtype=float)
    out = np.full(L.shape, "neutral", dtype=object)
    finite = np.isfinite(L)
    out[finite & (L < 0) & (np.abs(L) < neutral_threshold)] = "unstable"
    out[finite & (L > 0) & (np.abs(L) < neutral_threshold)] = "stable"
    return out


def blh_flag(hub_height, blh):
    """True where hub_height > 0.1 * boundary-layer height (guide 3.4)."""
    return np.asarray(hub_height, dtype=float) > 0.1 * np.asarray(blh, dtype=float)
