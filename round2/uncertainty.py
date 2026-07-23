"""WP7.1 — first-order (delta-method) uncertainty band.

Below rated power: (sigma_P/P)^2 ~= (3 sigma_v/v)^2 + (sigma_rho/rho)^2
+ (sigma_DF/DF)^2 + (sigma_curve/P)^2. The x3 wind amplification is the
headline; the approximation breaks near cut-in/rated (cross-check only).
"""

import numpy as np


def relative_power_sigma(sigma_v_rel: float,
                         sigma_rho_rel: float = 0.01,
                         sigma_df_rel: float = 0.01,
                         sigma_curve_rel: float = 0.03) -> float:
    """First-order relative sigma of power below rated."""
    return float(np.sqrt((3.0 * sigma_v_rel) ** 2 + sigma_rho_rel ** 2
                         + sigma_df_rel ** 2 + sigma_curve_rel ** 2))


def wind_sigma_from_class(era5_class: int, class_rmse: dict = None,
                          mean_wind: float = 5.0) -> float:
    """Relative wind-speed sigma per ERA5 quality class, from the WP2-A RMSE
    (class_rmse: {class: rmse in m/s}, default = Hu-threshold midpoints)."""
    rmse = (class_rmse or {1: 1.0, 2: 2.25, 3: 3.5}).get(int(era5_class), 2.25)
    return float(rmse / mean_wind)


def delta_band_summary(sigma_v_rel: float) -> dict:
    s = relative_power_sigma(sigma_v_rel)
    return {
        "sigma_v_rel": sigma_v_rel,
        "sigma_P_rel": s,
        "amplification": s / sigma_v_rel,
        "band_95_pct": 1.96 * s * 100.0,
    }
