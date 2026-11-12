"""Soiling: HSU deposition/rain-wash model (pvlib) as default, and the repaired legacy
erf-deposition model of the 2025 station script (which returned power * loss instead of power * (1 - loss))."""

import numpy as np
import pandas as pd
import pvlib
from scipy.special import erf


def hsu_ratio(rain_mm: pd.Series, pm2_5_gm3: pd.Series, pm10_gm3: pd.Series, tilt: float,
              cleaning_threshold_mm: float = 1.0, depo_veloc: dict = None, rain_accum_period: str = "24h") -> pd.Series:
    """Soiling ratio (1 = clean) from the Humboldt State (HSU) model; PM in g/m3, rain in mm per step."""
    depo = depo_veloc or {"2_5": 0.0009, "10": 0.004}
    r = pvlib.soiling.hsu(rain_mm.fillna(0.0), cleaning_threshold_mm, tilt, pm2_5_gm3.fillna(0.0), pm10_gm3.fillna(0.0),
                          depo_veloc=depo, rain_accum_period=pd.Timedelta(rain_accum_period))
    return r.clip(0.0, 1.0).rename("soiling_ratio")


def legacy_erf_loss(pm10_gm3: pd.Series, rain_mm: pd.Series, tilt: float, step_s: float = 3600.0,
                    v_depos: float = 0.009, cleaning_thresh_mm_day: float = 5.0, cleaning_factor_day: float = 0.1,
                    w0: float = 4.0) -> pd.Series:
    """Loss fraction of the legacy model: PM10 deposition (g/m2) accumulates, rain above the daily
    threshold (scaled to the step) washes a fraction off; loss = 0.3434 erf(0.17 w^0.8473)."""
    pm = pm10_gm3.fillna(0.0).values
    rain = rain_mm.fillna(0.0).values
    dep = pm * v_depos * step_s * np.cos(np.radians(tilt))
    steps_per_day = 86400.0 / step_s
    thresh = cleaning_thresh_mm_day / steps_per_day
    factor = cleaning_factor_day / steps_per_day
    w = np.empty(len(pm))
    w[0] = w0
    for i in range(1, len(pm)):
        w[i] = w[i - 1] * (1.0 - factor) if rain[i] >= thresh else w[i - 1] + dep[i]
    loss = 0.3434 * erf(0.17 * np.power(np.clip(w, 0, None), 0.8473))
    return pd.Series(loss, index=pm10_gm3.index, name="soiling_loss")


def apply(power: pd.Series, ratio: pd.Series) -> pd.Series:
    """power * soiling_ratio (ratio = 1 - loss)."""
    return power * ratio.reindex(power.index).fillna(1.0)
