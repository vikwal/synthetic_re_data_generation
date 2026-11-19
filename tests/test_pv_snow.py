"""parks_pv.generate snow models: depth_linear (release v1) and depth_slide (model e, slide-off rule)."""
import numpy as np
import pandas as pd
import pytest

from parks_pv import generate as GEN


def _idx(n, step="30min"):
    return pd.date_range("2024-01-10", periods=n, freq=step, tz="UTC")


def _pulse(n=21):
    """Ramp fully covered at the first step, then the depth driver drops to zero."""
    r = np.zeros(n)
    r[0] = 1.0
    return pd.Series(r, index=_idx(n))


def _const(v, n=21):
    return pd.Series(float(v), index=_idx(n))


def test_slide_off_steep_sheds_faster_than_flat_in_thaw():
    r = _pulse()
    thaw_t, night_poa = _const(5.0), _const(0.0)            # 5 > -0/80: can slide
    flat = GEN.slide_off(r, thaw_t, night_poa, tilt=8.0, slide_per_h=0.1, step_h=0.5)
    steep = GEN.slide_off(r, thaw_t, night_poa, tilt=60.0, slide_per_h=0.1, step_h=0.5)
    hours = 10.0                                             # 20 steps of 30 min after the pulse
    assert flat.iloc[-1] == pytest.approx(1.0 - 0.1 * np.sin(np.radians(8.0)) * hours)
    assert steep.iloc[-1] == pytest.approx(max(0.0, 1.0 - 0.1 * np.sin(np.radians(60.0)) * hours))
    assert steep.iloc[-1] < flat.iloc[-1]
    assert (np.diff(flat.values) <= 0).all() and (flat >= 0).all()


def test_slide_off_frost_holds_cover_unless_sun_warms_the_array():
    r = _pulse()
    frost = GEN.slide_off(r, _const(-10.0), _const(0.0), tilt=30.0, slide_per_h=0.1, step_h=0.5)
    assert (frost == 1.0).all()                              # -10 < -0/80: no sliding, cover kept
    sunny = GEN.slide_off(r, _const(-10.0), _const(1000.0), tilt=30.0, slide_per_h=0.1, step_h=0.5)
    assert sunny.iloc[-1] < 1.0                              # -10 > -1000/80 = -12.5: pvlib can-slide rule
    zero_tilt = GEN.slide_off(r, _const(5.0), _const(0.0), tilt=0.0, slide_per_h=0.1, step_h=0.5)
    assert (zero_tilt == 1.0).all()


def test_slide_off_follows_rising_ramp_at_once():
    r = pd.Series(np.linspace(0.0, 1.0, 11), index=_idx(11))
    cov = GEN.slide_off(r, _const(5.0, 11), _const(0.0, 11), tilt=30.0, slide_per_h=0.1, step_h=0.5)
    assert np.allclose(cov.values, r.values)


def test_snow_coverage_models():
    idx = _idx(4)
    f = pd.DataFrame({"snow_depth_cm": [0.0, 2.5, 10.0, 0.0], "icon_snow_depth_cm": [5.0, 0.0, 0.0, 0.0],
                      "temp_air": -5.0, "poa_global": 0.0}, index=idx)
    lin = GEN.snow_coverage(f, "depth_linear")
    assert list(lin.values) == [0.0, 0.5, 1.0, 0.0]          # v1: ERA5 only, 5 cm ramp
    sl = GEN.snow_coverage(f, "depth_slide", tilt=20.0)
    assert list(sl.values) == [1.0, 1.0, 1.0, 1.0]           # ICON fills step 0, frost keeps the cover
    with pytest.raises(KeyError):
        GEN.snow_coverage(f.drop(columns="icon_snow_depth_cm"), "depth_slide", tilt=20.0)
    with pytest.raises(ValueError):
        GEN.snow_params("marion")


def test_resnow_frame_keeps_other_columns_and_reproduces_v1():
    idx = _idx(6)
    depth = pd.Series([0.0, 1.0, 3.0, 6.0, 0.0, 0.0], index=idx)
    free = pd.Series([0.0, 1e6, 2e6, 2e6, 1e6, 0.0], index=idx)
    cov = (depth / 5.0).clip(0, 1)
    from pv import snow
    fac = snow.loss_factor(cov)
    fr = pd.DataFrame({"ghi": [0.0, 100.0, 200.0, 200.0, 100.0, 0.0], "temp_air": -3.0, "snow_depth_cm": depth, "poa_global": 0.0,
                       "snow_coverage": cov, "snow_factor": fac, "power_park_free": free, "power_park": free * fac}, index=idx)
    same = GEN.resnow_frame(fr, "depth_linear")
    pd.testing.assert_frame_equal(same, fr)
    new = GEN.resnow_frame(fr, "depth_slide", tilt=25.0, icon_depth_cm=pd.Series(0.0, index=idx))
    assert list(new.columns) == list(fr.columns)
    pd.testing.assert_frame_equal(new[["ghi", "temp_air", "snow_depth_cm", "poa_global", "power_park_free"]],
                                  fr[["ghi", "temp_air", "snow_depth_cm", "poa_global", "power_park_free"]])
    assert new["snow_coverage"].iloc[4] == 1.0               # frost after the 6 cm step: cover persists
    assert (new["power_park"] <= fr["power_park"] + 1e-9).all()
