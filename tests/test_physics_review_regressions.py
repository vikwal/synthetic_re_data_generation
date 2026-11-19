"""Independent regression checks for errors found during the physics review."""
import numpy as np
import pandas as pd
import pytest


def test_hub_saturation_pressure_uses_celsius():
    import generate_wind as gw
    index = pd.date_range("2024-01-01", periods=2, freq="h", tz="UTC")
    frame = pd.DataFrame({
        "u_wind_10m": [5., 6.], "v_wind_10m": [0., 0.],
        "u_wind_100m": [7., 8.], "v_wind_100m": [0., 0.],
        "temp_2m": [288.15, 288.15], "dew_point_2m": [283.15, 283.15],
        "pressure": [101325., 101325.],
    }, index=index)
    output = gw.get_features(frame, {"p_s_model": "huang", "temp_gradient": .00649},
                             hub_height=102., suffix="_hub")
    # 15 degC minus 0.649 K lapse gives a saturation pressure near 1.64 kPa.
    # A Kelvin-as-Celsius call instead gives several MPa.
    np.testing.assert_allclose(output["temp_2m_hub"], 287.501)
    assert output["sat_vap_pressure_hub"].between(1600., 1660.).all()
    assert (output["sat_vap_pressure_hub"] < output["sat_vap_pressure"]).all()


def test_enabled_wakes_fail_when_inputs_are_missing():
    from round2.chain import ParkCache
    park = ParkCache.__new__(ParkCache)
    park.park_id = "fixture"
    park.index = pd.date_range("2024-01-01", periods=2, freq="h", tz="UTC")
    park.v10 = park.v100 = np.ones(2)
    park.q_target = None
    park.ages = None
    park.turbines = park.hub_heights = []
    park.turbine_groups = {}
    park.wake = {}
    with pytest.raises(FileNotFoundError, match="No wake factors"):
        park.power_series({"wake_enabled": True})
    np.testing.assert_array_equal(park.power_series({"wake_enabled": False}), [0., 0.])


def test_wake_loader_clips_nonphysical_cached_factors(monkeypatch):
    import generate_wind as gw
    index = pd.date_range("2024-01-01", periods=2, freq="h", tz="UTC")
    monkeypatch.setattr(gw.os.path, "exists", lambda path: True)
    monkeypatch.setattr(gw.pd, "read_parquet", lambda path: pd.DataFrame({"w": [-.2, 1.2]}, index=index))
    actual = gw.load_wake_factor({"wake": {"k": .0625}, "wake_dir": "/fixture"}, "fixture", index)
    np.testing.assert_array_equal(actual, [1e-6, 1.])
