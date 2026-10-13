"""pv package: time conventions, transposition, arrays/terrain geometry, module/inverter
models, aging, soiling sign, config repair, chain energy balance, angle estimation."""

import numpy as np
import pandas as pd
import pytest

from pv import aging, angles, arrays, chain, config, inverter, metrics, modules, soiling, timebase, transposition

LAT, LON, ALT = 50.0, 10.0, 300.0


def _idx(days=3, start="2024-06-01"):
    return timebase.hourly_index(start, pd.Timestamp(start) + pd.Timedelta(days=days))


def _clear_driver(days=3):
    idx = _idx(days)
    cs = timebase.clearsky(idx, LAT, LON, ALT)
    drv = pd.DataFrame({"ghi": cs["ghi"], "dhi": cs["dhi"], "temp_air": 20.0, "wind_speed": 2.0,
                        "albedo": 0.2, "imputed_share": 0.0}, index=idx)
    return drv


# ------------------------------------------------------------- timebase

def test_hourly_mean_end_labels_shift_to_start():
    t = pd.date_range("2024-01-01 00:30", periods=4, freq="30min", tz="UTC")   # END labels of [00:00,00:30],...
    s = pd.Series([1.0, 3.0, 5.0, 7.0], index=t)
    h = timebase.to_hourly_mean(s, label="end", step="30min")
    assert h.index[0] == pd.Timestamp("2024-01-01 00:00", tz="UTC")
    assert h.iloc[0] == 2.0 and h.iloc[1] == 6.0


def test_hourly_mean_start_labels_keep_hour():
    t = pd.date_range("2024-01-01 00:00", periods=8, freq="15min", tz="UTC")
    s = pd.Series(np.arange(8.0), index=t)
    h = timebase.to_hourly_mean(s, label="start")
    assert list(h.values) == [1.5, 5.5]


def test_solar_position_uses_interval_centre():
    idx = timebase.hourly_index("2024-06-21", "2024-06-22")
    sp = timebase.solar_position(idx, LAT, LON, ALT)
    # solar noon at 10 E is ~11:20 UTC; the 11:00 interval (centre 11:30) must be the daily minimum zenith
    assert sp["apparent_zenith"].idxmin().hour == 11


# -------------------------------------------------------- transposition

def test_dni_closure_zero_at_night_and_limited():
    drv = _clear_driver(1)
    sp = timebase.solar_position(drv.index, LAT, LON, ALT)
    cs = timebase.clearsky(drv.index, LAT, LON, ALT)
    dni = transposition.dni_from_closure(drv["ghi"], drv["dhi"], sp["apparent_zenith"], cs["dni"])
    assert (dni[sp["apparent_zenith"] > 90] == 0).all()
    assert dni.max() <= cs["dni"].max() * 1.1 + 1e-6


def test_poa_south_tilt_beats_horizontal_in_winter():
    idx = timebase.hourly_index("2024-01-10", "2024-01-11")
    cs = timebase.clearsky(idx, LAT, LON, ALT)
    sp = timebase.solar_position(idx, LAT, LON, ALT)
    dni = transposition.dni_from_closure(cs["ghi"], cs["dhi"], sp["apparent_zenith"], cs["dni"])
    p0 = transposition.poa(cs["ghi"], cs["dhi"], dni, sp, 0.0, 180.0, 0.2)
    p30 = transposition.poa(cs["ghi"], cs["dhi"], dni, sp, 30.0, 180.0, 0.2)
    assert p30["poa_global"].sum() > 1.3 * p0["poa_global"].sum()


def test_horizon_blocks_direct_and_reduces_sky():
    drv = _clear_driver(1)
    sp = timebase.solar_position(drv.index, LAT, LON, ALT)
    dni = transposition.dni_from_closure(drv["ghi"], drv["dhi"], sp["apparent_zenith"])
    hz = pd.Series(30.0, index=np.arange(0, 360, 7.5))          # 30 deg wall all around
    free = transposition.poa(drv["ghi"], drv["dhi"], dni, sp, 25.0, 180.0, 0.2)
    sh = transposition.poa(drv["ghi"], drv["dhi"], dni, sp, 25.0, 180.0, 0.2, horizon=hz)
    low = (90.0 - sp["apparent_zenith"] < 30.0) & (sp["apparent_zenith"] < 90.0)
    assert (sh.loc[low, "poa_direct"] == 0).all()
    assert sh["horizon_shaded"].sum() > 0
    assert sh["poa_diffuse"].sum() < free["poa_diffuse"].sum()
    assert abs(transposition.sky_view_factor(hz) - (1 - np.sin(np.radians(30)) ** 2)) < 1e-9


# --------------------------------------------------------------- arrays

def test_effective_angles_flat_ground_identity():
    t, a = arrays.effective_angles(np.zeros(3), np.array([0.0, 90.0, 200.0]), 25.0, 180.0)
    assert np.allclose(t, 25.0) and np.allclose(a, 180.0)


def test_effective_angles_south_slope_adds_tilt():
    # ground falling to the south (aspect 180) by 10 deg: south-facing rows get 35 deg
    t, a = arrays.effective_angles(np.array([10.0]), np.array([180.0]), 25.0, 180.0)
    assert abs(t[0] - 35.0) < 1e-6 and abs(a[0] - 180.0) < 1e-6
    # ground falling north by 10 deg: 15 deg
    t, a = arrays.effective_angles(np.array([10.0]), np.array([0.0]), 25.0, 180.0)
    assert abs(t[0] - 15.0) < 1e-6
    # ground falling east: azimuth turns east, tilt grows slightly
    t, a = arrays.effective_angles(np.array([10.0]), np.array([90.0]), 25.0, 180.0)
    assert 150.0 < a[0] < 180.0 and t[0] > 25.0


def test_cluster_arrays_shares_sum_to_one():
    rng = np.random.default_rng(0)
    tilt = rng.normal(25, 4, 500)
    az = rng.normal(180, 15, 500)
    out = arrays.cluster_arrays(tilt, az, np.ones(500), 4)
    assert len(out) == 4 and abs(sum(a.share for a in out) - 1.0) < 1e-9
    te, ae = arrays.mean_angles([a.tilt for a in out], [a.azimuth for a in out], [a.share for a in out])
    assert abs(te - 25) < 2 and abs(ae - 180) < 5


def test_inter_row_factors_shade_at_low_sun():
    drv = _clear_driver(1)
    sp = timebase.solar_position(drv.index, LAT, LON, ALT)
    f = arrays.inter_row_factors(sp, 25.0, 180.0, gcr=0.5)
    low = (90.0 - sp["apparent_zenith"]).between(1, 12)
    assert (f.loc[low, "direct_shaded_fraction"] > 0).any()
    assert 0.9 < f["sky_diffuse_factor"].iloc[0] < 1.0


# -------------------------------------------------------------- modules

def test_module_lookup_bundled_and_2026():
    m = modules.module_params("REC Solar REC245PE")
    assert abs(m["STC"] - 245) < 2
    c = modules.module_params("CS6P-250P")
    assert abs(c["STC"] - 250) < 2
    with pytest.raises(KeyError):
        modules.module_params("no such module xyz")


def test_cec_dc_matches_stc_and_low_light_drops():
    m = modules.module_params("REC Solar REC245PE")
    out = modules.dc_cec(pd.Series([1000.0]), pd.Series([25.0]), m, 245.0)
    assert abs(out["p_mp"].iloc[0] - 245.0) / 245.0 < 0.03
    ll = modules.low_light_curve(m)
    assert ll.loc[1000] == pytest.approx(1.0) and 0.85 < ll.loc[200] < 1.02
    g = modules.temperature_coefficient(m)
    assert -0.006 < g < -0.003


def test_pvwatts_dc_linear():
    p = modules.dc_pvwatts(pd.Series([500.0, 1000.0]), pd.Series([25.0, 25.0]), 1000.0)
    assert np.allclose(p.values, [500.0, 1000.0])


def test_era_proxy_resolves():
    for y in (2005, 2012, 2017, 2020, 2024):
        m = modules.era_proxy_module(y)
        assert m["STC"] > 100


# ------------------------------------------------------------- inverter

def test_pick_inverter_scales_to_park():
    inv, n = inverter.pick_inverter(5_000_000.0, year=2013)
    assert n >= 1 and abs(n * inv["Paco"] - 5e6) / 5e6 < 0.25


def test_sandia_clips_and_has_night_tare():
    inv, n = inverter.pick_inverter(1_000_000.0, year=2020)
    pdc = pd.Series([0.0, 0.5e6, 1.3e6])
    vdc = pd.Series(float(inv["Vdco"]), index=pdc.index)
    ac = inverter.ac_sandia(pdc, vdc, inv, n, scale_dc=n * float(inv["Paco"]) / 1.0e6)
    assert ac.iloc[0] == 0.0 and ac.iloc[2] <= 1.0e6 * 1.001 and ac.iloc[1] < 0.5e6


# ---------------------------------------------------------------- aging

def test_degradation_monotone_and_era_ordered():
    ages = np.array([0.0, 1.0, 10.0, 25.0])
    f = aging.degradation_factor(ages, "multi_si")
    assert f[0] == 1.0 and np.all(np.diff(f) < 0)
    assert 0.90 < f[2] < 0.95                         # ~2 % LID + 0.6 %/a
    assert aging.degradation_factor(10.0, "topcon_bifacial") > aging.degradation_factor(10.0, "multi_si_early")


def test_degradation_series_exact_age():
    idx = pd.date_range("2024-01-01", periods=3, freq="365D", tz="UTC")
    s = aging.degradation_series(idx, "2014-01-01", "multi_si")
    assert s.iloc[0] < 1.0 and s.is_monotonic_decreasing


def test_inverter_availability_bounds():
    idx = _idx(60)
    a = aging.inverter_availability(idx, 10, np.random.default_rng(1), failures_per_unit_year=5.0, repair_days_mean=2.0)
    assert a.between(0, 1).all() and a.mean() < 1.0


# -------------------------------------------------------------- soiling

def test_soiling_reduces_power_not_inverts():
    idx = _idx(20)
    pm = pd.Series(20e-6 * 1e3 * 0 + 0.00002, index=idx)        # 20 ug/m3 -> g/m3
    rain = pd.Series(0.0, index=idx)
    r = soiling.hsu_ratio(rain, pm, pm, 25.0)
    assert r.between(0, 1).all() and r.iloc[-1] < 1.0
    loss = soiling.legacy_erf_loss(pm, rain, 25.0)
    assert loss.between(0, 0.35).all() and loss.is_monotonic_increasing
    p = pd.Series(1000.0, index=idx)
    assert (soiling.apply(p, 1 - loss) <= p).all()


# --------------------------------------------------------------- config

def test_legacy_config_repair():
    old = {"pv_params": {"tilt_az_list": [[[25, 180]]], "dc_rating_watts": 1436925, "dc_ac_ratio": 1, "gamma_pdc": -0.0035,
                         "albedo": 0.25, "mean_eta_inv_nom": 0.95, "eta_inv_ref": 0.9637, "faiman_u0": 25, "faiman_u1": 6.84,
                         "apply_ageing": False, "apply_soiling": False, "random_seed": 42}}
    c = config.from_legacy(old, "p", 50.0, 10.0)
    assert c["arrays"] == [{"name": "a0", "tilt": 25.0, "azimuth": 180.0, "share": 1.0, "gcr": None}]
    assert c["kwp_dc"] == pytest.approx(1436.925)


def test_normalise_shares_and_errors():
    c = config.normalise({"park_id": "x", "latitude": 50, "longitude": 10, "kwp_dc": 1, "arrays": [{"tilt": 20, "azimuth": 170, "share": 3}, {"tilt": 20, "azimuth": 190, "share": 1}]})
    assert [a["share"] for a in c["arrays"]] == [0.75, 0.25]
    with pytest.raises(ValueError):
        config.normalise({"park_id": "x", "latitude": 50, "longitude": 10, "kwp_dc": 1, "arrays": []})


# ---------------------------------------------------------------- chain

def _cfg(**over):
    base = {"park_id": "t", "latitude": LAT, "longitude": LON, "altitude": ALT, "kwp_dc": 1000.0, "dc_ac": 1.2,
            "arrays": [{"tilt": 25.0, "azimuth": 180.0, "share": 1.0}]}
    base.update(over)
    return config.normalise(base)


def test_chain_pvwatts_clear_day_plausible():
    drv = _clear_driver(2)
    r = chain.run_park(_cfg(), drv)
    assert r["p_ac_w"].max() <= 1000e3 / 1.2 * 1.001           # AC cap
    assert r["clipped"].sum() > 0                              # June clear sky with dc_ac 1.2 clips
    assert (r.loc[r["zenith"] > 90, "p_ac_w"] == 0).all()
    assert 0.1 < r["p_ac_w"].mean() / 1000e3 < 0.4


def test_chain_cec_sandia_runs_and_is_close_to_pvwatts():
    drv = _clear_driver(2)
    a = chain.run_park(_cfg(), drv)
    b = chain.run_park(_cfg(module={"model": "cec", "name": "REC Solar REC245PE", "year": 2013},
                            inverter={"model": "sandia", "unit_kw": 500}), drv)
    assert b["p_ac_w"].max() > 0
    assert abs(b["p_ac_w"].sum() / a["p_ac_w"].sum() - 1.0) < 0.12
    assert b.attrs["inverter"]["n_units"] >= 1


def test_chain_aging_and_losses_reduce_energy():
    drv = _clear_driver(2)
    a = chain.run_park(_cfg(), drv)["p_ac_w"].sum()
    b = chain.run_park(_cfg(aging={"enabled": True, "commissioning_date": "2010-01-01", "era": "multi_si"}, dc_ac=1.0), drv)["p_ac_w"].sum()
    c = chain.run_park(_cfg(losses={"dc": 0.05, "ac": 0.02, "availability": 0.95}, dc_ac=1.0), drv)["p_ac_w"].sum()
    ref = chain.run_park(_cfg(dc_ac=1.0), drv)["p_ac_w"].sum()
    assert b < ref and c < ref and a > 0


def test_chain_mixture_equals_weighted_sum_before_inverter():
    drv = _clear_driver(1)
    cfg_mix = _cfg(arrays=[{"tilt": 20.0, "azimuth": 90.0, "share": 0.5}, {"tilt": 20.0, "azimuth": 270.0, "share": 0.5}], dc_ac=1.0)
    e = chain.run_park(_cfg(arrays=[{"tilt": 20.0, "azimuth": 90.0, "share": 1.0}], dc_ac=1.0), drv)["p_dc_w"]
    w = chain.run_park(_cfg(arrays=[{"tilt": 20.0, "azimuth": 270.0, "share": 1.0}], dc_ac=1.0), drv)["p_dc_w"]
    m = chain.run_park(cfg_mix, drv)["p_dc_w"]
    assert np.allclose(m.values, 0.5 * e.values + 0.5 * w.values, rtol=1e-6, atol=1e-6)


# --------------------------------------------------------------- angles

def test_angle_estimators_recover_orientation():
    drv = _clear_driver(6)
    truth = _cfg(arrays=[{"tilt": 30.0, "azimuth": 200.0, "share": 1.0}], dc_ac=1.0)
    y = chain.run_park(truth, drv)["p_ac_w"] / 1000e3
    bank = angles.poa_bank(drv, LAT, LON, ALT, tilts=[10, 20, 30, 40], azimuths=[140, 160, 180, 200, 220, 240])
    mask = angles.clear_hours(drv["ghi"], LAT, LON, ALT)
    g = angles.grid_fit(y, bank, mask)
    assert g.iloc[0]["tilt"] == 30.0 and g.iloc[0]["azimuth"] == 200.0
    n = angles.nnls_mixture(y, bank, mask)
    assert abs(n["tilt_eff"] - 30.0) < 5 and abs(n["azimuth_eff"] - 200.0) < 8


# -------------------------------------------------------------- metrics

def test_plateau_detection():
    idx = _idx(1)
    p = pd.Series(np.r_[np.zeros(6), [0, 200, 400, 600], [800, 800, 800, 800], [600, 400, 200, 0], np.zeros(6)], index=idx)
    pl = metrics.plateaus(p, cap=1000.0)
    assert len(pl) == 1 and pl.iloc[0]["hours"] == 4
    s = metrics.plateau_summary(pl, 1000.0)
    assert s["hours_at_mode"] == 4


def test_basic_metrics_perfect():
    y = pd.Series(np.arange(20.0))
    assert metrics.basic(y, y, 100.0)["r2"] == pytest.approx(1.0)
