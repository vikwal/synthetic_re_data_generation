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


def test_suspect_mask_catches_outage_hours_and_grid_recovers():
    from pv import plausibility as PL
    drv = _clear_driver(8)
    truth = _cfg(arrays=[{"tilt": 30.0, "azimuth": 180.0, "share": 1.0}], dc_ac=1.0)
    y = chain.run_park(truth, drv)["p_ac_w"] / 1000e3
    bank = angles.poa_bank(drv, LAT, LON, ALT, tilts=[10, 20, 30, 40], azimuths=[160, 180, 200])
    mask = angles.clear_hours(drv["ghi"], LAT, LON, ALT)
    bad = mask & (drv.index.hour.isin([9, 10, 11, 12]) & (drv.index.day <= 3))     # curtailment on three mornings
    y_bad = y.where(~bad, 0.2 * y)
    sus = PL.suspect_mask(y_bad, bank, mask)
    assert sus[bad].all() and not sus[mask & ~bad].any()
    g = angles.grid_fit(y_bad, bank, mask & ~sus)
    assert g.iloc[0]["tilt"] == 30.0 and g.iloc[0]["azimuth"] == 180.0


def test_clipping_mask_uses_limit_or_quantile():
    from pv import plausibility as PL
    idx = _idx(1)
    p = pd.Series(np.r_[np.zeros(6), np.linspace(0, 1000, 6), [1000, 1000, 1000], np.linspace(1000, 0, 6), np.zeros(3)], index=idx)
    assert PL.clipping_mask(p, ac_limit_w=1000.0).sum() == 5
    assert PL.clipping_mask(p).sum() >= 3


def test_tilt_verdict_withheld_without_zenith_spread():
    from pv import plausibility as PL
    drv = _clear_driver(10)
    truth = _cfg(arrays=[{"tilt": 25.0, "azimuth": 180.0, "share": 1.0}], dc_ac=1.0)
    y = chain.run_park(truth, drv)["p_ac_w"] / 1000e3
    res = PL.assess(y, drv, LAT, LON, ALT, register_orientation="Süd", register_tilt_class="20 - 40 Grad", n_boot=0)
    assert res["zenith_spread"] < PL.TILT_ZENITH_SPREAD and res["tilt_identifiable"] is False
    assert res["plausibility"]["tilt_ok"] is None and res["plausibility"]["azimuth_ok"] is True
    assert res["n_suspect"] == 0
    # the same data declared identifiable gives a verdict
    res2 = PL.assess(y, drv, LAT, LON, ALT, register_orientation="Süd", register_tilt_class="20 - 40 Grad", n_boot=0, zenith_spread_min=0.0)
    assert res2["plausibility"]["tilt_ok"] is True


def test_flat_east_west_is_consistent_with_east_west_register():
    from pv import plausibility as PL
    drv = _clear_driver(10)
    truth = _cfg(arrays=[{"tilt": 7.0, "azimuth": 90.0, "share": 0.5}, {"tilt": 7.0, "azimuth": 270.0, "share": 0.5}], dc_ac=1.0)
    y = chain.run_park(truth, drv)["p_ac_w"] / 1000e3
    res = PL.assess(y, drv, LAT, LON, ALT, register_orientation="Ost-West", register_tilt_class="< 20 Grad", n_boot=0, zenith_spread_min=0.0)
    assert res["mixture"]["w_flat"] > 0.3 or res["grid"]["tilt"] <= 5.0
    assert res["plausibility"]["azimuth_ok"] in (True, None)
    res_s = PL.assess(y, drv, LAT, LON, ALT, register_orientation="Süd", register_tilt_class="< 20 Grad", n_boot=0, zenith_spread_min=0.0)
    assert res_s["plausibility"]["azimuth_ok"] is not False        # flat: undecidable, never a false alarm


def test_chain_bank_recovers_orientation_and_matches_chain():
    from pv import plausibility as PL
    drv = _clear_driver(6)
    cfg = _cfg(arrays=[{"tilt": 30.0, "azimuth": 200.0, "share": 1.0, "gcr": 0.4}], dc_ac=1.0, shading={"inter_row": True})
    y = chain.run_park(cfg, drv)["p_dc_w"]
    bank = PL.chain_bank(cfg, drv, tilts=[20, 30, 40], azimuths=[180, 200, 220])
    assert set(bank) == {(float(t), float(a)) for t in (20, 30, 40) for a in (180, 200, 220)}
    mask = angles.clear_hours(drv["ghi"], LAT, LON, ALT)
    g = angles.grid_fit(y, bank, mask)
    assert g.iloc[0]["tilt"] == 30.0 and g.iloc[0]["azimuth"] == 200.0 and g.iloc[0]["r2"] > 0.999


def test_east_west_hypothesis_prefers_pair_only_for_east_west_layout():
    from pv import plausibility as PL
    drv = _clear_driver(12)
    bank = angles.poa_bank(drv, LAT, LON, ALT, PL.TILTS, PL.AZIMUTHS)
    mask = angles.clear_hours(drv["ghi"], LAT, LON, ALT)
    ew = _cfg(arrays=[{"tilt": 15.0, "azimuth": 90.0, "share": 0.5}, {"tilt": 15.0, "azimuth": 270.0, "share": 0.5}], dc_ac=1.0)
    y_ew = chain.run_park(ew, drv)["p_dc_w"]
    h = PL.east_west_hypothesis(y_ew, bank, mask)
    assert h["preferred"] and h["tilt"] == 15.0 and (h["azimuth_east"], h["azimuth_west"]) == (90.0, 270.0)
    south = _cfg(arrays=[{"tilt": 25.0, "azimuth": 180.0, "share": 1.0}], dc_ac=1.0)
    y_s = chain.run_park(south, drv)["p_dc_w"]
    assert not PL.east_west_hypothesis(y_s, bank, mask)["preferred"]
    res = PL.assess(y_ew, drv, LAT, LON, ALT, register_orientation="Ost-West", register_tilt_class="< 20 Grad", n_boot=0, zenith_spread_min=0.0)
    assert res["mixture"]["east_west"] and res["plausibility"]["azimuth_ok"] is True


def test_azimuth_prior_restricts_bank_and_recovers_tilt():
    from pv import plausibility as PL
    drv = _clear_driver(8)
    truth = _cfg(arrays=[{"tilt": 20.0, "azimuth": 190.0, "share": 1.0}], dc_ac=1.0)
    y = chain.run_park(truth, drv)["p_dc_w"]
    res = PL.assess(y, drv, LAT, LON, ALT, n_boot=0, azimuth_prior=190.0, cfg=truth)
    assert res["bank"] == "chain" and res["grid"]["azimuth"] == 190.0 and res["grid"]["tilt"] == 20.0
    res_poa = PL.assess(y, drv, LAT, LON, ALT, n_boot=0, azimuth_prior=190.0)
    assert res_poa["grid"]["azimuth"] == 190.0 and abs(res_poa["grid"]["tilt"] - 20.0) <= 5.0     # plain POA: temperature/IAM bias one step
    # with a prior the azimuth verdict is the prior's: rows south vs register east-west -> implausible, and vice versa
    v = PL.assess(y, drv, LAT, LON, ALT, n_boot=0, azimuth_prior=190.0, cfg=truth, register_orientation="Ost-West", register_tilt_class="20 - 40 Grad", zenith_spread_min=0.0)
    assert v["plausibility"]["azimuth_ok"] is False and v["plausibility"]["tilt_ok"] is True
    v = PL.assess(y, drv, LAT, LON, ALT, n_boot=0, azimuth_prior=190.0, cfg=truth, register_orientation="Süd", register_tilt_class="20 - 40 Grad", zenith_spread_min=0.0)
    assert v["plausibility"]["azimuth_ok"] is True
    v = PL.assess(y, drv, LAT, LON, ALT, n_boot=0, azimuth_prior=80.0, prior_east_west=True, cfg=truth, register_orientation="Süd", register_tilt_class="20 - 40 Grad", zenith_spread_min=0.0)
    assert v["plausibility"]["azimuth_ok"] is False
    pr = PL.azimuth_prior_from_rows(100.0)
    assert not pr["east_west"] and pr["azimuth"] == 190.0
    pr = PL.azimuth_prior_from_rows(170.0)
    assert pr["east_west"] and pr["azimuths"] == (80.0, 260.0)


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


def test_grid_region_bootstrap_repeats_days_by_multiplicity():
    from pv import plausibility as PL
    drv = _clear_driver(12)
    truth = _cfg(arrays=[{"tilt": 30.0, "azimuth": 200.0, "share": 1.0}], dc_ac=1.0)
    y = chain.run_park(truth, drv)["p_dc_w"]
    bank = angles.poa_bank(drv, LAT, LON, ALT, [20, 30, 40], [180, 200, 220])
    mask = angles.clear_hours(drv["ghi"], LAT, LON, ALT)
    reg = PL.grid_region(y, bank, mask, n_boot=20)
    assert reg["n_days"] == 12 and reg["tilt_boot"][1] == 30.0 and reg["azimuth_boot"][1] == 200.0
    # a resample that drops the multiplicity would be a 63 % subsample; with repeats every draw still holds all hours of its days
    assert reg["tilt_boot"][0] <= 30.0 <= reg["tilt_boot"][2]
    reg0 = PL.grid_region(y, bank, mask, n_boot=0)
    assert "tilt_boot" not in reg0 and reg0["n_days"] == 12


def test_plausibility_prior_reports_prior_classes_and_tilt_none_without_class():
    from pv import plausibility as PL
    region = {"tilt": 20.0, "azimuth": 180.0, "r2": 0.99, "scale": 1.0, "tilt_range": (15.0, 25.0), "azimuth_range": (170.0, 190.0),
              "tilt_boot": (15.0, 20.0, 25.0), "azimuth_boot": (170.0, 180.0, 190.0), "n_hours": 100, "n_days": 10}
    mix = {"flat": False, "east_west": False}
    v = PL.plausibility(region, mix, "Süd", "20 - 40 Grad", azimuth_prior=100.0)
    assert v["azimuth_ok"] is False and v["estimated_azimuth_classes"] == ["Ost"]       # the prior's class, not the power's
    v = PL.plausibility(region, mix, "Süd")
    assert v["azimuth_ok"] is True and v["tilt_ok"] is None                           # no register tilt class -> no tilt verdict
    assert PL._interval_distance(170.0, 190.0, 180.0) == 0.0 and PL._interval_distance(60.0, 300.0, 0.0) == 60.0


def test_angle_diagnostics_run_on_assess_mask():
    from pv import angle_diagnostics, plausibility as PL
    drv = _clear_driver(8)
    truth = _cfg(arrays=[{"tilt": 25.0, "azimuth": 180.0, "share": 1.0}], dc_ac=1.0)
    y = chain.run_park(truth, drv)["p_dc_w"]
    res = PL.assess(y, drv, LAT, LON, ALT, n_boot=0)
    bank = angles.poa_bank(drv, LAT, LON, ALT, PL.TILTS, PL.AZIMUTHS)
    d = angle_diagnostics.run(y, drv, LAT, LON, ALT, bank, res["mask"], tilt_guess=25.0)
    assert abs(d["centroid_azimuth"]["azimuth"] - 180.0) < 10 and d["centroid_azimuth"]["n_days"] >= 5
    assert "tilt" in d["noon_tilt"]


def test_block_arrays_gates():
    from pv import geodata_geometry as GG
    rng = np.random.default_rng(0)

    def tables(specs):
        rows = []
        for tilt, az, n, area in specs:
            for _ in range(n):
                rows.append({"tilt": tilt + rng.normal(0, 1.0), "azimuth": az + rng.normal(0, 1.5), "area_m2": area, "axis_deg": (az - 90.0) % 180.0 + rng.normal(0, 2.0)})
        return pd.DataFrame(rows)
    # two blocks 21 deg apart: not worth separating -> one plane with the mean normal
    cl, info = GG.block_arrays(tables([(15.0, 184.0, 30, 100.0), (15.0, 205.0, 20, 100.0)]))
    assert info["reason"] == "merged" and len(cl) == 1 and 190 < cl[0].azimuth < 198
    # two blocks 46 deg apart with enough tables each: split, area shares
    cl, info = GG.block_arrays(tables([(15.0, 184.0, 30, 100.0), (15.0, 230.0, 20, 100.0)]))
    assert info["reason"] == "split" and len(cl) == 2 and abs(cl[0].share - 0.6) < 0.02 and abs(cl[1].azimuth - 230.0) < 3
    # a small second block (3 tables, 9 % of the area) merges into the main block
    cl, info = GG.block_arrays(tables([(15.0, 184.0, 30, 100.0), (15.0, 250.0, 3, 100.0)]))
    assert info["reason"] == "merged" and len(cl) == 1
    # an east-west tent: two faces 180 deg apart -> split
    cl, info = GG.block_arrays(tables([(10.0, 90.0, 25, 50.0), (10.0, 270.0, 25, 50.0)]))
    assert info["reason"] == "split" and len(cl) == 2 and {round(c.azimuth / 10) * 10 for c in cl} == {90, 270}
    # noisy tilts in one block (standard error of the median above 3 deg) -> collapsed
    t = tables([(15.0, 184.0, 30, 100.0), (15.0, 230.0, 20, 100.0)])
    t.loc[t.azimuth > 210, "tilt"] = rng.uniform(2.0, 40.0, (t.azimuth > 210).sum())
    cl, info = GG.block_arrays(t)
    assert info["reason"] == "collapsed:tilt_se" and len(cl) == 1


# ------------------------------------------------------------- 30-minute step

def test_step_helpers_30min():
    idx = timebase.step_index("2024-06-21", "2024-06-22", "30min")
    assert len(idx) == 48 and timebase.step_of(idx) == pd.Timedelta("30min")
    assert timebase.centre(idx)[0] == pd.Timestamp("2024-06-21 00:15", tz="UTC")
    t = pd.date_range("2024-01-01 00:15", periods=4, freq="15min", tz="UTC")      # END labels of 15-min intervals
    s = pd.Series([1.0, 3.0, 5.0, 7.0], index=t)
    h = timebase.to_step_mean(s, label="end", src_step="15min", step="30min")
    assert list(h.values) == [2.0, 6.0] and h.index[0] == pd.Timestamp("2024-01-01 00:00", tz="UTC")
    sp = timebase.solar_position(idx, LAT, LON, ALT)
    assert sp["apparent_zenith"].idxmin() == pd.Timestamp("2024-06-21 11:00", tz="UTC")   # centre 11:15, solar noon 11:20
    cs30 = timebase.clearsky(idx, LAT, LON, ALT)
    cs60 = timebase.clearsky(timebase.hourly_index("2024-06-21", "2024-06-22"), LAT, LON, ALT)
    assert abs(cs30["ghi"].sum() * 0.5 / cs60["ghi"].sum() - 1.0) < 1e-3


def test_chain_30min_matches_hourly_energy():
    idx = timebase.step_index("2024-06-20", "2024-06-23", "30min")
    cs = timebase.clearsky(idx, LAT, LON, ALT)
    drv = pd.DataFrame({"ghi": cs["ghi"], "dhi": cs["dhi"], "temp_air": 20.0, "wind_speed": 2.0, "albedo": 0.2, "imputed_share": 0.0}, index=idx)
    cfg = _cfg(arrays=[{"tilt": 25.0, "azimuth": 180.0, "share": 1.0, "gcr": 0.4}], dc_ac=1.2, shading={"inter_row": True})
    r30 = chain.run_park(cfg, drv)
    r60 = chain.run_park(cfg, _clear_driver(3).loc["2024-06-20":"2024-06-22"]) if False else chain.run_park(cfg, pd.DataFrame(
        {"ghi": timebase.clearsky(timebase.hourly_index("2024-06-20", "2024-06-23"), LAT, LON, ALT)["ghi"],
         "dhi": timebase.clearsky(timebase.hourly_index("2024-06-20", "2024-06-23"), LAT, LON, ALT)["dhi"],
         "temp_air": 20.0, "wind_speed": 2.0, "albedo": 0.2, "imputed_share": 0.0}, index=timebase.hourly_index("2024-06-20", "2024-06-23")))
    e30, e60 = r30["p_ac_w"].sum() * 0.5, r60["p_ac_w"].sum()
    assert abs(e30 / e60 - 1.0) < 0.01
    assert r30["clipped"].sum() * 0.5 >= r60["clipped"].sum() * 0.8          # the finer step resolves the clipping plateau at least as well


def test_instant_to_step_means_and_era5_solar_repeat():
    from pv import drivers as D
    t = pd.date_range("2024-01-01 00:00", periods=4, freq="1h", tz="UTC")
    inst = pd.DataFrame({"x": [0.0, 2.0, 4.0, 6.0]}, index=t)
    m = D.instant_to_step_means(inst, "30min")
    assert list(m.index[:3]) == list(pd.date_range("2024-01-01 00:00", periods=3, freq="30min", tz="UTC"))
    assert list(m["x"].values[:4]) == [0.5, 1.5, 2.5, 3.5]                     # means over [t, t+30min)
    h = D.instant_to_step_means(inst, "1h")
    assert list(h["x"].values[:3]) == [1.0, 3.0, 5.0]
