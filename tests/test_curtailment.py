"""Curtailment layers: switch-off identity, grid setpoints and thinning, AR(1)
disturbance, market thresholds and blocks, bat masks, reproducibility, node
clustering, calibration on a toy example, and (slow) bit identity of the
existing outputs."""

import copy
import datetime as dt
import hashlib
import json
import os
import subprocess

import numpy as np
import pandas as pd
import pytest
import yaml

import generate_wind as gw
from curtailment import apply as curt_apply
from curtailment import areas, calibrate, config, environment, grid, market, streams, timegrid

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _cfg(**over):
    c = config.get_curtailment_params({"curtailment": {"enabled": True, **over}})
    return c


# ---------------------------------------------------------------- config / switch-off

def test_disabled_returns_same_object():
    df = pd.DataFrame({"power_park": [1.0, 2.0]})
    assert gw.apply_curtailment(df, {}, {}) is df
    assert gw.apply_curtailment(df, {"curtailment": {"enabled": False, "grid": {"target_scale": 4}}}, {}) is df


def test_defaults_merge_and_validation():
    c = config.get_curtailment_params({"curtailment": {"grid": {"target_scale": 4.0, "areas": {
        "A1 SH": {"p0": 0.3}}}}})
    assert c["enabled"] is False and c["grid"]["target_scale"] == 4.0
    assert c["grid"]["areas"]["A1_SH"]["p0"] == 0.3 and c["grid"]["areas"]["A1_SH"]["a"] == -0.38   # deep merge
    assert c["grid"]["setpoint_first"] == {0.0: 0.78, 0.3: 0.12, 0.6: 0.10}
    assert c["market"]["cohorts"][-1]["from"] == dt.date(2025, 2, 25)
    for bad in ({"grid": {"setpoint_first": {0.0: 0.5, 0.3: 0.1}}}, {"step_minutes": 60},
                {"grid": {"areas": {"A9_X": {"p0": 1.0}}}}, {"market": {"post_eeg": "x"}},
                {"environment": {"bat": {"levels": {"standard": {"p": 0.9}}}}}):
        with pytest.raises(ValueError):
            config.get_curtailment_params({"curtailment": bad})
    assert config.area_key("A3 NI-O/ST") == "A3_NI_O_ST" and config.area_key("A6 Sued") == "A6_SUED"


def test_chain_configs_resolve():
    for name, scale in (("PARKS_v1_curt.yaml", 1.0), ("PARKS_v1_curt_x4.yaml", 4.0)):
        with open(os.path.join(REPO, "configs", "round2", name)) as f:
            chain = yaml.safe_load(f)
        c = config.get_curtailment_params(chain)
        assert c["enabled"] and c["grid"]["target_scale"] == scale
        with open(os.path.join(REPO, "configs", "round2", "PARKS_v1.yaml")) as f:
            assert chain["round2"] == yaml.safe_load(f)["round2"]


# ---------------------------------------------------------------- time grid / grid layer

def test_quarter_hour_mean_and_expand():
    h = pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC")
    q = timegrid.quarter_index(h)
    assert len(q) == 12 and q[1] - q[0] == pd.Timedelta(minutes=15)
    v = timegrid.to_quarter([1.0, 2.0, 3.0])
    np.testing.assert_array_equal(timegrid.to_hour(v), [1.0, 2.0, 3.0])
    np.testing.assert_allclose(timegrid.to_hour([0, 0, 1, 1, 4, 4, 4, 0]), [0.5, 3.0])


def test_setpoint_min_rule():
    p = np.array([0.0, 2.0, 5.0, 9.0])
    np.testing.assert_array_equal(grid.apply_setpoint(p, np.zeros(4), 10.0), np.zeros(4))
    np.testing.assert_array_equal(grid.apply_setpoint(p, np.ones(4), 10.0), p)
    np.testing.assert_array_equal(grid.apply_setpoint(p, np.full(4, 0.3), 10.0), [0.0, 2.0, 3.0, 3.0])


def test_setpoint_series_lowest_wins_and_release():
    ev = pd.DataFrame({"start": [2, 4], "n_main": [6, 2], "s0": [0.6, 0.0], "n_rel": [0, 3], "s_rel": [1.0, 0.3],
                       "event_id": ["a", "b"]})
    s, ids = grid.setpoint_series(14, ev)
    np.testing.assert_array_equal(s, [1, 1, .6, .6, 0, 0, .3, .3, .3, 1, 1, 1, 1, 1])
    assert list(ids[:9]) == ["", "", "a", "a", "b", "b", "b", "b", "b"]
    v = grid.segments_covering(14, ev["start"], ev["n_main"], ev["s0"], ev["n_rel"], ev["s_rel"])
    np.testing.assert_array_equal(v[0], s)


def _toy_drivers(days=60, seed=0):
    h = pd.date_range("2024-01-01", periods=24 * days, freq="h", tz="UTC")
    q = timegrid.quarter_index(h)
    r = np.random.default_rng(seed)
    cf = np.clip(0.3 + 0.25 * np.sin(np.arange(len(q)) / 300.0) + 0.05 * r.standard_normal(len(q)), 0, 0.9)
    price = 60 + 80 * np.sin(np.arange(len(q)) / 97.0)
    return h, q, pd.DataFrame({"cf_da": cf, "price": price, "r_m": 70.0}, index=q)


def test_thinning_superset_and_monotone():
    c = _cfg()
    h, q, drv = _toy_drivers()
    acfg = c["grid"]["areas"]["A1_SH"]
    u = grid.u_on_slots(q, ["A1_SH"], c["grid"]["disturbance"], c["seed"])["A1_SH"]
    rate0 = grid.base_rate(drv["cf_da"].to_numpy(), u, acfg, 0.01)
    ev1 = grid.node_events(c["seed"], "N", q, rate0, np.full(len(q), 1.0), acfg, c["grid"])
    ev4 = grid.node_events(c["seed"], "N", q, rate0, np.full(len(q), 4.0), acfg, c["grid"])
    assert len(ev1) > 0 and set(ev1.event_id) <= set(ev4.event_id) and len(ev4) > len(ev1)
    s1, _ = grid.setpoint_series(len(q), ev1)
    s4, _ = grid.setpoint_series(len(q), ev4)
    assert np.all(s4 <= s1)
    p = np.full(len(q), 1.0)
    assert grid.apply_setpoint(p, s4, 1.0).sum() <= grid.apply_setpoint(p, s1, 1.0).sum()
    # identical shapes of the common events (c-independent draws)
    m = ev4.set_index("event_id").loc[ev1.event_id]
    np.testing.assert_array_equal(m["n_main"].to_numpy(), ev1["n_main"].to_numpy())


def test_event_shapes_follow_config():
    c = _cfg()
    acfg = c["grid"]["areas"]["A1_SH"]
    e = grid.event_shapes(1, ("x",), np.arange(200_000), acfg, c["grid"])
    s0 = pd.Series(e["s0"]).value_counts(normalize=True)
    assert s0[0.0] == pytest.approx(0.78, abs=0.01) and s0[0.6] == pytest.approx(0.10, abs=0.01)
    med = np.median(e["n_main"]) * 0.25
    assert med == pytest.approx(np.exp(acfg["dur"][0]), rel=0.1)
    assert e["n_main"].max() * 0.25 <= c["grid"]["max_duration_h"] and e["n_main"].min() >= 1
    rel = e["n_rel"] > 0
    assert rel[e["s0"] > 0].sum() == 0
    assert rel[e["s0"] == 0].mean() == pytest.approx(acfg["release"]["p"], abs=0.01)
    assert (e["s_rel"][rel] == 0.3).mean() == pytest.approx(acfg["release"]["p30"], abs=0.02)


def test_ar1_moments_and_cross_correlation():
    d = {"phi": 0.4, "sigma_z": 1.0, "rho_z": 0.4}
    days = pd.date_range("2000-01-01", periods=40_000, freq="D")
    u = grid.disturbance(days, ["A", "B"], d, seed=7)
    z = {k: np.log(v) + 0.5 for k, v in u.items()}
    assert np.std(z["A"]) == pytest.approx(1.0, abs=0.03)
    assert np.mean(u["A"]) == pytest.approx(1.0, abs=0.05)
    assert np.corrcoef(z["A"][1:], z["A"][:-1])[0, 1] == pytest.approx(0.4, abs=0.03)
    assert np.corrcoef(z["A"], z["B"])[0, 1] == pytest.approx(0.4, abs=0.03)
    # a day's value does not depend on the requested period
    sub = grid.disturbance(days[100:200], ["A"], d, seed=7)["A"]
    np.testing.assert_array_equal(sub, u["A"][100:200])


# ---------------------------------------------------------------- market

COH = config._normalise(copy.deepcopy(config.CURTAILMENT_DEFAULTS))["market"]["cohorts"]


def test_cohort_rules():
    assert market.cohort_of("2018-05-01", 2999.0, COH)["rule_h"] is None          # < 3 MW: no 6-h rule
    assert market.cohort_of("2018-05-01", 3000.0, COH)["rule_h"] == 6
    assert market.cohort_of("2025-02-24", 4000.0, COH)["rule_h"] == 3
    assert market.cohort_of("2025-02-25", 4000.0, COH)["rule_h"] == 0.25           # every negative quarter-hour
    assert market.cohort_of("2004-06-01", 1500.0, COH)["post_eeg_from"] == dt.date(2025, 1, 1)
    assert market.cohort_of("1996-06-01", 500.0, COH)["post_eeg_from"] <= dt.date(2021, 1, 1)


def test_negative_blocks_and_rule_threshold():
    q = pd.date_range("2024-05-01", periods=4 * 24, freq="15min", tz="UTC")
    price = np.full(len(q), 50.0)
    price[4 * 2:4 * 7] = -5.0          # 5-h block
    price[4 * 10:4 * 16] = -5.0        # 6-h block
    bid, starts, lens = market.neg_blocks(price)
    assert list(lens) == [5.0, 6.0] and list(starts) == [8, 40]
    blen = np.where(bid >= 0, lens[np.maximum(bid, 0)], 0.0)
    coh = market.cohort_of("2018-05-01", 3600.0, COH)                              # 6-h rule, strike 7.6 ct
    p = market.threshold(q, price, np.full(len(q), 50.0), blen, coh, "unsubsidized")
    assert np.all(p[8:28] == -(10 * 7.6 - 50.0))                                   # 5 h < 6 h: -premium = -26
    assert np.all(p[40:64] == 0.0)                                                 # whole 6-h block at 0
    assert not np.any(price[8:28] < p[8:28]) and np.all(price[40:64] < p[40:64])
    # premium can not become a positive threshold
    p2 = market.threshold(q, price, np.full(len(q), 200.0), blen, coh, "unsubsidized")
    assert np.all(p2[8:28] == 0.0)


def test_post_eeg_switch_on_first_of_january():
    q = pd.date_range("2024-12-31 22:00", periods=16, freq="15min", tz="UTC")       # 23:00-02:45 local
    coh = market.cohort_of("2004-06-01", 1500.0, COH)
    price = np.full(len(q), -1.0)
    p = market.threshold(q, price, np.full(len(q), 60.0), np.full(len(q), 4.0), coh, "unsubsidized")
    local = q.tz_convert("Europe/Berlin")
    assert np.all(p[local.year == 2024] == -(85.0 - 60.0)) and np.all(p[local.year == 2025] == 0.0)
    pf = market.threshold(q, price, np.full(len(q), 60.0), np.full(len(q), 4.0), coh, "feed_in")
    assert np.all(np.isneginf(pf[local.year == 2025]))


def test_market_all_or_nothing_per_block():
    c = _cfg()
    q = pd.date_range("2024-05-01", periods=4 * 48, freq="15min", tz="UTC")
    price = np.full(len(q), 30.0)
    price[40:72] = -60.0
    price[120:140] = -150.0
    drv = pd.DataFrame({"price": price, "r_m": 60.0, "cf_da": 0.4}, index=q)
    blocks = market.neg_blocks(price)
    coh = [market.cohort_of("2004-01-01", 1500, COH), market.cohort_of("2018-01-01", 2000, COH)]
    pg = np.ones((len(q), 2))
    reacted = []
    for k in range(40):
        m, rec = market.park_masks(q, drv, blocks, coh, pg, f"P{k}", c, (-1.5, 0.8, 1.5))
        for b in rec.itertuples():
            sl = blocks[0] == b.block
            exposed = price[sl, None] < np.column_stack([market.threshold(
                q[sl], price[sl], drv["r_m"].to_numpy()[sl], np.full(sl.sum(), blocks[2][b.block]), co,
                "unsubsidized") for co in coh])
            assert np.array_equal(m[sl], exposed & b.react)
            reacted.append(b.react)
    assert 0 < np.mean(reacted) < 1


def test_expected_response_matches_mc():
    r = np.random.default_rng(1)
    eta = 1.5 * r.standard_normal(400_000)
    mc = market.response_prob(30.0, -1.5, 0.8, eta).mean()
    assert market.expected_response(30.0, -1.5, 0.8, 1.5) == pytest.approx(mc, abs=0.003)


# ---------------------------------------------------------------- environment

def test_sunset_kiel_midsummer():
    days = pd.DatetimeIndex(["2024-06-21"])
    rise, sset = environment.sun_times(54.32, 10.14, days)
    local = pd.Timestamp(sset[0], tz="UTC").tz_convert("Europe/Berlin")
    assert abs((local - pd.Timestamp("2024-06-21 22:02", tz="Europe/Berlin")).total_seconds()) < 6 * 60
    r = pd.Timestamp(rise[0], tz="UTC").tz_convert("Europe/Berlin")
    assert abs((r - pd.Timestamp("2024-06-21 04:44", tz="Europe/Berlin")).total_seconds()) < 6 * 60


def test_bat_mask_season_night_wind_temperature():
    c = _cfg()
    lev = c["environment"]["bat"]["levels"]["standard"]
    h = pd.date_range("2024-06-21 00:00", "2024-06-22 23:00", freq="h", tz="Europe/Berlin").tz_convert("UTC")
    q = timegrid.quarter_index(h)
    base = environment.night_season_mask(q, 54.32, 10.14, lev)
    loc = q.tz_convert("Europe/Berlin")
    at = lambda s: base[loc == pd.Timestamp(s, tz="Europe/Berlin")][0]           # noqa: E731
    assert not at("2024-06-21 20:45") and at("2024-06-21 21:15")                    # sunset 22:02 - 1 h
    assert at("2024-06-22 04:30") and not at("2024-06-22 05:00")
    assert not at("2024-06-21 13:00")
    q2 = timegrid.quarter_index(pd.date_range("2024-11-02", periods=24, freq="h", tz="UTC"))
    assert not environment.night_season_mask(q2, 54.32, 10.14, lev).any()            # out of season
    park = {"park_id": "P", "latitude": 54.32, "longitude": 10.14,
            "groups": [{"group_id": "t1", "commissioning_date": "2015-01-01", "hub_height": 100.0},
                       {"group_id": "t2", "commissioning_date": "2005-01-01", "hub_height": 100.0}]}
    night = np.asarray(loc.hour < 3)[::4]
    frame = pd.DataFrame({"wind_speed_hub_t1": np.where(night, 5.9, 6.0), "wind_speed_hub_t2": 1.0,
                          "temp_2m": 288.15}, index=h)
    for lvl in ("standard", "mild", "strict"):
        c["environment"]["bat"]["levels"][lvl] = {**lev, "p": c["environment"]["bat"]["levels"][lvl]["p"]}
    m, name = environment.bat_masks(q, frame, park, c)
    assert name in ("standard", "mild", "strict")
    assert not m[:, 1].any()                                                        # 2005 group: no permit
    assert m[timegrid.to_quarter(night), 0].all()
    assert not m[timegrid.to_quarter(~night) & ~base, 0].any()
    assert not m[timegrid.to_quarter(~night), 0].any()                              # 6.0 m/s is not < 6
    cold = frame.assign(temp_2m=273.15 + 9.99 + 98 * 0.00649)                       # hub temperature 9.99 C
    assert not environment.bat_masks(q, cold, park, c)[0].any()


# ---------------------------------------------------------------- nodes

def test_single_linkage_chain():
    lat = np.array([54.0, 54.0, 54.0, 50.0])
    lon = np.array([9.0, 9.0 + 8 / 65.4, 9.0 + 16 / 65.4, 9.0])                     # ~8 km steps at 54 N
    n = areas.cluster_nodes(["c", "b", "a", "d"], lat, lon, 10.0)
    assert n.to_dict() == {"c": "a", "b": "a", "a": "a", "d": "d"}
    perm = areas.cluster_nodes(["d", "a", "b", "c"], lat[::-1], lon[::-1], 10.0)
    assert perm.to_dict() == n.to_dict()
    assert areas.haversine_km(54.0, 9.0, 54.0, 9.0 + 8 / 65.4) == pytest.approx(8.0, abs=0.1)


# ---------------------------------------------------------------- end to end on a synthetic plan

def _toy_plan(cfg, h, q, drv, park_ids, c=2.0):
    years = np.unique(timegrid.utc_years(q))
    calib = {"grid": {a: {int(y): {"c": c} for y in years} for a in cfg["grid"]["areas"]},
             "market": {"theta0": -1.0, "theta1": 1.0, "tau": 1.0}}
    nodes = pd.DataFrame({"park_id": park_ids, "node": park_ids, "area": "A1_SH", "node_area": "A1_SH"})
    u = grid.u_on_slots(q, ["A1_SH"], cfg["grid"]["disturbance"], cfg["seed"])
    d = drv.copy()
    d["price"] = np.where(np.arange(len(q)) % 400 < 30, -80.0, d["price"])
    return curt_apply.Plan(cfg=cfg, qidx=q, drv=d, blocks=market.neg_blocks(d["price"].to_numpy()), u=u,
                           calib=calib, nodes=nodes.set_index("park_id"))


def _toy_park(pid, h, seed):
    r = np.random.default_rng(seed)
    p1 = np.clip(r.gamma(1.5, 6e5, len(h)), 0, 2e6)
    p2 = np.clip(r.gamma(1.5, 6e5, len(h)), 0, 3e6)
    df = pd.DataFrame({"temp_2m": 290.0, "wind_speed_hub_t1": r.uniform(2, 12, len(h)),
                       "wind_speed_hub_t2": r.uniform(2, 12, len(h)), "power_t1": p1, "power_t2": p2}, index=h)
    df["power_park_free"] = p1 + p2
    df["wake_factor"] = 0.9
    df["power_park"] = df["power_park_free"] * 0.9
    df["availability"] = 1.0
    meta = {"park_id": pid, "latitude": 54.3, "longitude": 9.5, "capacity_kw": 5000.0,
            "groups": [{"group_id": "t1", "commissioning_date": "2004-03-01", "unit_kw": 2000.0, "n": 1,
                        "hub_height": 100.0},
                       {"group_id": "t2", "commissioning_date": "2017-03-01", "unit_kw": 3000.0, "n": 1,
                        "hub_height": 120.0}]}
    return df, meta


def test_curtail_park_consistency():
    c = _cfg()
    h, q, drv = _toy_drivers(days=120)
    h = h.tz_convert("UTC")
    plan = _toy_plan(c, h, q, drv, ["P1", "P2"], c=40.0)
    df, meta = _toy_park("P1", h, 3)
    out, info = curt_apply.curtail_park(df, meta, plan, c, return_info=True)
    assert list(out.columns[:len(df.columns)]) == list(df.columns)
    assert list(out.columns[len(df.columns):]) == curt_apply.NEW_COLS
    for col in df.columns.drop("power_park"):
        pd.testing.assert_series_equal(out[col], df[col])
    np.testing.assert_array_equal(out["power_park_avail"], df["power_park"])
    assert (out["power_park"] <= out["power_park_avail"]).all() and (out["power_park"] >= 0).all()
    loss = out[["loss_env", "loss_mkt", "loss_grid"]].sum(axis=1)
    np.testing.assert_allclose(loss, out["power_park_avail"] - out["power_park"], atol=1e-6)
    assert (out[["loss_env", "loss_mkt", "loss_grid"]] >= -1e-9).all().all()
    assert out["loss_grid"].sum() > 0 and out["loss_mkt"].sum() > 0
    assert ((out["curt_flag"] == 1) == (loss > 0)).all()
    assert (out.loc[out["grid_setpoint"] < 1, "grid_event_id"] != "").all()
    assert (out.loc[out["grid_setpoint"] == 1, "grid_event_id"] == "").all()
    assert info["n_grid_events"] > 0 and len(info["cohorts"]) == 2
    # "only market" is exactly avail x mkt_factor when no other layer acts in the hour
    calm = (out["loss_env"] == 0) & (out["loss_grid"] == 0)
    np.testing.assert_allclose(out.loc[calm, "power_park"],
                               out.loc[calm, "power_park_avail"] * out.loc[calm, "mkt_factor"], rtol=1e-12)
    # layers can be switched off one by one
    c2 = copy.deepcopy(c)
    c2["layers"] = {"environment": False, "market": False, "grid": True}
    plan2 = _toy_plan(c2, h, q, drv, ["P1"], c=40.0)
    o2 = curt_apply.curtail_park(df, meta, plan2, c2)
    assert o2["loss_env"].sum() == 0 and o2["loss_mkt"].sum() == 0
    assert (o2["loss_grid"] >= out["loss_grid"] - 1e-6).all()          # more power reaches the grid cap


def test_reproducible_and_order_independent():
    c = _cfg()
    h, q, drv = _toy_drivers(days=40)
    parks = {p: _toy_park(p, h, i) for i, p in enumerate(["P1", "P2", "P3"])}
    plan_a = _toy_plan(c, h, q, drv, ["P1", "P2", "P3"], c=20.0)
    plan_b = _toy_plan(c, h, q, drv, ["P3", "P2", "P1"], c=20.0)
    out_a = {p: curt_apply.curtail_park(*parks[p], plan_a, c) for p in ["P1", "P2", "P3"]}
    out_b = {p: curt_apply.curtail_park(*parks[p], plan_b, c) for p in ["P3", "P1", "P2"]}
    for p in parks:
        pd.testing.assert_frame_equal(out_a[p], out_b[p])
    c2 = _cfg(seed=c["seed"] + 1)
    plan_c = _toy_plan(c2, h, q, drv, ["P1"], c=20.0)
    assert not curt_apply.curtail_park(*parks["P1"], plan_c, c2)["power_park"].equals(out_a["P1"]["power_park"])


def test_streams_are_keyed_not_ordered():
    a = streams.rng(1, "beta", "N1").random()
    streams.rng(1, "beta", "N2").random()
    assert streams.rng(1, "beta", "N1").random() == a
    u = streams.uniform_at(1, ("grid", "N"), np.arange(1000))
    np.testing.assert_array_equal(streams.uniform_at(1, ("grid", "N"), np.arange(500, 600)), u[500:600])
    assert 0 <= u.min() and u.max() < 1 and abs(u.mean() - 0.5) < 0.03


# ---------------------------------------------------------------- calibration

def test_grid_calibration_hits_toy_target():
    c = _cfg()
    c["grid"]["calibration"]["n_virtual_nodes"] = 150
    h, q, drv = _toy_drivers(days=90)
    acfg = c["grid"]["areas"]["A1_SH"]
    acfg["target_pct"] = {2024: 4.0}
    u = grid.u_on_slots(q, ["A1_SH"], c["grid"]["disturbance"], c["seed"])["A1_SH"]
    prof = np.clip(np.stack([drv["cf_da"].to_numpy() * f for f in (0.8, 1.0, 1.2)]), 0, 1)
    res = calibrate.calibrate_area(c, "A1_SH", q, drv["cf_da"].to_numpy(), u, prof, log=lambda *_: None)
    r = res["years"][2024]
    assert r["status"] == "converged" and abs(r["achieved"] / 0.04 - 1) <= 0.02
    c4 = copy.deepcopy(c)
    c4["grid"]["target_scale"] = 4.0
    r4 = calibrate.calibrate_area(c4, "A1_SH", q, drv["cf_da"].to_numpy(), u, prof, log=lambda *_: None)
    assert r4["years"][2024]["c"] > r["c"]


# ---------------------------------------------------------------- bit identity (slow)

def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


@pytest.mark.slow
def test_parks_v1_assemble_unchanged():
    """The parks_v1 assemble path still reproduces the released frame bit for bit,
    and apply_curtailment (off) passes it through as the same object."""
    from parks import assemble, paths
    lk = "SEL900568481641"
    free_p = os.path.join(paths.RUN_DIR, "free", f"free_{lk}.parquet")
    if not os.path.exists(free_p):
        pytest.skip("PARKS_v1 run directory not on disk")
    with open(os.path.join(paths.RELEASE_DIR, f"manifest_{lk}.json")) as f:
        man = json.load(f)
    free = pd.read_parquet(free_p)
    w = pd.read_parquet(os.path.join(paths.RUN_DIR, "wakes", f"w_{lk}_k0.075.parquet"))["w"]
    rel = assemble.release_frame(free, w, [g["group_id"] for g in man["groups"]])
    assert assemble.frame_sha256(rel) == man["release_sha256"]
    with open(os.path.join(REPO, "configs", "round2", "PARKS_v1.yaml")) as f:
        chain = yaml.safe_load(f)
    assert gw.apply_curtailment(rel, chain, {}) is rel
    released = pd.read_parquet(os.path.join(paths.RELEASE_DIR, f"synth_{lk}.parquet"))
    assert assemble.frame_sha256(released) == man["release_sha256"]


@pytest.mark.slow
def test_main_output_identical_with_curtailment_disabled(tmp_path):
    """generate_wind.main(): a config with 'curtailment: {enabled: false}' writes
    byte-identical output to the same config without the block."""
    scratch = str(tmp_path)
    base = yaml.safe_load(open(os.path.join(REPO, "configs", "config_wind.yaml")))
    r2 = yaml.safe_load(open(os.path.join(REPO, "configs", "round2", "M1.yaml")))["round2"]
    shas = []
    for k, extra in enumerate(({}, {"curtailment": {"enabled": False, "grid": {"target_scale": 4.0}}})):
        cfg = copy.deepcopy(base)
        cfg["data"]["synth_dir"] = os.path.join(scratch, str(k))
        cfg["round2"] = {**r2, "experiment_id": "curt_off"}
        cfg.update(extra)
        name = f"config_07374_curt{k}.yaml"
        path = os.path.join(REPO, "configs", name)
        try:
            with open(path, "w") as f:
                yaml.safe_dump(cfg, f)
            code = f"import sys; sys.argv=['x']; import generate_wind as m; m.main('{name}')"
            res = subprocess.run([os.path.join(REPO, "synthre", "bin", "python"), "-c", code], cwd=REPO,
                                 capture_output=True, text=True, timeout=3600)
            assert res.returncode == 0, res.stderr[-3000:]
        finally:
            os.remove(path)
        shas.append(_sha(os.path.join(scratch, str(k), "wind", "round2", "curt_off", f"synth_07374_curt{k}.csv")))
    assert shas[0] == shas[1]
