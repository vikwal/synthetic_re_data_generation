"""Tests for the parks_pv_v1 geometry, register view and chain parameters (no data access)."""
import numpy as np
import pandas as pd
import pytest

from parks_pv import geometry as G, params as P


def _row(**kw):
    base = {"park_id": "SELTEST", "orientation_class": "Süd", "tilt_class": "20 - 40 Grad", "secondary_orientation": np.nan,
            "secondary_tilt_class": np.nan, "uniform_flag": "1", "cohort": "2013-20", "cls": "C1_south_std", "mixed_orientation": False,
            "ac_kw": 2700.0, "dc_kwp": 3000.0, "year": 2016, "n_units": 1, "area_ha": 4.0, "power_limit": "Nein", "lat": 50.0, "lon": 10.0}
    base.update(kw)
    return pd.Series(base)


def test_nominal_arrays_shares_and_ranges():
    r = _row()
    rng = np.random.default_rng(1)
    for _ in range(50):
        arrs, dev = G.nominal_arrays(r, rng)
        assert abs(sum(a["share"] for a in arrs) - 1.0) < 1e-9
        for a in arrs:
            assert 0 <= a["azimuth"] < 360 and 5 <= a["tilt"] <= 45
        if dev is None:
            assert len(arrs) == 1 and 160 <= arrs[0]["azimuth"] <= 200 and 20 <= arrs[0]["tilt"] <= 28


def test_deviation_share_about_15_percent():
    n = sum(G.nominal_arrays(_row(park_id=f"P{i}"), G.park_rng(f"P{i}"))[1] is not None for i in range(1500))
    assert 0.11 < n / 1500 < 0.19


def test_east_west_and_secondary():
    arrs, _ = G.nominal_arrays(_row(orientation_class="Ost-West", tilt_class="< 20 Grad"), np.random.default_rng(3))
    assert len(arrs) == 2 and {round(a["azimuth"] / 90) for a in arrs} == {1, 3} and all(10 <= a["tilt"] <= 15 for a in arrs)
    arrs, _ = G.nominal_arrays(_row(secondary_orientation="West", park_id="X"), np.random.default_rng(5))
    assert len(arrs) >= 2 and any(a["source"] == "secondary" for a in arrs)
    sec = sum(a["share"] for a in arrs if a["source"] == "secondary")
    assert 0.15 <= sec <= 0.45


def test_register_classes_and_view():
    assert G.orientation_class(180.0) == "Süd" and G.orientation_class(359.0) == "Nord" and G.orientation_class(225.0) == "Süd-West"
    assert G.tilt_class(12.0) == "< 20 Grad" and G.tilt_class(25.0) == "20 - 40 Grad"
    v = G.register_view(_row(), [{"tilt": 23.0, "azimuth": 180.0, "share": 1.0}])
    assert v["tilt_deg_rounded5"] == 25 and v["orientation_class"] == "Süd" and v["secondary_orientation"] == ""


def test_park_parameters_and_config():
    par = P.park_parameters(_row())
    assert par["dc_ac"] == pytest.approx(3000 / 2700) and par["era"] == "multi_perc" and par["kwp_per_ha_src"] == "mastr_area"
    assert 0.2 <= par["gcr"] <= 0.75
    cfg = P.park_config("SELTEST", 50.0, 10.0, 100.0, par, [{"name": "a", "tilt": 25, "azimuth": 180, "share": 0.6},
                                                           {"name": "b", "tilt": 25, "azimuth": 200, "share": 0.4}])
    assert cfg["losses"] == P.LOSSES and cfg["shading"]["inter_row"] and cfg["arrays"][0]["gcr"] == par["gcr"]
    assert cfg["module"]["gamma_pdc"] == P.GAMMA["multi_perc"] and cfg["aging"]["commissioning_date"] == "2016-07-01"


def test_release_frame_and_clients_table():
    from parks_pv import assemble as A, metadata as M
    idx = pd.date_range("2024-01-01", periods=48, freq="h", tz="UTC")
    drv = pd.DataFrame({c: np.linspace(0, 1, 48) for c in A.DRIVER_COLS}, index=idx)
    res = pd.DataFrame({c: np.ones(48) for c in A.CHAIN_COLS + ["power_park_free"]}, index=idx)
    res["power_park"] = 0.5
    rel = A.release_frame(drv, res)
    assert list(rel.columns) == A.DRIVER_COLS + A.CHAIN_COLS + A.POWER_COLS and rel.index.name == "timestamp"
    assert (rel["power_park"] == 0.5).all() and (rel["availability"] == 1.0).all() and rel.dtypes.eq("float64").all()
    parks = pd.DataFrame({"client_id": ["R0", "R0", "N0", "holdout"], "client_kind": ["regional", "regional", "nationwide", "nationwide"],
                          "region": [1, 1, 2, 3], "capacity_kw": [1000.0, 2000.0, 500.0, 700.0], "capacity_kwp_dc": [1100.0, 2200.0, 550.0, 770.0],
                          "n_arrays": [2, 3, 1, 4]})
    c = M.clients_table(parks).set_index("client_id")
    assert c.loc["R0", "n_parks"] == 2 and c.loc["R0", "capacity_kw"] == 3000.0 and c.loc["R0", "region_id"] == 1
    assert c.loc["holdout", "archetype"] == "holdout" and c.loc["N0", "region_id"] == ""
