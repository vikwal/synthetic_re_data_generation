"""parks_v1: config generation, group aging, wake mapping, layout grouping,
ERA5 cell selection, and (slow) equivalence with the round-2 chain."""

import os

import numpy as np
import pandas as pd
import pytest
import yaml

from parks import config, era5_db, layout, library, synth, wakes
from round2 import aging
from round2 import wake as r2_wake

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------- library

def _curve(name="T"):
    ws = np.arange(1.0, 26.0, 1.0)
    p = np.clip((ws - 3.0) ** 3 * 8.0, 0.0, 2000.0)
    p[ws < 4.0] = 0.0
    p[ws > 22.0] = 0.0
    return pd.Series(p, index=ws, name=name)


def test_derive_specs_from_curve():
    d = library.derive_specs_from_curve(_curve())
    assert d["cut_in"] == 4.0            # first P > 0
    assert d["rated_ws"] == 10.0         # first P >= 0.99 Pmax (7^3*8 = 2744 -> clipped)
    assert d["cut_out"] == 22.0          # last P > 0


def test_spec_overrides_only_fill_missing_fields():
    pc = pd.DataFrame({"A": _curve("A"), "B": _curve("B")})
    specs = pd.DataFrame({"Rotordurchmesser": [80.0, 90.0],
                          "Einschaltgeschwindigkeit": ["3.0", "-"],
                          "Abschaltgeschwindigkeit": ["25.0", "-"],
                          "Nennwindgeschwindigkeit": ["-", "12.0"]}, index=["A", "B"])
    ov = library.build_spec_overrides(["A", "B"], pc, specs)
    assert set(zip(ov.turbine, ov.field)) == {("A", "rated_ws"), ("B", "cut_in"), ("B", "cut_out")}
    s = library.turbine_specs(["A", "B"], ov, specs)
    assert s["A"]["cut_in"] == 3.0 and s["A"]["rated"] == 10.0     # existing kept, missing derived
    assert s["B"]["rated"] == 12.0 and s["B"]["cut_out"] == 22.0
    with pytest.raises(ValueError):
        library.turbine_specs(["B"], None, specs)                    # '-' without override


# ---------------------------------------------------------------- group aging

def test_degradation_by_group_matches_single_park_logic():
    idx = pd.date_range("2023-07-24", "2026-08-31 23:00", freq="h", tz="UTC")
    dates = {"t1": "2002-03-14", "t2": "2017-05-23"}
    dv = aging.degradation_by_group(idx, dates, model="weibull")
    for g, d in dates.items():
        ref, _ = aging.get_degradation_vector(idx, model="weibull", commissioning_date=d)
        np.testing.assert_array_equal(dv[g], ref)
    # the older group is more degraded, both decline over time
    assert dv["t1"].mean() < dv["t2"].mean()
    assert dv["t1"][-1] < dv["t1"][0] and dv["t2"][-1] < dv["t2"][0]
    assert aging.DF_weibull(21.36) == pytest.approx(dv["t1"][0], abs=1e-3)


def test_degradation_by_group_clips_before_commissioning():
    idx = pd.date_range("2023-07-24", periods=24 * 400, freq="h", tz="UTC")
    dv = aging.degradation_by_group(idx, {"new": "2024-01-01"}, model="weibull")["new"]
    before = idx < pd.Timestamp("2024-01-01", tz="UTC")
    assert np.all(dv[before] == 1.0) and dv[-1] < 1.0


def test_degradation_by_group_rejects_unknown_model():
    with pytest.raises(ValueError):
        aging.degradation_by_group(pd.date_range("2024", periods=3, freq="h"), {"t1": "2010-01-01"}, model="x")


# ---------------------------------------------------------------- wake mapping

@pytest.fixture(scope="module")
def curves():
    return library.load_power_curves(), library.load_ct_curves()


def test_resolve_columns_automatic_by_lib_name(curves):
    pc, ct = curves
    assert r2_wake.resolve_columns("Enercon E-101", pc, ct) == ("Enercon E-101", "Enercon E-101")
    # library type without a Ct curve -> own power curve, generic Ct (None)
    assert r2_wake.resolve_columns("Enercon E-66/18.70", pc, ct) == ("Enercon E-66/18.70", None)


def test_resolve_columns_legacy_labels_unchanged(curves):
    pc, ct = curves
    for label, cols in r2_wake.MODEL_MAP.items():
        assert r2_wake.resolve_columns(label, pc, ct) == cols
    with pytest.raises(KeyError):
        r2_wake.resolve_columns("Not A Turbine 9000", pc, ct)


def test_build_windturbine_flags_generic_ct(curves):
    pc, ct = curves
    _, generic = r2_wake.build_windturbine("Enercon E-66/18.70", 98.0, 70.0, np.nan, pc, ct, name="x")
    _, own = r2_wake.build_windturbine("Enercon E-101", 135.0, 101.0, np.nan, pc, ct, name="y")
    assert generic is True and own is False


# ---------------------------------------------------------------- layout / groups

def _layout():
    return pd.DataFrame({
        "park_id": ["P"] * 5, "lib_name": ["A", "A", "A", "B", "A"],
        "hub_height": [100.0, 100.0, 100.0, 120.0, 100.0], "commissioning_year": [2010, 2010, 2010, 2015, 2010],
        "rated_kw": [2000.0, 2000.0, 1800.0, 3000.0, 2000.0], "rated_cap_kw": [2000.0, 2000.0, 1800.0, 3000.0, 2000.0],
        "era5_cell_id": [7, 7, 7, 7, 8], "era5_cell_dist_km": [5.0] * 5, "rotor_diameter": [80.0] * 5,
        "commissioning_date": ["2010-01-01", "2010-12-31", "2010-06-01", "2015-01-01", "2010-03-01"],
        "x_utm32": [0.0, 400.0, 800.0, 1200.0, 30000.0], "y_utm32": [0.0] * 5,
    })


def test_groups_split_by_type_hub_year_rating_and_cell():
    lay = _layout()
    lay = lay.assign(**{c: "" for c in layout.LAYOUT_COLS if c not in lay})
    lay = layout.assign_groups(lay)
    g = layout.group_table(lay)
    # A/2000/cell7 (2), A/1800/cell7 (1), A/2000/cell8 (1), B (1)
    assert sorted(g.n_turbines) == [1, 1, 1, 2]
    assert g.n_turbines.sum() == len(lay)
    two = g[g.n_turbines == 2].iloc[0]
    assert two.commissioning_date == "2010-07-02"           # mean of the two members
    assert g.group_id.tolist() == ["t1", "t2", "t3", "t4"]
    assert g.iloc[-1].lib_name == "B"                       # ordered by commissioning year first
    assert layout.primary_cell(g) == 7


def test_min_spacing_in_rotor_diameters():
    lay = _layout()
    assert layout.min_spacing_d(lay) == pytest.approx(5.0)
    assert layout.min_spacing_d(lay.iloc[:1]) == np.inf


def test_rating_cap_never_exceeds_curve():
    sel = pd.DataFrame({"lokation": ["P"], "name": ["x"], "client": ["R0"], "cls": ["B1_single"]})
    tur = pd.DataFrame({"lokation": ["P"], "EinheitMastrNummer": ["U1"], "operator": ["o"],
                        "Hersteller": ["h"], "lib_name": ["A"], "Nettonennleistung": [3600.0],
                        "Nabenhoehe": [150.0], "Rotordurchmesser": [136.0],
                        "Inbetriebnahmedatum": ["2021-12-21"]})
    osm = pd.DataFrame({"EinheitMastrNummer": ["U1"], "lat": [53.0], "lon": [10.0],
                        "coord_source": ["mastr"], "status": ["ok"], "d_osm_m": [3.0]})
    lay = layout.build_turbine_table(sel, tur, osm, {"A": 3450.0})
    assert lay.rated_kw.iloc[0] == 3600.0 and lay.rated_cap_kw.iloc[0] == 3450.0


# ---------------------------------------------------------------- configs

def _park_inputs():
    lay = _layout()
    lay = lay.assign(latitude=53.0, longitude=10.0, osm_status="ok", operator="o")
    lay = layout.assign_groups(lay.assign(**{c: "" for c in layout.LAYOUT_COLS if c not in lay}))
    lay["latitude"], lay["longitude"], lay["rotor_diameter"] = 53.0, 10.0, 80.0
    lay["rated_kw"] = [2000.0, 2000.0, 1800.0, 3000.0, 2000.0]
    lay["rated_cap_kw"] = lay["rated_kw"]
    groups = layout.group_table(lay)
    park = pd.Series({"lokation": "P", "name": "Park", "client": "R0", "cls": "B4_two_types", "replaced": ""})
    base = config.load_yaml(os.path.join(REPO, "configs", "config_wind.yaml"))
    chain = config.load_yaml(os.path.join(REPO, "configs", "round2", "PARKS_v1.yaml"))
    return park, groups, lay, base, chain


def test_build_config_round2_format():
    park, groups, lay, base, chain = _park_inputs()
    cfg = config.build_config(park, groups, lay, base, chain)
    p = cfg["params"]
    assert len({len(p[k]) for k in ("turbines", "hub_heights", "rated", "group_ids", "group_sizes",
                                    "commissioning_dates", "era5_cells")}) == 1
    assert sum(p["group_sizes"]) == 5
    assert cfg["park"]["capacity_kw"] == pytest.approx(10800.0)
    assert cfg["park"]["primary_era5_cell"] == 7
    assert cfg["round2"]["correction"] == "off" and cfg["round2"]["shear"] == "power_law"
    assert cfg["round2"]["wake"] == {"enabled": True, "model": "noj", "k": 0.075}
    assert "commissioning_date" not in p and p["noise"] == 0.0
    # base sections of the round-2 format are kept, and the YAML round-trips
    assert {"data", "features", "write", "params", "round2", "park"} <= set(cfg)
    assert yaml.safe_load(yaml.safe_dump(cfg)) == cfg
    synth.check_config(cfg)


def test_check_config_rejects_inconsistent_configs():
    park, groups, lay, base, chain = _park_inputs()
    cfg = config.build_config(park, groups, lay, base, chain)
    bad = {**cfg, "params": {**cfg["params"], "group_sizes": cfg["params"]["group_sizes"][:-1]}}
    with pytest.raises(ValueError):
        synth.check_config(bad)
    bad = {**cfg, "round2": {**cfg["round2"], "correction": "height_consistent"}}
    with pytest.raises(ValueError):
        synth.check_config(bad)


# ---------------------------------------------------------------- ERA5 cells

def test_nearest_cells_is_geodesic():
    pts = pd.DataFrame({"cell_id": [1, 2], "lat": [54.0, 54.25], "lon": [10.25, 10.0]})
    # 0.2 deg east (~13 km at 54 N) is closer than 0.25 deg north (~28 km)
    ids, d = era5_db.nearest_cells(pts, 54.0, 10.05)
    assert ids[0] == 1 and d[0] == pytest.approx(13.1, abs=0.3)


def test_check_complete_detects_gaps():
    idx = pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC")
    df = pd.DataFrame({c: 1.0 for c in era5_db.RAW_COLS}, index=idx)
    era5_db.check_complete(df, "2024-01-01", "2024-01-01 04:00", 1)
    with pytest.raises(ValueError):
        era5_db.check_complete(df.drop(idx[2]), "2024-01-01", "2024-01-01 04:00", 1)
    with pytest.raises(ValueError):
        era5_db.check_complete(df.assign(temp_2m=np.nan), "2024-01-01", "2024-01-01 04:00", 1)


# ---------------------------------------------------------------- wakes

def _two_turbine_layout(dx, dy):
    return pd.DataFrame({"lib_name": ["Enercon E-101"] * 2, "hub_height": [135.0] * 2,
                         "rotor_diameter": [101.0] * 2, "x_utm32": [0.0, dx], "y_utm32": [0.0, dy]})


def test_wake_factor_direction_dependence(curves):
    pc, ct = curves
    idx = pd.date_range("2024-01-01", periods=2, freq="h", tz="UTC")
    flow = pd.DataFrame({"ws": [8.0, 8.0], "wd": [270.0, 0.0]}, index=idx)   # from west, from north
    w, info = wakes.wake_factor(_two_turbine_layout(5 * 101.0, 0.0), flow, 0.075, pc, ct)
    assert w.iloc[0] < 0.9        # aligned with the row: downstream turbine waked
    assert w.iloc[1] == pytest.approx(1.0, abs=1e-6)
    assert not info.generic_ct.any()


def test_wake_factor_single_turbine_is_one(curves):
    pc, ct = curves
    idx = pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC")
    flow = pd.DataFrame({"ws": [5.0, 9.0, 14.0], "wd": [10.0, 200.0, 270.0]}, index=idx)
    w, _ = wakes.wake_factor(_two_turbine_layout(0.0, 0.0).iloc[:1], flow, 0.075, pc, ct)
    assert (w == 1.0).all()


def test_inflow_capacity_weighted():
    idx = pd.date_range("2024-01-01", periods=1, freq="h", tz="UTC")
    free = pd.DataFrame({"wind_speed_hub_t1": [6.0], "wind_speed_hub_t2": [9.0],
                         "wind_direction_100m_t1": [350.0], "wind_direction_100m_t2": [10.0]}, index=idx)
    groups = pd.DataFrame({"group_id": ["t1", "t2"], "n": [2, 1], "rated_cap_kw": [1000.0, 2000.0]})
    f = wakes.inflow(free, groups)
    assert f.ws.iloc[0] == pytest.approx(7.5)
    assert f.wd.iloc[0] == pytest.approx(0.0, abs=1e-9) or f.wd.iloc[0] == pytest.approx(360.0)


# ---------------------------------------------------------------- equivalence (slow)

SITE_RUN = "/mnt/nvme2/synthetic/wind/round2/SITE_v2"
ERA5_V2 = "/mnt/nvme2/synthetic/raw/wind_era5_v2"


@pytest.mark.slow
@pytest.mark.parametrize("station_id", ["00164", "00183"])
def test_group_driver_reproduces_round2_chain(station_id):
    """parks.synth on a single-cell config reproduces generate_wind.main()
    (SITE_v2 run, branch-C sites: power law, Weibull, no QM, no wakes)."""
    ref_path = os.path.join(SITE_RUN, f"synth_{station_id}.csv")
    if not os.path.exists(ref_path):
        pytest.skip("SITE_v2 run outputs not on disk")
    os.chdir(REPO)
    ref = pd.read_csv(ref_path, sep=";", index_col=0, parse_dates=True)
    base = config.load_yaml(os.path.join(REPO, "configs", "config_wind.yaml"))
    chain = config.load_yaml(os.path.join(REPO, "configs", "round2", "PARKS_v1.yaml"))
    comm = pd.read_csv(os.path.join(REPO, "data", "round2", "site_commissioning.csv"), dtype={"park_id": str})
    cd = comm.set_index("park_id").loc[station_id, "commissioning_date"]
    types, hubs = base["params"]["turbines"], base["params"]["hub_heights"]
    pc = library.load_power_curves()
    cfg = dict(base)
    cfg["round2"] = {**chain["round2"], "output_end": str(ref.index[-1].tz_convert(None))}
    cfg["params"] = {**base["params"], "turbines": types, "hub_heights": hubs,
                     "rated": [float(pc[t].max()) for t in types],
                     "group_ids": [f"t{i}" for i in range(1, 7)], "group_sizes": [1] * 6,
                     "commissioning_dates": [cd] * 6, "era5_cells": [0] * 6}
    cfg["park"] = {"primary_era5_cell": 0}
    era5 = pd.read_csv(os.path.join(ERA5_V2, f"Station_{station_id}.csv"),
                       parse_dates=["timestamp"], index_col="timestamp")[era5_db.RAW_COLS]
    era5 = era5.loc[cfg["round2"]["output_start"]:]
    out = synth.run_park(cfg, {0: era5})
    idx = ref.index.intersection(out.index)
    assert len(idx) == len(ref)
    for i in range(1, 7):
        np.testing.assert_allclose(out.loc[idx, f"power_t{i}"], ref.loc[idx, f"power_t{i}"], rtol=1e-9, atol=1e-6)
        np.testing.assert_allclose(out.loc[idx, f"wind_speed_hub_t{i}"], ref.loc[idx, f"wind_speed_t{i}"], atol=1e-9)


def test_git_state_keeps_full_paths(tmp_path):
    import subprocess
    from parks import assemble
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    for name in ("a.txt", "b.txt"):
        (tmp_path / name).write_text("x")
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "-c", "user.email=t@t", "-c", "user.name=t",
                    "commit", "-qm", "init"], check=True)
    (tmp_path / "a.txt").write_text("changed")
    (tmp_path / "untracked.txt").write_text("u")
    st = assemble.git_state(str(tmp_path))
    assert st["dirty_files"] == ["a.txt"] and len(st["commit"]) == 40
