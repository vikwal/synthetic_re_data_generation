"""Sample-data package: park import and forecast-run weather (pure logic; Parquet reads are faked)."""
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from fastapi_app import nwp, parkdata

PARK = "SELTEST0001"
Z_TOP = [20.0, 55.2, 100.3, 156.0]
Z_BOT = [0.0, 20.0, 55.2, 100.3]


def _icon_rows(offset: float = 0.0, days: int = 4) -> pd.DataFrame:
    rows = []
    for d in range(days):
        start = pd.Timestamp("2024-01-01 12:00", tz="UTC") + pd.Timedelta(days=d)
        for lead in range(0, 49):
            for top, bot in zip(Z_TOP, Z_BOT):
                z = (top + bot) / 2
                rows.append({"starttime": start, "forecasttime": float(lead), "toplevel": top, "bottomlevel": bot,
                             "u_wind": 0.05 * z + offset, "v_wind": 0.0, "temperature": 280.0 + lead, "pressure": 95000.0,
                             "qs": 0.005})
    return pd.DataFrame(rows)


def test_icon_frame_interpolates_winds_and_applies_lead_window():
    f = parkdata.icon_frame(_icon_rows(), 12, 36)
    assert f["lead"].min() == 12 and f["lead"].max() == 36
    assert np.allclose(f["u_wind_10m"], 0.5, atol=1e-6)               # lowest layer mid height is 10 m
    assert np.allclose(f["u_wind_100m"], 5.0, atol=1e-6)              # linear profile -> exact at 100 m
    assert (f["valid"] == f["starttime"] + pd.to_timedelta(f["lead"], unit="h")).all()
    assert 270 < f["dew_point_2m"].mean() < 290


def test_pick_latest_prefers_the_shortest_lead():
    f = parkdata.icon_frame(_icon_rows(), 12, 36)
    s = parkdata.pick_latest(f)
    assert s.index.is_unique
    # 2024-01-02 00:00 is lead 12 of the 01-01 run (the only run in the window for that hour)
    t = pd.Timestamp("2024-01-02 00:00", tz="UTC")
    assert t in s.index
    # an hour covered by two runs: the later run (smaller lead) wins -> value from lead < 24+
    both = pd.Timestamp("2024-01-03 00:00", tz="UTC")      # lead 36 of one run and lead 12 of the next
    run_a = f[(f["valid"] == both)]
    assert sorted(run_a["lead"]) == [12.0, 36.0] and s.loc[both, "temp_2m"] == 280.0 + 12.0


def test_combine_points_weights_and_complete_hours_only():
    a = parkdata.pick_latest(parkdata.icon_frame(_icon_rows(0.0), 12, 36))
    b = parkdata.pick_latest(parkdata.icon_frame(_icon_rows(4.0), 12, 36))
    b = b.iloc[5:]                                                    # point b is missing the first hours
    out = parkdata.combine_points({"a": a, "b": b}, {"a": 0.25, "b": 0.75})
    assert out.index.min() == b.index.min()                           # only hours where all points have data
    assert np.allclose(out["u_wind_100m"], 5.0 + 0.75 * 4.0)
    assert np.allclose(out["wind_speed_100m"], out["u_wind_100m"].abs())


def test_ecmwf_frame_renames_columns():
    n = 60
    df = pd.DataFrame({
        "starttime": pd.Timestamp("2024-01-01 12:00", tz="UTC"), "forecasttime": np.arange(n),
        "dew_point2m": 280.0, "temp2m": 285.0, "specific_rho": 0.01, "u_wind10m": 1.0, "v_wind10m": 2.0,
        "u_wind100m": 3.0, "v_wind100m": 4.0, "u_wind200m": 5.0, "v_wind200m": 6.0, "friction_velocity": 0.2,
        "grid_lat": 50.5, "grid_lon": 7.8})
    f = parkdata.ecmwf_frame(df, 0, 57)
    assert len(f) == 58 and {"temp_2m", "dew_point_2m", "u_wind_100m"} <= set(f.columns)
    assert "pressure" not in f.columns                                # ECMWF SL has no surface pressure
    s = parkdata.combine_points({"p": parkdata.pick_latest(f)}, {"p": 1.0})
    assert np.allclose(s["wind_speed_100m"], 5.0)


def _make_package(root: Path) -> None:
    w = root / "wind"
    (w / PARK).mkdir(parents=True)
    pd.DataFrame([{"park_id": PARK, "technology": "wind", "name": "Testpark", "capacity_kw": 1000.0, "n_turbines": 2,
                   "wake_efficiency": 0.97}]).to_csv(w / "parks.csv", index=False)
    pd.DataFrame([{"park_id": PARK, "lib_name": "Enercon E-82 E2 2.300", "rated_cap_kw": 500.0, "hub_height": 100.0,
                   "commissioning_date": "2015-01-01", "latitude": 51.0 + i * 0.01, "longitude": 10.0, "group_id": "t1"}
                  for i in range(2)]).to_csv(w / "park_layouts.csv", index=False)
    (w / PARK / "technik.json").write_text(json.dumps({"groups": [{"group_id": "t1", "n": 2, "rated_cap_kw": 500.0}]}))
    (w / PARK / "synth.parquet").write_bytes(b"x")
    run = w / PARK / "nwp" / "icon-d2" / "12"
    run.mkdir(parents=True)
    pd.DataFrame([{"turbine_id": f"T{i}", "point_key": k, "weight": wt}
                  for i in range(2) for k, wt in (("p1", 0.5), ("p2", 0.5))]).to_csv(run / "turbine_points.csv", index=False)
    for k in ("p1", "p2"):
        (run / f"{k}_ML.parquet").write_bytes(b"x")


def test_import_and_weather_roundtrip_with_fake_parquet():
    idx = pd.date_range("2024-01-01", periods=24 * 5, freq="h", tz="UTC", name="timestamp")
    synth = pd.DataFrame({"power_park": np.linspace(0, 8e5, len(idx)), "power_park_free": np.linspace(0, 9e5, len(idx)),
                          "power_t1": np.linspace(0, 9e5, len(idx)), "aging_factor_t1": 0.9, "wind_speed_hub_t1": 6.0,
                          "temp_2m": 280.0, "density": 1.2, "u_wind_100m": 3.0, "v_wind_100m": 4.0}, index=idx)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "beispieldaten_test"
        _make_package(root)
        real_package_dir, real_read = parkdata.package_dir, pd.read_parquet

        def fake_read(path, *a, **k):
            return synth.copy() if str(path).endswith("synth.parquet") else _icon_rows(4.0 if "p2" in str(path) else 0.0)

        parkdata.package_dir = lambda: root
        pd.read_parquet = fake_read
        parkdata.load_weather.cache_clear()
        try:
            assert [p["park_id"] for p in parkdata.wind_parks()] == [PARK]
            meta, df = parkdata.build_result(PARK)
            assert meta["package_park_id"] == PARK and meta["summary"]["capacity_w"] == 1e6
            assert "power" in df.columns and df["power"].max() == 8e5 and meta["kind"] == "wind"
            assert abs(meta["summary"]["park_wake_loss_pct"] - 3.0) < 1e-9
            lat, lon = nwp.site_of(meta)                               # the NWP helpers can locate the park
            assert abs(lat - 51.005) < 1e-9 and lon == 10.0
            wx, note = nwp.weather_for("pkg_icon_d2", meta, (12, 36))
            assert note["grid_points"] == 2 and note["lead_from"] == 12
            assert np.allclose(wx["u_wind_100m"], 5.0 + 2.0)           # equal weights of offsets 0 and 4
            try:
                nwp.weather_for("pkg_icon_d2", {"params": meta["params"]}, (12, 36))
                raise AssertionError("expected a ValueError for a non-package run")
            except ValueError as exc:
                assert "imported wind parks" in str(exc)
            src = {s["id"]: s for s in nwp.sources()}
            assert src["pkg_icon_d2"]["available"] and src["pkg_icon_d2"]["lead"]["max"] == 48
        finally:
            parkdata.package_dir, pd.read_parquet = real_package_dir, real_read
            parkdata.load_weather.cache_clear()


def test_imported_parks_become_scenarios():
    from fastapi_app import settings, storage

    real_dir = settings.STORAGE_DIR
    with tempfile.TemporaryDirectory() as tmp:
        settings.STORAGE_DIR = Path(tmp)
        try:
            _, rs = storage._dirs()
            rid = parkdata.result_id(PARK)
            meta = {"id": rid, "scenario_id": None, "scenario_name": "Testpark (package)", "kind": "wind",
                    "params": {"turbines": [{"lat": 51.0, "lon": 10.0, "turbine": "X", "hub_height": 100.0}]},
                    "package_park_id": PARK, "created": "2026-10-01T00:00:00+00:00"}
            storage._write_json(rs / f"{rid}.json", meta)
            other = "a" * 12                                              # an ordinary run is left alone
            storage._write_json(rs / f"{other}.json", {"id": other, "scenario_id": "b" * 12, "kind": "wind"})
            parkdata._SEEN.clear()
            assert parkdata.ensure_scenarios() == 1
            sc = storage.get_scenario(rid)
            assert sc["name"] == "Testpark" and sc["last_result_id"] == rid and sc["kind"] == "wind"   # old " (package)" removed
            assert storage._read_json(rs / f"{rid}.json")["scenario_id"] == rid
            assert storage._read_json(rs / f"{rid}.json")["scenario_name"] == "Testpark"
            assert parkdata.ensure_scenarios() == 0 and len(storage.list_scenarios()) == 1       # repeating adds nothing
            storage.update_scenario(rid, {"name": "Testpark (package)"})                         # a scenario already named the old way
            storage.update_result_meta(rid, {"scenario_name": "Testpark (package)"})
            parkdata._SEEN.clear()
            parkdata.ensure_scenarios()
            assert storage.get_scenario(rid)["name"] == "Testpark" and storage._read_json(rs / f"{rid}.json")["scenario_name"] == "Testpark"
            storage.delete_scenario(rid)                                  # deleting the scenario removes the park for good
            assert parkdata.ensure_scenarios(force=True) == 0 and not storage.list_scenarios()
        finally:
            settings.STORAGE_DIR = real_dir
            parkdata._SEEN.clear()


def test_result_id_is_stable_hex():
    a = parkdata.result_id("SEL1")
    assert a == parkdata.result_id("SEL1") and len(a) == 12 and int(a, 16) >= 0
