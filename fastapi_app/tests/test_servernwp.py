"""Server NWP reader: file-name parsing, nearest points, latest-run choice (Parquet reads are faked)."""
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fastapi_app import nwp, servernwp  # noqa: E402


def _ecmwf(speed: float) -> pd.DataFrame:
    rows = []
    for start in pd.to_datetime(["2024-01-01 00:00", "2024-01-01 12:00", "2024-01-02 00:00"], utc=True):
        for lead in range(0, 58):
            rows.append({"starttime": start, "forecasttime": lead, "dew_point2m": 270.0, "temp2m": 275.0,
                         "u_wind10m": speed, "v_wind10m": 0.0, "u_wind100m": speed * 1.5, "v_wind100m": 0.0})
    return pd.DataFrame(rows)


def _ecmwf_solar() -> pd.DataFrame:
    rows = []
    for start in pd.to_datetime(["2024-01-01 00:00", "2024-01-01 12:00", "2024-01-02 00:00"], utc=True):
        for lead in range(0, 58):                                              # lead 0 is empty in the real files
            rows.append({"starttime": start, "forecasttime": lead,
                         "ssrd": np.nan if lead == 0 else 400.0, "fdir": np.nan if lead == 0 else 150.0})
    return pd.DataFrame(rows)


def _icon(speed: float, run_hour: int) -> pd.DataFrame:
    tops = [20.0, 55.0, 100.0, 150.0]
    rows = []
    for day in ("2024-01-01", "2024-01-02"):
        start = pd.Timestamp(f"{day} {run_hour:02d}:00", tz="UTC")
        for lead in range(0, 49):
            for i, top in enumerate(tops):
                rows.append({"starttime": start, "forecasttime": float(lead), "toplevel": top,
                             "bottomlevel": 0.0 if i == 0 else tops[i - 1], "u_wind": speed, "v_wind": 0.0,
                             "temperature": 280.0, "pressure": 99000.0, "qs": 0.005})
    return pd.DataFrame(rows)


def _icon_sl(run_hour: int) -> pd.DataFrame:
    rows = []
    for day in ("2024-01-01", "2024-01-02"):
        start = pd.Timestamp(f"{day} {run_hour:02d}:00", tz="UTC")
        for lead in range(0, 49):
            rows.append({"starttime": start, "forecasttime": float(lead), "aswdir_s_avg": 100.0, "aswdifd_s_avg": 50.0})
            if lead == 3:                                                     # extra sub-hourly row, must be ignored
                rows.append({"starttime": start, "forecasttime": 3.25, "aswdir_s_avg": 999.0, "aswdifd_s_avg": 999.0})
    return pd.DataFrame(rows)


def test_parse_and_weights():
    assert servernwp._parse("47_2600_10_1100_wind_sl.parquet") == (47.26, 10.11)
    assert servernwp._parse("47_3796_10_2951_ML.parquet") == (47.3796, 10.2951)
    assert servernwp._parse("junk.parquet") is None
    assert servernwp.weights_of([0.2, 30.0, 40.0]) == [1.0, 0.0, 0.0]
    w = servernwp.weights_of([10.0, 20.0])
    assert abs(sum(w) - 1) < 1e-9 and w[0] > w[1]


def _reset():
    servernwp.index.cache_clear()
    servernwp._load.cache_clear()


def test_server_source_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        sl = Path(tmp) / "ecmwf/parquet/SL"
        sl.mkdir(parents=True)
        speeds = {"47_2600_10_1100": 4.0, "47_5100_10_1100": 8.0, "47_2600_10_3600": 8.0}
        solar = Path(tmp) / "ecmwf/parquet/solar"
        solar.mkdir(parents=True)
        for k in speeds:
            (sl / f"{k}_wind_sl.parquet").write_bytes(b"")
            (solar / f"{k}_solar_sl.parquet").write_bytes(b"")
        for run in ("06", "12"):                                              # ICON-D2: station folders per run hour
            d = Path(tmp) / f"icon-d2/parquet/ML/{run}/00096"
            d.mkdir(parents=True)
            (d / "47_3796_10_2951_ML.parquet").write_bytes(b"")
            sld = Path(tmp) / f"icon-d2/parquet/SL/{run}"
            sld.mkdir(parents=True)
            (sld / "47_3796_10_2951_SL.parquet").write_bytes(b"")
        real_read = pd.read_parquet

        def fake(p, *a, **k):
            p = Path(p)
            if "icon-d2" in p.parts:
                if "SL" in p.parts:
                    return _icon_sl(int(p.parts[-2]))
                return _icon(6.0, int(p.parts[-3]))
            if "solar" in p.parts:
                return _ecmwf_solar()
            return _ecmwf(speeds[p.name.replace("_wind_sl.parquet", "")])

        os.environ["SYNTH_RE_NWP_ROOT"] = tmp
        _reset()
        pd.read_parquet = fake
        try:
            src = {s["id"]: s for s in nwp.sources()}
            assert set(src) == {"srv_icon_d2", "srv_ecmwf"}                    # nothing else on the server
            assert src["srv_ecmwf"]["available"] and src["srv_ecmwf"]["points"] == 3
            assert src["srv_icon_d2"]["points"] == 1 and src["srv_icon_d2"]["lead"]["max"] == 48
            assert nwp.has_lead("srv_ecmwf") and not nwp.is_package("srv_ecmwf")

            wx, note = nwp.weather_for("srv_ecmwf", {"params": {"lat": 47.2601, "lon": 10.1101}}, (12, 36))
            assert note["grid_points"] == 1 and note["distance_km"] < 1
            assert np.allclose(wx["wind_speed_10m"], 4.0) and np.allclose(wx["wind_speed_100m"], 6.0)
            assert wx.index.min() == pd.Timestamp("2024-01-01 12:00", tz="UTC")
            assert np.allclose(wx["shortwave_radiation"], 400.0) and np.allclose(wx["direct_radiation"], 150.0)
            assert np.allclose(wx["diffuse_radiation"], 250.0)                  # global - direct
            wz, _ = nwp.weather_for("srv_ecmwf", {"params": {"lat": 47.2601, "lon": 10.1101}}, (0, 6))
            assert wz["shortwave_radiation"].notna().all()                       # lead 0 (empty) falls back to the next run
            wx2, note2 = nwp.weather_for("srv_ecmwf", {"params": {"lat": 47.38, "lon": 10.2}}, (12, 36))
            assert note2["grid_points"] == 3 and 4.0 < wx2["wind_speed_10m"].iloc[0] < 8.0

            wi, ni = nwp.weather_for("srv_icon_d2", {"params": {"lat": 47.38, "lon": 10.30}}, (12, 36))
            assert ni["grid_points"] == 1 and np.allclose(wi["wind_speed_100m"], 6.0)
            assert {"temp_2m", "pressure", "dew_point_2m"} <= set(wi.columns)
            assert np.allclose(wi["direct_radiation"], 100.0) and np.allclose(wi["diffuse_radiation"], 50.0)
            assert np.allclose(wi["shortwave_radiation"], 150.0)               # direct + diffuse, sub-hourly row ignored
            assert "shortwave_radiation" in src["srv_icon_d2"]["features"]
            assert "shortwave_radiation" in src["srv_ecmwf"]["features"]
            for bad in ({"lat": 40.0, "lon": 0.0},):
                try:
                    nwp.weather_for("srv_ecmwf", {"params": bad}, (12, 36))
                    raise AssertionError("expected a distance error")
                except ValueError as e:
                    assert "km" in str(e)
        finally:
            pd.read_parquet = real_read
            os.environ.pop("SYNTH_RE_NWP_ROOT")
            _reset()
        assert "srv_ecmwf" not in {s["id"] for s in nwp.sources()}            # off again without the setting


if __name__ == "__main__":
    test_parse_and_weights(); test_server_source_roundtrip(); print("ok")
