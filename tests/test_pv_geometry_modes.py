"""Run config geometry modes, the angle estimation store and the manual arrays of stage geometry."""
import pandas as pd
import pytest

import settings
from parks_pv import geometry_store as GSt, run_config, stages


def _cfg(tmp_path, body: str) -> str:
    p = tmp_path / "run.yaml"
    p.write_text("release: t\n" + body)
    return str(p)


def test_run_config_geometry_defaults_and_checks(tmp_path):
    rc = run_config.load(_cfg(tmp_path, ""))
    assert rc["geometry"] == {"mode": "estimate", "method": "v4", "manual": None}
    assert "geometry" in rc["stages"] and rc["stages"].index("geometry") == rc["stages"].index("baseline") + 1
    with pytest.raises(ValueError):
        run_config.load(_cfg(tmp_path, "geometry: {mode: guess}\n"))
    with pytest.raises(ValueError):
        run_config.load(_cfg(tmp_path, "geometry: {mode: manual}\n"))
    with pytest.raises(KeyError):
        run_config.load(_cfg(tmp_path, "geometry: {arrays: x.csv}\n"))


def test_store_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setitem(settings._S, "geometry_dir", str(tmp_path))   # settings are resolved once at import
    assert GSt.missing("m", ["A", "B"]) == ["A", "B"]
    src = pd.DataFrame([dict(pv="A", source="laz", status="accepted", best=True)])
    GSt.save("m", "A", src, [dict(park_id="A", array_id="g0_0", tilt=20.0, azimuth=180.0, share=1.0, source="windows:laz")])
    GSt.save("m", "B", src.assign(pv="B", best=False, status="rejected"), [])
    assert GSt.missing("m", ["A", "B", "C"]) == ["C"]
    assert len(GSt.arrays("m", "A")) == 1 and len(GSt.arrays("m", "B")) == 0
    assert set(GSt.all_sources("m").pv) == {"A", "B"}


def test_manual_rows(tmp_path):
    p = tmp_path / "manual.csv"
    pd.DataFrame(dict(park_id=["A", "A", "B"], tilt=[15, 15, 25], azimuth=[90, 270, 180], share=[0.5, 0.5, 1])).to_csv(p, index=False)
    rows = stages._manual_rows(str(p), pd.DataFrame(dict(park_id=["A", "B"])))
    assert list(rows["A"].array_id) == ["m0", "m1"] and (rows["B"].geometry_source == "manual").all()
    with pytest.raises(SystemExit):
        stages._manual_rows(str(p), pd.DataFrame(dict(park_id=["A", "C"])))
