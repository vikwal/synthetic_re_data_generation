"""Park spec (parks_pv.park_spec): validation, derived selection row, cohorts and dates without MaStR units."""
import json

import pandas as pd
import pytest

from parks_pv import params as P, park_spec as PS

BASE = dict(park_id="demo_01", lat=52.3, lon=13.6, ac_kw=10000, dc_kwp=12500, commissioning="2021-06-01")


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    monkeypatch.setattr(PS, "terrain_metrics", lambda lat, lon: dict(elevation=50.0, slope=0.5, aspect=180.0, tpi5=1.0, tdi=0.1, elev_std=10.0, tpi75=0.0))
    monkeypatch.setattr(PS, "admin_area", lambda lat, lon: dict(bundesland="Brandenburg", landkreis="X", gemeinde="Y"))


def test_validate():
    assert PS.validate(BASE) == []
    e = PS.validate(dict(BASE, park_id="a b", lat=91, ac_kw=0, foo=1, geometry={"mode": "manual"}))
    assert any("park_id" in x for x in e) and any("lat" in x for x in e) and any("ac_kw" in x for x in e)
    assert any("foo" in x for x in e) and any("arrays" in x for x in e)
    assert any("commissioning" in x for x in PS.validate({k: v for k, v in BASE.items() if k != "commissioning"}))
    assert PS.validate(dict(BASE, commissioning_full="2020-01-01")) == ["commissioning_full before commissioning"]


def test_row_defaults_and_classes():
    r = PS.to_row(BASE)
    assert r["orientation_class"] == "Süd" and r["tilt_class"] == "20 - 40 Grad" and r["cls"] == "C1_south_std"
    assert r["dc_ac"] == 1.25 and r["year"] == r["year_last"] == 2021 and r["cohort"] == "2021+" and r["size_bin"] == "5-20MW"
    assert r["bundesland"] == "Brandenburg" and r["kind"] == "spec" and r["unit_ids"] == ""
    ew = PS.to_row(dict(BASE, geometry={"mode": "manual", "arrays": [{"tilt": 12, "azimuth": 90, "share": 0.5}, {"tilt": 12, "azimuth": 270, "share": 0.5}]}))
    assert ew["orientation_class"] == "Ost-West" and ew["tilt_class"] == "< 20 Grad" and ew["geometry_mode"] == "manual"
    assert len(json.loads(ew["geometry_arrays"])) == 2


def test_cohorts_and_dates_without_units():
    s = dict(BASE, cohorts=[{"commissioning": "2023-04-01", "dc_kwp": 2500}, {"commissioning": "2021-06-01", "dc_kwp": 10000}])
    r = pd.Series(PS.to_row(s))
    assert r.commissioning_full == "2023-04-01" and r.year_last == 2023
    d = P.commissioning_dates(pd.DataFrame([r]))
    assert d.loc["demo_01", "commissioning_first"] == pd.Timestamp("2021-06-01")
    c = P.age_cohorts(r)
    assert [x["year"] for x in c] == [2021, 2023] and sum(x["share"] for x in c) == pytest.approx(1.0)


def test_load_list_and_duplicates(tmp_path):
    p = tmp_path / "p.yaml"
    PS.dump([BASE, dict(BASE, park_id="demo_02")], str(p))
    assert [s["park_id"] for s in PS.load(str(p))] == ["demo_01", "demo_02"]
    PS.dump([BASE, BASE], str(p))
    with pytest.raises(ValueError, match="twice"):
        PS.load(str(p))
