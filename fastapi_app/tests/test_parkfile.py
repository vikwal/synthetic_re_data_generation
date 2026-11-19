"""Turbine file import: park grouping, column detection and turbine type matching."""
import io

import pytest

pd = pytest.importorskip("pandas")

from fastapi_app import parkfile  # noqa: E402

KNOWN = ["Vestas V112-3.0 Offshore", "Enercon E-82 E2 2.000", "Vestas V136-3.45"]

CSV = (
    "Park ID;Park Name;Turbine Type;Latitude;Longitude;Hub height;Commissioning date\n"
    "P1;Alpha;Vestas V112-3.0 Offshore;53,40;8,64;94;31.12.2013\n"
    "P1;Alpha;Vestas V112-3.0 Offshore;53,41;8,65;94;31.12.2013\n"
    "P2;Beta;E-82;51.50;10.20;;2015-06-01\n"
    "P2;Beta;E-82;51.51;10.21;108;2015-06-01\n"
    "P3;Broken;Vestas V136-3.45;;;100;2020-01-01\n"
).encode()


def test_find_columns_variants():
    c = parkfile.find_columns(["ParkID", "ParkName", "turbine_type", "lat", "lon", "HubHeight", "commissioning_date"])
    assert c == {"park_id": "ParkID", "park_name": "ParkName", "type": "turbine_type", "lat": "lat", "lon": "lon",
                 "hub": "HubHeight", "date": "commissioning_date"}


def test_find_columns_german():
    c = parkfile.find_columns(["Parkname", "ParkID", "Hersteller", "Typ", "Nabenhöhe (m)", "Rotordurchmesser (m)",
                               "Koordinaten Breite (WGS 84)", "Koordinaten Länge (WGS 84)"])
    assert (c["park_name"], c["park_id"], c["maker"], c["type"], c["hub"], c["lat"], c["lon"]) == (
        "Parkname", "ParkID", "Hersteller", "Typ", "Nabenhöhe (m)", "Koordinaten Breite (WGS 84)", "Koordinaten Länge (WGS 84)")
    c = parkfile.find_columns(["Name des Windparks", "Parkkennung", "WEA-Typ", "Breitengrad", "Längengrad", "Nabenhoehe", "Baujahr"])
    assert (c["park_name"], c["park_id"], c["type"], c["lat"], c["lon"], c["hub"], c["date"]) == (
        "Name des Windparks", "Parkkennung", "WEA-Typ", "Breitengrad", "Längengrad", "Nabenhoehe", "Baujahr")


def test_match_turbine_exact_and_closest():
    assert parkfile.match_turbine("vestas v136 3.45", KNOWN) == ("Vestas V136-3.45", "exact")
    name, how = parkfile.match_turbine("E-82", KNOWN)
    assert name == "Enercon E-82 E2 2.000" and how == "suggested"


def test_parse_groups_parks_and_fills_turbines():
    out = parkfile.parse_parks("turbines.csv", CSV, KNOWN)
    labels = [p["label"] for p in out["parks"]]
    assert labels == ["Alpha (P1)", "Beta (P2)"]                       # park with no valid position is skipped
    alpha, beta = out["parks"]
    assert alpha["n"] == 2 and alpha["turbines"][0]["lat"] == 53.4 and alpha["turbines"][0]["hub_height"] == 94
    assert alpha["turbines"][0]["commissioning_date"] == "2013-12-31" and alpha["turbines"][0]["match"] == "exact"
    assert beta["turbines"][0]["hub_height"] is None and beta["turbines"][1]["hub_height"] == 108
    assert beta["turbines"][0]["turbine"] == "Enercon E-82 E2 2.000" and beta["turbines"][0]["match"] == "suggested"
    assert any("skipped" in w for w in out["warnings"])


def test_parse_excel_and_missing_columns():
    pytest.importorskip("openpyxl")
    buf = io.BytesIO()
    pd.DataFrame({"park_id": ["A", "A"], "park_name": ["Gamma", "Gamma"], "type": [KNOWN[0]] * 2, "lat": [50.0, 50.1],
                  "lon": [9.0, 9.1], "hub_height": [100, 100]}).to_excel(buf, index=False)
    out = parkfile.parse_parks("t.xlsx", buf.getvalue(), KNOWN)
    assert out["parks"][0]["label"] == "Gamma (A)" and out["parks"][0]["n"] == 2
    with pytest.raises(ValueError):
        parkfile.parse_parks("bad.csv", b"a;b\n1;2\n", KNOWN)
