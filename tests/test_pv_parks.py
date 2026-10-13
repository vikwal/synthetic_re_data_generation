"""Tests for the parks_pv_v1 selection chain: MaStR aggregation, park classes, same-operator merging and
CAMS ring geometry. Synthetic inputs only."""
import numpy as np
import pandas as pd
import pytest

from pv import cams_points as CP
from pv import mastr
from pv import selection as S


def _units():
    base = {"EinheitBetriebsstatus": "In Betrieb", "ArtDerSolaranlage": "Freifläche", "Hauptausrichtung": "Süd",
            "HauptausrichtungNeigungswinkel": "20 - 40 Grad", "Nebenausrichtung": None, "NebenausrichtungNeigungswinkel": None,
            "EinheitlicheAusrichtungUndNeigungswinkel": "1", "Leistungsbegrenzung": "Nein", "AnlagenbetreiberMastrNummer": "ABR1",
            "Bundesland": "Bayern", "Landkreis": "X", "Gemeinde": "Y", "Postleitzahl": "90000", "InAnspruchGenommeneFlaeche": 2.0,
            "ZugeordneteWirkleistungWechselrichter": 900.0, "AnzahlModule": 2000.0, "EegMaStRNummer": None}
    rows = [
        {**base, "EinheitMastrNummer": "SEE1", "LokationMaStRNummer": "SEL1", "Breitengrad": 49.0, "Laengengrad": 11.0,
         "Bruttoleistung": 1000.0, "Nettonennleistung": 900.0, "Inbetriebnahmedatum": "2012-05-01"},
        {**base, "EinheitMastrNummer": "SEE2", "LokationMaStRNummer": "SEL1", "Breitengrad": 49.001, "Laengengrad": 11.0,
         "Bruttoleistung": 3000.0, "Nettonennleistung": 2700.0, "Inbetriebnahmedatum": "2014-05-01", "Hauptausrichtung": "Ost-West",
         "HauptausrichtungNeigungswinkel": "< 20 Grad"},
        {**base, "EinheitMastrNummer": "SEE3", "LokationMaStRNummer": "SEL2", "Breitengrad": 49.002, "Laengengrad": 11.0,
         "Bruttoleistung": 2000.0, "Nettonennleistung": 1800.0, "Inbetriebnahmedatum": "2021-01-01"},
        {**base, "EinheitMastrNummer": "SEE4", "LokationMaStRNummer": "SEL3", "Breitengrad": 49.003, "Laengengrad": 11.0,
         "Bruttoleistung": 1500.0, "Nettonennleistung": 1400.0, "Inbetriebnahmedatum": "2021-01-01", "AnlagenbetreiberMastrNummer": "ABR2"},
        {**base, "EinheitMastrNummer": "SEE5", "LokationMaStRNummer": "SEL4", "Breitengrad": 51.0, "Laengengrad": 9.0,
         "Bruttoleistung": 1200.0, "Nettonennleistung": 1000.0, "Inbetriebnahmedatum": "2019-01-01", "Nebenausrichtung": "Ost"},
    ]
    u = pd.DataFrame(rows)
    u["Inbetriebnahmedatum"] = pd.to_datetime(u["Inbetriebnahmedatum"])
    return u


def test_aggregate_parks_sums_and_weighted_classes():
    p = mastr.aggregate_parks(_units()).set_index("park_id")
    assert p.loc["SEL1", "n_units"] == 2 and p.loc["SEL1", "dc_kwp"] == 4000.0 and p.loc["SEL1", "ac_kw"] == 3600.0
    assert p.loc["SEL1", "orientation_class"] == "Ost-West"          # class of the largest capacity share
    assert p.loc["SEL1", "mixed_orientation"] and p.loc["SEL1", "year"] == 2012
    assert p.loc["SEL4", "has_secondary"] and not p.loc["SEL2", "has_secondary"]
    assert abs(p.loc["SEL1", "dc_ac"] - 4000 / 3600) < 1e-9


def test_pv_class_rules():
    p = mastr.aggregate_parks(_units()).set_index("park_id")
    cls = {k: S.pv_class(r) for k, r in zip(p.index, p.itertuples())}
    assert cls == {"SEL1": "C5_multi", "SEL2": "C1_south_std", "SEL3": "C1_south_std", "SEL4": "C5_multi"}
    r = p.loc["SEL2"].copy()
    r["orientation_class"], r["tilt_class"] = "Süd", "< 20 Grad"
    assert S.pv_class(r) == "C2_south_flat"
    r["orientation_class"] = "Süd-West"
    assert S.pv_class(r) == "C4_off_south"
    r["orientation_class"] = "Nord"
    assert S.pv_class(r) is None


def test_merge_same_operator_and_foreign_flag():
    u = _units()
    parks = mastr.aggregate_parks(u)
    merged = S.merge_same_operator(parks, u, radius_km=0.5).set_index("park_id")
    # SEL1 (ABR1) and SEL2 (ABR1, 220 m north) merge under the larger SEL1; SEL3 belongs to ABR2 and stays
    assert "SEL2" not in merged.index and merged.loc["SEL1", "n_lokationen"] == 2
    assert merged.loc["SEL1", "dc_kwp"] == 6000.0 and merged.loc["SEL1", "merged_ids"] == "SEL1;SEL2"
    assert merged.loc["SEL3", "n_lokationen"] == 1 and merged.loc["SEL4", "n_lokationen"] == 1
    f = S.foreign_neighbour_km(merged.reset_index())
    fk = dict(zip(merged.index, f))
    assert 0.1 < fk["SEL3"] < 0.3 and np.isnan(fk["SEL4"])


def test_ring_points_geometry():
    pts = pd.DataFrame(CP.ring_points("P", 50.0, 10.0))
    assert len(pts) == 9 and pts.kind.tolist().count("ring") == 8
    n3 = pts[pts.point_id == "P_r3_N"].iloc[0]
    e8 = pts[pts.point_id == "P_r8_E"].iloc[0]
    assert abs(n3.lon - 10.0) < 1e-9 and abs((n3.lat - 50.0) * 111.2 - 3.0) < 0.05
    assert abs(e8.lat - 50.0) < 1e-3 and abs((e8.lon - 10.0) * 111.32 * np.cos(np.deg2rad(50)) - 8.0) < 0.05


def test_selection_fills_slots_with_spread(monkeypatch):
    rng = np.random.default_rng(0)
    n = 400
    pool = pd.DataFrame({
        "park_id": [f"S{i}" for i in range(n)], "reject": None, "cls": rng.choice(S.CLASSES, n),
        "region": rng.integers(0, 4, n), "cohort": rng.choice(["<=2012", "2013-20", "2021+"], n),
        "size_bin": rng.choice(S.SIZE_LABELS, n), "terrain_cls": rng.choice(S.TERRAIN_LABELS, n),
        "dc_kwp": rng.uniform(750, 20000, n), "ac_kw": 1000.0, "inv_kw": 1000.0, "dc_ac": 1.1, "year": 2015,
        "commissioned_last": pd.Timestamp("2015-01-01"), "n_units": 1, "orientation_class": "Süd", "tilt_class": "< 20 Grad",
        "secondary_orientation": None, "secondary_tilt_class": None, "uniform_flag": "1", "power_limit": None, "area_ha": 2.0,
        "n_modules": 1000.0, "lat": 50.0, "lon": 10.0, "elevation": 100.0, "slope": 1.0, "elev_std": 20.0, "bundesland": "B",
        "landkreis": "L", "operator": "O", "n_operators": 1, "eeg_tender": False, "n_lokationen": 1, "merged_ids": "x",
        "foreign_within_km": np.nan, "unit_ids": "u"})
    sel = S.select(pool, seed=1)
    assert len(sel) == 90 and sel.park_id.notna().all() and sel.park_id.is_unique
    assert (sel.groupby(["client", "cls"]).size() == 2).all()
    assert (sel[sel.kind == "regional"].groupby("client").region.nunique() == 1).all()
    assert (sel[sel.kind == "nationwide"].groupby("client").region.nunique() == 4).all()
