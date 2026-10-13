"""Marktstammdatenregister (MaStR) bulk export: stream the solar unit XML files, decode catalogue ids,
join the EEG plant data and aggregate units to parks (one park = one LokationMaStRNummer).

The export (UTF-16LE, 54 EinheitenSolar_*.xml of 150-360 MB) is parsed with ElementTree.iterparse,
about 3 s per file, so a full pass over the register takes a few minutes with a process pool. Only
units above a power threshold or on open ground (Freifläche) are kept - the register holds 5 M
rooftop units that are irrelevant for park synthesis."""
import glob
import os
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

# EinheitSolar children kept as columns (catalogue ids are decoded in decode_catalogue)
UNIT_FIELDS = ["EinheitMastrNummer", "LokationMaStRNummer", "EegMaStRNummer", "AnlagenbetreiberMastrNummer",
               "EinheitBetriebsstatus", "Inbetriebnahmedatum", "DatumEndgueltigeStilllegung", "Registrierungsdatum",
               "Bundesland", "Landkreis", "Gemeinde", "Gemeindeschluessel", "Postleitzahl", "Breitengrad", "Laengengrad",
               "Bruttoleistung", "Nettonennleistung", "ZugeordneteWirkleistungWechselrichter", "AnzahlModule",
               "ArtDerSolaranlage", "Hauptausrichtung", "HauptausrichtungNeigungswinkel", "Nebenausrichtung",
               "NebenausrichtungNeigungswinkel", "EinheitlicheAusrichtungUndNeigungswinkel", "Leistungsbegrenzung",
               "InAnspruchGenommeneFlaeche", "InAnspruchGenommeneAckerflaeche", "ArtDerFlaecheIds", "Einspeisungsart",
               "FernsteuerbarkeitNb", "FernsteuerbarkeitDv", "SpeicherAmGleichenOrt", "NameStromerzeugungseinheit"]
EEG_FIELDS = ["EegMaStRNummer", "AnlagenschluesselEeg", "InstalliertePVLeistung", "InstallierteLeistung",
              "EegInbetriebnahmedatum", "Zuschlagsnummer", "AusschreibungZuschlag", "AnlageBetriebsstatus"]
NUMERIC = ["Breitengrad", "Laengengrad", "Bruttoleistung", "Nettonennleistung", "ZugeordneteWirkleistungWechselrichter",
           "AnzahlModule", "InAnspruchGenommeneFlaeche", "InAnspruchGenommeneAckerflaeche"]
CATALOGUE = ["EinheitBetriebsstatus", "ArtDerSolaranlage", "Hauptausrichtung", "HauptausrichtungNeigungswinkel",
             "Nebenausrichtung", "NebenausrichtungNeigungswinkel", "Leistungsbegrenzung", "Einspeisungsart",
             "Bundesland", "FernsteuerbarkeitNb", "FernsteuerbarkeitDv"]
DATES = ["Inbetriebnahmedatum", "DatumEndgueltigeStilllegung", "Registrierungsdatum", "EegInbetriebnahmedatum"]
# catalogue ids used in filters (Katalogwerte.xml of the 2025-06-25 export)
ID_FREIFLAECHE, ID_IN_BETRIEB, ID_TRACKING_AZ, ID_TRACKING_TILT, ID_OST_WEST = 852, 35, 703, 811, 704
AZ_DEG = {"Nord": 0.0, "Nord-Ost": 45.0, "Ost": 90.0, "Süd-Ost": 135.0, "Süd": 180.0, "Süd-West": 225.0, "West": 270.0,
          "Nord-West": 315.0}


def load_catalogue(mastr_dir: str) -> dict:
    root = ET.parse(os.path.join(mastr_dir, "Katalogwerte.xml")).getroot()
    return {int(k.find("Id").text): k.find("Wert").text for k in root}


def _stream(path: str, tag: str, fields: list, keep) -> pd.DataFrame:
    rows = []
    for _, el in ET.iterparse(path, events=("end",)):
        if el.tag == tag:
            d = {c.tag: c.text for c in el}
            if keep(d):
                rows.append({f: d.get(f) for f in fields})
            el.clear()
    return pd.DataFrame(rows, columns=fields)


def _keep_unit(min_kw: float):
    def keep(d):
        try:
            kw = float(d.get("Bruttoleistung") or 0.0)
        except ValueError:
            kw = 0.0
        return kw >= min_kw or d.get("ArtDerSolaranlage") == str(ID_FREIFLAECHE)
    return keep


def parse_units_file(path: str, min_kw: float = 100.0) -> pd.DataFrame:
    """Units of one EinheitenSolar_*.xml with Bruttoleistung >= min_kw or on open ground."""
    df = _stream(path, "EinheitSolar", UNIT_FIELDS, _keep_unit(min_kw))
    df["source_file"] = os.path.basename(path)
    return df


def parse_eeg_file(path: str) -> pd.DataFrame:
    return _stream(path, "AnlageEegSolar", EEG_FIELDS, lambda d: True)


def decode_catalogue(df: pd.DataFrame, kat: dict) -> pd.DataFrame:
    """Numeric conversion, catalogue decoding (id columns kept as <col>_id) and dates."""
    df = df.copy()
    for c in NUMERIC:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in CATALOGUE:
        if c in df:
            ids = pd.to_numeric(df[c], errors="coerce")
            df[c + "_id"] = ids.astype("Int64")
            df[c] = ids.map(lambda v: kat.get(int(v)) if pd.notna(v) else None)
    for c in DATES:
        if c in df:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


def parse_register(mastr_dir: str, min_kw: float = 100.0, workers: int = 12) -> tuple:
    """All solar units (filtered) and all EEG plant rows of the export as two frames."""
    ufiles = sorted(glob.glob(os.path.join(mastr_dir, "EinheitenSolar_*.xml")))
    efiles = sorted(glob.glob(os.path.join(mastr_dir, "AnlagenEegSolar_*.xml")))
    with ProcessPoolExecutor(workers) as ex:
        units = pd.concat(ex.map(parse_units_file, ufiles, [min_kw] * len(ufiles)), ignore_index=True)
        eeg = pd.concat(ex.map(parse_eeg_file, efiles), ignore_index=True)
    kat = load_catalogue(mastr_dir)
    units = decode_catalogue(units, kat)
    eeg = decode_catalogue(eeg, kat)
    for c in ("InstalliertePVLeistung", "InstallierteLeistung"):
        eeg[c] = pd.to_numeric(eeg[c], errors="coerce")
    return units, eeg


def _mode_weighted(s: pd.Series, w: pd.Series):
    """Value with the largest summed weight (NaN ignored)."""
    m = s.notna()
    if not m.any():
        return None
    return w[m].groupby(s[m]).sum().idxmax()


def aggregate_parks(units: pd.DataFrame, eeg: pd.DataFrame = None) -> pd.DataFrame:
    """One row per LokationMaStRNummer: capacities summed, dates min/max, orientation/tilt classes of the
    largest share, flags for mixed orientation, secondary orientation and power limitation. Units without a
    Lokation form their own park keyed by the unit number."""
    u = units.copy()
    u["park_id"] = u["LokationMaStRNummer"].fillna("U_" + u["EinheitMastrNummer"].astype(str))
    w = u["Bruttoleistung"].fillna(0.0)
    g = u.groupby("park_id", sort=False)
    out = pd.DataFrame({
        "n_units": g.size(),
        "dc_kwp": g["Bruttoleistung"].sum(min_count=1),
        "ac_kw": g["Nettonennleistung"].sum(min_count=1),
        "inv_kw": g["ZugeordneteWirkleistungWechselrichter"].sum(min_count=1),
        "n_modules": g["AnzahlModule"].sum(min_count=1),
        "area_ha": g["InAnspruchGenommeneFlaeche"].sum(min_count=1),
        "lat": g.apply(lambda x: np.average(x["Breitengrad"], weights=x["Bruttoleistung"].fillna(0) + 1e-9) if x["Breitengrad"].notna().all() else np.nan),
        "lon": g.apply(lambda x: np.average(x["Laengengrad"], weights=x["Bruttoleistung"].fillna(0) + 1e-9) if x["Laengengrad"].notna().all() else np.nan),
        "coord_spread_km": g.apply(lambda x: float(np.hypot((x["Breitengrad"].max() - x["Breitengrad"].min()) * 111.2,
                                                           (x["Laengengrad"].max() - x["Laengengrad"].min()) * 111.2 * np.cos(np.deg2rad(x["Breitengrad"].mean()))))),
        "commissioned_first": g["Inbetriebnahmedatum"].min(),
        "commissioned_last": g["Inbetriebnahmedatum"].max(),
        "status": g["EinheitBetriebsstatus"].agg(lambda s: "/".join(sorted(set(s.dropna())))),
        "mounting": g["ArtDerSolaranlage"].agg(lambda s: "/".join(sorted(set(s.dropna())))),
        "orientation_class": g.apply(lambda x: _mode_weighted(x["Hauptausrichtung"], x["Bruttoleistung"].fillna(0))),
        "tilt_class": g.apply(lambda x: _mode_weighted(x["HauptausrichtungNeigungswinkel"], x["Bruttoleistung"].fillna(0))),
        "n_orientation_classes": g["Hauptausrichtung"].nunique(),
        "secondary_orientation": g["Nebenausrichtung"].agg(lambda s: "/".join(sorted(set(s.dropna()))) or None),
        "secondary_tilt_class": g["NebenausrichtungNeigungswinkel"].agg(lambda s: "/".join(sorted(set(s.dropna()))) or None),
        "uniform_flag": g["EinheitlicheAusrichtungUndNeigungswinkel"].agg(lambda s: "/".join(sorted(set(s.dropna().astype(str)))) or None),
        "power_limit": g["Leistungsbegrenzung"].agg(lambda s: "/".join(sorted(set(s.dropna()))) or None),
        "n_operators": g["AnlagenbetreiberMastrNummer"].nunique(),
        "operator": g["AnlagenbetreiberMastrNummer"].first(),
        "bundesland": g["Bundesland"].first(),
        "landkreis": g["Landkreis"].first(),
        "gemeinde": g["Gemeinde"].first(),
        "plz": g["Postleitzahl"].first(),
        "unit_ids": g["EinheitMastrNummer"].agg(lambda s: ";".join(s.astype(str))),
        "eeg_ids": g["EegMaStRNummer"].agg(lambda s: ";".join(sorted(set(s.dropna().astype(str))))),
    })
    out["has_secondary"] = out["secondary_orientation"].notna()
    out["mixed_orientation"] = out["n_orientation_classes"] > 1
    out["dc_ac"] = out["dc_kwp"] / out["ac_kw"]
    out["year"] = out["commissioned_first"].dt.year
    if eeg is not None and len(eeg):
        e = eeg.dropna(subset=["EegMaStRNummer"]).drop_duplicates("EegMaStRNummer").set_index("EegMaStRNummer")
        ids = out["eeg_ids"].str.split(";")
        out["eeg_keys"] = ids.map(lambda l: ";".join(sorted({str(e["AnlagenschluesselEeg"].get(i)) for i in l if i in e.index} - {"None", "nan"})) or None)
        out["eeg_tender"] = ids.map(lambda l: any(str(e["Zuschlagsnummer"].get(i) or "") not in ("", "None", "nan") for i in l if i in e.index))
    return out.reset_index()


def orientation_deg(cls: str):
    """Catalogue orientation class -> azimuth in degrees (pvlib convention, 180 = south); None for
    tracking / east-west / unknown."""
    return AZ_DEG.get(cls)
