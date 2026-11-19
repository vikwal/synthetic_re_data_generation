"""Park spec: the input of one PV park for generate_pv.py, typed in by hand (dashboard, fictitious park) or built from
the MaStR (from_mastr). A spec file (YAML or JSON) holds one park or {parks: [...]}; run config key `parks`.

Required:
    park_id        str, letters/digits/_/- (also the file and cache key of the park)
    lat, lon       float, WGS84
    ac_kw          float > 0, AC rating (MaStR Nettonennleistung)
    dc_kwp         float > 0, DC rating (MaStR Bruttoleistung); DC/AC is clipped to 1.0-1.6 by the chain
    commissioning  date, first commissioning (aging starts here) - not needed with mastr.unit_ids
Optional:
    commissioning_full  date the park reached its rating (the series starts here), default commissioning
    cohorts        [{commissioning: date, dc_kwp: float, area_ha: float?}] age cohorts (construction phases); default one
    area_ha        fenced area (density -> row spacing); default from the year table
    geometry       {mode: estimate | register | manual, arrays: [{tilt, azimuth, share}]}; mode default = run config;
                   arrays: tilt against the horizontal, azimuth true (180 = south), shares normalised
    orientation_class, tilt_class, secondary_orientation, secondary_tilt_class, uniform_flag, power_limit
                   register classes in MaStR wording (Süd, Ost-West, ... / < 20 Grad, 20 - 40 Grad, ...); default:
                   derived from geometry.arrays, else Süd / 20 - 40 Grad
    bundesland, landkreis, gemeinde   default: from the nearest MaStR unit within NEAREST_KM (state of the geodata
                   fetch of mode estimate); none abroad
    mastr          {unit_ids: [...], lokation_ids: [...]} provenance of a MaStR park; with unit_ids the commissioning
                   dates and age cohorts come from the units (as in the releases)
    name           free label

to_row() turns a spec into a selection row (the columns of pv.selection), so every stage runs unchanged."""
import datetime
import json
import os
import re

import numpy as np
import pandas as pd
import yaml

import settings
from pv import selection as SEL
from pv.plausibility import orientation_class as az_class, tilt_class as tilt_cls

GEOMETRY_MODES = ("estimate", "register", "manual")
REQUIRED = ("park_id", "lat", "lon", "ac_kw", "dc_kwp")
OPTIONAL = ("commissioning", "commissioning_full", "cohorts", "area_ha", "geometry", "orientation_class", "tilt_class",
            "secondary_orientation", "secondary_tilt_class", "uniform_flag", "power_limit", "bundesland", "landkreis", "gemeinde",
            "mastr", "name")
ORIENTATIONS = ("Süd", "Süd-Ost", "Süd-West", "Ost", "West", "Ost-West", "Nord", "Nord-Ost", "Nord-West")
TILT_CLASSES = ("< 20 Grad", "20 - 40 Grad", "40 - 60 Grad", "> 60 Grad")
NEAREST_KM = 15.0
ROW_COLS = ["park_id", "client", "kind", "cls", "region", "cohort", "size_bin", "terrain_cls", "dc_kwp", "ac_kw", "inv_kw", "dc_ac", "year",
            "year_last", "n_units", "orientation_class", "tilt_class", "secondary_orientation", "secondary_tilt_class", "uniform_flag",
            "power_limit", "area_ha", "n_modules", "lat", "lon", "elevation", "slope", "elev_std", "bundesland", "landkreis", "operator",
            "n_operators", "eeg_tender", "n_lokationen", "merged_ids", "foreign_within_km", "unit_ids", "gemeinde",
            # spec columns (empty for MaStR selections)
            "commissioning_first", "commissioning_full", "cohorts", "geometry_mode", "geometry_arrays", "name"]


# ---------------------------------------------------------------- load and validate

def load(path: str) -> list:
    """Park specs of a YAML/JSON file (one park or {parks: [...]}), validated; raises ValueError with all errors."""
    try:
        raw = yaml.safe_load(open(os.path.expanduser(path)))
    except (yaml.YAMLError, ValueError) as ex:                    # also invalid dates such as 2024-13-01
        raise ValueError(f"{path}: not readable: {ex}") from None
    specs = raw.get("parks") if isinstance(raw, dict) and "parks" in raw else [raw] if isinstance(raw, dict) else raw
    if not isinstance(specs, list) or not specs:
        raise ValueError(f"{path}: no park spec")
    errs = [f"parks[{i}] ({s.get('park_id', '?') if isinstance(s, dict) else '?'}): {e}" for i, s in enumerate(specs) for e in validate(s)]
    ids = [s.get("park_id") for s in specs if isinstance(s, dict)]
    errs += [f"park_id {p!r} twice" for p in sorted({p for p in ids if ids.count(p) > 1})]
    if errs:
        raise ValueError(f"{path}:\n  " + "\n  ".join(errs))
    return specs


def _date(v):
    try:
        return pd.Timestamp(v).normalize() if v is not None and str(v) != "" else None
    except (ValueError, TypeError):
        return "bad"


def _num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and np.isfinite(v)


def validate(s: dict) -> list:
    """Error messages of one spec ([] = valid)."""
    if not isinstance(s, dict):
        return ["not a mapping"]
    e = [f"unknown key {k!r}" for k in s if k not in REQUIRED + OPTIONAL]
    e += [f"{k} missing" for k in REQUIRED if s.get(k) is None]
    if s.get("park_id") is not None and not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", str(s["park_id"])):
        e.append("park_id: letters, digits, _ and - only (max 64)")
    if s.get("lat") is not None and not (_num(s["lat"]) and -90 <= s["lat"] <= 90):
        e.append("lat out of range")
    if s.get("lon") is not None and not (_num(s["lon"]) and -180 <= s["lon"] <= 180):
        e.append("lon out of range")
    for k in ("ac_kw", "dc_kwp"):
        if s.get(k) is not None and not (_num(s[k]) and s[k] > 0):
            e.append(f"{k} must be > 0")
    if s.get("area_ha") is not None and not (_num(s["area_ha"]) and s["area_ha"] > 0):
        e.append("area_ha must be > 0")
    units = (s.get("mastr") or {}).get("unit_ids")
    if s.get("commissioning") is None and not units:
        e.append("commissioning missing (or mastr.unit_ids)")
    for k in ("commissioning", "commissioning_full"):
        if _date(s.get(k)) == "bad":
            e.append(f"{k}: not a date")
    if _date(s.get("commissioning")) not in (None, "bad") and _date(s.get("commissioning_full")) not in (None, "bad") \
            and _date(s["commissioning_full"]) < _date(s["commissioning"]):
        e.append("commissioning_full before commissioning")
    for j, c in enumerate(s.get("cohorts") or []):
        if not isinstance(c, dict) or _date(c.get("commissioning")) in (None, "bad") or not (_num(c.get("dc_kwp")) and c["dc_kwp"] > 0):
            e.append(f"cohorts[{j}]: needs commissioning (date) and dc_kwp > 0")
    if s.get("orientation_class") is not None and s["orientation_class"] not in ORIENTATIONS:
        e.append(f"orientation_class one of {ORIENTATIONS}")
    if s.get("tilt_class") is not None and s["tilt_class"] not in TILT_CLASSES:
        e.append(f"tilt_class one of {TILT_CLASSES}")
    g = s.get("geometry") or {}
    if not isinstance(g, dict) or set(g) - {"mode", "arrays"}:
        e.append("geometry: keys mode, arrays")
        g = {}
    if g.get("mode") is not None and g["mode"] not in GEOMETRY_MODES:
        e.append(f"geometry.mode one of {GEOMETRY_MODES}")
    arr = g.get("arrays")
    if g.get("mode") == "manual" and not arr:
        e.append("geometry.mode manual needs geometry.arrays")
    for j, a in enumerate(arr or []):
        if not isinstance(a, dict) or not all(_num(a.get(k)) for k in ("tilt", "azimuth", "share")):
            e.append(f"geometry.arrays[{j}]: needs tilt, azimuth, share")
        elif not (0 <= a["tilt"] <= 90 and 0 <= a["azimuth"] <= 360 and a["share"] > 0):
            e.append(f"geometry.arrays[{j}]: tilt 0-90, azimuth 0-360, share > 0")
    return e


# ---------------------------------------------------------------- derived fields

def _units():
    global _U
    if "_U" not in globals():
        from parks_pv import paths
        _U = pd.read_parquet(paths.mastr_file("units"), columns=["EinheitMastrNummer", "LokationMaStRNummer", "Breitengrad", "Laengengrad",
                                                                   "Bundesland", "Landkreis", "Gemeinde"])
    return _U


def admin_area(lat: float, lon: float) -> dict:
    """Bundesland, Landkreis, Gemeinde of the nearest MaStR unit within NEAREST_KM (None abroad / far from any unit)."""
    global _TREE
    from scipy.spatial import cKDTree
    u = _units().dropna(subset=["Breitengrad", "Laengengrad", "Bundesland"])
    if "_TREE" not in globals():
        _TREE = (cKDTree(SEL._km_xy(u.Breitengrad.values, u.Laengengrad.values)), u.reset_index(drop=True))
    d, i = _TREE[0].query(SEL._km_xy(np.array([lat]), np.array([lon])))
    if float(d[0]) > NEAREST_KM:
        return dict(bundesland=None, landkreis=None, gemeinde=None)
    x = _TREE[1].iloc[int(i[0])]
    return dict(bundesland=x.Bundesland, landkreis=x.Landkreis, gemeinde=x.Gemeinde)


def terrain_metrics(lat: float, lon: float) -> dict:
    """Point terrain metrics as the MaStR pool table (pool_terrain_*.parquet): elevation, slope, aspect, tpi5, tdi,
    elev_std, tpi75 from the 90 m and 1 km DEM mosaics (round2.topo.location_metrics)."""
    global _DEM
    import rasterio
    from round2.topo import location_metrics
    if "_DEM" not in globals():
        def load(p):
            with rasterio.open(p) as ds:
                a = ds.read(1).astype(float)
                if ds.nodata is not None:
                    a[a == ds.nodata] = np.nan
                return {"arr": a, "transform": ds.transform}
        _DEM = load(settings.get("dem_mosaic_90m")), load(settings.get("dem_mosaic_1km"))
    m = location_metrics(_DEM[0], _DEM[1], lat, lon)
    return {k: m.get(k, np.nan) for k in ("elevation", "slope", "aspect", "tpi5", "tdi", "elev_std", "tpi75")}


def classes_from_arrays(arrays: list) -> tuple:
    """(orientation class, tilt class) of given sub-arrays: an opposite east/west pair is Ost-West, else the class of
    the largest array; tilt class of the share-weighted tilt."""
    from parks_pv.geometry_estimate import measured_classes
    az = [float(a["azimuth"]) for a in arrays]
    w = np.array([float(a["share"]) for a in arrays])
    o = "Ost-West" if "Ost-West" in measured_classes(az) else az_class(az[int(np.argmax(w))])
    return o, tilt_cls(float(np.average([float(a["tilt"]) for a in arrays], weights=w)))


def _cohort_rows(s: dict, first: pd.Timestamp) -> list:
    if s.get("cohorts"):
        return [dict(commissioning=f"{_date(c['commissioning']):%Y-%m-%d}", dc_kwp=float(c["dc_kwp"]),
                     area_ha=float(c["area_ha"]) if _num(c.get("area_ha")) else None)
                for c in sorted(s["cohorts"], key=lambda c: _date(c["commissioning"]))]
    return [dict(commissioning=f"{first:%Y-%m-%d}", dc_kwp=float(s["dc_kwp"]), area_ha=float(s["area_ha"]) if _num(s.get("area_ha")) else None)]


def to_row(s: dict) -> dict:
    """Selection row of a valid spec (columns ROW_COLS)."""
    units = list((s.get("mastr") or {}).get("unit_ids") or [])
    loks = list((s.get("mastr") or {}).get("lokation_ids") or [])
    g = s.get("geometry") or {}
    arrays = g.get("arrays") or []
    if units and s.get("commissioning") is None:              # commissioning dates from the MaStR units
        from parks_pv import params
        d = params.commissioning_dates(pd.DataFrame([dict(park_id=s["park_id"], unit_ids=";".join(units))]))
        first, full = d.commissioning_first.iloc[0], d.commissioning_full.iloc[0]
    else:
        first = _date(s["commissioning"])
        full = _date(s.get("commissioning_full")) or max([first] + [_date(c["commissioning"]) for c in s.get("cohorts") or []])
    o_cls, t_cls = s.get("orientation_class"), s.get("tilt_class")
    if (o_cls is None or t_cls is None) and arrays:
        o2, t2 = classes_from_arrays(arrays)
        o_cls, t_cls = o_cls or o2, t_cls or t2
    o_cls, t_cls = o_cls or "Süd", t_cls or "20 - 40 Grad"
    adm = {k: s.get(k) for k in ("bundesland", "landkreis", "gemeinde")}
    if adm["bundesland"] is None:
        adm = {k: adm[k] or v for k, v in admin_area(float(s["lat"]), float(s["lon"])).items()}
    t = terrain_metrics(float(s["lat"]), float(s["lon"]))
    sec, unif = s.get("secondary_orientation"), s.get("uniform_flag")
    cls = SEL.pv_class(pd.Series(dict(has_secondary=bool(sec), mixed_orientation=False, uniform_flag=unif, orientation_class=o_cls,
                                      tilt_class=t_cls))) or ""
    cohorts = _cohort_rows(s, first) if not units else None
    return dict(
        park_id=str(s["park_id"]), client="", kind="spec", cls=cls, region=np.nan, cohort=SEL.cohort(first.year),
        size_bin=str(pd.cut([float(s["dc_kwp"]) / 1e3], SEL.SIZE_BINS, labels=SEL.SIZE_LABELS, right=False)[0]),
        terrain_cls=str(pd.cut([t["elev_std"]], SEL.TERRAIN_BINS, labels=SEL.TERRAIN_LABELS)[0]),
        dc_kwp=float(s["dc_kwp"]), ac_kw=float(s["ac_kw"]), inv_kw=np.nan, dc_ac=round(float(s["dc_kwp"]) / float(s["ac_kw"]), 3),
        year=int(first.year), year_last=int(full.year), n_units=len(units) or len(cohorts or [1]),
        orientation_class=o_cls, tilt_class=t_cls, secondary_orientation=sec, secondary_tilt_class=s.get("secondary_tilt_class"),
        uniform_flag=unif, power_limit=s.get("power_limit"), area_ha=float(s["area_ha"]) if _num(s.get("area_ha")) else np.nan,
        n_modules=np.nan, lat=float(s["lat"]), lon=float(s["lon"]), elevation=round(float(t["elevation"]), 1),
        slope=round(float(t["slope"]), 2), elev_std=round(float(t["elev_std"]), 1), operator="", n_operators=1, eeg_tender=np.nan,
        n_lokationen=max(len(loks), 1), merged_ids=";".join(loks) if loks else np.nan, foreign_within_km=np.nan, unit_ids=";".join(units),
        commissioning_first=f"{first:%Y-%m-%d}", commissioning_full=f"{full:%Y-%m-%d}", cohorts=json.dumps(cohorts) if cohorts else "",
        geometry_mode=g.get("mode") or "", geometry_arrays=json.dumps(arrays) if arrays else "", name=s.get("name") or "", **adm)


def to_selection(specs: list) -> pd.DataFrame:
    return pd.DataFrame([to_row(s) for s in specs]).reindex(columns=ROW_COLS)


def write_selection(path_spec: str, out_csv: str) -> str:
    """Selection CSV of a spec file (rewritten on every run, the spec file is the source)."""
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    to_selection(load(path_spec)).to_csv(out_csv, index=False)
    return out_csv


# ---------------------------------------------------------------- MaStR adapter

def from_mastr(lokation_ids, park_id: str = None, geometry: dict = None) -> dict:
    """Spec of a MaStR park from its Lokation number(s) (a construction site registered as several Lokationen of one
    operator: all of them; the largest is the head and the default park_id, as pv.selection.merge_same_operator). The
    register fields come from the aggregated units (pv.mastr.aggregate_parks)."""
    from parks_pv import paths
    from pv import mastr
    loks = [lokation_ids] if isinstance(lokation_ids, str) else list(lokation_ids)
    u = pd.read_parquet(paths.mastr_file("units"))
    u = u[u.LokationMaStRNummer.isin(loks)].copy()
    if not len(u):
        raise KeyError(f"no MaStR units for Lokation {loks}")
    head = u.groupby("LokationMaStRNummer").Bruttoleistung.sum().idxmax()
    u["LokationMaStRNummer"] = head
    eeg = pd.read_parquet(paths.mastr_file("eeg")) if os.path.exists(paths.mastr_file("eeg")) else None
    p = mastr.aggregate_parks(u, eeg).iloc[0]
    def na(v):                                                    # NaN -> None, numpy scalars -> Python (YAML/JSON)
        if v is None or (not isinstance(v, (list, dict)) and pd.isna(v)):
            return None
        return v.item() if isinstance(v, np.generic) else v
    spec = dict(park_id=park_id or str(p.park_id), lat=round(float(p.lat), 5), lon=round(float(p.lon), 5),   # rounding as pv.selection
                ac_kw=round(float(p.ac_kw), 1), dc_kwp=round(float(p.dc_kwp), 1),
                area_ha=na(p.area_ha), orientation_class=na(p.orientation_class), tilt_class=na(p.tilt_class),
                secondary_orientation=na(p.secondary_orientation), secondary_tilt_class=na(p.secondary_tilt_class),
                uniform_flag=na(p.uniform_flag), power_limit=na(p.power_limit), bundesland=na(p.bundesland), landkreis=na(p.landkreis),
                gemeinde=na(p.gemeinde), mastr=dict(unit_ids=sorted(u.EinheitMastrNummer.astype(str)), lokation_ids=sorted(loks)))
    if geometry:
        spec["geometry"] = geometry
    return {k: v for k, v in spec.items() if v is not None}


def dump(specs: list, path: str):
    """Write specs as YAML ({parks: [...]})."""
    conv = lambda v: v.isoformat() if isinstance(v, (datetime.date, pd.Timestamp)) else v
    yaml.safe_dump({"parks": [{k: conv(v) for k, v in s.items()} for s in specs]}, open(path, "w"), allow_unicode=True, sort_keys=False)


def main():
    """CLI: check a spec file, or write the spec of a MaStR park.

        synthre/bin/python -m parks_pv.park_spec check parks.yaml          # errors, else the derived selection rows
        synthre/bin/python -m parks_pv.park_spec mastr SEL... [SEL...] --out parks.yaml [--park-id ID] [--geometry estimate]"""
    import argparse
    ap = argparse.ArgumentParser(description=main.__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("path")
    m = sub.add_parser("mastr")
    m.add_argument("lokation_ids", nargs="+")
    m.add_argument("--out", required=True)
    m.add_argument("--park-id", default=None)
    m.add_argument("--geometry", choices=GEOMETRY_MODES, default=None)
    a = ap.parse_args()
    if a.cmd == "check":
        try:
            specs = load(a.path)
        except ValueError as ex:
            raise SystemExit(str(ex))
        rows = to_selection(specs)
        print(f"{len(specs)} parks valid")
        print(rows[["park_id", "dc_kwp", "ac_kw", "year", "year_last", "orientation_class", "tilt_class", "cls", "bundesland", "terrain_cls",
                    "geometry_mode"]].to_string(index=False))
    else:
        dump([from_mastr(a.lokation_ids, a.park_id, {"mode": a.geometry} if a.geometry else None)], a.out)
        print("written", a.out)


if __name__ == "__main__":
    main()
