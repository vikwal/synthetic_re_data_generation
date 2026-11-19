"""Shared loaders for the Trianel PV benchmark. Everything here touches CONFIDENTIAL inputs
(names, coordinates, meter ids); callers must only persist the anonymised views (pv1..pv7 by
descending capacity) that these functions return.

Inputs
  data/wind_farm_meterdata/masterdata_solar_20240601.csv, meterdata_solar_20240601.csv
  MaStR match (optional): a candidates CSV (pv, EinheitMastrNummer, dist_km, ...) and a JSON of
  XML unit blocks keyed by EinheitMastrNummer (scripts in the FL_Contribution scratch; see report)
  CAMS ADS CSVs per park (CAMS_rad_<pv>_<year>.csv), PVGIS horizon JSON {pv: {az: elev}}
"""

import json
import os

import numpy as np
import pandas as pd

from pv import drivers, timebase, config as pvconfig, modules

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
METER_DIR = os.path.join(REPO, "data", "wind_farm_meterdata")
DEM_MOSAIC = "/mnt/nvme2/synthetic/raw/round2/dem/mosaic_90m.tif"
STATIONS_MASTER = os.path.expanduser("~/Work/stations_master.csv")

# Trianel module strings -> library names (bundled SAM CEC 2019 edition); /Bp variants are not
# listed, /Bx is the same cell/wattage class.
KNOWN_MODULES = {"JC245M": "Renesola America JC245M-24/Bx", "JC250M": "Renesola America JC250M-24/Bx",
                 "REC 245": "REC Solar REC245PE", "CS6P": "Canadian Solar Inc. CS6P-250P"}
MODULE_YEAR_HINT = {"JC245M": 2012, "JC250M": 2012, "REC 245": 2012, "CS6P": 2013}


def load_masterdata() -> pd.DataFrame:
    m = pd.read_csv(os.path.join(METER_DIR, "masterdata_solar_20240601.csv"), sep=None, engine="python")
    m = m.rename(columns={"Installierte Leistung (kW)": "kw", "Azimut von (°)": "az0", "Azimut bis (°)": "az1",
                          "Neigung von (°)": "tilt0", "Neigung bis (°)": "tilt1", "Koordinaten Breite (WGS 84)": "lat",
                          "Koordinaten Länge (WGS 84)": "lon", "Malo-ID": "malo", "ParkID": "park", "Typ": "module_type",
                          "Hersteller": "manufacturer", "Abregelungsmodus": "curtail_mode"})
    rank = m.groupby("park")["kw"].sum().sort_values(ascending=False)
    m["pv"] = m["park"].map({p: f"pv{i + 1}" for i, p in enumerate(rank.index)})
    return m


def dem_altitude(lat: float, lon: float) -> float:
    import rasterio
    with rasterio.open(DEM_MOSAIC) as d:
        return float(list(d.sample([(lon, lat)]))[0][0])


def parks_table(m: pd.DataFrame) -> pd.DataFrame:
    """One row per park (anonymised id, capacity, coordinates, arrays from the master data)."""
    rows = []
    for pv, g in m.groupby("pv"):
        kw = float(g["kw"].sum())
        arrays = []
        for _, r in g.iterrows():
            tilts = sorted({float(r["tilt0"]), float(r["tilt1"])})
            az = float(np.mean([r["az0"], r["az1"]]))
            for t in tilts:
                arrays.append({"tilt": t, "azimuth": az, "share": float(r["kw"]) / kw / len(tilts)})
        mt = str(g["module_type"].iloc[0]) if g["module_type"].notna().any() else ""
        key = next((k for k in KNOWN_MODULES if k in mt), None)
        rows.append({"pv": pv, "kw_master": kw, "lat": float(g["lat"].iloc[0]), "lon": float(g["lon"].iloc[0]),
                     "coords_rounded": bool(abs(g["lat"].iloc[0] * 1000 - round(g["lat"].iloc[0] * 1000)) < 1e-6),
                     "n_rows": len(g), "arrays": arrays, "module_key": key,
                     "module_lib": KNOWN_MODULES.get(key), "year_hint": MODULE_YEAR_HINT.get(key),
                     "malo": str(int(g["malo"].iloc[0])), "curtail_mode": str(g["curtail_mode"].iloc[0]),
                     "tilt_master": float(np.average([a["tilt"] for a in arrays], weights=[a["share"] for a in arrays])),
                     "az_master": float(np.average([a["azimuth"] for a in arrays], weights=[a["share"] for a in arrays]))})
    p = pd.DataFrame(rows).set_index("pv").sort_index(key=lambda s: s.str[2:].astype(int))
    p["alt"] = [dem_altitude(r.lat, r.lon) for r in p.itertuples()]
    return p


def load_meter_hourly(parks: pd.DataFrame, min_quarters: int = 3) -> pd.DataFrame:
    """Hourly mean power in W per park (15-min values are labelled by interval START)."""
    md = pd.read_csv(os.path.join(METER_DIR, "meterdata_solar_20240601.csv"), sep=";", index_col=0, parse_dates=True)
    md.index = pd.to_datetime(md.index, utc=True)
    out = {}
    for pv, r in parks.iterrows():
        if r["malo"] not in md.columns:
            continue
        s = md[r["malo"]].astype(float).clip(lower=0.0) * 1000.0
        h = s.resample("1h", closed="left", label="left").mean()
        n = s.resample("1h", closed="left", label="left").count()
        out[pv] = h.where(n >= min_quarters)
    return pd.DataFrame(out)


# MaStR locations of the parks whose coordinates miss the register by kilometres (capacity match of trianel_mastr_match.py,
# accepted 2026-10-21: nearest candidate within the tolerance; pv1 register point 27 km off, pv6 Trianel point 3 km off)
MASTR_LOCATIONS = {"pv1": ["SEL991360642453"], "pv6": ["SEL918966805471", "SEL957974947865"]}
MASTR_UNITS = os.path.join(os.environ.get("DATA_ROOT", "/mnt/lambda1/nvme1"), "synthetic", "solar", "mastr", "solar_units_20250625.parquet")


def _facts_from_locations(locs: list, units_path: str = MASTR_UNITS) -> dict:
    """mastr_facts row from all units in operation at the given MaStR locations (register snapshot parquet)."""
    u = pd.read_parquet(units_path)
    u = u[u["LokationMaStRNummer"].isin(locs) & (u["EinheitBetriebsstatus"] == "In Betrieb")]
    if u.empty:
        raise ValueError(f"no units in operation at {locs}")
    num = lambda k: float(pd.to_numeric(u[k], errors="coerce").sum()) or None
    txt = lambda k: "/".join(sorted({str(v) for v in u[k].dropna()} - {"", "nan", "None"})) or None
    d = pd.to_datetime(u["Inbetriebnahmedatum"])
    return {"mastr_units": len(u), "mastr_dist_km": np.nan, "ac_kw_mastr": num("Nettonennleistung"), "dc_kwp_mastr": num("Bruttoleistung"),
            "inv_kw_mastr": num("ZugeordneteWirkleistungWechselrichter"), "n_modules": num("AnzahlModule"), "year_mastr": int(d.min().year),
            "date_mastr": f"{d.min():%Y-%m-%d}", "orientation_class": txt("Hauptausrichtung"), "tilt_class": txt("HauptausrichtungNeigungswinkel"),
            "secondary_orientation": txt("Nebenausrichtung"), "area_ha": num("InAnspruchGenommeneFlaeche"), "power_limit": txt("Leistungsbegrenzung"),
            "uniform_flag": txt("EinheitlicheAusrichtungUndNeigungswinkel"), "mastr_src": "location:" + "+".join(locs)}


def mastr_facts(parks: pd.DataFrame, cand_csv: str = None, xml_json: str = None, max_km: float = 0.5, min_ac_kw: float = 1000.0,
                locations: dict = MASTR_LOCATIONS) -> pd.DataFrame:
    """Per park: MaStR units within max_km with AC >= min_ac_kw (empty rows where none); parks in `locations` take the
    units of their matched MaStR locations instead (coordinates unusable)."""
    cols = ["mastr_units", "mastr_dist_km", "ac_kw_mastr", "dc_kwp_mastr", "inv_kw_mastr", "n_modules", "year_mastr", "date_mastr",
            "orientation_class", "tilt_class", "secondary_orientation", "area_ha", "power_limit", "uniform_flag", "mastr_src"]
    out = pd.DataFrame(index=parks.index, columns=cols, dtype=object)
    for pv, locs in (locations or {}).items():
        if pv in out.index:
            for k, v in _facts_from_locations(locs).items():
                out.loc[pv, k] = v
    if not cand_csv or not os.path.exists(cand_csv):
        return out
    c = pd.read_csv(cand_csv, dtype={"EinheitMastrNummer": str})
    xml = json.load(open(xml_json)) if xml_json and os.path.exists(xml_json) else {}
    import xml.etree.ElementTree as ET
    katp = os.path.join(REPO, "data", "Marktstammdatenregister_20250625_25", "Katalogwerte.xml")
    kat = {int(k.find("Id").text): k.find("Wert").text for k in ET.parse(katp).getroot()} if os.path.exists(katp) else {}
    dec = lambda v: kat.get(int(float(v)), str(v)) if v not in (None, "", "nan") and pd.notna(v) else None
    for pv in parks.index:
        u = c[(c["pv"] == pv) & (c["dist_km"] <= max_km) & (pd.to_numeric(c["Nettonennleistung"], errors="coerce") >= min_ac_kw)]
        if u.empty or pv in (locations or {}):
            continue
        xs = [xml.get(i, {}) for i in u["EinheitMastrNummer"]]
        f = lambda k: sum(float(x.get(k, 0) or 0) for x in xs)
        out.loc[pv, "mastr_units"] = len(u)
        out.loc[pv, "mastr_dist_km"] = float(u["dist_km"].max())
        out.loc[pv, "ac_kw_mastr"] = float(pd.to_numeric(u["Nettonennleistung"]).sum())
        out.loc[pv, "dc_kwp_mastr"] = f("Bruttoleistung") or None
        out.loc[pv, "inv_kw_mastr"] = f("ZugeordneteWirkleistungWechselrichter") or None
        out.loc[pv, "n_modules"] = float(pd.to_numeric(u["AnzahlModule"], errors="coerce").sum())
        out.loc[pv, "year_mastr"] = int(pd.to_datetime(u["Inbetriebnahmedatum"], errors="coerce").dt.year.min())
        out.loc[pv, "date_mastr"] = f"{pd.to_datetime(u['Inbetriebnahmedatum'], errors='coerce').min():%Y-%m-%d}"
        out.loc[pv, "mastr_src"] = f"coordinate<={max_km}km"
        out.loc[pv, "orientation_class"] = dec(u["Hauptausrichtung"].iloc[0])
        out.loc[pv, "tilt_class"] = dec(u["HauptausrichtungNeigungswinkel"].iloc[0])
        out.loc[pv, "secondary_orientation"] = "/".join(filter(None, [dec(x.get("Nebenausrichtung")) for x in xs])) or None
        out.loc[pv, "area_ha"] = f("InAnspruchGenommeneFlaeche") or None
        out.loc[pv, "power_limit"] = "/".join(sorted({dec(x.get("Leistungsbegrenzung")) or "" for x in xs} - {""})) or None
        out.loc[pv, "uniform_flag"] = "/".join(sorted({str(x.get("EinheitlicheAusrichtungUndNeigungswinkel", "")) for x in xs} - {""})) or None
    return out


def ac_cap_from_meter(meter: pd.Series, cap_w: float) -> tuple:
    """AC limit estimate (W, source): a recurring flat-top level (>= 50 plateau hours within 3 % of
    one level, above 60 % of the series maximum) that the meter exceeds by > 3 % in fewer than 1 %
    of its daytime hours; otherwise the 99.9 % quantile. Recurring levels that are regularly
    exceeded are curtailment set-points, not the inverter/grid limit."""
    from pv import metrics
    s = meter.dropna()
    pl = metrics.plateaus(s, cap_w, lo=0.6)
    if not pl.empty:
        lev = pl["level_norm"].round(2)
        mode = lev.mode().iloc[0]
        hours = int(pl.loc[(lev - mode).abs() <= 0.03, "hours"].sum())
        day = s[s > 0.02 * cap_w]
        exceed = float((day > 1.03 * mode * cap_w).mean())
        if hours >= 50 and exceed < 0.01:
            return float(mode * cap_w), f"plateau_{hours}h"
    return float(s.quantile(0.999)), "p99.9"


def park_parameters(parks: pd.DataFrame, facts: pd.DataFrame, meter: pd.DataFrame, use_meter_ac: bool = True) -> pd.DataFrame:
    """Decide kwp_dc, dc_ac, year, module per park from master data + MaStR + meter plateau.
    use_meter_ac=False ignores the meter for the AC limit (register-only twins, D2 2026-10-20): DC/AC from MaStR, else 1.10."""
    rows = []
    for pv, p in parks.iterrows():
        f = facts.loc[pv]
        dc_mastr = f["dc_kwp_mastr"] if pd.notna(f["dc_kwp_mastr"]) and float(f["dc_kwp_mastr"]) > 0 else None
        ac_mastr = f["ac_kw_mastr"] if pd.notna(f["ac_kw_mastr"]) else None
        kwp = float(dc_mastr) if dc_mastr else float(p["kw_master"])
        ac_meter, ac_meter_src = (ac_cap_from_meter(meter[pv], kwp * 1000.0)) if (pv in meter and use_meter_ac) else (np.nan, None)
        ac_meter = ac_meter / 1000.0 if np.isfinite(ac_meter) else np.nan
        plateau = bool(ac_meter_src and ac_meter_src.startswith("plateau"))
        if dc_mastr and ac_mastr and not (plateau and ac_meter < 0.95 * float(ac_mastr)):
            dc_ac, dc_ac_src = kwp / float(ac_mastr), "mastr"
        elif np.isfinite(ac_meter) and ac_meter < 0.97 * kwp:
            dc_ac, dc_ac_src = kwp / ac_meter, f"meter_{ac_meter_src}"
        elif dc_mastr and ac_mastr:
            dc_ac, dc_ac_src = kwp / float(ac_mastr), "mastr"
        else:
            dc_ac, dc_ac_src = 1.10, "default_median"
        commissioning = None
        if pd.notna(f["year_mastr"]):
            year, year_src = int(f["year_mastr"]), "mastr"
            commissioning = f["date_mastr"] if pd.notna(f["date_mastr"]) else None
        elif pd.notna(p["year_hint"]):
            year, year_src = int(p["year_hint"]), "module_type"
        elif pv in meter and meter[pv].first_valid_index() is not None:
            year, year_src = int(meter[pv].first_valid_index().year) - 1, "meter_start_minus_1"
        else:
            year, year_src = 2014, "default"
        wp = (kwp * 1000.0 / float(f["n_modules"])) if pd.notna(f["n_modules"]) and float(f["n_modules"]) > 0 else np.nan
        wp_m2 = {"multi_si_early": 130, "multi_si": 150, "multi_perc": 165, "mono_perc": 195, "topcon_bifacial": 215}[modules.era_of(year)]
        if pd.notna(f["area_ha"]) and float(f["area_ha"]) > 0:
            kwp_ha, kwp_ha_src = kwp / float(f["area_ha"]), "mastr_area"
        else:
            kwp_ha, kwp_ha_src = float(np.interp(year, [2010, 2014, 2018, 2022, 2025], [450, 600, 850, 1150, 1250])), "year_table"
        gcr = float(np.clip(kwp_ha * 1000.0 / 10000.0 / wp_m2, 0.2, 0.75))
        rows.append({"pv": pv, "kwp_dc": kwp, "kwp_src": "mastr" if dc_mastr else "master_kw", "dc_ac": float(np.clip(dc_ac, 1.0, 1.6)),
                     "gcr": gcr, "kwp_per_ha": kwp_ha, "kwp_per_ha_src": kwp_ha_src,
                     "dc_ac_src": dc_ac_src, "ac_cap_meter_kw": ac_meter, "year": year, "year_src": year_src,
                     "commissioning_date": commissioning or f"{year}-07-01", "mastr_src": f["mastr_src"],
                     "module_lib": p["module_lib"], "wp_per_module": wp, "era": modules.era_of(year)})
    return pd.DataFrame(rows).set_index("pv")


def base_config(pv: str, p: pd.Series, par: pd.Series, arrays: list = None, variant: str = "C", albedo="era5") -> dict:
    """Variants: L legacy-like PVWatts (dc_ac 1, albedo 0.3, no IAM, no aging); P repaired PVWatts;
    C CEC + Sandia + aging + horizon; CR = C + inter-row shading (gcr from MaStR density); the
    caller may override the arrays (fitted angles)."""
    arr = arrays if arrays is not None else p["arrays"]
    rows = variant.endswith("R")
    variant = variant.rstrip("R") if variant not in ("L", "P", "C") else variant
    cfg = {"park_id": pv, "latitude": p["lat"], "longitude": p["lon"], "altitude": p["alt"], "kwp_dc": float(par["kwp_dc"]),
           "arrays": [dict(a, gcr=float(par["gcr"]) if rows else None) for a in arr], "random_seed": 42}
    if variant == "L":
        cfg.update({"dc_ac": 1.0, "module": {"model": "pvwatts", "gamma_pdc": -0.0035, "year": int(par["year"])},
                    "inverter": {"model": "pvwatts", "eta_inv_nom": 0.955}, "transposition": {"albedo": 0.3, "iam": None},
                    "aging": {"enabled": False}, "shading": {"horizon": False}})
    elif variant == "P":
        gamma = dict(zip(["multi_si_early", "multi_si", "multi_perc", "mono_perc", "topcon_bifacial"], [-0.0045, -0.0042, -0.0040, -0.0036, -0.0030]))[par["era"]]
        cfg.update({"dc_ac": float(par["dc_ac"]), "module": {"model": "pvwatts", "gamma_pdc": gamma, "year": int(par["year"])},
                    "inverter": {"model": "pvwatts", "eta_inv_nom": 0.97}, "transposition": {"albedo": albedo, "iam": "physical"},
                    "losses": {"dc": 0.02, "ac": 0.01, "availability": 1.0},
                    "aging": {"enabled": True, "commissioning_date": par.get("commissioning_date") or f"{int(par['year'])}-07-01", "era": par["era"]},
                    "shading": {"horizon": False}})
    else:
        cfg.update({"dc_ac": float(par["dc_ac"]), "module": {"model": "cec", "name": par["module_lib"], "year": int(par["year"])},
                    "inverter": {"model": "sandia", "unit_kw": None}, "transposition": {"albedo": albedo, "iam": "physical"},
                    "losses": {"dc": 0.02, "ac": 0.01, "availability": 1.0},
                    "aging": {"enabled": True, "commissioning_date": par.get("commissioning_date") or f"{int(par['year'])}-07-01", "era": par["era"]},
                    "shading": {"horizon": True, "inter_row": rows}})
    return pvconfig.normalise(cfg)


def build_drivers(p: pd.Series, start: str, end: str, conn, cams_dir: str, pv: str, n_stations: int = 1) -> tuple:
    """{'station': df, 'cams': df, 'era5': df} merged with ERA5 met on the hourly index, plus meta."""
    idx = timebase.hourly_index(start, end)
    met, meta = drivers.era5_met(conn, p["lat"], p["lon"], start, end)
    out, info = {}, dict(meta)
    master = pd.read_csv(STATIONS_MASTER, dtype={"station_id": str})
    ids = drivers.imputed_stations()
    sid, dist = drivers.nearest_station(p["lat"], p["lon"], master, ids)
    st = drivers.station_imputed(sid, start, end)
    out["station"] = drivers.merge_driver(st, met, idx)
    info.update({"station_id": sid, "station_dist_km": dist, "station_imputed_share": float(st["imputed_share"].mean()),
                 "station_missing_share": float(st["missing_share"].mean())})
    cams_files = [os.path.join(cams_dir, f) for f in os.listdir(cams_dir) if f.startswith(f"CAMS_rad_{pv}_") and f.endswith(".csv")]
    if cams_files:
        cams = drivers.cams_from_csv(cams_files, start, end)
        out["cams"] = drivers.merge_driver(cams, met, idx)
        info["cams_unreliable_share"] = float(cams["imputed_share"].mean())
    e5, m5 = drivers.era5_solar(conn, p["lat"], p["lon"], start, end)
    out["era5"] = drivers.merge_driver(e5, met, idx)
    info.update(m5)
    for k in out:
        if "albedo" not in out[k] or out[k]["albedo"].isna().all():
            out[k]["albedo"] = out["era5"]["albedo"]
        else:
            out[k]["albedo"] = out[k]["albedo"].fillna(out["era5"]["albedo"])
    return out, info


def load_horizon(path: str, pv: str) -> pd.Series:
    hz = json.load(open(path))[pv]
    s = pd.Series({float(k): float(v) for k, v in hz.items()}).sort_index()
    return s
