"""Park parameters from the selection row (MaStR) and the chain config: PVWatts variant P of the Trianel
benchmark (era gamma, physical IAM, ERA5 albedo, era aging), plus horizon and inter-row shading and the
v1 loss level (E3: DC 5 %, AC 2 %, availability 0.98 ~ 0.93 of a loss-free chain)."""

import numpy as np
import pandas as pd

from parks_pv import paths
from pv import config as pvconfig, modules

GAMMA = {"multi_si_early": -0.0045, "multi_si": -0.0042, "multi_perc": -0.0040, "mono_perc": -0.0036, "topcon_bifacial": -0.0030}
WP_M2 = {"multi_si_early": 130, "multi_si": 150, "multi_perc": 165, "mono_perc": 195, "topcon_bifacial": 215}
DENSITY_YEARS, DENSITY_KWP_HA = [2010, 2014, 2018, 2022, 2025], [450, 600, 850, 1150, 1250]
LOSSES = {"dc": 0.05, "ac": 0.02, "availability": 0.98}
AGE_COHORTS = True       # 2026-11-05 (Viktor, on): arrays = geometry groups x age cohorts of the MaStR units; off = park-level era/aging
COHORT_GAP_YEARS = 3     # units of one era further apart than this in commissioning year form their own cohort


def has_units(r) -> bool:
    """True for a selection row with MaStR unit ids (spec parks without units carry their dates and cohorts)."""
    u = getattr(r, "unit_ids", None)
    return isinstance(u, str) and u.strip() not in ("", "nan")


def commissioning_dates(sel: pd.DataFrame, units_path: str = None) -> pd.DataFrame:
    """Per park (index park_id) the MaStR commissioning date of the first unit (`commissioning_first`, aging starts here)
    and of the last unit (`commissioning_full`: the park has its registered capacity, the release series starts here).
    Spec parks without units: the columns commissioning_first / commissioning_full of the row."""
    rows = []
    u = None
    for r in sel.itertuples():
        if not has_units(r):
            rows.append({"park_id": r.park_id, "commissioning_first": pd.Timestamp(r.commissioning_first).normalize(),
                         "commissioning_full": pd.Timestamp(r.commissioning_full).normalize()})
            continue
        if u is None:
            units_path = units_path or paths.mastr_file("units")
            u = pd.read_parquet(units_path, columns=["EinheitMastrNummer", "Inbetriebnahmedatum"]).set_index("EinheitMastrNummer")
        d = pd.to_datetime(u["Inbetriebnahmedatum"].reindex(str(r.unit_ids).split(";")))
        if d.isna().any():
            raise ValueError(f"{r.park_id}: units without commissioning date in {units_path}")
        rows.append({"park_id": r.park_id, "commissioning_first": d.min().normalize(), "commissioning_full": d.max().normalize()})
    return pd.DataFrame(rows).set_index("park_id")


def park_parameters(r) -> dict:
    """r: selection row. kwp from Bruttoleistung, AC from Nettonennleistung, year = first commissioning."""
    kwp = float(r.dc_kwp)
    ac = float(r.ac_kw) if np.isfinite(r.ac_kw) and r.ac_kw > 0 else kwp / 1.1
    dc_ac = float(np.clip(kwp / ac, 1.0, 1.6))
    year = int(r.year)
    era = modules.era_of(year)
    if np.isfinite(r.area_ha) and r.area_ha > 0:
        kwp_ha, kwp_ha_src = kwp / float(r.area_ha), "mastr_area"
    else:
        kwp_ha, kwp_ha_src = float(np.interp(year, DENSITY_YEARS, DENSITY_KWP_HA)), "year_table"
    gcr = gcr_of(kwp_ha, era)
    return {"kwp_dc": kwp, "ac_kw": ac, "dc_ac": dc_ac, "year": year, "era": era, "gamma_pdc": GAMMA[era],
            "kwp_per_ha": kwp_ha, "kwp_per_ha_src": kwp_ha_src, "gcr": gcr,
            "power_limit": r.power_limit if isinstance(r.power_limit, str) else ""}


def gcr_of(kwp_ha: float, era: str) -> float:
    """Ground coverage ratio from the capacity density and the module power per m2 of the era."""
    return float(np.clip(kwp_ha * 1000.0 / 10000.0 / WP_M2[era], 0.2, 0.75))


def age_cohorts(r, units_path: str = None) -> list:
    """Age cohorts of a park from its MaStR units (r: selection row with unit_ids): units sorted by commissioning date,
    a new cohort starts when the technology era changes or the year lies more than COHORT_GAP_YEARS after the first
    year of the current cohort. Per cohort: kWp share (Bruttoleistung), first commissioning date, era, era gamma and
    the gcr from the cohort units' MaStR area if every unit has one, else from the year table. One cohort = no split.
    Spec parks without units: the cohorts of the spec (column cohorts, JSON), one cohort each."""
    global _UNIT_TAB
    if not has_units(r):
        return spec_cohorts(r)
    units_path = units_path or paths.mastr_file("units")
    if "_UNIT_TAB" not in globals() or _UNIT_TAB[0] != units_path:
        u = pd.read_parquet(units_path, columns=["EinheitMastrNummer", "Inbetriebnahmedatum", "Bruttoleistung", "InAnspruchGenommeneFlaeche"])
        _UNIT_TAB = (units_path, u.set_index("EinheitMastrNummer"))
    u = _UNIT_TAB[1].reindex(str(r.unit_ids).split(";")).dropna(subset=["Inbetriebnahmedatum", "Bruttoleistung"])
    u = u.assign(date=pd.to_datetime(u.Inbetriebnahmedatum).dt.normalize()).sort_values("date")
    groups, cur = [], None
    for x in u.itertuples():
        y, era = x.date.year, modules.era_of(x.date.year)
        if cur is None or era != cur["era"] or y - cur["year"] > COHORT_GAP_YEARS:
            cur = {"year": y, "era": era, "rows": []}
            groups.append(cur)
        cur["rows"].append(x)
    tot = float(u.Bruttoleistung.sum())
    out = []
    for g in groups:
        kwp = float(sum(x.Bruttoleistung for x in g["rows"]))
        area = [x.InAnspruchGenommeneFlaeche for x in g["rows"]]
        if all(np.isfinite(a) and a > 0 for a in area):
            kwp_ha = kwp / float(sum(area))
        else:
            kwp_ha = float(np.interp(g["year"], DENSITY_YEARS, DENSITY_KWP_HA))
        out.append({"year": g["year"], "commissioning_date": f"{g['rows'][0].date:%Y-%m-%d}", "era": g["era"], "kwp": kwp,
                    "share": kwp / tot, "gamma_pdc": GAMMA[g["era"]], "gcr": gcr_of(kwp_ha, g["era"]), "n_units": len(g["rows"])})
    return out


def spec_cohorts(r) -> list:
    """Age cohorts of a spec park (parks_pv.park_spec): [{commissioning, dc_kwp, area_ha}] in the column cohorts."""
    import json
    cs = json.loads(r.cohorts) if isinstance(getattr(r, "cohorts", None), str) and r.cohorts else []
    tot = float(sum(c["dc_kwp"] for c in cs)) or 1.0
    out = []
    for c in cs:
        y = pd.Timestamp(c["commissioning"]).year
        era = modules.era_of(y)
        kwp_ha = c["dc_kwp"] / c["area_ha"] if c.get("area_ha") else float(np.interp(y, DENSITY_YEARS, DENSITY_KWP_HA))
        out.append({"year": y, "commissioning_date": f"{pd.Timestamp(c['commissioning']):%Y-%m-%d}", "era": era, "kwp": float(c["dc_kwp"]),
                    "share": float(c["dc_kwp"]) / tot, "gamma_pdc": GAMMA[era], "gcr": gcr_of(kwp_ha, era), "n_units": 1})
    return out


def cohort_arrays(arrays: list, cohorts: list) -> list:
    """Geometry groups x age cohorts, proportional split: every geometry array is divided by the cohort kWp shares
    (the unit coordinates are often too coarse to place a cohort in one geometry group; a spatial assignment would
    replace this split where they are not). Names <array>_<cohort year>."""
    return [dict(a, name=f"{a['name']}_{c['year']}", share=float(a["share"]) * c["share"], gamma_pdc=c["gamma_pdc"], era=c["era"],
                 commissioning_date=c["commissioning_date"], gcr=c["gcr"]) for a in arrays for c in cohorts]


def park_config(park_id: str, lat: float, lon: float, alt: float, par: dict, arrays: list, inter_row: bool = True,
                horizon: bool = True, seed: int = 42, cohorts: list = None) -> dict:
    """cohorts (age_cohorts, with AGE_COHORTS): more than one -> the arrays are split by cohort and carry the cohort
    era, gamma, commissioning date and gcr; None or one cohort -> park-level values as before."""
    split = bool(cohorts) and len(cohorts) > 1
    if split:
        arrays = cohort_arrays(arrays, cohorts)
    cfg = {"park_id": park_id, "latitude": float(lat), "longitude": float(lon), "altitude": float(alt), "kwp_dc": par["kwp_dc"],
           "dc_ac": par["dc_ac"], "random_seed": seed,
           "arrays": [{"name": a["name"], "tilt": round(float(a["tilt"]), 3), "azimuth": round(float(a["azimuth"]), 3),
                       "share": round(float(a["share"]), 6), "gcr": (a["gcr"] if split else par["gcr"]) if inter_row else None,
                       **({k: a[k] for k in ("gamma_pdc", "era", "commissioning_date")} if split else {})} for a in arrays],
           "module": {"model": "pvwatts", "gamma_pdc": par["gamma_pdc"], "year": par["year"]},
           "inverter": {"model": "pvwatts", "eta_inv_nom": 0.97},
           "transposition": {"albedo": "era5", "iam": "physical"},
           "losses": dict(LOSSES),
           "aging": {"enabled": True, "commissioning_date": par.get("commissioning_first") or f"{par['year']}-07-01", "era": par["era"]},
           "shading": {"horizon": horizon, "inter_row": inter_row}}
    return pvconfig.normalise(cfg)
