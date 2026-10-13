"""Park parameters from the selection row (MaStR) and the chain config: PVWatts variant P of the Trianel
benchmark (era gamma, physical IAM, ERA5 albedo, era aging), plus horizon and inter-row shading and the
v1 loss level (E3: DC 5 %, AC 2 %, availability 0.98 ~ 0.93 of a loss-free chain)."""
import numpy as np

from pv import config as pvconfig, modules

GAMMA = {"multi_si_early": -0.0045, "multi_si": -0.0042, "multi_perc": -0.0040, "mono_perc": -0.0036, "topcon_bifacial": -0.0030}
WP_M2 = {"multi_si_early": 130, "multi_si": 150, "multi_perc": 165, "mono_perc": 195, "topcon_bifacial": 215}
DENSITY_YEARS, DENSITY_KWP_HA = [2010, 2014, 2018, 2022, 2025], [450, 600, 850, 1150, 1250]
LOSSES = {"dc": 0.05, "ac": 0.02, "availability": 0.98}


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
    gcr = float(np.clip(kwp_ha * 1000.0 / 10000.0 / WP_M2[era], 0.2, 0.75))
    return {"kwp_dc": kwp, "ac_kw": ac, "dc_ac": dc_ac, "year": year, "era": era, "gamma_pdc": GAMMA[era],
            "kwp_per_ha": kwp_ha, "kwp_per_ha_src": kwp_ha_src, "gcr": gcr,
            "power_limit": r.power_limit if isinstance(r.power_limit, str) else ""}


def park_config(park_id: str, lat: float, lon: float, alt: float, par: dict, arrays: list, inter_row: bool = True,
                horizon: bool = True, seed: int = 42) -> dict:
    cfg = {"park_id": park_id, "latitude": float(lat), "longitude": float(lon), "altitude": float(alt), "kwp_dc": par["kwp_dc"],
           "dc_ac": par["dc_ac"], "random_seed": seed,
           "arrays": [{"name": a["name"], "tilt": round(float(a["tilt"]), 3), "azimuth": round(float(a["azimuth"]), 3),
                       "share": round(float(a["share"]), 6), "gcr": par["gcr"] if inter_row else None} for a in arrays],
           "module": {"model": "pvwatts", "gamma_pdc": par["gamma_pdc"], "year": par["year"]},
           "inverter": {"model": "pvwatts", "eta_inv_nom": 0.97},
           "transposition": {"albedo": "era5", "iam": "physical"},
           "losses": dict(LOSSES),
           "aging": {"enabled": True, "commissioning_date": f"{par['year']}-07-01", "era": par["era"]},
           "shading": {"horizon": horizon, "inter_row": inter_row}}
    return pvconfig.normalise(cfg)
