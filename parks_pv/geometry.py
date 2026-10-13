"""True sub-array geometry of a park and the register view of it.

Truth (pv_arrays.csv) is built from three layers:
 1. nominal mount orientation(s) from the register class and era (tilt ranges per class x cohort, azimuth
    = class centre + parcel-alignment noise); several nominal arrays for east-west, secondary orientation,
    mixed unit classes or the register's "not uniform" flag;
 2. a deliberate register deviation in ~15 % of parks (E4): the true main orientation is NOT the registered
    class (pv7 case: registered south, built east-west), to let step 3 measure what wrong master data cost;
 3. terrain: rows follow the ground inside the footprint (OSM plant polygon, else MaStR area as a disc) ->
    effective angles per 30 m cell -> k-means clusters (<= 4 arrays per park) with area shares.
The register view (pv_masterdata_view.csv) is what a register delivers: the MaStR classes (including the
deviating ones), the tilt as class and rounded to 5 deg, secondary orientation only where MaStR has it.
All randomness is seeded per park_id."""
import hashlib

import numpy as np
import pandas as pd

from pv import arrays as arr, terrain

AZ_CENTRE = {"Süd": 180.0, "Süd-West": 225.0, "Süd-Ost": 135.0, "Ost": 90.0, "West": 270.0, "Nord": 0.0,
             "Nord-Ost": 45.0, "Nord-West": 315.0}
# true mount tilt (deg) by register tilt class and cohort: uniform ranges
TILT_RANGE = {("20 - 40 Grad", "<=2012"): (25.0, 32.0), ("20 - 40 Grad", "2013-20"): (20.0, 28.0), ("20 - 40 Grad", "2021+"): (15.0, 25.0),
              ("< 20 Grad", "<=2012"): (15.0, 20.0), ("< 20 Grad", "2013-20"): (12.0, 19.0), ("< 20 Grad", "2021+"): (10.0, 18.0),
              ("40 - 60 Grad", "<=2012"): (40.0, 45.0), ("40 - 60 Grad", "2013-20"): (40.0, 45.0), ("40 - 60 Grad", "2021+"): (40.0, 45.0)}
EW_TILT = (10.0, 15.0)
AZ_SIGMA = 6.0               # parcel alignment noise around the class centre
DEVIATION_SHARE = 0.15       # parks whose true main orientation differs from the register class
MAX_ARRAYS = 4


def park_rng(park_id: str, seed: int = 42) -> np.random.Generator:
    h = int(hashlib.sha256(f"{seed}:{park_id}".encode()).hexdigest()[:12], 16)
    return np.random.default_rng(h)


def _tilt(rng, tilt_class: str, cohort: str) -> float:
    lo, hi = TILT_RANGE.get((tilt_class, cohort), TILT_RANGE[("20 - 40 Grad", cohort)])
    return float(rng.uniform(lo, hi))


def _azimuth(rng, cls: str) -> float:
    return float((AZ_CENTRE[cls] + np.clip(rng.normal(0.0, AZ_SIGMA), -20.0, 20.0)) % 360.0)


def nominal_arrays(row, rng) -> tuple:
    """(list of dicts {name, tilt, azimuth, share, source}, deviation note or None) for one selected park.
    row needs: orientation_class, tilt_class, secondary_orientation, secondary_tilt_class, uniform_flag,
    cohort, cls."""
    main, tilt_cls, coh = row.orientation_class, row.tilt_class, row.cohort
    deviation = None
    if rng.random() < DEVIATION_SHARE and main == "Süd":
        main = "Ost-West" if rng.random() < 0.4 else rng.choice(["Süd-West", "Süd-Ost"])
        deviation = f"registered Süd, built {main}"
    out = []
    if main == "Ost-West":
        t = float(rng.uniform(*EW_TILT))
        s = float(rng.uniform(0.45, 0.55))
        out += [{"name": "east", "tilt": t, "azimuth": _azimuth(rng, "Ost"), "share": s, "source": "east-west"},
                {"name": "west", "tilt": t, "azimuth": _azimuth(rng, "West"), "share": 1.0 - s, "source": "east-west"}]
    else:
        out.append({"name": "main", "tilt": _tilt(rng, tilt_cls, coh), "azimuth": _azimuth(rng, main), "share": 1.0, "source": "main class"})
    sec = [s for s in str(row.secondary_orientation or "").split("/") if s]
    if sec:
        s_tot = float(rng.uniform(0.15, 0.45))
        sec_tilt_cls = str(row.secondary_tilt_class or "").split("/")[0] or tilt_cls
        for a in out:
            a["share"] *= 1.0 - s_tot
        for j, sc in enumerate(sec[:2]):
            if sc == "Ost-West":
                t = float(rng.uniform(*EW_TILT))
                out += [{"name": f"sec{j}e", "tilt": t, "azimuth": _azimuth(rng, "Ost"), "share": s_tot / len(sec[:2]) / 2, "source": "secondary"},
                        {"name": f"sec{j}w", "tilt": t, "azimuth": _azimuth(rng, "West"), "share": s_tot / len(sec[:2]) / 2, "source": "secondary"}]
            elif sc in AZ_CENTRE:
                out.append({"name": f"sec{j}", "tilt": _tilt(rng, sec_tilt_cls, coh), "azimuth": _azimuth(rng, sc),
                            "share": s_tot / len(sec[:2]), "source": "secondary"})
    elif "0" in str(row.uniform_flag or "") or bool(getattr(row, "mixed_orientation", False)):
        s2 = float(rng.uniform(0.1, 0.3))
        for a in out:
            a["share"] *= 1.0 - s2
        az0 = out[0]["azimuth"]
        out.append({"name": "second", "tilt": _tilt(rng, tilt_cls, coh), "azimuth": float((az0 + rng.choice([-1, 1]) * rng.uniform(15, 45)) % 360.0),
                    "share": s2, "source": "non-uniform flag"})
    tot = sum(a["share"] for a in out)
    for a in out:
        a["share"] /= tot
    return out, deviation


def footprint(park, osm_elements: list, area_ha_fallback: float):
    """Shapely polygon: OSM solar plant near the centre, else a disc of the MaStR area."""
    poly = terrain.pick_plant_near(osm_elements, park.lat, park.lon, 2.5) if osm_elements else None
    if poly is not None and terrain.area_ha(poly) >= 0.3:
        return poly, "osm"
    return terrain.buffer_polygon(park.lat, park.lon, max(area_ha_fallback, 0.5)), "mastr_area" if pd.notna(park.area_ha) and park.area_ha > 0 else "density_buffer"


def terrain_arrays(poly, nominal: list, k_total: int = MAX_ARRAYS, seed: int = 0) -> tuple:
    """Rows follow the ground: effective angles per DEM cell for every nominal array, clustered into
    <= k_total arrays with shares = nominal share x area share. Returns (arrays, terrain stats)."""
    k_each = max(1, k_total // len(nominal))
    out, stats = [], {}
    for a in nominal:
        t = terrain.terrain_table(poly, a["tilt"], a["azimuth"])
        if t.empty:
            out.append(dict(a))
            continue
        cl = arr.cluster_arrays(t["tilt_eff"].values, t["az_eff"].values, t["area_m2"].values, k_each, seed=seed)
        for j, c in enumerate(cl):
            out.append({"name": f"{a['name']}_{j}", "tilt": c.tilt, "azimuth": c.azimuth, "share": a["share"] * c.share,
                        "source": a["source"], "mount_tilt": a["tilt"], "mount_azimuth": a["azimuth"]})
        if not stats:
            stats = {"n_cells": int(len(t)), "elev_min": float(t.elev.min()), "elev_max": float(t.elev.max()),
                     "slope_mean": float(t.slope.mean()), "slope_p90": float(t.slope.quantile(0.9)), "slope_max": float(t.slope.max()),
                     "tilt_eff_std": float(t.tilt_eff.std()), "az_eff_std": float(t.az_eff.std())}
    tot = sum(a["share"] for a in out)
    for a in out:
        a["share"] /= tot
    return out, stats


def orientation_class(azimuth: float) -> str:
    """Azimuth (deg from north) -> MaStR 8-class orientation."""
    names = ["Nord", "Nord-Ost", "Ost", "Süd-Ost", "Süd", "Süd-West", "West", "Nord-West"]
    return names[int(((azimuth + 22.5) % 360.0) // 45.0)]


def tilt_class(tilt: float) -> str:
    return "< 20 Grad" if tilt < 20 else "20 - 40 Grad" if tilt < 40 else "40 - 60 Grad" if tilt < 60 else "> 60 Grad"


def register_view(sel_row, arrays: list) -> dict:
    """What a register delivers for this park: MaStR fields as registered (classes, secondary orientation only
    where MaStR has it, power limit, counts) plus the true mean tilt rounded to 5 deg - the deviating
    register class is kept on purpose."""
    tilts = np.array([a["tilt"] for a in arrays])
    shares = np.array([a["share"] for a in arrays])
    t_mean = float((tilts * shares).sum())
    return {"park_id": sel_row.park_id, "capacity_kw_register": sel_row.ac_kw, "dc_kwp_register": sel_row.dc_kwp,
            "orientation_class": sel_row.orientation_class, "tilt_class": sel_row.tilt_class, "tilt_deg_rounded5": 5 * round(t_mean / 5),
            "secondary_orientation": sel_row.secondary_orientation if isinstance(sel_row.secondary_orientation, str) else "",
            "secondary_tilt_class": sel_row.secondary_tilt_class if isinstance(sel_row.secondary_tilt_class, str) else "",
            "uniform_flag": sel_row.uniform_flag if isinstance(sel_row.uniform_flag, str) else "",
            "power_limit": sel_row.power_limit if isinstance(sel_row.power_limit, str) else "",
            "commissioning_year": sel_row.year, "n_units": sel_row.n_units, "area_ha": sel_row.area_ha}
