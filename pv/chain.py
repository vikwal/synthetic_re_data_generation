"""Run one park config on one driver frame: solar position -> DNI -> POA per array (horizon,
inter-row, IAM) -> cell temperature -> DC (PVWatts or CEC) -> aging -> soiling -> DC losses ->
inverter (clipping) -> AC losses / availability."""

import numpy as np
import pandas as pd
import pvlib

from pv import aging, arrays as arr, inverter as inv, modules as mod, soiling, timebase, transposition

OUT_COLS = ["ghi", "dhi", "dni", "temp_air", "wind_speed", "albedo", "zenith", "azimuth_sun", "poa_global",
            "poa_eff", "t_cell", "p_dc_raw_w", "aging_factor", "soiling_ratio", "p_dc_w", "p_ac_w",
            "clipped", "availability", "imputed_share", "horizon_shaded"]


def _albedo(cfg: dict, drv: pd.DataFrame) -> pd.Series:
    a = cfg["transposition"]["albedo"]
    if a == "era5" and "albedo" in drv:
        return drv["albedo"].fillna(0.2).clip(0.05, 0.9)
    return pd.Series(float(a) if a != "era5" else 0.2, index=drv.index)


def prepare_sun(cfg: dict, drv: pd.DataFrame) -> tuple:
    """Solar position (interval centre), clear-sky DNI and closure DNI on the driver frame."""
    sp = timebase.solar_position(drv.index, cfg["latitude"], cfg["longitude"], cfg["altitude"],
                                 temp_air=drv["temp_air"].fillna(12.0).values if "temp_air" in drv else None)
    cs = timebase.clearsky(drv.index, cfg["latitude"], cfg["longitude"], cfg["altitude"])
    dni = transposition.dni_from_closure(drv["ghi"], drv["dhi"], sp["apparent_zenith"], clearsky_dni=cs["dni"])
    return sp, cs, dni


def run_park(cfg: dict, drv: pd.DataFrame, horizon: pd.Series = None, pm: pd.DataFrame = None,
             rain_mm: pd.Series = None, return_arrays: bool = False) -> pd.DataFrame:
    """drv: pv.drivers frame with ghi, dhi, temp_air, wind_speed (+ albedo, imputed_share)."""
    drv = drv.copy()
    idx = drv.index
    sp, cs, dni = prepare_sun(cfg, drv)
    albedo = _albedo(cfg, drv)
    dni_extra = pvlib.irradiance.get_extra_radiation(idx + timebase.CENTRE)
    dni_extra.index = idx
    hz = horizon if cfg["shading"]["horizon"] else None

    kwp = float(cfg["kwp_dc"])
    pdc0_w = kwp * 1000.0
    pac0_w = pdc0_w / float(cfg["dc_ac"])
    m = cfg["module"]
    mod_row = None
    if m["model"] == "cec":
        mod_row = mod.module_params(m["name"]) if m.get("name") else mod.era_proxy_module(int(m.get("year", 2015)))

    # aging (needed for the CEC current loss as well)
    if cfg["aging"]["enabled"]:
        era = cfg["aging"]["era"] or mod.era_of(int(m.get("year", 2015)))
        age_f = aging.degradation_series(idx, cfg["aging"]["commissioning_date"], era)
        mean_age = float(np.nanmean(aging.age_years(idx, cfg["aging"]["commissioning_date"])))
    else:
        era, age_f, mean_age = None, pd.Series(1.0, index=idx), 0.0

    poa_sum = pd.Series(0.0, index=idx)
    poa_eff_sum = pd.Series(0.0, index=idx)
    tcell_sum = pd.Series(0.0, index=idx)
    pdc = pd.Series(0.0, index=idx)
    shaded = pd.Series(0.0, index=idx)
    vmp = pd.Series(0.0, index=idx)
    per_array = {}
    for a in cfg["arrays"]:
        p = transposition.poa(drv["ghi"], drv["dhi"], dni, sp, a["tilt"], a["azimuth"], albedo, dni_extra,
                              horizon=hz, model=cfg["transposition"]["model"], iam_model=cfg["transposition"]["iam"])
        if cfg["shading"]["inter_row"] and a.get("gcr"):
            f = arr.inter_row_factors(sp, a["tilt"], a["azimuth"], a["gcr"])
            direct = p["poa_direct"] * (1.0 - f["direct_shaded_fraction"])
            diffuse = p["poa_diffuse"] * f["sky_diffuse_factor"]
            scale = (direct + diffuse) / p["poa_global"].replace(0, np.nan)
            p["poa_eff"] = (p["poa_eff"] * scale).fillna(0.0)
            p["poa_global"] = direct + diffuse
        tc = mod.cell_temperature(p["poa_global"], drv["temp_air"].fillna(12.0), drv["wind_speed"].fillna(2.0),
                                  model=cfg["temperature"]["model"], u0=cfg["temperature"]["u0"], u1=cfg["temperature"]["u1"])
        share_w = pdc0_w * float(a["share"])
        if m["model"] == "pvwatts":
            gamma = aging.aged_gamma(float(m["gamma_pdc"]), mean_age, era) if era else float(m["gamma_pdc"])
            dc_a = mod.dc_pvwatts(p["poa_eff"], tc, share_w, gamma)
        else:
            r_s = aging.aged_r_s_growth(mean_age, era) if era else 0.0
            out = mod.dc_cec(p["poa_eff"], tc, mod_row, share_w, r_s_growth=r_s)
            dc_a = out["p_mp"]
            vmp = vmp + out["v_mp"] * float(a["share"])
        poa_sum += p["poa_global"] * a["share"]
        poa_eff_sum += p["poa_eff"] * a["share"]
        tcell_sum += tc * a["share"]
        shaded = np.maximum(shaded, p["horizon_shaded"])
        pdc += dc_a
        if return_arrays:
            per_array[a["name"]] = pd.DataFrame({"poa_global": p["poa_global"], "poa_eff": p["poa_eff"], "t_cell": tc, "p_dc_w": dc_a})

    res = pd.DataFrame(index=idx)
    res["ghi"], res["dhi"], res["dni"] = drv["ghi"], drv["dhi"], dni
    res["temp_air"], res["wind_speed"], res["albedo"] = drv.get("temp_air"), drv.get("wind_speed"), albedo
    res["zenith"], res["azimuth_sun"] = sp["apparent_zenith"], sp["azimuth"]
    res["poa_global"], res["poa_eff"], res["t_cell"] = poa_sum, poa_eff_sum, tcell_sum
    res["p_dc_raw_w"] = pdc
    res["aging_factor"] = age_f
    pdc = pdc * age_f

    if cfg["soiling"]["enabled"] and pm is not None and rain_mm is not None:
        tilt_mean = float(np.average([a["tilt"] for a in cfg["arrays"]], weights=[a["share"] for a in cfg["arrays"]]))
        ratio = soiling.hsu_ratio(rain_mm.reindex(idx), pm["pm2_5"].reindex(idx), pm["pm10"].reindex(idx), tilt_mean,
                                  cleaning_threshold_mm=cfg["soiling"]["cleaning_threshold_mm"])
    else:
        ratio = pd.Series(1.0, index=idx)
    res["soiling_ratio"] = ratio
    pdc = pdc * ratio * (1.0 - float(cfg["losses"]["dc"]))
    res["p_dc_w"] = pdc

    iv = cfg["inverter"]
    if iv["model"] == "pvwatts":
        pac = inv.ac_pvwatts(pdc, pac0_w, eta_inv_nom=float(iv["eta_inv_nom"]), eta_inv_ref=float(iv.get("eta_inv_ref", 0.9637)))
    else:
        inv_row, n_units = (inv.inverter_params(iv["name"]), None) if iv.get("name") else inv.pick_inverter(pac0_w, int(m.get("year", 2015)), iv.get("unit_kw"))
        if n_units is None:
            n_units = max(1, int(round(pac0_w / float(inv_row["Paco"]))))
        scale_dc = (n_units * float(inv_row["Paco"])) / pac0_w            # n library units carry exactly the park AC rating
        if m["model"] == "cec":
            n_series = inv.strings_for(inv_row, float(mod_row["V_mp_ref"]))
            vdc = (vmp * n_series).where(vmp > 0, float(inv_row["Vdco"]))
        else:
            vdc = pd.Series(float(inv_row["Vdco"]), index=idx)
        pac = inv.ac_sandia(pdc, vdc, inv_row, n_units, scale_dc=scale_dc)
        pac = pac.clip(upper=pac0_w)
        res.attrs["inverter"] = {"name": inv_row["name"], "n_units": n_units, "paco_w": float(inv_row["Paco"])}
    res["clipped"] = (pdc * 0.97 > pac0_w).astype(float)
    pac = pac * (1.0 - float(cfg["losses"]["ac"]))
    avail = float(cfg["losses"]["availability"])
    res["availability"] = avail
    res["p_ac_w"] = (pac * avail).clip(lower=0.0)
    res["imputed_share"] = drv["imputed_share"] if "imputed_share" in drv else 0.0
    res["horizon_shaded"] = shaded
    res.attrs["module"] = None if mod_row is None else str(mod_row["name"])
    res.attrs["kwp_dc"] = kwp
    res.attrs["pac0_w"] = pac0_w
    if return_arrays:
        res.attrs["arrays"] = per_array
    return res[OUT_COLS]
