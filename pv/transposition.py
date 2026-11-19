"""DNI closure, plane-of-array irradiance, horizon shading, incidence-angle modifier."""

import numpy as np
import pandas as pd
import pvlib

ZENITH_DNI_ZERO = 88.0      # deg; below this elevation DNI from closure is unreliable


def dni_from_closure(ghi: pd.Series, dhi: pd.Series, zenith: pd.Series, clearsky_dni: pd.Series = None) -> pd.Series:
    """DNI = (GHI - DHI) / cos(z), limited to a clear-sky DNI (x 1.1) where given, zero near the horizon."""
    dni = pvlib.irradiance.dni(ghi, dhi, zenith, dni_clear=clearsky_dni, clearsky_tolerance=1.1,
                               zenith_threshold_for_zero_dni=ZENITH_DNI_ZERO,
                               zenith_threshold_for_clearsky_limit=80.0)
    return dni.fillna(0.0).clip(lower=0.0)


def horizon_elevation(horizon: pd.Series, solar_azimuth: pd.Series) -> np.ndarray:
    """Interpolate a horizon profile (index: azimuth deg 0..360, values: elevation deg) at the sun azimuth."""
    az = np.asarray(horizon.index, dtype=float) % 360.0
    el = np.asarray(horizon.values, dtype=float)
    order = np.argsort(az)
    az, el = az[order], el[order]
    az_ext = np.concatenate([az, [az[0] + 360.0]])
    el_ext = np.concatenate([el, [el[0]]])
    return np.interp(np.asarray(solar_azimuth, dtype=float) % 360.0, az_ext, el_ext)


def sky_view_factor(horizon: pd.Series) -> float:
    """Fraction of the isotropic sky dome seen by a horizontal receiver: 1 - mean(sin^2 h)."""
    el = np.radians(np.asarray(horizon.values, dtype=float))
    return float(1.0 - np.mean(np.sin(el) ** 2))


def poa(ghi, dhi, dni, solpos: pd.DataFrame, tilt: float, azimuth: float, albedo, dni_extra=None,
        horizon: pd.Series = None, model: str = "haydavies", iam_model: str = "physical") -> pd.DataFrame:
    """POA components for one (tilt, azimuth). Columns: poa_global, poa_direct, poa_diffuse,
    poa_eff (after IAM), aoi, horizon_shaded (0/1)."""
    idx = solpos.index
    zen, saz = solpos["apparent_zenith"], solpos["azimuth"]
    if dni_extra is None:
        dni_extra = pvlib.irradiance.get_extra_radiation(idx)
    dni_eff = pd.Series(np.asarray(dni, dtype=float), index=idx)
    shaded = pd.Series(0.0, index=idx)
    svf = 1.0
    if horizon is not None:
        hz = horizon_elevation(horizon, saz)
        elev = 90.0 - zen.values
        shaded = pd.Series(((elev < hz) & (elev > 0)).astype(float), index=idx)
        dni_eff = dni_eff.where(shaded == 0, 0.0)
        svf = sky_view_factor(horizon)
    irr = pvlib.irradiance.get_total_irradiance(tilt, azimuth, zen, saz, dni_eff, ghi, dhi,
                                                dni_extra=dni_extra, albedo=albedo, model=model)
    irr = irr.fillna(0.0)
    sky = irr["poa_sky_diffuse"] * svf
    ground = irr["poa_ground_diffuse"]
    direct = irr["poa_direct"]
    aoi = pvlib.irradiance.aoi(tilt, azimuth, zen, saz)
    if iam_model:
        iam_b = pvlib.iam.physical(aoi)
        iam_d = pvlib.iam.marion_diffuse(iam_model, tilt)
        eff = direct * iam_b + sky * iam_d["sky"] + ground * iam_d["ground"]
    else:
        eff = direct + sky + ground
    out = pd.DataFrame({"poa_global": direct + sky + ground, "poa_direct": direct,
                        "poa_diffuse": sky + ground, "poa_eff": eff, "aoi": aoi,
                        "horizon_shaded": shaded}, index=idx)
    out.loc[zen >= 90.0, ["poa_global", "poa_direct", "poa_diffuse", "poa_eff"]] = 0.0
    return out.clip(lower=0.0)
