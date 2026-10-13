"""Snow cover losses. Driver: ERA5 single-level snow fields (snowfall sf, snow depth sd, snow density
rsn, 2 m temperature) read from monthly GRIB files; model: Marion et al. (2013) as implemented in
pvlib.snow (coverage_nrel, dc_loss_nrel), optionally with a snow-depth gate from ERA5."""
import glob
import os

import numpy as np
import pandas as pd

ACCUMULATED = {"sf", "tp"}            # [t-1h, t] sums from the 06/18 UTC runs, step 1..12
INSTANT = {"sd", "rsn", "2t"}


def era5_snow_point(grib_dir: str, lat: float, lon: float) -> pd.DataFrame:
    """Hourly series at the nearest ERA5 grid point: snowfall_mm_we (per hour), snow_depth_m (physical
    depth = sd * 1000 / rsn), snow_density, temp_air (deg C), precip_mm. Index UTC, interval END for
    accumulated fields (as public.era5_solar), instant for the others."""
    import pygrib
    rows = {}
    files = sorted(glob.glob(os.path.join(grib_dir, "*.grib")))
    if not files:
        raise FileNotFoundError(grib_dir)
    ij = None
    for f in files:
        g = pygrib.open(f)
        for m in g:
            if ij is None:
                lats, lons = m.latlons()
                d2 = (lats - lat) ** 2 + ((lons - lon) * np.cos(np.radians(lat))) ** 2
                ij = np.unravel_index(np.argmin(d2), d2.shape)
            t = pd.Timestamp(m.validDate, tz="UTC")
            rows.setdefault(t, {})[m.shortName] = float(m.values[ij])
        g.close()
    df = pd.DataFrame.from_dict(rows, orient="index").sort_index()
    out = pd.DataFrame(index=df.index)
    out["snowfall_mm_we"] = df["sf"] * 1000.0
    out["precip_mm"] = df["tp"] * 1000.0
    out["snow_density"] = df["rsn"]
    out["snow_depth_m"] = df["sd"] * 1000.0 / df["rsn"].clip(lower=50.0)
    out["snow_depth_we_mm"] = df["sd"] * 1000.0
    out["temp_air"] = df["2t"] - 273.15
    return out


def snowfall_cm(snowfall_mm_we: pd.Series, fresh_density: float = 100.0) -> pd.Series:
    """Water equivalent -> fresh snow height (cm/h) with a fresh-snow density (kg/m3)."""
    return snowfall_mm_we * 1000.0 / fresh_density / 10.0


def coverage(snowfall_cm_h: pd.Series, poa: pd.Series, temp_air: pd.Series, tilt: float, snow_depth_cm: pd.Series = None,
             threshold_snowfall: float = 1.0, threshold_depth: float = 1.0, can_slide_coefficient: float = -80.0,
             slide_amount_coefficient: float = 0.197) -> pd.Series:
    """Fraction of the array covered (Marion model). Hourly series on a common index."""
    from pvlib import snow as pvsnow
    sf = snowfall_cm_h.fillna(0.0)
    cov = pvsnow.coverage_nrel(sf, poa.fillna(0.0), temp_air.ffill().bfill(), tilt,
                               snow_depth=None if snow_depth_cm is None else snow_depth_cm.fillna(0.0),
                               threshold_snowfall=threshold_snowfall, threshold_depth=threshold_depth,
                               can_slide_coefficient=can_slide_coefficient, slide_amount_coefficient=slide_amount_coefficient)
    return cov.clip(0.0, 1.0)


def depth_gate(coverage_series: pd.Series, snow_depth_cm: pd.Series, min_depth_cm: float = 0.5) -> pd.Series:
    """Zero coverage where the reanalysis says there is no snow on the ground at all - the Marion
    sliding rule knows no melt, a depth gate ends the event when the snow pack is gone."""
    gate = snow_depth_cm.reindex(coverage_series.index).ffill().bfill() >= min_depth_cm
    return coverage_series.where(gate, 0.0)


def loss_factor(coverage_series: pd.Series, num_strings: int = 100) -> pd.Series:
    """Power multiplier 1 - loss. With many strings the NREL step loss tends to the coverage itself."""
    from pvlib import snow as pvsnow
    return 1.0 - pvsnow.dc_loss_nrel(coverage_series, num_strings)
