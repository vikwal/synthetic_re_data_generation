"""Timestamp conventions and resampling.

Every source labels its intervals differently (verified 2026-10-13 by correlating against the
Ineichen clear-sky curve, see FL_Contribution/reports/pv_generator_review.md):
  DWD 10 min raw / imputed 30 min / ERA5 / CAMS  -> label = interval END
  Trianel 15 min meter data                       -> label = interval START
The chain works on hourly means labelled by the interval START (UTC). Solar position is
evaluated at the interval centre (start + 30 min).
"""

import numpy as np
import pandas as pd
import pvlib

CENTRE = pd.Timedelta("30min")


def to_hourly_mean(s: pd.Series | pd.DataFrame, label: str, step: str = None) -> pd.Series | pd.DataFrame:
    """Hourly mean of an interval series; result labelled by interval start.

    label: 'start' | 'end' | 'centre' -- how the source labels its intervals.
    step:  source interval (e.g. '15min'); needed for 'centre' and inferred otherwise.
    """
    if label not in ("start", "end", "centre"):
        raise ValueError(f"label must be start|end|centre, got {label}")
    if label == "end":
        if step is None:
            step = pd.infer_freq(s.index[:50]) or _median_step(s.index)
        s = s.copy()
        s.index = s.index - pd.Timedelta(step)
    elif label == "centre":
        if step is None:
            raise ValueError("centre labels need the source step")
        s = s.copy()
        s.index = s.index - pd.Timedelta(step) / 2
    return s.resample("1h", closed="left", label="left").mean()


def _median_step(idx: pd.DatetimeIndex) -> pd.Timedelta:
    d = pd.Series(idx[1:] - idx[:-1])
    return d.median()


def hourly_index(start, end) -> pd.DatetimeIndex:
    """[start, end) hourly UTC index, interval-start labels."""
    return pd.date_range(pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC"),
                         freq="1h", inclusive="left")


def solar_position(index: pd.DatetimeIndex, lat: float, lon: float, alt: float = 0.0,
                   temp_air=None, pressure=None) -> pd.DataFrame:
    """Solar position at the interval centre, returned on the interval-start index."""
    sp = pvlib.solarposition.get_solarposition(index + CENTRE, lat, lon, altitude=alt,
                                               temperature=12.0 if temp_air is None else np.asarray(temp_air),
                                               pressure=pressure)
    sp.index = index
    return sp


def clearsky(index: pd.DatetimeIndex, lat: float, lon: float, alt: float = 0.0,
             model: str = "ineichen") -> pd.DataFrame:
    """Clear-sky GHI/DNI/DHI as hourly means: 10-min Ineichen curve averaged over the hour."""
    fine = pd.date_range(index[0], index[-1] + pd.Timedelta(minutes=50), freq="10min")
    loc = pvlib.location.Location(lat, lon, altitude=alt, tz="UTC")
    cs = loc.get_clearsky(fine, model=model)
    return cs.resample("1h", closed="left", label="left").mean().reindex(index)
