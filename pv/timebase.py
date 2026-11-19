"""Timestamp conventions and resampling.

Every source labels its intervals differently (verified 2026-10-13 by correlating against the
Ineichen clear-sky curve, see FL_Contribution/reports/pv_generator_review.md):
  DWD 10 min raw / imputed 30 min / ERA5 / CAMS  -> label = interval END
  Trianel 15 min meter data                       -> label = interval START
The chain works on interval means labelled by the interval START (UTC) on a regular step (1 h for
the hourly release, 30 min for parks_pv_v1_30min). The step is read from the index (step_of);
solar position and extraterrestrial irradiance are evaluated at the interval centre.
"""

import numpy as np
import pandas as pd
import pvlib

DEFAULT_STEP = pd.Timedelta("1h")
FINE = pd.Timedelta("5min")            # clear-sky curve resolution before averaging over a step


def step_of(index: pd.DatetimeIndex) -> pd.Timedelta:
    """Regular step of an interval-start index (freq, else the median spacing; 1 h for a single stamp)."""
    if getattr(index, "freq", None) is not None:
        return pd.Timedelta(index.freq)
    if len(index) < 2:
        return DEFAULT_STEP
    return pd.Series(index[1:] - index[:-1]).median()


def centre(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Interval centres of an interval-start index."""
    return index + step_of(index) / 2


def to_step_mean(s: pd.Series | pd.DataFrame, label: str, src_step: str = None, step="1h") -> pd.Series | pd.DataFrame:
    """Mean of an interval series over `step`, result labelled by interval start.

    label:    'start' | 'end' | 'centre' -- how the source labels its intervals.
    src_step: source interval (e.g. '15min'); needed for 'centre' and inferred otherwise.
    """
    if label not in ("start", "end", "centre"):
        raise ValueError(f"label must be start|end|centre, got {label}")
    if label == "end":
        if src_step is None:
            src_step = pd.infer_freq(s.index[:50]) or _median_step(s.index)
        s = s.copy()
        s.index = s.index - pd.Timedelta(src_step)
    elif label == "centre":
        if src_step is None:
            raise ValueError("centre labels need the source step")
        s = s.copy()
        s.index = s.index - pd.Timedelta(src_step) / 2
    return s.resample(pd.Timedelta(step), closed="left", label="left").mean()


def to_hourly_mean(s: pd.Series | pd.DataFrame, label: str, step: str = None) -> pd.Series | pd.DataFrame:
    """Hourly mean of an interval series (`step` = source interval); see to_step_mean."""
    return to_step_mean(s, label, step, "1h")


def _median_step(idx: pd.DatetimeIndex) -> pd.Timedelta:
    d = pd.Series(idx[1:] - idx[:-1])
    return d.median()


def step_index(start, end, step="1h") -> pd.DatetimeIndex:
    """[start, end) UTC index with interval-start labels on a regular step."""
    return pd.date_range(pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC"),
                         freq=pd.Timedelta(step), inclusive="left")


def hourly_index(start, end) -> pd.DatetimeIndex:
    """[start, end) hourly UTC index, interval-start labels."""
    return step_index(start, end, "1h")


def solar_position(index: pd.DatetimeIndex, lat: float, lon: float, alt: float = 0.0,
                   temp_air=None, pressure=None) -> pd.DataFrame:
    """Solar position at the interval centre, returned on the interval-start index."""
    sp = pvlib.solarposition.get_solarposition(centre(index), lat, lon, altitude=alt,
                                               temperature=12.0 if temp_air is None else np.asarray(temp_air),
                                               pressure=pressure)
    sp.index = index
    return sp


def clearsky(index: pd.DatetimeIndex, lat: float, lon: float, alt: float = 0.0,
             model: str = "ineichen") -> pd.DataFrame:
    """Clear-sky GHI/DNI/DHI as interval means over the index step: 5-min Ineichen curve averaged."""
    step = step_of(index)
    fine = pd.date_range(index[0], index[-1] + step - FINE, freq=FINE)
    loc = pvlib.location.Location(lat, lon, altitude=alt, tz="UTC")
    cs = loc.get_clearsky(fine, model=model)
    return cs.resample(step, closed="left", label="left").mean().reindex(index)
