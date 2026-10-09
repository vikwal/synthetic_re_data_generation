"""15-minute grid of the curtailment model.

Hourly inputs are piecewise constant on the four quarter-hours of the hour;
outputs are hourly means of the four quarter-hours. Calendar days for the grid
disturbance term change at 00:00 Europe/Berlin. Slot numbers are absolute
(unix seconds // 900), so any period maps to the same slot ids.
"""

import numpy as np
import pandas as pd

STEP = pd.Timedelta(minutes=15)
STEP_H = 0.25
PER_HOUR = 4
TZ_LOCAL = "Europe/Berlin"


def quarter_index(hourly: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """The four quarter-hours of every hour (UTC); hourly index must be regular."""
    if hourly.tz is None:
        raise ValueError("hourly index must be tz-aware (UTC)")
    if len(hourly) > 1 and not (np.diff(hourly.asi8) == 3_600_000_000_000).all():
        raise ValueError("hourly index must be regular and gap-free")
    return pd.date_range(hourly[0], periods=len(hourly) * PER_HOUR, freq=STEP, tz=hourly.tz)


def period_index(start, end) -> pd.DatetimeIndex:
    """Quarter-hours of the hours [start, end] (end = last hour start, UTC)."""
    hours = pd.date_range(pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC"), freq="h")
    return quarter_index(hours)


def to_quarter(values) -> np.ndarray:
    """Hourly array (n,) or (n, k) -> quarter-hour array, piecewise constant."""
    return np.repeat(np.asarray(values), PER_HOUR, axis=0)


def to_hour(values) -> np.ndarray:
    """Quarter-hour array (4n,) or (4n, k) -> hourly mean."""
    v = np.asarray(values, dtype=float)
    return v.reshape((-1, PER_HOUR) + v.shape[1:]).mean(axis=1)


def hour_any(values) -> np.ndarray:
    v = np.asarray(values)
    return v.reshape((-1, PER_HOUR) + v.shape[1:]).any(axis=1)


def hour_max(values) -> np.ndarray:
    v = np.asarray(values)
    return v.reshape((-1, PER_HOUR) + v.shape[1:]).max(axis=1)


def slot_numbers(qidx: pd.DatetimeIndex) -> np.ndarray:
    return (qidx.asi8 // 900_000_000_000).astype(np.int64)


def local_dates(qidx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Local (Europe/Berlin) calendar date of every slot, tz-naive midnight."""
    return qidx.tz_convert(TZ_LOCAL).tz_localize(None).normalize()


def day_codes(qidx: pd.DatetimeIndex) -> tuple:
    """(unique local days, index of each slot's day in that array)."""
    days = local_dates(qidx)
    uniq, inv = np.unique(days.values, return_inverse=True)
    return pd.DatetimeIndex(uniq), inv


def utc_years(qidx: pd.DatetimeIndex) -> np.ndarray:
    return qidx.year.to_numpy()
