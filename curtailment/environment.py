"""Environment layer: bat curtailment as a permit condition (report 6.4).

Per park one permit level (same practice for the whole park), stream
rng(seed, "bat", park_id); it applies to the turbine groups commissioned in or
after min_commissioning_year. Mask per quarter-hour and group:

  m = 1{local date in [start, end]}
      * 1{t >= sunset(d) - delta(month) or t < sunrise(d)}
      * 1{v_hub,g < v_max} * 1{T_hub,g > t_min}

Sunrise/sunset: pvlib Location.get_sun_rise_set_transit (SPA) at the park
centroid, once per local calendar day. T_hub,g from temp_2m with
generate_wind.get_temperature_at_height and the group hub height.
No precipitation criterion: the ERA5 table has no precipitation (conservative,
slightly higher losses than permits with a rain exception).
"""

import numpy as np
import pandas as pd

from curtailment import streams, timegrid

DEFAULT_TEMP_GRADIENT = 0.00649   # K/m, configs/config_wind.yaml params.temp_gradient


def draw_level(seed: int, park_id: str, levels: dict) -> str:
    """Permit level of a park; levels in config order with probabilities p."""
    u = streams.rng(seed, "bat", park_id).random()
    names = list(levels)
    cum = np.cumsum([levels[n]["p"] for n in names])
    return names[min(int(np.searchsorted(cum, u, side="right")), len(names) - 1)]


def sun_times(lat: float, lon: float, days: pd.DatetimeIndex) -> tuple:
    """Sunrise and sunset (UTC ns int64) for local calendar days (tz-naive dates)."""
    from pvlib.location import Location
    loc = Location(lat, lon, tz=timegrid.TZ_LOCAL)
    # SPA works on the UTC date of the stamp: local noon keeps it on the local day
    t = (days + pd.Timedelta(hours=12)).tz_localize(timegrid.TZ_LOCAL)
    st = loc.get_sun_rise_set_transit(t, method="spa")
    rise = pd.DatetimeIndex(st["sunrise"]).tz_convert("UTC").asi8
    sset = pd.DatetimeIndex(st["sunset"]).tz_convert("UTC").asi8
    return rise, sset


def _mmdd(s: str) -> int:
    m, d = s.split("-")
    return int(m) * 100 + int(d)


def _delta_by_month(spec: dict) -> np.ndarray:
    """before_sunset_h {'04-08': 1, '09-10': 3} or {'all': 0} -> hours per month 1..12 (index 0 unused)."""
    out = np.zeros(13)
    for k, h in spec.items():
        if k == "all":
            out[1:] = float(h)
        else:
            a, b = (int(x) for x in k.split("-"))
            out[a:b + 1] = float(h)
    return out


def night_season_mask(qidx: pd.DatetimeIndex, lat: float, lon: float, level: dict) -> np.ndarray:
    """1{date in season} * 1{night incl. the pre-sunset margin}, per quarter-hour."""
    days, inv = timegrid.day_codes(qidx)
    rise, sset = sun_times(lat, lon, days)
    dates = timegrid.local_dates(qidx)
    md = dates.month.to_numpy() * 100 + dates.day.to_numpy()
    season = (md >= _mmdd(level["start"])) & (md <= _mmdd(level["end"]))
    delta_ns = (_delta_by_month(level["before_sunset_h"])[dates.month.to_numpy()] * 3.6e12).astype(np.int64)
    t = qidx.asi8
    night = (t >= sset[inv] - delta_ns) | (t < rise[inv])
    return season & night


def hub_temperature_c(temp_2m_k, hub_height: float, params: dict = None) -> np.ndarray:
    import generate_wind as gw   # lazy: generate_wind imports this package
    p = {"temp_gradient": (params or {}).get("temp_gradient", DEFAULT_TEMP_GRADIENT)}
    t = gw.get_temperature_at_height(pd.DataFrame({"temp_2m": np.asarray(temp_2m_k, float)}), p, float(hub_height))
    return np.asarray(t, float) - 273.15


def bat_masks(qidx: pd.DatetimeIndex, frame: pd.DataFrame, park: dict, cfg: dict, params: dict = None) -> tuple:
    """Masks m_env (n_slots, n_groups) bool and the drawn level (None if no group is eligible).
    frame: hourly release frame (wind_speed_hub_t<g>, temp_2m); park: park meta with groups."""
    bat = cfg["environment"]["bat"]
    groups = park["groups"]
    m = np.zeros((len(qidx), len(groups)), bool)
    eligible = [int(str(g["commissioning_date"])[:4]) >= bat["min_commissioning_year"] for g in groups]
    if not any(eligible):
        return m, None
    name = draw_level(cfg["seed"], park["park_id"], bat["levels"])
    level = bat["levels"][name]
    base = night_season_mask(qidx, park["latitude"], park["longitude"], level)
    for k, (g, ok) in enumerate(zip(groups, eligible)):
        if not ok:
            continue
        v = timegrid.to_quarter(frame[g.get("wind_col", f"wind_speed_hub_{g['group_id']}")].to_numpy())
        t = timegrid.to_quarter(hub_temperature_c(frame["temp_2m"].to_numpy(), g["hub_height"], params))
        m[:, k] = base & (v < level["v_max"]) & (t > level["t_min_c"])
    return m, name
