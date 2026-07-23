"""WP1 — Trianel benchmark meter data access.

Meter data: 15-min park power in kW per Malo-ID (UTC). The Malo-ID <->
config-park_id mapping is data/round2/park_mapping.csv (wp0_park_layouts.py).
All loaders return power in WATT to match the synthetic chain.
"""

import os

import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
METER_PATH = os.path.join(REPO, "data", "Trianel_Benchmark", "meterdata_wind_20240601.csv")
MAPPING_PATH = os.path.join(REPO, "data", "round2", "park_mapping.csv")
LAYOUTS_PATH = os.path.join(REPO, "data", "round2", "park_layouts.csv")

_cache = {}


def load_mapping() -> pd.DataFrame:
    m = pd.read_csv(MAPPING_PATH, dtype={"park_id": str, "malo_id": str})
    assert len(m) == 13, f"expected 13 mapped parks, got {len(m)}"
    return m


def load_layouts() -> pd.DataFrame:
    return pd.read_csv(LAYOUTS_PATH, dtype={"park_id": str, "malo_id": str})


def rated_power_w(park_id: str) -> float:
    lay = load_layouts()
    kw = lay.loc[lay["park_id"] == str(park_id), "rated_kw"].sum()
    return float(kw) * 1000.0


def _meterdata() -> pd.DataFrame:
    if "meter" not in _cache:
        df = pd.read_csv(METER_PATH, sep=";")
        df["Zeit"] = pd.to_datetime(df["Zeit"], utc=True)
        df.set_index("Zeit", inplace=True)
        _cache["meter"] = df
    return _cache["meter"]


def load_park_power(park_id: str, window=None, freq: str = "1h") -> pd.Series:
    """Measured park power (W), resampled to `freq` by mean, clipped to window.

    An hour is kept only if all four 15-min values are present.
    """
    mapping = load_mapping()
    row = mapping.loc[mapping["park_id"] == str(park_id)]
    assert len(row) == 1, f"park {park_id} not in mapping"
    malo = row["malo_id"].iloc[0]
    s = _meterdata()[malo].astype(float) * 1000.0  # kW -> W
    grouper = s.resample(freq)
    out = grouper.mean().where(grouper.count() >= 4 if freq == "1h" else 1)
    if window is not None:
        out = out.loc[window[0]:window[1]]
    return out.rename(f"meas_{park_id}")


def effective_window(park_id: str, window) -> tuple:
    """Actual data coverage of the meter series inside the requested window."""
    s = load_park_power(park_id, window)
    valid = s.dropna()
    if valid.empty:
        return None, None, 0.0
    return str(valid.index.min()), str(valid.index.max()), float(s.notna().mean())
