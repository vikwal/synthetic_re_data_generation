"""Authoritative real-park commissioning dates (HANDOFF ask 12).

Source = the user-researched dates in configs/real_wind_parks_era5/ (NOT
data/comm_dates.csv — that file holds the synthesized station-dataset dates
and disagreed by up to +-18 years for the seven 5-digit park ids).
Materialized to data/round2/park_commissioning.csv so figures, exports and
Table 4.1 share one source.
"""

import glob
import os

import pandas as pd
import yaml

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TABLE = os.path.join(REPO, "data", "round2", "park_commissioning.csv")
AGE_REF = "2023-12-01"


def build_table() -> pd.DataFrame:
    rows = []
    for f in sorted(glob.glob(os.path.join(
            REPO, "configs", "real_wind_parks_era5", "config_*.yaml"))):
        pid = os.path.basename(f)[len("config_"):-len(".yaml")]
        cd = yaml.safe_load(open(f))["params"]["commissioning_date"]
        age = (pd.Timestamp(AGE_REF) - pd.Timestamp(cd)).days / 365.25
        rows.append({"park_id": pid, "commissioning_date": cd,
                     "age_years_2023_12": round(age, 2),
                     "source": "park config (user-researched)"})
    df = pd.DataFrame(rows)
    df.to_csv(TABLE, index=False)
    return df


def commissioning_date(park_id: str) -> str:
    df = load()
    hit = df[df["park_id"] == str(park_id)]
    if len(hit):
        return hit["commissioning_date"].iloc[0]
    raise KeyError(f"park {park_id} not in {TABLE}")


def age_years(park_id: str) -> float:
    df = load()
    return float(df.set_index("park_id").loc[str(park_id), "age_years_2023_12"])


_cache = {}


def load() -> pd.DataFrame:
    if "t" not in _cache:
        if not os.path.exists(TABLE):
            build_table()
        _cache["t"] = pd.read_csv(TABLE, dtype={"park_id": str})
    return _cache["t"]
