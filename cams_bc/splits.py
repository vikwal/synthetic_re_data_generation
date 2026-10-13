"""Station splits: a held-out test set and a 3-fold cross-validation over DISJOINT station groups.

Stratified over a lat/lon grid so that every fold and the test set cover north/south and east/west
alike; deterministic through a seeded permutation. The table is stored in data/cams_bc_splits.csv
and read back by every script, so no two scripts can disagree about which station is a test station.
"""
import numpy as np
import pandas as pd

from .config import N_FOLDS, SEED, SPLITS, STATIONS_LIST, TEST_FRACTION


def station_table(ids: list) -> pd.DataFrame:
    st = pd.read_csv(STATIONS_LIST, dtype={"Stations_id": str})
    st["Stations_id"] = st["Stations_id"].str.zfill(5)
    st = st.drop_duplicates("Stations_id").set_index("Stations_id")
    return pd.DataFrame({"station_id": ids, "lat": [float(st.loc[s, "geoBreite"]) for s in ids],
                         "lon": [float(st.loc[s, "geoLaenge"]) for s in ids],
                         "elev": [float(st.loc[s, "Stationshoehe"]) for s in ids]}).set_index("station_id")


def make_splits(ids: list, seed: int = SEED, test_fraction: float = TEST_FRACTION, n_folds: int = N_FOLDS) -> pd.DataFrame:
    t = station_table(ids)
    t["cell"] = pd.qcut(t["lat"], 3, labels=False).astype(str) + "_" + pd.qcut(t["lon"], 2, labels=False).astype(str)
    rng = np.random.default_rng(seed)
    t["role"], t["fold"] = "train", -1
    for cell, grp in t.groupby("cell"):
        order = grp.index.to_numpy()[rng.permutation(len(grp))]
        n_test = int(round(test_fraction * len(order)))
        t.loc[order[:n_test], "role"] = "test"
        rest = order[n_test:]
        start = rng.integers(n_folds)
        for i, s in enumerate(rest):
            t.loc[s, "fold"] = int((i + start) % n_folds)
    return t.reset_index()


def load_splits() -> pd.DataFrame:
    return pd.read_csv(SPLITS, dtype={"station_id": str})


def train_ids(sp: pd.DataFrame) -> list:
    return sorted(sp.loc[sp.role == "train", "station_id"])


def test_ids(sp: pd.DataFrame) -> list:
    return sorted(sp.loc[sp.role == "test", "station_id"])


def fold_ids(sp: pd.DataFrame, k: int) -> tuple:
    tr = sp[(sp.role == "train") & (sp.fold != k)]
    va = sp[(sp.role == "train") & (sp.fold == k)]
    return sorted(tr.station_id), sorted(va.station_id)
