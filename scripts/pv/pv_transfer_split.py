"""Station split for the PV transfer study (model applied to unseen sites), 2026-10-18.

90 synthetic parks (parks_pv_v1_30min) -> 30 test parks + 60 training parks; the 60 are dealt into
3 disjoint folds of 20 for the HPO (fixed time split: train 2023-07..2024-07, validate 2024-08..2025-07;
test year 2025-08..2026-07 reserved for the 30 test parks).

Method (cams_bc.splits / utils.pv_parks.station_folds, extended by a class stratum; Viktor 2026-10-18:
fully random, no fixed holdout parks, sets and folds must not be correlated by proximity or park homogeneity):
  * 30 test parks drawn from all 90, stratified over lat/lon quantile cells (3 x 2, as in the CAMS
    correction model) x PV class (C1..C5), largest-remainder rounding -> every set covers all regions and classes;
  * the 60 training parks are shuffled class-first within the same strata and dealt round-robin to 3 folds with
    one running counter (classes exactly balanced, cells approximately);
  * near-duplicate sites (pairs closer than NEAR_KM, 7 pairs < 10 km, min 3.8 km) must share role and fold,
    otherwise an "unseen" park would have a training neighbour with practically the same weather. Enforced by
    searching seeds: the first N_SEEDS seeds are drawn, the valid ones scored by balance (cells x role/fold) and
    test->train distance, the best one is written.
Deterministic. Output: data/pv_transfer_split_v1.csv (+ chosen seed on stdout).
"""
import argparse
import os

import numpy as np
import pandas as pd
from pyproj import Geod

ROOT = os.environ.get("DATA_ROOT", "/mnt/lambda1/nvme1")
RELEASE = f"{ROOT}/synthetic/solar/parks_pv_v1_30min"
SELECTION = os.path.expanduser("~/Work/FL_Contribution/data/mastr/pv_park_selection_v1.csv")
OUT = os.path.expanduser("~/Work/FL_Contribution/data/pv_transfer_split_v1.csv")
NEAR_KM = 10.0
N_SEEDS = 500


def park_table() -> pd.DataFrame:
    p = pd.read_csv(f"{RELEASE}/parks.csv").set_index("park_id")
    sel = pd.read_csv(SELECTION).set_index("park_id")
    t = p[["latitude", "longitude", "client_id", "region", "size_bin", "terrain_cls", "cohort"]].rename(
        columns={"latitude": "lat", "longitude": "lon"})
    t["cls"] = sel.loc[t.index, "cls"]
    t["cell"] = (pd.qcut(t.lat, 3, labels=False).astype(str) + "_" + pd.qcut(t.lon, 2, labels=False).astype(str))
    t["stratum"] = t.cell + "|" + t.cls
    return t


def largest_remainder(counts: pd.Series, total: int) -> pd.Series:
    """Integer allocation of `total` over strata proportional to `counts`."""
    raw = counts / counts.sum() * total
    base = np.floor(raw).astype(int)
    rest = int(total - base.sum())
    # NOTE: default (unstable) sort kept on purpose: the frozen split_v1 (seed 256) depends on its tie
    # order; kind='stable' changes the result. Reproduce with pandas 2.3.3 / numpy 1.26.4 (frcst/).
    order = (raw - base).sort_values(ascending=False).index[:rest]
    base.loc[order] += 1
    return base


def units_of(t: pd.DataFrame, dist: pd.DataFrame, km: float = NEAR_KM) -> list:
    """Connected components of the 'closer than km' graph: near-duplicate sites move as one unit."""
    ids = list(t.index)
    parent = {i: i for i in ids}

    def root(i):
        while parent[i] != i:
            i = parent[i]
        return i

    for a, b in near_pairs(dist, km):
        parent[root(a)] = root(b)
    comp = {}
    for i in ids:
        comp.setdefault(root(i), []).append(i)
    return list(comp.values())


def make_split(t: pd.DataFrame, seed: int, units: list, n_test: int = 30, n_folds: int = 3) -> pd.DataFrame:
    """Stratified draw at unit level (unit stratum = stratum of its first park; a near pair spanning two cells or
    classes is counted against the first member's quota, which only blurs the stratification, sizes are exact). Greedy per stratum up to the
    park quota, shortfall filled with singleton units; folds dealt class-first with a running counter that
    advances by unit size. Exact sizes are not guaranteed for every seed -> see search()."""
    rng = np.random.default_rng(seed)
    t = t.copy()
    t["role"], t["fold"] = "train", -1
    u = pd.DataFrame({"members": units, "stratum": [t.loc[m[0], "stratum"] for m in units],
                      "cls": [t.loc[m[0], "cls"] for m in units], "cell": [t.loc[m[0], "cell"] for m in units]})
    u["size"] = u.members.str.len()
    u = u.iloc[rng.permutation(len(u))].reset_index(drop=True)
    quota = largest_remainder(t.groupby("stratum").size(), n_test)
    u["role"] = "train"
    for stratum, q in quota.items():
        got = 0
        for i in u.index[u.stratum == stratum]:
            if got + u.at[i, "size"] <= q:
                u.at[i, "role"] = "test"
                got += u.at[i, "size"]
    short = n_test - int(u.loc[u.role == "test", "size"].sum())
    if short > 0:
        deficit = (quota - u[u.role == "test"].groupby("stratum")["size"].sum().reindex(quota.index, fill_value=0))
        cand = u[(u.role == "train") & (u["size"] == 1)].copy()
        cand["deficit"] = cand.stratum.map(deficit)
        cand = cand.sort_values("deficit", ascending=False, kind="stable")
        u.loc[cand.index[:short], "role"] = "test"
    for i in u.index[u.role == "test"]:
        t.loc[u.at[i, "members"], "role"] = "test"
    k = int(rng.integers(n_folds))
    tr_units = u[u.role == "train"].sort_values(["cls", "cell"], kind="stable")
    for i in tr_units.index:
        t.loc[u.at[i, "members"], "fold"] = k % n_folds
        k += u.at[i, "size"]
    return t.reset_index()


def distances(t: pd.DataFrame) -> pd.DataFrame:
    """Geodesic km between all parks (index order of t), diagonal inf."""
    g = Geod(ellps="WGS84")
    lat, lon = t.lat.values, t.lon.values
    d = np.array([g.inv(np.full(len(t), lon[i]), np.full(len(t), lat[i]), lon, lat)[2] / 1000 for i in range(len(t))])
    np.fill_diagonal(d, np.inf)
    return pd.DataFrame(d, index=t.index, columns=t.index)


def near_pairs(dist: pd.DataFrame, km: float = NEAR_KM) -> list:
    iu = np.triu_indices(len(dist), 1)
    m = dist.values[iu] < km
    return [(dist.index[i], dist.index[j]) for i, j in zip(iu[0][m], iu[1][m])]


def valid(t: pd.DataFrame, pairs: list) -> bool:
    """Near-duplicate pairs share role (and fold when both train)."""
    r = t.set_index("park_id") if "park_id" in t.columns else t
    return all(r.loc[a, "role"] == r.loc[b, "role"] and r.loc[a, "fold"] == r.loc[b, "fold"] for a, b in pairs)


def balance_score(t: pd.DataFrame) -> float:
    """Sum of squared deviations of the cell counts (test vs expected 1/3; folds vs expected 1/3 of train)."""
    cells = t.cell.value_counts()
    te = t[t.role == "test"].cell.value_counts().reindex(cells.index, fill_value=0)
    sc = float(((te - cells / 3) ** 2).sum())
    tr = t[t.role == "train"]
    for k in sorted(tr.fold.unique()):
        fk = tr[tr.fold == k].cell.value_counts().reindex(cells.index, fill_value=0)
        sc += float(((fk - tr.cell.value_counts().reindex(cells.index, fill_value=0) / 3) ** 2).sum())
    return sc


def nn_test_train(t: pd.DataFrame, dist: pd.DataFrame) -> np.ndarray:
    r = t.set_index("park_id") if "park_id" in t.columns else t
    te, tr = r.index[r.role == "test"], r.index[r.role == "train"]
    return dist.loc[te, tr].min(axis=1).values


def search(t: pd.DataFrame, n_seeds: int = N_SEEDS, n_test: int = 30, n_folds: int = 3) -> tuple:
    """Best seed with exact set sizes (30 test, 20/20/20): lowest cell-balance score, ties by larger median
    test->train distance. Near pairs are units, so the constraint holds by construction."""
    dist = distances(t)
    pairs = near_pairs(dist)
    units = units_of(t, dist)
    best = None
    n_tr = (len(t) - n_test) // n_folds
    for seed in range(n_seeds):
        s = make_split(t, seed, units, n_test=n_test, n_folds=n_folds)
        sizes = s[s.role == "train"].fold.value_counts().sort_index().tolist()
        if int((s.role == "test").sum()) != n_test or sizes != [n_tr] * n_folds:
            continue
        key = (balance_score(s), -np.median(nn_test_train(s, dist)))
        if best is None or key < best[0]:
            best = (key, seed, s)
    if best is None:
        raise RuntimeError("no seed gives exact set sizes")
    return best[1], best[2], pairs, dist


def report(t: pd.DataFrame, dist: pd.DataFrame, pairs: list) -> None:
    tr, te = t[t.role == "train"], t[t.role == "test"]
    d = nn_test_train(t, dist)
    print(f"test {len(te)}, train {len(tr)}, folds {tr.fold.value_counts().sort_index().tolist()}, "
          f"near pairs (< {NEAR_KM:.0f} km): {len(pairs)}, all in the same role/fold: {valid(t, pairs)}")
    print(f"test -> nearest train park: min {np.min(d):.0f}, median {np.median(d):.0f}, max {np.max(d):.0f} km")
    r = t.set_index("park_id")
    for k in sorted(tr.fold.unique()):
        va, rest = r.index[(r.role == "train") & (r.fold == k)], r.index[(r.role == "train") & (r.fold != k)]
        dk = dist.loc[va, rest].min(axis=1).values
        print(f"fold {k}: val -> nearest train park min {np.min(dk):.0f}, median {np.median(dk):.0f} km")
    for c in ["cls", "cell", "region", "client_id", "size_bin", "terrain_cls"]:
        print(f"-- {c}: test {pd.crosstab(t[c], t.role)['test'].to_dict()}")
        print(f"   folds {pd.crosstab(tr[c], tr.fold).to_dict('index')}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=None, help="fixed seed instead of the seed search")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    base = park_table()
    if a.seed is None:
        seed, t, pairs, dist = search(base)
    else:
        dist = distances(base); pairs = near_pairs(dist)
        seed, t = a.seed, make_split(base, a.seed, units_of(base, dist))
    t["seed"] = seed
    cols = ["park_id", "role", "fold", "cell", "cls", "client_id", "region", "size_bin", "terrain_cls", "cohort",
            "lat", "lon", "seed"]
    t[cols].sort_values(["role", "fold", "park_id"]).to_csv(a.out, index=False)
    print("written", a.out, "seed", seed)
    report(t, dist, pairs)
