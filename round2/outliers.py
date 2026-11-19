"""DBSCAN outlier cleaning of measured park series (evaluation_real_wind.ipynb).

Two-stage per park, replicating the notebook's cell-15 logic:
  1. cut-in filter: hours with measured power == 0 while the slowest turbine's
     hub wind is already >= its cut-in speed -> curtailment/data artefact.
  2. DBSCAN on StandardScaler-transformed [v_hub, measured] with the per-park
     parameters from evaluation/best_dbscan_params.pkl; label -1 -> outlier.

Masks are model-dependent (v_hub comes from the experiment's synth output) —
a user decision; outlier shares per model are reported for transparency.
Degenerate/missing pkl entries fall back to the notebook's grid search
(maximize R2 subject to outlier share < 20 %), cached in
data/round2/dbscan_params_filled.pkl.
"""

import os
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PARAMS_PKL = os.path.join(REPO, "evaluation", "best_dbscan_params.pkl")
FILLED_PKL = os.path.join(REPO, "data", "round2", "dbscan_params_filled.pkl")
MAX_OUTLIER_SHARE = 0.20

_cache = {}


def load_params() -> dict:
    if "params" in _cache:
        return _cache["params"]
    with open(PARAMS_PKL, "rb") as f:
        params = dict(pickle.load(f))
    if os.path.exists(FILLED_PKL):
        with open(FILLED_PKL, "rb") as f:
            params.update(pickle.load(f))
    _cache["params"] = params
    return params


def _grid_search(X: np.ndarray, meas: np.ndarray, synth: np.ndarray) -> dict:
    """Notebook's (commented-out) grid search: maximize post-filter R2 of
    synth vs meas, constraint outlier share < 20 %."""
    best = {"eps": 0.1, "min_samples": 10, "r2": -np.inf}
    Xs = StandardScaler().fit_transform(X)
    for eps in np.arange(0.02, 0.31, 0.02):
        for min_samples in (3, 5, 8, 10, 15, 25, 40):
            labels = DBSCAN(eps=float(eps), min_samples=int(min_samples)) \
                .fit_predict(Xs)
            keep = labels != -1
            share = 1.0 - keep.mean()
            if share >= MAX_OUTLIER_SHARE or keep.sum() < 100:
                continue
            r2 = r2_score(meas[keep], synth[keep])
            if r2 > best["r2"]:
                best = {"eps": float(eps), "min_samples": int(min_samples),
                        "r2": float(r2)}
    return best


def get_park_params(park_id: str, X: np.ndarray = None, meas: np.ndarray = None,
                    synth: np.ndarray = None) -> dict:
    params = load_params()
    p = params.get(park_id)
    if p is not None and np.isfinite(p.get("r2", -np.inf)):
        return p
    # degenerate or missing -> grid-search fallback, cached
    assert X is not None, f"no valid DBSCAN params for {park_id} and no data given"
    p = _grid_search(X, meas, synth)
    filled = {}
    if os.path.exists(FILLED_PKL):
        with open(FILLED_PKL, "rb") as f:
            filled = pickle.load(f)
    filled[park_id] = p
    with open(FILLED_PKL, "wb") as f:
        pickle.dump(filled, f)
    _cache.pop("params", None)
    print(f"[outliers] grid-search fallback for {park_id}: {p}")
    return p


def compute_mask(park_id: str, meas: pd.Series, v_hub: pd.Series,
                 min_cut_in: float, synth: pd.Series = None) -> pd.Series:
    """Boolean keep-mask on the intersection index of meas and v_hub.

    meas: measured park power (any unit), hourly. v_hub: model hub wind (m/s).
    min_cut_in: smallest turbine cut-in of the park (cut-in filter).
    synth: model park power — only needed for the grid-search fallback R2.
    """
    df = pd.concat([meas.rename("real"), v_hub.rename("v")], axis=1).dropna()
    keep = pd.Series(True, index=df.index)
    # stage 1: cut-in filter
    keep[(df["real"] == 0) & (df["v"] >= min_cut_in)] = False
    # stage 2: DBSCAN on the remaining points
    sub = df[keep]
    X = sub[["v", "real"]].values
    synth_sub = synth.reindex(sub.index).values if synth is not None else None
    p = get_park_params(park_id, X, sub["real"].values, synth_sub)
    labels = DBSCAN(eps=p["eps"], min_samples=int(p["min_samples"])) \
        .fit_predict(StandardScaler().fit_transform(X))
    keep.loc[sub.index[labels == -1]] = False
    return keep
