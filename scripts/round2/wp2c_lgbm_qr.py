#!/usr/bin/env python3
"""WP2-C — regionalized quantile regression (Houndekindo-style).

13 LightGBM quantile models (pinball loss): topo features + ERA5-10m quantiles
at the location -> observed DWD-10m quantile. Validation: leave-station-out
spatial CV. Baselines: (i) raw ERA5 quantile, (ii) nearest-station QM
(nearest station's DWD quantile scaled by the ERA5-quantile ratio).

Outputs: data/round2/lso_cv_skill.csv, data/round2/lgbm_models.pkl,
data/round2/model_quantiles.csv (predicted local quantiles for all locations)
"""

import os
import pickle
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
from round2.correction import QUANTILES  # noqa: E402

TOPO = os.path.join(REPO, "data", "round2", "topo_features.csv")
TABLE = os.path.join(REPO, "data", "round2", "station_correction_table.parquet")
TOPO_FEATURES = ["elevation", "slope", "aspect", "tpi5", "tpi75", "tdi",
                 "elev_std", "z0"]
LGB_PARAMS = dict(objective="quantile", n_estimators=300, learning_rate=0.05,
                  num_leaves=15, min_child_samples=10, subsample=0.9,
                  colsample_bytree=0.9, random_state=42, verbosity=-1)


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def pinball(y, yhat, q):
    d = y - yhat
    return float(np.mean(np.maximum(q * d, (q - 1) * d)))


def main():
    topo = pd.read_csv(TOPO, dtype={"location_id": str})
    table = pd.read_parquet(TABLE)
    st = topo[topo["kind"] == "station"].merge(
        table, left_on="location_id", right_on="station_id").dropna(
        subset=TOPO_FEATURES)
    print(f"stations: {len(st)}")

    q_cols_era5 = [f"q_era5_{q:.3f}" for q in QUANTILES]
    q_cols_dwd = [f"q_dwd_{q:.3f}" for q in QUANTILES]

    # ---- leave-one-station-out CV ----
    logo = LeaveOneGroupOut()
    groups = st["station_id"].values
    rows = []
    preds_cv = {q: np.full(len(st), np.nan) for q in QUANTILES}
    for qi, q in enumerate(QUANTILES):
        feats = TOPO_FEATURES + q_cols_era5
        X = st[feats].values
        y = st[q_cols_dwd[qi]].values
        model = lgb.LGBMRegressor(alpha=float(q), **LGB_PARAMS)
        for tr, te in logo.split(X, y, groups):
            model.fit(X[tr], y[tr])
            preds_cv[q][te] = model.predict(X[te])

    # nearest-station baseline (also leave-one-out: nearest OTHER station)
    lat, lon = st["latitude"].values, st["longitude"].values
    n = len(st)
    d = haversine_km(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
    np.fill_diagonal(d, np.inf)
    nearest = d.argmin(axis=1)

    for qi, q in enumerate(QUANTILES):
        y = st[q_cols_dwd[qi]].values
        yhat_model = preds_cv[q]
        yhat_raw = st[q_cols_era5[qi]].values
        # nearest-station QM: this location's ERA5 quantile x neighbour ratio
        ratio = st[q_cols_dwd[qi]].values[nearest] / \
            np.clip(st[q_cols_era5[qi]].values[nearest], 1e-6, None)
        yhat_nn = yhat_raw * ratio
        rows.append({
            "quantile": q,
            "pinball_lgbm": pinball(y, yhat_model, q),
            "pinball_raw_era5": pinball(y, yhat_raw, q),
            "pinball_nearest_qm": pinball(y, yhat_nn, q),
            "rmse_lgbm": float(np.sqrt(np.mean((y - yhat_model) ** 2))),
            "rmse_raw_era5": float(np.sqrt(np.mean((y - yhat_raw) ** 2))),
            "rmse_nearest_qm": float(np.sqrt(np.mean((y - yhat_nn) ** 2))),
        })
    skill = pd.DataFrame(rows)
    skill.to_csv(os.path.join(REPO, "data", "round2", "lso_cv_skill.csv"), index=False)
    print(skill.round(4).to_string(index=False))
    mean_lgbm = skill["pinball_lgbm"].mean()
    mean_raw = skill["pinball_raw_era5"].mean()
    mean_nn = skill["pinball_nearest_qm"].mean()
    print(f"\nmean pinball: lgbm={mean_lgbm:.4f} raw={mean_raw:.4f} nn_qm={mean_nn:.4f}")
    print("LGBM beats nearest-station QM:", mean_lgbm < mean_nn)

    # ---- final models on all stations + predictions for all locations ----
    models = {}
    for qi, q in enumerate(QUANTILES):
        model = lgb.LGBMRegressor(alpha=float(q), **LGB_PARAMS)
        model.fit(st[TOPO_FEATURES + q_cols_era5].values, st[q_cols_dwd[qi]].values)
        models[float(q)] = model
    with open(os.path.join(REPO, "data", "round2", "lgbm_models.pkl"), "wb") as f:
        pickle.dump({"models": models, "topo_features": TOPO_FEATURES,
                     "q_cols_era5": q_cols_era5}, f)

    # ERA5 quantiles for parks/sites: their driving station's ERA5 quantiles
    # (chain runs on station wind). location_id 'site_XXXXX'/'park_XXXXX_N'.
    locs = topo.dropna(subset=TOPO_FEATURES).copy()

    def station_of(loc):
        if loc["kind"] == "station":
            return loc["location_id"]
        if loc["kind"] == "site":
            return loc["location_id"].split("_")[1]
        return loc["location_id"].split("_")[1][:5]  # park_XXXXX(_N)

    tbl = table.set_index("station_id")
    out_rows = []
    for _, loc in locs.iterrows():
        sid = station_of(loc)
        if sid not in tbl.index:
            continue
        q_era5 = tbl.loc[sid, q_cols_era5].values.astype(float)
        X = np.concatenate([loc[TOPO_FEATURES].values.astype(float), q_era5])[None, :]
        pred = [models[float(q)].predict(X)[0] for q in QUANTILES]
        pred = np.maximum.accumulate(np.clip(pred, 0.0, None))  # enforce monotone
        out_rows.append({"location_id": loc["location_id"], "kind": loc["kind"],
                         "station_id": sid,
                         **{f"q_model_{q:.3f}": v for q, v in zip(QUANTILES, pred)}})
    pd.DataFrame(out_rows).to_csv(
        os.path.join(REPO, "data", "round2", "model_quantiles.csv"), index=False)
    print(f"model quantiles written for {len(out_rows)} locations")


if __name__ == "__main__":
    main()
