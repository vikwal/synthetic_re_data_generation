"""TI-GBOOST — time-invariant scaling factor with gradient boosting.

Follows TRWindBC (TI_GBOOST/computeGBFeatureImp.py, train_predict.py):
- target OBS_SF = mean(ws_obs) / mean(ws_era5) over the training period
  (paired valid hours), one sample per station
- random-search over the paper's parameter grid; here scored on the fixed
  50-station validation set instead of the authors' k-fold CV
- feature selection: median permutation importance across all options with
  val R^2 above a threshold; keep positively-important features (capped)
- final prediction: mean of n bagged refits (authors: 100)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .data import load_processed


def scaling_factor_target(cfg: dict, station_ids: list[str]) -> pd.Series:
    """OBS_SF per station over the TRAINING period, paired valid hours only."""
    per = cfg["periods"]
    out = {}
    for sid in station_ids:
        df = load_processed(cfg, sid).loc[per["train_start"]:per["train_end"]]
        paired = df[df["y_valid"]]
        out[sid] = float(paired["ws_obs"].mean() / paired["ws10"].mean())
    s = pd.Series(out, name="OBS_SF")
    s.index.name = "station_id"
    return s


def sample_options(cfg: dict, rng: np.random.Generator) -> list[dict]:
    """Random-search options from the paper grid (Table 1)."""
    space = cfg["hpo"]["gboost"]["space"]
    n = cfg["hpo"]["gboost"]["n_options"]
    opts = []
    for _ in range(n):
        opts.append(dict(
            loss="squared_error",
            criterion="friedman_mse",
            validation_fraction=0.3,
            learning_rate=float(rng.choice(space["learning_rate"])),
            subsample=float(rng.choice(space["subsample"])),
            max_depth=int(rng.choice(space["max_depth"])),
            max_features=float(rng.choice(space["max_features"])),
            n_estimators=int(rng.choice(space["n_estimators"])),
            n_iter_no_change=int(rng.choice(space["n_iter_no_change"])),
            min_samples_split=float(rng.choice(space["min_samples_split"])),
        ))
    return opts


def eval_option(X: pd.DataFrame, y: pd.Series, train_ids: list[str],
                val_ids: list[str], params: dict, features: list[str],
                seed: int) -> tuple[dict, pd.Series]:
    """Fit on train stations, score + permutation importance on val stations."""
    model = GradientBoostingRegressor(random_state=seed, **params)
    model.fit(X.loc[train_ids, features], y.loc[train_ids])

    x_val, y_val = X.loc[val_ids, features], y.loc[val_ids]
    pred = np.clip(model.predict(x_val), 0, 1e5)
    scores = {"MAE": mean_absolute_error(y_val, pred),
              "R2": r2_score(y_val, pred),
              "MSE": mean_squared_error(y_val, pred)}
    pfi = permutation_importance(model, x_val, y_val, n_repeats=20,
                                 scoring="r2", random_state=seed)
    return scores, pd.Series(pfi.importances_mean, index=features)


def select_features(hpo_results: pd.DataFrame, importances: pd.DataFrame,
                    cfg: dict) -> list[str]:
    """Median PFI across options with val R^2 above threshold; keep features
    with positive importance, at most n_select_features (paper: top 11)."""
    gb_cfg = cfg["hpo"]["gboost"]
    good = hpo_results[hpo_results["R2"] > gb_cfg["r2_threshold"]].index
    assert len(good) > 0, (
        f"no option reached val R2 > {gb_cfg['r2_threshold']} — "
        "inspect gboost_hpo.csv, consider lowering the threshold")
    med = importances.loc[good].median(axis=0).sort_values(ascending=False)
    selected = med[med > 0].index[:gb_cfg["n_select_features"]].tolist()
    assert selected, "no positively important features"
    return selected


def bagged_predict(X: pd.DataFrame, y: pd.Series, fit_ids: list[str],
                   predict_ids: list[str], params: dict, features: list[str],
                   n_bagging: int) -> pd.Series:
    """Authors' final scheme: mean prediction over n stochastic refits."""
    preds = []
    for i in range(n_bagging):
        model = GradientBoostingRegressor(random_state=i, **params)
        model.fit(X.loc[fit_ids, features], y.loc[fit_ids])
        preds.append(np.clip(model.predict(X.loc[predict_ids, features]), 0, 1e5))
    return pd.Series(np.mean(preds, axis=0), index=pd.Index(predict_ids, name="station_id"),
                     name="sf_pred")
