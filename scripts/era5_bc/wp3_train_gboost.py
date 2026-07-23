#!/usr/bin/env python3
"""WP3 — TI-GBOOST training: random search, feature selection, final bagging.

Steps (TRWindBC scheme adapted to the fixed 101/50/50 split):
1. target OBS_SF per station (training period)
2. random search (n_options) fit on train stations, scored on val stations
3. feature selection via median permutation importance over good options
4. best params re-scored with selected features
5. final: bagged fit (n_bagging) on train+val, predictions for ALL stations
   (train/val diagnostics + test evaluation), scaler refit on train+val

Run with the synthre venv:
    synthre/bin/python scripts/era5_bc/wp3_train_gboost.py

Outputs:
    results/era5_bc/gboost_hpo.csv          option scores (val MAE/R2/MSE)
    data/era5_bc/gboost_selected.json       selected features + best params
    results/era5_bc/gboost_predictions.csv  per-station predicted + observed SF
"""

import json
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
from era5_bc import gboost as G  # noqa: E402
from era5_bc import static as S  # noqa: E402
from era5_bc.config import load_config  # noqa: E402


def main():
    cfg = load_config()
    st = cfg["stations"]
    train_ids, val_ids = st["train_effective"], st["val_effective"]
    test_ids = st["test_effective"]
    all_ids = train_ids + val_ids + test_ids

    table = S.load_static_table(cfg)
    features_all = list(table.columns)

    # scale with train-only min-max for the HPO phase (paper convention)
    X_hpo = S.minmax_apply(table, S.minmax_fit(table, train_ids))
    y = G.scaling_factor_target(cfg, all_ids)

    # ---- random search + permutation importance ----
    rng = np.random.default_rng(cfg["model_common"]["seed"])
    options = G.sample_options(cfg, rng)
    scores, imps = [], []
    for i, params in enumerate(options):
        sc, imp = G.eval_option(X_hpo, y, train_ids, val_ids, params,
                                features_all, seed=i)
        scores.append(sc)
        imps.append(imp)
        if (i + 1) % 25 == 0:
            print(f"option {i + 1}/{len(options)}")
    hpo = pd.DataFrame(scores)
    hpo.index.name = "option"
    imps = pd.DataFrame(imps)

    os.makedirs(cfg["paths"]["results_dir"], exist_ok=True)
    hpo_path = os.path.join(cfg["paths"]["results_dir"], "gboost_hpo.csv")
    pd.concat([hpo, pd.DataFrame(options)], axis=1).to_csv(hpo_path)

    # ---- feature selection + best params ----
    selected = G.select_features(hpo, imps, cfg)
    best_opt = int(hpo["R2"].idxmax())
    best_params = options[best_opt]
    print(f"best option {best_opt}: R2={hpo.loc[best_opt, 'R2']:.3f} "
          f"MAE={hpo.loc[best_opt, 'MAE']:.3f}")
    print(f"selected features ({len(selected)}): {selected}")

    # re-score best params with the selected feature subset
    sc_sel, _ = G.eval_option(X_hpo, y, train_ids, val_ids, best_params,
                              selected, seed=best_opt)
    print(f"best params on selected features: R2={sc_sel['R2']:.3f} "
          f"MAE={sc_sel['MAE']:.3f}")

    sel_path = os.path.join(os.path.dirname(cfg["paths"]["static_features"]),
                            "gboost_selected.json")
    with open(sel_path, "w") as f:
        json.dump({"selectedFeatures": selected, "bestParams": best_params,
                   "val_scores_all_features": hpo.loc[best_opt].to_dict(),
                   "val_scores_selected": sc_sel}, f, indent=1)

    # ---- final: one bagged fit per evaluation split ----
    # "val" run: fit on train stations only -> val stations are unseen
    # "test" run: fit on train+val -> test stations are unseen
    for eval_split, fit_ids in (("val", list(train_ids)),
                                ("test", train_ids + val_ids)):
        unseen = val_ids if eval_split == "val" else test_ids
        assert not set(fit_ids) & set(unseen), f"{eval_split} leakage"
        X_final = S.minmax_apply(table, S.minmax_fit(table, fit_ids))
        pred = G.bagged_predict(X_final, y, fit_ids, all_ids, best_params,
                                selected, cfg["hpo"]["gboost"]["n_bagging"])

        out = pd.DataFrame({"sf_pred": pred, "sf_obs": y.loc[all_ids]})
        out["split"] = (["train"] * len(train_ids) + ["val"] * len(val_ids)
                        + ["test"] * len(test_ids))
        pred_path = os.path.join(cfg["paths"]["results_dir"],
                                 f"gboost_predictions_{eval_split}.csv")
        out.to_csv(pred_path)
        mae = (out.loc[out.split == eval_split, "sf_pred"]
               - out.loc[out.split == eval_split, "sf_obs"]).abs().mean()
        print(f"final bagged fit ({eval_split} run, {len(fit_ids)} fit "
              f"stations): unseen-{eval_split} SF-MAE={mae:.3f} -> {pred_path}")

    print(f"wrote {hpo_path}\n      {sel_path}")


if __name__ == "__main__":
    main()
