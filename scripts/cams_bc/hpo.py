#!/usr/bin/env python
"""Optuna HPO of the CAMS correction, one worker per call (several workers share the study through
OPTUNA_STORAGE). Objective: mean over the 3 station folds of 0.5 (RMSE_ghi + RMSE_dhi)/RMSE_cams.

    python scripts/cams_bc/hpo.py --model xgb --trials 40 --gpu 0 --groups cams geom ring3 ring8 ring_agg context era5_solar era5_met
"""
import argparse
import os
import sys

import optuna

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
from cams_bc import features as F, hpo  # noqa: E402
from cams_bc.config import RESULTS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["xgb", "mlp"], required=True)
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--groups", nargs="+", default=F.FULL)
    ap.add_argument("--form", default="kc", help="target form, or 'search'")
    ap.add_argument("--version", default="v1")
    ap.add_argument("--storage", default=os.environ.get("OPTUNA_STORAGE"))
    args = ap.parse_args()
    df, sp = C.load_train(args.groups)
    name = hpo.study_name(args.model, args.version)
    study = optuna.create_study(study_name=name, storage=args.storage, direction="minimize", load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(multivariate=True, seed=None),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=1))
    study.set_user_attr("groups", list(args.groups))
    study.set_user_attr("form", args.form)
    device = f"cuda:{args.gpu}"

    def objective(trial):
        form = None if args.form == "search" else args.form
        p = hpo.sample_xgb(trial, form) if args.model == "xgb" else hpo.sample_mlp(trial, form)
        form = p.pop("form")

        def report(k, val):
            trial.report(val, k)
            if trial.should_prune():
                raise optuna.TrialPruned()

        obj, folds = C.cv_run(args.model, p, form, args.groups, df, sp, device, report=report)
        for f in folds:
            for key in ("rel_rmse_ghi", "rel_rmse_dhi", "r2_ghi", "r2_dhi", "best_iter_ghi", "best_iter_dhi", "epochs"):
                if key in f:
                    trial.set_user_attr(f"fold{f['fold']}_{key}", float(f[key]))
        return obj

    study.optimize(objective, n_trials=args.trials, catch=(RuntimeError, ValueError))
    os.makedirs(RESULTS, exist_ok=True)
    study.trials_dataframe().to_csv(os.path.join(RESULTS, f"{name}_trials.csv"), index=False)
    print("best", study.best_value, study.best_params)


if __name__ == "__main__":
    main()
