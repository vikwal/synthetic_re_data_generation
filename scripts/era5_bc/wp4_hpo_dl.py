#!/usr/bin/env python3
"""WP4/WP5 — Optuna HPO for the TR models (LSTM / Transformer).

v2 studies (era5_bc_{model}_v2) in the shared PostgreSQL storage (env
OPTUNA_STORAGE, same DB as the forecasting_framework Optuna dashboard).
Continuous full search space (see era5_bc/hpo.py and config hpo.dl.space):
lr (log), dropout, past_len (multiples of 24), static branch width/dropout,
weight_decay (log), plus LSTM hidden/num_layers or Transformer
nheads*head_dim/nblock. TPE sampler. The paper-default reference lives in
the v1 studies (trial 0 there).

Multiple workers (one per GPU and model) share each study via the storage;
every worker stops once the study reaches --target-trials COMPLETE trials.

Run with the frcst venv:
    frcst/bin/python scripts/era5_bc/wp4_hpo_dl.py --model lstm --setup
    frcst/bin/python scripts/era5_bc/wp4_hpo_dl.py --model lstm --target-trials 100 --device cuda:0

Checkpoints per trial: {checkpoints_dir}/{model}_v2/trial_{N}/
(best.pt, log.csv, params.json, summary.json).
"""

import argparse
import json
import os
import sys

import optuna

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
from era5_bc import static as S  # noqa: E402
from era5_bc import train as T  # noqa: E402
from era5_bc.config import load_config  # noqa: E402
from era5_bc.hpo import STUDY_VERSION, sample_params, study_name  # noqa: E402
from era5_bc.models import build_model  # noqa: E402
from era5_bc.windows import build_windows  # noqa: E402

N_STARTUP_TRIALS = 15  # random exploration before TPE kicks in


def get_study(model: str) -> optuna.Study:
    return optuna.create_study(
        study_name=study_name(model), storage=os.environ["OPTUNA_STORAGE"],
        direction="minimize",
        sampler=optuna.samplers.TPESampler(n_startup_trials=N_STARTUP_TRIALS),
        load_if_exists=True)


def make_objective(cfg: dict, model_name: str, device: str):
    st, per = cfg["stations"], cfg["periods"]
    space = cfg["hpo"]["dl"]["space"]

    # static scaling: min-max on TRAIN stations only (HPO phase)
    table = S.load_static_table(cfg)
    scaler = S.minmax_fit(table, st["train_effective"])
    static_scaled = S.minmax_apply(table, scaler)

    window_cache = {}  # past_len -> (train_ws, val_ws); reused across trials

    def objective(trial: optuna.Trial) -> float:
        params = sample_params(trial, space, model_name)
        print(f"[{study_name(model_name)}] trial {trial.number}: {params}")

        if params["past_len"] not in window_cache:
            window_cache.clear()  # past_len varies per trial; keep 1 entry
            kw = dict(past_len=params["past_len"],
                      period_start=per["train_start"], period_end=per["train_end"],
                      min_coverage=st["min_obs_coverage"])
            window_cache[params["past_len"]] = (
                build_windows(cfg, st["train_effective"], static_scaled, **kw),
                build_windows(cfg, st["val_effective"], static_scaled, **kw))
        train_ws, val_ws = window_cache[params["past_len"]]

        ckpt_dir = os.path.join(cfg["paths"]["checkpoints_dir"],
                                f"{model_name}_{STUDY_VERSION}",
                                f"trial_{trial.number}")
        os.makedirs(ckpt_dir, exist_ok=True)
        with open(os.path.join(ckpt_dir, "params.json"), "w") as f:
            json.dump(params, f, indent=1)
        S.save_scaler(scaler, os.path.join(ckpt_dir, "static_scaler.json"))

        model = build_model(cfg, model_name, overrides=params)
        log = T.fit(model, train_ws, val_ws, cfg, lr=params["lr"],
                    weight_decay=params["weight_decay"],
                    batch_size=params["batch_size"],
                    ckpt_dir=ckpt_dir, device=device,
                    epoch_callback=lambda ep, tl, vl: trial.report(vl, ep))

        best = log.loc[log["vloss"].idxmin()]
        trial.set_user_attr("best_epoch", int(best["epoch"]))
        trial.set_user_attr("n_epochs_run", len(log))
        trial.set_user_attr("n_params", model.count_parameters())
        trial.set_user_attr("ckpt_dir", ckpt_dir)
        with open(os.path.join(ckpt_dir, "summary.json"), "w") as f:
            json.dump({"model": model_name, "trial": trial.number, **params,
                       "best_epoch": int(best["epoch"]),
                       "best_vloss": float(best["vloss"]),
                       "n_epochs_run": len(log)}, f, indent=1)
        return float(best["vloss"])

    return objective


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["lstm", "transformer"])
    ap.add_argument("--setup", action="store_true",
                    help="create the study, then exit")
    ap.add_argument("--target-trials", type=int, default=100,
                    help="total COMPLETE trials for the study; workers stop "
                         "once the study reaches this (shared across workers)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = load_config()
    study = get_study(args.model)

    if args.setup:
        print(f"{study.study_name}: ready ({len(study.trials)} trials)")
        return

    def stop_when_reached(st: optuna.Study, _trial):
        n_done = len(st.get_trials(deepcopy=False,
                                   states=(optuna.trial.TrialState.COMPLETE,)))
        if n_done >= args.target_trials:
            st.stop()

    # catch per-trial runtime errors (marked FAIL in the study) so one bad
    # config or OOM doesn't kill a long-running worker
    study.optimize(make_objective(cfg, args.model, args.device),
                   n_trials=args.target_trials, gc_after_trial=True,
                   callbacks=[stop_when_reached],
                   catch=(RuntimeError, ValueError))
    print(f"worker done; study best: {study.best_value:.4f} {study.best_params}")


if __name__ == "__main__":
    main()
