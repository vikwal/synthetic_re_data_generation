#!/usr/bin/env python3
"""WP6 — final training of the best HPO config, one run per evaluation split.

Two runs per model:
  --eval-split val   fit on the TRAIN stations; early stopping on 10 of the
                     50 val stations; BC quality is then evaluated on the
                     remaining 40 val stations.
  --eval-split test  fit on TRAIN+VAL stations; early stopping on 10 of the
                     50 test stations; evaluation on the remaining 40.

Using a few stations of the eval split for early stopping is not perfectly
clean (they share the split with the evaluation stations) but avoids the worse
alternative of early-stopping on training stations. The early-stop stations
are seeded/deterministic (era5_bc.config.early_stop_split), recorded in
summary.json, and NEVER part of the evaluation set.

Run with the frcst venv:
    frcst/bin/python scripts/era5_bc/wp5_train_final.py --model lstm --eval-split test --device cuda:0

Output: {checkpoints_dir}/{model}_v2/final_{split}/{best.pt, log.csv,
         params.json, static_scaler.json, summary.json}
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
from era5_bc import hpo as H  # noqa: E402
from era5_bc import static as S  # noqa: E402
from era5_bc import train as T  # noqa: E402
from era5_bc.config import early_stop_split, load_config  # noqa: E402
from era5_bc.models import build_model  # noqa: E402
from era5_bc.windows import build_windows  # noqa: E402


def best_trial_params(cfg: dict, core: str) -> dict:
    """Best trial params from the Optuna study (OPTUNA_STORAGE); falls back
    to scanning the per-trial summary.json files if the DB is unreachable."""
    try:
        import optuna
        study = optuna.load_study(study_name=H.study_name(core),
                                  storage=os.environ["OPTUNA_STORAGE"])
        bt = study.best_trial
        params = H.resolve_params(bt.params, core)
        print(f"best {core} trial {bt.number} "
              f"(study {study.study_name}, vloss {bt.value:.4f}): {params}")
        return params
    except Exception as exc:  # noqa: BLE001 — fallback path
        print(f"optuna study unavailable ({exc}); scanning summary files")
        pattern = os.path.join(cfg["paths"]["checkpoints_dir"],
                               f"{core}_{H.STUDY_VERSION}",
                               "trial_*", "summary.json")
        summaries = [json.load(open(p)) for p in glob.glob(pattern)]
        assert summaries, f"no completed trials found under {pattern}"
        best = min(summaries, key=lambda s: s["best_vloss"])
        print(f"best {core} trial (file scan): {best}")
        return H.resolve_params(
            {k: v for k, v in best.items()
             if k not in ("model", "trial", "best_epoch", "best_vloss",
                          "n_epochs_run")}, core)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["lstm", "transformer"])
    ap.add_argument("--eval-split", required=True, choices=["val", "test"])
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = load_config()
    st, per = cfg["stations"], cfg["periods"]
    params = best_trial_params(cfg, args.model)

    if args.eval_split == "val":
        fit_ids = list(st["train_effective"])
    else:
        fit_ids = st["train_effective"] + st["val_effective"]
    es_ids, eval_ids = early_stop_split(cfg, args.eval_split)
    assert not set(fit_ids) & set(es_ids) and not set(fit_ids) & set(eval_ids)
    assert not set(es_ids) & set(eval_ids)

    ckpt_dir = os.path.join(cfg["paths"]["checkpoints_dir"],
                            f"{args.model}_{H.STUDY_VERSION}",
                            f"final_{args.eval_split}")
    os.makedirs(ckpt_dir, exist_ok=True)

    table = S.load_static_table(cfg)
    scaler = S.minmax_fit(table, fit_ids)
    S.save_scaler(scaler, os.path.join(ckpt_dir, "static_scaler.json"))
    static_scaled = S.minmax_apply(table, scaler)

    kw = dict(period_start=per["train_start"], period_end=per["train_end"],
              past_len=params["past_len"],
              min_coverage=st["min_obs_coverage"])
    train_ws = build_windows(cfg, fit_ids, static_scaled, **kw)
    monitor_ws = build_windows(cfg, es_ids, static_scaled, **kw)
    assert not set(np.unique(train_ws.station_ids)) & set(es_ids + eval_ids)

    with open(os.path.join(ckpt_dir, "params.json"), "w") as f:
        json.dump(params, f, indent=1)

    model = build_model(cfg, args.model, overrides=params)
    # monitor: median over the 10 ES stations (robust — a pooled mean is
    # dominated by extreme-factor stations like summits, see train.py)
    log = T.fit(model, train_ws, monitor_ws, cfg, lr=params["lr"],
                weight_decay=params.get("weight_decay", 0.0),
                batch_size=params.get("batch_size"),
                ckpt_dir=ckpt_dir, device=args.device,
                monitor="station_median")

    bst = log.loc[log["vloss"].idxmin()]
    with open(os.path.join(ckpt_dir, "summary.json"), "w") as f:
        json.dump({"model": args.model, "eval_split": args.eval_split,
                   **params,
                   "best_epoch": int(bst["epoch"]),
                   "best_monitor_loss": float(bst["vloss"]),
                   "n_fit_stations": len(fit_ids),
                   "early_stop_stations": es_ids,
                   "eval_stations": eval_ids}, f, indent=1)
    print(f"final training done ({args.model}, eval split {args.eval_split})")


if __name__ == "__main__":
    main()
