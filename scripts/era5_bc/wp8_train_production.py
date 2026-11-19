#!/usr/bin/env python3
"""WP8 — production model for park downscaling.

Trains on ALL stations (train+val+test, 201 effective) with the best v2-HPO
config. No early stopping: a fixed epoch budget runs through, and a seeded
5 % WINDOW holdout (random station-days; every station stays in training)
selects the best checkpoint. Rationale (discussed 2026-07-10): with dropout
the val curve plateaus (median post-minimum drift +1.8 % over 102 logged
runs), so early stopping is not needed as overfitting protection — but
checkpoint selection is nearly free and catches the ~18 % of runs that do
drift. Park-near stations must be IN training (the BC at a park uses the
ERA5 grid point of its name-giving station), so no station-based holdout.

This model is for DEPLOYMENT ONLY — reported performance numbers remain
those of the final_val/final_test runs (wp5/wp6).

Run with the frcst venv:
    frcst/bin/python scripts/era5_bc/wp8_train_production.py --model lstm --device cuda:0

Output: {checkpoints_dir}/{model}_v2/production/
"""

import argparse
import json
import os
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
from era5_bc import static as S  # noqa: E402
from era5_bc import train as T  # noqa: E402
from era5_bc.config import load_config  # noqa: E402
from era5_bc.hpo import STUDY_VERSION  # noqa: E402
from era5_bc.models import build_model  # noqa: E402
from era5_bc.windows import WindowSet, build_windows  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from wp5_train_final import best_trial_params  # noqa: E402

HOLDOUT_FRAC = 0.05


def split_windows(ws: WindowSet, frac: float, seed: int):
    rng = np.random.default_rng(seed)
    hold = np.zeros(len(ws), dtype=bool)
    hold[rng.choice(len(ws), size=int(len(ws) * frac), replace=False)] = True

    def sub(m):
        return WindowSet(ws.Xd[m], ws.Xs[m], ws.Xdate[m], ws.y[m], ws.mask[m],
                         ws.station_ids[m], ws.day_start[m])
    return sub(~hold), sub(hold)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["lstm", "transformer"])
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = load_config()
    st, per = cfg["stations"], cfg["periods"]
    params = best_trial_params(cfg, args.model)

    fit_ids = (st["train_effective"] + st["val_effective"]
               + st["test_effective"])  # ALL stations

    ckpt_dir = os.path.join(cfg["paths"]["checkpoints_dir"],
                            f"{args.model}_{STUDY_VERSION}", "production")
    os.makedirs(ckpt_dir, exist_ok=True)

    table = S.load_static_table(cfg)
    scaler = S.minmax_fit(table, fit_ids)
    S.save_scaler(scaler, os.path.join(ckpt_dir, "static_scaler.json"))
    static_scaled = S.minmax_apply(table, scaler)

    ws = build_windows(cfg, fit_ids, static_scaled,
                       period_start=per["train_start"],
                       period_end=per["train_end"],
                       past_len=params["past_len"],
                       min_coverage=st["min_obs_coverage"])
    train_ws, monitor_ws = split_windows(ws, HOLDOUT_FRAC,
                                         cfg["model_common"]["seed"])
    print(f"windows: {len(train_ws):,} train / {len(monitor_ws):,} holdout "
          f"({monitor_ws.station_ids.shape[0] and len(np.unique(monitor_ws.station_ids))} stations in holdout)")

    with open(os.path.join(ckpt_dir, "params.json"), "w") as f:
        json.dump(params, f, indent=1)

    model = build_model(cfg, args.model, overrides=params)
    # patience >= max_epochs -> no early stop; the holdout only picks the
    # best checkpoint at the end
    max_epochs = int(cfg["model_common"]["max_epochs"])
    log = T.fit(model, train_ws, monitor_ws, cfg, lr=params["lr"],
                weight_decay=params.get("weight_decay", 0.0),
                batch_size=params.get("batch_size"),
                ckpt_dir=ckpt_dir, device=args.device,
                monitor="pooled", patience=max_epochs)

    bst = log.loc[log["vloss"].idxmin()]
    with open(os.path.join(ckpt_dir, "summary.json"), "w") as f:
        json.dump({"model": args.model, "purpose": "production", **params,
                   "best_epoch": int(bst["epoch"]),
                   "best_monitor_loss": float(bst["vloss"]),
                   "n_fit_stations": len(fit_ids),
                   "holdout_frac": HOLDOUT_FRAC,
                   "n_epochs_run": len(log)}, f, indent=1)
    print(f"production training done ({args.model}); "
          f"best epoch {int(bst['epoch'])} of {len(log)}")


if __name__ == "__main__":
    main()
