#!/usr/bin/env python3
"""WP7a — hourly scaling-factor predictions of a trained TR model.

For the given evaluation split, loads the matching final checkpoint
(final_val / final_test) and predicts the 40 evaluation stations (the
early-stop stations of the split are excluded) for both time windows:
- spatial:  TRAINING period (spatial generalization)
- temporal: EVAL window Jul 2023 - Jun 2024 (spatiotemporal, round2 window)

Run with the frcst venv:
    frcst/bin/python scripts/era5_bc/wp6_predict_dl.py --model lstm --eval-split test --device cuda:0

Output: results/era5_bc/pred_{model}_{split}_{window}.parquet
        (station_id, timestamp, sf_pred)
"""

import argparse
import json
import os
import sys

import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
from era5_bc import static as S  # noqa: E402
from era5_bc.config import early_stop_split, load_config  # noqa: E402
from era5_bc.hpo import STUDY_VERSION  # noqa: E402
from era5_bc.models import build_model  # noqa: E402
from era5_bc.train import predict  # noqa: E402
from era5_bc.windows import build_windows  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["lstm", "transformer"])
    ap.add_argument("--eval-split", required=True, choices=["val", "test"])
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = load_config()
    per = cfg["periods"]
    _, eval_ids = early_stop_split(cfg, args.eval_split)

    ckpt_dir = os.path.join(cfg["paths"]["checkpoints_dir"],
                            f"{args.model}_{STUDY_VERSION}",
                            f"final_{args.eval_split}")
    params = json.load(open(os.path.join(ckpt_dir, "params.json")))
    scaler = S.load_scaler(os.path.join(ckpt_dir, "static_scaler.json"))
    static_scaled = S.minmax_apply(S.load_static_table(cfg), scaler)

    model = build_model(cfg, args.model, overrides=params)
    model.load_state_dict(torch.load(os.path.join(ckpt_dir, "best.pt"),
                                     weights_only=True))

    windows = {"spatial": (per["train_start"], per["train_end"]),
               "temporal": (per["eval_start"], per["eval_end"])}
    os.makedirs(cfg["paths"]["results_dir"], exist_ok=True)
    for name, (start, end) in windows.items():
        ws = build_windows(cfg, eval_ids, static_scaled,
                           period_start=start, period_end=end,
                           past_len=params["past_len"])
        pred = predict(model, ws, cfg, device=args.device)
        out = os.path.join(cfg["paths"]["results_dir"],
                           f"pred_{args.model}_{args.eval_split}_{name}.parquet")
        pred.to_parquet(out, index=False)
        print(f"{args.eval_split}/{name}: {len(pred):,} hourly predictions "
              f"({pred.station_id.nunique()} stations) -> {out}")


if __name__ == "__main__":
    main()
