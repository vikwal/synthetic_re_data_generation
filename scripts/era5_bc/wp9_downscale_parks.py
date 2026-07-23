#!/usr/bin/env python3
"""WP9 — ERA5 downscaling at the 13 real wind parks with the production model.

For each park: static covariates at the PARK centroid (topo + dist_coast +
ERA5 quantiles), dynamics from the park's name-giving DWD station (the round2
chain drives parks by station id, so this is the exact ERA5 series the chain
consumes). Predicts hourly scaling factors over the full ERA5 span and writes
corrected 10 m wind. The same factor can be applied height-consistently to
v100 (round2/correction.py QM convention) by the chain.

Run with the frcst venv (after wp8):
    frcst/bin/python scripts/era5_bc/wp9_downscale_parks.py --model lstm --device cuda:0

Output: results/era5_bc/park_sf_{model}.parquet
        (park_id, timestamp UTC, sf_pred, ws10_era5, ws10_corr)
"""

import argparse
import json
import os
import re
import sys

import pandas as pd
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
from era5_bc import static as S  # noqa: E402
from era5_bc.config import load_config  # noqa: E402
from era5_bc.data import load_processed  # noqa: E402
from era5_bc.hpo import STUDY_VERSION  # noqa: E402
from era5_bc.models import build_model  # noqa: E402
from era5_bc.train import predict  # noqa: E402
from era5_bc.windows import build_windows  # noqa: E402


def park_station_map(cfg: dict) -> dict:
    """park location_id -> name-giving DWD station id (park_00198_1 -> 00198)."""
    topo = pd.read_csv(cfg["paths"]["topo_features"],
                       dtype={"location_id": str})
    parks = topo.loc[topo["kind"] == "park", "location_id"]
    out = {}
    for pid in parks:
        m = re.fullmatch(r"park_(\d{5})(?:_\d+)?", pid)
        assert m, f"unexpected park id: {pid}"
        out[pid] = m.group(1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["lstm", "transformer"])
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = load_config()
    ckpt_dir = os.path.join(cfg["paths"]["checkpoints_dir"],
                            f"{args.model}_{STUDY_VERSION}", "production")
    params = json.load(open(os.path.join(ckpt_dir, "params.json")))
    scaler = S.load_scaler(os.path.join(ckpt_dir, "static_scaler.json"))

    p2s = park_station_map(cfg)
    park_static = S.assemble_park_static_table(cfg, p2s)
    static_scaled = S.minmax_apply(park_static, scaler)
    outside = ((static_scaled < -0.05) | (static_scaled > 1.05)).sum().sum()
    if outside:
        print(f"note: {outside} park static values slightly outside the "
              f"station min-max range (extrapolated)")

    model = build_model(cfg, args.model, overrides=params)
    model.load_state_dict(torch.load(os.path.join(ckpt_dir, "best.pt"),
                                     weights_only=True))

    # full ERA5 span; first ceil(past_len/24) days lack context
    any_frame = load_processed(cfg, next(iter(p2s.values())))
    start, end = any_frame.index[0], any_frame.index[-1]
    ws = build_windows(cfg, list(p2s), static_scaled,
                       period_start=start, period_end=end,
                       past_len=params["past_len"],
                       require_targets=False, frame_ids=p2s)
    sf = predict(model, ws, cfg, device=args.device)
    sf = sf.rename(columns={"station_id": "park_id"})

    # join uncorrected ERA5 ws10 of the corresponding station grid point
    parts = []
    for pid, sid in p2s.items():
        ws10 = load_processed(cfg, sid)["ws10"]
        sub = sf[sf["park_id"] == pid].set_index("timestamp")
        sub["ws10_era5"] = ws10.reindex(sub.index)
        parts.append(sub.reset_index())
    out = pd.concat(parts, ignore_index=True)
    out["ws10_corr"] = out["ws10_era5"] * out["sf_pred"]

    path = os.path.join(cfg["paths"]["results_dir"],
                        f"park_sf_{args.model}.parquet")
    out[["park_id", "timestamp", "sf_pred", "ws10_era5", "ws10_corr"]] \
        .to_parquet(path, index=False)

    print(f"wrote {path}: {len(out):,} hours, {out.park_id.nunique()} parks, "
          f"{out.timestamp.min():%Y-%m-%d} .. {out.timestamp.max():%Y-%m-%d}")
    print(out.groupby("park_id")["sf_pred"].median().round(3).to_string())


if __name__ == "__main__":
    main()
