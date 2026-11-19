#!/usr/bin/env python3
"""WP2 — static covariate table for all stations with ERA5 data.

Joins round2 topo features + distance to coast and computes ERA5 ws10
quantiles (5/50/95 %) over the TRAINING period. Written UNSCALED — the
min-max scaler is fitted at model-training time on the respective fit set
(train stations for HPO, train+val for the final models).

Run with the synthre venv:
    synthre/bin/python scripts/era5_bc/wp2_static_features.py
"""

import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
from era5_bc.config import load_config  # noqa: E402
from era5_bc.static import assemble_static_table  # noqa: E402


def main():
    cfg = load_config()
    ids = (cfg["stations"]["train_effective"] + cfg["stations"]["val_effective"]
           + cfg["stations"]["test_effective"])
    table = assemble_static_table(cfg, ids)

    out = cfg["paths"]["static_features"]
    os.makedirs(os.path.dirname(out), exist_ok=True)
    table.to_csv(out)

    print(f"wrote {out}: {table.shape[0]} stations x {table.shape[1]} covariates")
    print(table.describe().round(3).to_string())


if __name__ == "__main__":
    main()
