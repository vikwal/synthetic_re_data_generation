#!/usr/bin/env python
"""Measurement network as INPUT: do the k nearest stations' measured clear-sky index and CAMS error
help at an unseen location, and can they replace the CAMS ring? Same protocol as the predictor study.

    python scripts/cams_bc/neighbour_study.py --part 0 --nparts 4 --gpu 0
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
from cams_bc import features as F  # noqa: E402
from cams_bc.config import RESULTS  # noqa: E402

BASE = ["cams", "geom", "ring3", "ring8", "ring_agg", "context", "era5_solar", "era5_met"]
NO_RING = ["cams", "geom", "context", "era5_solar", "era5_met"]


def nb_cols(k, agg=True):
    c = [f"nb{i}_{v}" for i in range(k) for v in ("kc", "kd", "err", "dist_km")]
    return c + (["nb_err_wmean", "nb_kc_wmean", "nb_err_std"] if agg and k == F.N_NEIGHBOURS else [])


CONFIGS = {
    "base_no_nb": (BASE, []),
    "base+nb1": (BASE, nb_cols(1)),
    "base+nb2": (BASE, nb_cols(2)),
    "base+nb4": (BASE, nb_cols(4)),
    "noring+nb4": (NO_RING, nb_cols(4)),
    "noring_no_nb": (NO_RING, []),
    "base+nb4_err_only": (BASE, [f"nb{i}_err" for i in range(4)] + [f"nb{i}_dist_km" for i in range(4)] + ["nb_err_wmean"]),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--nparts", type=int, default=1)
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    names = [n for i, n in enumerate(CONFIGS) if i % args.nparts == args.part]
    df, sp = C.load_train(F.FULL)
    rows = []
    for name in names:
        groups, extra = CONFIGS[name]
        cols = F.columns(groups) + extra
        t0 = time.time()
        folds = []
        for k in range(3):
            tr, va = C.fold_frames(df, sp, k)
            pred = C.fit_predict("xgb", {}, "kc", cols, tr, va, f"cuda:{args.gpu}")
            s = C.score(va, pred)
            s.update({"fold": k, "config": name, "n_features": len(cols)})
            folds.append(s)
        rows += folds
        print(f"{name}: objective {np.mean([f['objective'] for f in folds]):.4f} rel ghi {np.mean([f['rel_rmse_ghi'] for f in folds]):.4f} "
              f"dhi {np.mean([f['rel_rmse_dhi'] for f in folds]):.4f} {time.time() - t0:.0f}s", flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(RESULTS, f"neighbour_study_part{args.part}.csv"), index=False)


if __name__ == "__main__":
    main()
