#!/usr/bin/env python
"""Does it have to be a full ring? Subsets of the eight CAMS ring points (3 and 8 km, N/O/S/W) with
otherwise identical predictors: CAMS centre + geometry + ERA5 solar/met + the centre's time context,
plus for the chosen points their kc/kd, the subset mean kc and its +-30 min lags.

    python scripts/cams_bc/ring_subset_study.py --part 0 --nparts 4 --gpu 0
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
from cams_bc import data, features as F, splits  # noqa: E402
from cams_bc.config import RESULTS  # noqa: E402

BASE_GROUPS = ["cams", "geom", "era5_solar", "era5_met"]
CENTRE_CONTEXT = ["kc_lag_m1", "kc_lag_p1", "kc_lag_m2", "kc_lag_p2", "kc_std5", "kc_mean5", "kc_daymean", "kc_daystd", "kd_lag_m1", "kd_lag_p1"]
SUBSETS = {
    "none": [],
    "1pt_3N": ["3N"],
    "1pt_8N": ["8N"],
    "2pt_3N_8S": ["3N", "8S"],
    "2pt_3N_3S": ["3N", "3S"],
    "3pt_3N_8O_8S": ["3N", "8O", "8S"],
    "3pt_3N_3O_8S": ["3N", "3O", "8S"],
    "4pt_mixed_3N_3S_8O_8W": ["3N", "3S", "8O", "8W"],
    "4pt_3km": ["3N", "3O", "3S", "3W"],
    "4pt_8km": ["8N", "8O", "8S", "8W"],
    "8pt_full": ["3N", "3O", "3S", "3W", "8N", "8O", "8S", "8W"],
}


def add_subset_features(df: pd.DataFrame, pts: list) -> tuple:
    cols = []
    if not pts:
        return df, cols
    kc = [f"ring{p}_kc" for p in pts]
    kd = [f"ring{p}_kd" for p in pts]
    cols += kc + kd
    df = df.copy()
    df["sub_kc_mean"] = df[kc].mean(axis=1)
    if len(pts) > 1:
        df["sub_kc_std"] = df[kc].std(axis=1)
        cols.append("sub_kc_std")
    df["sub_centre_minus"] = df["kc_cams"] - df["sub_kc_mean"]
    # lags within each station's regular 30-min series (rows are daytime only; shift inside station)
    g = df.groupby("station_id", sort=False)["sub_kc_mean"]
    df["sub_kc_lag_m1"], df["sub_kc_lag_p1"] = g.shift(1), g.shift(-1)
    cols += ["sub_kc_mean", "sub_centre_minus", "sub_kc_lag_m1", "sub_kc_lag_p1"]
    return df, cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--nparts", type=int, default=1)
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    names = [n for i, n in enumerate(SUBSETS) if i % args.nparts == args.part]
    df, sp = C.load_train(F.FULL)
    base_cols = F.columns(BASE_GROUPS) + CENTRE_CONTEXT
    rows = []
    for name in names:
        t0 = time.time()
        d, extra = add_subset_features(df, SUBSETS[name])
        cols = base_cols + extra
        folds = []
        for k in range(3):
            tr, va = C.fold_frames(d, sp, k)
            pred = C.fit_predict("xgb", {}, "kc", cols, tr, va, f"cuda:{args.gpu}")
            s = C.score(va, pred)
            s.update({"fold": k, "subset": name, "n_points": len(SUBSETS[name]), "n_features": len(cols)})
            folds.append(s)
        rows += folds
        obj = np.mean([f["objective"] for f in folds])
        print(f"{name}: objective {obj:.4f} rel ghi {np.mean([f['rel_rmse_ghi'] for f in folds]):.4f} dhi {np.mean([f['rel_rmse_dhi'] for f in folds]):.4f} {time.time() - t0:.0f}s", flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(RESULTS, f"ring_subset_study_part{args.part}.csv"), index=False)


if __name__ == "__main__":
    main()
