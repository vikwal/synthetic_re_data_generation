#!/usr/bin/env python
"""Review follow-up: does the imputed share of the neighbour-station input matter, and do the constant
distance columns help or act as a station identifier?

The neighbour input is observed-else-imputed. The imputer used other stations' measurements, so where
a neighbour has a gap, the target station's own measurement can enter its neighbour features
indirectly (a leak at the test stations; at a park the imputer never saw the site). Here the
neighbour features of the training stations are rebuilt from the pure observation (ghi_raw, NaN at
gaps) and compared with the filled version under the default XGB, 3 station folds, same rows.

    python scripts/cams_bc/neighbour_source_study.py --gpu 0
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
from cams_bc import data, features as F  # noqa: E402
from cams_bc.config import RESULTS  # noqa: E402

BASE = ["cams", "geom", "ring3", "ring8", "ring_agg", "context", "era5_solar", "era5_met"]
# run on the dataset of 2026-10-13 morning (filled neighbour input incl. nb*_dist_km); result: raw ==
# filled, dropping the distances +0.35 Pp -> features.NEIGHBOURS has no distances since, and the
# dataset carries the raw input. Re-running needs a frame that still has the dist columns.
NB_NODIST = F.columns(["neighbours"])
NB = NB_NODIST + [f"nb{i}_dist_km" for i in range(F.N_NEIGHBOURS)]

CONFIGS = {
    "base_no_nb": (None, []),
    "base+nb4_filled": ("filled", NB),
    "base+nb4_filled_nodist": ("filled", NB_NODIST),
    "base+nb4_raw": ("raw", NB),
    "base+nb4_raw_nodist": ("raw", NB_NODIST),
}


def with_neighbours(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Replace the nb* columns of the training frame by features built from the given table source."""
    table = data.neighbour_table(source=source)
    coords = data.coords_all()
    out = df.copy()
    for sid, pos in df.groupby("station_id").indices.items():
        idx = df.index[pos]
        nb = data.neighbour_features(float(coords.loc[sid, "lat"]), float(coords.loc[sid, "lon"]), idx, table, coords, exclude=[sid])
        out.iloc[pos, [out.columns.get_loc(c) for c in NB]] = nb[NB].to_numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    df, sp = C.load_train(F.FULL)
    frames = {"filled": df, "raw": with_neighbours(df, "raw")}
    rows = []
    for name, (source, extra) in CONFIGS.items():
        d = frames[source] if source else df
        cols = F.columns(BASE) + extra
        t0 = time.time()
        folds = []
        for k in range(3):
            tr, va = C.fold_frames(d, sp, k)
            pred = C.fit_predict("xgb", {}, "kc", cols, tr, va, f"cuda:{args.gpu}")
            s = C.score(va, pred)
            s.update({"fold": k, "config": name, "n_features": len(cols)})
            folds.append(s)
        rows += folds
        print(f"{name}: objective {np.mean([f['objective'] for f in folds]):.4f} rel ghi {np.mean([f['rel_rmse_ghi'] for f in folds]):.4f} "
              f"dhi {np.mean([f['rel_rmse_dhi'] for f in folds]):.4f} {time.time() - t0:.0f}s", flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "neighbour_source_study.csv"), index=False)


if __name__ == "__main__":
    main()
