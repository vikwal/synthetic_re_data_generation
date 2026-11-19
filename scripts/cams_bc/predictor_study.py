#!/usr/bin/env python
"""Predictor study before any HPO: which feature groups and which target form carry, measured with a
default XGBoost in the 3-fold station cross-validation (validation stations unseen in training).

Three blocks: (A) cumulative ladder of feature groups, (B) target forms on the full set, (C) leave-one-
group-out from the full set. Configurations are split over --nparts processes (one GPU each).

    python scripts/cams_bc/predictor_study.py --part 0 --nparts 4 --gpu 0
"""
import argparse
import json
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
from cams_bc import features as F  # noqa: E402
from cams_bc.config import RESULTS  # noqa: E402
from cams_bc.targets import FORMS  # noqa: E402


def configurations() -> list:
    cfgs = [{"block": "A_ladder", "name": "+".join(g), "groups": g, "form": "kc"} for g in F.LADDER]
    cfgs += [{"block": "B_form", "name": f"full/{f}", "groups": F.FULL, "form": f} for f in FORMS if f != "kc"]
    cfgs += [{"block": "C_loo", "name": f"full-{g}", "groups": [x for x in F.FULL if x != g], "form": "kc"} for g in F.FULL]
    return cfgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--nparts", type=int, default=1)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", default=RESULTS)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    cfgs = [c for i, c in enumerate(configurations()) if i % args.nparts == args.part]
    df, sp = C.load_train()
    print(f"part {args.part}: {len(cfgs)} configurations, {len(df):,} rows", flush=True)
    rows = []
    for c in cfgs:
        t0 = time.time()
        obj, folds = C.cv_run("xgb", {}, c["form"], c["groups"], df, sp, f"cuda:{args.gpu}")
        for f in folds:
            rows.append({**{k: v for k, v in c.items() if k != "groups"}, "n_features": len(F.columns(c["groups"])), **f})
        print(f"{c['block']} {c['name']}: objective {obj:.4f}  rel_rmse ghi {sum(f['rel_rmse_ghi'] for f in folds) / 3:.4f} "
              f"dhi {sum(f['rel_rmse_dhi'] for f in folds) / 3:.4f}  {time.time() - t0:.0f}s", flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(args.out, f"predictor_study_part{args.part}.csv"), index=False)
        if c["block"] == "A_ladder" and c["groups"] == F.FULL:
            # gain importance of the full model, fold 0
            tr, va = C.fold_frames(df, sp, 0)
            pred = C.fit_predict("xgb", {}, "kc", F.columns(F.FULL), tr, va, f"cuda:{args.gpu}")
            imp = {v: pred["info"][f"model_{v}"].importance(F.columns(F.FULL)) for v in C.VARS}
            json.dump(imp, open(os.path.join(args.out, "importance_full_fold0.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
