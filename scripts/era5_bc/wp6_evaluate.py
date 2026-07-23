#!/usr/bin/env python3
"""WP7b — evaluation of all BC methods on unseen stations.

Evaluation runs: split in {val, test} x window in {spatial, temporal}.
Per split only the 40 evaluation stations are used (the 10 early-stop
stations are excluded); the corresponding models were fitted WITHOUT any of
them (val run: fit on train stations; test run: fit on train+val).

Models: UC-ERA5 (uncorrected), TI-GBOOST (constant per-station factor from
gboost_predictions_{split}.csv), TR-LSTM, TR-Transformer (hourly factors
from wp6_predict_dl.py). Corrected wind = ws10_era5 * sf_pred vs DWD obs.

Metrics per station: hourly MAE/MBE/PCC/RMSE/R2, PSS (all/LWT/UPT),
quantiles, and the skill score
    skill = 1 - RMSE_model / RMSE_UC-ERA5
(>0: the BC improves over raw ERA5 at that unseen station).

Run with the synthre venv:
    synthre/bin/python scripts/era5_bc/wp6_evaluate.py

Outputs per (split, window):
    results/era5_bc/per_station_{split}_{window}.csv
    results/era5_bc/summary_{split}_{window}.csv
    results/era5_bc/summary_by_area_{split}_{window}.csv
"""

import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
from era5_bc import metrics as M  # noqa: E402
from era5_bc import static as S  # noqa: E402
from era5_bc.config import early_stop_split, load_config  # noqa: E402
from era5_bc.data import load_processed  # noqa: E402

DL_MODELS = ["lstm", "transformer"]
PERF = {"mae": 0.0, "rmse": 0.0, "mbe": 0.0, "pcc": 1.0}  # perfect scores
TS_COLS = ["mae", "mbe", "pcc", "rmse", "r2", "skill",
           "pss_all", "pss_lwt", "pss_upt"]


def classify_area(static_raw: pd.DataFrame) -> pd.Series:
    """Approximation of the paper's area types (their Table S3 criteria are
    in the supplement; thresholds here chosen for German conditions):
    coastal < 30 km to coast; hilly/mountainous: elev_std > 100 m;
    high SRL: z0 >= 0.6 (forest/urban); else low SRL. First match wins."""
    def one(r):
        if r["dist_coast"] < 30:
            return "coastal"
        if r["elev_std"] > 100:
            return "hilly"
        if r["z0"] >= 0.6:
            return "high_srl"
        return "low_srl"
    return static_raw.apply(one, axis=1)


def hourly_factors(cfg: dict, model: str, split: str, window: str):
    path = os.path.join(cfg["paths"]["results_dir"],
                        f"pred_{model}_{split}_{window}.parquet")
    if not os.path.exists(path):
        print(f"WARNING: {path} missing — skipping {model}/{split}/{window}")
        return None
    df = pd.read_parquet(path)
    return df.set_index(["station_id", "timestamp"])["sf_pred"]


def main():
    cfg = load_config()
    per, ev = cfg["periods"], cfg["evaluation"]
    res_dir = cfg["paths"]["results_dir"]
    area = classify_area(S.load_static_table(cfg))

    windows = {"spatial": (per["train_start"], per["train_end"]),
               "temporal": (per["eval_start"], per["eval_end"])}

    for split in ("val", "test"):
        gb_path = os.path.join(res_dir, f"gboost_predictions_{split}.csv")
        gb = pd.read_csv(gb_path, dtype={"station_id": str}) \
               .set_index("station_id")["sf_pred"]
        es_ids, eval_ids = early_stop_split(cfg, split)

        for window, (start, end) in windows.items():
            sf_dl = {m: hourly_factors(cfg, m, split, window)
                     for m in DL_MODELS}
            rows = []
            for sid in eval_ids:
                frame = load_processed(cfg, sid).loc[start:end]
                base = frame[frame["y_valid"]][["ws_obs", "ws10"]]
                if len(base) == 0:
                    continue
                obs = base["ws_obs"].to_numpy()

                preds = {"uc_era5": base["ws10"].to_numpy(),
                         "gboost": base["ws10"].to_numpy() * gb.loc[sid]}
                for m, sf in sf_dl.items():
                    if sf is None:
                        continue
                    aligned = sf.loc[sid].reindex(base.index)
                    ok = aligned.notna()
                    preds[m] = (base["ws10"][ok] * aligned[ok]).to_numpy()
                    preds[f"_obs_{m}"] = obs[ok.to_numpy()]

                for model in ("uc_era5", "gboost", *DL_MODELS):
                    if model not in preds:
                        continue
                    o = preds.get(f"_obs_{model}", obs)
                    rows.append({"station_id": sid, "model": model,
                                 **M.station_metrics(o, preds[model], ev)})

            per_station = pd.DataFrame(rows)
            # skill = 1 - RMSE_model / RMSE_UC-ERA5, per station
            rmse_uc = (per_station.query("model == 'uc_era5'")
                       .set_index("station_id")["rmse"])
            per_station["skill"] = (
                1.0 - per_station["rmse"]
                / per_station["station_id"].map(rmse_uc).to_numpy())
            per_station["area"] = per_station["station_id"].map(area)
            per_station.to_csv(
                os.path.join(res_dir, f"per_station_{split}_{window}.csv"),
                index=False)

            by_area = (per_station.groupby(["area", "model"])[TS_COLS]
                       .median().round(3))
            by_area.to_csv(os.path.join(
                res_dir, f"summary_by_area_{split}_{window}.csv"))

            # ---- summary: median across stations + improvements ----
            summary = []
            med = per_station.groupby("model")[TS_COLS].median()
            for model, r in med.iterrows():
                mws = M.median_ws_metrics(
                    per_station.query("model == @model")
                    .set_index("station_id"))
                row = {"model": model,
                       **{f"ts_{k}": v for k, v in r.items()},
                       **{f"medws_{k}": v for k, v in mws.items()}}
                for metric in ("mae", "rmse", "pcc"):
                    row[f"impr_{metric}_pct"] = M.pct_improvement(
                        r[metric], med.loc["uc_era5", metric], PERF[metric])
                summary.append(row)
            summary = pd.DataFrame(summary).set_index("model")
            summary.to_csv(os.path.join(res_dir,
                                        f"summary_{split}_{window}.csv"))

            print(f"\n=== split={split} window={window} "
                  f"({start:%Y-%m-%d} .. {end:%Y-%m-%d}), "
                  f"{per_station.station_id.nunique()} eval stations "
                  f"(ES stations excluded: {len(es_ids)}) ===")
            counts = (per_station.drop_duplicates("station_id")["area"]
                      .value_counts())
            print(f"areas: {counts.to_dict()}")
            cols = ["ts_mae", "ts_rmse", "ts_r2", "ts_pcc", "ts_skill",
                    "impr_rmse_pct", "medws_mae", "medws_r2"]
            print(summary[cols].round(3).to_string())


if __name__ == "__main__":
    main()
