# era5_bc — ERA5 10 m wind speed bias correction at DWD stations

Reimplementation of **Houndekindo & Ouarda (2025)**, *"LSTM and
Transformer-based framework for bias correction of ERA5 hourly wind speeds"*,
Energy 328, 136498 (paper PDF in `literature/`). Reference code:
https://github.com/kindo/TRWindBC (architecture details verified against it).

## Idea

Models predict a **scaling factor**, not the wind speed:

- **TI-GBOOST** (time-invariant): `SF = mean(ws_obs)/mean(ws_era5)` per
  station from static covariates only. Corrects the long-term level, cannot
  fix temporal variability.
- **TR-LSTM / TR-Transformer** (time-resolved): hourly
  `y_t = ws_obs,t / ws_era5,t` from static covariates + a 144 h ERA5 sequence
  (120 h context + the 24 h target day; one window per station-day).
  Corrected wind: `ws_corr,t = ws_era5,t * ŷ_t`.

## Setup (this repo)

- 203 DWD stations, split 103 train / 50 val / 50 test copied from
  `forecasting_framework/configs/dcrnn/config_wind_dcrnn.yaml`.
  `03196`, `15813` have no ERA5 v2 CSV → dropped (101 train).
- **Periods are round2-conform** (`scripts/round2/wp2a_station_table.py`):
  training **Jul 2024 – Apr 2026**, evaluation **Jul 2023 – Jun 2024**
  (the round2 validation window; hard leakage assert in `config.py`).
- HPO trains on train stations, early-stops/scores on the 50 val stations
  (same period → spatial generalization). Final training runs TWICE per
  model: the "val" run fits on the train stations, the "test" run on
  train+val; in each run 10 seeded stations of the respective eval split
  drive early stopping and the remaining 40 are the untouched evaluation
  set. Evaluated on both the training period (*spatial*) and the eval
  window (*temporal*). Headline metrics: hourly MAE/RMSE/R² plus
  skill = 1 − RMSE_model/RMSE_UC-ERA5 at unseen stations.
- Dynamic covariates (ERA5, nearest grid point): u10/v10 (as unit direction
  vectors), ws10, blh, sp, t2m — standardized per grid point over the
  training period. Static covariates: ERA5 ws10 quantiles (5/50/95 %,
  training period), elevation, slope, aspect (sin/cos), tpi5, tpi75, tdi,
  elev_std, z0 (round2 `topo_features.csv`) + **distance to coast** (new,
  Natural Earth 10 m coastline; `Dcoast` is also in the paper's candidate
  pool). Min-max scaled on the respective fit-station set.

## Documented deviations from the paper

1. **Masked loss instead of complete days**: DWD hourly data has gaps; invalid
   target hours get weight 0 (their ECCC data only contains complete days).
2. **Calm guard**: hours with `ws_era5 < 0.5 m/s` are masked from the loss and
   training targets are clipped to `[0, 8]` (the reference divides raw values;
   summit stations produce factor spikes otherwise). Config-switchable.
3. **Fixed CET (UTC+1)** for day boundaries and the temporal embedding
   (reference uses station-local time; DST would break the fixed 24 h day).
4. **Early stopping** (patience 10 on val loss) instead of a fixed 30 epochs +
   min-val-loss checkpoint selection; equivalent selection, less compute.
5. **Fixed val-station set** (the 50-station split) instead of the authors'
   random 30 % station split / k-fold; GBOOST random search scored on the same
   val set instead of 6-fold CV.
6. **Final retrain monitor**: val stations are consumed by the final
   train+val fit → early stopping on a seeded random 10 % window holdout.
7. **Terrain feature set**: `elev_std/tpi5/tpi75/tdi` stand in for the paper's
   multi-scale Gaussian-filter DEM features (SDS/DME/TAC at SG1..SG100);
   ~22 months of training data vs. their 10 years.

## Pipeline

`scripts/era5_bc/run_all.sh` documents the order and the two interpreters
(`synthre` for data/GBOOST/eval — no usable torch; `frcst` from
forecasting_framework for the DL models — torch 2.7 + 4 GPUs; never share
pickles between the two, scalers/stats are JSON).

| Step | Script | Output |
|---|---|---|
| WP0 | `wp0_distance_to_coast.py` | `data/era5_bc/dist_coast.csv` |
| WP1 | `wp1_build_dataset.py` | `/mnt/nvme2/synthetic/era5_bc/processed/*.parquet`, `norm_stats.json` |
| WP2 | `wp2_static_features.py` | `data/era5_bc/static_features.csv` (unscaled) |
| WP3 | `wp3_train_gboost.py` | `results/era5_bc/gboost_{hpo,predictions}.csv` |
| WP4/5 | `wp4_hpo_dl.py`, `run_hpo.sh` | `checkpoints/{model}/trial_K/` |
| WP6 | `wp5_train_final.py` | `checkpoints/{model}/final/` |
| WP7 | `wp6_predict_dl.py`, `wp6_evaluate.py`, `wp7_report_figs.py` | `results/era5_bc/{pred_*,per_station_*,summary_*}`, `figs/era5_bc/` |

Tests: `wp4_smoke_test.py` (windows/models/overfit), `test_metrics.py`.
