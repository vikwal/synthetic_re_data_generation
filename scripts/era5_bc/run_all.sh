#!/usr/bin/env bash
# ERA5 bias correction — full pipeline (Houndekindo & Ouarda 2025 reimpl.).
# Two interpreters: synthre (data/GBOOST/eval, no torch) and frcst (torch+CUDA).
# Steps are idempotent; comment out what is already done.
set -eu
cd "$(dirname "$0")/../.."

PY_DATA=./synthre/bin/python
PY_TORCH=/home/viktor/Work/forecasting_framework/frcst/bin/python

# ---- WP0-WP2: data preparation (synthre) ----
$PY_DATA scripts/era5_bc/wp0_distance_to_coast.py
$PY_DATA scripts/era5_bc/wp1_build_dataset.py
$PY_DATA scripts/era5_bc/wp2_static_features.py

# ---- WP3: TI-GBOOST baseline (synthre) ----
$PY_DATA scripts/era5_bc/wp3_train_gboost.py

# ---- WP4 sanity + WP5 HPO (frcst; also see l1 workers in README) ----
$PY_TORCH scripts/era5_bc/wp4_smoke_test.py
bash scripts/era5_bc/run_hpo.sh 100

# ---- WP6: final training, one run per eval split (frcst) ----
# "val":  fit train stations,      early stop on 10 val stations
# "test": fit train+val stations,  early stop on 10 test stations
for model in lstm transformer; do
  for split in val test; do
    $PY_TORCH scripts/era5_bc/wp5_train_final.py --model $model --eval-split $split --device cuda:0
  done
done

# ---- WP7: predictions + evaluation + figures ----
for model in lstm transformer; do
  for split in val test; do
    $PY_TORCH scripts/era5_bc/wp6_predict_dl.py --model $model --eval-split $split --device cuda:0
  done
done
$PY_DATA scripts/era5_bc/test_metrics.py
$PY_DATA scripts/era5_bc/wp6_evaluate.py
$PY_DATA scripts/era5_bc/wp7_report_figs.py

# ---- WP8/WP9: production model (all stations) + park downscaling ----
for model in lstm transformer; do
  $PY_TORCH scripts/era5_bc/wp8_train_production.py --model $model --device cuda:0
  $PY_TORCH scripts/era5_bc/wp9_downscale_parks.py --model $model --device cuda:0
done
