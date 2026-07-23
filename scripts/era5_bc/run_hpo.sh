#!/usr/bin/env bash
# WP5 — Optuna HPO for both TR models, all GPUs, both models CONCURRENTLY.
# Studies era5_bc_{lstm,transformer} in $OPTUNA_STORAGE (shared PostgreSQL,
# visible in the forecasting_framework Optuna dashboard).
#
# Launches 2*N_GPUS workers at once: LSTM on GPU 0..3 AND Transformer on
# GPU 0..3 (two workers share each GPU — fine on an A100 80GB). Each study
# stops once it reaches TARGET_TRIALS COMPLETE trials (enforced via callback
# on the shared study, so worker count and restarts don't change the total).
#
# Usage: bash scripts/era5_bc/run_hpo.sh [TARGET_TRIALS]
set -eu
cd "$(dirname "$0")/../.."

PY_TORCH=/home/viktor/Work/forecasting_framework/frcst/bin/python
TARGET_TRIALS=${1:-16}
N_GPUS=4
LOGDIR=logs/era5_bc
mkdir -p "$LOGDIR"

: "${OPTUNA_STORAGE:?OPTUNA_STORAGE must be set}"

# create studies + enqueue paper defaults (idempotent)
for model in lstm transformer; do
  "$PY_TORCH" scripts/era5_bc/wp4_hpo_dl.py --model "$model" --setup
done

# launch all workers concurrently: both models across all GPUs
for model in lstm transformer; do
  for g in $(seq 0 $((N_GPUS - 1))); do
    "$PY_TORCH" scripts/era5_bc/wp4_hpo_dl.py \
      --model "$model" --target-trials "$TARGET_TRIALS" --device "cuda:${g}" \
      > "$LOGDIR/hpo_${model}_gpu${g}.log" 2>&1 &
  done
done
wait
echo "HPO complete (both studies reached $TARGET_TRIALS trials)."
