#!/usr/bin/env bash
# Round 3: regenerate the 13-park ladder rungs with the IEC density
# normalisation (commit 4ca1380). run_ladder.py takes the YAML name
# (configs/round2/<NAME>.yaml); the output directory is the YAML's internal
# experiment_id (/mnt/nvme2/synthetic/wind/round2/<experiment_id>/).
# The round-2 series were backed up to round2_r2backup_20261102 (kept).
# Sequential; run_ladder also evaluates each rung (results/round2/ladder_metrics.csv).
# Run on l2:  nohup scripts/round2/regen_rungs_iec.sh > logs/regen_rungs_iec.out 2>&1 &
set -u
cd "$(dirname "$0")/.."
cd "$(git rev-parse --show-toplevel)"
RUNGS=(M1 M2 M3 M4 M4_noage M4_noWake S1 S2 S3 S3_noage S3_noQM)
BASE=/mnt/nvme2/synthetic/wind/round2
BACKUP=/mnt/nvme2/synthetic/wind/round2_r2backup_20261102
LOG=logs/regen_rungs_iec.log
mkdir -p "$BACKUP" logs
echo "$(date +%F_%T) REGEN_START ${RUNGS[*]}" | tee -a "$LOG"
for exp in "${RUNGS[@]}"; do
  yaml=configs/round2/$exp.yaml
  [ -f "$yaml" ] || { echo "$(date +%F_%T) MISSING $yaml" | tee -a "$LOG"; continue; }
  exp_id=$(grep -m1 "^ *experiment_id" "$yaml" | sed -E "s/.*experiment_id: *'?([A-Za-z0-9_]+)'?.*/\1/")
  if [ -d "$BASE/$exp_id" ] && [ ! -d "$BACKUP/$exp_id" ]; then
    cp -a "$BASE/$exp_id" "$BACKUP/$exp_id" && echo "$(date +%F_%T) backup $exp_id -> $BACKUP" >> "$LOG"
  fi
  echo "$(date +%F_%T) START $exp ($exp_id)" | tee -a "$LOG"
  synthre/bin/python scripts/round2/run_ladder.py "$exp" > "logs/regen_$exp.log" 2>&1
  rc=$?
  n=$(ls "$BASE/$exp_id"/synth_*.csv 2>/dev/null | wc -l)
  newest=$(ls -t "$BASE/$exp_id"/synth_*.csv 2>/dev/null | head -1 | xargs -r stat -c %y | cut -c1-16)
  echo "$(date +%F_%T) END $exp ($exp_id) rc=$rc files=$n newest=$newest" | tee -a "$LOG"
done
echo "$(date +%F_%T) REGEN_DONE" | tee -a "$LOG"
