#!/bin/bash
# Full sequential pipeline: aging experiments -> ladder restore -> K rungs ->
# LOO runs -> analyses -> scoring -> summary. One job, no inter-job polling.
set -e
cd /home/viktor/Work/synthetic_re_data_generation
PY=synthre/bin/python
LOG=${1:-/tmp/round2_full_sequence.log}

step() { echo "=== $(date +%H:%M:%S) $1"; }

step "aging experiments"
for exp in M3allconst M4all M4ball M5all; do
  $PY scripts/round2/run_ladder.py "$exp" >> "$LOG" 2>&1
  step "$exp done"
done

step "ladder restore (skip-generate)"
for exp in M1 M2 M2noage M3 M3noage M4 M4b M5; do
  $PY scripts/round2/run_ladder.py "$exp" --skip-generate >> "$LOG" 2>&1
done
step "restore done"

step "K rungs"
for exp in K1 K3 K5; do
  $PY scripts/round2/run_ladder.py "$exp" >> "$LOG" 2>&1
  step "$exp done"
done

step "LOO runs"
for exp in M5all_noQM M5all_PL M5all_noage; do
  $PY scripts/round2/run_ladder.py "$exp" >> "$LOG" 2>&1
  step "$exp done"
done

step "analyses"
$PY scripts/round2/analyze_ladder.py --stability-exp M3 >> "$LOG" 2>&1
$PY scripts/round2/ablation_table.py >> "$LOG" 2>&1

step "scoring (DBSCAN masks + 3 variants)"
$PY scripts/round2/score_ladder.py --check-vs-ladder >> "$LOG" 2>&1

step "summary figures + tables"
$PY scripts/round2/make_summary.py >> "$LOG" 2>&1

step "ALL DONE"
