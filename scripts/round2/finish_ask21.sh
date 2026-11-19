#!/bin/bash
# Detached ask-21a job: similarity-gate runs (M2_simgate, M5_simgate),
# scoring, error decomposition (incl. M5_simgate), export. Idempotent.
set -e
cd /home/viktor/Work/synthetic_re_data_generation
PY=$PWD/synthre/bin/python
STATE=/tmp/round2_ask21.state
LOG=/tmp/round2_ask21.log
touch "$STATE"

step() {
  local name="$1"; shift
  grep -q "^$name$" "$STATE" && { echo "skip $name" >> "$LOG"; return 0; }
  echo "$(date +%H:%M:%S) start $name" >> "$LOG"
  if "$@" >> "$LOG" 2>&1; then
    echo "$name" >> "$STATE"; echo "$(date +%H:%M:%S) OK $name" >> "$LOG"
  else
    echo "$(date +%H:%M:%S) FAILED $name (rc=$?)" >> "$LOG"; exit 1
  fi
}

step run_M2_simgate $PY scripts/round2/run_ladder.py M2_simgate
step run_M5_simgate $PY scripts/round2/run_ladder.py M5_simgate
step score  $PY scripts/round2/score_ladder.py --experiments M2_simgate M5_simgate
step decomp $PY scripts/round2/error_decomposition.py --rungs M0 M1noage M2noage M3noage M4all M5all K5 M5_simgate
step export $PY scripts/round2/export_paper_tables.py
echo "$(date +%H:%M:%S) ASK21 ALL FINISHED" >> "$LOG"
