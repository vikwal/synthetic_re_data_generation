#!/bin/bash
# Detached ask-18a job: waits for the 02483 rerun, then runs the restrictive-
# gate variant M5_gate5, scores it and re-exports the paper tables. 18b (ACF
# band) is code-level and already rendered by the 02483 finisher. Idempotent.
set -e
cd /home/viktor/Work/synthetic_re_data_generation
PY=$PWD/synthre/bin/python
STATE=/tmp/round2_ask18.state
LOG=/tmp/round2_ask18.log
touch "$STATE"

echo "$(date +%H:%M:%S) waiting for 02483 rerun" >> "$LOG"
until grep -q "02483 RERUN ALL FINISHED" /tmp/round2_02483.log 2>/dev/null; do sleep 30; done
echo "$(date +%H:%M:%S) 02483 rerun finished, starting ask 18a" >> "$LOG"

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

step run_gate5   $PY scripts/round2/run_ladder.py M5_gate5
step score_gate5 $PY scripts/round2/score_ladder.py --experiments M5_gate5
step export      $PY scripts/round2/export_paper_tables.py
echo "$(date +%H:%M:%S) ASK18 ALL FINISHED" >> "$LOG"
