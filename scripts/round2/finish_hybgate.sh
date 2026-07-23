#!/bin/bash
# M5_hybgate: run + score (merge mode, appends the new experiment only).
# Deliberately NO export/figures — old results stay untouched.
set -e
cd /home/viktor/Work/synthetic_re_data_generation
PY=$PWD/synthre/bin/python
STATE=/tmp/round2_hybgate.state
LOG=/tmp/round2_hybgate.log
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
step run_M5_hybgate $PY scripts/round2/run_ladder.py M5_hybgate
step score $PY scripts/round2/score_ladder.py --experiments M5_hybgate
echo "$(date +%H:%M:%S) HYBGATE FINISHED" >> "$LOG"
