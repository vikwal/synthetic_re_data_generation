#!/bin/bash
# Detached ask-14 job: waits for the ask-12/13 finisher, then runs the
# no-aging twins, scores them and re-exports the paper tables. Idempotent.
cd /home/viktor/Work/synthetic_re_data_generation
PY=$PWD/synthre/bin/python
STATE=/tmp/round2_ask14.state
LOG=/tmp/round2_ask14.log
touch "$STATE"

echo "$(date +%H:%M:%S) waiting for ask-12/13 finisher" >> "$LOG"
until grep -q "^figs$" /tmp/round2_finish.state 2>/dev/null; do sleep 30; done
echo "$(date +%H:%M:%S) ask-12/13 finished, starting ask 14" >> "$LOG"

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

step run_M4all_noage $PY scripts/round2/run_ladder.py M4all_noage
step run_K5_noage    $PY scripts/round2/run_ladder.py K5_noage
step score_twins     $PY scripts/round2/score_ladder.py --experiments M4all_noage K5_noage
step export          $PY scripts/round2/export_paper_tables.py
echo "$(date +%H:%M:%S) ASK14 ALL FINISHED" >> "$LOG"
