#!/bin/bash
# Detached finisher for HANDOFF asks 12+13 (survives VS Code / Claude exit).
# Idempotent: milestone markers in /tmp/round2_finish.state skip done steps.
cd /home/viktor/Work/synthetic_re_data_generation
PY=$PWD/synthre/bin/python
STATE=/tmp/round2_finish.state
LOG=/tmp/round2_finish.log
touch "$STATE"

step() {
  local name="$1"; shift
  if grep -q "^$name$" "$STATE"; then
    echo "$(date +%H:%M:%S) skip $name (done)" >> "$LOG"
    return 0
  fi
  echo "$(date +%H:%M:%S) start $name" >> "$LOG"
  if "$@" >> "$LOG" 2>&1; then
    echo "$name" >> "$STATE"
    echo "$(date +%H:%M:%S) OK $name" >> "$LOG"
  else
    echo "$(date +%H:%M:%S) FAILED $name (rc=$?)" >> "$LOG"
    exit 1
  fi
}

step wp4      $PY scripts/round2/wp4_fleet_regression.py --experiment M3noage
step morris   $PY scripts/round2/wp6_morris.py
step sobol    $PY scripts/round2/wp6_sobol.py
step ablation $PY scripts/round2/ablation_table.py
step export   $PY scripts/round2/export_paper_tables.py
step figs     $PY scripts/round2/regen_paper_figs.py
echo "$(date +%H:%M:%S) ALL FINISHED" >> "$LOG"
