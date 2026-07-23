#!/bin/bash
# Detached rerun after the 00198_1 (Saxony-Anhalt) commissioning-date fix
# (2014-04-01 -> 2013-09-01, capacity-weighted from MaStR). Aging-active
# rungs for the park only, then full rescoring + downstream. Idempotent.
set -e
cd /home/viktor/Work/synthetic_re_data_generation
PY=$PWD/synthre/bin/python
STATE=/tmp/round2_00198.state
LOG=/tmp/round2_00198.log
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

for exp in M1 M2 M3 M4 M4b M5 M2all M3allconst M4all M4ball M5all M5all_PL M5all_noQM M5_gate5 K1 K3 K5 MK; do
  step "run_$exp" $PY scripts/round2/run_ladder.py "$exp" --parks 00198_1
done
step score  $PY scripts/round2/score_ladder.py --check-vs-ladder
step m0     $PY scripts/round2/score_m0.py
step wp4    $PY scripts/round2/wp4_fleet_regression.py --experiment M3noage
step tables $PY scripts/round2/make_summary.py --tables-only
step export $PY scripts/round2/export_paper_tables.py
step figs   $PY scripts/round2/regen_paper_figs.py
step tornado $PY scripts/round2/wp6_tornado.py
echo "$(date +%H:%M:%S) 00198_1 RERUN ALL FINISHED" >> "$LOG"
