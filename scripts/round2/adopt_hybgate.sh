#!/bin/bash
# Hybrid-gate adoption: data/round2/correction now carries the hybrid gate
# (distance-gate backup: data/round2_distgate/). Only 3 parks change branch
# (05347 B->A, 01200_1/01200_2 A->C), so the 15 correction-active main
# experiments rerun for those parks only, then full downstream. Idempotent.
set -e
cd /home/viktor/Work/synthetic_re_data_generation
PY=$PWD/synthre/bin/python
STATE=/tmp/round2_adopt.state
LOG=/tmp/round2_adopt.log
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
for exp in M2 M2noage M2all M3 M3noage M3allconst M4 M4all M4all_noage \
           M4b M4ball M5 M5all M5all_noage M5all_PL; do
  step "run_$exp" $PY scripts/round2/run_ladder.py "$exp" --parks 05347 01200_1 01200_2
done
step score   $PY scripts/round2/score_ladder.py --check-vs-ladder
step m0      $PY scripts/round2/score_m0.py
step wp4     $PY scripts/round2/wp4_fleet_regression.py --experiment M3noage
step tables  $PY scripts/round2/make_summary.py --tables-only
step export  $PY scripts/round2/export_paper_tables.py
step decomp  $PY scripts/round2/error_decomposition.py --rungs M0 M1noage M2noage M3noage M4all M5all K5 M5_simgate
step figs    $PY scripts/round2/regen_paper_figs.py
step tornado $PY scripts/round2/wp6_tornado.py
step morris  $PY scripts/round2/wp6_morris.py
step sobol   $PY scripts/round2/wp6_sobol.py
step sa_copy cp results/round2/morris_ranking.csv results/round2/sobol_indices.csv results/round2/summary/
echo "$(date +%H:%M:%S) HYBGATE ADOPTION FINISHED" >> "$LOG"
