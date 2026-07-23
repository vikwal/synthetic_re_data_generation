#!/bin/bash
# Detached rerun after the 05426 (Palatinate) turbine fix
# (Vestas V126-3.3/137m -> V112-3.3/140m, sources: MaStR SEE969028349266,
# SLT transport reference, thewindpower.net cluster). The turbine change
# affects EVERY rung, so all 26 experiments rerun for the park, then full
# rescoring + downstream incl. SA (chain baseline shifts). Idempotent.
set -e
cd /home/viktor/Work/synthetic_re_data_generation
PY=$PWD/synthre/bin/python
STATE=/tmp/round2_05426.state
LOG=/tmp/round2_05426.log
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

for exp in M1 M1noage M2 M2noage M2all M2_simgate M3 M3noage M3allconst \
           M4 M4all M4all_noage M4b M4ball M5 M5all M5all_noage M5all_PL \
           M5all_noQM M5_gate5 M5_simgate K1 K3 K5 K5_noage MK; do
  step "run_$exp" $PY scripts/round2/run_ladder.py "$exp" --parks 05426
done
step score   $PY scripts/round2/score_ladder.py --check-vs-ladder
step wp4     $PY scripts/round2/wp4_fleet_regression.py --experiment M3noage
step tables  $PY scripts/round2/make_summary.py --tables-only
step export  $PY scripts/round2/export_paper_tables.py
step decomp  $PY scripts/round2/error_decomposition.py --rungs M0 M1noage M2noage M3noage M4all M5all K5 M5_simgate
step figs    $PY scripts/round2/regen_paper_figs.py
step tornado $PY scripts/round2/wp6_tornado.py
step morris  $PY scripts/round2/wp6_morris.py
step sobol   $PY scripts/round2/wp6_sobol.py
step sa_copy cp results/round2/morris_ranking.csv results/round2/sobol_indices.csv results/round2/summary/
echo "$(date +%H:%M:%S) 05426 RERUN ALL FINISHED" >> "$LOG"
