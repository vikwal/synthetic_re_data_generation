#!/bin/bash
# Morris + Sobol refresh after the 00198_1 commissioning-date fix
# (age 9.67 -> 10.25 y) so the SA numbers match the corrected-age state.
set -e
cd /home/viktor/Work/synthetic_re_data_generation
PY=$PWD/synthre/bin/python
STATE=/tmp/round2_sa00198.state
LOG=/tmp/round2_sa00198.log
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

step morris $PY scripts/round2/wp6_morris.py
step sobol  $PY scripts/round2/wp6_sobol.py
step copy   cp results/round2/morris_ranking.csv results/round2/sobol_indices.csv results/round2/summary/
step fig    $PY -c "
import sys, matplotlib
matplotlib.use('Agg')
sys.path.insert(0, '.'); sys.path.insert(0, 'scripts/round2')
from round2.paper_style import apply_print_style
apply_print_style()
import regen_paper_figs as r
r.fig_morris_sobol()
print('morris_mu_star regeneriert')
"
echo "$(date +%H:%M:%S) SA REFRESH FINISHED" >> "$LOG"
