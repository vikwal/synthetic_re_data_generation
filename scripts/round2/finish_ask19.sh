#!/bin/bash
# Detached ask-19 job: clean-ladder runs (M1noage, M2all), scoring, export,
# and the three ladder-based figures. Idempotent via state file.
set -e
cd /home/viktor/Work/synthetic_re_data_generation
PY=$PWD/synthre/bin/python
STATE=/tmp/round2_ask19.state
LOG=/tmp/round2_ask19.log
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

step run_M1noage $PY scripts/round2/run_ladder.py M1noage
step run_M2all   $PY scripts/round2/run_ladder.py M2all
step score       $PY scripts/round2/score_ladder.py --experiments M1noage M2all
step export      $PY scripts/round2/export_paper_tables.py
step figs        $PY -c "
import sys, matplotlib
matplotlib.use('Agg')
sys.path.insert(0, '.'); sys.path.insert(0, 'scripts/round2')
from round2.paper_style import apply_print_style
apply_print_style()
import regen_paper_figs as r
sc = r.scores('clean')
r.fig_bars_round1_style(sc)
r.fig_ladder_spaghetti(sc)
r.fig_effect_matrix(sc)
print('bars_r2, ladder_spaghetti, effect_matrix regeneriert')
"
echo "$(date +%H:%M:%S) ASK19 ALL FINISHED" >> "$LOG"
