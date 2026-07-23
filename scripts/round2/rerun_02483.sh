#!/bin/bash
# Detached rerun after the 02483 commissioning-date fix (2015-12-01).
# Aging-active rungs for park 02483 only, then full rescoring + downstream
# (wp4/SA/exports/figures) via the idempotent finisher with cleared state.
set -e
cd /home/viktor/Work/synthetic_re_data_generation
PY=$PWD/synthre/bin/python
LOG=/tmp/round2_02483.log
echo "$(date +%H:%M:%S) start 02483 rerun" > "$LOG"

for exp in M1 M2 M3 M4 M4b M5 M3allconst M4all M4ball M5all M5all_PL M5all_noQM K1 K3 K5 MK; do
  $PY scripts/round2/run_ladder.py "$exp" --parks 02483 >> "$LOG" 2>&1
  echo "$(date +%H:%M:%S) $exp done" >> "$LOG"
done

$PY scripts/round2/score_ladder.py --check-vs-ladder >> "$LOG" 2>&1
echo "$(date +%H:%M:%S) scoring done" >> "$LOG"
$PY scripts/round2/score_m0.py >> "$LOG" 2>&1
echo "$(date +%H:%M:%S) m0 done" >> "$LOG"

rm -f /tmp/round2_finish.state
bash scripts/round2/finish_ask12_13.sh
echo "$(date +%H:%M:%S) 02483 RERUN ALL FINISHED" >> "$LOG"
