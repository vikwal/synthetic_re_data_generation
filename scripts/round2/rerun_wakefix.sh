#!/bin/bash
# Wake staleness fix (2026-07-08): the WP5 wake parquets (Jul 3) were computed
# on the pre-final-gate M3noage wind; the Jul-8 hybrid-gate adoption changed
# the corrected wind of 5 parks (05347 B->A, 01200_1/01200_2 A->C,
# 00282/02483 no-B) but wp5 was not rerun. Sequence: backup old parquets,
# recompute w(t) on the canonical M3noage wind, verify (8 unchanged parks must
# reproduce exactly), rerun the 8 canonical wake-active experiments for the
# 5 affected parks, full downstream. Gate-diagnostic variants
# (M5_gate5/M5_simgate/M5_hybgate) stay frozen as documentation. Idempotent.
set -e
cd /home/viktor/Work/synthetic_re_data_generation
PY=$PWD/synthre/bin/python
STATE=/tmp/round2_wakefix.state
LOG=/tmp/round2_wakefix.log
WAKES=/mnt/nvme2/synthetic/raw/round2/wakes
BACKUP=/mnt/nvme2/synthetic/raw/round2/wakes_stale_jul3
PARKS="05347 01200_1 01200_2 00282 02483"
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

backup_wakes() { [ -d "$BACKUP" ] || cp -a "$WAKES" "$BACKUP"; }
step backup backup_wakes

step wp5 $PY scripts/round2/wp5_precompute_wakes.py \
  --era5-dir /mnt/nvme2/synthetic/raw/wind_era5_v2 --from-experiment M3noage

verify_wakes() {
  $PY - <<'EOF'
import glob, os, sys
import numpy as np, pandas as pd
WAKES = "/mnt/nvme2/synthetic/raw/round2/wakes"
BACKUP = "/mnt/nvme2/synthetic/raw/round2/wakes_stale_jul3"
CHANGED = {"05347", "01200_1", "01200_2", "00282", "02483"}
bad = []
for f in sorted(glob.glob(os.path.join(BACKUP, "w_*_k0.075.parquet"))):
    pid = os.path.basename(f)[len("w_"):-len("_k0.075.parquet")]
    old = pd.read_parquet(f)["w"]
    new = pd.read_parquet(os.path.join(WAKES, os.path.basename(f)))["w"]
    both = pd.concat([old.rename("o"), new.rename("n")], axis=1).dropna()
    d = float((both.o - both.n).abs().max())
    print(f"{pid}: max|dw| old vs new = {d:.6f} "
          f"({'expected-changed' if pid in CHANGED else 'must-match'})")
    if pid in CHANGED and d < 1e-9:
        bad.append(f"{pid}: expected wake change, got none")
    if pid not in CHANGED and d > 1e-9:
        bad.append(f"{pid}: unexpected wake change {d}")
for b in bad:
    print("VERIFY FAIL:", b)
sys.exit(1 if bad else 0)
EOF
}
step verify verify_wakes

for exp in M5 M5all M5all_noage M5all_noQM M5all_PL K5 K5_noage MK; do
  step "run_$exp" $PY scripts/round2/run_ladder.py "$exp" --parks $PARKS
done

step score   $PY scripts/round2/score_ladder.py --check-vs-ladder
step tables  $PY scripts/round2/make_summary.py --tables-only
step export  $PY scripts/round2/export_paper_tables.py
step decomp  $PY scripts/round2/error_decomposition.py --rungs M0 M1noage M2noage M3noage M4all M5all K5 M5_simgate
step figs    $PY scripts/round2/regen_paper_figs.py
step tornado $PY scripts/round2/wp6_tornado.py
step morris  $PY scripts/round2/wp6_morris.py
step sobol   $PY scripts/round2/wp6_sobol.py
step sa_copy cp results/round2/morris_ranking.csv results/round2/sobol_indices.csv results/round2/summary/
step ask25   $PY scripts/round2/ask25_placebo.py --n-perm 1000
echo "$(date +%H:%M:%S) WAKEFIX RERUN FINISHED" >> "$LOG"
