#!/usr/bin/env bash
# Keep fetching CAMS points in small batches until all are present; sleep on the ADS quota.
# usage: nohup scripts/pv/fetch_cams_loop.sh [batch=60] [sleep_s=1800] > results/pv/logs/fetch_cams_loop.log 2>&1 &
cd "$(dirname "$0")/../.."
BATCH=${1:-60}; SLEEP=${2:-1800}
while true; do
  while pgrep -f "fetch_cams_parks.py" > /dev/null; do sleep 60; done
  out=$(synthre/bin/python scripts/pv/fetch_cams_parks.py --max-requests "$BATCH" --threads 2 2>&1 | grep -v Warning)
  echo "$(date '+%F %T') $(echo "$out" | head -1) | $(echo "$out" | grep '^result' )"
  if echo "$out" | grep -q "0 to fetch"; then echo "all points fetched"; break; fi
  if echo "$out" | grep -qi "quota"; then sleep "$SLEEP"; else sleep 30; fi
done
