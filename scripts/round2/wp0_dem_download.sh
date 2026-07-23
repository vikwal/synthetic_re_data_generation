#!/bin/bash
# WP0.4 — Copernicus DEM GLO-30 tiles for Germany (public S3, no auth).
# Germany bbox: lat 47..55, lon 5..15 -> 1x1 degree tiles.
set -u
OUT=/mnt/nvme2/synthetic/raw/round2/dem/glo30
mkdir -p "$OUT"
BASE="https://copernicus-dem-30m.s3.amazonaws.com"
ok=0; miss=0
for lat in $(seq 47 55); do
  for lon in $(seq 5 15); do
    LON=$(printf "E%03d" "$lon")
    NAME="Copernicus_DSM_COG_10_N${lat}_00_${LON}_00_DEM"
    DEST="$OUT/${NAME}.tif"
    if [ -s "$DEST" ]; then ok=$((ok+1)); continue; fi
    URL="$BASE/${NAME}/${NAME}.tif"
    if curl -sfL --retry 3 -o "${DEST}.part" "$URL"; then
      mv "${DEST}.part" "$DEST"; ok=$((ok+1))
    else
      rm -f "${DEST}.part"; miss=$((miss+1)); echo "MISS $NAME"
    fi
  done
done
echo "tiles ok=$ok missing=$miss"
ls -la "$OUT" | head -3
du -sh "$OUT"
