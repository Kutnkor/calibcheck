#!/usr/bin/env bash
# Replays the storyboard. To record a terminal session, run it under a recorder of your
# choice (e.g. `asciinema rec -c launch/record_demo.sh`). Uses only local, synthetic data.
set -euo pipefail
OUT="${1:-/tmp/calibcheck-demo}"
rm -rf "$OUT"
calibcheck demo --out "$OUT"
head -3 "$OUT/data/test.csv"
calibcheck inspect "$OUT/data/test.csv" --prob p_model --id row_id --group segment
calibcheck fit "$OUT/data/calibration.csv" --prob p_model --id row_id --method isotonic --out "$OUT/isotonic.json" --overwrite
calibcheck recalibrate --calibration "$OUT/data/calibration.csv" --test "$OUT/data/test.csv" \
  --prob p_model --id row_id --method sigmoid --method isotonic --bootstrap --seed 7 --out "$OUT/recal" --quiet
echo "Open $OUT/recal/report.html"
