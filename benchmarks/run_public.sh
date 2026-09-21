#!/usr/bin/env bash
# Reproduces the public-benchmark tables in the README.
#   bash benchmarks/run_public.sh                       # default model, 200 rows per window
#   MODEL=mlx-community/Qwen3-0.6B-8bit N=100 bash benchmarks/run_public.sh
# Each dataset is shuffled with a fixed seed and split into two disjoint windows:
# rows [0, N) are the test set, rows [N, 2N) are only used to fit the calibration profile.
set -euo pipefail

MODEL=${MODEL:-mlx-community/Qwen3-1.7B-8bit}
N=${N:-200}
OUT=${OUT:-results/public-$(basename "$MODEL")}
DATASETS=${DATASETS:-"boolq sst5 agnews arc banking77"}
LD=${LD:-localdecision}

mkdir -p data "$OUT"
for ds in $DATASETS; do
  [ -f "data/$ds-test.jsonl" ] || $LD fetch "$ds" --limit "$N" -o "data/$ds-test.jsonl"
  [ -f "data/$ds-calib.jsonl" ] || $LD fetch "$ds" --limit "$N" --skip "$N" -o "data/$ds-calib.jsonl"
done

for ds in $DATASETS; do
  $LD eval "data/$ds-test.jsonl" --model "$MODEL" --debias none,auto --output "$OUT/$ds-test"
  $LD eval "data/$ds-calib.jsonl" --model "$MODEL" --debias auto --output "$OUT/$ds-calib"
done

$LD calibrate "$OUT"/*-calib/records-auto.jsonl -o "$OUT/calibration.json" --alpha 0.1
echo "== uncalibrated (auto) =="
$LD report "$OUT"/*-test/records-auto.jsonl
echo "== calibrated on the disjoint window =="
$LD report "$OUT"/*-test/records-auto.jsonl --calibration "$OUT/calibration.json"
