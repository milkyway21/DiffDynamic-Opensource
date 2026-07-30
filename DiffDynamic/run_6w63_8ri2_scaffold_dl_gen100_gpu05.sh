#!/usr/bin/env bash
# GPU0 → 8RI2，GPU5 → 6W63：scaffold dynamic_locked ×100 并行
# Usage: bash run_6w63_8ri2_scaffold_dl_gen100_gpu05.sh
set -eo pipefail
cd /data/ye/DiffDynamic
DATE_TAG="${DATE_TAG:-$(date +%Y%m%d)}"
BASE_SEED="${BASE_SEED:-20260719}"
mkdir -p outputs

LOG8="outputs/8ri2_scaffold_dl_gen100_gpu0_${DATE_TAG}.log"
LOG6="outputs/6w63_scaffold_dl_gen100_gpu5_${DATE_TAG}.log"

echo "LAUNCH parallel DATE=$DATE_TAG SEED=$BASE_SEED START=$(date -Iseconds)"
GPU=0 BASE_SEED="$BASE_SEED" DATE_TAG="$DATE_TAG" \
  bash run_8ri2_scaffold_dynamic_locked_gen100.sh > "$LOG8" 2>&1 &
PID8=$!
GPU=5 BASE_SEED="$BASE_SEED" DATE_TAG="$DATE_TAG" \
  bash run_6w63_scaffold_dynamic_locked_gen100.sh > "$LOG6" 2>&1 &
PID6=$!

echo "8RI2 pid=$PID8 log=$LOG8"
echo "6W63 pid=$PID6 log=$LOG6"

EC8=0
EC6=0
wait "$PID8" || EC8=$?
wait "$PID6" || EC6=$?
echo "DONE 8RI2_exit=$EC8 6W63_exit=$EC6 END=$(date -Iseconds)"
[[ "$EC8" -eq 0 && "$EC6" -eq 0 ]]
