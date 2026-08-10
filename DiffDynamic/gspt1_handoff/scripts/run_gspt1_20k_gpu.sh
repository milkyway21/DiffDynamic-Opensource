#!/usr/bin/env bash
# GSPT1 60000 分子批量生成：600 个种子 x 100 分子/种子 = 60000
# 三 GPU 并行：GPU3(seed 20270000-20270199) GPU4(20270200-20270399) GPU5(20270400-20270599)
# Usage:
#   GPU=3 JOB_START=0 JOB_END=199 bash run_gspt1_20k_gpu.sh
#   GPU=4 JOB_START=200 JOB_END=399 bash run_gspt1_20k_gpu.sh
#   GPU=5 JOB_START=400 JOB_END=599 bash run_gspt1_20k_gpu.sh
set -eo pipefail

GPU="${GPU:?GPU required (3|4|5)}"
TARGET=gspt1
JOB_START="${JOB_START:?JOB_START required}"
JOB_END="${JOB_END:?JOB_END required}"
N_JOBS=$((JOB_END - JOB_START + 1))
ROOT="/data/zhang/Ye/DiffDynamic_outputs/hsvpol/molglue_ikzf2_gspt1/diffdynamic/dl_gspt1_60k"
SCRIPT_DIR="/home/user/Desktop/Ye/DiffDynamic/hsvpol/molglue_ikzf2_gspt1/scripts"
CAMPAIGN_LOG="$ROOT/$TARGET/campaign_gpu${GPU}.log"

export NUM_SAMPLES=100
export ROOT
export GPU
export TARGET

# 每个 GPU 独立 BASE_SEED，确保 600 个种子全部不同
export BASE_SEED=20270000

mkdir -p "$ROOT/$TARGET/jobs"

echo "=== GSPT1 Campaign GPU${GPU} start $(date -Iseconds) JOB $JOB_START-$JOB_END ===" | tee "$CAMPAIGN_LOG"

DONE_COUNT=0
FAIL_COUNT=0
for JOB_ID in $(seq $JOB_START $JOB_END); do
  JOB_TAG=$(printf 'job_%04d' "$JOB_ID")
  echo ">>> [$(date +%H:%M:%S)] GPU${GPU} Job $JOB_TAG ($((JOB_ID - JOB_START + 1))/$N_JOBS) ..." | tee -a "$CAMPAIGN_LOG"
  if JOB_ID="$JOB_ID" bash "$SCRIPT_DIR/run_scaffold_fast_100k_one_job.sh" 2>&1 | tee -a "$CAMPAIGN_LOG"; then
    DONE_COUNT=$((DONE_COUNT + 1))
  else
    FAIL_COUNT=$((FAIL_COUNT + 1))
    echo "!!! GPU${GPU} Job $JOB_TAG FAILED" | tee -a "$CAMPAIGN_LOG"
  fi
  echo ">>> GPU${GPU} Progress: done=$DONE_COUNT fail=$FAIL_COUNT / $N_JOBS" | tee -a "$CAMPAIGN_LOG"
done

echo "=== GSPT1 Campaign GPU${GPU} DONE $(date -Iseconds) success=$DONE_COUNT fail=$FAIL_COUNT ===" | tee -a "$CAMPAIGN_LOG"
