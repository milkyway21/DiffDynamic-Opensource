#!/usr/bin/env bash
# 6W63：10 批 × 100 = 1000，GPU3 顺序；每批独立 BASE_SEED
# Usage: bash run_6w63_10batch_1000_gpu3.sh
set -eo pipefail
source /home/user/anaconda3/etc/profile.d/conda.sh
conda activate diffdynamic
export PYTHONPATH=/data/ye/DiffDynamic:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /data/ye/DiffDynamic

ROOT="experiments/6w63_10batch_1000_finalfill"
mkdir -p "$ROOT" outputs
LOG="$ROOT/launcher.log"
GPU=3
# 与 8RI2 同一组种子，便于跨口袋对照；每批仍独立 OUT
SEEDS=(20260719 31415927 27182819 16180340 66260702 14142136 17320509 22360680 11235813 39817321)

echo "ROOT=$ROOT GPU=$GPU SEEDS=${SEEDS[*]} START=$(date -Iseconds)" | tee "$LOG"

for i in "${!SEEDS[@]}"; do
  seed="${SEEDS[$i]}"
  batch=$((i + 1))
  out="${ROOT}/batch$(printf '%02d' "$batch")_seed_${seed}"
  cfg="configs/run_6w63_batch${batch}_seed_${seed}_1000.yml"
  sed "s/^  seed: .*/  seed: ${seed}/" configs/run_6w63_4wi_ligandsize_normal_gen100_prudent_tbr10.yml > "$cfg"
  if grep -q '^      seed:' "$cfg"; then
    sed -i "s/^      seed: .*/      seed: ${seed}/" "$cfg"
  fi
  echo "=== batch=${batch}/10 seed=${seed} GPU=${GPU} OUT=${out} START=$(date -Iseconds) ===" | tee -a "$LOG"
  OUT="$out" BASE_SEED="$seed" CFG="$cfg" GPU="$GPU" \
    bash run_6w63_4wi_ligandsize_normal_gen100_prudent_tbr10.sh \
    > "outputs/6w63_b${batch}_seed${seed}_gpu${GPU}.log" 2>&1
  ec=$?
  echo "=== batch=${batch}/10 seed=${seed} DONE exit=${ec} $(date -Iseconds) ===" | tee -a "$LOG"
  [[ $ec -eq 0 ]] || { echo "BATCH ${batch} FAILED" | tee -a "$LOG"; exit "$ec"; }
done

echo "ALL_1000_DONE $(date -Iseconds)" | tee -a "$LOG"
ls -d "$ROOT"/batch*/run/result_custom_*.pt 2>/dev/null | wc -l | xargs -I{} echo "pt_files={}"
