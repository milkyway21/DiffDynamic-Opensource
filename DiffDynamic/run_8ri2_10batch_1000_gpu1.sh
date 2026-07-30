#!/usr/bin/env bash
# 8RI2：10 批 × 100 = 1000，GPU1 顺序；每批独立 BASE_SEED
# Usage: bash run_8ri2_10batch_1000_gpu1.sh
set -eo pipefail
source /home/user/anaconda3/etc/profile.d/conda.sh
conda activate diffdynamic
export PYTHONPATH=/data/ye/DiffDynamic:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /data/ye/DiffDynamic

ROOT="experiments/8ri2_10batch_1000_finalfill"
mkdir -p "$ROOT" outputs
LOG="$ROOT/launcher.log"
GPU=1
SEEDS=(20260719 31415927 27182819 16180340 66260702 14142136 17320509 22360680 11235813 39817321)

echo "ROOT=$ROOT GPU=$GPU SEEDS=${SEEDS[*]} START=$(date -Iseconds)" | tee "$LOG"

for i in "${!SEEDS[@]}"; do
  seed="${SEEDS[$i]}"
  batch=$((i + 1))
  out="${ROOT}/batch$(printf '%02d' "$batch")_seed_${seed}"
  cfg="configs/run_8ri2_batch${batch}_seed_${seed}_1000.yml"
  sed "s/^  seed: .*/  seed: ${seed}/" configs/run_8ri2_ligand_instant_gen100_prudent_tbr10.yml > "$cfg"
  # 同步 grow.seed
  if grep -q '^      seed:' "$cfg"; then
    sed -i "s/^      seed: .*/      seed: ${seed}/" "$cfg"
  fi
  echo "=== batch=${batch}/10 seed=${seed} GPU=${GPU} OUT=${out} START=$(date -Iseconds) ===" | tee -a "$LOG"
  OUT="$out" BASE_SEED="$seed" CFG="$cfg" GPU="$GPU" \
    bash run_8ri2_ligand_instant_gen100_prudent_tbr10.sh \
    > "outputs/8ri2_b${batch}_seed${seed}_gpu${GPU}.log" 2>&1
  ec=$?
  echo "=== batch=${batch}/10 seed=${seed} DONE exit=${ec} $(date -Iseconds) ===" | tee -a "$LOG"
  [[ $ec -eq 0 ]] || { echo "BATCH ${batch} FAILED" | tee -a "$LOG"; exit "$ec"; }
done

echo "ALL_1000_DONE $(date -Iseconds)" | tee -a "$LOG"
ls -d "$ROOT"/batch*/run/result_custom_*.pt 2>/dev/null | wc -l | xargs -I{} echo "pt_files={}"
