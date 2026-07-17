#!/usr/bin/env bash
# 6W63 4WI：8 个随机种子 × 100 = 800 分子，GPU 1/2/3/5 各跑 2 个 seed（波次内 4 卡并行）
# Usage: bash run_6w63_8seed_800_gpu1235.sh
set -eo pipefail
source /home/user/anaconda3/etc/profile.d/conda.sh
conda activate diffdynamic
export PYTHONPATH=/data/ye/DiffDynamic:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /data/ye/DiffDynamic

ROOT="experiments/6w63_8seed_800_radial_scafbond"
mkdir -p "$ROOT" outputs
LOG="$ROOT/launcher.log"

# 与 8RI2 800 批次相同的 8 个种子（可复现）
SEEDS=(20260716 31415926 27182818 16180339 66260701 14142135 17320508 22360679)
GPUS=(1 2 3 5)

run_one_seed() {
  local seed="$1" gpu="$2"
  local out="${ROOT}/seed_${seed}"
  local cfg="configs/run_6w63_seed_${seed}_800.yml"
  sed "s/^  seed: .*/  seed: ${seed}/" configs/run_6w63_4wi_ligandsize_normal_gen100_prudent_tbr10.yml > "$cfg"
  echo "=== seed=${seed} GPU=${gpu} OUT=${out} START=$(date -Iseconds) ===" | tee -a "$LOG"
  OUT="$out" BASE_SEED="$seed" CFG="$cfg" GPU="$gpu" \
    bash run_6w63_4wi_ligandsize_normal_gen100_prudent_tbr10.sh \
    > "outputs/6w63_seed${seed}_gpu${gpu}.log" 2>&1
  local ec=$?
  echo "=== seed=${seed} GPU=${gpu} DONE exit=${ec} $(date -Iseconds) ===" | tee -a "$LOG"
  return $ec
}

echo "ROOT=$ROOT SEEDS=${SEEDS[*]} GPUS=${GPUS[*]} START=$(date -Iseconds)" | tee "$LOG"

for wave in 0 1; do
  pids=()
  for slot in 0 1 2 3; do
    idx=$((wave * 4 + slot))
    [[ $idx -lt ${#SEEDS[@]} ]] || continue
    seed="${SEEDS[$idx]}"
    gpu="${GPUS[$slot]}"
    run_one_seed "$seed" "$gpu" &
    pids+=($!)
  done
  fail=0
  for pid in "${pids[@]}"; do
    wait "$pid" || fail=1
  done
  [[ $fail -eq 0 ]] || { echo "WAVE $wave FAILED" | tee -a "$LOG"; exit 1; }
  echo "WAVE $wave OK $(date -Iseconds)" | tee -a "$LOG"
done

echo "ALL_800_DONE $(date -Iseconds)" | tee -a "$LOG"
ls -d "$ROOT"/seed_*/run/result_custom_*.pt 2>/dev/null | wc -l | xargs -I{} echo "pt_files={}"
