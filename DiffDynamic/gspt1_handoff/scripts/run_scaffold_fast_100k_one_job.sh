#!/usr/bin/env bash
# 单 job：Fast 自定义骨架 × num_samples=100；n_extra 10–25；仅生成
# Usage:
#   TARGET=ikzf2 JOB_ID=0 GPU=1 bash scripts/run_scaffold_fast_100k_one_job.sh
set -eo pipefail
source /home/user/anaconda3/etc/profile.d/conda.sh
set +u
conda activate diffdynamic
export PYTHONPATH="/data/ye/DiffDynamic${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -u
cd /data/ye/DiffDynamic

ROOT="${ROOT:-/data/zhang/Ye/DiffDynamic_outputs/hsvpol/molglue_ikzf2_gspt1/diffdynamic/scaffold_fast_100k}"
CFG_SRC="${CFG_SRC:-/home/user/Desktop/Ye/DiffDynamic/hsvpol/molglue_ikzf2_gspt1/configs/sampling_fast_scaffold_custom_warhead_nextra10_25.yml}"
BASE_SEED="${BASE_SEED:-20260804}"
GPU="${GPU:?GPU required}"
JOB_ID="${JOB_ID:?JOB_ID required (0..999)}"
TARGET="${TARGET:?TARGET required: ikzf2|gspt1}"
SEED=$((BASE_SEED + JOB_ID))

STRUCT_ROOT="/data/ye/e-drug-lab/Scientist_In_E-Drug-Lab/runs/molecular_glue_structures_ikzf2_gspt1_20260803_235811"
case "$TARGET" in
  ikzf2)
    PROTEIN="${PROTEIN:-$STRUCT_ROOT/01_ikzf2_molecular_glue/receptor/7U8F_receptor_clean.pdb}"
    LIGAND="${LIGAND:-$STRUCT_ROOT/01_ikzf2_molecular_glue/ligand/7U8F_LWK_D_502_native.sdf}"
    ;;
  gspt1)
    PROTEIN="${PROTEIN:-$STRUCT_ROOT/02_gspt1_or_gstp1_molecular_glue/receptor/5HXB_receptor_clean.pdb}"
    LIGAND="${LIGAND:-$STRUCT_ROOT/02_gspt1_or_gstp1_molecular_glue/ligand/5HXB_85C_C_502_native.sdf}"
    ;;
  *)
    echo "ERROR: TARGET must be ikzf2 or gspt1, got: $TARGET" >&2
    exit 1
    ;;
esac

JOB_TAG=$(printf 'job_%04d' "$JOB_ID")
JOB_DIR="$ROOT/$TARGET/jobs/$JOB_TAG"
RUN_DIR="$JOB_DIR/run"
LOG_DIR="$JOB_DIR/logs"
CFG_DIR="$JOB_DIR/configs"
DONE_MARK="$JOB_DIR/.sample_done"

mkdir -p "$RUN_DIR" "$LOG_DIR" "$CFG_DIR"

FINAL_PT=$(find "$RUN_DIR" -maxdepth 1 -name 'result_custom_*.pt' ! -name '*_seed*_gen*' ! -name '*_chains*' -type f 2>/dev/null | head -1 || true)
if [[ -f "$DONE_MARK" && -n "$FINAL_PT" && -s "$FINAL_PT" ]]; then
  echo "[SKIP] $TARGET/$JOB_TAG already done: $FINAL_PT"
  exit 0
fi

CFG="$CFG_DIR/run_scaffold_seed_${SEED}.yml"
cp "$CFG_SRC" "$CFG"
sed -i "s/^  seed: .*/  seed: ${SEED}/" "$CFG"
if [[ -n "${NUM_SAMPLES:-}" ]]; then
  python3 - "$CFG" "$NUM_SAMPLES" <<'PY'
import sys
from pathlib import Path
cfg, n = Path(sys.argv[1]), sys.argv[2]
lines = cfg.read_text().splitlines(True)
out, in_sc, done = [], False, False
for line in lines:
    if line.startswith('  scaffold:'):
        in_sc = True
    elif in_sc and line[:2] == '  ' and not line.startswith('    ') and line.strip() and not line.strip().startswith('#'):
        # next top-level key under sample (2-space indent, not 4)
        if line.startswith('  ') and not line.startswith('    '):
            in_sc = False
    if in_sc and (not done) and line.startswith('    num_samples:'):
        line = f'    num_samples: {n}\n'
        done = True
    out.append(line)
cfg.write_text(''.join(out))
print(f'NUM_SAMPLES override -> {n} ok={done}')
PY
fi

LOG="$LOG_DIR/generation_gpu${GPU}_seed${SEED}.log"
echo "=== $TARGET/$JOB_TAG START $(date -Iseconds) GPU=$GPU SEED=$SEED ===" | tee "$LOG"
echo "PROTEIN=$PROTEIN" | tee -a "$LOG"
echo "LIGAND=$LIGAND" | tee -a "$LOG"
echo "CFG=$CFG" | tee -a "$LOG"

python -u scripts/sample_diffusion.py "$CFG" \
  --protein_path "$PROTEIN" \
  --ligand_path "$LIGAND" \
  --result_path "$RUN_DIR" \
  --device "cuda:${GPU}" \
  >> "$LOG" 2>&1

FINAL_PT=$(find "$RUN_DIR" -maxdepth 1 -name 'result_custom_*.pt' ! -name '*_seed*_gen*' ! -name '*_chains*' -type f 2>/dev/null | head -1 || true)
if [[ -z "$FINAL_PT" || ! -s "$FINAL_PT" ]]; then
  echo "[ERROR] $TARGET/$JOB_TAG no final result pt" | tee -a "$LOG"
  exit 1
fi

python - <<PY >> "$LOG" 2>&1
import torch
from pathlib import Path
pt = Path("$FINAL_PT")
obj = torch.load(pt, map_location="cpu", weights_only=False)
n = None
if isinstance(obj, dict):
    for k in ("pred_ligand_pos", "pred_pos", "pos"):
        if k in obj and obj[k] is not None:
            try:
                n = len(obj[k])
            except TypeError:
                n = None
            print(f"key={k} n={n}")
            break
print("FINAL_N=", n)
print("PT=", pt)
PY

date -Iseconds > "$DONE_MARK"
echo "$FINAL_PT" > "$JOB_DIR/final_pt.txt"
echo "=== $TARGET/$JOB_TAG DONE $(date -Iseconds) PT=$FINAL_PT ===" | tee -a "$LOG"
