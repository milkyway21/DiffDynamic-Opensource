#!/usr/bin/env bash
# GSPT1 single-job reconstruction; Vina is explicitly disabled.
# Usage:
#   TARGET=gspt1 JOB_ID=0 bash scripts/reconstruct_gspt1_scaffold_job.sh
set -eo pipefail
source /home/user/anaconda3/etc/profile.d/conda.sh
set +u
conda activate diffdynamic
export PYTHONPATH="/data/ye/DiffDynamic${PYTHONPATH:+:$PYTHONPATH}"
# Three scaffold jobs run concurrently on the 80-core host.  Keep the total
# evaluator fan-out near the host capacity while leaving headroom for Python.
export EVAL_PARALLEL_WORKERS="${EVAL_PARALLEL_WORKERS:-24}"
set -u
cd /data/ye/DiffDynamic

ROOT="${ROOT:-/data/zhang/Ye/DiffDynamic_outputs/hsvpol/molglue_ikzf2_gspt1/diffdynamic/gspt1_scaffold_exact_loop}"
TARGET="${TARGET:?TARGET required: ikzf2|gspt1}"
JOB_ID="${JOB_ID:?JOB_ID required}"

STRUCT_ROOT="/data/ye/e-drug-lab/Scientist_In_E-Drug-Lab/runs/molecular_glue_structures_ikzf2_gspt1_20260803_235811"
case "$TARGET" in
  ikzf2)
    PROTEIN="$STRUCT_ROOT/01_ikzf2_molecular_glue/receptor/7U8F_receptor_clean.pdb"
    LIGAND="$STRUCT_ROOT/01_ikzf2_molecular_glue/ligand/7U8F_LWK_D_502_native.sdf"
    ;;
  gspt1)
    PROTEIN="$STRUCT_ROOT/02_gspt1_or_gstp1_molecular_glue/receptor/5HXB_receptor_clean.pdb"
    LIGAND="$STRUCT_ROOT/02_gspt1_or_gstp1_molecular_glue/ligand/5HXB_85C_C_502_native.sdf"
    ;;
  *)
    echo "ERROR: TARGET must be ikzf2 or gspt1, got: $TARGET" >&2
    exit 1
    ;;
esac

JOB_TAG=$(printf 'job_%04d' "$JOB_ID")
JDIR="$ROOT/$TARGET/jobs/$JOB_TAG"
OUT="$ROOT/$TARGET/extract_cleaned/$JOB_TAG"
MARK="$OUT/.extract_done"
PROTEIN_ROOT="$(dirname "$LIGAND")"
REF_LIG="$(basename "$LIGAND")"

if [[ ! -f "$JDIR/.sample_done" ]]; then
  echo "[SKIP-NO-SAMPLE] $TARGET/$JOB_TAG"
  exit 0
fi
if [[ -f "$MARK" ]]; then
  echo "[SKIP] $TARGET/$JOB_TAG extract done"
  exit 0
fi

PT=""
if [[ -f "$JDIR/final_pt.txt" ]]; then
  PT=$(head -1 "$JDIR/final_pt.txt" | tr -d '\r\n')
fi
if [[ -z "$PT" || ! -s "$PT" ]]; then
  PT=$(find "$JDIR/run" -maxdepth 1 -name 'result_custom_*.pt' \
    ! -name '*_seed*_gen*' ! -name '*_chains*' -type f 2>/dev/null | head -1 || true)
fi
if [[ -z "$PT" || ! -s "$PT" ]]; then
  echo "[MISS-PT] $TARGET/$JOB_TAG" >&2
  exit 1
fi

mkdir -p "$OUT"
echo "[RUN] $TARGET/$JOB_TAG PT=$PT"
echo "[POLICY] Vina disabled: --vina-modes none"
echo "[POLICY] EVAL_PARALLEL_WORKERS=$EVAL_PARALLEL_WORKERS"
python -u evaluate_pt_with_correct_reconstruct.py "$PT" \
  --vina-modes none \
  --receptor_pdb "$PROTEIN" \
  --protein_root "$PROTEIN_ROOT" \
  --reference_ligand "$REF_LIG" \
  --output_dir "$OUT" \
  --enable_isolation \
  --no-distribution-plots \
  > "$OUT/extract.log" 2>&1

N_SDF=$(find "$OUT" -name '*.sdf' -type f 2>/dev/null | wc -l)
if [[ "$N_SDF" -lt 1 ]]; then
  echo "[ERROR] $TARGET/$JOB_TAG no SDF after extract; see $OUT/extract.log" >&2
  exit 1
fi

date -Iseconds > "$MARK"
echo "$PT" > "$OUT/final_pt.txt"
echo "[OK] $TARGET/$JOB_TAG sdf=$N_SDF"
