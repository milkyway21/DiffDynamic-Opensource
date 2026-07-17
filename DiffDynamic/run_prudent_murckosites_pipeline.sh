#!/usr/bin/env bash
# Prudent murckosites production: generate + eval + active-ligand plots
# Usage:
#   bash run_prudent_murckosites_pipeline.sh [JOB ...]
#   JOB: 6w63 | 8ri2 | all (default: all)
#   VERSION=v2 (default)
set -eo pipefail
source /home/user/anaconda3/etc/profile.d/conda.sh
conda activate diffdynamic
export PYTHONPATH=/data/ye/DiffDynamic:$PYTHONPATH
cd /data/ye/DiffDynamic

JOBS="${*:-all}"
VERSION="${VERSION:-v2}"

pick_gpu() {
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | sort -t, -k2 -nr | head -1 | cut -d, -f1 | tr -d ' '
}

GPU_6W63="${GPU_6W63:-$(pick_gpu)}"
GPU_8RI2="${GPU_8RI2:-$(pick_gpu)}"
if [[ "$GPU_6W63" == "$GPU_8RI2" ]]; then
  GPU_8RI2=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | sort -t, -k2 -nr | sed -n '2p' | cut -d, -f1 | tr -d ' ')
fi
echo "VERSION=$VERSION | GPUs: 6W63=$GPU_6W63, 8RI2=$GPU_8RI2"

check_actives() {
  local CSV="$1" SDF_DIR="$2"
  local n_csv n_sdf
  n_csv=$(tail -n +2 "$CSV" | wc -l)
  n_sdf=$(ls "$SDF_DIR"/*.sdf 2>/dev/null | wc -l)
  if [[ "$n_csv" -eq 0 || "$n_sdf" -eq 0 ]]; then
    echo "ERROR: active CSV rows=$n_csv, SDF count=$n_sdf"
    return 1
  fi
  echo "OK: $n_csv actives, $n_sdf SDFs"
}

resolve_scaffold() {
  local DIR="$1"
  local SCAFFOLD
  SCAFFOLD=$(ls -t "$DIR"/scaffold/*murcko*.sdf "$DIR"/scaffold/*.sdf 2>/dev/null | head -1)
  if [[ -n "$SCAFFOLD" && -s "$SCAFFOLD" ]]; then
    echo "$SCAFFOLD"
    return
  fi
  if [[ "$DIR" == *6w63* ]]; then
    echo "experiments/6w63prudent_murckosites_v1/scaffold/6W63_murcko_scaffold.sdf"
  elif [[ "$DIR" == *8ri2* ]]; then
    echo "experiments/8ri2prudent_murckosites_v1/scaffold/8RI2_murcko_scaffold.sdf"
  fi
}

run_plots() {
  local DIR="$1" TAG="$2" ACTIVE_CSV="$3" ACTIVE_SDF="$4"
  check_actives "$ACTIVE_CSV" "$ACTIVE_SDF"
  XLSX=$(ls -t "$DIR"/evaluation_results_*.xlsx | head -1)
  python3 - <<PY
import pandas as pd
from pathlib import Path
xlsx = Path("$XLSX")
df = pd.read_excel(xlsx, sheet_name="评估结果")
out = pd.DataFrame({
    "smiles": df["SMILES"],
    "vina_dock": df["Vina_Dock_亲和力"],
    "qed": df["QED评分"],
    "sa": df["SA评分"],
    "logp": df["logP"],
    "molwt": df["分子量"],
})
out.to_csv(Path("$DIR") / "generated_dock_qed_sa.csv", index=False)
print(f"CSV: {len(out)} rows")
PY
  mkdir -p "$DIR/plots"
  python /data/ye/protein-ligand/plot_generic_violin.py \
    --gen "$DIR/generated_dock_qed_sa.csv" --act "$ACTIVE_CSV" \
    --out_dir "$DIR/plots" --tag "$TAG"
  python /data/ye/protein-ligand/plot_generic_violin_expanded.py \
    --xlsx "$XLSX" --act "$ACTIVE_CSV" \
    --out_dir "$DIR/plots" --tag "$TAG"
  python /data/ye/protein-ligand/plot_generic_results.py \
    --gen_sdf "$DIR/reconstructed_molecules" --act_sdf "$ACTIVE_SDF" \
    --out_dir "$DIR/plots" --tag "$TAG" --gen_csv "$DIR/generated_dock_qed_sa.csv"
  SCAFFOLD=$(resolve_scaffold "$DIR")
  if [[ -n "$SCAFFOLD" && -f "$SCAFFOLD" ]]; then
    python /data/ye/protein-ligand/postprocess_sdf.py \
      --pocket-name "$TAG" --sdf-dir "$DIR/reconstructed_molecules" \
      --active-sdf "$ACTIVE_SDF" --scaffold-sdf "$SCAFFOLD" \
      --output-dir "$DIR/plots" --top-n 50
  fi
}

run_job() {
  local KEY="$1" DIR="$2" CONFIG="$3" TAG="$4" PROTEIN="$5" LIGAND="$6" \
        ACTIVE_CSV="$7" ACTIVE_SDF="$8" GPU="$9"
  echo "========== $KEY -> $DIR (GPU $GPU) =========="
  mkdir -p "$DIR"
  python -u scripts/sample_diffusion.py "$CONFIG" \
    --protein_path "$PROTEIN" --ligand_path "$LIGAND" \
    --result_path "$DIR" --device "cuda:${GPU}" \
    2>&1 | tee "$DIR/generation.log"
  PT=$(ls -t "$DIR"/result_custom_*.pt | head -1)
  python split_and_eval_parallel.py \
    --pt_file "$PT" --protein_root /data/ye/protein-ligand \
    --output_dir "$DIR" --exhaustiveness 8 --num_workers 8 \
    2>&1 | tee "$DIR/eval_parallel.log"
  run_plots "$DIR" "$TAG" "$ACTIVE_CSV" "$ACTIVE_SDF"
  echo "DONE: $DIR"
}

want() {
  [[ "$JOBS" == "all" || " $JOBS " == *" $1 "* ]]
}

P6W63_PROTEIN="/data/ye/protein-ligand/6W63_receptor_nowater_noligand.pdb"
P6W63_LIGAND="/data/ye/protein-ligand/6W63_ligand_X77_pose.sdf"
P6W63_ACT_CSV="/data/ye/protein-ligand/6W63/active_dock_qed_sa.csv"
P6W63_ACT_SDF="/data/ye/protein-ligand/6W63/active_ligands/sdf"

P8RI2_PROTEIN="/data/ye/protein-ligand/8RI2/8RI2_apo.pdb"
P8RI2_LIGAND="/data/ye/protein-ligand/8RI2/8ri2ligand.sdf"
P8RI2_ACT_CSV="/data/ye/protein-ligand/8RI2/active_dock_qed_sa.csv"
P8RI2_ACT_SDF="/data/ye/protein-ligand/8RI2/active_ligands/sdf"

pids=()
fail=0

if want 6w63; then
  run_job 6w63 "experiments/6w63prudent_murckosites_${VERSION}" configs/run_6w63_prudent_murckosites.yml \
    "6W63_murckosites_${VERSION}" "$P6W63_PROTEIN" "$P6W63_LIGAND" \
    "$P6W63_ACT_CSV" "$P6W63_ACT_SDF" "$GPU_6W63" &
  pids+=($!)
fi

if want 8ri2; then
  run_job 8ri2 "experiments/8ri2prudent_murckosites_${VERSION}" configs/run_8ri2_prudent_murckosites.yml \
    "8RI2_murckosites_${VERSION}" "$P8RI2_PROTEIN" "$P8RI2_LIGAND" \
    "$P8RI2_ACT_CSV" "$P8RI2_ACT_SDF" "$GPU_8RI2" &
  pids+=($!)
fi

for pid in "${pids[@]}"; do wait "$pid" || fail=1; done
exit $fail
