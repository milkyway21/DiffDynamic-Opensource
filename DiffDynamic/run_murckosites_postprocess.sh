#!/usr/bin/env bash
# Post-process murckosites_v1 project: eval + CSV + plots (with active ligands)
# Usage:
#   bash run_murckosites_postprocess.sh           # full: eval + plots
#   bash run_murckosites_postprocess.sh plot_only # replot only (skip eval)
set -eo pipefail
source /home/user/anaconda3/etc/profile.d/conda.sh
conda activate diffdynamic
export PYTHONPATH=/data/ye/DiffDynamic:$PYTHONPATH
cd /data/ye/DiffDynamic

PLOT_ONLY="${1:-}"

check_actives() {
  local CSV="$1" SDF_DIR="$2"
  local n_csv n_sdf
  n_csv=$(tail -n +2 "$CSV" | wc -l)
  n_sdf=$(ls "$SDF_DIR"/*.sdf 2>/dev/null | wc -l)
  if [[ "$n_csv" -eq 0 || "$n_sdf" -eq 0 ]]; then
    echo "ERROR: active CSV rows=$n_csv, SDF count=$n_sdf — fix before plotting"
    return 1
  fi
  echo "OK: $n_csv actives in CSV, $n_sdf SDFs in $SDF_DIR"
}

run_plots() {
  local DIR="$1" TAG="$2" ACTIVE_CSV="$3" ACTIVE_SDF="$4" SCAFFOLD_SDF="$5"
  check_actives "$ACTIVE_CSV" "$ACTIVE_SDF"

  XLSX=$(ls -t "$DIR"/evaluation_results_*.xlsx | head -1)
  if [[ ! -f "$DIR/generated_dock_qed_sa.csv" ]]; then
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
print(f"CSV: {len(out)} rows -> $DIR/generated_dock_qed_sa.csv")
PY
  fi

  mkdir -p "$DIR/plots"
  python /data/ye/protein-ligand/plot_generic_violin.py \
    --gen "$DIR/generated_dock_qed_sa.csv" \
    --act "$ACTIVE_CSV" \
    --out_dir "$DIR/plots" \
    --tag "$TAG"
  python /data/ye/protein-ligand/plot_generic_violin_expanded.py \
    --xlsx "$XLSX" \
    --act "$ACTIVE_CSV" \
    --out_dir "$DIR/plots" \
    --tag "$TAG"
  python /data/ye/protein-ligand/plot_generic_results.py \
    --gen_sdf "$DIR/reconstructed_molecules" \
    --act_sdf "$ACTIVE_SDF" \
    --out_dir "$DIR/plots" \
    --tag "$TAG" \
    --gen_csv "$DIR/generated_dock_qed_sa.csv"

  if [[ -f "$SCAFFOLD_SDF" ]]; then
    python /data/ye/protein-ligand/postprocess_sdf.py \
      --pocket-name "$TAG" \
      --sdf-dir "$DIR/reconstructed_molecules" \
      --active-sdf "$ACTIVE_SDF" \
      --scaffold-sdf "$SCAFFOLD_SDF" \
      --output-dir "$DIR/plots" \
      --top-n 50
  fi
}

run_one() {
  local DIR="$1" TAG="$2" ACTIVE_CSV="$3" ACTIVE_SDF="$4" SCAFFOLD_SDF="$5"
  echo "========== Post-process: $DIR (mode=${PLOT_ONLY:-full}) =========="

  if [[ "$PLOT_ONLY" != "plot_only" ]]; then
    PT=$(ls -t "$DIR"/result_custom_*.pt 2>/dev/null | head -1)
    if [[ -z "$PT" ]]; then echo "ERROR: no .pt in $DIR"; return 1; fi
    echo "PT: $PT"

    python split_and_eval_parallel.py \
      --pt_file "$PT" \
      --protein_root /data/ye/protein-ligand \
      --output_dir "$DIR" \
      --exhaustiveness 8 \
      --num_workers 8 \
      2>&1 | tee "$DIR/eval_parallel.log"
  else
    XLSX=$(ls -t "$DIR"/evaluation_results_*.xlsx 2>/dev/null | head -1)
    if [[ -z "$XLSX" ]]; then echo "ERROR: no evaluation_results_*.xlsx in $DIR"; return 1; fi
    echo "Replot only, using: $XLSX"
  fi

  run_plots "$DIR" "$TAG" "$ACTIVE_CSV" "$ACTIVE_SDF" "$SCAFFOLD_SDF"
  echo "DONE: $DIR"
}

run_one "experiments/6w63prudent_murckosites_v1" "6W63_murckosites_v1" \
  "/data/ye/protein-ligand/6W63/active_dock_qed_sa.csv" \
  "/data/ye/protein-ligand/6W63/active_ligands/sdf" \
  "experiments/6w63prudent_murckosites_v1/scaffold/6W63_murcko_scaffold.sdf"

run_one "experiments/8ri2prudent_murckosites_v1" "8RI2_murckosites_v1" \
  "/data/ye/protein-ligand/8RI2/active_dock_qed_sa.csv" \
  "/data/ye/protein-ligand/8RI2/active_ligands/sdf" \
  "experiments/8ri2prudent_murckosites_v1/scaffold/8RI2_murcko_scaffold.sdf"
