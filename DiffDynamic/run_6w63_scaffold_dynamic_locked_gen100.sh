#!/usr/bin/env bash
# 6W63：scaffold dynamic_locked 生成 100 → 评估 → n100 出图
# Usage: GPU=5 bash run_6w63_scaffold_dynamic_locked_gen100.sh
set -eo pipefail
source /home/user/anaconda3/etc/profile.d/conda.sh
conda activate diffdynamic
export PYTHONPATH=/data/ye/DiffDynamic:$PYTHONPATH
cd /data/ye/DiffDynamic

DATE_TAG="${DATE_TAG:-$(date +%Y%m%d)}"
GPU="${GPU:-5}"
BASE_SEED="${BASE_SEED:-20260719}"
OUT="${OUT:-experiments/6w63_scaffold_dl_gen100_gpu${GPU}_${DATE_TAG}}"
CFG_SRC="${CFG:-configs/run_6w63_scaffold_dynamic_locked_gen100.yml}"
PROTEIN="/data/ye/protein-ligand/6W63_receptor_nowater_noligand.pdb"
LIGAND="/data/ye/protein-ligand/6W63_ligand_4WI_pose.sdf"

mkdir -p "$OUT"/{run,reconstructed_molecules_n100,plots}
CFG="$OUT/run_config.yml"
cp "$CFG_SRC" "$CFG"
if grep -q '^  seed:' "$CFG"; then
  sed -i "s/^  seed: .*/  seed: ${BASE_SEED}/" "$CFG"
fi
if grep -q '^      seed:' "$CFG"; then
  sed -i "s/^      seed: .*/      seed: ${BASE_SEED}/" "$CFG"
fi

echo "OUT=$OUT GPU=$GPU CFG=$CFG SEED=$BASE_SEED START=$(date -Iseconds)"

echo "========== Generation (scaffold dynamic_locked) =========="
python -u scripts/sample_diffusion.py "$CFG" \
  --protein_path "$PROTEIN" --ligand_path "$LIGAND" \
  --result_path "$OUT/run" --device "cuda:${GPU}" \
  2>&1 | tee "$OUT/generation.log"

echo "========== Evaluation =========="
PT=$(ls -t "$OUT"/run/result_custom_*.pt | head -1)
python split_and_eval_parallel.py \
  --pt_file "$PT" --protein_root /data/ye/protein-ligand \
  --output_dir "$OUT/run" --exhaustiveness 8 --num_workers 8 \
  2>&1 | tee "$OUT/eval_parallel.log"

echo "========== Random extract n=100 (valid Vina) + plot =========="
python3 scripts/select_n100_valid_and_plot.py \
  --out "$OUT" \
  --act /data/ye/protein-ligand/6W63/active_dock_qed_sa.csv \
  --tag 6W63_scaffold_dl_n100 \
  --prefix 6W63_scaffold_dl_n100 \
  --seed "${SELECT_SEED:-$BASE_SEED}" --n 100 \
  2>&1 | tee "$OUT/n100_plot.log"

echo "========== t-SNE + LogP-MW =========="
python3 /data/ye/protein-ligand/plot_generic_results.py \
  --gen_sdf "$OUT/reconstructed_molecules_n100" \
  --gen_csv "$OUT/generated_dock_qed_sa_n100.csv" \
  --act_sdf /data/ye/protein-ligand/6W63/active_ligands/sdf \
  --out_dir "$OUT/plots" \
  --tag 6W63_scaffold_dl_n100 \
  2>&1 | tee -a "$OUT/n100_plot.log"

echo "END=$(date -Iseconds)"
echo "ALL_DONE: $OUT"
ls -la "$OUT"
ls "$OUT/run"/result_custom_*.pt 2>/dev/null | head
ls "$OUT/reconstructed_molecules_n100" | wc -l
ls "$OUT/plots" | head
