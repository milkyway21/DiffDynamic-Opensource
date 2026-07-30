#!/usr/bin/env bash
# 8RI2：scaffold dynamic_locked 生成 1000 → 全量评估 → n100 出图
# Usage: GPU=0 bash run_8ri2_scaffold_dynamic_locked_gen1000.sh
set -eo pipefail
source /home/user/anaconda3/etc/profile.d/conda.sh
conda activate diffdynamic
export PYTHONPATH=/data/ye/DiffDynamic:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /data/ye/DiffDynamic

DATE_TAG="${DATE_TAG:-$(date +%Y%m%d)}"
GPU="${GPU:-0}"
BASE_SEED="${BASE_SEED:-20260719}"
OUT="${OUT:-experiments/8ri2_scaffold_dl_gen1000_gpu${GPU}_${DATE_TAG}}"
CFG_SRC="${CFG:-configs/run_8ri2_scaffold_dynamic_locked_gen1000.yml}"
PROTEIN="/data/ye/protein-ligand/8RI2/8RI2_apo.pdb"
LIGAND="/data/ye/protein-ligand/8RI2/8RI2_ligand.sdf"

mkdir -p "$OUT"/{run,reconstructed_molecules_n100,plots}
CFG="$OUT/run_config.yml"
cp "$CFG_SRC" "$CFG"
sed -i "s/^  seed: .*/  seed: ${BASE_SEED}/" "$CFG" || true
sed -i "s/^      seed: .*/      seed: ${BASE_SEED}/" "$CFG" || true

echo "OUT=$OUT GPU=$GPU CFG=$CFG SEED=$BASE_SEED START=$(date -Iseconds)"

echo "========== Generation (scaffold dynamic_locked n=1000) =========="
python -u scripts/sample_diffusion.py "$CFG" \
  --protein_path "$PROTEIN" --ligand_path "$LIGAND" \
  --result_path "$OUT/run" --device "cuda:${GPU}" \
  2>&1 | tee "$OUT/generation.log"

echo "========== Evaluation (all) =========="
PT=$(ls -t "$OUT"/run/result_custom_*.pt | head -1)
python split_and_eval_parallel.py \
  --pt_file "$PT" --protein_root /data/ye/protein-ligand \
  --output_dir "$OUT/run" --exhaustiveness 8 --num_workers 8 \
  2>&1 | tee "$OUT/eval_parallel.log"

echo "========== Random extract n=100 (valid Vina) + plot =========="
python3 scripts/select_n100_valid_and_plot.py \
  --out "$OUT" \
  --act /data/ye/protein-ligand/8RI2/active_dock_qed_sa.csv \
  --tag 8RI2_scaffold_dl_n1000 \
  --prefix 8RI2_scaffold_dl_n1000 \
  --seed "${SELECT_SEED:-$BASE_SEED}" --n 100 \
  2>&1 | tee "$OUT/n100_plot.log"

echo "========== t-SNE + LogP-MW =========="
python3 /data/ye/protein-ligand/plot_generic_results.py \
  --gen_sdf "$OUT/reconstructed_molecules_n100" \
  --gen_csv "$OUT/generated_dock_qed_sa_n100.csv" \
  --act_sdf /data/ye/protein-ligand/8RI2/active_ligands/sdf \
  --out_dir "$OUT/plots" \
  --tag 8RI2_scaffold_dl_n1000 \
  2>&1 | tee -a "$OUT/n100_plot.log"

echo "END=$(date -Iseconds)"
echo "ALL_DONE: $OUT"
ls "$OUT/run"/result_custom_*.pt
wc -l "$OUT/eval_parallel.log" | awk '{print "eval_log_lines",$1}'
ls "$OUT/run/reconstructed_molecules" 2>/dev/null | wc -l | xargs -I{} echo "recon_sdf={}"
