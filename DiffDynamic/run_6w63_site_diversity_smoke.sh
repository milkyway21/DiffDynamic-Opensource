#!/usr/bin/env bash
# Path1: 4WI scaffold v3 — full chain (gen + eval max_samples=5 + plots)
# Path2: X77 exit-vector — generation smoke only
set -eo pipefail
source /home/user/anaconda3/etc/profile.d/conda.sh
conda activate diffdynamic
export PYTHONPATH=/data/ye/DiffDynamic:$PYTHONPATH
cd /data/ye/DiffDynamic

PROTEIN="/data/ye/protein-ligand/6W63_receptor_nowater_noligand.pdb"
LIG_4WI="/data/ye/protein-ligand/6W63_ligand_4WI_pose.sdf"
LIG_X77="/data/ye/protein-ligand/6W63_ligand_X77_pose.sdf"
ACT_CSV="/data/ye/protein-ligand/6W63/active_dock_qed_sa.csv"
ACT_SDF="/data/ye/protein-ligand/6W63/active_ligands/sdf"
GPU="${GPU:-0}"

run_path1() {
  local DIR="outputs/smoke_6w63_4wi_v3"
  local CFG="configs/run_6w63_4wi_dl_baseline30_v3.yml"
  local TAG="6W63_4WI_v3_smoke"
  rm -rf "$DIR"
  mkdir -p "$DIR"
  echo "========== PATH1: 4WI v3 full chain → $DIR =========="
  python scripts/sample_diffusion.py "$CFG" \
    --protein_path "$PROTEIN" --ligand_path "$LIG_4WI" \
    --result_path "$DIR" --device "cuda:${GPU}" \
    2>&1 | tee "$DIR/generation.log"

  # expect >=4 murcko sites in log / json
  python3 - <<'PY'
import json, glob, sys
js = glob.glob('outputs/smoke_6w63_4wi_v3/scaffold/*murcko_sites*.json')
assert js, 'no sites json'
data = json.load(open(js[0]))
n = len(data['attachment_sites'])
print(f'[check] 4WI sites={n}')
assert n >= 4, n
PY

  PT=$(ls -t "$DIR"/result_custom_*.pt | head -1)
  python split_and_eval_parallel.py \
    --pt_file "$PT" \
    --protein_root /data/ye/protein-ligand \
    --output_dir "$DIR" \
    --exhaustiveness 8 \
    --num_workers 4 \
    --max_samples 5 \
    --vina_timeout 20 \
    2>&1 | tee "$DIR/eval_parallel.log"

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
})
out.to_csv("$DIR/generated_dock_qed_sa.csv", index=False)
print(out.describe())
PY

  mkdir -p "$DIR/plots"
  python /data/ye/protein-ligand/plot_generic_violin.py \
    --gen "$DIR/generated_dock_qed_sa.csv" --act "$ACT_CSV" \
    --out_dir "$DIR/plots" --tag "$TAG"
  python /data/ye/protein-ligand/plot_generic_violin_expanded.py \
    --xlsx "$XLSX" --act "$ACT_CSV" \
    --out_dir "$DIR/plots" --tag "$TAG"
  python /data/ye/protein-ligand/plot_generic_results.py \
    --gen_sdf "$DIR/reconstructed_molecules" --act_sdf "$ACT_SDF" \
    --out_dir "$DIR/plots" --tag "$TAG" --gen_csv "$DIR/generated_dock_qed_sa.csv"
  SCAFFOLD=$(ls -t "$DIR"/scaffold/*murcko*.sdf "$DIR"/scaffold/*.sdf 2>/dev/null | head -1 || true)
  if [[ -z "$SCAFFOLD" || ! -s "$SCAFFOLD" ]]; then
    SCAFFOLD="experiments/6w63prudent_murckosites_v1/scaffold/6W63_murcko_scaffold.sdf"
  fi
  if [[ -f "$SCAFFOLD" ]]; then
    python /data/ye/protein-ligand/postprocess_sdf.py \
      --pocket-name "$TAG" --sdf-dir "$DIR/reconstructed_molecules" \
      --active-sdf "$ACT_SDF" --scaffold-sdf "$SCAFFOLD" \
      --output-dir "$DIR/plots" --top-n 50 || true
  fi

  echo "PATH1_DONE: $DIR"
  ls -la "$DIR"/evaluation_results_*.xlsx "$DIR"/generated_dock_qed_sa.csv "$DIR"/plots/*active* "$DIR"/plots/*combined* 2>/dev/null | head -30
}

run_path2() {
  local DIR="outputs/smoke_6w63_x77_exitvector"
  local CFG="configs/run_6w63_x77_exitvector_smoke.yml"
  rm -rf "$DIR"
  mkdir -p "$DIR"
  echo "========== PATH2: X77 exit-vector smoke → $DIR =========="
  python scripts/sample_diffusion.py "$CFG" \
    --protein_path "$PROTEIN" --ligand_path "$LIG_X77" \
    --result_path "$DIR" --device "cuda:${GPU}" \
    2>&1 | tee "$DIR/generation.log"

  python3 - <<'PY'
import json, glob, re
js = glob.glob('outputs/smoke_6w63_x77_exitvector/scaffold/*murcko_sites*.json')
assert js, 'no sites json'
data = json.load(open(js[0]))
sites = data['attachment_sites']
n_ev = sum(1 for s in sites if s.get('site_kind')=='exit_vector')
n_m = sum(1 for s in sites if s.get('site_kind')!='exit_vector')
print(f'[check] murcko={n_m} exit_vector={n_ev} total={len(sites)}')
assert n_m >= 1 and n_ev >= 5 and len(sites) >= 6, (n_m, n_ev, len(sites))
log = open('outputs/smoke_6w63_x77_exitvector/generation.log').read()
assert 'exit_vector' in log or 'sequential_random' in log
# allocations should vary across sites (not only site 0)
allocs = re.findall(r'sequential_random: 各位点分配 \[([^\]]+)\]', log)
print('[check] alloc samples:', allocs[:5])
assert allocs, 'no sequential_random lines'
print('PATH2_OK')
PY
  echo "PATH2_DONE: $DIR"
}

MODE="${1:-both}"
case "$MODE" in
  path1) run_path1 ;;
  path2) run_path2 ;;
  both)
    run_path2
    run_path1
    ;;
  *) echo "usage: $0 [path1|path2|both]"; exit 1 ;;
esac
echo "ALL_SMOKE_DONE"
