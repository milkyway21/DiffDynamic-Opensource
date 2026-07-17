#!/usr/bin/env bash
# 路径1 生产满跑：6W63(4WI) + 8RI2(RM5) 各生成→对接评估→出图
# 目标：每靶点 ≥100 有效唯一 SMILES
# Usage: bash run_path1_multisite_v3.sh
set -eo pipefail
source /home/user/anaconda3/etc/profile.d/conda.sh
conda activate diffdynamic
export PYTHONPATH=/data/ye/DiffDynamic:$PYTHONPATH
cd /data/ye/DiffDynamic

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
echo "GPUs: 6W63=$GPU_6W63  8RI2=$GPU_8RI2"
echo "START $(date -Iseconds)"

resolve_scaffold() {
  local DIR="$1"
  local SCAFFOLD
  SCAFFOLD=$(ls -t "$DIR"/scaffold/*murcko*.sdf "$DIR"/scaffold/*.sdf 2>/dev/null | head -1 || true)
  if [[ -n "$SCAFFOLD" && -s "$SCAFFOLD" ]]; then echo "$SCAFFOLD"; return; fi
  if [[ "$DIR" == *6w63* ]]; then
    echo "experiments/6w63prudent_murckosites_v1/scaffold/6W63_murcko_scaffold.sdf"
  else
    echo "experiments/8ri2prudent_murckosites_v1/scaffold/8RI2_murcko_scaffold.sdf"
  fi
}

run_plots() {
  local DIR="$1" TAG="$2" ACTIVE_CSV="$3" ACTIVE_SDF="$4"
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
      --output-dir "$DIR/plots" --top-n 50 || true
  fi
}

verify_unique() {
  local DIR="$1" MIN_UNIQ="${2:-100}"
  python3 - <<PY
import glob, pandas as pd
from rdkit import Chem
xlsx = sorted(glob.glob("$DIR/evaluation_results_*.xlsx"))[-1]
df = pd.read_excel(xlsx, sheet_name="评估结果")
smi = df["SMILES"].dropna().astype(str)
smi = smi[smi.str.len() > 8]
# valid unique
valid = []
for s in smi:
    m = Chem.MolFromSmiles(s)
    if m is not None:
        valid.append(Chem.MolToSmiles(m))
uniq = len(set(valid))
print(f"$DIR: eval_rows={len(smi)} valid={len(valid)} unique={uniq} (need>={$MIN_UNIQ})")
open("$DIR/unique_count.txt","w").write(str(uniq))
if uniq < $MIN_UNIQ:
    raise SystemExit(f"FAIL: unique={uniq} < {$MIN_UNIQ}")
print("OK_UNIQUE")
PY
}

run_job() {
  local KEY="$1" DIR="$2" CFG="$3" TAG="$4" PROTEIN="$5" LIGAND="$6" \
        ACT_CSV="$7" ACT_SDF="$8" GPU="$9"
  echo "========== $KEY -> $DIR (GPU $GPU) =========="
  mkdir -p "$DIR"
  python -u scripts/sample_diffusion.py "$CFG" \
    --protein_path "$PROTEIN" --ligand_path "$LIGAND" \
    --result_path "$DIR" --device "cuda:${GPU}" \
    2>&1 | tee "$DIR/generation.log"

  # site count check
  python3 - <<PY
import json, glob
js = glob.glob("$DIR/scaffold/*murcko_sites*.json")
assert js, "no sites json"
n = len(json.load(open(js[0]))["attachment_sites"])
print(f"[$KEY] murcko sites={n}")
assert n >= 3, n
PY

  PT=$(ls -t "$DIR"/result_custom_*.pt | head -1)
  python split_and_eval_parallel.py \
    --pt_file "$PT" --protein_root /data/ye/protein-ligand \
    --output_dir "$DIR" --exhaustiveness 8 --num_workers 8 \
    2>&1 | tee "$DIR/eval_parallel.log"

  run_plots "$DIR" "$TAG" "$ACT_CSV" "$ACT_SDF"
  verify_unique "$DIR" 100
  echo "DONE: $KEY"
}

# --- jobs (parallel across GPUs) ---
run_job 6w63_4wi_v3 \
  "experiments/6w63_4wi_dl_baseline30_v3" \
  "configs/run_6w63_4wi_dl_baseline30_v3.yml" \
  "6W63_4WI_v3" \
  "/data/ye/protein-ligand/6W63_receptor_nowater_noligand.pdb" \
  "/data/ye/protein-ligand/6W63_ligand_4WI_pose.sdf" \
  "/data/ye/protein-ligand/6W63/active_dock_qed_sa.csv" \
  "/data/ye/protein-ligand/6W63/active_ligands/sdf" \
  "$GPU_6W63" &
PID6=$!

run_job 8ri2_rm5_v3 \
  "experiments/8ri2_rm5_dl_baseline30_v3" \
  "configs/run_8ri2_rm5_dl_baseline30_v3.yml" \
  "8RI2_RM5_v3" \
  "/data/ye/protein-ligand/8RI2/8RI2_apo.pdb" \
  "/data/ye/protein-ligand/8RI2/8RI2_ligand_RM5_pose.sdf" \
  "/data/ye/protein-ligand/8RI2/active_dock_qed_sa.csv" \
  "/data/ye/protein-ligand/8RI2/active_ligands/sdf" \
  "$GPU_8RI2" &
PID8=$!

fail=0
wait $PID6 || fail=1
wait $PID8 || fail=1

echo "========== Verification =========="
for d in experiments/6w63_4wi_dl_baseline30_v3 experiments/8ri2_rm5_dl_baseline30_v3; do
  pt=$(ls "$d"/result_custom_*.pt 2>/dev/null | wc -l)
  xlsx=$(ls "$d"/evaluation_results_*.xlsx 2>/dev/null | wc -l)
  act=$(ls "$d"/plots/*_active_ligands.png 2>/dev/null | wc -l)
  uniq=$(cat "$d"/unique_count.txt 2>/dev/null || echo 0)
  echo "  $d: pt=$pt eval=$xlsx active_png=$act unique=$uniq"
done

if [[ "$fail" -eq 0 ]]; then
  echo "ALL_OK"
else
  echo "FAILED"
fi
echo "END $(date -Iseconds) fail=$fail"
exit $fail
