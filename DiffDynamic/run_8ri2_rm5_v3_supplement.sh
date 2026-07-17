#!/usr/bin/env bash
# 8RI2 RM5 补充批次：生成→评估→与主批次合并，确保 ≥100 有效唯一
set -eo pipefail
source /home/user/anaconda3/etc/profile.d/conda.sh
conda activate diffdynamic
export PYTHONPATH=/data/ye/DiffDynamic:$PYTHONPATH
cd /data/ye/DiffDynamic

GPU="${GPU_8RI2:-1}"
MAIN="experiments/8ri2_rm5_dl_baseline30_v3"
SUPP="experiments/8ri2_rm5_dl_baseline30_v3_supp"
CFG="configs/run_8ri2_rm5_dl_baseline30_v3_supp.yml"
PROTEIN="/data/ye/protein-ligand/8RI2/8RI2_apo.pdb"
LIGAND="/data/ye/protein-ligand/8RI2/8RI2_ligand_RM5_pose.sdf"
ACT_CSV="/data/ye/protein-ligand/8RI2/active_dock_qed_sa.csv"
ACT_SDF="/data/ye/protein-ligand/8RI2/active_ligands/sdf"
TAG="8RI2_RM5_v3"

echo "========== SUPP 8RI2 RM5 (GPU $GPU) =========="
mkdir -p "$SUPP"
python -u scripts/sample_diffusion.py "$CFG" \
  --protein_path "$PROTEIN" --ligand_path "$LIGAND" \
  --result_path "$SUPP" --device "cuda:${GPU}" \
  2>&1 | tee "$SUPP/generation.log"

PT=$(ls -t "$SUPP"/result_custom_*.pt | head -1)
python split_and_eval_parallel.py \
  --pt_file "$PT" --protein_root /data/ye/protein-ligand \
  --output_dir "$SUPP" --exhaustiveness 8 --num_workers 8 \
  2>&1 | tee "$SUPP/eval_parallel.log"

# Merge main + supp evaluation xlsx into main dir
python3 - <<'PY'
import glob, shutil
from pathlib import Path
import pandas as pd
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
from rdkit import Chem

main = Path('experiments/8ri2_rm5_dl_baseline30_v3')
supp = Path('experiments/8ri2_rm5_dl_baseline30_v3_supp')
dfs = []
for d in [main, supp]:
    xlsx = sorted(d.glob('evaluation_results_*.xlsx'))[-1]
    df = pd.read_excel(xlsx, sheet_name='评估结果')
    df['_src'] = d.name
    dfs.append(df)
    print(f'{d.name}: {len(df)} rows from {xlsx.name}')

merged = pd.concat(dfs, ignore_index=True)
# dedupe by canonical SMILES, keep first (prefer main)
canon = []
keep = []
seen = set()
for i, s in enumerate(merged['SMILES'].astype(str)):
    m = Chem.MolFromSmiles(s) if len(s) > 8 else None
    if m is None:
        continue
    c = Chem.MolToSmiles(m)
    if c in seen:
        continue
    seen.add(c)
    keep.append(i)
    canon.append(c)

uniq_df = merged.iloc[keep].copy()
uniq_df = uniq_df.drop(columns=['_src'], errors='ignore')
print(f'merged unique valid: {len(uniq_df)}')

# write merged xlsx
from datetime import datetime
ts = datetime.now().strftime('%Y%m%d_%H%M%S')
out_xlsx = main / f'evaluation_results_merged_{ts}.xlsx'
# use same sheet name
with pd.ExcelWriter(out_xlsx, engine='openpyxl') as w:
    uniq_df.to_excel(w, sheet_name='评估结果', index=False)
print('wrote', out_xlsx)

# also copy reconstructed SDFs from supp into main (optional)
src = supp / 'reconstructed_molecules'
dst = main / 'reconstructed_molecules'
if src.exists():
    dst.mkdir(exist_ok=True)
    for f in src.glob('*.sdf'):
        target = dst / f'supp_{f.name}'
        if not target.exists():
            shutil.copy2(f, target)

# CSV
out = pd.DataFrame({
    'smiles': uniq_df['SMILES'],
    'vina_dock': uniq_df['Vina_Dock_亲和力'],
    'qed': uniq_df['QED评分'],
    'sa': uniq_df['SA评分'],
    'logp': uniq_df['logP'],
    'molwt': uniq_df['分子量'],
})
out.to_csv(main / 'generated_dock_qed_sa.csv', index=False)
(main / 'unique_count.txt').write_text(str(len(uniq_df)))
if len(uniq_df) < 100:
    raise SystemExit(f'FAIL unique={len(uniq_df)} < 100')
print('OK_UNIQUE', len(uniq_df))
PY

# Re-plot on merged
XLSX=$(ls -t "$MAIN"/evaluation_results_merged_*.xlsx | head -1)
mkdir -p "$MAIN/plots"
python /data/ye/protein-ligand/plot_generic_violin.py \
  --gen "$MAIN/generated_dock_qed_sa.csv" --act "$ACT_CSV" \
  --out_dir "$MAIN/plots" --tag "$TAG"
python /data/ye/protein-ligand/plot_generic_violin_expanded.py \
  --xlsx "$XLSX" --act "$ACT_CSV" \
  --out_dir "$MAIN/plots" --tag "$TAG"
python /data/ye/protein-ligand/plot_generic_results.py \
  --gen_sdf "$MAIN/reconstructed_molecules" --act_sdf "$ACT_SDF" \
  --out_dir "$MAIN/plots" --tag "$TAG" --gen_csv "$MAIN/generated_dock_qed_sa.csv"
SCAFFOLD=$(ls -t "$MAIN"/scaffold/*murcko*.sdf experiments/8ri2prudent_murckosites_v1/scaffold/8RI2_murcko_scaffold.sdf 2>/dev/null | head -1)
if [[ -f "$SCAFFOLD" ]]; then
  python /data/ye/protein-ligand/postprocess_sdf.py \
    --pocket-name "$TAG" --sdf-dir "$MAIN/reconstructed_molecules" \
    --active-sdf "$ACT_SDF" --scaffold-sdf "$SCAFFOLD" \
    --output-dir "$MAIN/plots" --top-n 50 || true
fi

echo "SUPP_DONE unique=$(cat $MAIN/unique_count.txt)"
