#!/bin/bash
# 测试骨架生成的分子连通性
# 用法: bash test_scaffold_connectivity.sh [GPU_ID] [TARGET] [BATCH_SIZE]
# 生成 -> 重建(vina none) -> 统计连通率 -> 判定 PASS/FAIL

set -eo pipefail

GPU_ID="${1:-3}"
TARGET="${2:-ikzf2}"
BATCH_SIZE="${3:-50}"

cd /data/ye/DiffDynamic
source /home/user/anaconda3/etc/profile.d/conda.sh
conda activate diffdynamic
export PYTHONPATH="/data/ye/DiffDynamic${PYTHONPATH:+:$PYTHONPATH}"

STRUCT_ROOT="/data/ye/e-drug-lab/Scientist_In_E-Drug-Lab/runs/molecular_glue_structures_ikzf2_gspt1_20260803_235811"
CONFIG="/home/user/Desktop/Ye/DiffDynamic/hsvpol/molglue_ikzf2_gspt1/configs/sampling_fast_scaffold_custom_warhead_nextra10_25.yml"
OUT="/tmp/test_connectivity_${TARGET}_gpu${GPU_ID}"

if [ "$TARGET" = "ikzf2" ]; then
  PROTEIN="$STRUCT_ROOT/01_ikzf2_molecular_glue/receptor/7U8F_receptor_clean.pdb"
  LIGAND="$STRUCT_ROOT/01_ikzf2_molecular_glue/ligand/7U8F_LWK_D_502_native.sdf"
elif [ "$TARGET" = "gspt1" ]; then
  PROTEIN="$STRUCT_ROOT/02_gspt1_or_gstp1_molecular_glue/receptor/5HXB_receptor_clean.pdb"
  LIGAND="$STRUCT_ROOT/02_gspt1_or_gstp1_molecular_glue/ligand/5HXB_85C_C_502_native.sdf"
else
  echo "ERROR: TARGET must be ikzf2 or gspt1"
  exit 1
fi

echo "=========================================="
echo " 测试骨架生成连通性"
echo " GPU: $GPU_ID | Target: $TARGET | Batch: $BATCH_SIZE"
echo "=========================================="

rm -rf "$OUT"; mkdir -p "$OUT"

# Step 1: 生成
echo "[1/3] 生成 $BATCH_SIZE 分子 (prudent 模式, anchor_strength=0.3, start_t=150)..."
python -u scripts/sample_diffusion.py "$CONFIG" \
  --protein_path "$PROTEIN" \
  --ligand_path "$LIGAND" \
  --molecule_path "$LIGAND" \
  --result_path "$OUT" \
  -gpu "$GPU_ID" \
  --batch_size "$BATCH_SIZE" \
  2>&1 | tee "$OUT/run.log" | tail -3

# 找到生成的 .pt 文件
PT_FILE=$(ls -t "$OUT"/result_custom_*.pt 2>/dev/null | head -1)
if [ -z "$PT_FILE" ]; then
  echo "ERROR: 未找到生成的 .pt 文件"
  exit 1
fi
echo "生成完成: $PT_FILE"

# Step 2: 重建 (vina none)
echo "[2/3] 重建分子 (vina none)..."
EXTRACT_DIR="$OUT/extract_novina"
mkdir -p "$EXTRACT_DIR"
python -u evaluate_pt_with_correct_reconstruct.py \
  "$PT_FILE" \
  --protein_root ./data/crossdocked_pocket10_test_only \
  --receptor_pdb "$PROTEIN" \
  --output_dir "$EXTRACT_DIR" \
  --vina-modes none \
  --max_samples "$BATCH_SIZE" \
  2>&1 | tee "$OUT/extract.log" | tail -5

# Step 3: 统计连通率
echo "[3/3] 统计连通率..."
python3 -u <<'PYEOF'
import os, sys, glob, json
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

extract_dir = os.environ.get("EXTRACT_DIR", "")
if not extract_dir:
    # fallback
    extract_dir = "/tmp/test_connectivity_extract"

sdf_files = glob.glob(os.path.join(extract_dir, "**", "*.sdf"), recursive=True)
if not sdf_files:
    sdf_files = glob.glob(os.path.join(extract_dir, "*.sdf"))

total = 0
connected = 0
fragmented = 0
no_scaffold = 0
scaffold_smarts = "O=C1CCC(N2Cc3ccccc3C2=O)C(=O)N1"
scaffold_mol = Chem.MolFromSmarts(scaffold_smarts)

for sdf_path in sorted(sdf_files):
    suppl = Chem.SDMolSupplier(sdf_path, sanitize=False)
    for mol in suppl:
        if mol is None:
            continue
        try:
            Chem.SanitizeMol(mol, sanitizeOps=Chem.SANITIZE_ALL ^ Chem.SANITIZE_PROPERTIES)
            Chem.SetAromaticity(mol)
        except Exception:
            continue
        total += 1
        smiles = Chem.MolToSmiles(mol)
        frags = smiles.split(".")
        if len(frags) == 1:
            connected += 1
        else:
            fragmented += 1
        if not mol.HasSubstructMatch(scaffold_mol):
            no_scaffold += 1

print(f"\n{'='*50}")
print(f" 连通性测试结果")
print(f"{'='*50}")
print(f" 总分子数:     {total}")
print(f" 连通(1片段):  {connected} ({100*connected/max(total,1):.1f}%)")
print(f" 碎片化(多片段): {fragmented} ({100*fragmented/max(total,1):.1f}%)")
print(f" 含骨架:       {total - no_scaffold} ({100*(total-no_scaffold)/max(total,1):.1f}%)")
print(f"{'='*50}")
if total > 0:
    rate = connected / total
    if rate >= 0.8:
        print(f" 结果: PASS (连通率 {100*rate:.1f}% >= 80%)")
        sys.exit(0)
    else:
        print(f" 结果: FAIL (连通率 {100*rate:.1f}% < 80%)")
        sys.exit(1)
else:
    print(" 结果: FAIL (无有效分子)")
    sys.exit(1)
PYEOF

RESULT=$?
echo ""
if [ $RESULT -eq 0 ]; then
  echo "✅ 测试通过！可以进行批量生成。"
else
  echo "❌ 测试失败，需要调整参数。"
  echo "建议: 降低 start_t 或提高 extra_anchor_strength"
fi
exit $RESULT
