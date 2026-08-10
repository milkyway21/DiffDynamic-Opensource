---
name: "drug-gen-similarity-search"
description: "End-to-end drug molecule generation pipeline: scaffold-based DiffDynamic generation, SDF reconstruction, fragment cleanup, dedup, and Tanimoto similarity search against reference molecules. Invoke when user asks to generate molecules and find similar ones to known compounds."
---

# Drug Generation & Similarity Search

Complete pipeline: DiffDynamic scaffold generation → PT→SDF reconstruction → fragment removal → dedup → Tanimoto similarity comparison.

## When to Invoke

- User wants to generate drug-like molecules using DiffDynamic scaffold mode
- User wants to compare generated molecules against reference SDF files
- User asks for "生成分子并查找相似分子" or similar
- User wants batch molecule generation with similarity analysis

## Prerequisites

- Conda env: `diffdynamic` at `/home/user/anaconda3/envs/diffdynamic`
- Working dir: `/data/ye/DiffDynamic`
- Config template: `/home/user/Desktop/Ye/DiffDynamic/hsvpol/molglue_ikzf2_gspt1/configs/sampling_fast_scaffold_custom_warhead_nextra10_25.yml`
- Job script: `/home/user/Desktop/Ye/DiffDynamic/hsvpol/molglue_ikzf2_gspt1/scripts/run_scaffold_fast_100k_one_job.sh`
- Reference SDFs: `/data/ye/sdf/GSPT1.sdf`, `/data/ye/sdf/IKZF2.sdf` (55 molecules total, CRLF line endings, no `$$$$` between records — must split by `$$$$` after CRLF→LF conversion)
- Protein/ligand structures: `/data/ye/e-drug-lab/Scientist_In_E-Drug-Lab/runs/molecular_glue_structures_ikzf2_gspt1_20260803_235811/`

### Target Structures

| Target | Protein PDB | Ligand SDF |
|--------|------------|------------|
| IKZF2 | `.../01_ikzf2_molecular_glue/receptor/7U8F_receptor_clean.pdb` | `.../01_ikzf2_molecular_glue/ligand/7U8F_LWK_D_502_native.sdf` |
| GSPT1 | `.../02_gspt1_or_gstp1_molecular_glue/receptor/5HXB_receptor_clean.pdb` | `.../02_gspt1_or_gstp1_molecular_glue/ligand/5HXB_85C_C_502_native.sdf` |

## Pipeline Steps

### Step 1: Batch Generation

Generate molecules using scaffold `dynamic_locked` mode. Each job uses a unique seed and produces `num_samples` molecules.

**Key config parameters** (in `sampling_fast_scaffold_custom_warhead_nextra10_25.yml`):
- `scaffold.mode: dynamic_locked`
- `scaffold_source: custom`
- `scaffold_smarts: "O=C1CCC(N2Cc3ccccc3C2=O)C(=O)N1"` (glutarimide→isoindolin-1-one)
- `fix_scaffold_pos: true`, `fix_scaffold_type: true`
- `grow.n_extra_min: 10`, `grow.n_extra_max: 22`
- `grow.extra_anchor_strength: 0.0` (no anchor)
- `targetdiff_baseline_refine.enable: true`, `start_t: 50`, `refine_batch_size: 100`
- `dynamic.large_step.batch_size: 100`
- `scaffold.num_samples: 100` (per seed)

**Launcher script** (`run_10k_gpu3.sh` pattern):
```bash
#!/usr/bin/env bash
set -eo pipefail
GPU=<GPU_ID>
TARGET=<ikzf2|gspt1>
N_JOBS=<N>  # e.g. 100 for 10k molecules with num_samples=100
ROOT="<OUTPUT_ROOT>"
SCRIPT_DIR="/home/user/Desktop/Ye/DiffDynamic/hsvpol/molglue_ikzf2_gspt1/scripts"
CAMPAIGN_LOG="$ROOT/$TARGET/campaign.log"
export NUM_SAMPLES=100
export ROOT; export GPU; export TARGET
mkdir -p "$ROOT/$TARGET/jobs"
echo "=== Campaign start $(date -Iseconds) GPU=$GPU TARGET=$TARGET N_JOBS=$N_JOBS ===" | tee "$CAMPAIGN_LOG"
DONE_COUNT=0; FAIL_COUNT=0
for JOB_ID in $(seq 0 $((N_JOBS - 1))); do
  JOB_TAG=$(printf 'job_%04d' "$JOB_ID")
  echo ">>> [$(date +%H:%M:%S)] Job $JOB_TAG ($((JOB_ID+1))/$N_JOBS) on GPU$GPU ..." | tee -a "$CAMPAIGN_LOG"
  if JOB_ID="$JOB_ID" bash "$SCRIPT_DIR/run_scaffold_fast_100k_one_job.sh" 2>&1 | tee -a "$CAMPAIGN_LOG"; then
    DONE_COUNT=$((DONE_COUNT + 1))
  else
    FAIL_COUNT=$((FAIL_COUNT + 1))
    echo "!!! Job $JOB_TAG FAILED" | tee -a "$CAMPAIGN_LOG"
  fi
  echo ">>> Progress: done=$DONE_COUNT fail=$FAIL_COUNT / total=$N_JOBS" | tee -a "$CAMPAIGN_LOG"
done
echo "=== Campaign DONE $(date -Iseconds) success=$DONE_COUNT fail=$FAIL_COUNT ===" | tee -a "$CAMPAIGN_LOG"
```

**Multi-target parallel**: Launch separate campaigns for each target on different GPUs.

### Step 2: Batch PT→SDF Reconstruction (80 cores)

```bash
cd /data/ye/DiffDynamic
source /home/user/anaconda3/etc/profile.d/conda.sh
conda activate diffdynamic
export PYTHONPATH="/data/ye/DiffDynamic${PYTHONPATH:+:$PYTHONPATH}"

ROOT="<JOBS_ROOT>"  # e.g. .../dl_10k_test/ikzf2/jobs
EXTRACT_BASE="<EXTRACT_ROOT>"  # e.g. .../dl_10k_test/ikzf2/extract
PROTEIN="<RECEPTOR_PDB>"
PROTEIN_ROOT="<RECEPTOR_DIR>"  # parent dir of receptor PDB
mkdir -p "$EXTRACT_BASE"

for JOB_DIR in "$ROOT"/job_*/; do
  JOB_TAG=$(basename "$JOB_DIR")
  PT=$(find "$JOB_DIR/run" -maxdepth 1 -name 'result_custom_*.pt' ! -name '*_seed*' ! -name '*_chains*' -type f 2>/dev/null | head -1)
  [[ -z "$PT" || ! -s "$PT" ]] && continue
  OUT_DIR="$EXTRACT_BASE/$JOB_TAG"
  mkdir -p "$OUT_DIR"
  CUDA_VISIBLE_DEVICES="" python -u evaluate_pt_with_correct_reconstruct.py "$PT" \
    --output_dir "$OUT_DIR" --receptor_pdb "$PROTEIN" --protein_root "$PROTEIN_ROOT" \
    --vina-modes none --enable_isolation > "$OUT_DIR/extract.log" 2>&1 &
  DONE_JOBS=$(( (DONE_JOBS+1) % 80 ))
  [[ $DONE_JOBS -eq 0 ]] && wait
done
wait
echo "All extractions done."
```

**Critical**: `--vina-modes none` disables Vina docking. `--enable_isolation` prevents RDKit segfaults.

### Step 3: Fragment Removal (keep largest fragment)

```python
import glob, os
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

sdf_dir = '<EXTRACT_ROOT>'
out_dir = '<CLEANED_ROOT>'
# IMPORTANT: preserve job subdirectory structure to avoid filename collisions
# (SDF filenames are timestamp-based and identical across jobs)

for f in glob.glob(os.path.join(sdf_dir, '**', '*.sdf'), recursive=True):
    mol = Chem.SDMolSupplier(f, sanitize=True)[0]
    if mol is None: continue
    rel_path = os.path.relpath(f, sdf_dir)
    out_path = os.path.join(out_dir, rel_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    if len(frags) > 1:
        largest = max(frags, key=lambda m: m.GetNumAtoms())
        try: Chem.SanitizeMol(largest)
        except: pass
        writer = Chem.SDWriter(out_path); writer.write(largest); writer.close()
    else:
        writer = Chem.SDWriter(out_path); writer.write(mol); writer.close()
```

**Pitfall**: SDF filenames are like `7U8F_20260808_111152_0p00.sdf` — timestamp-based, identical across jobs. Must preserve directory structure (`job_XXXX/...`) when writing cleaned files.

### Step 4: Dedup & Tanimoto Similarity Search

```python
import glob, os
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import AllChem, DataStructs
import numpy as np
RDLogger.DisableLog('rdApp.*')

# Load generated (cleaned) SDFs
sdf_dir = '<CLEANED_ROOT>'
sdf_files = sorted(glob.glob(os.path.join(sdf_dir, '**', '*.sdf'), recursive=True))
mols = []
for f in sdf_files:
    mol = Chem.SDMolSupplier(f, sanitize=True)[0]
    if mol is not None:
        mols.append((Chem.MolToSmiles(mol), mol, os.path.basename(f)))

# Deduplicate
smiles_set = {}
for smi, mol, fname in mols:
    if smi not in smiles_set:
        smiles_set[smi] = (mol, fname)

# Load reference SDFs — CRITICAL: CRLF + no $$$$ between records
# Must read as binary, replace \r\n with \n, split by $$$$ manually
ref_files = ['/data/ye/sdf/GSPT1.sdf', '/data/ye/sdf/IKZF2.sdf']
ref_mols = []
for rf in ref_files:
    with open(rf, 'rb') as f:
        content = f.read().decode('utf-8', errors='replace')
    content = content.replace('\r\n', '\n')
    blocks = content.split('$$$$')
    for block in blocks:
        block = block.strip()
        if not block: continue
        mol = Chem.MolFromMolBlock(block + '\n', sanitize=False)
        if mol is None: continue
        try: Chem.SanitizeMol(mol)
        except: pass
        if mol is not None:
            ref_mols.append((Chem.MolToSmiles(mol), mol, os.path.basename(rf)))

# Safe fingerprint (some ref molecules have invalid RingInfo)
def safe_fp(mol):
    try:
        return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    except Exception:
        try:
            mol2 = Chem.Mol(mol); Chem.SanitizeMol(mol2)
            return AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)
        except Exception:
            return None

ref_fps = [(safe_fp(m), s, src) for s, m, src in ref_mols if safe_fp(m) is not None]
gen_fps = [(safe_fp(m), s, f) for s, (m, f) in smiles_set.items() if safe_fp(m) is not None]

# Compute best Tanimoto per generated molecule
exact_matches = []  # sim == 1.0
high_sim = []       # 0.7 < sim < 1.0
all_scores = []
for gfp, gsmi, gfname in gen_fps:
    best_sim, best_ref, best_src = 0.0, None, None
    for rfp, rsmi, rsrc in ref_fps:
        sim = DataStructs.TanimotoSimilarity(gfp, rfp)
        if sim > best_sim:
            best_sim, best_ref, best_src = sim, rsmi, rsrc
        if sim == 1.0:
            exact_matches.append((gsmi, gfname, rsmi, rsrc))
    all_scores.append(best_sim)
    if 0.7 < best_sim < 1.0:
        high_sim.append((gsmi, gfname, best_ref, best_src, best_sim))
```

### Reference SDF Parsing Pitfalls

1. **CRLF line endings**: Files use `\r\n`. Must convert to `\n` before parsing.
2. **No `$$$$` between records**: `SDMolSupplier` only parses first molecule. Must manually split by `$$$$` after CRLF conversion, then parse each block with `Chem.MolFromMolBlock(block + '\n', sanitize=False)`.
3. **Invalid RingInfo**: 2 of 55 reference molecules fail `GetMorganFingerprintAsBitVect` due to unsanitized ring info. Use `safe_fp()` wrapper with fallback.
4. **Expected count**: GSPT1.sdf = 17 molecules, IKZF2.sdf = 38 molecules, total = 55.

## Output

- **Generated SDFs**: `<OUTPUT_ROOT>/<target>/extract_cleaned/job_XXXX/.../molecule.sdf`
- **Similarity report**: `<OUTPUT_ROOT>/<target>/similarity_report_cleaned.txt`
- Report includes: total count, unique SMILES, exact matches, Tanimoto > 0.7 list, similarity distribution, top 200 closest molecules.

## Testing Rules (MUST FOLLOW)

- Test generation with `batch_size=5`, not default 100
- Test evaluation with `max_samples=5`
- Vina timeout 20s (`vina_timeout=20`), or use `--vina-modes none` to skip entirely
- Never delete `DiffDynamic/diffdynamic.db`
- API params: `{"mode":"dynamic","data_id":0,"batch_size":5,"auto_evaluate":true,"auto_extract":true}`
