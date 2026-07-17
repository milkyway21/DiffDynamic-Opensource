#!/usr/bin/env bash
# 8RI2 RM5：100 瞬时+完整生成（gaussian 团簇释放）→ experiments/8ri2_rm5_instant_gen100_gaussian
# Usage: bash run_8ri2_rm5_instant_gen100.sh
set -eo pipefail
source /home/user/anaconda3/etc/profile.d/conda.sh
conda activate diffdynamic
export PYTHONPATH=/data/ye/DiffDynamic:$PYTHONPATH
cd /data/ye/DiffDynamic

OUT="experiments/8ri2_rm5_instant_gen100_gaussian"
CFG="configs/run_8ri2_rm5_instant_gen100_gaussian.yml"
PROTEIN="/data/ye/protein-ligand/8RI2/8RI2_apo.pdb"
LIGAND="/data/ye/protein-ligand/8RI2/8RI2_ligand_RM5_pose.sdf"
GPU="${GPU:-0}"
BASE_SEED=42

mkdir -p "$OUT"/{instant,final,run}
echo "OUT=$OUT GPU=$GPU START=$(date -Iseconds)"

# ---------- 1) 100 instant SDFs（RNG 与 sample_diffusion 一致）----------
python3 - <<'PY'
import json
from pathlib import Path
import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
import torch

from utils.scaffold_sites import (
    extract_murcko_attachment_sites,
    build_extra_atom_positions,
    DEFAULT_MURCKO_SITES_CFG,
)

OUT = Path('experiments/8ri2_rm5_instant_gen100_gaussian')
INSTANT = OUT / 'instant'
INSTANT.mkdir(parents=True, exist_ok=True)
LIGAND = '/data/ye/protein-ligand/8RI2/8RI2_ligand_RM5_pose.sdf'
BASE_SEED = 42
N = 100

sites_cfg = {
    **DEFAULT_MURCKO_SITES_CFG,
    'p_active': 0.5,
    'per_site_add_mode': 'fragment_prior',
    'fragment_sizes': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 18, 20],
    'fragment_weights': [0.08, 0.06, 0.05, 0.05, 0.05, 0.14, 0.07, 0.06, 0.05, 0.10, 0.10, 0.10, 0.04, 0.03, 0.01, 0.005, 0.005],
    'max_per_site': 20,
    'min_per_site': 0,
    'per_site_count_mode': 'sequential_random',
    'jitter_std': 1.0,
    'jitter_mode': 'gaussian',
    'max_active_sites': 1,
    'prefer_murcko_sites': True,
    'overflow_mode': 'cap',
    'dedup_dist': 0.5,
    'n_extra_min_clamp': 1,
    'n_extra_max_clamp': 12,
}

mol = Chem.RemoveHs(Chem.SDMolSupplier(LIGAND, removeHs=False)[0])
Chem.SanitizeMol(mol)
# bond length sanity
conf = mol.GetConformer()
for b in mol.GetBonds():
    i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
    pi = np.array([conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z])
    pj = np.array([conf.GetAtomPosition(j).x, conf.GetAtomPosition(j).y, conf.GetAtomPosition(j).z])
    d = float(np.linalg.norm(pi - pj))
    assert 0.5 < d < 2.2, (i, j, d)

core = MurckoScaffold.GetScaffoldForMol(mol)
match = mol.GetSubstructMatch(core)
scaffold_set = set(match)
sites = extract_murcko_attachment_sites(mol, list(match))
print(f'sites={len(sites)} scaffold_atoms={len(scaffold_set)}')

rw = Chem.RWMol(Chem.Mol(mol))
for idx in sorted((i for i in range(rw.GetNumAtoms()) if i not in scaffold_set), reverse=True):
    rw.RemoveAtom(idx)
scaffold_mol = rw.GetMol()
Chem.SanitizeMol(scaffold_mol)

# scaffold-only reference
w = Chem.SDWriter(str(INSTANT / '00_scaffold_only.sdf'))
sc0 = Chem.Mol(scaffold_mol)
sc0.SetProp('_Name', '8RI2_RM5_murcko_scaffold')
w.write(sc0); w.close()

pos_arr = np.array([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z]
                    for i in range(mol.GetNumAtoms())])
center = torch.tensor(pos_arr.mean(axis=0), dtype=torch.float32)

manifest = []
all_path = INSTANT / 'all_100_instant.sdf'
aw = Chem.SDWriter(str(all_path))
aw.SetKekulize(False)

for sample_idx in range(N):
    # MUST match scripts/sample_diffusion._murcko_sample_rng(grow_cfg, sample_idx)
    # grow_cfg normally has no seed → base=42
    rng_seed = BASE_SEED + sample_idx * 9973
    rng = np.random.default_rng(rng_seed)
    extra_pos, meta = build_extra_atom_positions(
        0, sites, sites_cfg, center, 'cpu', rng=rng,
    )
    n_extra = int(extra_pos.shape[0])
    counts = meta.get('site_allocation')

    combo = Chem.RWMol(Chem.Mol(scaffold_mol))
    conf0 = combo.GetConformer()
    n_sc = combo.GetNumAtoms()
    for _ in range(n_extra):
        combo.AddAtom(Chem.Atom(6))
    conf_new = Chem.Conformer(combo.GetNumAtoms())
    for i in range(n_sc):
        conf_new.SetAtomPosition(i, conf0.GetAtomPosition(i))
    for i in range(n_extra):
        x, y, z = extra_pos[i].tolist()
        conf_new.SetAtomPosition(n_sc + i, (float(x), float(y), float(z)))
    combo.RemoveAllConformers()
    combo.AddConformer(conf_new, assignId=True)
    out_mol = combo.GetMol()
    assert out_mol.GetNumBonds() == scaffold_mol.GetNumBonds()
    assert all(out_mol.GetAtomWithIdx(i).GetDegree() == 0 for i in range(n_sc, n_sc + n_extra))

    name = f'instant_{sample_idx:03d}_nextra{n_extra}'
    out_mol.SetProp('_Name', name)
    out_mol.SetProp('sample_idx', str(sample_idx))
    out_mol.SetProp('rng_seed', str(rng_seed))
    out_mol.SetProp('n_scaffold', str(n_sc))
    out_mol.SetProp('n_extra', str(n_extra))
    out_mol.SetProp('site_allocation', str(counts))
    out_mol.SetProp('scaffold_smiles', Chem.MolToSmiles(scaffold_mol))

    path = INSTANT / f'{name}.sdf'
    w = Chem.SDWriter(str(path))
    w.SetKekulize(False)
    w.write(out_mol)
    w.close()
    aw.write(out_mol)

    # multimol companion
    multi = INSTANT / f'{name}_multimol.sdf'
    mw = Chem.SDWriter(str(multi))
    sc = Chem.Mol(scaffold_mol); sc.SetProp('_Name', f'scaffold_{sample_idx:03d}'); sc.SetProp('role', 'scaffold')
    mw.write(sc)
    for i in range(n_extra):
        am = Chem.RWMol(); am.AddAtom(Chem.Atom(6))
        c = Chem.Conformer(1)
        x, y, z = extra_pos[i].tolist()
        c.SetAtomPosition(0, (float(x), float(y), float(z)))
        am.AddConformer(c, assignId=True)
        m = am.GetMol(); m.SetProp('_Name', f'extra_{sample_idx:03d}_{i:02d}'); m.SetProp('role', 'discrete_extra')
        mw.write(m)
    mw.close()

    manifest.append({
        'sample_idx': sample_idx,
        'rng_seed': rng_seed,
        'n_extra': n_extra,
        'site_allocation': counts,
        'instant_sdf': str(path),
        'instant_multimol_sdf': str(multi),
    })
    if sample_idx < 5 or sample_idx % 20 == 0:
        print(f'[{sample_idx:03d}] n_extra={n_extra} seed={rng_seed}')

aw.close()
(OUT / 'instant_manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str))
print(f'Wrote {N} instant SDFs → {INSTANT}')
PY

# ---------- 2) 完整生成 ----------
echo "========== Generation =========="
python -u scripts/sample_diffusion.py "$CFG" \
  --protein_path "$PROTEIN" --ligand_path "$LIGAND" \
  --result_path "$OUT/run" --device "cuda:${GPU}" \
  2>&1 | tee "$OUT/generation.log"

# ---------- 3) 对接评估 ----------
echo "========== Evaluation =========="
PT=$(ls -t "$OUT"/run/result_custom_*.pt | head -1)
python split_and_eval_parallel.py \
  --pt_file "$PT" --protein_root /data/ye/protein-ligand \
  --output_dir "$OUT/run" --exhaustiveness 8 --num_workers 8 \
  2>&1 | tee "$OUT/eval_parallel.log"

# ---------- 4) 从 .pt 按生成顺序重建 final，并去碎片 ----------
echo "========== Pair finals from .pt =========="
python3 scripts/pack_8ri2_rm5_instant_gen100_gaussian.py

echo "END=$(date -Iseconds)"
echo "ALL_DONE: $OUT"
ls -la "$OUT"
ls "$OUT/instant" | wc -l
ls "$OUT/final" | wc -l
