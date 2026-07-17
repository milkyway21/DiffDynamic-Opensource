#!/usr/bin/env bash
# 6W63 4WI：配体原子数 Normal(μ,σ)−骨架 → n_extra；prudent 生成 100；再随机抽 100
# Usage: GPU=3 bash run_6w63_4wi_ligandsize_normal_gen100_prudent_tbr10.sh
set -eo pipefail
source /home/user/anaconda3/etc/profile.d/conda.sh
conda activate diffdynamic
export PYTHONPATH=/data/ye/DiffDynamic:$PYTHONPATH
cd /data/ye/DiffDynamic

OUT="${OUT:-experiments/6w63_4wi_ligandsize_normal_gen100_prudent_tbr10_fragprior_adaptgate_lockpos_radial_scafbond}"
CFG="${CFG:-configs/run_6w63_4wi_ligandsize_normal_gen100_prudent_tbr10.yml}"
PROTEIN="/data/ye/protein-ligand/6W63_receptor_nowater_noligand.pdb"
LIGAND="/data/ye/protein-ligand/6W63_ligand_4WI_pose.sdf"
GPU="${GPU:-1}"
BASE_SEED="${BASE_SEED:-42}"
LIGAND_SIZE_MEAN=40
LIGAND_SIZE_STD=5.0
LIGAND_SIZE_MIN=34
LIGAND_SIZE_MAX=60

mkdir -p "$OUT"/{instant,final,run,reconstructed_molecules_n100}
echo "OUT=$OUT GPU=$GPU START=$(date -Iseconds)"
echo "n_extra_mode=ligand_size_normal_minus_scaffold μ=$LIGAND_SIZE_MEAN σ=$LIGAND_SIZE_STD"

# ---------- 1) 100 instant SDFs（与 sample_diffusion ligand_size RNG 对齐）----------
python3 - <<PY
import json
from pathlib import Path
import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
import torch

from utils.scaffold_sites import (
    extract_murcko_attachment_sites,
    build_extra_atom_positions,
    resolve_preserved_sidechains,
    count_removed_sidechain_sites,
    DEFAULT_MURCKO_SITES_CFG,
)

OUT = Path('$OUT')
INSTANT = OUT / 'instant'
INSTANT.mkdir(parents=True, exist_ok=True)
LIGAND = '$LIGAND'
BASE_SEED = $BASE_SEED
N = 100
MU, SIGMA = $LIGAND_SIZE_MEAN, $LIGAND_SIZE_STD
SIZE_MIN, SIZE_MAX = $LIGAND_SIZE_MIN, $LIGAND_SIZE_MAX

sites_cfg = {
    **DEFAULT_MURCKO_SITES_CFG,
    'p_active': 0.5,
    'max_per_site': 20,
    'min_per_site': 0,
    'per_site_count_mode': 'sequential_random',
    'per_site_add_mode': 'fragment_prior',
    'jitter_std': 1.0,
    'jitter_mode': 'gaussian',
    'radial_step': 0.25,
    'radial_bulk': 0.35,
    'radial_bulk_start': 4,
    'max_active_sites': 'n_removed_sidechains',
    'prefer_murcko_sites': True,
    'overflow_mode': 'cap',
    'dedup_dist': 0.5,
    'preserve_zero_allocation_sidechains': True,
}

mol = Chem.RemoveHs(Chem.SDMolSupplier(LIGAND, removeHs=False)[0])
Chem.SanitizeMol(mol)
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
n_sc = len(scaffold_set)
n_removed_sites = count_removed_sidechain_sites(sites)
print(f'sites={len(sites)} removed_sidechain_sites={n_removed_sites} scaffold_atoms={n_sc}')

rw = Chem.RWMol(Chem.Mol(mol))
for idx in sorted((i for i in range(rw.GetNumAtoms()) if i not in scaffold_set), reverse=True):
    rw.RemoveAtom(idx)
scaffold_mol = rw.GetMol()
Chem.SanitizeMol(scaffold_mol)

w = Chem.SDWriter(str(INSTANT / '00_scaffold_only.sdf'))
sc0 = Chem.Mol(scaffold_mol)
sc0.SetProp('_Name', '6W63_4WI_murcko_scaffold')
w.write(sc0); w.close()

pos_arr = np.array([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z]
                    for i in range(mol.GetNumAtoms())])
center = torch.tensor(pos_arr.mean(axis=0), dtype=torch.float32)
ref_pos = torch.tensor(pos_arr, dtype=torch.float32)
# 类型占位：instant 只关心坐标；log_v 用均匀 one-hot 形状
n_classes = 13
ref_log_v = torch.zeros(ref_pos.shape[0], n_classes)
ref_log_v[:, 1] = 1.0
ref_log_v = torch.log(ref_log_v.clamp_min(1e-8))

manifest = []
aw = Chem.SDWriter(str(INSTANT / 'all_100_instant.sdf'))
aw.SetKekulize(False)
n_extras = []

for sample_idx in range(N):
    # size RNG: base + sample_idx*9973 + 1 （与 sample_diffusion._sample_n_extra_atoms 对齐）
    size_rng = np.random.default_rng(BASE_SEED + sample_idx * 9973 + 1)
    total = int(round(float(size_rng.normal(MU, SIGMA))))
    total = int(np.clip(total, SIZE_MIN, SIZE_MAX))
    n_extra = int(np.clip(total - n_sc, 6, 45))
    n_extras.append(n_extra)

    place_cfg = dict(sites_cfg)
    place_cfg['n_extra_min_clamp'] = n_extra
    place_cfg['n_extra_max_clamp'] = n_extra
    # placement RNG: base + sample_idx*9973 （不消耗 size 流）
    rng_seed = BASE_SEED + sample_idx * 9973
    rng = np.random.default_rng(rng_seed)
    extra_pos, meta = build_extra_atom_positions(
        0, sites, place_cfg, center, 'cpu', rng=rng,
    )
    n_placed = int(extra_pos.shape[0])
    counts = meta.get('site_allocation')
    preserved_pos, _pv, preserved_indices, zero_ids = resolve_preserved_sidechains(
        sites, meta, place_cfg, ref_pos, ref_log_v,
    )
    n_preserved = int(preserved_pos.shape[0])

    combo = Chem.RWMol(Chem.Mol(scaffold_mol))
    conf0 = combo.GetConformer()
    # 保留侧链：按参考原子元素添加
    for idx in preserved_indices:
        combo.AddAtom(Chem.Atom(mol.GetAtomWithIdx(int(idx)).GetAtomicNum()))
    for _ in range(n_placed):
        combo.AddAtom(Chem.Atom(6))
    conf_new = Chem.Conformer(combo.GetNumAtoms())
    for i in range(n_sc):
        conf_new.SetAtomPosition(i, conf0.GetAtomPosition(i))
    for i in range(n_preserved):
        x, y, z = preserved_pos[i].tolist()
        conf_new.SetAtomPosition(n_sc + i, (float(x), float(y), float(z)))
    for i in range(n_placed):
        x, y, z = extra_pos[i].tolist()
        conf_new.SetAtomPosition(n_sc + n_preserved + i, (float(x), float(y), float(z)))
    combo.RemoveAllConformers()
    combo.AddConformer(conf_new, assignId=True)
    out_mol = combo.GetMol()

    name = f'instant_{sample_idx:03d}_nextra{n_placed}_npres{n_preserved}'
    out_mol.SetProp('_Name', name)
    out_mol.SetProp('sample_idx', str(sample_idx))
    out_mol.SetProp('rng_seed', str(rng_seed))
    out_mol.SetProp('n_scaffold', str(n_sc))
    out_mol.SetProp('n_preserved', str(n_preserved))
    out_mol.SetProp('n_extra', str(n_placed))
    out_mol.SetProp('zero_allocation_site_ids', str(zero_ids))
    out_mol.SetProp('preserved_atom_indices', str(preserved_indices))
    out_mol.SetProp('target_total_atoms', str(total))
    out_mol.SetProp('site_allocation', str(counts))
    out_mol.SetProp('scaffold_smiles', Chem.MolToSmiles(scaffold_mol))

    path = INSTANT / f'{name}.sdf'
    ww = Chem.SDWriter(str(path)); ww.SetKekulize(False); ww.write(out_mol); ww.close()
    aw.write(out_mol)

    multi = INSTANT / f'{name}_multimol.sdf'
    mw = Chem.SDWriter(str(multi))
    sc = Chem.Mol(scaffold_mol); sc.SetProp('_Name', f'scaffold_{sample_idx:03d}'); sc.SetProp('role', 'scaffold')
    mw.write(sc)
    for i, idx in enumerate(preserved_indices):
        am = Chem.RWMol(); am.AddAtom(Chem.Atom(mol.GetAtomWithIdx(int(idx)).GetAtomicNum()))
        c = Chem.Conformer(1)
        x, y, z = preserved_pos[i].tolist()
        c.SetAtomPosition(0, (float(x), float(y), float(z)))
        am.AddConformer(c, assignId=True)
        m = am.GetMol(); m.SetProp('_Name', f'preserved_{sample_idx:03d}_{i:02d}'); m.SetProp('role', 'preserved_sidechain')
        mw.write(m)
    for i in range(n_placed):
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
        'target_total_atoms': total,
        'n_extra': n_placed,
        'n_preserved': n_preserved,
        'zero_allocation_site_ids': zero_ids,
        'preserved_atom_indices': preserved_indices,
        'site_allocation': counts,
        'instant_sdf': str(path),
        'instant_multimol_sdf': str(multi),
    })
    if sample_idx < 5 or sample_idx % 20 == 0:
        print(f'[{sample_idx:03d}] total={total} n_extra={n_placed} n_preserved={n_preserved} zero={zero_ids} seed={rng_seed}')

aw.close()
arr = np.array(n_extras)
print(f'n_extra stats: mean={arr.mean():.2f} std={arr.std():.2f} min={arr.min()} max={arr.max()}')
print(f'total_atoms stats: mean={(arr+n_sc).mean():.2f} min={(arr+n_sc).min()} max={(arr+n_sc).max()}')
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
  --prudent_final_only \
  2>&1 | tee "$OUT/eval_parallel.log"

# ---------- 4) 从 .pt 按生成顺序重建 final ----------
echo "========== Pack finals =========="
OUT_DIR="$OUT" python3 scripts/pack_6w63_4wi_ligandsize_normal_gen100_prudent_tbr10.py

# ---------- 5) 随机抽取 100 个 valid-Vina + 新 palette 出图 ----------
echo "========== Random extract n=100 (valid Vina) + plot =========="
python3 scripts/select_n100_valid_and_plot.py \
  --out "$OUT" \
  --act /data/ye/protein-ligand/6W63/active_dock_qed_sa.csv \
  --tag 6W63_4WI_radial_scafbond_n100 \
  --prefix 6W63_radial_scafbond_n100 \
  --seed "${SELECT_SEED:-$BASE_SEED}" --n 100 \
  2>&1 | tee "$OUT/n100_plot.log"

# ---------- 6) t-SNE + LogP–MW (paper plots skill) ----------
echo "========== t-SNE + LogP-MW =========="
python3 /data/ye/protein-ligand/plot_generic_results.py \
  --gen_sdf "$OUT/reconstructed_molecules_n100" \
  --gen_csv "$OUT/generated_dock_qed_sa_n100.csv" \
  --act_sdf /data/ye/protein-ligand/6W63/active_ligands/sdf \
  --out_dir "$OUT/plots" \
  --tag 6W63_4WI_radial_scafbond_n100 \
  2>&1 | tee -a "$OUT/n100_plot.log"

echo "END=$(date -Iseconds)"
echo "ALL_DONE: $OUT"
ls -la "$OUT"
ls "$OUT/instant" | wc -l
ls "$OUT/final" | wc -l
ls "$OUT/reconstructed_molecules_n100" | wc -l
ls "$OUT/plots" | head
