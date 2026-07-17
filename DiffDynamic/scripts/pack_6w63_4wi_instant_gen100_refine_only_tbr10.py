#!/usr/bin/env python3
"""Rebuild final_*.sdf from generation .pt (1:1 with instant), defragment, write pairs_manifest."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

from utils import reconstruct
import utils.transforms as trans

OUT = Path('experiments/6w63_4wi_instant_gen100_refine_only_tbr10')
FINAL = OUT / 'final'
FINAL.mkdir(parents=True, exist_ok=True)
for f in FINAL.glob('*'):
    f.unlink()

manifest = json.loads((OUT / 'instant_manifest.json').read_text())
pt = sorted((OUT / 'run').glob('result_custom_*.pt'))[-1]
data = torch.load(pt, map_location='cpu')
pos_list = data['pred_ligand_pos']
v_list = data['pred_ligand_v']
n_entries = len(pos_list)
print(f'pt={pt.name} n_entries={n_entries}')

xlsx_list = sorted((OUT / 'run').glob('evaluation_results_*.xlsx'))
df = pd.read_excel(xlsx_list[-1], sheet_name='评估结果') if xlsx_list else None
chooser = rdMolStandardize.LargestFragmentChooser()


def largest_frag(mol):
    if mol is None:
        return None
    try:
        out = chooser.choose(mol)
        if out is not None:
            return out
    except Exception:
        pass
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    return max(frags, key=lambda m: m.GetNumAtoms()) if frags else mol


def reconstruct_one(pos, v):
    pos = pos.numpy() if hasattr(pos, 'numpy') else np.asarray(pos)
    v = v.numpy() if hasattr(v, 'numpy') else np.asarray(v)
    if v.ndim > 1:
        atom_type = v.argmax(axis=-1)
    else:
        atom_type = v.astype(int)
    try:
        atomic_nums = trans.get_atomic_number_from_index(atom_type, mode='add_aromatic')
        aromatic = trans.is_aromatic_from_index(atom_type, mode='add_aromatic')
        mol = reconstruct.reconstruct_from_generated(pos, atomic_nums, aromatic)
        return mol, True
    except Exception:
        return None, False


n_sc = int(Chem.SDMolSupplier(str(OUT / 'instant' / '00_scaffold_only.sdf'), removeHs=False)[0].GetNumAtoms())
gen_indices = []
for i in range(n_entries):
    n_at = int(np.asarray(pos_list[i]).shape[0])
    if n_at == n_sc and len(gen_indices) == 0 and i == 0:
        continue
    gen_indices.append(i)
print(f'scaffold_atoms={n_sc} gen_indices={len(gen_indices)} (expect 100)')

all_w = Chem.SDWriter(str(FINAL / 'all_100_final.sdf'))
pairs = []
n_ok = 0
for sample_idx, entry_i in enumerate(gen_indices[:100]):
    row = manifest[sample_idx] if sample_idx < len(manifest) else {'sample_idx': sample_idx}
    mol, ok = reconstruct_one(pos_list[entry_i], v_list[entry_i])
    had_frag = False
    smiles = None
    vina = qed = sa = logp = None
    path = FINAL / f'final_{sample_idx:03d}.sdf'
    if mol is not None:
        frags = Chem.GetMolFrags(mol, asMols=False)
        had_frag = len(frags) > 1
        mol = largest_frag(mol)
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            pass
        try:
            smiles = Chem.MolToSmiles(mol)
        except Exception:
            pass
        mol.SetProp('_Name', f'final_{sample_idx:03d}')
        mol.SetProp('sample_idx', str(sample_idx))
        mol.SetProp('reconstruct_ok', str(ok))
        mol.SetProp('fragments_removed', str(had_frag).lower())
        if smiles:
            mol.SetProp('SMILES', smiles)
        if row.get('instant_sdf'):
            mol.SetProp('paired_instant', str(row['instant_sdf']))
        w = Chem.SDWriter(str(path))
        w.write(mol)
        w.close()
        all_w.write(mol)
        n_ok += 1

    if df is not None and smiles:
        hit = df[df['SMILES'].astype(str) == smiles]
        if len(hit) == 0:
            for j, r in df.iterrows():
                s = str(r.get('SMILES', ''))
                m2 = Chem.MolFromSmiles(s) if s and s != 'nan' else None
                if m2 is None:
                    continue
                try:
                    s2 = Chem.MolToSmiles(chooser.choose(m2))
                except Exception:
                    s2 = Chem.MolToSmiles(m2)
                if s2 == smiles:
                    hit = df.iloc[[df.index.get_loc(j)]]
                    break
        if len(hit):
            r = hit.iloc[0]
            vina = r.get('Vina_Dock_亲和力')
            qed = r.get('QED评分')
            sa = r.get('SA评分')
            logp = r.get('logP')

    def _f(x):
        if x is None:
            return None
        try:
            if pd.isna(x):
                return None
        except Exception:
            pass
        return float(x)

    out = dict(row)
    out.update({
        'final_sdf': str(path),
        'final_n_atoms': None if mol is None else mol.GetNumAtoms(),
        'smiles': smiles,
        'reconstruct_ok': ok,
        'fragments_removed': had_frag,
        'vina_dock': _f(vina),
        'qed': _f(qed),
        'sa': _f(sa),
        'logp': _f(logp),
    })
    pairs.append(out)
    if sample_idx < 3 or sample_idx % 20 == 0:
        print(f'[{sample_idx:03d}] ok={ok} frag={had_frag} n={out["final_n_atoms"]} smi={smiles}')

all_w.close()
(OUT / 'pairs_manifest.json').write_text(json.dumps(pairs, indent=2, ensure_ascii=False, default=str))
pd.DataFrame(pairs).to_csv(OUT / 'pairs_manifest.csv', index=False)
(OUT / 'README.txt').write_text(
    '6W63 4WI refine_only: 100 instant + DiffDynamic refine only\n'
    '================================================\n'
    'Placement: removed-site centroid + N(0,1), max_active_sites=1\n'
    'instant/ — scaffold bonded + clustered discrete C\n'
    'final/   — post-gen reconstructed (defragmented), 1:1 with instant\n'
    'run/     — .pt + evaluation_results\n'
    'RNG: np.random.default_rng(42 + sample_idx * 9973)\n'
)
print(f'DONE reconstruct_ok={n_ok}/100 → {OUT}')
