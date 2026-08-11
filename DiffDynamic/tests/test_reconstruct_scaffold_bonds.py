"""骨架键保真：参考拓扑在距离成键时不被破坏。"""
from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from utils.reconstruct import (
    _reconnect_scaffold_extra_components,
    _repair_scaffold_extra_bonds,
    reconstruct_from_generated,
)


def _benzene_scaffold_with_nearby_extra():
    """苯环骨架 + 一个极近的额外 C（易被乱连键）。"""
    mol = Chem.MolFromSmiles('c1ccccc1')
    AllChem.EmbedMolecule(mol, randomSeed=0)
    AllChem.UFFOptimizeMolecule(mol)
    conf = mol.GetConformer()
    xyz = []
    atomic = []
    for i in range(mol.GetNumAtoms()):
        p = conf.GetAtomPosition(i)
        xyz.append([p.x, p.y, p.z])
        atomic.append(6)
    # 额外原子贴在原子 0 附近
    p0 = np.array(xyz[0])
    xyz.append((p0 + np.array([0.8, 0.0, 0.0])).tolist())
    atomic.append(6)

    bonds = []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        aromatic = b.GetIsAromatic() or b.GetBondType() == Chem.BondType.AROMATIC
        order = 5 if aromatic else int(round(b.GetBondTypeAsDouble()))
        bonds.append((i, j, order, aromatic))
    return np.asarray(xyz, dtype=np.float64), atomic, bonds, 6


def test_scaffold_bonds_preserve_benzene_ring():
    xyz, atomic, bonds, n_sc = _benzene_scaffold_with_nearby_extra()
    mol = reconstruct_from_generated(
        xyz, atomic, aromatic=[True] * n_sc + [False],
        basic_mode=False,
        scaffold_bonds=bonds,
        n_scaffold=n_sc,
    )
    assert mol is not None
    # 骨架 6 原子之间应保留环键（度数 ≥2）
    for i in range(n_sc):
        deg = mol.GetAtomWithIdx(i).GetDegree()
        assert deg >= 2, (i, deg)
    em = Chem.EditableMol(Chem.Mol())
    amap = {}
    for i in range(n_sc):
        amap[i] = em.AddAtom(Chem.Atom(mol.GetAtomWithIdx(i).GetAtomicNum()))
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if i < n_sc and j < n_sc:
            em.AddBond(amap[i], amap[j], b.GetBondType())
    sc = em.GetMol()
    try:
        Chem.SanitizeMol(sc)
    except Exception:
        pass
    smi = Chem.MolToSmiles(sc)
    assert 'c1ccccc1' in smi or smi.count('c') >= 6 or 'C1=CC=CC=C1' in smi, smi


def test_scaffold_attachment_bond_keeps_distant_generated_branch():
    xyz, atomic, bonds, n_sc = _benzene_scaffold_with_nearby_extra()
    xyz[-1] = xyz[0] + np.array([4.0, 0.0, 0.0])
    mol = reconstruct_from_generated(
        xyz, atomic, aromatic=[True] * n_sc + [False],
        basic_mode=False,
        scaffold_bonds=bonds,
        n_scaffold=n_sc,
        extra_attachment_bonds=[(0, n_sc, 1, False)],
    )
    assert mol is not None
    assert len(Chem.GetMolFrags(mol, asMols=False)) == 1
    assert mol.GetBondBetweenAtoms(0, n_sc) is not None


def test_scaffold_component_reconnect_prefers_generated_atoms():
    mol = Chem.MolFromSmiles('c1ccccc1C.C')
    conformer = Chem.Conformer(mol.GetNumAtoms())
    for idx in range(mol.GetNumAtoms()):
        conformer.SetAtomPosition(idx, (float(idx), 0.0, 0.0))
    conformer.SetAtomPosition(6, (1.5, 0.0, 0.0))
    conformer.SetAtomPosition(7, (3.3, 0.0, 0.0))
    mol.AddConformer(conformer)

    repaired = _reconnect_scaffold_extra_components(mol, n_scaffold=6)
    assert len(Chem.GetMolFrags(repaired, asMols=False)) == 1
    assert repaired.GetBondBetweenAtoms(6, 7) is not None


def test_without_scaffold_bonds_still_runs():
    xyz, atomic, _, _ = _benzene_scaffold_with_nearby_extra()
    mol = reconstruct_from_generated(xyz, atomic, basic_mode=True)
    assert mol is not None


def test_scaffold_repair_downgrades_erroneous_aromatic_external_double_bond():
    scaffold = Chem.MolFromSmiles('c1ccccc1')
    editable = Chem.RWMol(scaffold)
    extra_idx = editable.AddAtom(Chem.Atom(6))
    editable.AddBond(0, extra_idx, Chem.BondType.DOUBLE)
    noisy = editable.GetMol()
    scaffold_bonds = [
        (
            bond.GetBeginAtomIdx(),
            bond.GetEndAtomIdx(),
            1,
            True,
        )
        for bond in scaffold.GetBonds()
    ]

    repaired = _repair_scaffold_extra_bonds(
        noisy, scaffold_bonds, n_scaffold=6,
    )
    external = repaired.GetBondBetweenAtoms(0, extra_idx)
    assert external is not None
    assert external.GetBondType() == Chem.BondType.SINGLE
    Chem.SanitizeMol(repaired)


if __name__ == '__main__':
    test_scaffold_bonds_preserve_benzene_ring()
    test_without_scaffold_bonds_still_runs()
    print('All reconstruct scaffold bond tests passed.')
