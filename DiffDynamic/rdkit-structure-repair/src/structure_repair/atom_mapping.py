"""Permanent atom-map assignment for audit stability."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from rdkit import Chem


def assign_atom_maps(mol: Chem.Mol) -> Chem.Mol:
    """Assign fixed atom maps (idx+1) where unset. Never overwrite existing maps."""
    copied = Chem.Mol(mol)
    for atom in copied.GetAtoms():
        if atom.GetAtomMapNum() == 0:
            atom.SetAtomMapNum(atom.GetIdx() + 1)
    return copied


def clear_atom_maps_for_smiles(mol: Chem.Mol) -> Chem.Mol:
    """Return a copy with atom maps cleared (for clean canonical SMILES)."""
    copied = Chem.Mol(mol)
    for atom in copied.GetAtoms():
        atom.SetAtomMapNum(0)
    return copied


def atom_map_to_idx(mol: Chem.Mol) -> Dict[int, int]:
    return {atom.GetAtomMapNum(): atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomMapNum() > 0}


def idx_to_atom_map(mol: Chem.Mol) -> Dict[int, int]:
    return {atom.GetIdx(): atom.GetAtomMapNum() for atom in mol.GetAtoms()}


def get_atom_by_map(mol: Chem.Mol, atom_map: int) -> Optional[Chem.Atom]:
    for atom in mol.GetAtoms():
        if atom.GetAtomMapNum() == atom_map:
            return atom
    return None


def bond_map_pair(mol: Chem.Mol, begin_idx: int, end_idx: int) -> Tuple[int, int]:
    m1 = mol.GetAtomWithIdx(begin_idx).GetAtomMapNum()
    m2 = mol.GetAtomWithIdx(end_idx).GetAtomMapNum()
    return (m1, m2) if m1 <= m2 else (m2, m1)


def ordered_ring_atom_maps(mol: Chem.Mol, ring_idxs: List[int]) -> List[int]:
    return [mol.GetAtomWithIdx(i).GetAtomMapNum() for i in ring_idxs]
