"""Permanent atom-map assignment for audit stability."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from rdkit import Chem

AUDIT_MAP_PROP = "_sr_atom_map"


def atom_map_id(atom: Chem.Atom) -> int:
    """Read the public map or the provenance retained on a clean output copy."""
    value = atom.GetAtomMapNum()
    if not value and atom.HasProp(AUDIT_MAP_PROP):
        value = atom.GetIntProp(AUDIT_MAP_PROP)
    return value


def assign_atom_maps(mol: Chem.Mol) -> Chem.Mol:
    """Fill missing maps without collisions; reject ambiguous existing maps."""
    copied = Chem.Mol(mol)
    existing = [atom_map_id(atom) for atom in copied.GetAtoms()]
    positive = [value for value in existing if value > 0]
    if len(positive) != len(set(positive)):
        raise ValueError("duplicate_atom_maps")
    used = set(positive)
    next_map = 1
    for atom in copied.GetAtoms():
        value = atom_map_id(atom)
        if value <= 0:
            while next_map in used:
                next_map += 1
            value = next_map
            used.add(value)
        atom.SetAtomMapNum(value)
        atom.SetIntProp(AUDIT_MAP_PROP, value)
    return copied


def clear_atom_maps_for_smiles(mol: Chem.Mol) -> Chem.Mol:
    """Return a copy with atom maps cleared (for clean canonical SMILES)."""
    copied = Chem.Mol(mol)
    for atom in copied.GetAtoms():
        value = atom_map_id(atom)
        if value > 0:
            atom.SetIntProp(AUDIT_MAP_PROP, value)
        atom.SetAtomMapNum(0)
    return copied


def atom_map_to_idx(mol: Chem.Mol) -> Dict[int, int]:
    result = {}
    for atom in mol.GetAtoms():
        value = atom_map_id(atom)
        if value <= 0:
            continue
        if value in result:
            raise ValueError("duplicate_atom_maps")
        result[value] = atom.GetIdx()
    return result


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
