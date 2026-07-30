"""Shared helpers for applying bond/atom edits on RWMol copies."""

from __future__ import annotations

from typing import List, Optional, Sequence

from rdkit import Chem

from .atom_mapping import atom_map_to_idx, get_atom_by_map
from .models import AtomEdit, BondEdit
from .staged_sanitize import try_full_sanitize


def bond_type_name(bt: Chem.BondType) -> str:
    return str(bt).replace("BondType.", "")


def parse_bond_type(name: Optional[str]) -> Optional[Chem.BondType]:
    if name is None:
        return None
    return getattr(Chem.BondType, name)


def clear_aromatic_flags(rw: Chem.RWMol, atom_idxs: Sequence[int]) -> List[AtomEdit]:
    edits: List[AtomEdit] = []
    atom_set = set(atom_idxs)
    for idx in atom_idxs:
        atom = rw.GetAtomWithIdx(idx)
        if atom.GetIsAromatic():
            edits.append(
                AtomEdit(
                    atom_map_id=atom.GetAtomMapNum(),
                    property_name="is_aromatic",
                    old_value=True,
                    new_value=False,
                )
            )
            atom.SetIsAromatic(False)
    for bond in rw.GetBonds():
        if bond.GetBeginAtomIdx() in atom_set and bond.GetEndAtomIdx() in atom_set:
            if bond.GetIsAromatic():
                bond.SetIsAromatic(False)
    return edits


def set_bond_type(
    rw: Chem.RWMol,
    map1: int,
    map2: int,
    new_type: Chem.BondType,
    aromatic: bool = False,
) -> Optional[BondEdit]:
    mapping = atom_map_to_idx(rw)
    if map1 not in mapping or map2 not in mapping:
        return None
    i1, i2 = mapping[map1], mapping[map2]
    bond = rw.GetBondBetweenAtoms(i1, i2)
    if bond is None:
        return None
    old = bond_type_name(bond.GetBondType())
    bond.SetIsAromatic(False)
    bond.SetBondType(new_type)
    if aromatic:
        bond.SetIsAromatic(True)
        bond.SetBondType(Chem.BondType.AROMATIC)
    new_name = "AROMATIC" if aromatic else bond_type_name(new_type)
    return BondEdit(
        atom_map_1=min(map1, map2),
        atom_map_2=max(map1, map2),
        old_bond_type=old,
        new_bond_type=new_name,
        action="set",
    )


def delete_bond(rw: Chem.RWMol, map1: int, map2: int) -> Optional[BondEdit]:
    mapping = atom_map_to_idx(rw)
    if map1 not in mapping or map2 not in mapping:
        return None
    i1, i2 = mapping[map1], mapping[map2]
    bond = rw.GetBondBetweenAtoms(i1, i2)
    if bond is None:
        return None
    old = bond_type_name(bond.GetBondType())
    rw.RemoveBond(i1, i2)
    return BondEdit(
        atom_map_1=min(map1, map2),
        atom_map_2=max(map1, map2),
        old_bond_type=old,
        new_bond_type=None,
        action="delete",
    )


def set_formal_charge(rw: Chem.RWMol, atom_map: int, charge: int) -> Optional[AtomEdit]:
    atom = get_atom_by_map(rw, atom_map)
    if atom is None:
        return None
    old = atom.GetFormalCharge()
    if old == charge:
        return None
    atom.SetFormalCharge(charge)
    return AtomEdit(
        atom_map_id=atom_map,
        property_name="formal_charge",
        old_value=old,
        new_value=charge,
    )


def finalize_candidate_mol(rw: Chem.RWMol) -> Optional[Chem.Mol]:
    mol = rw.GetMol()
    try:
        mol.UpdatePropertyCache(strict=False)
        Chem.GetSymmSSSR(mol)
    except Exception:  # noqa: BLE001
        pass
    sanitized, _ = try_full_sanitize(mol)
    return sanitized


def molecule_state_hash(mol: Chem.Mol) -> str:
    """Deterministic hash of heavy-atom graph state."""
    parts = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1:
            continue
        parts.append(
            f"{atom.GetAtomMapNum()}:{atom.GetAtomicNum()}:{atom.GetFormalCharge()}:"
            f"{int(atom.GetIsAromatic())}"
        )
    bonds = []
    for bond in mol.GetBonds():
        a1, a2 = bond.GetBeginAtom(), bond.GetEndAtom()
        if a1.GetAtomicNum() == 1 or a2.GetAtomicNum() == 1:
            continue
        m1, m2 = a1.GetAtomMapNum(), a2.GetAtomMapNum()
        if m1 > m2:
            m1, m2 = m2, m1
        bonds.append(
            f"{m1}-{m2}:{bond_type_name(bond.GetBondType())}:{int(bond.GetIsAromatic())}"
        )
    parts.sort()
    bonds.sort()
    return "|".join(parts) + "#" + ";".join(bonds)
