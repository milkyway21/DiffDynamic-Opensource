"""Identity protection comparisons between original and candidate molecules."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

from rdkit import Chem

from .atom_mapping import atom_map_to_idx


def _heavy_atoms(mol: Chem.Mol) -> List[Chem.Atom]:
    return [a for a in mol.GetAtoms() if a.GetAtomicNum() > 1]


def compare_atom_elements(original: Chem.Mol, candidate: Chem.Mol) -> bool:
    o_map = atom_map_to_idx(original)
    c_map = atom_map_to_idx(candidate)
    if set(o_map) != set(c_map):
        # Allow missing H maps only
        o_heavy = {
            m
            for m, i in o_map.items()
            if original.GetAtomWithIdx(i).GetAtomicNum() > 1
        }
        c_heavy = {
            m
            for m, i in c_map.items()
            if candidate.GetAtomWithIdx(i).GetAtomicNum() > 1
        }
        if o_heavy != c_heavy:
            return False
        shared = o_heavy
    else:
        shared = set(o_map)
    for m in shared:
        if original.GetAtomWithIdx(o_map[m]).GetAtomicNum() != candidate.GetAtomWithIdx(
            c_map[m]
        ).GetAtomicNum():
            return False
    return True


def compare_heavy_atom_count(original: Chem.Mol, candidate: Chem.Mol) -> bool:
    return len(_heavy_atoms(original)) == len(_heavy_atoms(candidate))


def _heavy_connectivity(mol: Chem.Mol) -> Set[Tuple[int, int]]:
    edges = set()
    for bond in mol.GetBonds():
        a1, a2 = bond.GetBeginAtom(), bond.GetEndAtom()
        if a1.GetAtomicNum() == 1 or a2.GetAtomicNum() == 1:
            continue
        m1, m2 = a1.GetAtomMapNum(), a2.GetAtomMapNum()
        edges.add((m1, m2) if m1 <= m2 else (m2, m1))
    return edges


def compare_heavy_atom_connectivity(
    original: Chem.Mol,
    candidate: Chem.Mol,
    allowed_deleted: Optional[Set[Tuple[int, int]]] = None,
    allowed_added: Optional[Set[Tuple[int, int]]] = None,
) -> bool:
    o = _heavy_connectivity(original)
    c = _heavy_connectivity(candidate)
    deleted = o - c
    added = c - o
    allowed_deleted = allowed_deleted or set()
    allowed_added = allowed_added or set()
    return deleted.issubset(allowed_deleted) and added.issubset(allowed_added)


def compare_ring_membership(original: Chem.Mol, candidate: Chem.Mol) -> bool:
    """Soft check: same number of SSSR rings (strict ring-atom sets may differ after aromaticity)."""
    try:
        Chem.GetSymmSSSR(original)
        Chem.GetSymmSSSR(candidate)
    except Exception:  # noqa: BLE001
        return True
    return original.GetRingInfo().NumRings() == candidate.GetRingInfo().NumRings()


def compare_stereochemistry(original: Chem.Mol, candidate: Chem.Mol) -> bool:
    """Reject if tetrahedral chirality tags flip for mapped atoms."""
    o_map = atom_map_to_idx(original)
    c_map = atom_map_to_idx(candidate)
    for m, oi in o_map.items():
        if m not in c_map:
            continue
        o_atom = original.GetAtomWithIdx(oi)
        c_atom = candidate.GetAtomWithIdx(c_map[m])
        ot = o_atom.GetChiralTag()
        ct = c_atom.GetChiralTag()
        if ot in (
            Chem.ChiralType.CHI_TETRAHEDRAL_CW,
            Chem.ChiralType.CHI_TETRAHEDRAL_CCW,
        ) and ct in (
            Chem.ChiralType.CHI_TETRAHEDRAL_CW,
            Chem.ChiralType.CHI_TETRAHEDRAL_CCW,
        ):
            if ot != ct:
                return False
    return True


def compare_double_bond_stereo(original: Chem.Mol, candidate: Chem.Mol) -> bool:
    o_bonds = {}
    for bond in original.GetBonds():
        if bond.GetBondType() != Chem.BondType.DOUBLE:
            continue
        m1, m2 = bond.GetBeginAtom().GetAtomMapNum(), bond.GetEndAtom().GetAtomMapNum()
        key = (m1, m2) if m1 <= m2 else (m2, m1)
        o_bonds[key] = bond.GetStereo()
    for bond in candidate.GetBonds():
        if bond.GetBondType() != Chem.BondType.DOUBLE:
            continue
        m1, m2 = bond.GetBeginAtom().GetAtomMapNum(), bond.GetEndAtom().GetAtomMapNum()
        key = (m1, m2) if m1 <= m2 else (m2, m1)
        if key in o_bonds:
            ost = o_bonds[key]
            cst = bond.GetStereo()
            if ost in (Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOZ) and cst in (
                Chem.BondStereo.STEREOE,
                Chem.BondStereo.STEREOZ,
            ):
                if ost != cst:
                    return False
    return True


def compare_total_formal_charge(original: Chem.Mol, candidate: Chem.Mol) -> int:
    o = sum(a.GetFormalCharge() for a in original.GetAtoms())
    c = sum(a.GetFormalCharge() for a in candidate.GetAtoms())
    return c - o


def identity_ok(
    original: Chem.Mol,
    candidate: Chem.Mol,
    config: Dict[str, Any],
    allowed_deleted_bonds: Optional[Set[Tuple[int, int]]] = None,
    allowed_added_bonds: Optional[Set[Tuple[int, int]]] = None,
    allow_charge_change: bool = True,
) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    ident = config.get("identity", {})
    stereo = config.get("stereo", {})

    if ident.get("preserve_elements", True) and not compare_atom_elements(original, candidate):
        errors.append("element_identity_changed")
    if ident.get("preserve_heavy_atom_count", True) and not compare_heavy_atom_count(
        original, candidate
    ):
        errors.append("heavy_atom_count_changed")

    if not compare_heavy_atom_connectivity(
        original,
        candidate,
        allowed_deleted=allowed_deleted_bonds if ident.get("allow_documented_bond_deletion", True) else set(),
        allowed_added=allowed_added_bonds or set(),
    ):
        errors.append("unauthorized_connectivity_change")

    if stereo.get("preserve_tetrahedral_centers", True) and not compare_stereochemistry(
        original, candidate
    ):
        errors.append("stereochemistry_changed")
    if stereo.get("preserve_double_bond_stereo", True) and not compare_double_bond_stereo(
        original, candidate
    ):
        errors.append("double_bond_stereo_changed")

    if not allow_charge_change:
        if compare_total_formal_charge(original, candidate) != 0:
            errors.append("total_formal_charge_changed")

    return len(errors) == 0, errors
