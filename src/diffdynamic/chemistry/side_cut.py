#!/usr/bin/env python3
"""Largest-fragment + single-dummy side-chain cuts.

Whole molecule: keep the heaviest connected fragment, drop the rest.
Scaffold extras: keep the largest attached component and place exactly one
attachment dummy on the distal (NHO / SMARTS-last) cut bond.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from rdkit import Chem
from rdkit.Geometry import Point3D

DUMMY_ATOMIC_NUM = 0
DUMMY_ISOTOPE = 1
DUMMY_MAP = 1


def heavy_atom_count(molecule: Chem.Mol) -> int:
    return sum(1 for atom in molecule.GetAtoms() if atom.GetAtomicNum() > 1)


def dummy_count(molecule: Chem.Mol) -> int:
    return sum(
        1 for atom in molecule.GetAtoms() if atom.GetAtomicNum() == DUMMY_ATOMIC_NUM
    )


def fragment_count(molecule: Chem.Mol) -> int:
    return len(Chem.GetMolFrags(molecule, asMols=False))


def keep_largest_fragment(molecule: Chem.Mol) -> Optional[Chem.Mol]:
    """Keep the connected component with the most heavy atoms."""
    if molecule is None:
        return None
    fragments = Chem.GetMolFrags(molecule, asMols=True, sanitizeFrags=False)
    if not fragments:
        return None
    best = max(
        fragments,
        key=lambda item: (heavy_atom_count(item), item.GetNumAtoms()),
    )
    _lenient_sanitize(best)
    return best


def _lenient_sanitize(molecule: Chem.Mol) -> bool:
    try:
        Chem.SanitizeMol(molecule)
        return True
    except Exception:
        pass
    try:
        Chem.SanitizeMol(
            molecule,
            sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL
            ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE,
        )
        return True
    except Exception:
        return False


def is_scoreable_side(molecule: Optional[Chem.Mol]) -> bool:
    """Hard gate: one fragment, exactly one dummy of degree 1."""
    if molecule is None:
        return False
    if dummy_count(molecule) != 1:
        return False
    if fragment_count(molecule) != 1:
        return False
    dummy = next(
        atom for atom in molecule.GetAtoms() if atom.GetAtomicNum() == DUMMY_ATOMIC_NUM
    )
    if dummy.GetDegree() != 1:
        return False
    if "." in (Chem.MolToSmiles(molecule, isomericSmiles=False) or ""):
        return False
    return True


def is_scoreable_side_smiles(smiles: str) -> bool:
    if not smiles or "." in smiles:
        return False
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        molecule = Chem.MolFromSmiles(smiles, sanitize=False)
    return is_scoreable_side(molecule)


def dummy_stripped_largest(molecule: Chem.Mol) -> Optional[Chem.Mol]:
    editable = Chem.RWMol(Chem.Mol(molecule))
    dummy_indices = [
        int(atom.GetIdx())
        for atom in molecule.GetAtoms()
        if atom.GetAtomicNum() == DUMMY_ATOMIC_NUM
    ]
    for index in sorted(dummy_indices, reverse=True):
        editable.RemoveAtom(index)
    stripped = editable.GetMol()
    return keep_largest_fragment(stripped)


def _components(molecule: Chem.Mol, atom_indices: Iterable[int]) -> List[set]:
    allowed = {int(index) for index in atom_indices}
    components: List[set] = []
    seen: set = set()
    for start in sorted(allowed):
        if start in seen:
            continue
        queue = [start]
        seen.add(start)
        component: set = set()
        while queue:
            current = queue.pop()
            component.add(current)
            for neighbour in molecule.GetAtomWithIdx(current).GetNeighbors():
                index = int(neighbour.GetIdx())
                if index in allowed and index not in seen:
                    seen.add(index)
                    queue.append(index)
        components.append(component)
    return components


def cut_largest_attached_side(
    molecule: Chem.Mol,
    scaffold_indices: Sequence[int],
) -> Tuple[Optional[Chem.Mol], Dict[str, Any]]:
    """Cut extras outside ``scaffold_indices``; one dummy on the distal bond."""
    scaffold = set(int(index) for index in scaffold_indices)
    extra = {
        int(atom.GetIdx())
        for atom in molecule.GetAtoms()
        if int(atom.GetIdx()) not in scaffold and atom.GetAtomicNum() > 1
    }
    if not extra:
        return None, {"status": "no_target_side"}

    attached: List[Tuple[set, List[Tuple[int, int, Chem.BondType, int]]]] = []
    detached: List[set] = []
    for component in _components(molecule, extra):
        cross_bonds: List[Tuple[int, int, Chem.BondType, int]] = []
        for extra_index in sorted(component):
            for neighbour in molecule.GetAtomWithIdx(extra_index).GetNeighbors():
                scaffold_index = int(neighbour.GetIdx())
                if scaffold_index not in scaffold:
                    continue
                bond = molecule.GetBondBetweenAtoms(extra_index, scaffold_index)
                if bond is None:
                    continue
                bond_type = bond.GetBondType()
                if bond_type == Chem.BondType.AROMATIC:
                    bond_type = Chem.BondType.SINGLE
                cross_bonds.append(
                    (extra_index, scaffold_index, bond_type, int(bond.GetIdx()))
                )
        if cross_bonds:
            attached.append((component, cross_bonds))
        else:
            detached.append(component)
    if not attached:
        return None, {
            "status": "no_scaffold_cross_bond",
            "attached_component_count": 0,
            "detached_component_count": len(detached),
        }

    attached.sort(key=lambda item: (len(item[0]), -min(item[0])), reverse=True)
    keep, cross_bonds = attached[0]
    rank = {int(index): position for position, index in enumerate(scaffold_indices)}
    cross_bonds.sort(
        key=lambda item: (rank.get(int(item[1]), -1), item[0], item[3]),
        reverse=True,
    )
    extra_index, scaffold_index, bond_type, _bond_index = cross_bonds[0]

    conformer_coordinates: List[Optional[Tuple[float, float, float]]] = []
    for conformer in molecule.GetConformers():
        pos = conformer.GetAtomPosition(int(scaffold_index))
        conformer_coordinates.append((float(pos.x), float(pos.y), float(pos.z)))

    editable = Chem.RWMol(Chem.Mol(molecule))
    for atom in editable.GetAtoms():
        atom.SetIntProp("_side_cut_original_index", int(atom.GetIdx()))
    for atom_index in sorted(set(range(molecule.GetNumAtoms())) - keep, reverse=True):
        editable.RemoveAtom(int(atom_index))
    local_by_original = {
        int(atom.GetIntProp("_side_cut_original_index")): int(atom.GetIdx())
        for atom in editable.GetAtoms()
    }
    local = local_by_original.get(int(extra_index))
    if local is None:
        return None, {"status": "side_attachment_atom_missing"}
    dummy = Chem.Atom(DUMMY_ATOMIC_NUM)
    dummy.SetIsotope(DUMMY_ISOTOPE)
    dummy.SetAtomMapNum(DUMMY_MAP)
    dummy.SetNoImplicit(True)
    dummy_index = editable.AddAtom(dummy)
    try:
        editable.AddBond(local, dummy_index, bond_type)
    except Exception:
        return None, {"status": "side_attachment_bond_failed"}
    side = editable.GetMol()
    if not _lenient_sanitize(side):
        return None, {"status": "side_sanitize_failed"}
    for conformer_index, position in enumerate(conformer_coordinates):
        if position is None or conformer_index >= side.GetNumConformers():
            continue
        side.GetConformer(conformer_index).SetAtomPosition(
            dummy_index, Point3D(*position)
        )
    for atom in side.GetAtoms():
        if atom.HasProp("_side_cut_original_index"):
            atom.ClearProp("_side_cut_original_index")
    side.SetProp("_SideDummy", "[1*:1]")
    if not is_scoreable_side(side):
        return None, {
            "status": "not_scoreable",
            "dummy_count": dummy_count(side),
            "fragment_count": fragment_count(side),
        }
    return side, {
        "status": "ok",
        "outside_component_count": 1,
        "attached_component_count": 1,
        "raw_attached_component_count": int(len(attached)),
        "absorbed_scaffold_substituent_count": int(len(attached) - 1),
        "detached_component_count": len(detached),
        "detached_heavy_atoms": sum(len(component) for component in detached),
        "side_heavy_atoms": int(len(keep)),
        "cross_bond_count": 1,
        "cut_rule": "largest_attached_single_dummy",
        "sanitize_mode": "lenient",
    }


def match_scaffold_smarts(
    molecule: Chem.Mol,
    smarts_list: Sequence[str],
) -> Tuple[Optional[Tuple[int, ...]], str]:
    for smarts in smarts_list:
        if not smarts:
            continue
        query = Chem.MolFromSmarts(smarts)
        if query is None:
            continue
        match = molecule.GetSubstructMatch(query)
        if match:
            return tuple(int(index) for index in match), smarts
    return None, ""


def cut_ikzf2_molecule(
    molecule: Chem.Mol,
    smarts_list: Sequence[str],
    n_scaffold_fallback: int = 0,
) -> Tuple[Optional[Chem.Mol], Dict[str, Any]]:
    """Largest fragment, then SMARTS (NHO) or locked prefix fallback."""
    whole = keep_largest_fragment(molecule)
    if whole is None:
        return None, {"status": "no_fragment"}
    match, used = match_scaffold_smarts(whole, smarts_list)
    cut_rule = "smarts"
    if (
        match is None
        and n_scaffold_fallback
        and whole.GetNumAtoms() > n_scaffold_fallback
    ):
        match = tuple(range(int(n_scaffold_fallback)))
        cut_rule = "n_scaffold_fallback"
        used = "n_scaffold"
    if match is None:
        return None, {"status": "no_scaffold"}
    side, meta = cut_largest_attached_side(whole, match)
    if side is None:
        return None, meta
    meta = dict(meta)
    meta["cut_rule"] = cut_rule + "/" + str(meta.get("cut_rule") or "")
    meta["smarts"] = used
    return side, meta
