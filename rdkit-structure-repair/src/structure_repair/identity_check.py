"""Identity protection comparisons between original and candidate molecules."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

from rdkit import Chem

from .atom_mapping import atom_map_id, atom_map_to_idx


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
        m1, m2 = atom_map_id(a1), atom_map_id(a2)
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


def _parent_indices(original, candidate, parent_index_map=None):
    """Return candidate-index -> original-index correspondence, without guessing."""
    if parent_index_map is not None:
        mapping = dict(parent_index_map)
    else:
        o_map = atom_map_to_idx(original)
        c_map = atom_map_to_idx(candidate)
        mapping = {ci: o_map[m] for m, ci in c_map.items() if m in o_map}
    if len(set(mapping.values())) != len(mapping):
        raise ValueError("non_unique_parent_mapping")
    if any(
        ci < 0 or ci >= candidate.GetNumAtoms()
        or oi < 0 or oi >= original.GetNumAtoms()
        for ci, oi in mapping.items()
    ):
        raise ValueError("parent_mapping_out_of_range")
    return mapping


def compare_atom_isotopes(original, candidate, parent_index_map=None) -> bool:
    """Preserve isotopes on surviving atoms and reject deletion of labelled atoms."""
    mapping = _parent_indices(original, candidate, parent_index_map)
    inverse = {oi: ci for ci, oi in mapping.items()}
    for atom in original.GetAtoms():
        ci = inverse.get(atom.GetIdx())
        if ci is None:
            if atom.GetIsotope():
                return False
        else:
            other = candidate.GetAtomWithIdx(ci)
            if atom.GetIsotope() != other.GetIsotope():
                return False
            if atom.GetIsotope() and atom.GetAtomicNum() != other.GetAtomicNum():
                return False
    return True


def _stereo_neighbours(atom, index_labels):
    labels = []
    for neighbour in atom.GetNeighbors():
        if neighbour.GetAtomicNum() == 1 and neighbour.GetIsotope() == 0:
            labels.append(-1)  # An explicit ordinary H equals the implicit H.
        else:
            label = index_labels.get(neighbour.GetIdx())
            if label is None:
                return None
            labels.append(label)
    if len(labels) == 3:
        labels.append(-1)
    return labels if len(labels) == 4 and len(set(labels)) == 4 else None


def compare_stereochemistry(
    original: Chem.Mol, candidate: Chem.Mol, parent_index_map=None,
) -> bool:
    """Preserve mapped tetrahedral parity, including specified-to-unspecified loss.

    Raw CW/CCW tags depend on neighbour order. Comparing mapped ligand order
    instead of CIP labels also tolerates remote substitutions changing priorities.
    """
    mapping = _parent_indices(original, candidate, parent_index_map)
    inverse = {oi: ci for ci, oi in mapping.items()}
    original_labels = {i: i for i in range(original.GetNumAtoms())}
    chiral = {
        Chem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.ChiralType.CHI_TETRAHEDRAL_CCW,
    }
    for atom in original.GetAtoms():
        before = atom.GetChiralTag()
        if before not in chiral:
            continue
        ci = inverse.get(atom.GetIdx())
        if ci is None:
            return False
        after_atom = candidate.GetAtomWithIdx(ci)
        after = after_atom.GetChiralTag()
        if after not in chiral:
            return False
        old_order = _stereo_neighbours(atom, original_labels)
        new_order = _stereo_neighbours(after_atom, mapping)
        if old_order is None or new_order is None or set(old_order) != set(new_order):
            return False
        permutation = [old_order.index(label) for label in new_order]
        odd = sum(
            permutation[i] > permutation[j]
            for i in range(4) for j in range(i + 1, 4)
        ) % 2
        if (before != after) != bool(odd):
            return False
    return True


def compare_double_bond_stereo(
    original: Chem.Mol, candidate: Chem.Mol, parent_index_map=None,
) -> bool:
    """Preserve specified E/Z (or cis/trans), accounting for reference ligands."""
    mapping = _parent_indices(original, candidate, parent_index_map)
    inverse = {oi: ci for ci, oi in mapping.items()}
    parity = {
        Chem.BondStereo.STEREOZ: 0, Chem.BondStereo.STEREOCIS: 0,
        Chem.BondStereo.STEREOE: 1, Chem.BondStereo.STEREOTRANS: 1,
    }
    for bond in original.GetBonds():
        if bond.GetStereo() not in parity:
            continue
        oi, oj = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if oi not in inverse or oj not in inverse:
            return False
        other = candidate.GetBondBetweenAtoms(inverse[oi], inverse[oj])
        if (other is None or other.GetBondType() != Chem.BondType.DOUBLE
                or other.GetStereo() not in parity):
            return False
        old_refs = list(bond.GetStereoAtoms())
        new_refs = [mapping.get(i) for i in other.GetStereoAtoms()]
        if len(old_refs) != 2 or len(new_refs) != 2 or None in new_refs:
            return False
        if mapping[other.GetBeginAtomIdx()] != oi:
            new_refs.reverse()
        flips = 0
        for endpoint, opposite, old_ref, new_ref in zip(
            (oi, oj), (oj, oi), old_refs, new_refs
        ):
            neighbours = {
                a.GetIdx() for a in original.GetAtomWithIdx(endpoint).GetNeighbors()
                if a.GetIdx() != opposite
            }
            if new_ref not in neighbours:
                return False
            flips += old_ref != new_ref
        if parity[bond.GetStereo()] != (parity[other.GetStereo()] ^ (flips % 2)):
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
    if not compare_atom_isotopes(original, candidate):
        errors.append("isotope_identity_changed")
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
