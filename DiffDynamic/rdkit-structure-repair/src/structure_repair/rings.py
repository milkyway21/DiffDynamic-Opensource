"""Ring-structure anomaly detection."""

from __future__ import annotations

from typing import List, Sequence, Set, Tuple

from rdkit import Chem

from .atom_mapping import bond_map_pair
from .models import StructureIssue


def _ring_bonds(mol: Chem.Mol, ring: Sequence[int]) -> List[Chem.Bond]:
    bonds = []
    n = len(ring)
    for i in range(n):
        a, b = ring[i], ring[(i + 1) % n]
        bond = mol.GetBondBetweenAtoms(a, b)
        if bond is not None:
            bonds.append(bond)
    return bonds


def _is_simple_c6_carbon_ring(mol: Chem.Mol, ring: Sequence[int]) -> bool:
    if len(ring) != 6:
        return False
    if any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6 for i in ring):
        return False
    bonds = _ring_bonds(mol, ring)
    if len(bonds) != 6:
        return False
    if any(b.GetBondType() == Chem.BondType.TRIPLE for b in bonds):
        return False
    return True


def _consecutive_double_in_ring(bonds: List[Chem.Bond]) -> bool:
    types = [b.GetBondType() for b in bonds]
    n = len(types)
    for i in range(n):
        if types[i] == Chem.BondType.DOUBLE and types[(i + 1) % n] == Chem.BondType.DOUBLE:
            return True
    return False


def detect_ring_issues(mol: Chem.Mol) -> List[StructureIssue]:
    issues: List[StructureIssue] = []
    try:
        Chem.GetSymmSSSR(mol)
    except Exception:  # noqa: BLE001
        pass
    ring_info = mol.GetRingInfo()

    for ring in ring_info.AtomRings():
        bonds = _ring_bonds(mol, list(ring))
        maps = [mol.GetAtomWithIdx(i).GetAtomMapNum() for i in ring]
        bond_pairs = [
            bond_map_pair(mol, b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in bonds
        ]

        if _is_simple_c6_carbon_ring(mol, ring):
            types = [b.GetBondType() for b in bonds]
            all_single = all(t == Chem.BondType.SINGLE for t in types) and not any(
                b.GetIsAromatic() for b in bonds
            )
            all_double = all(t == Chem.BondType.DOUBLE for t in types)
            has_consec = _consecutive_double_in_ring(bonds)
            aromatic_atoms = all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring)

            # Exocyclic unsaturation / substitution hint for aromatic recovery
            exo_double = False
            for idx in ring:
                atom = mol.GetAtomWithIdx(idx)
                for bond in atom.GetBonds():
                    other = bond.GetOtherAtomIdx(idx)
                    if other not in ring and bond.GetBondType() == Chem.BondType.DOUBLE:
                        exo_double = True

            if all_single and not aromatic_atoms:
                # Estimate ring hydrogen count to distinguish cyclohexane (~12H)
                # from benzene-like all-single corruption (~6H or failed sanitize).
                h_count = 0
                for idx in ring:
                    atom = mol.GetAtomWithIdx(idx)
                    try:
                        h_count += int(atom.GetTotalNumHs(includeNeighbors=False))
                    except Exception:  # noqa: BLE001
                        h_count += int(atom.GetNumExplicitHs())
                sanitize_ok = True
                try:
                    tmp = Chem.Mol(mol)
                    Chem.SanitizeMol(tmp)
                except Exception:  # noqa: BLE001
                    sanitize_ok = False
                looks_aromatic_corrupt = (h_count <= 7) or (not sanitize_ok and h_count <= 9)
                if looks_aromatic_corrupt:
                    issues.append(
                        StructureIssue(
                            issue_code="ALL_SINGLE_CONJUGATED_RING",
                            severity="warning",
                            atom_map_ids=maps,
                            bond_atom_map_pairs=bond_pairs,
                            description="Six-membered all-carbon ring with all single bonds",
                            evidence={"ring_idxs": list(ring), "h_count": h_count},
                            repairable=True,
                            candidate_rule_ids=["C6_AROMATIC_RECOVERY"],
                        )
                    )
                    issues.append(
                        StructureIssue(
                            issue_code="C6_RING_AROMATICITY_LOST",
                            severity="warning",
                            atom_map_ids=maps,
                            bond_atom_map_pairs=bond_pairs,
                            description="Possible lost C6 aromaticity (all single bonds)",
                            evidence={
                                "ring_idxs": list(ring),
                                "exo_double": exo_double,
                                "h_count": h_count,
                            },
                            repairable=True,
                            candidate_rule_ids=["C6_AROMATIC_RECOVERY"],
                        )
                    )

            if all_double:
                issues.append(
                    StructureIssue(
                        issue_code="ALL_DOUBLE_RING",
                        severity="error",
                        atom_map_ids=maps,
                        bond_atom_map_pairs=bond_pairs,
                        description="Ring with all double bonds",
                        evidence={"ring_idxs": list(ring)},
                        repairable=True,
                        candidate_rule_ids=["C6_AROMATIC_RECOVERY", "CUMULENE_VALIDATION_AND_REPAIR"],
                    )
                )

            if has_consec:
                issues.append(
                    StructureIssue(
                        issue_code="CONSECUTIVE_DOUBLE_BONDS_IN_RING",
                        severity="error",
                        atom_map_ids=maps,
                        bond_atom_map_pairs=bond_pairs,
                        description="Consecutive double bonds inside C6 carbon ring",
                        evidence={"ring_idxs": list(ring)},
                        repairable=True,
                        candidate_rule_ids=["C6_AROMATIC_RECOVERY", "CUMULENE_VALIDATION_AND_REPAIR"],
                    )
                )

        # C5 hetero aromaticity lost
        if len(ring) == 5:
            atomic_nums = [mol.GetAtomWithIdx(i).GetAtomicNum() for i in ring]
            hetero = [z for z in atomic_nums if z in (7, 8, 16)]
            if hetero and not any(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
                types = [b.GetBondType() for b in bonds]
                if all(t == Chem.BondType.SINGLE for t in types):
                    issues.append(
                        StructureIssue(
                            issue_code="C5_HETEROAROMATICITY_LOST",
                            severity="warning",
                            atom_map_ids=maps,
                            bond_atom_map_pairs=bond_pairs,
                            description="Five-membered heterocycle with all single bonds",
                            evidence={"ring_idxs": list(ring), "hetero_nums": hetero},
                            repairable=True,
                            candidate_rule_ids=["C5_HETEROAROMATIC_RECOVERY"],
                        )
                    )

        # Multiple double bonds sharing an atom in ring
        for idx in ring:
            dbl = 0
            for bond in mol.GetAtomWithIdx(idx).GetBonds():
                other = bond.GetOtherAtomIdx(idx)
                if other in ring and bond.GetBondType() == Chem.BondType.DOUBLE:
                    dbl += 1
            if dbl >= 2:
                issues.append(
                    StructureIssue(
                        issue_code="MULTIPLE_DOUBLE_BONDS_SHARING_ATOM_IN_RING",
                        severity="error",
                        atom_map_ids=[mol.GetAtomWithIdx(idx).GetAtomMapNum()],
                        bond_atom_map_pairs=[],
                        description="Ring atom participates in multiple double bonds",
                        evidence={"ring_idxs": list(ring)},
                        repairable=True,
                        candidate_rule_ids=["CUMULENE_VALIDATION_AND_REPAIR", "C6_AROMATIC_RECOVERY"],
                    )
                )

        # Triple bond in small ring
        for bond in bonds:
            if bond.GetBondType() == Chem.BondType.TRIPLE and len(ring) <= 7:
                issues.append(
                    StructureIssue(
                        issue_code="TRIPLE_BOND_IN_SMALL_RING",
                        severity="error",
                        atom_map_ids=[
                            bond.GetBeginAtom().GetAtomMapNum(),
                            bond.GetEndAtom().GetAtomMapNum(),
                        ],
                        bond_atom_map_pairs=[
                            bond_map_pair(mol, bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
                        ],
                        description="Triple bond inside small ring",
                        evidence={"ring_size": len(ring)},
                        repairable=False,
                        candidate_rule_ids=[],
                    )
                )

    return issues


def order_ring_atoms(mol: Chem.Mol, ring: Sequence[int]) -> List[int]:
    """Return ring atom indices in cyclic bond order starting at min index."""
    ring_set: Set[int] = set(ring)
    if not ring:
        return []
    start = min(ring)
    ordered = [start]
    prev = -1
    current = start
    for _ in range(len(ring) - 1):
        neighbors = []
        atom = mol.GetAtomWithIdx(current)
        for nbr in atom.GetNeighbors():
            nidx = nbr.GetIdx()
            if nidx in ring_set and nidx != prev:
                neighbors.append(nidx)
        if not neighbors:
            break
        # Prefer unused
        nxt = None
        for n in neighbors:
            if n not in ordered:
                nxt = n
                break
        if nxt is None:
            nxt = neighbors[0]
        prev, current = current, nxt
        ordered.append(current)
    return ordered
