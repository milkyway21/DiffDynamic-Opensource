"""Ring-structure anomaly detection."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from rdkit import Chem

from .atom_mapping import bond_map_pair
from .models import StructureIssue
from .planarity import (
    has_3d_conformer,
    planarity_thresholds_from_config,
    ring_planarity,
)

# Aromatization opportunities are *not* legality defects, so they carry
# severity "info" and repairable=False: the repair layer ignores them
# (see repair_engine actionable filters) while the medchem optimize layer
# treats them as candidate sites.
AROMATIZABLE_C6_RING = "AROMATIZABLE_C6_RING"
AROMATIZABLE_C5_HETEROCYCLE = "AROMATIZABLE_C5_HETEROCYCLE"
AROMATIZATION_OPPORTUNITY_CODES = (
    AROMATIZABLE_C6_RING,
    AROMATIZABLE_C5_HETEROCYCLE,
)


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


def _heavy_degree(mol: Chem.Mol, idx: int) -> int:
    atom = mol.GetAtomWithIdx(int(idx))
    return sum(1 for nbr in atom.GetNeighbors() if nbr.GetAtomicNum() > 1)


def _has_exocyclic_unsaturation(mol: Chem.Mol, idx: int, ring_set: Set[int]) -> bool:
    """True when the atom carries a double/triple bond leaving the ring.

    Cyclohexadienones and exocyclic methylenes are legitimately non-aromatic;
    aromatizing them would destroy a real functional group.
    """
    atom = mol.GetAtomWithIdx(int(idx))
    for bond in atom.GetBonds():
        other = bond.GetOtherAtomIdx(int(idx))
        if other in ring_set:
            continue
        if bond.GetBondType() in (Chem.BondType.DOUBLE, Chem.BondType.TRIPLE):
            return True
    return False


def _carbon_can_be_aromatic(mol: Chem.Mol, idx: int, ring_set: Set[int]) -> bool:
    atom = mol.GetAtomWithIdx(int(idx))
    if atom.GetAtomicNum() != 6:
        return False
    if atom.GetFormalCharge() != 0:
        return False
    if atom.GetNumRadicalElectrons() > 0:
        return False
    if _heavy_degree(mol, idx) > 3:
        return False
    return not _has_exocyclic_unsaturation(mol, idx, ring_set)


def _heteroatom_can_be_aromatic(mol: Chem.Mol, idx: int, ring_set: Set[int]) -> bool:
    atom = mol.GetAtomWithIdx(int(idx))
    z = atom.GetAtomicNum()
    if atom.GetFormalCharge() != 0 or atom.GetNumRadicalElectrons() > 0:
        return False
    if _has_exocyclic_unsaturation(mol, idx, ring_set):
        return False
    degree = _heavy_degree(mol, idx)
    if z == 7:
        # pyrrole-type NH (2 heavy) or N-substituted (3 heavy)
        return degree in (2, 3)
    if z in (8, 16):
        # furan / thiophene oxygen and sulfur are strictly divalent
        return degree == 2
    return False


def _ring_is_fused_to_aromatic(mol: Chem.Mol, ring: Sequence[int]) -> bool:
    ring_set = set(int(i) for i in ring)
    for other in mol.GetRingInfo().AtomRings():
        other_set = set(int(i) for i in other)
        if other_set == ring_set:
            continue
        if len(ring_set & other_set) < 2:
            continue
        if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in other_set):
            return True
    return False


def _ring_hydrogen_count(mol: Chem.Mol, ring: Sequence[int]) -> int:
    total = 0
    for idx in ring:
        atom = mol.GetAtomWithIdx(int(idx))
        try:
            total += int(atom.GetTotalNumHs(includeNeighbors=False))
        except Exception:  # noqa: BLE001
            total += int(atom.GetNumExplicitHs())
    return total


def detect_aromatization_opportunities(
    mol: Chem.Mol,
    config: Optional[Dict[str, Any]] = None,
) -> List[StructureIssue]:
    """Report rings that could plausibly be aromatic but currently are not.

    These are opportunities, not defects: severity is ``info`` and
    ``repairable`` is False so the legality repair layer leaves them alone.
    The medchem optimize layer consumes them and decides using the recorded
    evidence (ring double-bond count, 3D planarity, aromatic fusion).
    """
    config = config or {}
    thresholds = planarity_thresholds_from_config(config)
    have_3d = has_3d_conformer(mol)

    try:
        Chem.GetSymmSSSR(mol)
    except Exception:  # noqa: BLE001
        pass

    issues: List[StructureIssue] = []
    for ring in mol.GetRingInfo().AtomRings():
        size = len(ring)
        if size not in (5, 6):
            continue
        ring_set = set(int(i) for i in ring)
        if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring_set):
            continue

        ordered = order_ring_atoms(mol, list(ring))
        if len(ordered) != size:
            continue
        bonds = _ring_bonds(mol, ordered)
        if len(bonds) != size:
            continue
        if any(b.GetBondType() == Chem.BondType.TRIPLE for b in bonds):
            continue

        atomic_nums = [mol.GetAtomWithIdx(i).GetAtomicNum() for i in ordered]
        hetero_idxs = [i for i in ordered if mol.GetAtomWithIdx(i).GetAtomicNum() != 6]

        if size == 6:
            # Only all-carbon benzene recovery here; aza-scan lives in the T1 catalog.
            if any(z != 6 for z in atomic_nums):
                continue
            if not all(_carbon_can_be_aromatic(mol, i, ring_set) for i in ordered):
                continue
            issue_code = AROMATIZABLE_C6_RING
        else:
            if not hetero_idxs or len(hetero_idxs) > 2:
                continue
            ok = True
            for i in ordered:
                if mol.GetAtomWithIdx(i).GetAtomicNum() == 6:
                    ok = _carbon_can_be_aromatic(mol, i, ring_set)
                else:
                    ok = _heteroatom_can_be_aromatic(mol, i, ring_set)
                if not ok:
                    break
            if not ok:
                continue
            issue_code = AROMATIZABLE_C5_HETEROCYCLE

        n_double = sum(1 for b in bonds if b.GetBondType() == Chem.BondType.DOUBLE)
        planarity = (
            ring_planarity(
                mol,
                ordered,
                rmsd_threshold=thresholds["rmsd_threshold"],
                torsion_threshold=thresholds["torsion_threshold"],
            )
            if have_3d
            else None
        )
        planar = planarity["planar"] if planarity else None

        # A fully saturated, clearly puckered ring is a real aliphatic ring.
        if n_double == 0 and planar is not True:
            continue

        evidence: Dict[str, Any] = {
            "ring_idxs": list(ordered),
            "ring_size": size,
            "n_ring_double_bonds": n_double,
            "ring_h_count": _ring_hydrogen_count(mol, ordered),
            "fused_to_aromatic": _ring_is_fused_to_aromatic(mol, ordered),
            "hetero_atomic_nums": [
                mol.GetAtomWithIdx(i).GetAtomicNum() for i in hetero_idxs
            ],
            "planar": planar,
        }
        if planarity:
            evidence.update(planarity)

        issues.append(
            StructureIssue(
                issue_code=issue_code,
                severity="info",
                atom_map_ids=[mol.GetAtomWithIdx(i).GetAtomMapNum() for i in ordered],
                bond_atom_map_pairs=[
                    bond_map_pair(mol, b.GetBeginAtomIdx(), b.GetEndAtomIdx())
                    for b in bonds
                ],
                description=(
                    f"{size}-membered ring with {n_double} ring double bond(s) "
                    f"is a candidate for aromatization"
                ),
                evidence=evidence,
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
