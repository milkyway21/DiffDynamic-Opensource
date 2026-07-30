"""Aromaticity inconsistency detection."""

from __future__ import annotations

from typing import List

from rdkit import Chem

from .atom_mapping import bond_map_pair
from .models import StructureIssue
from .staged_sanitize import staged_sanitize


def detect_aromaticity_issues(mol: Chem.Mol) -> List[StructureIssue]:
    issues: List[StructureIssue] = []
    try:
        mol.UpdatePropertyCache(strict=False)
        Chem.GetSymmSSSR(mol)
    except Exception:  # noqa: BLE001
        pass

    report = staged_sanitize(mol)
    if "KEKULIZATION_FAILED" in report.failure_types:
        aromatic_atoms = [a.GetAtomMapNum() for a in mol.GetAtoms() if a.GetIsAromatic()]
        issues.append(
            StructureIssue(
                issue_code="KEKULIZATION_FAILED",
                severity="error",
                atom_map_ids=aromatic_atoms,
                bond_atom_map_pairs=[],
                description="Kekulization failed during staged sanitization",
                evidence={"failure_types": report.failure_types},
                repairable=True,
                candidate_rule_ids=["C6_AROMATIC_RECOVERY", "C5_HETEROAROMATIC_RECOVERY"],
            )
        )

    ring_info = mol.GetRingInfo()
    ring_bonds = set()
    for ring in ring_info.BondRings():
        ring_bonds.update(ring)

    for bond in mol.GetBonds():
        if bond.GetIsAromatic() and bond.GetIdx() not in ring_bonds:
            issues.append(
                StructureIssue(
                    issue_code="AROMATIC_BOND_OUTSIDE_RING",
                    severity="error",
                    atom_map_ids=[
                        bond.GetBeginAtom().GetAtomMapNum(),
                        bond.GetEndAtom().GetAtomMapNum(),
                    ],
                    bond_atom_map_pairs=[
                        bond_map_pair(mol, bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
                    ],
                    description="Aromatic bond flagged outside any ring",
                    evidence={},
                    repairable=True,
                    candidate_rule_ids=["C6_AROMATIC_RECOVERY"],
                )
            )

        a1, a2 = bond.GetBeginAtom(), bond.GetEndAtom()
        if bond.GetIsAromatic() and (not a1.GetIsAromatic() or not a2.GetIsAromatic()):
            issues.append(
                StructureIssue(
                    issue_code="AROMATIC_BOND_WITH_NONAROMATIC_ATOM",
                    severity="warning",
                    atom_map_ids=[a1.GetAtomMapNum(), a2.GetAtomMapNum()],
                    bond_atom_map_pairs=[
                        bond_map_pair(mol, bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
                    ],
                    description="Aromatic bond between non-aromatic atom(s)",
                    evidence={},
                    repairable=True,
                    candidate_rule_ids=["C6_AROMATIC_RECOVERY", "C5_HETEROAROMATIC_RECOVERY"],
                )
            )

    for atom in mol.GetAtoms():
        if not atom.GetIsAromatic():
            continue
        has_aro_bond = any(b.GetIsAromatic() for b in atom.GetBonds())
        if not has_aro_bond:
            issues.append(
                StructureIssue(
                    issue_code="AROMATIC_ATOM_WITHOUT_AROMATIC_BOND",
                    severity="warning",
                    atom_map_ids=[atom.GetAtomMapNum()],
                    bond_atom_map_pairs=[],
                    description=f"Aromatic atom map {atom.GetAtomMapNum()} has no aromatic bonds",
                    evidence={},
                    repairable=True,
                    candidate_rule_ids=["C6_AROMATIC_RECOVERY", "C5_HETEROAROMATIC_RECOVERY"],
                )
            )

    # Inconsistent flags: mix of aromatic/non-aromatic in same small ring
    for ring in ring_info.AtomRings():
        if len(ring) not in (5, 6):
            continue
        flags = [mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring]
        if any(flags) and not all(flags):
            maps = [mol.GetAtomWithIdx(i).GetAtomMapNum() for i in ring]
            issues.append(
                StructureIssue(
                    issue_code="INCONSISTENT_AROMATIC_FLAGS",
                    severity="warning",
                    atom_map_ids=maps,
                    bond_atom_map_pairs=[],
                    description="Mixed aromatic flags within the same ring",
                    evidence={"ring_size": len(ring)},
                    repairable=True,
                    candidate_rule_ids=(
                        ["C6_AROMATIC_RECOVERY"]
                        if len(ring) == 6
                        else ["C5_HETEROAROMATIC_RECOVERY"]
                    ),
                )
            )

    return issues
