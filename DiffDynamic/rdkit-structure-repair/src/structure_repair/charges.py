"""Formal charge anomaly detection."""

from __future__ import annotations

from typing import List

from rdkit import Chem

from .models import StructureIssue


def detect_charge_issues(mol: Chem.Mol, config: dict = None) -> List[StructureIssue]:
    config = config or {}
    charge_cfg = config.get("charge", {})
    allowed = set(charge_cfg.get("allowed_total_charges", [-2, -1, 0, 1, 2]))
    issues: List[StructureIssue] = []

    try:
        mol.UpdatePropertyCache(strict=False)
    except Exception:  # noqa: BLE001
        pass

    total = sum(a.GetFormalCharge() for a in mol.GetAtoms())
    if total not in allowed:
        issues.append(
            StructureIssue(
                issue_code="INVALID_TOTAL_CHARGE",
                severity="warning",
                atom_map_ids=[a.GetAtomMapNum() for a in mol.GetAtoms() if a.GetFormalCharge() != 0],
                bond_atom_map_pairs=[],
                description=f"Total formal charge {total} outside allowed {sorted(allowed)}",
                evidence={"total_charge": total, "allowed": sorted(allowed)},
                repairable=False,
                candidate_rule_ids=[],
            )
        )

    # Neutral hypervalent N/O → missing positive charge
    for atom in mol.GetAtoms():
        z = atom.GetAtomicNum()
        fc = atom.GetFormalCharge()
        degree = atom.GetTotalDegree()
        if z == 7 and fc == 0 and degree == 4:
            issues.append(
                StructureIssue(
                    issue_code="MISSING_POSITIVE_CHARGE",
                    severity="error",
                    atom_map_ids=[atom.GetAtomMapNum()],
                    bond_atom_map_pairs=[],
                    description=f"Tetracoordinate neutral N map {atom.GetAtomMapNum()}",
                    evidence={"atomic_num": 7, "degree": degree},
                    repairable=True,
                    candidate_rule_ids=["CHARGE_REPAIR"],
                )
            )
            issues.append(
                StructureIssue(
                    issue_code="IMPLAUSIBLE_NEUTRAL_HYPERVALENT_ATOM",
                    severity="error",
                    atom_map_ids=[atom.GetAtomMapNum()],
                    bond_atom_map_pairs=[],
                    description=f"Neutral hypervalent N map {atom.GetAtomMapNum()}",
                    evidence={},
                    repairable=True,
                    candidate_rule_ids=["CHARGE_REPAIR"],
                )
            )

    # Permanent ions / zwitterions (informational, not repairable forcibly)
    pos = [a for a in mol.GetAtoms() if a.GetFormalCharge() > 0]
    neg = [a for a in mol.GetAtoms() if a.GetFormalCharge() < 0]
    if pos and not neg and total > 0:
        # quaternary ammonium etc.
        for a in pos:
            if a.GetAtomicNum() == 7 and a.GetTotalDegree() == 4:
                issues.append(
                    StructureIssue(
                        issue_code="PERMANENT_ION",
                        severity="info",
                        atom_map_ids=[a.GetAtomMapNum()],
                        bond_atom_map_pairs=[],
                        description="Permanent quaternary ammonium ion",
                        evidence={"total_charge": total},
                        repairable=False,
                        candidate_rule_ids=[],
                    )
                )
    if pos and neg:
        issues.append(
            StructureIssue(
                issue_code="ZWITTERION",
                severity="info",
                atom_map_ids=[a.GetAtomMapNum() for a in pos + neg],
                bond_atom_map_pairs=[],
                description="Zwitterionic charge separation detected",
                evidence={"total_charge": total},
                repairable=False,
                candidate_rule_ids=[],
            )
        )

    return issues


def detect_radical_issues(mol: Chem.Mol, allow: bool = False) -> List[StructureIssue]:
    if allow:
        return []
    issues: List[StructureIssue] = []
    radicals = [
        a for a in mol.GetAtoms() if a.GetNumRadicalElectrons() > 0 and a.GetAtomicNum() > 1
    ]
    if not radicals:
        return []
    maps = [a.GetAtomMapNum() for a in radicals]
    code = "UNEXPECTED_RADICAL" if len(radicals) == 1 else "MULTIPLE_UNEXPECTED_RADICALS"
    issues.append(
        StructureIssue(
            issue_code=code,
            severity="error",
            atom_map_ids=maps,
            bond_atom_map_pairs=[],
            description=f"Unexpected radical electrons on atoms {maps}",
            evidence={"counts": {a.GetAtomMapNum(): a.GetNumRadicalElectrons() for a in radicals}},
            repairable=True,
            candidate_rule_ids=["VALENCE_REPAIR", "CUMULENE_VALIDATION_AND_REPAIR"],
        )
    )
    return issues
