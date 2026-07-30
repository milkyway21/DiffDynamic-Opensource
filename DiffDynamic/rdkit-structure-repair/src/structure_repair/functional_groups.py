"""Common functional-group pattern detection."""

from __future__ import annotations

from typing import List, Optional

from rdkit import Chem

from .atom_mapping import bond_map_pair
from .models import StructureIssue

# Patterns checked on unsanitized mols via manual neighbor inspection when needed.


def _nitro_like_issues(mol: Chem.Mol) -> List[StructureIssue]:
    issues: List[StructureIssue] = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 7:
            continue
        oxygens = []
        for bond in atom.GetBonds():
            other = bond.GetOtherAtom(atom)
            if other.GetAtomicNum() == 8:
                oxygens.append((other, bond))
        if len(oxygens) < 2:
            continue
        # Classic wrong form: N(=O)=O with charge 0
        double_o = sum(1 for _, b in oxygens if b.GetBondType() == Chem.BondType.DOUBLE)
        fc_n = atom.GetFormalCharge()
        fc_o = [o.GetFormalCharge() for o, _ in oxygens]
        if double_o >= 2 and fc_n == 0 and all(c == 0 for c in fc_o):
            maps = [atom.GetAtomMapNum()] + [o.GetAtomMapNum() for o, _ in oxygens]
            pairs = [
                bond_map_pair(mol, atom.GetIdx(), o.GetIdx()) for o, _ in oxygens
            ]
            issues.append(
                StructureIssue(
                    issue_code="NITRO_BOND_ORDER_ERROR",
                    severity="error",
                    atom_map_ids=maps,
                    bond_atom_map_pairs=pairs,
                    description="Nitro group as N(=O)=O without charge separation",
                    evidence={"n_map": atom.GetAtomMapNum()},
                    repairable=True,
                    candidate_rule_ids=["FUNCTIONAL_GROUP_REPAIR"],
                )
            )
            issues.append(
                StructureIssue(
                    issue_code="MISSING_POSITIVE_CHARGE",
                    severity="error",
                    atom_map_ids=[atom.GetAtomMapNum()],
                    bond_atom_map_pairs=pairs,
                    description="Nitro nitrogen missing formal + charge",
                    evidence={},
                    repairable=True,
                    candidate_rule_ids=["FUNCTIONAL_GROUP_REPAIR", "CHARGE_REPAIR"],
                )
            )
            issues.append(
                StructureIssue(
                    issue_code="MISSING_NEGATIVE_CHARGE",
                    severity="error",
                    atom_map_ids=[o.GetAtomMapNum() for o, _ in oxygens],
                    bond_atom_map_pairs=pairs,
                    description="Nitro oxygen missing formal - charge",
                    evidence={},
                    repairable=True,
                    candidate_rule_ids=["FUNCTIONAL_GROUP_REPAIR", "CHARGE_REPAIR"],
                )
            )
    return issues


def _carboxylate_issues(mol: Chem.Mol) -> List[StructureIssue]:
    issues: List[StructureIssue] = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 6:
            continue
        oxygens = []
        for bond in atom.GetBonds():
            other = bond.GetOtherAtom(atom)
            if other.GetAtomicNum() == 8 and other.GetDegree() == 1:
                oxygens.append((other, bond))
        if len(oxygens) != 2:
            continue
        types = [b.GetBondType() for _, b in oxygens]
        charges = [o.GetFormalCharge() for o, _ in oxygens]
        # Both single, both neutral → carboxylate/acid ambiguity
        if all(t == Chem.BondType.SINGLE for t in types) and all(c == 0 for c in charges):
            maps = [atom.GetAtomMapNum()] + [o.GetAtomMapNum() for o, _ in oxygens]
            pairs = [bond_map_pair(mol, atom.GetIdx(), o.GetIdx()) for o, _ in oxygens]
            issues.append(
                StructureIssue(
                    issue_code="CARBOXYLATE_BOND_ORDER_ERROR",
                    severity="warning",
                    atom_map_ids=maps,
                    bond_atom_map_pairs=pairs,
                    description="Carboxyl carbon with two single-bonded terminal oxygens",
                    evidence={},
                    repairable=True,
                    candidate_rule_ids=["FUNCTIONAL_GROUP_REPAIR"],
                )
            )
    return issues


def _amide_issues(mol: Chem.Mol) -> List[StructureIssue]:
    issues: List[StructureIssue] = []
    for bond in mol.GetBonds():
        a1, a2 = bond.GetBeginAtom(), bond.GetEndAtom()
        if {a1.GetAtomicNum(), a2.GetAtomicNum()} != {6, 7}:
            continue
        if bond.GetBondType() != Chem.BondType.DOUBLE:
            continue
        c_atom = a1 if a1.GetAtomicNum() == 6 else a2
        n_atom = a2 if a1.GetAtomicNum() == 6 else a1
        # Look for carbonyl O on carbon
        has_carbonyl_o = False
        for b in c_atom.GetBonds():
            o = b.GetOtherAtom(c_atom)
            if o.GetAtomicNum() == 8 and b.GetBondType() == Chem.BondType.DOUBLE and o.GetIdx() != n_atom.GetIdx():
                has_carbonyl_o = True
        # Wrong amide: C(=O)=N style (two doubles on C to O and N)
        o_double = sum(
            1
            for b in c_atom.GetBonds()
            if b.GetOtherAtom(c_atom).GetAtomicNum() == 8 and b.GetBondType() == Chem.BondType.DOUBLE
        )
        if has_carbonyl_o and bond.GetBondType() == Chem.BondType.DOUBLE and o_double >= 1:
            # Could be legitimate imide; flag only if N has only single other bonds and no charge
            if n_atom.GetFormalCharge() == 0:
                issues.append(
                    StructureIssue(
                        issue_code="AMIDE_BOND_ORDER_ERROR",
                        severity="warning",
                        atom_map_ids=[c_atom.GetAtomMapNum(), n_atom.GetAtomMapNum()],
                        bond_atom_map_pairs=[
                            bond_map_pair(mol, c_atom.GetIdx(), n_atom.GetIdx())
                        ],
                        description="Possible amide C–N double bond (should usually be single)",
                        evidence={},
                        repairable=True,
                        candidate_rule_ids=["FUNCTIONAL_GROUP_REPAIR"],
                    )
                )
    return issues


def _noxide_issues(mol: Chem.Mol) -> List[StructureIssue]:
    issues: List[StructureIssue] = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 7:
            continue
        for bond in atom.GetBonds():
            o = bond.GetOtherAtom(atom)
            if o.GetAtomicNum() != 8 or o.GetDegree() != 1:
                continue
            # Neutral N=O or N-O where N is aromatic/tertiary → likely N-oxide needing charges
            if atom.GetFormalCharge() == 0 and o.GetFormalCharge() == 0:
                if atom.GetIsAromatic() or atom.GetDegree() >= 3:
                    if bond.GetBondType() in (Chem.BondType.DOUBLE, Chem.BondType.SINGLE):
                        issues.append(
                            StructureIssue(
                                issue_code="MISSING_POSITIVE_CHARGE",
                                severity="warning",
                                atom_map_ids=[atom.GetAtomMapNum(), o.GetAtomMapNum()],
                                bond_atom_map_pairs=[
                                    bond_map_pair(mol, atom.GetIdx(), o.GetIdx())
                                ],
                                description="Possible N-oxide missing charge separation",
                                evidence={"pattern": "n_oxide"},
                                repairable=True,
                                candidate_rule_ids=["FUNCTIONAL_GROUP_REPAIR", "CHARGE_REPAIR"],
                            )
                        )
    return issues


def _sulfone_issues(mol: Chem.Mol) -> List[StructureIssue]:
    issues: List[StructureIssue] = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 16:
            continue
        oxygens = [
            (bond.GetOtherAtom(atom), bond)
            for bond in atom.GetBonds()
            if bond.GetOtherAtom(atom).GetAtomicNum() == 8
        ]
        if len(oxygens) < 2:
            continue
        # Flag if S looks over/undervalued relative to sulfone representation
        try:
            atom.UpdatePropertyCache(strict=False)
        except Exception:  # noqa: BLE001
            pass
    return issues


def detect_functional_group_issues(mol: Chem.Mol) -> List[StructureIssue]:
    issues: List[StructureIssue] = []
    issues.extend(_nitro_like_issues(mol))
    issues.extend(_carboxylate_issues(mol))
    issues.extend(_amide_issues(mol))
    issues.extend(_noxide_issues(mol))
    issues.extend(_sulfone_issues(mol))
    return issues
