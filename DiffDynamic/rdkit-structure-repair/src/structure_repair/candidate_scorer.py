"""Deterministic candidate scoring (lower is better)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from rdkit import Chem

from .detector import detect_issues
from .io import mol_to_canonical_smiles
from .models import AtomEdit, BondEdit, RepairCandidate, StructureIssue
from .staged_sanitize import staged_sanitize, try_full_sanitize


_ERROR_CODES_VALENCE = {
    "CARBON_OVERVALENT",
    "CARBON_UNDERVALENT",
    "NITROGEN_OVERVALENT",
    "OXYGEN_OVERVALENT",
    "HALOGEN_OVERVALENT",
    "BORON_VALENCE_ERROR",
    "PHOSPHORUS_VALENCE_ERROR",
    "SULFUR_VALENCE_ERROR",
    "UNEXPLAINED_VALENCE_STATE",
}
_ERROR_CODES_AROMATIC = {
    "KEKULIZATION_FAILED",
    "AROMATIC_BOND_OUTSIDE_RING",
    "AROMATIC_ATOM_WITHOUT_AROMATIC_BOND",
    "AROMATIC_BOND_WITH_NONAROMATIC_ATOM",
    "INCONSISTENT_AROMATIC_FLAGS",
}
_ERROR_CODES_BOND = {
    "IMPOSSIBLE_BOND_ORDER",
    "CONSECUTIVE_DOUBLE_BONDS_IN_RING",
    "IMPLAUSIBLE_CUMULENE",
    "ALL_SINGLE_CONJUGATED_RING",
    "ALL_DOUBLE_RING",
    "TRIPLE_BOND_IN_SMALL_RING",
    "AMIDE_BOND_ORDER_ERROR",
    "CARBOXYLATE_BOND_ORDER_ERROR",
    "NITRO_BOND_ORDER_ERROR",
}
_ERROR_CODES_CHARGE = {
    "MISSING_POSITIVE_CHARGE",
    "MISSING_NEGATIVE_CHARGE",
    "IMPLAUSIBLE_NEUTRAL_HYPERVALENT_ATOM",
    "INVALID_TOTAL_CHARGE",
}
_ERROR_CODES_RING = {
    "C6_RING_AROMATICITY_LOST",
    "C5_HETEROAROMATICITY_LOST",
    "RING_BOND_ORDER_CONFLICT",
    "MULTIPLE_DOUBLE_BONDS_SHARING_ATOM_IN_RING",
    "IMPOSSIBLE_RING_UNSATURATION",
    "FUSED_RING_AROMATICITY_CONFLICT",
}
_RADICAL_CODES = {"UNEXPECTED_RADICAL", "MULTIPLE_UNEXPECTED_RADICALS", "RADICAL_CREATED_BY_WRONG_BOND_ORDER"}


def compute_edit_cost(
    bond_edits: List[BondEdit],
    atom_edits: List[AtomEdit],
    config: Dict[str, Any],
) -> float:
    costs = config.get("scoring", {}).get("edit_costs", {})
    total = 0.0
    for be in bond_edits:
        if be.action == "delete":
            total += float(costs.get("bond_deletion", 8))
        elif be.action == "add":
            total += float(costs.get("bond_addition", 10))
        else:
            # set bond type / aromatic
            if be.old_bond_type != be.new_bond_type:
                if {be.old_bond_type, be.new_bond_type} <= {"AROMATIC", "SINGLE", "DOUBLE", "TRIPLE", None}:
                    if be.new_bond_type == "AROMATIC" or be.old_bond_type == "AROMATIC":
                        total += float(costs.get("aromatic_flag_change", 0.5))
                    total += float(costs.get("bond_order_change", 1))
                else:
                    total += float(costs.get("bond_order_change", 1))
    for ae in atom_edits:
        if ae.property_name == "formal_charge":
            total += float(costs.get("formal_charge_change", 1))
        elif ae.property_name in ("num_explicit_hs", "no_implicit"):
            total += float(costs.get("implicit_h_change", 1))
        elif ae.property_name == "is_aromatic":
            total += float(costs.get("aromatic_flag_change", 0.5))
        elif ae.property_name == "atomic_num":
            total += float(costs.get("element_change", 10000))
    return total


def score_issues(issues: List[StructureIssue], config: Dict[str, Any], mol: Optional[Chem.Mol] = None) -> float:
    w = config.get("scoring", {})
    codes = [i.issue_code for i in issues]

    parse_failure = 0
    sanitize_failure = 0
    if mol is not None:
        report = staged_sanitize(mol)
        sanitize_failure = 0 if report.overall_success else 1
        smi = mol_to_canonical_smiles(mol)
        if smi is None:
            parse_failure = 1

    invalid_valence = sum(1 for c in codes if c in _ERROR_CODES_VALENCE)
    kekule_fail = sum(1 for c in codes if c == "KEKULIZATION_FAILED")
    radicals = sum(1 for c in codes if c in _RADICAL_CODES)
    aromatic = sum(1 for c in codes if c in _ERROR_CODES_AROMATIC)
    bond_bad = sum(1 for c in codes if c in _ERROR_CODES_BOND)
    charge_bad = sum(1 for c in codes if c in _ERROR_CODES_CHARGE)
    ring_bad = sum(1 for c in codes if c in _ERROR_CODES_RING)

    frag_pen = 0
    if mol is not None:
        frags = Chem.GetMolFrags(mol, asMols=False)
        if len(frags) > 1:
            frag_pen = len(frags) - 1

    penalty = (
        float(w.get("parse_failure", 10000)) * parse_failure
        + float(w.get("sanitize_failure", 8000)) * sanitize_failure
        + float(w.get("invalid_valence_count", 3000)) * invalid_valence
        + float(w.get("kekulization_failure_count", 2000)) * kekule_fail
        + float(w.get("unexpected_radical_count", 1500)) * radicals
        + float(w.get("aromaticity_conflict_count", 1000)) * aromatic
        + float(w.get("impossible_bond_order_count", 800)) * bond_bad
        + float(w.get("implausible_charge_count", 500)) * charge_bad
        + float(w.get("ring_conflict_count", 300)) * ring_bad
        + float(w.get("disconnected_fragment_penalty", 100)) * frag_pen
    )
    return penalty


def score_candidate(
    candidate: RepairCandidate,
    config: Dict[str, Any],
    issues: Optional[List[StructureIssue]] = None,
) -> float:
    if issues is None:
        issues = detect_issues(candidate.mol, config)
    base = score_issues(issues, config, mol=candidate.mol)
    return base + float(candidate.edit_cost)


def score_molecule(mol: Chem.Mol, config: Dict[str, Any]) -> float:
    issues = detect_issues(mol, config)
    return score_issues(issues, config, mol=mol)
