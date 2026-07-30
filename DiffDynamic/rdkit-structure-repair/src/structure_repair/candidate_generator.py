"""Candidate generation orchestration across repair rules."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

from rdkit import Chem

from .candidate_scorer import compute_edit_cost
from .models import RepairCandidate, StructureIssue
from .rules import (
    C5HeteroaromaticRecoveryRule,
    C6AromaticRecoveryRule,
    ChargeRepairRule,
    ConnectivityRepairRule,
    CumuleneValidationAndRepairRule,
    FunctionalGroupRepairRule,
    RepairRule,
    ValenceRepairRule,
)


def build_rules(config: Dict[str, Any]) -> List[RepairRule]:
    conn = config.get("connectivity_repair", {})
    arom = config.get("aromaticity", {})
    rules: List[RepairRule] = [
        ConnectivityRepairRule(
            allow_bond_deletion=bool(conn.get("allow_bond_deletion", True)),
            max_deleted=int(conn.get("maximum_deleted_bonds", 1)),
        ),
        ValenceRepairRule(allow_bond_deletion=bool(conn.get("allow_bond_deletion", True))),
        ChargeRepairRule(),
        FunctionalGroupRepairRule(),
    ]
    if arom.get("repair_c6_carbon_rings", True):
        rules.append(C6AromaticRecoveryRule())
    if arom.get("repair_c5_heteroaromatic_rings", True):
        rules.append(C5HeteroaromaticRecoveryRule())
    rules.append(CumuleneValidationAndRepairRule())
    rules.sort(key=lambda r: r.priority)
    return rules


def generate_candidates_for_issue(
    mol: Chem.Mol,
    issue: StructureIssue,
    rules: Sequence[RepairRule],
    config: Dict[str, Any],
) -> List[RepairCandidate]:
    rule_by_id = {r.rule_id: r for r in rules}
    target_ids = issue.candidate_rule_ids or [r.rule_id for r in rules]
    out: List[RepairCandidate] = []
    seen_ids = set()
    for rid in target_ids:
        rule = rule_by_id.get(rid)
        if rule is None:
            continue
        for cand in rule.generate_candidates(mol, issue):
            if cand.candidate_id in seen_ids:
                continue
            seen_ids.add(cand.candidate_id)
            if cand.edit_cost == 0.0 and (cand.bond_edits or cand.atom_edits):
                cand.edit_cost = compute_edit_cost(cand.bond_edits, cand.atom_edits, config)
            out.append(cand)
    # Also try all rules if specific ones produced nothing
    if not out:
        for rule in rules:
            for cand in rule.generate_candidates(mol, issue):
                if cand.candidate_id in seen_ids:
                    continue
                seen_ids.add(cand.candidate_id)
                if cand.edit_cost == 0.0 and (cand.bond_edits or cand.atom_edits):
                    cand.edit_cost = compute_edit_cost(cand.bond_edits, cand.atom_edits, config)
                out.append(cand)
    return out
