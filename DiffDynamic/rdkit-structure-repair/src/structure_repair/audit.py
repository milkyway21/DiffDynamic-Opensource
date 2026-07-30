"""Audit logging and summary CSV."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

from .io import mol_to_canonical_smiles
from .models import BondEdit, RepairResult, StructureIssue

PathLike = Union[str, Path]


def result_to_audit_dict(result: RepairResult) -> Dict[str, Any]:
    return {
        "molecule_id": result.molecule_id,
        "status": result.status,
        "original_smiles": mol_to_canonical_smiles(result.original_mol),
        "repaired_smiles": (
            mol_to_canonical_smiles(result.repaired_mol) if result.repaired_mol is not None else None
        ),
        "issues_before": [i.issue_code for i in result.issues_before],
        "issues_after": [i.issue_code for i in result.issues_after],
        "rules_applied": list(result.applied_rule_ids),
        "bond_edits": [
            {
                "atom_map_1": e.atom_map_1,
                "atom_map_2": e.atom_map_2,
                "old_bond_type": e.old_bond_type,
                "new_bond_type": e.new_bond_type,
                "action": e.action,
            }
            for e in result.bond_edits
        ],
        "atom_edits": [
            {
                "atom_map_id": e.atom_map_id,
                "property_name": e.property_name,
                "old_value": e.old_value,
                "new_value": e.new_value,
            }
            for e in result.atom_edits
        ],
        "charge_edits": [
            {
                "atom_map_id": e.atom_map_id,
                "old_value": e.old_value,
                "new_value": e.new_value,
            }
            for e in result.atom_edits
            if e.property_name == "formal_charge"
        ],
        "score_before": result.score_before,
        "score_after": result.score_after,
        "confidence": result.confidence,
        "reject_reason": result.reject_reason,
        "selected_candidate_id": result.selected_candidate_id,
    }


def append_audit_jsonl(path: PathLike, result: RepairResult) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(result_to_audit_dict(result), ensure_ascii=False) + "\n")


def write_summary_csv(path: PathLike, results: Iterable[RepairResult]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "molecule_id",
        "status",
        "original_smiles",
        "repaired_smiles",
        "issues_before",
        "issues_after",
        "rules_applied",
        "bond_edits",
        "charge_edits",
        "score_before",
        "score_after",
        "confidence",
        "reject_reason",
    ]
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            d = result_to_audit_dict(result)
            writer.writerow(
                {
                    "molecule_id": d["molecule_id"],
                    "status": d["status"],
                    "original_smiles": d["original_smiles"],
                    "repaired_smiles": d["repaired_smiles"],
                    "issues_before": ";".join(d["issues_before"]),
                    "issues_after": ";".join(d["issues_after"]),
                    "rules_applied": ";".join(d["rules_applied"]),
                    "bond_edits": len(d["bond_edits"]),
                    "charge_edits": len(d["charge_edits"]),
                    "score_before": d["score_before"],
                    "score_after": d["score_after"],
                    "confidence": d["confidence"],
                    "reject_reason": d["reject_reason"] or "",
                }
            )


def load_audit_jsonl(path: PathLike) -> List[Dict[str, Any]]:
    path = Path(path)
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def explain_audit_record(record: Dict[str, Any]) -> str:
    lines = [
        f"molecule_id: {record.get('molecule_id')}",
        f"status: {record.get('status')}",
        f"original_smiles: {record.get('original_smiles')}",
        f"repaired_smiles: {record.get('repaired_smiles')}",
        f"issues_before: {record.get('issues_before')}",
        f"issues_after: {record.get('issues_after')}",
        f"rules_applied: {record.get('rules_applied')}",
        f"score_before: {record.get('score_before')} → score_after: {record.get('score_after')}",
        f"confidence: {record.get('confidence')}",
        f"reject_reason: {record.get('reject_reason')}",
        "bond_edits:",
    ]
    for edit in record.get("bond_edits") or []:
        lines.append(
            f"  {edit.get('atom_map_1')}-{edit.get('atom_map_2')}: "
            f"{edit.get('old_bond_type')} → {edit.get('new_bond_type')} ({edit.get('action')})"
        )
    return "\n".join(lines)
