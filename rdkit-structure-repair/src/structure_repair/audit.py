"""Audit logging and summary CSV."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

from .io import mol_to_canonical_smiles
from .models import BondEdit, OptimizeResult, RepairResult, StructureIssue

PathLike = Union[str, Path]

OPTIMIZE_AUDIT_FIELDS = [
    "molecule_id",
    "status",
    "smiles_before",
    "smiles_after",
    "transform_chain",
    "transform_tiers",
    "n_transforms",
    "total_reward",
    "reward_qed",
    "reward_sa",
    "reward_rotb",
    "reward_alerts",
    "reward_tpsa",
    "reward_le",
    "reward_cost",
    "reward_bonus",
    "qed_before",
    "qed_after",
    "sa_before",
    "sa_after",
    "rotb_before",
    "rotb_after",
    "alerts_before",
    "alerts_after",
    "heavy_before",
    "heavy_after",
    "top_reject_reasons",
    "reject_reason",
]


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


def _heavy_atom_smiles(mol) -> Optional[str]:
    """Canonical SMILES without the explicit hydrogens the 3D pipeline carries."""
    if mol is None:
        return None
    try:
        from rdkit import Chem

        return mol_to_canonical_smiles(Chem.RemoveHs(Chem.Mol(mol)))
    except Exception:  # noqa: BLE001
        return mol_to_canonical_smiles(mol)


def optimize_result_to_audit_dict(result: OptimizeResult) -> Dict[str, Any]:
    before = result.properties_before or {}
    after = result.properties_after or {}
    breakdown = result.reward_breakdown or {}
    top_rejects = sorted(
        (result.rejected_counts or {}).items(), key=lambda kv: -kv[1]
    )[:5]
    return {
        "molecule_id": result.molecule_id,
        "status": result.status,
        "smiles_before": _heavy_atom_smiles(result.original_mol),
        "smiles_after": _heavy_atom_smiles(result.optimized_mol),
        "transform_chain": result.transform_chain,
        "transform_tiers": ">".join(a.tier for a in result.applied),
        "n_transforms": len(result.applied),
        "total_reward": result.total_reward,
        "reward_qed": breakdown.get("qed", 0.0),
        "reward_sa": breakdown.get("sa", 0.0),
        "reward_rotb": breakdown.get("rotb", 0.0),
        "reward_alerts": breakdown.get("alerts", 0.0),
        "reward_tpsa": breakdown.get("tpsa", 0.0),
        "reward_le": breakdown.get("le", 0.0),
        "reward_cost": breakdown.get("cost", 0.0),
        "reward_bonus": breakdown.get("bonus", 0.0),
        "qed_before": round(before.get("qed", 0.0), 4),
        "qed_after": round(after.get("qed", 0.0), 4),
        "sa_before": round(before.get("sa", 0.0), 3),
        "sa_after": round(after.get("sa", 0.0), 3),
        "rotb_before": int(before.get("rotb", 0)),
        "rotb_after": int(after.get("rotb", 0)),
        "alerts_before": int(before.get("alerts", 0)),
        "alerts_after": int(after.get("alerts", 0)),
        "heavy_before": int(before.get("heavy_atoms", 0)),
        "heavy_after": int(after.get("heavy_atoms", 0)),
        "top_reject_reasons": ";".join(f"{k}:{v}" for k, v in top_rejects),
        "reject_reason": result.reject_reason or "",
    }


def write_optimize_audit_csv(path: PathLike, results: Iterable[OptimizeResult]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=OPTIMIZE_AUDIT_FIELDS)
        writer.writeheader()
        for result in results:
            writer.writerow(optimize_result_to_audit_dict(result))


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
