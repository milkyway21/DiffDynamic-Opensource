"""CLI entry point: structure-repair."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from .audit import append_audit_jsonl, explain_audit_record, load_audit_jsonl, write_summary_csv
from .config import default_config_path, load_config
from .detector import detect_issues
from .io import iter_input_molecules, mol_to_canonical_smiles, write_sdf
from .models import (
    STATUS_AMBIGUOUS,
    STATUS_REJECTED,
    STATUS_REPAIRED,
    STATUS_UNCHANGED,
    RepairResult,
)
from .repair_engine import repair_molecule
from .atom_mapping import assign_atom_maps


def _cmd_validate(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    molecules = iter_input_molecules(
        args.input,
        smiles_column=args.smiles_column,
        id_column=args.id_column,
        sanitize=False,
    )
    results: List[RepairResult] = []
    for mid, mol in molecules:
        mapped = assign_atom_maps(mol)
        issues = detect_issues(mapped, config)
        from .candidate_scorer import score_molecule

        score = score_molecule(mapped, config)
        needs_repair = any(i.repairable and i.severity in {"error", "warning"} for i in issues)
        result = RepairResult(
            molecule_id=mid,
            status=STATUS_UNCHANGED if not needs_repair else STATUS_REJECTED,
            original_mol=mapped,
            repaired_mol=None,
            issues_before=issues,
            issues_after=issues,
            selected_candidate_id=None,
            applied_rule_ids=[],
            bond_edits=[],
            atom_edits=[],
            score_before=score,
            score_after=None,
            confidence=0.0,
            reject_reason=None if not needs_repair else "validate_only_issues_found",
        )
        results.append(result)
        append_audit_jsonl(out_dir / "audit.jsonl", result)

    write_summary_csv(out_dir / "summary.csv", results)
    ok = sum(1 for r in results if r.status == STATUS_UNCHANGED)
    print(f"Validated {len(results)} molecules; clean={ok}, flagged={len(results) - ok}")
    print(f"Wrote {out_dir / 'summary.csv'}")
    return 0


def _cmd_repair(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    molecules = iter_input_molecules(
        args.input,
        smiles_column=args.smiles_column,
        id_column=args.id_column,
        sanitize=False,
    )

    buckets = {
        STATUS_REPAIRED: [],
        STATUS_UNCHANGED: [],
        STATUS_AMBIGUOUS: [],
        STATUS_REJECTED: [],
    }
    results: List[RepairResult] = []

    # num_workers reserved for M4; run serially for now
    for mid, mol in molecules:
        result = repair_molecule(mol, config=config, molecule_id=mid)
        results.append(result)
        append_audit_jsonl(out_dir / "audit.jsonl", result)
        status = result.status if result.status in buckets else STATUS_REJECTED
        if result.status == STATUS_REPAIRED and result.repaired_mol is not None:
            buckets[STATUS_REPAIRED].append((mid, result.repaired_mol))
        elif result.status == STATUS_UNCHANGED:
            buckets[STATUS_UNCHANGED].append((mid, result.repaired_mol or result.original_mol))
        elif result.status == STATUS_AMBIGUOUS:
            buckets[STATUS_AMBIGUOUS].append((mid, result.original_mol))
        else:
            buckets[STATUS_REJECTED].append((mid, result.original_mol))

    write_sdf(out_dir / "repaired.sdf", buckets[STATUS_REPAIRED])
    write_sdf(out_dir / "unchanged.sdf", buckets[STATUS_UNCHANGED])
    write_sdf(out_dir / "ambiguous.sdf", buckets[STATUS_AMBIGUOUS])
    write_sdf(out_dir / "rejected.sdf", buckets[STATUS_REJECTED])
    write_summary_csv(out_dir / "summary.csv", results)

    counts = {k: len(v) for k, v in buckets.items()}
    print(f"Repaired batch: {counts}")
    print(f"Output: {out_dir}")
    return 0


def _cmd_explain(args: argparse.Namespace) -> int:
    records = load_audit_jsonl(args.input)
    found = False
    for rec in records:
        if args.molecule_id and rec.get("molecule_id") != args.molecule_id:
            continue
        print(explain_audit_record(rec))
        print("-" * 40)
        found = True
        if args.molecule_id:
            break
    if not found:
        print(f"No audit record for molecule_id={args.molecule_id}", file=sys.stderr)
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="structure-repair", description="Deterministic RDKit structure repair")
    sub = parser.add_subparsers(dest="command", required=True)

    p_val = sub.add_parser("validate", help="Detect issues without repairing")
    p_val.add_argument("input", help="SDF/MOL/SMILES/CSV input")
    p_val.add_argument("--config", default=str(default_config_path()))
    p_val.add_argument("--output", required=True)
    p_val.add_argument("--smiles-column", default="reconstructed_smiles")
    p_val.add_argument("--id-column", default="molecule_id")
    p_val.set_defaults(func=_cmd_validate)

    p_rep = sub.add_parser("repair", help="Detect and repair molecules")
    p_rep.add_argument("input", help="SDF/MOL/SMILES/CSV input")
    p_rep.add_argument("--config", default=str(default_config_path()))
    p_rep.add_argument("--output", required=True)
    p_rep.add_argument("--smiles-column", default="reconstructed_smiles")
    p_rep.add_argument("--id-column", default="molecule_id")
    p_rep.add_argument("--num-workers", type=int, default=1, help="Reserved (serial in M1-M3)")
    p_rep.set_defaults(func=_cmd_repair)

    p_exp = sub.add_parser("explain", help="Explain an audit.jsonl record")
    p_exp.add_argument("input", help="Path to audit.jsonl")
    p_exp.add_argument("--molecule-id", required=True)
    p_exp.set_defaults(func=_cmd_explain)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
