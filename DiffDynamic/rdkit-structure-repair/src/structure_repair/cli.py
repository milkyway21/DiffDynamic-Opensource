"""CLI entry point: structure-repair."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from .audit import (
    append_audit_jsonl,
    explain_audit_record,
    load_audit_jsonl,
    write_optimize_audit_csv,
    write_summary_csv,
)
from .config import (
    default_config_path,
    default_optimize_config_path,
    load_config,
    load_optimize_config,
)
from .detector import detect_issues
from .io import iter_input_molecules, write_sdf
from .models import (
    OPT_STATUS_OPTIMIZED,
    STATUS_AMBIGUOUS,
    STATUS_REJECTED,
    STATUS_REPAIRED,
    STATUS_UNCHANGED,
    OptimizeResult,
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


def _embed_single_conformer(mol):
    from rdkit import Chem
    from rdkit.Chem import AllChem

    try:
        sanitized = Chem.Mol(mol)
        Chem.SanitizeMol(sanitized)
        with_h = Chem.AddHs(sanitized)
        if AllChem.EmbedMolecule(with_h, randomSeed=0xF00D) != 0:
            return mol
        AllChem.MMFFOptimizeMolecule(with_h)
        return with_h
    except Exception:  # noqa: BLE001
        return mol


def _cmd_optimize(args: argparse.Namespace) -> int:
    from .optimize import optimize_molecule
    from .optimize.catalog import validate_catalog

    if args.validate_catalog:
        report = validate_catalog()
        print(f"Catalog entries OK: {len(report['ok'])}")
        for key in ("bad_smarts", "duplicate_id"):
            for item in report[key]:
                print(f"  {key}: {item}", file=sys.stderr)
        return 1 if report["bad_smarts"] or report["duplicate_id"] else 0

    repair_config = load_config(args.config)
    optimize_config = load_optimize_config(args.optimize_config)
    if args.tiers:
        optimize_config["tiers"] = [t.strip().upper() for t in args.tiers.split(",") if t.strip()]

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    molecules = iter_input_molecules(
        args.input,
        smiles_column=args.smiles_column,
        id_column=args.id_column,
        sanitize=False,
    )

    results: List[OptimizeResult] = []
    optimized_out = []
    preopt_out = []
    for mid, mol in molecules:
        if args.embed_if_missing and mol is not None and mol.GetNumConformers() == 0:
            mol = _embed_single_conformer(mol)
        result = optimize_molecule(
            mol,
            molecule_id=mid,
            config=optimize_config,
            protein_path=args.protein,
            repair_config=repair_config,
        )
        results.append(result)
        if result.optimized_mol is not None:
            optimized_out.append((mid, result.optimized_mol))
        if args.keep_preopt_sdf != "none" and result.changed:
            preopt_out.append((mid, result.original_mol))

    write_sdf(out_dir / "optimized.sdf", optimized_out)
    if preopt_out:
        write_sdf(out_dir / "preopt.sdf", preopt_out)
    write_optimize_audit_csv(out_dir / "optimize_audit.csv", results)

    changed = sum(1 for r in results if r.status == OPT_STATUS_OPTIMIZED)
    print(f"Optimized {changed}/{len(results)} molecules")
    if any(r.reject_reason == "no_conformer" for r in results):
        print(
            "Some inputs had no 3D conformer; rerun with --embed-if-missing to "
            "generate one (planarity evidence will then be synthetic).",
            file=sys.stderr,
        )
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
    parser = argparse.ArgumentParser(
        prog="structure-repair", description="Deterministic RDKit structure repair / medchem optimize"
    )
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

    p_opt = sub.add_parser(
        "optimize",
        help="Medchem optimization (changes molecular formula; off by default elsewhere)",
    )
    p_opt.add_argument("input", nargs="?", help="SDF/MOL/SMILES/CSV input")
    p_opt.add_argument("--config", default=str(default_config_path()))
    p_opt.add_argument("--optimize-config", default=str(default_optimize_config_path()))
    p_opt.add_argument("--output", default="optimize_out")
    p_opt.add_argument("--smiles-column", default="reconstructed_smiles")
    p_opt.add_argument("--id-column", default="molecule_id")
    p_opt.add_argument("--protein", default=None, help="Pocket PDB for clash rejection")
    p_opt.add_argument("--tiers", default=None, help="Comma-separated tier override, e.g. T0,T3")
    p_opt.add_argument(
        "--keep-preopt-sdf",
        choices=["none", "structure_only", "evaluated"],
        default="structure_only",
    )
    p_opt.add_argument(
        "--embed-if-missing",
        action="store_true",
        help="Generate a 3D conformer for inputs that lack one (e.g. SMILES)",
    )
    p_opt.add_argument(
        "--validate-catalog",
        action="store_true",
        help="Only check that every catalog transform compiles",
    )
    p_opt.set_defaults(func=_cmd_optimize)

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
