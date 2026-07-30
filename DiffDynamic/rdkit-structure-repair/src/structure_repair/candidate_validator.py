"""Validate repair candidates against chemical and identity gates."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

from rdkit import Chem

from .detector import detect_issues
from .identity_check import identity_ok
from .io import mol_to_canonical_smiles
from .models import BondEdit, RepairCandidate
from .staged_sanitize import try_full_sanitize
from .valence import detect_valence_issues


def _allowed_bond_edits(bond_edits: List[BondEdit]) -> Tuple[Set[Tuple[int, int]], Set[Tuple[int, int]]]:
    deleted = set()
    added = set()
    for e in bond_edits:
        key = (e.atom_map_1, e.atom_map_2) if e.atom_map_1 <= e.atom_map_2 else (e.atom_map_2, e.atom_map_1)
        if e.action == "delete":
            deleted.add(key)
        elif e.action == "add":
            added.add(key)
    return deleted, added


def validate_candidate(
    original: Chem.Mol,
    candidate: RepairCandidate,
    config: Dict[str, Any],
) -> RepairCandidate:
    errors: List[str] = []
    sanitized, err = try_full_sanitize(candidate.mol)
    if sanitized is None:
        errors.append(f"sanitize_failure:{err}")
        candidate.validation_errors = errors
        return candidate

    candidate.mol = sanitized

    valence_issues = detect_valence_issues(candidate.mol)
    if valence_issues:
        errors.append(f"invalid_valence_count:{len(valence_issues)}")

    for atom in candidate.mol.GetAtoms():
        if atom.GetNumRadicalElectrons() > 0 and atom.GetAtomicNum() > 1:
            if not config.get("radicals", {}).get("allow", False):
                errors.append("unexpected_radical")
                break

    # Kekulization already part of sanitize; check aromatic consistency lightly
    smi = mol_to_canonical_smiles(candidate.mol)
    if smi is None:
        errors.append("canonical_smiles_failed")

    deleted, added = _allowed_bond_edits(candidate.bond_edits)
    ok, id_errors = identity_ok(
        original,
        candidate.mol,
        config,
        allowed_deleted_bonds=deleted,
        allowed_added_bonds=added,
        allow_charge_change=True,
    )
    if not ok:
        errors.extend(id_errors)

    candidate.validation_errors = errors
    return candidate


def candidate_passes_acceptance(
    original_penalty: float,
    candidate: RepairCandidate,
    config: Dict[str, Any],
) -> bool:
    if candidate.validation_errors:
        return False
    if candidate.score is None:
        return False
    repair_cfg = config.get("repair", {})
    if candidate.score >= original_penalty:
        return False
    improvement = original_penalty - candidate.score
    if improvement < float(repair_cfg.get("minimum_improvement", 100)):
        return False
    if candidate.edit_cost > float(repair_cfg.get("maximum_edit_cost", 12)):
        return False
    return True
