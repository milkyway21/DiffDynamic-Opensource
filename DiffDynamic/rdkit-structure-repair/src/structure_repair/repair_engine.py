"""Main repair engine: detect → candidates → validate → score → select."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from rdkit import Chem

from .atom_mapping import assign_atom_maps, clear_atom_maps_for_smiles
from .candidate_generator import build_rules, generate_candidates_for_issue
from .candidate_scorer import score_candidate, score_molecule
from .candidate_validator import candidate_passes_acceptance, validate_candidate
from .config import load_config
from .detector import detect_issues
from .models import (
    STATUS_AMBIGUOUS,
    STATUS_REJECTED,
    STATUS_REPAIRED,
    STATUS_UNCHANGED,
    AtomEdit,
    BondEdit,
    RepairCandidate,
    RepairResult,
)
from .mol_edit import molecule_state_hash
from .staged_sanitize import try_full_sanitize
from .standardize import apply_standardize


PathLike = Union[str, Path]


def _confidence_from_scores(before: float, after: float, margin: float) -> float:
    if before <= 0:
        return 1.0
    improvement = max(0.0, before - (after if after is not None else before))
    frac = min(1.0, improvement / max(before, 1.0))
    margin_boost = min(0.2, margin / 500.0)
    return round(min(0.99, 0.5 + 0.4 * frac + margin_boost), 3)


def _finalize_output_mol(mol: Chem.Mol, config: Dict[str, Any]) -> Chem.Mol:
    out = Chem.Mol(mol)
    std = apply_standardize(out, config)
    if std is not None:
        out = std
    repair_cfg = config.get("repair", {})
    if repair_cfg.get("clear_atom_maps_on_output", True):
        out = clear_atom_maps_for_smiles(out)
    return out


def _dedupe_candidates(cands: List[RepairCandidate]) -> List[RepairCandidate]:
    from .io import mol_to_canonical_smiles

    uniq: List[RepairCandidate] = []
    seen_keys = set()
    for cand in cands:
        smi = mol_to_canonical_smiles(cand.mol) or molecule_state_hash(cand.mol)
        if smi in seen_keys:
            continue
        seen_keys.add(smi)
        uniq.append(cand)
    return uniq


def repair_molecule(
    mol: Chem.Mol,
    config: Optional[Dict[str, Any]] = None,
    config_path: Optional[PathLike] = None,
    molecule_id: str = "mol",
) -> RepairResult:
    """Repair a reconstructed RDKit molecule using deterministic rules.

    Never mutates the input molecule.
    """
    if config is None:
        config = load_config(config_path)

    original = assign_atom_maps(Chem.Mol(mol))
    try:
        original.UpdatePropertyCache(strict=False)
        Chem.GetSymmSSSR(original)
    except Exception:  # noqa: BLE001
        pass

    issues_before = detect_issues(original, config)
    score_before = score_molecule(original, config)

    sanitized, _ = try_full_sanitize(original)
    repairable = [i for i in issues_before if i.repairable and i.severity != "info"]
    if sanitized is not None and not repairable:
        actionable = [
            i
            for i in issues_before
            if i.severity in {"error", "warning"} and i.repairable
        ]
        if not actionable:
            return RepairResult(
                molecule_id=molecule_id,
                status=STATUS_UNCHANGED,
                original_mol=original,
                repaired_mol=_finalize_output_mol(sanitized, config),
                issues_before=issues_before,
                issues_after=detect_issues(sanitized, config),
                selected_candidate_id=None,
                applied_rule_ids=[],
                bond_edits=[],
                atom_edits=[],
                score_before=score_before,
                score_after=score_molecule(sanitized, config),
                confidence=1.0,
                reject_reason=None,
            )

    rules = build_rules(config)
    repair_cfg = config.get("repair", {})
    max_iter = int(repair_cfg.get("max_iterations", 10))
    min_margin = float(repair_cfg.get("minimum_winner_margin", 50))
    reject_repeated = bool(repair_cfg.get("reject_repeated_state", True))
    prefer_best = bool(repair_cfg.get("prefer_best_on_ambiguous", False))
    accept_best_effort = bool(repair_cfg.get("accept_best_effort", False))

    current = Chem.Mol(original)
    visited = {molecule_state_hash(current)}
    applied_rules: List[str] = []
    all_bond_edits: List[BondEdit] = []
    all_atom_edits: List[AtomEdit] = []
    selected_ids: List[str] = []
    last_reject: Optional[str] = None

    for _iteration in range(max_iter):
        issues = detect_issues(current, config)
        actionable = [i for i in issues if i.repairable and i.severity in {"error", "warning"}]
        if not actionable:
            break

        current_score = score_molecule(current, config)
        accepted: List[RepairCandidate] = []
        improving: List[RepairCandidate] = []
        for issue in actionable:
            cands = generate_candidates_for_issue(current, issue, rules, config)
            for cand in cands:
                validate_candidate(original, cand, config)
                if cand.validation_errors:
                    continue
                cand.score = score_candidate(cand, config)
                if candidate_passes_acceptance(current_score, cand, config):
                    accepted.append(cand)
                elif (
                    accept_best_effort
                    and cand.score is not None
                    and cand.score < current_score
                    and cand.edit_cost <= float(repair_cfg.get("maximum_edit_cost", 40))
                ):
                    improving.append(cand)

        all_cands = accepted if accepted else (improving if accept_best_effort else [])
        if not all_cands:
            last_reject = "no_valid_candidates"
            break

        all_cands = _dedupe_candidates(all_cands)
        all_cands.sort(key=lambda c: (c.score if c.score is not None else 1e18, c.edit_cost))
        best = all_cands[0]
        second_score = all_cands[1].score if len(all_cands) > 1 else None
        if (
            not prefer_best
            and second_score is not None
            and best.score is not None
            and (second_score - best.score) < min_margin
        ):
            return RepairResult(
                molecule_id=molecule_id,
                status=STATUS_AMBIGUOUS,
                original_mol=original,
                repaired_mol=None,
                issues_before=issues_before,
                issues_after=detect_issues(current, config),
                selected_candidate_id=None,
                applied_rule_ids=applied_rules,
                bond_edits=all_bond_edits,
                atom_edits=all_atom_edits,
                score_before=score_before,
                score_after=None,
                confidence=0.0,
                reject_reason="winner_margin_too_small",
            )

        new_hash = molecule_state_hash(best.mol)
        if reject_repeated and new_hash in visited:
            if prefer_best:
                break
            return RepairResult(
                molecule_id=molecule_id,
                status=STATUS_AMBIGUOUS,
                original_mol=original,
                repaired_mol=None,
                issues_before=issues_before,
                issues_after=detect_issues(current, config),
                selected_candidate_id=None,
                applied_rule_ids=applied_rules,
                bond_edits=all_bond_edits,
                atom_edits=all_atom_edits,
                score_before=score_before,
                score_after=None,
                confidence=0.0,
                reject_reason="repeated_state",
            )
        visited.add(new_hash)
        current = best.mol
        applied_rules.append(best.rule_id)
        selected_ids.append(best.candidate_id)
        all_bond_edits.extend(best.bond_edits)
        all_atom_edits.extend(best.atom_edits)

    final_sanitized, err = try_full_sanitize(current)
    if final_sanitized is not None:
        final_sanitized = _finalize_output_mol(final_sanitized, config)
        # re-sanitize after standardize/map clear
        final_sanitized, err2 = try_full_sanitize(final_sanitized)
        if final_sanitized is None:
            err = err2 or err

    issues_after = detect_issues(final_sanitized or current, config)
    score_after = score_molecule(final_sanitized or current, config) if final_sanitized else None

    if not applied_rules:
        if final_sanitized is not None and score_after is not None and score_after <= score_before:
            remaining = [i for i in issues_after if i.repairable and i.severity in {"error", "warning"}]
            if not remaining:
                return RepairResult(
                    molecule_id=molecule_id,
                    status=STATUS_UNCHANGED,
                    original_mol=original,
                    repaired_mol=final_sanitized,
                    issues_before=issues_before,
                    issues_after=issues_after,
                    selected_candidate_id=None,
                    applied_rule_ids=[],
                    bond_edits=[],
                    atom_edits=[],
                    score_before=score_before,
                    score_after=score_after,
                    confidence=1.0,
                    reject_reason=None,
                )
        return RepairResult(
            molecule_id=molecule_id,
            status=STATUS_REJECTED,
            original_mol=original,
            repaired_mol=None,
            issues_before=issues_before,
            issues_after=issues_after,
            selected_candidate_id=None,
            applied_rule_ids=[],
            bond_edits=[],
            atom_edits=[],
            score_before=score_before,
            score_after=score_after,
            confidence=0.0,
            reject_reason=last_reject or err or "repair_failed",
        )

    if final_sanitized is None:
        return RepairResult(
            molecule_id=molecule_id,
            status=STATUS_REJECTED,
            original_mol=original,
            repaired_mol=None,
            issues_before=issues_before,
            issues_after=issues_after,
            selected_candidate_id=selected_ids[-1] if selected_ids else None,
            applied_rule_ids=applied_rules,
            bond_edits=all_bond_edits,
            atom_edits=all_atom_edits,
            score_before=score_before,
            score_after=None,
            confidence=0.0,
            reject_reason=err or "final_sanitize_failed",
        )

    margin = (score_before - score_after) if score_after is not None else 0.0
    return RepairResult(
        molecule_id=molecule_id,
        status=STATUS_REPAIRED,
        original_mol=original,
        repaired_mol=final_sanitized,
        issues_before=issues_before,
        issues_after=issues_after,
        selected_candidate_id=selected_ids[-1] if selected_ids else None,
        applied_rule_ids=applied_rules,
        bond_edits=all_bond_edits,
        atom_edits=all_atom_edits,
        score_before=score_before,
        score_after=score_after,
        confidence=_confidence_from_scores(score_before, score_after or score_before, margin),
        reject_reason=None,
    )
