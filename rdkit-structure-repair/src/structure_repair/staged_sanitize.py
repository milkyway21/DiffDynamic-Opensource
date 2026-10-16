"""Staged RDKit sanitization with typed failure reporting."""

from __future__ import annotations

from typing import List, Optional, Tuple

from rdkit import Chem

from .models import SanitizationReport, SanitizationStepResult

# Ordered sanitize stages (subset of Chem.SanitizeFlags)
_STAGES: List[Tuple[str, int]] = [
    ("SANITIZE_CLEANUP", Chem.SanitizeFlags.SANITIZE_CLEANUP),
    ("SANITIZE_PROPERTIES", Chem.SanitizeFlags.SANITIZE_PROPERTIES),
    ("SANITIZE_SYMMRINGS", Chem.SanitizeFlags.SANITIZE_SYMMRINGS),
    ("SANITIZE_KEKULIZE", Chem.SanitizeFlags.SANITIZE_KEKULIZE),
    ("SANITIZE_FINDRADICALS", Chem.SanitizeFlags.SANITIZE_FINDRADICALS),
    ("SANITIZE_SETAROMATICITY", Chem.SanitizeFlags.SANITIZE_SETAROMATICITY),
    ("SANITIZE_SETCONJUGATION", Chem.SanitizeFlags.SANITIZE_SETCONJUGATION),
    ("SANITIZE_SETHYBRIDIZATION", Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION),
    ("SANITIZE_CLEANUPCHIRALITY", Chem.SanitizeFlags.SANITIZE_CLEANUPCHIRALITY),
    ("SANITIZE_ADJUSTHS", Chem.SanitizeFlags.SANITIZE_ADJUSTHS),
]


def _classify_error(step_name: str, exc: BaseException) -> str:
    msg = str(exc).lower()
    if "valence" in msg or "explicit valence" in msg:
        return "INVALID_VALENCE"
    if "kekul" in msg or step_name == "SANITIZE_KEKULIZE":
        return "KEKULIZATION_FAILED"
    if "aromatic" in msg or step_name == "SANITIZE_SETAROMATICITY":
        return "AROMATICITY_FAILED"
    if "radical" in msg or step_name == "SANITIZE_FINDRADICALS":
        return "RADICAL_DETECTED"
    if "hydrogen" in msg or "h count" in msg or step_name == "SANITIZE_ADJUSTHS":
        return "HYDROGEN_ADJUSTMENT_FAILED"
    return "SANITIZE_ERROR"


def staged_sanitize(mol: Chem.Mol, copy: bool = True) -> SanitizationReport:
    """Run sanitization stages individually and record failures.

    Does not raise; returns a SanitizationReport. When ``copy`` is True the
    input molecule is not mutated (each stage runs on a working copy that is
    discarded on failure of later stages — overall success means a full
    SanitizeMol on a fresh copy succeeded).
    """
    working = Chem.Mol(mol) if copy else mol
    steps: List[SanitizationStepResult] = []
    failure_types: List[str] = []

    try:
        working.UpdatePropertyCache(strict=False)
    except Exception as exc:  # noqa: BLE001
        err = _classify_error("UPDATE_CACHE", exc)
        failure_types.append(err)
        return SanitizationReport(
            overall_success=False,
            steps=[
                SanitizationStepResult(
                    step_name="UPDATE_CACHE",
                    success=False,
                    error_type=err,
                    error_message=str(exc),
                )
            ],
            failure_types=failure_types,
        )

    for step_name, flag in _STAGES:
        try:
            Chem.SanitizeMol(working, sanitizeOps=flag)
            steps.append(SanitizationStepResult(step_name=step_name, success=True))
        except Exception as exc:  # noqa: BLE001
            err = _classify_error(step_name, exc)
            failure_types.append(err)
            steps.append(
                SanitizationStepResult(
                    step_name=step_name,
                    success=False,
                    error_type=err,
                    error_message=str(exc),
                )
            )
            # Continue remaining stages on a restored copy for diagnostics
            working = Chem.Mol(mol) if copy else mol
            try:
                working.UpdatePropertyCache(strict=False)
            except Exception:  # noqa: BLE001
                break

    # Final full sanitize check on a fresh copy
    overall = False
    try:
        final = Chem.Mol(mol)
        final.UpdatePropertyCache(strict=False)
        Chem.SanitizeMol(final)
        overall = True
        # Apply successful sanitize back if not copying
        if not copy:
            Chem.SanitizeMol(mol)
    except Exception as exc:  # noqa: BLE001
        err = _classify_error("SANITIZE_FULL", exc)
        if err not in failure_types:
            failure_types.append(err)
        steps.append(
            SanitizationStepResult(
                step_name="SANITIZE_FULL",
                success=False,
                error_type=err,
                error_message=str(exc),
            )
        )
        overall = False

    if overall and not any(not s.success for s in steps if s.step_name != "SANITIZE_FULL"):
        failure_types = []

    return SanitizationReport(
        overall_success=overall,
        steps=steps,
        failure_types=failure_types,
    )


def try_full_sanitize(mol: Chem.Mol) -> Tuple[Optional[Chem.Mol], Optional[str]]:
    """Return (sanitized_copy, None) or (None, error_type)."""
    try:
        out = Chem.Mol(mol)
        out.UpdatePropertyCache(strict=False)
        Chem.SanitizeMol(out)
        Chem.AssignStereochemistry(out, cleanIt=True, force=True)
        return out, None
    except Exception as exc:  # noqa: BLE001
        return None, _classify_error("SANITIZE_FULL", exc)
