"""Aggregate structure issue detection."""

from __future__ import annotations

from typing import Any, Dict, List

from rdkit import Chem

from .aromaticity import detect_aromaticity_issues
from .atom_mapping import assign_atom_maps
from .charges import detect_charge_issues, detect_radical_issues
from .functional_groups import detect_functional_group_issues
from .models import StructureIssue
from .rings import detect_ring_issues
from .staged_sanitize import staged_sanitize
from .valence import detect_valence_issues


def detect_issues(mol: Chem.Mol, config: Dict[str, Any] = None) -> List[StructureIssue]:
    """Run all detectors on a molecule (atom maps assigned if missing)."""
    config = config or {}
    mapped = assign_atom_maps(mol)
    try:
        mapped.UpdatePropertyCache(strict=False)
        Chem.GetSymmSSSR(mapped)
    except Exception:  # noqa: BLE001
        pass

    issues: List[StructureIssue] = []
    issues.extend(detect_valence_issues(mapped))
    issues.extend(detect_aromaticity_issues(mapped))
    issues.extend(detect_ring_issues(mapped))
    issues.extend(detect_functional_group_issues(mapped))
    issues.extend(detect_charge_issues(mapped, config))
    radicals_allow = bool(config.get("radicals", {}).get("allow", False))
    issues.extend(detect_radical_issues(mapped, allow=radicals_allow))

    # Deduplicate by (issue_code, frozenset(atom_maps), frozenset(bond_pairs))
    seen = set()
    unique: List[StructureIssue] = []
    for issue in issues:
        key = (
            issue.issue_code,
            tuple(sorted(issue.atom_map_ids)),
            tuple(sorted(issue.bond_atom_map_pairs)),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(issue)
    return unique


def count_issue_codes(issues: List[StructureIssue]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for issue in issues:
        counts[issue.issue_code] = counts.get(issue.issue_code, 0) + 1
    return counts


def sanitize_related_counts(mol: Chem.Mol) -> Dict[str, int]:
    report = staged_sanitize(mol)
    return {
        "sanitize_failure": 0 if report.overall_success else 1,
        "kekulization_failure_count": int("KEKULIZATION_FAILED" in report.failure_types),
        "invalid_valence_from_sanitize": int("INVALID_VALENCE" in report.failure_types),
    }
