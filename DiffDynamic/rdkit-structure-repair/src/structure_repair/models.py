"""Core dataclasses for rdkit-structure-repair."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from rdkit import Chem


STATUS_UNCHANGED = "UNCHANGED"
STATUS_REPAIRED = "REPAIRED"
STATUS_AMBIGUOUS = "AMBIGUOUS"
STATUS_REJECTED = "REJECTED"
VALID_STATUSES = (
    STATUS_UNCHANGED,
    STATUS_REPAIRED,
    STATUS_AMBIGUOUS,
    STATUS_REJECTED,
)


@dataclass
class StructureIssue:
    issue_code: str
    severity: str
    atom_map_ids: List[int]
    bond_atom_map_pairs: List[Tuple[int, int]]
    description: str
    evidence: Dict[str, Any]
    repairable: bool
    candidate_rule_ids: List[str] = field(default_factory=list)


@dataclass
class BondEdit:
    atom_map_1: int
    atom_map_2: int
    old_bond_type: Optional[str]
    new_bond_type: Optional[str]
    action: str  # set | delete | add


@dataclass
class AtomEdit:
    atom_map_id: int
    property_name: str
    old_value: Any
    new_value: Any


@dataclass
class RepairCandidate:
    candidate_id: str
    rule_id: str
    mol: Chem.Mol
    bond_edits: List[BondEdit]
    atom_edits: List[AtomEdit]
    edit_cost: float
    validation_errors: List[str]
    score: Optional[float] = None


@dataclass
class RepairResult:
    molecule_id: str
    status: str
    original_mol: Chem.Mol
    repaired_mol: Optional[Chem.Mol]
    issues_before: List[StructureIssue]
    issues_after: List[StructureIssue]
    selected_candidate_id: Optional[str]
    applied_rule_ids: List[str]
    bond_edits: List[BondEdit]
    atom_edits: List[AtomEdit]
    score_before: float
    score_after: Optional[float]
    confidence: float
    reject_reason: Optional[str]


@dataclass
class SanitizationStepResult:
    step_name: str
    success: bool
    error_type: Optional[str] = None
    error_message: Optional[str] = None


@dataclass
class SanitizationReport:
    overall_success: bool
    steps: List[SanitizationStepResult] = field(default_factory=list)
    failure_types: List[str] = field(default_factory=list)

    @property
    def failed(self) -> bool:
        return not self.overall_success
