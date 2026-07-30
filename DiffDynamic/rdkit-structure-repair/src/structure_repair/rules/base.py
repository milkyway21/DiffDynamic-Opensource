"""Repair rule abstract base class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from rdkit import Chem

from ..models import RepairCandidate, StructureIssue


class RepairRule(ABC):
    rule_id: str
    priority: int = 100

    @abstractmethod
    def detect(self, mol: Chem.Mol) -> List[StructureIssue]:
        ...

    @abstractmethod
    def generate_candidates(
        self,
        mol: Chem.Mol,
        issue: StructureIssue,
    ) -> List[RepairCandidate]:
        ...

    @abstractmethod
    def explain(self, issue: StructureIssue, candidate: RepairCandidate) -> str:
        ...
