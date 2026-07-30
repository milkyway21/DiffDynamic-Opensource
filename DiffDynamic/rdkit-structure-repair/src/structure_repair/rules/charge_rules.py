"""Charge-focused repair rules."""

from __future__ import annotations

from typing import List

from rdkit import Chem

from ..atom_mapping import get_atom_by_map
from ..charges import detect_charge_issues
from ..models import RepairCandidate, StructureIssue
from ..mol_edit import finalize_candidate_mol, set_formal_charge
from .base import RepairRule
from .functional_group_rules import FunctionalGroupRepairRule


class ChargeRepairRule(RepairRule):
    rule_id = "CHARGE_REPAIR"
    priority = 35

    def detect(self, mol: Chem.Mol) -> List[StructureIssue]:
        return [
            i
            for i in detect_charge_issues(mol)
            if i.repairable
        ]

    def generate_candidates(
        self, mol: Chem.Mol, issue: StructureIssue
    ) -> List[RepairCandidate]:
        # Defer nitro-like to FG rule
        fg = FunctionalGroupRepairRule()
        fg_cands = fg.generate_candidates(mol, issue)
        if fg_cands:
            for c in fg_cands:
                c.rule_id = self.rule_id
                c.candidate_id = c.candidate_id.replace("FUNCTIONAL_GROUP_REPAIR", self.rule_id)
            return fg_cands

        out = []
        if issue.issue_code in {"MISSING_POSITIVE_CHARGE", "IMPLAUSIBLE_NEUTRAL_HYPERVALENT_ATOM"}:
            for m in issue.atom_map_ids:
                atom = get_atom_by_map(mol, m)
                if atom is None:
                    continue
                if atom.GetAtomicNum() == 7 and atom.GetFormalCharge() == 0 and atom.GetTotalDegree() == 4:
                    rw = Chem.RWMol(Chem.Mol(mol))
                    ae = set_formal_charge(rw, m, 1)
                    cand = finalize_candidate_mol(rw)
                    if cand is None:
                        continue
                    out.append(
                        RepairCandidate(
                            candidate_id=f"{self.rule_id}_nplus_{m}",
                            rule_id=self.rule_id,
                            mol=cand,
                            bond_edits=[],
                            atom_edits=[ae] if ae else [],
                            edit_cost=1.0,
                            validation_errors=[],
                        )
                    )
        return out

    def explain(self, issue: StructureIssue, candidate: RepairCandidate) -> str:
        return f"{self.rule_id}: charge repair for {issue.issue_code}"
