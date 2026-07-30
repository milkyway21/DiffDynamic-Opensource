"""Valence and connectivity repair rules."""

from __future__ import annotations

from typing import List, Set, Tuple

from rdkit import Chem

from ..atom_mapping import atom_map_to_idx, bond_map_pair, get_atom_by_map
from ..models import RepairCandidate, StructureIssue
from ..mol_edit import delete_bond, finalize_candidate_mol
from ..valence import detect_valence_issues
from .base import RepairRule


class ValenceRepairRule(RepairRule):
    """Conservative: try deleting one over-valent bond when connectivity repair allows."""

    rule_id = "VALENCE_REPAIR"
    priority = 30

    def __init__(self, allow_bond_deletion: bool = True):
        self.allow_bond_deletion = allow_bond_deletion

    def detect(self, mol: Chem.Mol) -> List[StructureIssue]:
        return detect_valence_issues(mol)

    def generate_candidates(
        self, mol: Chem.Mol, issue: StructureIssue
    ) -> List[RepairCandidate]:
        if not self.allow_bond_deletion:
            return []
        if "OVERVALENT" not in issue.issue_code and "VALENCE_ERROR" not in issue.issue_code:
            return []
        out = []
        for m in issue.atom_map_ids:
            atom = get_atom_by_map(mol, m)
            if atom is None:
                continue
            for bond in atom.GetBonds():
                other = bond.GetOtherAtom(atom)
                if other.GetAtomicNum() == 1:
                    continue
                # Prefer deleting bonds that are in small rings / bridge-like
                m2 = other.GetAtomMapNum()
                rw = Chem.RWMol(Chem.Mol(mol))
                be = delete_bond(rw, m, m2)
                if be is None:
                    continue
                cand = finalize_candidate_mol(rw)
                if cand is None:
                    continue
                # Must remain a single heavy-atom connected component
                frags = Chem.GetMolFrags(cand, asMols=True)
                heavy_frags = [
                    f for f in frags if any(a.GetAtomicNum() > 1 for a in f.GetAtoms())
                ]
                if len(heavy_frags) != 1:
                    continue
                out.append(
                    RepairCandidate(
                        candidate_id=f"{self.rule_id}_del_{m}_{m2}",
                        rule_id=self.rule_id,
                        mol=cand,
                        bond_edits=[be],
                        atom_edits=[],
                        edit_cost=8.0,
                        validation_errors=[],
                    )
                )
        return out

    def explain(self, issue: StructureIssue, candidate: RepairCandidate) -> str:
        return f"{self.rule_id}: reduce overvalence for {issue.issue_code}"


class ConnectivityRepairRule(RepairRule):
    rule_id = "CONNECTIVITY_REPAIR"
    priority = 20

    def __init__(self, allow_bond_deletion: bool = True, max_deleted: int = 1):
        self.allow_bond_deletion = allow_bond_deletion
        self.max_deleted = max_deleted
        self._valence_rule = ValenceRepairRule(allow_bond_deletion=allow_bond_deletion)

    def detect(self, mol: Chem.Mol) -> List[StructureIssue]:
        # Reuse overvalent as connectivity cue
        return [
            i
            for i in detect_valence_issues(mol)
            if "OVERVALENT" in i.issue_code
        ]

    def generate_candidates(
        self, mol: Chem.Mol, issue: StructureIssue
    ) -> List[RepairCandidate]:
        cands = self._valence_rule.generate_candidates(mol, issue)
        for c in cands:
            c.rule_id = self.rule_id
            c.candidate_id = c.candidate_id.replace("VALENCE_REPAIR", self.rule_id)
        return cands[: max(1, self.max_deleted) * 4]

    def explain(self, issue: StructureIssue, candidate: RepairCandidate) -> str:
        return f"{self.rule_id}: delete spurious bond for {issue.issue_code}"
