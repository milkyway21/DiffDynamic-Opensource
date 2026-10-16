"""Charge-focused repair rules."""

from __future__ import annotations

from typing import List

from rdkit import Chem

from ..atom_mapping import get_atom_by_map
from ..charges import detect_charge_issues
from ..models import RepairCandidate, StructureIssue
from ..mol_edit import finalize_candidate_mol, set_bond_type, set_formal_charge
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
        # Correct all unambiguous neutral tetracoordinate nitrogens together.
        # Sanitizing one site at a time cannot repair a molecule with two such
        # sites, since the untouched site still prevents full sanitization.
        neutral_four = [
            atom for atom in mol.GetAtoms()
            if atom.GetAtomicNum() == 7 and atom.GetFormalCharge() == 0
            and atom.GetTotalDegree() == 4 and not atom.GetIsAromatic()
            and all(b.GetBondType() == Chem.BondType.SINGLE for b in atom.GetBonds())
        ]
        fg = FunctionalGroupRepairRule()
        sites = {a.GetAtomMapNum(): ("nplus", []) for a in neutral_four}
        for group in fg.detect(mol):
            if group.issue_code == "NITRO_BOND_ORDER_ERROR":
                n_map = group.evidence["n_map"]
                nitrogen = get_atom_by_map(mol, n_map)
                oxygen_maps = [
                    a.GetAtomMapNum() for a in nitrogen.GetNeighbors()
                    if a.GetAtomicNum() == 8
                ]
                sites[n_map] = ("nitro", oxygen_maps)
            elif group.evidence.get("pattern") == "n_oxide":
                n_map, o_map = group.atom_map_ids
                sites[n_map] = ("noxide", [o_map])
        if len(sites) > 1:
            rw = Chem.RWMol(mol)
            edits, bonds = [], []
            for n_map, (kind, oxygens) in sites.items():
                edits.append(set_formal_charge(rw, n_map, 1))
                if kind == "nitro":
                    bonds.append(set_bond_type(rw, n_map, oxygens[0], Chem.BondType.DOUBLE))
                    edits.append(set_formal_charge(rw, oxygens[0], 0))
                if kind in {"nitro", "noxide"}:
                    negative_o = oxygens[-1]
                    bonds.append(set_bond_type(rw, n_map, negative_o, Chem.BondType.SINGLE))
                    edits.append(set_formal_charge(rw, negative_o, -1))
            cand = finalize_candidate_mol(rw)
            if cand is not None:
                return [RepairCandidate(
                    candidate_id=f"{self.rule_id}_joint_charge", rule_id=self.rule_id,
                    mol=cand, bond_edits=[e for e in bonds if e],
                    atom_edits=[e for e in edits if e], edit_cost=0.0,
                    validation_errors=[],
                )]

        # Defer nitro-like to FG rule
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
