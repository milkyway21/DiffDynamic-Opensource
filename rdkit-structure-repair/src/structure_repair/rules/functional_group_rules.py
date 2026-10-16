"""Functional group bond-order and charge joint repair."""

from __future__ import annotations

from typing import List, Optional

from rdkit import Chem

from ..atom_mapping import atom_map_to_idx, get_atom_by_map
from ..functional_groups import detect_functional_group_issues
from ..models import AtomEdit, BondEdit, RepairCandidate, StructureIssue
from ..mol_edit import finalize_candidate_mol, set_bond_type, set_formal_charge
from .base import RepairRule


class FunctionalGroupRepairRule(RepairRule):
    rule_id = "FUNCTIONAL_GROUP_REPAIR"
    priority = 40

    def detect(self, mol: Chem.Mol) -> List[StructureIssue]:
        return detect_functional_group_issues(mol)

    def generate_candidates(
        self, mol: Chem.Mol, issue: StructureIssue
    ) -> List[RepairCandidate]:
        generators = {
            "NITRO_BOND_ORDER_ERROR": self._nitro_candidates,
            "MISSING_POSITIVE_CHARGE": self._nitro_or_noxide_from_charge,
            "MISSING_NEGATIVE_CHARGE": self._nitro_or_noxide_from_charge,
            "CARBOXYLATE_BOND_ORDER_ERROR": self._carboxylate_candidates,
            "AMIDE_BOND_ORDER_ERROR": self._amide_candidates,
        }
        fn = generators.get(issue.issue_code)
        if fn is None:
            # n-oxide pattern in evidence
            if issue.evidence.get("pattern") == "n_oxide":
                return self._noxide_candidates(mol, issue)
            return []
        return fn(mol, issue)

    def _nitro_candidates(self, mol: Chem.Mol, issue: StructureIssue) -> List[RepairCandidate]:
        mapping = atom_map_to_idx(mol)
        # Find N in issue maps
        n_map = None
        for m in issue.atom_map_ids:
            atom = get_atom_by_map(mol, m)
            if atom is not None and atom.GetAtomicNum() == 7:
                n_map = m
                break
        if n_map is None:
            return []
        n_atom = get_atom_by_map(mol, n_map)
        o_maps = []
        for bond in n_atom.GetBonds():
            o = bond.GetOtherAtom(n_atom)
            if o.GetAtomicNum() == 8:
                o_maps.append(o.GetAtomMapNum())
        if len(o_maps) < 2:
            return []

        rw = Chem.RWMol(Chem.Mol(mol))
        bond_edits: List[BondEdit] = []
        atom_edits: List[AtomEdit] = []

        # Set N+
        ae = set_formal_charge(rw, n_map, 1)
        if ae:
            atom_edits.append(ae)
        # First O: =O (double, charge 0); Second O: -O- (single, charge -1)
        be1 = set_bond_type(rw, n_map, o_maps[0], Chem.BondType.DOUBLE)
        if be1:
            bond_edits.append(be1)
        be2 = set_bond_type(rw, n_map, o_maps[1], Chem.BondType.SINGLE)
        if be2:
            bond_edits.append(be2)
        ae_o = set_formal_charge(rw, o_maps[1], -1)
        if ae_o:
            atom_edits.append(ae_o)
        # Clear charge on first O
        ae_o0 = set_formal_charge(rw, o_maps[0], 0)
        if ae_o0:
            atom_edits.append(ae_o0)

        cand_mol = finalize_candidate_mol(rw)
        if cand_mol is None:
            return []
        return [
            RepairCandidate(
                candidate_id=f"{self.rule_id}_nitro",
                rule_id=self.rule_id,
                mol=cand_mol,
                bond_edits=bond_edits,
                atom_edits=atom_edits,
                edit_cost=float(len(bond_edits) + len(atom_edits)),
                validation_errors=[],
            )
        ]

    def _nitro_or_noxide_from_charge(
        self, mol: Chem.Mol, issue: StructureIssue
    ) -> List[RepairCandidate]:
        if issue.evidence.get("pattern") == "n_oxide":
            return self._noxide_candidates(mol, issue)
        # Try nitro if N has two O neighbors
        for m in issue.atom_map_ids:
            atom = get_atom_by_map(mol, m)
            if atom is None or atom.GetAtomicNum() != 7:
                continue
            o_count = sum(1 for n in atom.GetNeighbors() if n.GetAtomicNum() == 8)
            if o_count >= 2:
                return self._nitro_candidates(mol, issue)
        return self._noxide_candidates(mol, issue)

    def _noxide_candidates(self, mol: Chem.Mol, issue: StructureIssue) -> List[RepairCandidate]:
        n_map = None
        o_map = None
        for m in issue.atom_map_ids:
            atom = get_atom_by_map(mol, m)
            if atom is None:
                continue
            if atom.GetAtomicNum() == 7:
                n_map = m
            if atom.GetAtomicNum() == 8:
                o_map = m
        if n_map is None or o_map is None:
            return []
        rw = Chem.RWMol(Chem.Mol(mol))
        bond_edits = []
        atom_edits = []
        be = set_bond_type(rw, n_map, o_map, Chem.BondType.SINGLE)
        if be:
            bond_edits.append(be)
        ae_n = set_formal_charge(rw, n_map, 1)
        if ae_n:
            atom_edits.append(ae_n)
        ae_o = set_formal_charge(rw, o_map, -1)
        if ae_o:
            atom_edits.append(ae_o)
        cand_mol = finalize_candidate_mol(rw)
        if cand_mol is None:
            return []
        return [
            RepairCandidate(
                candidate_id=f"{self.rule_id}_noxide",
                rule_id=self.rule_id,
                mol=cand_mol,
                bond_edits=bond_edits,
                atom_edits=atom_edits,
                edit_cost=float(len(bond_edits) + len(atom_edits)),
                validation_errors=[],
            )
        ]

    def _carboxylate_candidates(
        self, mol: Chem.Mol, issue: StructureIssue
    ) -> List[RepairCandidate]:
        c_map = None
        o_maps = []
        for m in issue.atom_map_ids:
            atom = get_atom_by_map(mol, m)
            if atom is None:
                continue
            if atom.GetAtomicNum() == 6:
                c_map = m
            elif atom.GetAtomicNum() == 8:
                o_maps.append(m)
        if c_map is None or len(o_maps) < 2:
            return []

        out = []
        # Candidate A: carboxylate C(=O)[O-]
        rw = Chem.RWMol(Chem.Mol(mol))
        bes = []
        aes = []
        be1 = set_bond_type(rw, c_map, o_maps[0], Chem.BondType.DOUBLE)
        if be1:
            bes.append(be1)
        be2 = set_bond_type(rw, c_map, o_maps[1], Chem.BondType.SINGLE)
        if be2:
            bes.append(be2)
        ae = set_formal_charge(rw, o_maps[1], -1)
        if ae:
            aes.append(ae)
        cand = finalize_candidate_mol(rw)
        if cand is not None:
            out.append(
                RepairCandidate(
                    candidate_id=f"{self.rule_id}_carboxylate",
                    rule_id=self.rule_id,
                    mol=cand,
                    bond_edits=bes,
                    atom_edits=aes,
                    edit_cost=float(len(bes) + len(aes)),
                    validation_errors=[],
                )
            )
        # Candidate B: neutral carboxylic acid C(=O)O
        rw2 = Chem.RWMol(Chem.Mol(mol))
        bes2 = []
        aes2 = []
        be1 = set_bond_type(rw2, c_map, o_maps[0], Chem.BondType.DOUBLE)
        if be1:
            bes2.append(be1)
        be2 = set_bond_type(rw2, c_map, o_maps[1], Chem.BondType.SINGLE)
        if be2:
            bes2.append(be2)
        # set H on OH
        atom_o = get_atom_by_map(rw2, o_maps[1])
        if atom_o is not None:
            old_h = atom_o.GetNumExplicitHs()
            atom_o.SetNumExplicitHs(1)
            aes2.append(AtomEdit(o_maps[1], "num_explicit_hs", old_h, 1))
        cand2 = finalize_candidate_mol(rw2)
        if cand2 is not None:
            out.append(
                RepairCandidate(
                    candidate_id=f"{self.rule_id}_carboxylic_acid",
                    rule_id=self.rule_id,
                    mol=cand2,
                    bond_edits=bes2,
                    atom_edits=aes2,
                    edit_cost=float(len(bes2) + len(aes2)),
                    validation_errors=[],
                )
            )
        return out

    def _amide_candidates(self, mol: Chem.Mol, issue: StructureIssue) -> List[RepairCandidate]:
        if len(issue.atom_map_ids) < 2:
            return []
        maps = issue.atom_map_ids
        c_map = n_map = None
        for m in maps:
            atom = get_atom_by_map(mol, m)
            if atom is None:
                continue
            if atom.GetAtomicNum() == 6:
                c_map = m
            elif atom.GetAtomicNum() == 7:
                n_map = m
        if c_map is None or n_map is None:
            return []
        rw = Chem.RWMol(Chem.Mol(mol))
        be = set_bond_type(rw, c_map, n_map, Chem.BondType.SINGLE)
        if be is None:
            return []
        cand = finalize_candidate_mol(rw)
        if cand is None:
            return []
        return [
            RepairCandidate(
                candidate_id=f"{self.rule_id}_amide",
                rule_id=self.rule_id,
                mol=cand,
                bond_edits=[be],
                atom_edits=[],
                edit_cost=1.0,
                validation_errors=[],
            )
        ]

    def explain(self, issue: StructureIssue, candidate: RepairCandidate) -> str:
        return f"{self.rule_id}: fix functional group {issue.issue_code} via {candidate.candidate_id}"
