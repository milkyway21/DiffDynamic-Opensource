"""Cumulene validation and ring consecutive-double-bond repair."""

from __future__ import annotations

from typing import List, Optional, Tuple

from rdkit import Chem

from ..atom_mapping import atom_map_to_idx
from ..models import RepairCandidate, StructureIssue
from ..rings import detect_ring_issues, order_ring_atoms
from .aromatic_ring_rules import generate_c6_kekule_candidates
from .base import RepairRule

# Legal linear patterns (SMARTS) — matched on sanitized copies when possible
_LEGAL_SMARTS = [
    "[C]=[C]=[C]",  # allene
    "[C]=[C]=[C]=[C]",  # cumulene
    "[C]=[C]=O",  # ketene
    "[N]=[C]=[N]",  # carbodiimide
    "[N]=[C]=O",  # isocyanate
    "O=[C]=O",  # CO2-like
]


def _matches_legal_cumulene(mol: Chem.Mol) -> bool:
    for smarts in _LEGAL_SMARTS:
        patt = Chem.MolFromSmarts(smarts)
        if patt is None:
            continue
        try:
            if mol.HasSubstructMatch(patt):
                return True
        except Exception:  # noqa: BLE001
            continue
    return False


def _find_consecutive_double_chains(mol: Chem.Mol) -> List[List[int]]:
    """Return lists of atom indices participating in C=C=C chains."""
    chains = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() not in (6, 7, 8):
            continue
        dbl_neighbors = []
        for bond in atom.GetBonds():
            if bond.GetBondType() == Chem.BondType.DOUBLE:
                dbl_neighbors.append(bond.GetOtherAtomIdx(atom.GetIdx()))
        if len(dbl_neighbors) >= 2:
            chains.append([dbl_neighbors[0], atom.GetIdx(), dbl_neighbors[1]])
    return chains


class CumuleneValidationAndRepairRule(RepairRule):
    rule_id = "CUMULENE_VALIDATION_AND_REPAIR"
    priority = 60

    def detect(self, mol: Chem.Mol) -> List[StructureIssue]:
        issues: List[StructureIssue] = []
        # Ring consecutive doubles already from ring detector
        for issue in detect_ring_issues(mol):
            if issue.issue_code in {
                "CONSECUTIVE_DOUBLE_BONDS_IN_RING",
                "MULTIPLE_DOUBLE_BONDS_SHARING_ATOM_IN_RING",
                "ALL_DOUBLE_RING",
            }:
                issue.candidate_rule_ids = list(
                    set(issue.candidate_rule_ids + [self.rule_id])
                )
                issues.append(issue)

        chains = _find_consecutive_double_chains(mol)
        ring_info = mol.GetRingInfo()
        for chain in chains:
            in_ring = any(ring_info.NumAtomRings(i) > 0 for i in chain)
            maps = [mol.GetAtomWithIdx(i).GetAtomMapNum() for i in chain]
            if not in_ring and _matches_legal_cumulene(mol):
                # Legal — informational only
                issues.append(
                    StructureIssue(
                        issue_code="IMPLAUSIBLE_CUMULENE",
                        severity="info",
                        atom_map_ids=maps,
                        bond_atom_map_pairs=[],
                        description="Linear cumulene matches known legal pattern — keep",
                        evidence={"legal": True, "chain": chain},
                        repairable=False,
                        candidate_rule_ids=[],
                    )
                )
            elif in_ring:
                issues.append(
                    StructureIssue(
                        issue_code="CONSECUTIVE_DOUBLE_BONDS_IN_RING",
                        severity="error",
                        atom_map_ids=maps,
                        bond_atom_map_pairs=[],
                        description="Ring cumulene-like consecutive doubles",
                        evidence={"chain": chain},
                        repairable=True,
                        candidate_rule_ids=[self.rule_id, "C6_AROMATIC_RECOVERY"],
                    )
                )
            else:
                # Ambiguous linear — mark for possible reject
                issues.append(
                    StructureIssue(
                        issue_code="IMPLAUSIBLE_CUMULENE",
                        severity="warning",
                        atom_map_ids=maps,
                        bond_atom_map_pairs=[],
                        description="Linear consecutive doubles without known legal pattern",
                        evidence={"legal": False, "chain": chain},
                        repairable=False,
                        candidate_rule_ids=[],
                    )
                )
        # Dedup
        seen = set()
        unique = []
        for i in issues:
            key = (i.issue_code, tuple(sorted(i.atom_map_ids)))
            if key in seen:
                continue
            seen.add(key)
            unique.append(i)
        return unique

    def generate_candidates(
        self, mol: Chem.Mol, issue: StructureIssue
    ) -> List[RepairCandidate]:
        if not issue.repairable:
            return []
        # Prefer C6 kekule if ring maps available
        mapping = atom_map_to_idx(mol)
        ring_idxs = []
        evidence_ring = issue.evidence.get("ring_idxs")
        if evidence_ring:
            ring_idxs = list(evidence_ring)
        elif len(issue.atom_map_ids) >= 6:
            ring_idxs = [mapping[m] for m in issue.atom_map_ids if m in mapping]
            if len(ring_idxs) >= 6:
                # take first 6 that form a ring
                ring_info = mol.GetRingInfo()
                for ring in ring_info.AtomRings():
                    if len(ring) == 6 and all(
                        mol.GetAtomWithIdx(i).GetAtomicNum() == 6 for i in ring
                    ):
                        if set(issue.atom_map_ids) & {
                            mol.GetAtomWithIdx(i).GetAtomMapNum() for i in ring
                        }:
                            ring_idxs = list(ring)
                            break

        out: List[RepairCandidate] = []
        if len(ring_idxs) == 6:
            ordered = order_ring_atoms(mol, ring_idxs)
            for phase, cand_mol in enumerate(generate_c6_kekule_candidates(mol, ordered)):
                out.append(
                    RepairCandidate(
                        candidate_id=f"{self.rule_id}_kekule_{phase}",
                        rule_id=self.rule_id,
                        mol=cand_mol,
                        bond_edits=[],
                        atom_edits=[],
                        edit_cost=3.0,
                        validation_errors=[],
                    )
                )
            # Also try reducing one double bond to single at consecutive pair
            for i in range(6):
                rw = Chem.RWMol(Chem.Mol(mol))
                for idx in ordered:
                    rw.GetAtomWithIdx(idx).SetIsAromatic(False)
                b = rw.GetBondBetweenAtoms(ordered[i], ordered[(i + 1) % 6])
                if b is None or b.GetBondType() != Chem.BondType.DOUBLE:
                    continue
                b.SetIsAromatic(False)
                b.SetBondType(Chem.BondType.SINGLE)
                cand = rw.GetMol()
                try:
                    cand.UpdatePropertyCache(strict=False)
                    Chem.SanitizeMol(cand)
                except Exception:  # noqa: BLE001
                    continue
                out.append(
                    RepairCandidate(
                        candidate_id=f"{self.rule_id}_reduce_{i}",
                        rule_id=self.rule_id,
                        mol=cand,
                        bond_edits=[],
                        atom_edits=[],
                        edit_cost=1.0,
                        validation_errors=[],
                    )
                )
        return out

    def explain(self, issue: StructureIssue, candidate: RepairCandidate) -> str:
        return f"{self.rule_id}: address consecutive doubles ({issue.issue_code})"
