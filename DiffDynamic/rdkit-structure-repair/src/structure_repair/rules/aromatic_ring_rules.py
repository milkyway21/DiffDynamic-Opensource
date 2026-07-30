"""C6 aromatic recovery and C5 heteroaromatic recovery rules."""

from __future__ import annotations

from typing import List, Optional, Sequence

from rdkit import Chem

from ..atom_mapping import atom_map_to_idx
from ..candidate_scorer import compute_edit_cost
from ..models import AtomEdit, BondEdit, RepairCandidate, StructureIssue
from ..mol_edit import clear_aromatic_flags, finalize_candidate_mol, set_bond_type
from ..rings import detect_ring_issues, order_ring_atoms
from .base import RepairRule


def generate_c6_kekule_candidates(
    mol: Chem.Mol,
    ordered_ring_atoms: List[int],
) -> List[Chem.Mol]:
    if len(ordered_ring_atoms) != 6:
        return []
    if any(mol.GetAtomWithIdx(idx).GetAtomicNum() != 6 for idx in ordered_ring_atoms):
        return []

    candidates: List[Chem.Mol] = []
    for phase in (0, 1):
        rw = Chem.RWMol(Chem.Mol(mol))
        for atom_idx in ordered_ring_atoms:
            rw.GetAtomWithIdx(atom_idx).SetIsAromatic(False)
            # Clear explicit H so sanitize can reassign for aromatic carbons
            rw.GetAtomWithIdx(atom_idx).SetNumExplicitHs(0)
            rw.GetAtomWithIdx(atom_idx).SetNoImplicit(False)

        valid = True
        for i in range(6):
            begin_idx = ordered_ring_atoms[i]
            end_idx = ordered_ring_atoms[(i + 1) % 6]
            bond = rw.GetBondBetweenAtoms(begin_idx, end_idx)
            if bond is None:
                valid = False
                break
            bond.SetIsAromatic(False)
            if (i + phase) % 2 == 0:
                bond.SetBondType(Chem.BondType.DOUBLE)
            else:
                bond.SetBondType(Chem.BondType.SINGLE)
        if not valid:
            continue

        candidate = rw.GetMol()
        try:
            candidate.UpdatePropertyCache(strict=False)
            Chem.SanitizeMol(candidate)
        except Exception:  # noqa: BLE001
            continue
        candidates.append(candidate)
    return candidates


def _c6_should_attempt_aromatic(mol: Chem.Mol, ring: Sequence[int]) -> bool:
    """Reject saturated cyclohexane: each C typically has 2 implicit/explicit H and degree 2."""
    if len(ring) != 6:
        return False
    h_count = 0
    for idx in ring:
        atom = mol.GetAtomWithIdx(idx)
        try:
            h_count += atom.GetTotalNumHs(includeNeighbors=False)
        except Exception:  # noqa: BLE001
            h_count += atom.GetNumExplicitHs()
        if atom.GetDegree() > 3:
            return False
    bonds = []
    for i in range(6):
        b = mol.GetBondBetweenAtoms(int(ring[i]), int(ring[(i + 1) % 6]))
        if b is None:
            return False
        bonds.append(b)
    has_double = any(b.GetBondType() == Chem.BondType.DOUBLE for b in bonds)
    all_single = all(b.GetBondType() == Chem.BondType.SINGLE for b in bonds)
    consec = False
    for i in range(6):
        if (
            bonds[i].GetBondType() == Chem.BondType.DOUBLE
            and bonds[(i + 1) % 6].GetBondType() == Chem.BondType.DOUBLE
        ):
            consec = True
            break

    if consec or (has_double and sum(1 for b in bonds if b.GetBondType() == Chem.BondType.DOUBLE) >= 3):
        return True
    # Aggressive: attempt aromatic recovery for all-single C6 when H suggests
    # partial unsaturation (benzene~6, diene~8, even up to ~10 for substituted).
    if all_single and h_count <= 10:
        if h_count <= 8:
            return True
        try:
            tmp = Chem.Mol(mol)
            Chem.SanitizeMol(tmp)
            # Valid saturated-ish molecule with many H — skip cyclohexane-like
            return h_count <= 7
        except Exception:  # noqa: BLE001
            return True
    return False


class C6AromaticRecoveryRule(RepairRule):
    rule_id = "C6_AROMATIC_RECOVERY"
    priority = 50

    def detect(self, mol: Chem.Mol) -> List[StructureIssue]:
        issues = detect_ring_issues(mol)
        return [
            i
            for i in issues
            if i.issue_code
            in {
                "ALL_SINGLE_CONJUGATED_RING",
                "ALL_DOUBLE_RING",
                "CONSECUTIVE_DOUBLE_BONDS_IN_RING",
                "C6_RING_AROMATICITY_LOST",
                "MULTIPLE_DOUBLE_BONDS_SHARING_ATOM_IN_RING",
            }
            and len(i.atom_map_ids) == 6
        ]

    def generate_candidates(
        self, mol: Chem.Mol, issue: StructureIssue
    ) -> List[RepairCandidate]:
        mapping = atom_map_to_idx(mol)
        if not all(m in mapping for m in issue.atom_map_ids):
            return []
        ring_idxs = [mapping[m] for m in issue.atom_map_ids]
        # Prefer ring_idxs from evidence when available (already a proper ring)
        evidence_ring = issue.evidence.get("ring_idxs")
        if evidence_ring and len(evidence_ring) == 6:
            ring_idxs = list(evidence_ring)
        ordered = order_ring_atoms(mol, ring_idxs)
        if len(ordered) != 6:
            # Fall back to SSSR ring containing these atoms
            try:
                Chem.GetSymmSSSR(mol)
            except Exception:  # noqa: BLE001
                pass
            target = set(ring_idxs)
            ordered = []
            for ring in mol.GetRingInfo().AtomRings():
                if len(ring) == 6 and target.issubset(set(ring)):
                    ordered = order_ring_atoms(mol, list(ring))
                    break
            if len(ordered) != 6:
                return []
        if not _c6_should_attempt_aromatic(mol, ordered):
            return []

        out: List[RepairCandidate] = []
        kekule_mols = generate_c6_kekule_candidates(mol, ordered)
        for phase, cand_mol in enumerate(kekule_mols):
            # Rebuild edits description from maps
            bond_edits: List[BondEdit] = []
            atom_edits: List[AtomEdit] = []
            for i in range(6):
                a = mol.GetAtomWithIdx(ordered[i])
                if a.GetIsAromatic():
                    atom_edits.append(
                        AtomEdit(a.GetAtomMapNum(), "is_aromatic", True, False)
                    )
            for i in range(6):
                i1, i2 = ordered[i], ordered[(i + 1) % 6]
                b_old = mol.GetBondBetweenAtoms(i1, i2)
                b_new = cand_mol.GetBondBetweenAtoms(i1, i2)
                if b_old is None or b_new is None:
                    continue
                m1 = mol.GetAtomWithIdx(i1).GetAtomMapNum()
                m2 = mol.GetAtomWithIdx(i2).GetAtomMapNum()
                old_t = str(b_old.GetBondType()).replace("BondType.", "")
                new_t = str(b_new.GetBondType()).replace("BondType.", "")
                if b_new.GetIsAromatic():
                    new_t = "AROMATIC"
                if old_t != new_t or b_old.GetIsAromatic() != b_new.GetIsAromatic():
                    bond_edits.append(
                        BondEdit(
                            min(m1, m2),
                            max(m1, m2),
                            old_t,
                            new_t,
                            "set",
                        )
                    )
            cand = RepairCandidate(
                candidate_id=f"{self.rule_id}_p{phase}",
                rule_id=self.rule_id,
                mol=cand_mol,
                bond_edits=bond_edits,
                atom_edits=atom_edits,
                edit_cost=0.0,
                validation_errors=[],
            )
            out.append(cand)
        return out

    def explain(self, issue: StructureIssue, candidate: RepairCandidate) -> str:
        return (
            f"{self.rule_id}: recover benzene-like C6 ring for issue "
            f"{issue.issue_code} via Kekulé phase candidate {candidate.candidate_id}"
        )


def _enumerate_c5_kekule(mol: Chem.Mol, ordered: List[int]) -> List[Chem.Mol]:
    """Enumerate a few single/double patterns for 5-membered heterocycles."""
    patterns = []
    # Common patterns: positions of double bonds (relative indices)
    # pyrrole-like: doubles at 1-2, 3-4
    patterns.append({0, 2})
    patterns.append({1, 3})
    patterns.append({0, 3})
    patterns.append({1, 4})

    results = []
    for dbl_set in patterns:
        rw = Chem.RWMol(Chem.Mol(mol))
        for idx in ordered:
            atom = rw.GetAtomWithIdx(idx)
            atom.SetIsAromatic(False)
            atom.SetNumExplicitHs(0)
            atom.SetNoImplicit(False)
        valid = True
        for i in range(5):
            b = rw.GetBondBetweenAtoms(ordered[i], ordered[(i + 1) % 5])
            if b is None:
                valid = False
                break
            b.SetIsAromatic(False)
            b.SetBondType(Chem.BondType.DOUBLE if i in dbl_set else Chem.BondType.SINGLE)
        if not valid:
            continue
        # Ensure heteroatom H for pyrrole-like N
        for idx in ordered:
            atom = rw.GetAtomWithIdx(idx)
            if atom.GetAtomicNum() == 7 and atom.GetFormalCharge() == 0:
                # If N has degree 2 in ring only (plus maybe H), set explicit H=1
                if atom.GetDegree() == 2:
                    atom.SetNumExplicitHs(1)
        cand = rw.GetMol()
        try:
            cand.UpdatePropertyCache(strict=False)
            Chem.SanitizeMol(cand)
            results.append(cand)
        except Exception:  # noqa: BLE001
            continue
    return results


class C5HeteroaromaticRecoveryRule(RepairRule):
    rule_id = "C5_HETEROAROMATIC_RECOVERY"
    priority = 55

    def detect(self, mol: Chem.Mol) -> List[StructureIssue]:
        return [
            i
            for i in detect_ring_issues(mol)
            if i.issue_code == "C5_HETEROAROMATICITY_LOST"
        ]

    def generate_candidates(
        self, mol: Chem.Mol, issue: StructureIssue
    ) -> List[RepairCandidate]:
        mapping = atom_map_to_idx(mol)
        if not all(m in mapping for m in issue.atom_map_ids):
            return []
        ring_idxs = [mapping[m] for m in issue.atom_map_ids]
        ordered = order_ring_atoms(mol, ring_idxs)
        if len(ordered) != 5:
            return []
        out = []
        for i, cand_mol in enumerate(_enumerate_c5_kekule(mol, ordered)):
            out.append(
                RepairCandidate(
                    candidate_id=f"{self.rule_id}_{i}",
                    rule_id=self.rule_id,
                    mol=cand_mol,
                    bond_edits=[],
                    atom_edits=[],
                    edit_cost=2.0,
                    validation_errors=[],
                )
            )
        return out

    def explain(self, issue: StructureIssue, candidate: RepairCandidate) -> str:
        return f"{self.rule_id}: restore 5-membered heteroaromaticity ({issue.issue_code})"
