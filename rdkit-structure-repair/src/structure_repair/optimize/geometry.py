"""3D contract: pin surviving atoms, embed only new ones, clash-check the pocket."""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D

CORE_DISPLACEMENT_TOLERANCE = 0.01  # Angstrom; surviving atoms must not drift


def has_conformer(mol: Chem.Mol) -> bool:
    if mol is None or mol.GetNumConformers() == 0:
        return False
    conf = mol.GetConformer()
    return bool(conf.Is3D() and np.isfinite(conf.GetPositions()).all())


def transfer_conformer(
    parent: Chem.Mol,
    candidate: Chem.Mol,
    parent_index_map: Dict[int, int],
) -> bool:
    """Copy parent coordinates onto every mapped candidate atom.

    Returns False when any candidate heavy atom lacks a parent mapping (would
    leave an atom without a pose).
    """
    if not has_conformer(parent) or not valid_mapping(parent, candidate, parent_index_map):
        return False
    # Every candidate atom must be mapped for a pure transfer.
    if any(i not in parent_index_map for i in range(candidate.GetNumAtoms())):
        return False
    parent_conf = parent.GetConformer()
    if candidate.GetNumConformers():
        candidate.RemoveAllConformers()
    conf = Chem.Conformer(candidate.GetNumAtoms())
    conf.Set3D(True)
    for cand_idx, parent_idx in parent_index_map.items():
        conf.SetAtomPosition(cand_idx, parent_conf.GetAtomPosition(parent_idx))
    candidate.AddConformer(conf, assignId=True)
    return True


def max_core_displacement(
    parent: Chem.Mol,
    candidate: Chem.Mol,
    parent_index_map: Dict[int, int],
) -> Optional[float]:
    if not has_conformer(parent) or not has_conformer(candidate):
        return None
    pc = parent.GetConformer()
    cc = candidate.GetConformer()
    max_d = 0.0
    for cand_idx, parent_idx in parent_index_map.items():
        d = pc.GetAtomPosition(parent_idx).Distance(cc.GetAtomPosition(cand_idx))
        max_d = max(max_d, float(d))
    return max_d


def valid_mapping(parent, candidate, index_map):
    """Reject aliases and out-of-bounds provenance; never guess from atom counts."""
    return bool(index_map) and len(set(index_map.values())) == len(index_map) and all(
        isinstance(c, int) and isinstance(p, int)
        and 0 <= c < candidate.GetNumAtoms() and 0 <= p < parent.GetNumAtoms()
        for c, p in index_map.items()
    )


def affected_atom_indices(parent, candidate, index_map):
    """Changed element/charge/bond/connectivity plus its immediate neighbors."""
    changed = {i for i in range(candidate.GetNumAtoms()) if i not in index_map}
    reverse = {p: c for c, p in index_map.items()}
    for ci, pi in index_map.items():
        atom, before = candidate.GetAtomWithIdx(ci), parent.GetAtomWithIdx(pi)
        signature = lambda a: (a.GetAtomicNum(), a.GetFormalCharge(), a.GetIsAromatic(),
                               a.GetTotalNumHs(), str(a.GetHybridization()))
        if signature(atom) != signature(before):
            changed.add(ci)
        original_edges = {
            (reverse.get(b.GetOtherAtomIdx(pi), -1), str(b.GetBondType()))
            for b in before.GetBonds()
        }
        candidate_edges = {(b.GetOtherAtomIdx(ci), str(b.GetBondType())) for b in atom.GetBonds()}
        if original_edges != candidate_edges:
            changed.add(ci)
    # Relax adjacent atoms as bond angles and torsions can change without new atoms.
    expanded = set(changed)
    for idx in changed:
        expanded.update(a.GetIdx() for a in candidate.GetAtomWithIdx(idx).GetNeighbors())
    return expanded


def validate_geometry(mol, atom_indices=None):
    """Finite 3D coordinates, bond lengths, local angles, clashes and assigned stereo."""
    if not has_conformer(mol):
        return {"status": "failed", "reason": "missing_or_nonfinite_3d"}
    xyz = np.asarray(mol.GetConformer().GetPositions())
    focus = set(range(mol.GetNumAtoms()) if atom_indices is None else atom_indices)
    table = Chem.GetPeriodicTable()
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if i not in focus and j not in focus:
            continue
        radius = sum(table.GetRcovalent(mol.GetAtomWithIdx(k).GetAtomicNum()) for k in (i, j))
        distance = float(np.linalg.norm(xyz[i] - xyz[j]))
        if not 0.65 * radius < distance < 1.35 * radius:
            return {"status": "failed", "reason": "bond_length", "atoms": [i, j], "distance": distance}
    for i in focus:
        atom = mol.GetAtomWithIdx(i)
        neighbors = [a.GetIdx() for a in atom.GetNeighbors() if a.GetAtomicNum() > 1]
        # Genuine small rings have necessarily compressed angles.
        if atom.IsInRingSize(3) or atom.IsInRingSize(4):
            continue
        hybrid = str(atom.GetHybridization())
        limits = {"SP": (155, 180.01), "SP2": (95, 150), "SP3": (94, 150)}
        if hybrid not in limits:
            continue
        for n, j in enumerate(neighbors):
            for k in neighbors[n + 1:]:
                v, w = xyz[j] - xyz[i], xyz[k] - xyz[i]
                denom = np.linalg.norm(v) * np.linalg.norm(w)
                if denom < 1e-10:
                    return {"status": "failed", "reason": "collapsed_coordinates"}
                angle = float(np.degrees(np.arccos(np.clip(np.dot(v, w) / denom, -1, 1))))
                lo, hi = limits[hybrid]
                if not lo <= angle <= hi:
                    return {"status": "failed", "reason": "bond_angle", "atoms": [j, i, k], "angle": angle}
    for i in focus:
        for j in range(i + 1, mol.GetNumAtoms()):
            if mol.GetBondBetweenAtoms(i, j) is None and np.linalg.norm(xyz[i] - xyz[j]) < 0.9:
                return {"status": "failed", "reason": "intramolecular_clash", "atoms": [i, j]}
    try:
        from ..identity_check import compare_stereochemistry, compare_double_bond_stereo
        posed = Chem.Mol(mol)
        Chem.AssignStereochemistryFrom3D(posed, replaceExistingTags=True)
        identity = {i: i for i in range(mol.GetNumAtoms())}
        if not compare_stereochemistry(mol, posed, identity):
            return {"status": "failed", "reason": "3d_stereochemistry_mismatch"}
        if not compare_double_bond_stereo(mol, posed, identity):
            return {"status": "failed", "reason": "3d_double_bond_stereo_mismatch"}
    except Exception as exc:
        return {"status": "failed", "reason": "stereo_validation_failed", "detail": str(exc)}
    return {"status": "passed", "checked_atoms": sorted(focus)}


def _remove_added_hydrogens(posed, template):
    """Remove only H atoms appended by AddHs, preserving every input index."""
    count = template.GetNumAtoms()
    editable = Chem.RWMol(posed)
    for idx in range(posed.GetNumAtoms() - 1, count - 1, -1):
        if editable.GetAtomWithIdx(idx).GetAtomicNum() != 1:
            raise ValueError("Unexpected non-H atom appended during embedding")
        editable.RemoveAtom(idx)
    out = editable.GetMol()
    for idx in range(count):
        source, atom = template.GetAtomWithIdx(idx), out.GetAtomWithIdx(idx)
        atom.SetNumExplicitHs(source.GetNumExplicitHs())
        atom.SetNoImplicit(source.GetNoImplicit())
        # Removing appended H restores the exact original neighbor order.
        atom.SetChiralTag(source.GetChiralTag())
    Chem.SanitizeMol(out)
    return out


def generate_conformer(mol, seed=42):
    """Make a molecular conformer without claiming any protein alignment."""
    out = Chem.AddHs(Chem.Mol(mol))
    out.RemoveAllConformers()
    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    params.numThreads = 1
    if AllChem.EmbedMolecule(out, params) != 0:
        return None
    try:
        props = AllChem.MMFFGetMoleculeProperties(out, mmffVariant="MMFF94s")
        ff = AllChem.MMFFGetMoleculeForceField(out, props) if props else AllChem.UFFGetMoleculeForceField(out)
        if ff is None or ff.Minimize(maxIts=1000) != 0:
            return None
    except Exception:
        return None
    out = _remove_added_hydrogens(out, mol)
    # ETKDG embeds disconnected ions independently at the origin. Separate
    # freshly generated components for inspection without dropping or neutralizing
    # them; this operation is never used on a supplied protein-aligned pose.
    fragments = Chem.GetMolFrags(out)
    if len(fragments) > 1:
        conf = out.GetConformer()
        cursor = None
        for fragment in fragments:
            positions = np.asarray([list(conf.GetAtomPosition(i)) for i in fragment])
            minimum, maximum = float(positions[:, 0].min()), float(positions[:, 0].max())
            shift = 0.0 if cursor is None else cursor + 4.0 - minimum
            for idx in fragment:
                point = conf.GetAtomPosition(idx)
                conf.SetAtomPosition(idx, (point.x + shift, point.y, point.z))
            cursor = maximum + shift
    return out if has_conformer(out) else None


def embed_new_atoms(
    parent: Chem.Mol,
    candidate: Chem.Mol,
    parent_index_map: Dict[int, int],
    seed: int = 0xC0FFEE,
    attempts: int = 10,
    time_budget_s: float = 3.0,
) -> Optional[Chem.Mol]:
    """Relax every affected region, fixing only atoms whose environment is unchanged.

    Element/bond/ring edits need optimization even when no atoms are added.
    Every failure is explicit; a transferred but chemically stale pose is never
    returned as a successful optimization.
    """
    if not has_conformer(parent) or not valid_mapping(parent, candidate, parent_index_map):
        return None
    affected = affected_atom_indices(parent, candidate, parent_index_map)
    if not affected and len(parent_index_map) == candidate.GetNumAtoms():
        out = Chem.Mol(candidate)
        if transfer_conformer(parent, out, parent_index_map) and validate_geometry(out)["status"] == "passed":
            return out
        return None
    fixed = {ci: pi for ci, pi in parent_index_map.items() if ci not in affected}
    deadline = time.monotonic() + max(0.01, float(time_budget_s))
    for attempt in range(max(1, int(attempts))):
        if time.monotonic() > deadline:
            break
        try:
            out = Chem.AddHs(Chem.Mol(candidate))
            out.RemoveAllConformers()
            params = AllChem.ETKDGv3()
            params.randomSeed = (int(seed) + attempt) % 2147483647
            params.numThreads = 1
            coords = {ci: parent.GetConformer().GetAtomPosition(pi) for ci, pi in fixed.items()}
            if coords:
                params.SetCoordMap(coords)
            if AllChem.EmbedMolecule(out, params) != 0:
                continue
            # Coordinate constraints preserve distances, not the absolute frame.
            alignment = fixed or parent_index_map
            AllChem.AlignMol(out, parent, atomMap=list(alignment.items()))
            for ci, pi in fixed.items():
                out.GetConformer().SetAtomPosition(ci, parent.GetConformer().GetAtomPosition(pi))
            props = AllChem.MMFFGetMoleculeProperties(out, mmffVariant="MMFF94s")
            ff = AllChem.MMFFGetMoleculeForceField(out, props) if props else AllChem.UFFGetMoleculeForceField(out)
            if ff is None:
                continue
            for ci in fixed:
                ff.AddFixedPoint(ci)
            ff.Initialize()
            if ff.Minimize(maxIts=1000) != 0:
                continue
            # Keep original explicit H and all original atom indices unchanged.
            out = _remove_added_hydrogens(out, candidate)
            if max_core_displacement(parent, out, fixed) > CORE_DISPLACEMENT_TOLERANCE:
                continue
            if validate_geometry(out)["status"] != "passed":
                continue
            return out
        except Exception:
            continue
    return None


def restore_hydrogens(mol: Chem.Mol) -> Chem.Mol:
    try:
        return Chem.AddHs(mol, addCoords=True)
    except Exception:  # noqa: BLE001
        return mol


def load_pocket_coords(pdb_path: str) -> Optional[np.ndarray]:
    try:
        from rdkit.Chem import rdmolfiles

        prot = rdmolfiles.MolFromPDBFile(pdb_path, sanitize=False, removeHs=False)
        if prot is None or prot.GetNumConformers() == 0:
            return None
        conf = prot.GetConformer()
        coords = []
        for atom in prot.GetAtoms():
            if atom.GetAtomicNum() <= 1:
                continue
            p = conf.GetAtomPosition(atom.GetIdx())
            coords.append([p.x, p.y, p.z])
        if not coords:
            return None
        return np.asarray(coords, dtype=float)
    except Exception:  # noqa: BLE001
        return None


class PocketClashChecker:
    def __init__(self, protein_path: Optional[str], cutoff: float = 2.2):
        self.cutoff = float(cutoff)
        self._coords: Optional[np.ndarray] = None
        if protein_path:
            self._coords = load_pocket_coords(protein_path)

    @property
    def active(self) -> bool:
        return self._coords is not None and len(self._coords) > 0

    def has_clash(
        self, mol: Chem.Mol, atom_indices: Optional[Sequence[int]] = None
    ) -> bool:
        if not self.active or not has_conformer(mol):
            return False
        conf = mol.GetConformer()
        pts = []
        indices = (
            list(atom_indices)
            if atom_indices is not None
            else [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
        )
        for idx in indices:
            atom = mol.GetAtomWithIdx(int(idx))
            if atom.GetAtomicNum() <= 1:
                continue
            p = conf.GetAtomPosition(int(idx))
            pts.append([p.x, p.y, p.z])
        if not pts:
            return False
        lig = np.asarray(pts, dtype=float)
        try:
            from scipy.spatial import cKDTree

            tree = cKDTree(self._coords)
            dists, _ = tree.query(lig, k=1)
            return bool(np.min(dists) < self.cutoff)
        except Exception:  # noqa: BLE001
            dmin = np.sqrt(
                np.min(((lig[:, None, :] - self._coords[None, :, :]) ** 2).sum(-1))
            )
            return bool(dmin < self.cutoff)
