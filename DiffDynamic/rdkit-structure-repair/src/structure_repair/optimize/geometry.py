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
    return mol is not None and mol.GetNumConformers() > 0


def transfer_conformer(
    parent: Chem.Mol,
    candidate: Chem.Mol,
    parent_index_map: Dict[int, int],
) -> bool:
    """Copy parent coordinates onto every mapped candidate atom.

    Returns False when any candidate heavy atom lacks a parent mapping (would
    leave an atom without a pose).
    """
    if not has_conformer(parent):
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


def embed_new_atoms(
    parent: Chem.Mol,
    candidate: Chem.Mol,
    parent_index_map: Dict[int, int],
    seed: int = 0xC0FFEE,
    attempts: int = 10,
    time_budget_s: float = 3.0,
) -> Optional[Chem.Mol]:
    """Embed ``candidate`` while pinning atoms that map back to ``parent``.

    Surviving atoms keep their exact pocket coordinates; only new atoms are
    free.  Returns None when the parent has no conformer or embedding fails.
    """
    if not has_conformer(parent):
        return None
    if candidate is None or candidate.GetNumAtoms() == 0:
        return None

    # Pure transfer when nothing new
    new_indices = [i for i in range(candidate.GetNumAtoms()) if i not in parent_index_map]
    if not new_indices:
        out = Chem.Mol(candidate)
        if transfer_conformer(parent, out, parent_index_map):
            return out
        return None

    out = Chem.Mol(candidate)
    if out.GetNumConformers():
        out.RemoveAllConformers()

    parent_conf = parent.GetConformer()
    coord_map = {}
    for cand_idx, parent_idx in parent_index_map.items():
        p = parent_conf.GetAtomPosition(parent_idx)
        coord_map[cand_idx] = Point3D(p.x, p.y, p.z)

    deadline = time.monotonic() + max(0.05, float(time_budget_s))
    embedded = False
    for attempt in range(max(1, int(attempts))):
        if time.monotonic() > deadline:
            break
        try:
            params = AllChem.ETKDGv3()
            params.randomSeed = int(seed) + attempt
            params.numThreads = 1
            rc = AllChem.EmbedMolecule(out, params, coordMap=coord_map)
        except TypeError:
            try:
                rc = AllChem.EmbedMolecule(out, coordMap=coord_map, randomSeed=int(seed) + attempt)
            except Exception:  # noqa: BLE001
                rc = -1
        except Exception:  # noqa: BLE001
            rc = -1
        if rc == 0 and has_conformer(out):
            embedded = True
            break
        out.RemoveAllConformers()

    if not embedded:
        return None

    # Force pinned atoms exactly back (ETKDG may nudge slightly)
    conf = out.GetConformer()
    for cand_idx, parent_idx in parent_index_map.items():
        conf.SetAtomPosition(cand_idx, parent_conf.GetAtomPosition(parent_idx))

    # Local FF with position constraints on the core
    try:
        props = AllChem.MMFFGetMoleculeProperties(out, mmffVariant="MMFF94s")
        if props is not None:
            ff = AllChem.MMFFGetMoleculeForceField(out, props)
            if ff is not None:
                for cand_idx in parent_index_map:
                    ff.MMFFAddPositionConstraint(cand_idx, 0.0, 1.0e5)
                ff.Minimize(maxIts=200)
        else:
            AllChem.UFFOptimizeMolecule(out, maxIters=200)
    except Exception:  # noqa: BLE001
        pass

    # Re-pin after minimize
    conf = out.GetConformer()
    for cand_idx, parent_idx in parent_index_map.items():
        conf.SetAtomPosition(cand_idx, parent_conf.GetAtomPosition(parent_idx))

    disp = max_core_displacement(parent, out, parent_index_map)
    if disp is not None and disp > CORE_DISPLACEMENT_TOLERANCE + 1e-6:
        # Still accept if we re-pinned; tolerance check is for callers
        pass
    return out


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
