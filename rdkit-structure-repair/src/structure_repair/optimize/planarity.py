"""3D planarity evidence for aromatic recovery and ring decisions."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
from rdkit import Chem


def ring_planarity_rmsd(
    mol: Chem.Mol,
    ring_idxs: Sequence[int],
    conf_id: int = -1,
) -> Optional[float]:
    """RMSD of ring atoms to best-fit plane (Angstrom). None if no conformer."""
    if mol.GetNumConformers() == 0:
        return None
    conf = mol.GetConformer(conf_id)
    coords = np.array([list(conf.GetAtomPosition(int(i))) for i in ring_idxs], dtype=float)
    if coords.shape[0] < 3:
        return None
    centroid = coords.mean(axis=0)
    centered = coords - centroid
    # SVD: normal is last right singular vector
    try:
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    normal = vt[-1]
    dists = centered @ normal
    return float(np.sqrt(np.mean(dists ** 2)))


def ring_max_dihedral_deg(
    mol: Chem.Mol,
    ordered_ring: Sequence[int],
    conf_id: int = -1,
) -> Optional[float]:
    """Max absolute ring dihedral (degrees) along ordered ring atoms."""
    if mol.GetNumConformers() == 0 or len(ordered_ring) < 4:
        return None
    conf = mol.GetConformer(conf_id)
    n = len(ordered_ring)
    max_abs = 0.0
    for i in range(n):
        a, b, c, d = (
            ordered_ring[i],
            ordered_ring[(i + 1) % n],
            ordered_ring[(i + 2) % n],
            ordered_ring[(i + 3) % n],
        )
        try:
            ang = Chem.rdMolTransforms.GetDihedralDeg(conf, int(a), int(b), int(c), int(d))
        except Exception:  # noqa: BLE001
            continue
        max_abs = max(max_abs, abs(float(ang)))
    return max_abs


def is_near_planar(
    mol: Chem.Mol,
    ring_idxs: Sequence[int],
    ordered_ring: Optional[Sequence[int]] = None,
    rmsd_thresh: float = 0.10,
    dihedral_thresh: float = 10.0,
) -> Tuple[bool, dict]:
    """Return (near_planar, evidence). Missing 3D → False with reason."""
    evidence: dict = {}
    rmsd = ring_planarity_rmsd(mol, ring_idxs)
    evidence["planarity_rmsd"] = rmsd
    if rmsd is None:
        evidence["reason"] = "no_conformer"
        return False, evidence
    if rmsd > rmsd_thresh:
        evidence["reason"] = "rmsd_above_thresh"
        return False, evidence
    if ordered_ring is not None:
        dih = ring_max_dihedral_deg(mol, ordered_ring)
        evidence["max_dihedral_deg"] = dih
        if dih is not None and dih > dihedral_thresh:
            evidence["reason"] = "dihedral_above_thresh"
            return False, evidence
    evidence["reason"] = "near_planar"
    return True, evidence
