"""3D planarity evidence used to decide whether a ring should be aromatized.

Reconstructed molecules carry the pocket-conditioned conformer produced by the
diffusion sampler.  A ring that is geometrically flat but was assigned single /
isolated double bonds by OpenBabel bond perception is strong evidence that the
ring was meant to be aromatic.  These helpers turn that geometry into numbers.

All functions return ``None`` when there is no usable 3D conformer so callers
can distinguish "not planar" from "no evidence available".
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from rdkit import Chem


DEFAULT_RMSD_THRESHOLD = 0.10
DEFAULT_TORSION_THRESHOLD = 10.0


def has_3d_conformer(mol: Chem.Mol) -> bool:
    try:
        if mol is None or mol.GetNumConformers() == 0:
            return False
        return bool(mol.GetConformer().Is3D())
    except Exception:  # noqa: BLE001
        return False


def _ring_coords(
    mol: Chem.Mol,
    ring_idxs: Sequence[int],
    conf_id: int = -1,
) -> Optional[np.ndarray]:
    if not has_3d_conformer(mol):
        return None
    try:
        conf = mol.GetConformer(conf_id)
    except Exception:  # noqa: BLE001
        return None
    n_atoms = mol.GetNumAtoms()
    points: List[Sequence[float]] = []
    for idx in ring_idxs:
        i = int(idx)
        if i < 0 or i >= n_atoms:
            return None
        p = conf.GetAtomPosition(i)
        points.append((p.x, p.y, p.z))
    if not points:
        return None
    return np.asarray(points, dtype=float)


def plane_rmsd(coords: np.ndarray) -> float:
    """RMS distance of points from their SVD best-fit plane (Angstrom)."""
    centered = coords - coords.mean(axis=0)
    # The right-singular vector with the smallest singular value is the normal
    # of the least-squares plane.
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    normal = vt[-1]
    distances = centered @ normal
    return float(np.sqrt(np.mean(distances ** 2)))


def ring_plane_rmsd(
    mol: Chem.Mol,
    ring_idxs: Sequence[int],
    conf_id: int = -1,
) -> Optional[float]:
    coords = _ring_coords(mol, ring_idxs, conf_id)
    if coords is None or len(coords) < 4:
        return None
    return plane_rmsd(coords)


def _dihedral(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> Optional[float]:
    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2
    n1 = np.cross(b0, b1)
    n2 = np.cross(b1, b2)
    norm1 = np.linalg.norm(n1)
    norm2 = np.linalg.norm(n2)
    if norm1 < 1e-8 or norm2 < 1e-8:
        return None
    n1 = n1 / norm1
    n2 = n2 / norm2
    b1n = b1 / max(np.linalg.norm(b1), 1e-8)
    m = np.cross(n1, b1n)
    x = float(np.dot(n1, n2))
    y = float(np.dot(m, n2))
    return math.degrees(math.atan2(y, x))


def max_ring_torsion_deviation(
    mol: Chem.Mol,
    ordered_ring: Sequence[int],
    conf_id: int = -1,
) -> Optional[float]:
    """Largest deviation from planarity across all ring torsions, in degrees.

    A planar ring has every endocyclic torsion near 0 or 180 degrees, so both
    are treated as zero deviation.  A cyclohexane chair lands near 55.
    """
    coords = _ring_coords(mol, ordered_ring, conf_id)
    if coords is None or len(coords) < 4:
        return None
    n = len(coords)
    worst = 0.0
    for i in range(n):
        angle = _dihedral(
            coords[i],
            coords[(i + 1) % n],
            coords[(i + 2) % n],
            coords[(i + 3) % n],
        )
        if angle is None:
            continue
        magnitude = abs(angle)
        deviation = min(magnitude, abs(180.0 - magnitude))
        worst = max(worst, deviation)
    return worst


def ring_planarity(
    mol: Chem.Mol,
    ordered_ring: Sequence[int],
    rmsd_threshold: float = DEFAULT_RMSD_THRESHOLD,
    torsion_threshold: float = DEFAULT_TORSION_THRESHOLD,
    conf_id: int = -1,
) -> Optional[Dict[str, Any]]:
    """Full planarity report, or ``None`` when there is no 3D evidence."""
    rmsd = ring_plane_rmsd(mol, ordered_ring, conf_id)
    if rmsd is None:
        return None
    torsion = max_ring_torsion_deviation(mol, ordered_ring, conf_id)
    planar = rmsd <= rmsd_threshold
    if torsion is not None:
        planar = planar and torsion <= torsion_threshold
    return {
        "plane_rmsd": round(rmsd, 4),
        "max_torsion_deviation": round(torsion, 2) if torsion is not None else None,
        "rmsd_threshold": rmsd_threshold,
        "torsion_threshold": torsion_threshold,
        "planar": bool(planar),
    }


def is_ring_planar(
    mol: Chem.Mol,
    ordered_ring: Sequence[int],
    rmsd_threshold: float = DEFAULT_RMSD_THRESHOLD,
    torsion_threshold: float = DEFAULT_TORSION_THRESHOLD,
    conf_id: int = -1,
) -> Optional[bool]:
    report = ring_planarity(mol, ordered_ring, rmsd_threshold, torsion_threshold, conf_id)
    if report is None:
        return None
    return bool(report["planar"])


def planarity_thresholds_from_config(config: Dict[str, Any]) -> Dict[str, float]:
    arom = (config or {}).get("aromaticity", {}) or {}
    return {
        "rmsd_threshold": float(arom.get("planarity_rmsd_threshold", DEFAULT_RMSD_THRESHOLD)),
        "torsion_threshold": float(
            arom.get("planarity_torsion_threshold", DEFAULT_TORSION_THRESHOLD)
        ),
    }
