"""Ring opportunity detection: diene aromatize, medium/macrocycle break."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from rdkit import Chem

from ..atom_mapping import bond_map_pair
from ..models import StructureIssue
from ..rings import order_ring_atoms
from .planarity import is_near_planar


def _ring_bonds(mol: Chem.Mol, ring: Sequence[int]) -> List[Chem.Bond]:
    bonds = []
    n = len(ring)
    for i in range(n):
        b = mol.GetBondBetweenAtoms(int(ring[i]), int(ring[(i + 1) % n]))
        if b is not None:
            bonds.append(b)
    return bonds


def _count_ring_doubles(bonds: List[Chem.Bond]) -> int:
    return sum(1 for b in bonds if b.GetBondType() == Chem.BondType.DOUBLE)


def _is_all_carbon(mol: Chem.Mol, ring: Sequence[int]) -> bool:
    return all(mol.GetAtomWithIdx(int(i)).GetAtomicNum() == 6 for i in ring)


def detect_c6_diene_aromatize_opportunities(
    mol: Chem.Mol,
    config: Optional[Dict[str, Any]] = None,
) -> List[StructureIssue]:
    """C6 carbon rings with 1–2 isolated doubles + near-planar → benzene opportunity."""
    cfg = (config or {}).get("planarity", {})
    rmsd_thresh = float(cfg.get("rmsd_thresh", 0.10))
    dih_thresh = float(cfg.get("dihedral_thresh", 10.0))
    require_3d = bool(cfg.get("require_3d", True))

    issues: List[StructureIssue] = []
    try:
        Chem.GetSymmSSSR(mol)
    except Exception:  # noqa: BLE001
        pass

    for ring in mol.GetRingInfo().AtomRings():
        if len(ring) != 6 or not _is_all_carbon(mol, ring):
            continue
        if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
            continue
        bonds = _ring_bonds(mol, ring)
        if len(bonds) != 6:
            continue
        n_dbl = _count_ring_doubles(bonds)
        if n_dbl not in (1, 2):
            continue
        # Reject consecutive doubles (cumulene-like) — repair layer handles those
        types = [b.GetBondType() for b in bonds]
        consec = any(
            types[i] == Chem.BondType.DOUBLE and types[(i + 1) % 6] == Chem.BondType.DOUBLE
            for i in range(6)
        )
        if consec:
            continue

        ordered = order_ring_atoms(mol, list(ring))
        planar, pev = is_near_planar(
            mol, list(ring), ordered, rmsd_thresh=rmsd_thresh, dihedral_thresh=dih_thresh
        )
        if require_3d and not planar:
            # Still flag as soft opportunity if 2 doubles (chemist's default without 3D)
            if n_dbl < 2:
                continue
        maps = [mol.GetAtomWithIdx(i).GetAtomMapNum() for i in ring]
        bond_pairs = [
            bond_map_pair(mol, b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in bonds
        ]
        issues.append(
            StructureIssue(
                issue_code="C6_DIENE_NEAR_PLANAR",
                severity="warning",
                atom_map_ids=maps,
                bond_atom_map_pairs=bond_pairs,
                description=f"C6 carbon ring with {n_dbl} isolated double(s); aromatize to benzene",
                evidence={
                    "ring_idxs": list(ring),
                    "n_doubles": n_dbl,
                    "planar": planar,
                    **pev,
                },
                repairable=True,
                candidate_rule_ids=["T0_C6_DIENE_TO_BENZENE"],
            )
        )
    return issues


def _bond_break_score(mol: Chem.Mol, bond: Chem.Bond) -> float:
    """Higher = better break candidate (prefer long, unsubstituted, single bonds)."""
    score = 0.0
    if bond.GetBondType() == Chem.BondType.SINGLE:
        score += 5.0
    elif bond.GetBondType() == Chem.BondType.DOUBLE:
        score += 1.0
    else:
        score -= 10.0
    a1, a2 = bond.GetBeginAtom(), bond.GetEndAtom()
    # Prefer C–C over heteroatom bonds for opening (keep FG intact)
    if a1.GetAtomicNum() == 6 and a2.GetAtomicNum() == 6:
        score += 3.0
    # Prefer less substituted
    score -= 0.5 * (a1.GetDegree() + a2.GetDegree())
    # Prefer longer bonds if 3D available
    if mol.GetNumConformers() > 0:
        conf = mol.GetConformer()
        p1 = conf.GetAtomPosition(a1.GetIdx())
        p2 = conf.GetAtomPosition(a2.GetIdx())
        dist = ((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2 + (p1.z - p2.z) ** 2) ** 0.5
        score += dist
    return score


def pick_ring_bond_to_break(
    mol: Chem.Mol, ring: Sequence[int]
) -> Optional[Tuple[int, int]]:
    """Return (begin_idx, end_idx) of preferred ring bond to delete."""
    bonds = _ring_bonds(mol, ring)
    if not bonds:
        return None
    best = max(bonds, key=lambda b: _bond_break_score(mol, b))
    return best.GetBeginAtomIdx(), best.GetEndAtomIdx()


def detect_oversized_ring_issues(
    mol: Chem.Mol,
    config: Optional[Dict[str, Any]] = None,
) -> List[StructureIssue]:
    """Flag medium (7–8) and macro (≥9 / ≥10) rings for break or contract."""
    ring_cfg = (config or {}).get("rings", {})
    macro_min = int(ring_cfg.get("macrocycle_min_size", 10))
    medium_sizes = set(ring_cfg.get("medium_ring_sizes", [7, 8]))
    medium_min = int(ring_cfg.get("medium_ring_min_size", 7))

    issues: List[StructureIssue] = []
    try:
        Chem.GetSymmSSSR(mol)
    except Exception:  # noqa: BLE001
        pass

    for ring in mol.GetRingInfo().AtomRings():
        size = len(ring)
        # Skip already-aromatic 6-membered
        if size == 6 and all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
            continue

        maps = [mol.GetAtomWithIdx(i).GetAtomMapNum() for i in ring]
        bonds = _ring_bonds(mol, ring)
        bond_pairs = [
            bond_map_pair(mol, b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in bonds
        ]
        pick = pick_ring_bond_to_break(mol, ring)
        evidence: Dict[str, Any] = {
            "ring_idxs": list(ring),
            "ring_size": size,
            "break_bond_idxs": list(pick) if pick else None,
        }

        if size >= macro_min:
            issues.append(
                StructureIssue(
                    issue_code="MACROCYCLE_OVERSIZED",
                    severity="warning",
                    atom_map_ids=maps,
                    bond_atom_map_pairs=bond_pairs,
                    description=f"Macrocycle size {size} (≥{macro_min}): prefer bond break → chain",
                    evidence=evidence,
                    repairable=True,
                    candidate_rule_ids=["T0_BREAK_MACROCYCLE", "T2_CHAIN_LOCK_AFTER_OPEN"],
                )
            )
        elif size in medium_sizes or (medium_min <= size < macro_min and size not in (5, 6)):
            # 7–8 (and optionally 9): try contract to benzene-like 6, else break
            n_dbl = _count_ring_doubles(bonds)
            all_c = _is_all_carbon(mol, ring)
            ordered = order_ring_atoms(mol, list(ring))
            planar, pev = is_near_planar(mol, list(ring), ordered)
            evidence.update({"n_doubles": n_dbl, "all_carbon": all_c, "planar": planar, **pev})
            issues.append(
                StructureIssue(
                    issue_code="MEDIUM_RING_7_8",
                    severity="warning",
                    atom_map_ids=maps,
                    bond_atom_map_pairs=bond_pairs,
                    description=(
                        f"Medium ring size {size}: prefer contract→benzene/heteroarene; "
                        f"else break→lock"
                    ),
                    evidence=evidence,
                    repairable=True,
                    candidate_rule_ids=[
                        "T0_MEDIUM_RING_CONTRACT",
                        "T0_MEDIUM_RING_CONTRACT_AZA",
                        "T0_BREAK_MEDIUM_RING",
                        "T2_CHAIN_LOCK_AFTER_OPEN",
                    ],
                )
            )
    return issues


def detect_optimize_opportunities(
    mol: Chem.Mol, config: Optional[Dict[str, Any]] = None
) -> List[StructureIssue]:
    """All structural opportunities for the medchem optimize pass."""
    out: List[StructureIssue] = []
    out.extend(detect_c6_diene_aromatize_opportunities(mol, config))
    out.extend(detect_oversized_ring_issues(mol, config))
    return out
