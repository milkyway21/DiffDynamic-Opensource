"""Contract 7/8-membered rings to benzene / heteroarenes without bond-breaking open-chain.

Diffusion often builds a near-planar 7- or 8-ring that is chemically meant to be a
6π aromatic (benzene, pyridine, …) with one or two extra CH₂ inserted. Prefer:

  size 7 → delete one endocyclic CH₂/CH  → 6-arom
  size 8 → delete two adjacent CH₂      → 6-arom

Optional: all-carbon 6 → aza (pyridine) as an extra medchem candidate.

Breaking the ring into a chain remains a fallback in ``ring_break.py``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from rdkit import Chem

from .aromatize import _aromatize_ring
from .ring_ops import detect_oversized_ring_issues, order_ring_atoms
from .transforms import PROP_PARENT_IDX, TransformCandidate, canonical_smiles, heavy_atom_count

TRANSFORM_CONTRACT_6 = "T0_MEDIUM_RING_CONTRACT"
TRANSFORM_CONTRACT_AZA = "T0_MEDIUM_RING_CONTRACT_AZA"


def _ring_neighbors(mol: Chem.Mol, idx: int, ring_set: Set[int]) -> List[int]:
    atom = mol.GetAtomWithIdx(int(idx))
    return [n.GetIdx() for n in atom.GetNeighbors() if n.GetIdx() in ring_set]


def _exo_heavy_count(mol: Chem.Mol, idx: int, ring_set: Set[int]) -> int:
    atom = mol.GetAtomWithIdx(int(idx))
    return sum(1 for n in atom.GetNeighbors() if n.GetIdx() not in ring_set and n.GetAtomicNum() > 1)


def _is_removable_ring_atom(mol: Chem.Mol, idx: int, ring_set: Set[int]) -> bool:
    """CH₂ / unsubstituted CH carbon with exactly two ring neighbors."""
    atom = mol.GetAtomWithIdx(int(idx))
    if atom.GetAtomicNum() != 6:
        return False
    if len(_ring_neighbors(mol, idx, ring_set)) != 2:
        return False
    # Keep substituted carbons (branch points) — deleting them loses R groups.
    if _exo_heavy_count(mol, idx, ring_set) > 0:
        return False
    # Prefer saturated insertion (CH2) but also allow CH if H-count fits.
    return atom.GetTotalNumHs() >= 1


def _delete_atoms_reconnect(
    mol: Chem.Mol,
    delete: Sequence[int],
    reconnect: Sequence[Tuple[int, int]],
) -> Optional[Chem.Mol]:
    """Delete atoms and add reconnect bonds; returns sanitized mol or None.

    Kekulize first so existing aromatic rings (e.g. a pendant phenyl) survive as
    explicit double bonds instead of being wiped to single bonds.
    """
    delete_set = set(int(i) for i in delete)
    work = Chem.Mol(mol)
    try:
        Chem.Kekulize(work, clearAromaticFlags=True)
    except Exception:  # noqa: BLE001
        # Non-aromatic mols are fine; continue with a clean copy.
        work = Chem.Mol(mol)

    rw = Chem.RWMol(work)
    for a, b in reconnect:
        a, b = int(a), int(b)
        if a in delete_set or b in delete_set:
            continue
        if rw.GetBondBetweenAtoms(a, b) is None:
            rw.AddBond(a, b, Chem.BondType.SINGLE)

    for idx in sorted(delete_set, reverse=True):
        rw.RemoveAtom(idx)

    out = rw.GetMol()
    try:
        out.UpdatePropertyCache(strict=False)
        Chem.SanitizeMol(out)
    except Exception:  # noqa: BLE001
        try:
            Chem.SanitizeMol(
                out,
                sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL
                ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES,
            )
        except Exception:  # noqa: BLE001
            return None
    if len(Chem.GetMolFrags(out)) != 1:
        return None
    return out


def _force_aromatic_on_remaining(
    mol: Chem.Mol,
    keep_old_idxs: Sequence[int],
    old_to_new: Dict[int, int],
) -> Optional[Chem.Mol]:
    """Aromatize the contracted 6-ring via the shared T0 aromatize helper."""
    new_ring = []
    for old in keep_old_idxs:
        if old not in old_to_new:
            return None
        new_ring.append(old_to_new[old])
    if len(new_ring) != 6:
        return None

    arom = _aromatize_ring(mol, new_ring)
    if arom is None:
        return None
    # Reject "false aromatic" flags: kekulized ring must have 3 doubles (benzene-like)
    if not _ring_has_benzene_kekule(arom, new_ring):
        return None
    return arom


def _ring_has_benzene_kekule(mol: Chem.Mol, ring_idxs: Sequence[int]) -> bool:
    """True iff the 6-ring kekulizes to exactly three double bonds."""
    ring_set = {int(i) for i in ring_idxs}
    if len(ring_set) != 6:
        return False
    if not all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring_set):
        return False
    cp = Chem.Mol(mol)
    try:
        Chem.Kekulize(cp, clearAromaticFlags=True)
    except Exception:  # noqa: BLE001
        return False
    n_dbl = 0
    for bond in cp.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if a in ring_set and b in ring_set and bond.GetBondType() == Chem.BondType.DOUBLE:
            n_dbl += 1
    return n_dbl == 3


def _index_map_after_delete(n_atoms: int, delete: Sequence[int]) -> Dict[int, int]:
    """old_idx → new_idx after deleting atoms (new→old inverted later)."""
    delete_set = set(int(i) for i in delete)
    mapping: Dict[int, int] = {}
    new_i = 0
    for old in range(n_atoms):
        if old in delete_set:
            continue
        mapping[old] = new_i
        new_i += 1
    return mapping


def _stamp_provenance(product: Chem.Mol, old_to_new: Dict[int, int]) -> Dict[int, int]:
    """Return new→old map and stamp PROP_PARENT_IDX."""
    new_to_old = {v: k for k, v in old_to_new.items()}
    for ni, oi in new_to_old.items():
        if ni < product.GetNumAtoms():
            product.GetAtomWithIdx(ni).SetIntProp(PROP_PARENT_IDX, int(oi))
    return new_to_old


def _aza_variants(mol: Chem.Mol, ring_new_idxs: Sequence[int]) -> List[Chem.Mol]:
    """Replace one aromatic CH with N (pyridine-like)."""
    out: List[Chem.Mol] = []
    for ni in ring_new_idxs:
        atom = mol.GetAtomWithIdx(int(ni))
        if atom.GetAtomicNum() != 6:
            continue
        if atom.GetTotalNumHs() < 1:
            continue
        if any(n.GetAtomicNum() > 1 and n.GetIdx() not in set(ring_new_idxs) for n in atom.GetNeighbors()):
            continue
        rw = Chem.RWMol(Chem.Mol(mol))
        a = rw.GetAtomWithIdx(int(ni))
        a.SetAtomicNum(7)
        a.SetNoImplicit(False)
        a.SetNumExplicitHs(0)
        try:
            p = rw.GetMol()
            Chem.SanitizeMol(p)
        except Exception:  # noqa: BLE001
            continue
        out.append(p)
    return out


def _contract_ring_once(
    mol: Chem.Mol,
    ring: Sequence[int],
    delete: Sequence[int],
) -> Optional[Tuple[Chem.Mol, Dict[int, int], List[int]]]:
    """Delete atoms from ring, reconnect, aromatize remaining 6."""
    ring_set = set(int(i) for i in ring)
    delete = [int(i) for i in delete]
    if any(d not in ring_set for d in delete):
        return None
    keep = [i for i in ring if i not in set(delete)]
    if len(keep) != 6:
        return None

    ordered = order_ring_atoms(mol, list(ring))
    if not ordered:
        ordered = list(ring)

    # Find reconnect pairs: neighbors of deleted segment endpoints that remain
    reconnect: List[Tuple[int, int]] = []
    del_set = set(delete)
    # Walk ordered ring; for each gap created by deletion, link flanking keepers
    n = len(ordered)
    for i, idx in enumerate(ordered):
        if idx not in del_set:
            continue
        # find previous keeper
        prev = None
        for k in range(1, n):
            cand = ordered[(i - k) % n]
            if cand not in del_set:
                prev = cand
                break
        nxt = None
        for k in range(1, n):
            cand = ordered[(i + k) % n]
            if cand not in del_set:
                nxt = cand
                break
        if prev is not None and nxt is not None and prev != nxt:
            reconnect.append((prev, nxt))

    # Dedup reconnect
    uniq = []
    seen_b = set()
    for a, b in reconnect:
        key = (min(a, b), max(a, b))
        if key not in seen_b:
            seen_b.add(key)
            uniq.append((a, b))

    old_to_new = _index_map_after_delete(mol.GetNumAtoms(), delete)
    contracted = _delete_atoms_reconnect(mol, delete, uniq)
    if contracted is None:
        return None

    arom = _force_aromatic_on_remaining(contracted, keep, old_to_new)
    if arom is None:
        return None

    new_to_old = _stamp_provenance(arom, old_to_new)
    new_ring = [old_to_new[i] for i in keep]
    return arom, new_to_old, new_ring


def _deletion_plans(mol: Chem.Mol, ring: Sequence[int]) -> List[Tuple[int, ...]]:
    """Enumerate atom-deletion plans for size 7 (1 atom) / 8 (2 adjacent)."""
    ring_set = set(int(i) for i in ring)
    ordered = order_ring_atoms(mol, list(ring)) or list(ring)
    removable = [i for i in ordered if _is_removable_ring_atom(mol, i, ring_set)]
    plans: List[Tuple[int, ...]] = []

    if len(ring) == 7:
        for i in removable:
            plans.append((i,))
    elif len(ring) == 8:
        # Prefer two adjacent removable carbons
        for k, a in enumerate(ordered):
            b = ordered[(k + 1) % len(ordered)]
            if a in removable and b in removable:
                plans.append((a, b))
        # Fallback: single removable + any adjacent carbon with ≤1 exo heavy
        if not plans:
            for k, a in enumerate(ordered):
                if a not in removable:
                    continue
                for b in (ordered[(k - 1) % len(ordered)], ordered[(k + 1) % len(ordered)]):
                    atom_b = mol.GetAtomWithIdx(int(b))
                    if atom_b.GetAtomicNum() != 6:
                        continue
                    if _exo_heavy_count(mol, b, ring_set) > 0:
                        continue
                    if len(_ring_neighbors(mol, b, ring_set)) != 2:
                        continue
                    plans.append((a, b))
    elif len(ring) == 9:
        # Rare: delete 3 consecutive removable if possible
        for k in range(len(ordered)):
            trip = tuple(ordered[(k + j) % len(ordered)] for j in range(3))
            if all(t in removable for t in trip):
                plans.append(trip)

    # Dedup
    uniq = []
    seen = set()
    for p in plans:
        key = tuple(sorted(p))
        if key not in seen:
            seen.add(key)
            uniq.append(p)
    return uniq


def generate_ring_contract_candidates(
    parent: Chem.Mol,
    config: Optional[Dict[str, Any]] = None,
    rejected: Optional[Dict[str, int]] = None,
) -> List[TransformCandidate]:
    config = config or {}
    ring_cfg = config.get("rings", {})
    enable_aza = bool(ring_cfg.get("contract_offer_aza", True))
    max_plans = int(ring_cfg.get("contract_max_plans_per_ring", 4))

    issues = detect_oversized_ring_issues(parent, config)
    candidates: List[TransformCandidate] = []
    seen = set()
    parent_heavy = heavy_atom_count(parent)

    for issue in issues:
        if issue.issue_code != "MEDIUM_RING_7_8":
            continue
        ring = issue.evidence.get("ring_idxs") or []
        if len(ring) not in (7, 8, 9):
            continue

        plans = _deletion_plans(parent, ring)[:max_plans]
        if not plans:
            if rejected is not None:
                rejected["contract_no_removable_atom"] = (
                    rejected.get("contract_no_removable_atom", 0) + 1
                )
            continue

        for plan in plans:
            result = _contract_ring_once(parent, ring, plan)
            if result is None:
                if rejected is not None:
                    rejected["contract_sanitize_failed"] = (
                        rejected.get("contract_sanitize_failed", 0) + 1
                    )
                continue
            mol, index_map, new_ring = result
            smi = canonical_smiles(mol)
            if not smi or smi in seen:
                continue
            seen.add(smi)
            hetero = any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6 for i in new_ring)
            label = "杂芳六元环" if hetero else "苯环"
            candidates.append(
                TransformCandidate(
                    transform_id=TRANSFORM_CONTRACT_6,
                    tier="T0",
                    name=f"中环缩成{label}",
                    rationale=(
                        f"{len(ring)} 元环删 {len(plan)} 个插原子 → {label} "
                        f"(delete={list(plan)})"
                    ),
                    mol=mol,
                    cost=1.0,
                    requires_3d=False,
                    reward_bonus=0.35,
                    parent_index_map=index_map,
                    new_atom_indices=[],
                    delta_heavy=heavy_atom_count(mol) - parent_heavy,
                    smiles=smi,
                )
            )

            if enable_aza and not hetero:
                for aza in _aza_variants(mol, new_ring)[:3]:
                    asmi = canonical_smiles(aza)
                    if not asmi or asmi in seen:
                        continue
                    seen.add(asmi)
                    # Provenance: same heavy skeleton as contracted mol
                    aza_map = dict(index_map)
                    for atom in aza.GetAtoms():
                        if atom.HasProp(PROP_PARENT_IDX):
                            aza_map[atom.GetIdx()] = int(atom.GetProp(PROP_PARENT_IDX))
                        elif atom.GetIdx() in index_map:
                            atom.SetIntProp(PROP_PARENT_IDX, index_map[atom.GetIdx()])
                    candidates.append(
                        TransformCandidate(
                            transform_id=TRANSFORM_CONTRACT_AZA,
                            tier="T0",
                            name="中环缩成吡啶类",
                            rationale=f"{len(ring)} 元环 → 苯 → aza 杂芳",
                            mol=aza,
                            cost=1.2,
                            requires_3d=False,
                            reward_bonus=0.32,
                            parent_index_map=aza_map,
                            new_atom_indices=[],
                            delta_heavy=heavy_atom_count(aza) - parent_heavy,
                            smiles=asmi,
                        )
                    )

    return candidates
