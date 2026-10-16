"""Fragments responsibilities extracted from scaffold_sites.py."""

from __future__ import annotations

from diffdynamic.chemistry.sites.state import (
    Any,
    Dict,
    List,
    _ALIPHATIC_RING_KINDS,
    _ARYL_KINDS,
    _LINKER_KINDS,
    np,
)


def _site_frame(
    site: Dict[str, Any],
    direction_mode: str = "exit",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return a scaffold-derived exit frame without reading target-side atoms."""
    from diffdynamic.chemistry.sites.gaussian import _attachment_direction

    anchor = np.asarray(site["anchor_pos"], dtype=np.float64)
    mode = str(direction_mode or "exit").strip().lower()
    raw_direction = None
    if mode in ("pocket", "interface", "record_centroid"):
        raw_direction = site.get("pocket_center_offset")
    if raw_direction is None:
        raw_direction = site.get("exit_direction") or _attachment_direction(site)
    direction = np.asarray(raw_direction or [1.0, 0.0, 0.0], dtype=np.float64)
    direction /= max(float(np.linalg.norm(direction)), 1e-8)

    raw_tangent = site.get("tangent_direction") or [0.0, 1.0, 0.0]
    tangent = np.asarray(raw_tangent, dtype=np.float64)
    tangent -= np.dot(tangent, direction) * direction
    if float(np.linalg.norm(tangent)) <= 1e-8:
        basis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(float(np.dot(basis, direction))) > 0.9:
            basis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        tangent = basis - np.dot(basis, direction) * direction
    tangent /= max(float(np.linalg.norm(tangent)), 1e-8)

    raw_binormal = site.get("binormal_direction")
    binormal = np.asarray(
        raw_binormal or np.cross(direction, tangent), dtype=np.float64
    )
    binormal /= max(float(np.linalg.norm(binormal)), 1e-8)
    return anchor, direction, tangent, binormal


def _fragment_atom_keys(fragment: Dict[str, Any]) -> List[str]:
    keys: List[str] = []
    for key, count in sorted((fragment.get("atom_counts") or {}).items()):
        keys.extend([str(key)] * int(count))
    return keys


def _fragment_has_type(fragment: Dict[str, Any], type_key: str) -> bool:
    """Return whether a generic motif contains one requested atom class."""
    return int((fragment.get("atom_counts") or {}).get(str(type_key), 0)) > 0


def _order_fragments_for_attachment(
    fragments: List[Dict[str, Any]],
    attachment_type_keys: List[str],
    rng: np.random.Generator,
) -> List[Dict[str, Any]]:
    """Order generic motifs from the scaffold-facing end outward.

    ``fragment_patterns`` deliberately contain no reference graph.  Their
    historical order is nevertheless not guaranteed to start at the fixed
    scaffold, which can put a distal aromatic ring at the first exit point.
    Use only the unordered atom classes observed at the exit to choose a
    chemically attachable first motif.  Ties remain randomized so this is a
    prior, not a copied target-side atom order.
    """
    if len(fragments) <= 1:
        return list(fragments)

    preferred = {str(key) for key in attachment_type_keys}
    # Once the head motif is selected, keep plausible linkers before rings and
    # terminal substituents.  The old priority put every one-atom F/Cl/N
    # fragment before a carbonyl, which made the first few generic points
    # chemically disconnected even though the element multiset was correct.
    kind_priority = {
        "carbonyl": 0,
        "chain": 1,
        "aliphatic_ring": 2,
        "aromatic_ring": 3,
        "fused_aromatic": 4,
        "fused_ring": 5,
        "hetero_atom": 9,
    }

    def head_kind_rank(kind: str) -> int:
        if "N|0" in preferred:
            return {
                "hetero_atom": 0,
                "carbonyl": 1,
                "chain": 2,
            }.get(kind, 8)
        if "C|0" in preferred:
            return {
                "carbonyl": 0,
                "chain": 1,
                "aliphatic_ring": 2,
                "aromatic_ring": 3,
                "fused_aromatic": 4,
                "fused_ring": 4,
                "hetero_atom": 8,
            }.get(kind, 8)
        if "C|1" in preferred:
            return {
                "aromatic_ring": 0,
                "fused_aromatic": 0,
                "fused_ring": 0,
                "aliphatic_ring": 1,
                "chain": 2,
                "carbonyl": 3,
                "hetero_atom": 8,
            }.get(kind, 8)
        return 0

    # Pick one generic motif as the scaffold-facing head.  If two rings share
    # the same requested class, the choice is randomized; all other motifs
    # are then ordered by generic linker/ring/substituent priority.
    head_candidates = [
        (head_kind_rank(str(fragment.get("kind", "chain"))), index)
        for index, fragment in enumerate(fragments)
        if any(_fragment_has_type(fragment, key) for key in preferred)
    ]
    head_index = None
    if head_candidates:
        best_rank = min(rank for rank, _index in head_candidates)
        tied = [index for rank, index in head_candidates if rank == best_rank]
        head_index = int(rng.choice(tied))

    scored: list[tuple[tuple[float, ...], int, Dict[str, Any]]] = []
    for index, fragment in enumerate(fragments):
        keys = _fragment_atom_keys(fragment)
        kind = str(fragment.get("kind", "chain"))
        head_rank = 0 if index == head_index else 1
        score = (
            float(head_rank),
            float(kind_priority.get(kind, 99)),
            float(len(keys)),
        )
        scored.append((score, index, fragment))

    # Randomize equal-priority motifs before the stable sort.  This preserves
    # diversity across seeds while making the attachment rule deterministic.
    rng.shuffle(scored)
    scored.sort(key=lambda item: item[0])
    return [item[2] for item in scored]


def _shuffle_bucket(
    fragments: List[Dict[str, Any]],
    rng: np.random.Generator,
) -> List[Dict[str, Any]]:
    bucket = list(fragments)
    rng.shuffle(bucket)
    return bucket


def _order_fragments_aliphatic_aromatic(
    fragments: List[Dict[str, Any]],
    rng: np.random.Generator,
) -> List[Dict[str, Any]]:
    """Order remaining motifs as aliphatic ring, linker, distal aryl.

    Use when the proximal aryl is already locked in the scaffold.  The
    three-ring ``aryl_aliphatic_aryl`` prior would then put a spurious
    phenyl at the lock exit and bury the piperidine.
    """
    aliphatics: List[Dict[str, Any]] = []
    aryls: List[Dict[str, Any]] = []
    linkers: List[Dict[str, Any]] = []
    others: List[Dict[str, Any]] = []
    for fragment in fragments:
        kind = str(fragment.get("kind", "chain"))
        if kind in _ALIPHATIC_RING_KINDS:
            aliphatics.append(fragment)
        elif kind in _ARYL_KINDS:
            aryls.append(fragment)
        elif kind in _LINKER_KINDS:
            linkers.append(fragment)
        else:
            others.append(fragment)
    return (
        _shuffle_bucket(aliphatics, rng)
        + _shuffle_bucket(linkers, rng)
        + _shuffle_bucket(aryls, rng)
        + _shuffle_bucket(others, rng)
    )


def _order_fragments_aryl_aliphatic_aryl(
    fragments: List[Dict[str, Any]],
    rng: np.random.Generator,
) -> List[Dict[str, Any]]:
    """Order generic motifs as proximal aryl, aliphatic ring, distal aryl.

    Linkers sit between rings.  Ties within a kind are shuffled so this is a
    topology prior, not a copied target-side atom order.
    """
    aryls: List[Dict[str, Any]] = []
    aliphatics: List[Dict[str, Any]] = []
    linkers: List[Dict[str, Any]] = []
    others: List[Dict[str, Any]] = []
    for fragment in fragments:
        kind = str(fragment.get("kind", "chain"))
        if kind in _ARYL_KINDS:
            aryls.append(fragment)
        elif kind in _ALIPHATIC_RING_KINDS:
            aliphatics.append(fragment)
        elif kind in _LINKER_KINDS:
            linkers.append(fragment)
        else:
            others.append(fragment)
    aryls = _shuffle_bucket(aryls, rng)
    aliphatics = _shuffle_bucket(aliphatics, rng)
    linkers = _shuffle_bucket(linkers, rng)
    others = _shuffle_bucket(others, rng)
    proximal = aryls[:1]
    distal = aryls[1:]
    split = (len(linkers) + 1) // 2
    return proximal + linkers[:split] + aliphatics + linkers[split:] + distal + others


def _order_fragments_amide_fused(
    fragments: List[Dict[str, Any]],
    rng: np.random.Generator,
) -> List[Dict[str, Any]]:
    """Put amide/chain linkers at the exit, then a fused or aromatic ring."""
    fused: List[Dict[str, Any]] = []
    linkers: List[Dict[str, Any]] = []
    others: List[Dict[str, Any]] = []
    for fragment in fragments:
        kind = str(fragment.get("kind", "chain"))
        if kind in _ARYL_KINDS:
            fused.append(fragment)
        elif kind in _LINKER_KINDS:
            linkers.append(fragment)
        else:
            others.append(fragment)
    return (
        _shuffle_bucket(linkers, rng)
        + _shuffle_bucket(fused, rng)
        + _shuffle_bucket(others, rng)
    )


def _order_fragments_for_mode(
    fragments: List[Dict[str, Any]],
    sites_cfg: Dict[str, Any],
    rng: np.random.Generator,
) -> List[Dict[str, Any]]:
    """Apply a topology prior, keep profile order, or shuffle."""
    order_mode = str(sites_cfg.get("fragment_order_mode", "shuffle")).strip().lower()
    ordered = list(fragments)
    if order_mode in ("profile", "as_profile", "keep"):
        return ordered
    if order_mode in ("aryl_aliphatic_aryl", "aromatic_aliphatic_aromatic"):
        return _order_fragments_aryl_aliphatic_aryl(ordered, rng)
    if order_mode in (
        "aliphatic_aromatic",
        "aliphatic_then_aromatic",
        "aliphatic_aryl",
    ):
        return _order_fragments_aliphatic_aromatic(ordered, rng)
    if order_mode in ("amide_fused", "linker_then_fused"):
        return _order_fragments_amide_fused(ordered, rng)
    rng.shuffle(ordered)
    return ordered


def _put_preferred_types_first(
    atom_keys: List[str],
    preferred_types: List[str],
    rng: np.random.Generator,
) -> List[str]:
    """Move one or more requested classes to the generic head of a motif."""
    if not atom_keys or not preferred_types:
        return atom_keys
    remaining = list(atom_keys)
    head: List[str] = []
    for preferred in preferred_types:
        try:
            index = remaining.index(str(preferred))
        except ValueError:
            continue
        head.append(remaining.pop(index))
    rng.shuffle(head)
    rng.shuffle(remaining)
    return head + remaining
