"""Pocket Geometry responsibilities extracted from scaffold_sites.py."""

from __future__ import annotations

from diffdynamic.chemistry.sites.state import Any, Dict, List, Optional

from diffdynamic.chemistry.sites.state import (
    np,
)


def _anchor_radial_site_position(
    site: Dict[str, Any],
    jitter_std: float,
    rng: np.random.Generator,
    atom_idx: int = 0,
    site_count: int = 1,
    radial_step: float = 1.3,
    radial_bulk: float = 0.35,
    radial_bulk_start: int = 4,
) -> np.ndarray:
    """Place atoms from the scaffold anchor along the exit direction."""
    from diffdynamic.chemistry.sites.gaussian import _attachment_direction

    anchor = np.asarray(site["anchor_pos"], dtype=np.float64)
    r = float(radial_step) * (float(atom_idx) + 1.0) + float(radial_bulk) * max(
        0.0,
        float(atom_idx) + 1.0 - float(radial_bulk_start),
    )
    direction = _attachment_direction(site)
    if direction is not None and r > 0.0:
        position = anchor + direction * r
    else:
        position = anchor + np.array([1.5, 0.0, 0.0], dtype=np.float64)
    return position + rng.normal(0.0, float(jitter_std), size=3)


def _original_site_position(
    site: Dict[str, Any],
    jitter_std: float,
    rng: np.random.Generator,
    atom_idx: int = 0,
    site_count: int = 1,
    radial_step: float = 1.3,
    min_dist: float = 2.0,
    max_dist: float = 4.0,
) -> np.ndarray:
    """Place extra atom using the original removed sidechain atom's 3D position.

    Falls back to radial extrapolation when original positions are exhausted
    or unavailable.
    """
    from diffdynamic.chemistry.sites.gaussian import _attachment_direction

    anchor = np.array(site["anchor_pos"], dtype=np.float64)
    orig_positions = site.get("removed_atom_positions")

    if orig_positions and atom_idx < len(orig_positions):
        pos = np.array(orig_positions[atom_idx], dtype=np.float64)
        dist = float(np.linalg.norm(pos - anchor))
        if dist < 1e-6:
            direction = np.array([1.0, 0.0, 0.0])
        else:
            direction = (pos - anchor) / dist
        if dist < min_dist:
            pos = anchor + direction * min_dist
        elif dist > max_dist:
            pos = anchor + direction * max_dist
        return pos + rng.normal(0.0, float(jitter_std), size=3)

    if orig_positions:
        last = np.array(orig_positions[-1], dtype=np.float64)
        diff = last - anchor
        norm = float(np.linalg.norm(diff))
        direction = diff / norm if norm > 1e-6 else np.array([1.0, 0.0, 0.0])
        extra = atom_idx - len(orig_positions) + 1
        pos = last + direction * float(radial_step) * float(extra)
    else:
        direction = _attachment_direction(site)
        if direction is not None:
            pos = anchor + direction * float(radial_step) * (float(atom_idx) + 1.0)
        else:
            pos = anchor + np.array([1.5, 0.0, 0.0])
    return pos + rng.normal(0.0, float(jitter_std), size=3)


def _protein_clearance(
    position: np.ndarray,
    protein_positions: Optional[np.ndarray],
) -> float:
    if protein_positions is None or protein_positions.size == 0:
        return float("inf")
    distances = np.linalg.norm(protein_positions - position, axis=1)
    return float(distances.min())


def _random_unit_vector(rng: np.random.Generator) -> np.ndarray:
    direction = rng.normal(size=3)
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-8:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    return direction / norm


def _grow_pocket_position(
    anchor: np.ndarray,
    placed: List[np.ndarray],
    protein_positions: Optional[np.ndarray],
    rng: np.random.Generator,
    sites_cfg: dict,
) -> np.ndarray:
    from diffdynamic.chemistry.sites.gaussian import _attachment_direction

    min_protein_dist = float(sites_cfg.get("pocket_min_protein_dist", 1.5))
    min_point_dist = float(sites_cfg.get("pocket_min_point_dist", 0.9))
    max_anchor_dist = float(sites_cfg.get("pocket_max_anchor_dist", 10.0))
    candidates = int(sites_cfg.get("pocket_growth_candidates", 128))
    bond_min = float(sites_cfg.get("pocket_growth_step_min", 1.25))
    bond_max = float(sites_cfg.get("pocket_growth_step_max", 1.55))
    parents = placed[-8:] if placed else [anchor]
    best = None
    best_score = -float("inf")
    for _ in range(max(candidates, 1)):
        parent = parents[int(rng.integers(0, len(parents)))]
        candidate = parent + _random_unit_vector(rng) * float(
            rng.uniform(bond_min, bond_max)
        )
        anchor_dist = float(np.linalg.norm(candidate - anchor))
        if anchor_dist > max_anchor_dist:
            continue
        clearance = _protein_clearance(candidate, protein_positions)
        if clearance < min_protein_dist:
            continue
        if placed:
            point_distances = np.linalg.norm(
                np.asarray(placed, dtype=np.float64) - candidate, axis=1
            )
            if float(point_distances.min()) < min_point_dist:
                continue
        score = min(clearance, 4.0) - 0.03 * anchor_dist
        if score > best_score:
            best = candidate
            best_score = score
    if best is not None:
        return best
    direction = _attachment_direction(
        {
            "anchor_pos": anchor,
            "centroid_pos": placed[-1] if placed else anchor + [1.0, 0.0, 0.0],
        }
    )
    direction = direction if direction is not None else _random_unit_vector(rng)
    parent = placed[-1] if placed else anchor
    return parent + direction * float(rng.uniform(bond_min, bond_max))


def _relax_positions_from_protein(
    positions: np.ndarray,
    anchor: np.ndarray,
    protein_positions: Optional[np.ndarray],
    rng: np.random.Generator,
    sites_cfg: dict,
) -> np.ndarray:
    if positions.size == 0 or protein_positions is None or protein_positions.size == 0:
        return positions
    relaxed = np.asarray(positions, dtype=np.float64).copy()
    min_protein_dist = float(sites_cfg.get("pocket_min_protein_dist", 1.5))
    min_point_dist = float(sites_cfg.get("pocket_min_point_dist", 0.9))
    max_anchor_dist = float(sites_cfg.get("pocket_max_anchor_dist", 10.0))
    iterations = int(sites_cfg.get("pocket_relax_iterations", 120))
    for _ in range(max(iterations, 1)):
        moved = False
        for index in range(len(relaxed)):
            deltas = relaxed[index] - protein_positions
            distances = np.linalg.norm(deltas, axis=1)
            nearest = int(np.argmin(distances))
            distance = float(distances[nearest])
            if distance < min_protein_dist:
                direction = deltas[nearest]
                norm = float(np.linalg.norm(direction))
                direction = (
                    direction / norm if norm > 1e-8 else _random_unit_vector(rng)
                )
                relaxed[index] += direction * (min_protein_dist - distance + 0.03)
                moved = True
        for left in range(len(relaxed)):
            for right in range(left + 1, len(relaxed)):
                delta = relaxed[left] - relaxed[right]
                distance = float(np.linalg.norm(delta))
                if distance >= min_point_dist:
                    continue
                direction = (
                    delta / distance if distance > 1e-8 else _random_unit_vector(rng)
                )
                shift = 0.5 * (min_point_dist - distance + 0.02) * direction
                relaxed[left] += shift
                relaxed[right] -= shift
                moved = True
        for index in range(len(relaxed)):
            delta = relaxed[index] - anchor
            distance = float(np.linalg.norm(delta))
            if distance > max_anchor_dist:
                relaxed[index] = anchor + delta / distance * max_anchor_dist
                moved = True
        if not moved:
            break
    return relaxed


def _pocket_aware_site_positions(
    site: Dict[str, Any],
    count: int,
    protein_positions: Optional[np.ndarray],
    jitter_std: float,
    rng: np.random.Generator,
    sites_cfg: dict,
) -> List[np.ndarray]:
    anchor = np.asarray(site["anchor_pos"], dtype=np.float64)
    local_protein = protein_positions
    if protein_positions is not None and protein_positions.size > 0:
        local_radius = float(sites_cfg.get("pocket_local_radius", 13.0))
        local_mask = np.linalg.norm(protein_positions - anchor, axis=1) <= local_radius
        if local_mask.any():
            local_protein = protein_positions[local_mask]
    template = [
        np.asarray(position, dtype=np.float64)
        for position in (site.get("removed_atom_positions") or [])[:count]
    ]
    placed = [
        position + rng.normal(0.0, float(jitter_std), size=3) for position in template
    ]
    while len(placed) < count:
        placed.append(
            _grow_pocket_position(anchor, placed, local_protein, rng, sites_cfg)
        )
    array = _relax_positions_from_protein(
        np.asarray(placed), anchor, local_protein, rng, sites_cfg
    )
    return [array[index] for index in range(len(array))]


def _is_gaussian_jitter(jitter_mode: str) -> bool:
    mode = str(jitter_mode or "gaussian").lower()
    return mode in (
        "gaussian",
        "isotropic",
        "clustered",
        "denovo",
        "random",
        "pocket_center_gaussian",
        "interface_corridor_gaussian",
    )
