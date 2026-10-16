"""Fragment Geometry responsibilities extracted from scaffold_sites.py."""

from __future__ import annotations

from diffdynamic.chemistry.sites.state import Optional

from diffdynamic.chemistry.sites.state import (
    Any,
    Dict,
    List,
    load_scaffold_profile,
    np,
)


def _regular_polygon(
    n_atoms: int,
    radius: float,
    tangent: np.ndarray,
    binormal: np.ndarray,
    angle: float,
) -> np.ndarray:
    values = []
    for index in range(int(n_atoms)):
        theta = angle + 2.0 * np.pi * index / max(int(n_atoms), 1)
        values.append(radius * (np.cos(theta) * tangent + np.sin(theta) * binormal))
    return np.asarray(values, dtype=np.float64)


def _fragment_ring_atom_count(
    fragment: Dict[str, Any],
    total: int,
    kind: str,
) -> int:
    ring_sizes = [
        int(value) for value in fragment.get("ring_sizes", []) if int(value) >= 3
    ]
    if kind in ("aromatic_ring", "aliphatic_ring"):
        size = ring_sizes[0] if ring_sizes else total
        return max(3, min(int(total), int(size)))
    if kind in ("fused_aromatic", "fused_ring"):
        ring_sizes = ring_sizes or [6, 5]
        # Two fused rings share two generic positions.  This is a geometric
        # motif summary, not a copied bond graph.
        estimate = sum(ring_sizes) - 2 * max(len(ring_sizes) - 1, 1)
        return max(3, min(int(total), int(estimate)))
    return 0


def _split_fragment_ring_keys(
    atom_keys: List[str],
    ring_count: int,
    rng: np.random.Generator,
) -> tuple[List[str], List[str]]:
    """Choose generic ring versus substituent atom classes.

    The profile only gives element/aromatic counts.  This ordering keeps
    obvious substituents such as O/F/Cl out of a generic carbon ring when the
    marginal counts make that possible; it does not encode a reference graph.
    """
    aromatic = [key for key in atom_keys if key.endswith("|1")]
    non_aromatic = [key for key in atom_keys if not key.endswith("|1")]
    priority = {"C": 0, "N": 1, "S": 2, "P": 3, "O": 4, "F": 5, "Cl": 6}
    non_aromatic.sort(key=lambda key: priority.get(str(key).rsplit("|", 1)[0], 99))
    ordered = aromatic + non_aromatic
    ring_keys = ordered[: int(ring_count)]
    extra_keys = ordered[int(ring_count) :]
    rng.shuffle(ring_keys)
    rng.shuffle(extra_keys)
    return ring_keys, extra_keys


def _generic_fragment_geometry(
    fragment: Dict[str, Any],
    center: np.ndarray,
    direction: np.ndarray,
    tangent: np.ndarray,
    binormal: np.ndarray,
    rng: np.random.Generator,
    sites_cfg: Dict[str, Any],
    preferred_anchor_types: Optional[List[str]] = None,
) -> tuple[list[np.ndarray], List[str]]:
    """Create randomized motif geometry and aligned type hints.

    These are generic motif clouds, not reference graphs: no bond list,
    reference atom index, or target-side coordinate enters this function.
    The resulting coordinates are still forward-noised before denoising.
    """
    from diffdynamic.chemistry.sites.fragments import (
        _fragment_atom_keys,
        _put_preferred_types_first,
    )

    kind = str(fragment.get("kind", "chain"))
    atom_keys = _fragment_atom_keys(fragment)
    if not atom_keys:
        return [], []
    jitter = float(sites_cfg.get("fragment_geometry_sigma", 0.16))
    ring_radius = float(sites_cfg.get("fragment_ring_radius", 1.38))
    aliphatic_ring_radius = float(
        sites_cfg.get("fragment_aliphatic_ring_radius", ring_radius * 0.92)
    )
    angle = float(rng.uniform(0.0, 2.0 * np.pi))
    rot_t = np.cos(angle) * tangent + np.sin(angle) * binormal
    rot_b = -np.sin(angle) * tangent + np.cos(angle) * binormal
    total = len(atom_keys)
    attachment_aware = bool(sites_cfg.get("fragment_attachment_aware", False))
    preferred_anchor_types = [str(key) for key in (preferred_anchor_types or [])]

    if kind in ("aromatic_ring", "aliphatic_ring"):
        radius = ring_radius if kind == "aromatic_ring" else aliphatic_ring_radius
        ring_count = _fragment_ring_atom_count(fragment, total, kind)
        ring_keys, extra_keys = _split_fragment_ring_keys(atom_keys, ring_count, rng)
        local_ring = _regular_polygon(ring_count, radius, direction, rot_t, np.pi)
        local_points = [point for point in local_ring]
        for index, point in enumerate(extra_keys):
            local_points.append(
                local_ring[index % ring_count]
                + (1.35 + 0.20 * (index // ring_count)) * direction
                + 0.25 * ((-1) ** index) * rot_t
            )
        local = np.asarray(local_points, dtype=np.float64)
        atom_keys = ring_keys + extra_keys
        if attachment_aware and preferred_anchor_types:
            # Assign a preferred head class to the ring atom closest to the
            # scaffold-facing direction.  The ring geometry itself remains a
            # generic regular polygon.
            projections = local @ direction
            nearest = int(np.argmin(projections))
            for preferred in preferred_anchor_types:
                try:
                    source = atom_keys.index(preferred)
                except ValueError:
                    continue
                atom_keys[nearest], atom_keys[source] = (
                    atom_keys[source],
                    atom_keys[nearest],
                )
                break
    elif kind in ("fused_aromatic", "fused_ring") and total >= 8:
        # Generic fused rings with shared positions.  The exact reference
        # fusion topology is intentionally not retained.
        ring_sizes = [
            int(value) for value in fragment.get("ring_sizes", []) if int(value) >= 3
        ] or [6, 5]
        first_size = ring_sizes[0]
        second_size = ring_sizes[1] if len(ring_sizes) > 1 else 5
        first = _regular_polygon(first_size, ring_radius, direction, rot_t, np.pi)
        second_center = 1.55 * rot_t
        second = second_center + _regular_polygon(
            second_size, ring_radius * 0.96, direction, rot_t, np.pi + 0.35
        )
        ring_count = _fragment_ring_atom_count(fragment, total, kind)
        ring_keys, extra_keys = _split_fragment_ring_keys(atom_keys, ring_count, rng)
        local_points = [first[0], first[1]]
        local_points.extend(first[2:])
        local_points.extend(second[2:])
        local_points = local_points[:ring_count]
        while len(local_points) < ring_count:
            extra_index = len(local_points)
            local_points.append(
                second_center + (extra_index + 1.0) * 0.75 * rot_t + 0.35 * direction
            )
        for index, point in enumerate(extra_keys):
            local_points.append(
                local_points[index % ring_count]
                + (1.30 + 0.20 * (index // ring_count)) * direction
                + 0.25 * ((-1) ** index) * rot_t
            )
        local = np.asarray(local_points[:total], dtype=np.float64)
        atom_keys = ring_keys + extra_keys
        if attachment_aware and preferred_anchor_types:
            projections = local @ direction
            nearest = int(np.argmin(projections))
            for preferred in preferred_anchor_types:
                try:
                    source = atom_keys.index(preferred)
                except ValueError:
                    continue
                atom_keys[nearest], atom_keys[source] = (
                    atom_keys[source],
                    atom_keys[nearest],
                )
                break
    elif kind == "carbonyl" and total >= 2:
        local = np.zeros((total, 3), dtype=np.float64)
        if attachment_aware:
            # Keep the carbonyl carbon on the scaffold-facing end and put the
            # oxygen on a short generic branch.  No target-side bond is used.
            local[0] = -0.60 * direction
            local[1] = -0.60 * direction + 1.22 * rot_t
        else:
            carbonyl_axis = np.cos(angle) * rot_t + np.sin(angle) * rot_b
            local[0] = -0.615 * carbonyl_axis
            local[1] = 0.615 * carbonyl_axis
        for index in range(2, total):
            local[index] = (index - 1) * 1.25 * direction + 0.35 * (
                (-1) ** index
            ) * rot_t
    elif kind == "hetero_atom" or total == 1:
        local = np.zeros((total, 3), dtype=np.float64)
        for index in range(1, total):
            local[index] = 0.45 * index * rot_t
    else:
        local = np.zeros((total, 3), dtype=np.float64)
        for index in range(total):
            local[index] = (index - 0.5 * (total - 1)) * 1.48 * direction + 0.42 * (
                (-1) ** index
            ) * rot_t

    positions = [center + offset + rng.normal(0.0, jitter, size=3) for offset in local]
    if kind == "chain":
        if attachment_aware and preferred_anchor_types:
            atom_keys = _put_preferred_types_first(
                atom_keys, preferred_anchor_types, rng
            )
        else:
            rng.shuffle(atom_keys)
    return positions, atom_keys


def _align_fragment_to_anchor(
    positions: List[np.ndarray],
    atom_types: List[str],
    anchor: np.ndarray,
    direction: np.ndarray,
    target_distance: float,
) -> tuple[List[np.ndarray], List[str], float]:
    """Put the nearest generic fragment atom at a scaffold-facing distance."""
    if not positions:
        return [], [], 0.0
    array = np.asarray(positions, dtype=np.float64)
    projections = np.asarray(
        [float(np.dot(point - anchor, direction)) for point in array],
        dtype=np.float64,
    )
    shift = float(target_distance) - float(projections.min())
    array = array + shift * direction
    order = np.argsort(projections, kind="stable")
    aligned_positions = [array[int(index)] for index in order]
    aligned_types = [atom_types[int(index)] for index in order]
    return aligned_positions, aligned_types, float(target_distance)


def _strict_fragment_gaussian_site_positions(
    active_sites: List[Dict[str, Any]],
    counts: List[int],
    rng: np.random.Generator,
    sites_cfg: Dict[str, Any],
) -> tuple[List[np.ndarray], List[str], List[Dict[str, Any]]]:
    """Place generic fragment clouds around scaffold exit frames."""
    from diffdynamic.chemistry.sites.fragments import (
        _order_fragments_for_attachment,
        _order_fragments_for_mode,
        _site_frame,
    )

    profile_path = sites_cfg.get("fragment_prior_profile") or sites_cfg.get(
        "reference_exit_profile"
    )
    if not profile_path:
        raise ValueError("fragment_gaussian requires fragment_prior_profile")
    profile = load_scaffold_profile(profile_path)
    n_extra = int(sum(int(value) for value in counts))
    allocation = {
        str(site.get("profile_slot")): int(count)
        for site, count in zip(active_sites, counts)
        if int(count) > 0
    }
    candidates = []
    expected_site_type_counts = {
        str(slot): {str(key): int(value) for key, value in counts.items()}
        for slot, counts in (
            sites_cfg.get("reference_exact_site_type_counts") or {}
        ).items()
    }
    for pattern in profile.get("fragment_patterns", []):
        if int(pattern.get("n_extra", -1)) != n_extra:
            continue
        pattern_counts = {}
        for slot, fragments in (pattern.get("site_fragments") or {}).items():
            pattern_counts[str(slot)] = sum(
                sum(
                    int(value) for value in (fragment.get("atom_counts") or {}).values()
                )
                for fragment in fragments
            )
        if pattern_counts != allocation:
            continue
        if expected_site_type_counts:
            pattern_types: Dict[str, Dict[str, int]] = {}
            for slot, fragments in (pattern.get("site_fragments") or {}).items():
                merged: Dict[str, int] = {}
                for fragment in fragments:
                    for key, value in (fragment.get("atom_counts") or {}).items():
                        merged[str(key)] = merged.get(str(key), 0) + int(value)
                pattern_types[str(slot)] = merged
            if pattern_types != expected_site_type_counts:
                continue
        candidates.append(pattern)
    if not candidates:
        raise ValueError(
            f"no fragment pattern matches n_extra={n_extra} allocation={allocation}"
        )
    weights = np.asarray(
        [max(float(pattern.get("weight", 0.0)), 0.0) for pattern in candidates],
        dtype=np.float64,
    )
    selected = candidates[
        int(rng.choice(len(candidates), p=weights / max(float(weights.sum()), 1e-12)))
    ]

    positions: List[np.ndarray] = []
    type_hints: List[str] = []
    layout: List[Dict[str, Any]] = []
    fragment_spacing = float(sites_cfg.get("fragment_center_spacing", 1.52))
    center_offset = float(sites_cfg.get("strict_fragment_center_offset", 1.55))
    attachment_aware = bool(sites_cfg.get("fragment_attachment_aware", False))
    scale_to_record_centroid = bool(
        sites_cfg.get("fragment_scale_to_record_centroid", False)
    )
    lateral_sigma = float(sites_cfg.get("fragment_center_lateral_sigma", 0.45))
    shared_lateral_sigma = float(
        sites_cfg.get("fragment_center_shared_lateral_sigma", 0.0)
    )
    attachment_distance = float(sites_cfg.get("fragment_attachment_distance", 1.55))
    attachment_spacing = float(
        sites_cfg.get("fragment_attachment_spacing", fragment_spacing)
    )
    spacing_by_count = sites_cfg.get("fragment_attachment_spacing_by_n") or {}
    if spacing_by_count:
        count_key = str(n_extra)
        if count_key in spacing_by_count:
            attachment_spacing = float(spacing_by_count[count_key])
        elif n_extra in spacing_by_count:
            attachment_spacing = float(spacing_by_count[n_extra])
    for site, count in zip(active_sites, counts):
        if int(count) <= 0:
            continue
        slot = str(site.get("profile_slot"))
        fragments = [
            dict(fragment)
            for fragment in ((selected.get("site_fragments") or {}).get(slot) or [])
        ]
        fragments = _order_fragments_for_mode(fragments, sites_cfg, rng)
        attachment_counts = {
            str(key): int(value)
            for key, value in (
                (
                    sites_cfg.get("reference_exact_site_attachment_type_counts") or {}
                ).get(slot)
                or {}
            ).items()
            if int(value) > 0
        }
        attachment_type_keys = [
            key
            for key, value in sorted(attachment_counts.items())
            for _ in range(int(value))
        ]
        if attachment_aware:
            fragments = _order_fragments_for_attachment(
                fragments, attachment_type_keys, rng
            )
        anchor, direction, tangent, binormal = _site_frame(
            site,
            direction_mode=str(sites_cfg.get("fragment_axis_mode", "exit")),
        )
        site_center = anchor + center_offset * direction
        shared_lateral = np.zeros(2, dtype=np.float64)
        if shared_lateral_sigma > 0.0 and len(fragments) > 1:
            shared_lateral = rng.normal(0.0, shared_lateral_sigma, size=2)
        target_distance = None
        if scale_to_record_centroid:
            raw_offset = site.get("pocket_center_offset")
            if raw_offset is not None:
                candidate_distance = float(
                    np.linalg.norm(np.asarray(raw_offset, dtype=np.float64))
                )
                if np.isfinite(candidate_distance) and candidate_distance > 1e-6:
                    target_distance = candidate_distance
        for fragment_index, fragment in enumerate(fragments):
            lateral = rng.normal(0.0, lateral_sigma, size=2)
            progress = fragment_index / max(len(fragments) - 1, 1)
            lateral += progress * shared_lateral
            if target_distance is not None:
                projected_target = attachment_distance + progress * max(
                    target_distance - attachment_distance, 0.0
                )
                center = (
                    anchor
                    + projected_target * direction
                    + lateral[0] * tangent
                    + lateral[1] * binormal
                )
            else:
                step = (
                    attachment_spacing if attachment_spacing > 0.0 else fragment_spacing
                )
                projected_target = attachment_distance + fragment_index * step
                center = (
                    site_center
                    + fragment_index * step * direction
                    + lateral[0] * tangent
                    + lateral[1] * binormal
                )
            preferred_for_fragment = (
                list(attachment_type_keys)
                if attachment_aware and fragment_index == 0
                else []
            )
            fragment_positions, fragment_types = _generic_fragment_geometry(
                fragment,
                center,
                direction,
                tangent,
                binormal,
                rng,
                sites_cfg,
                preferred_anchor_types=preferred_for_fragment,
            )
            fragment_positions, fragment_types, projected_distance = (
                _align_fragment_to_anchor(
                    fragment_positions,
                    fragment_types,
                    anchor,
                    direction,
                    projected_target,
                )
            )
            positions.extend(fragment_positions)
            type_hints.extend(fragment_types)
            layout.append(
                {
                    "profile_slot": slot,
                    "kind": str(fragment.get("kind", "chain")),
                    "ring_sizes": [int(v) for v in fragment.get("ring_sizes", [])],
                    "atom_count": len(fragment_types),
                    "atom_types": list(fragment_types),
                    "anchor_projection_distance": projected_distance,
                }
            )

    if len(positions) != n_extra or len(type_hints) != n_extra:
        raise ValueError(
            f"fragment placement produced {len(positions)} atoms, expected {n_extra}"
        )
    return positions, type_hints, layout
