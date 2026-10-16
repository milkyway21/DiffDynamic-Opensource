"""Gaussian responsibilities extracted from scaffold_sites.py."""

from __future__ import annotations

from diffdynamic.chemistry.sites.state import Any, Dict

from diffdynamic.chemistry.sites.state import (
    List,
    Optional,
    np,
)


def _attachment_direction(site: Dict[str, Any]) -> Optional[np.ndarray]:
    """返回 normalize(centroid - anchor)；缺锚点或退化时返回 None。"""
    anchor = site.get("anchor_pos")
    centroid = site.get("centroid_pos")
    if anchor is None or centroid is None:
        return None
    anchor = np.asarray(anchor, dtype=np.float64).reshape(3)
    centroid = np.asarray(centroid, dtype=np.float64).reshape(3)
    direction = centroid - anchor
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-6:
        return None
    return direction / norm


def _directional_site_position(
    site: Dict[str, Any],
    atom_idx: int,
    jitter_std: float,
    rng: np.random.Generator,
    directional_step: float = 1.0,
    min_base_dist: float = 1.5,
) -> np.ndarray:
    """沿 anchor→centroid 方向外推放置（旧方案，易呈离散珠串）。"""
    anchor = site.get("anchor_pos")
    centroid = site.get("centroid_pos")
    if anchor is None or centroid is None:
        centroid = np.array(site["centroid_pos"], dtype=np.float64)
        return centroid + rng.normal(0.0, jitter_std, size=3)

    anchor = np.array(anchor, dtype=np.float64)
    centroid = np.array(centroid, dtype=np.float64)
    direction = centroid - anchor
    norm = float(np.linalg.norm(direction))
    if norm > 1e-6:
        direction = direction / norm
    else:
        direction = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    base_dist = max(norm, float(min_base_dist))
    dist = base_dist + float(atom_idx) * float(directional_step)
    pos = anchor + direction * dist
    noise = rng.normal(0.0, jitter_std * 0.3, size=3)
    return pos + noise


def _gaussian_site_position(
    site: Dict[str, Any],
    jitter_std: float,
    rng: np.random.Generator,
    atom_idx: int = 0,
    site_count: int = 1,
    radial_step: float = 0.25,
    radial_bulk: float = 0.35,
    radial_bulk_start: int = 4,
) -> np.ndarray:
    """Place extra atoms in a Gaussian cloud around the site centroid."""
    centroid = np.array(site["centroid_pos"], dtype=np.float64)
    r = float(radial_step) * float(atom_idx) + float(radial_bulk) * max(
        0.0,
        float(site_count) - float(radial_bulk_start),
    )
    direction = _attachment_direction(site)
    if direction is not None and r > 0.0:
        centroid = centroid + direction * r
    return centroid + rng.normal(0.0, float(jitter_std), size=3)


def _strict_anchor_gaussian_site_positions(
    site: Dict[str, Any],
    count: int,
    jitter_std: float,
    rng: np.random.Generator,
    sites_cfg: dict,
) -> List[np.ndarray]:
    """Sample an iid Gaussian cloud from a scaffold exit anchor.

    Strict scaffold mode deliberately uses only the scaffold anchor and a
    scaffold-derived exit frame.  It never reads ``removed_atom_positions``
    and never performs protein-clearance rejection or post-hoc relaxation, so
    the initial distribution remains the model's usual Gaussian cloud with a
    scaffold-local mean.
    """
    anchor = np.asarray(site["anchor_pos"], dtype=np.float64)
    raw_direction = site.get("exit_direction")
    if raw_direction is None:
        raw_direction = _attachment_direction(site)
    if raw_direction is None:
        raw_direction = [1.0, 0.0, 0.0]
    direction = np.asarray(raw_direction, dtype=np.float64)
    direction_norm = float(np.linalg.norm(direction))
    direction = direction / max(direction_norm, 1e-8)

    raw_tangent = site.get("tangent_direction")
    if raw_tangent is None:
        raw_tangent = [0.0, 1.0, 0.0]
    tangent = np.asarray(raw_tangent, dtype=np.float64)
    tangent -= np.dot(tangent, direction) * direction
    tangent_norm = float(np.linalg.norm(tangent))
    if tangent_norm <= 1e-8:
        basis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(float(np.dot(basis, direction))) > 0.9:
            basis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        tangent = basis - np.dot(basis, direction) * direction
        tangent_norm = float(np.linalg.norm(tangent))
    tangent /= max(tangent_norm, 1e-8)

    raw_binormal = site.get("binormal_direction")
    if raw_binormal is None:
        raw_binormal = np.cross(direction, tangent)
    binormal = np.asarray(raw_binormal, dtype=np.float64)
    binormal_norm = float(np.linalg.norm(binormal))
    binormal = binormal / max(binormal_norm, 1e-8)

    slot = str(site.get("profile_slot", ""))
    slot_offsets = sites_cfg.get("strict_gaussian_slot_offsets", {}) or {}
    slot_tangent_shifts = sites_cfg.get("strict_gaussian_slot_tangent_shifts", {}) or {}
    slot_binormal_shifts = (
        sites_cfg.get("strict_gaussian_slot_binormal_shifts", {}) or {}
    )
    offset = float(
        slot_offsets.get(slot, sites_cfg.get("strict_gaussian_center_offset", 1.5))
    )
    tangent_shift = float(
        slot_tangent_shifts.get(
            slot, sites_cfg.get("strict_gaussian_tangent_shift", 0.0)
        )
    )
    binormal_shift = float(
        slot_binormal_shifts.get(
            slot, sites_cfg.get("strict_gaussian_binormal_shift", 0.0)
        )
    )
    sigma = float(sites_cfg.get("strict_gaussian_sigma", jitter_std))
    if sigma <= 0.0:
        raise ValueError("strict_gaussian_sigma must be positive")

    center = (
        anchor
        + offset * direction
        + tangent_shift * tangent
        + binormal_shift * binormal
    )
    axial_sigma = sites_cfg.get("strict_gaussian_axial_sigma")
    lateral_sigma = sites_cfg.get("strict_gaussian_lateral_sigma")
    if axial_sigma is None or lateral_sigma is None:
        return [center + rng.normal(0.0, sigma, size=3) for _ in range(int(count))]

    axial_sigma = float(axial_sigma)
    lateral_sigma = float(lateral_sigma)
    if axial_sigma <= 0.0 or lateral_sigma <= 0.0:
        raise ValueError(
            "strict_gaussian_axial_sigma and "
            "strict_gaussian_lateral_sigma must be positive"
        )
    return [
        center
        + rng.normal(0.0, axial_sigma) * direction
        + rng.normal(0.0, lateral_sigma) * tangent
        + rng.normal(0.0, lateral_sigma) * binormal
        for _ in range(int(count))
    ]


def _pocket_center_gaussian_site_positions(
    site: Dict[str, Any],
    count: int,
    protein_positions: Optional[np.ndarray],
    rng: np.random.Generator,
    sites_cfg: dict,
) -> List[np.ndarray]:
    """Sample a discrete Gaussian cloud around a pocket-internal center.

    The site carries only an anchor-to-pocket-center vector derived from the
    reference pose.  No target-side atom coordinates, element labels, or bonds
    are used.  Protein clearance is enforced only for the initial cloud; the
    normal DiffDynamic coordinate reverse process remains fully free afterward.
    """
    from diffdynamic.chemistry.sites.pocket_geometry import (
        _grow_pocket_position,
        _protein_clearance,
    )

    anchor = np.asarray(site["anchor_pos"], dtype=np.float64)
    raw_offset = site.get("pocket_center_offset")
    if raw_offset is None:
        raise ValueError(
            "pocket_center_gaussian requires a pocket_center_offset on the "
            "reference exit site"
        )
    offset = np.asarray(raw_offset, dtype=np.float64)
    offset_norm = float(np.linalg.norm(offset))
    centered_custom_gaussian = (
        bool(sites_cfg.get("custom_region_strict"))
        and site.get("custom_region", {}).get("method") == "gaussian"
    )
    if offset_norm <= 1e-8 and not centered_custom_gaussian:
        raise ValueError(
            "pocket_center_gaussian requires a non-zero pocket center offset"
        )
    axis = offset / offset_norm if offset_norm > 1e-8 else np.array([1.0, 0.0, 0.0])
    center = anchor + offset

    # Use the scaffold exit direction only to orient the transverse Gaussian
    # frame.  It does not define the cloud mean.
    raw_tangent = site.get("exit_direction") or [0.0, 1.0, 0.0]
    tangent = np.asarray(raw_tangent, dtype=np.float64)
    tangent -= np.dot(tangent, axis) * axis
    tangent_norm = float(np.linalg.norm(tangent))
    if tangent_norm <= 1e-8:
        basis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(float(np.dot(basis, axis))) > 0.9:
            basis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        tangent = basis - np.dot(basis, axis) * axis
        tangent_norm = float(np.linalg.norm(tangent))
    tangent /= max(tangent_norm, 1e-8)
    binormal = np.cross(axis, tangent)
    binormal /= max(float(np.linalg.norm(binormal)), 1e-8)

    axial_sigma = float(sites_cfg.get("pocket_gaussian_axial_sigma", 1.0))
    lateral_sigma = float(sites_cfg.get("pocket_gaussian_lateral_sigma", 1.25))
    if axial_sigma <= 0.0 or lateral_sigma <= 0.0:
        raise ValueError("pocket Gaussian sigmas must be positive")

    local_protein = protein_positions
    if protein_positions is not None and protein_positions.size > 0:
        local_radius = float(sites_cfg.get("pocket_local_radius", 12.0))
        local_mask = np.linalg.norm(protein_positions - anchor, axis=1) <= (
            local_radius
        )
        if local_mask.any():
            local_protein = protein_positions[local_mask]

    min_protein_dist = float(sites_cfg.get("pocket_min_protein_dist", 1.8))
    min_point_dist = float(sites_cfg.get("pocket_min_point_dist", 0.8))
    max_anchor_dist = float(sites_cfg.get("pocket_max_anchor_dist", 8.0))
    attempts = int(sites_cfg.get("pocket_gaussian_attempts", 256))
    placed: List[np.ndarray] = []

    for _ in range(int(count)):
        accepted = None
        best = None
        best_score = -float("inf")
        for _attempt in range(max(attempts, 1)):
            candidate = (
                center
                + rng.normal(0.0, axial_sigma) * axis
                + rng.normal(0.0, lateral_sigma) * tangent
                + rng.normal(0.0, lateral_sigma) * binormal
            )
            anchor_distance = float(np.linalg.norm(candidate - anchor))
            clearance = _protein_clearance(candidate, local_protein)
            point_clearance = (
                float(np.linalg.norm(np.asarray(placed) - candidate, axis=1).min())
                if placed
                else float("inf")
            )
            score = min(clearance, 4.0) - 0.04 * anchor_distance
            if placed:
                score -= max(0.0, min_point_dist - point_clearance)
            if score > best_score:
                best = candidate
                best_score = score
            if (
                clearance >= min_protein_dist
                and anchor_distance <= max_anchor_dist
                and point_clearance >= min_point_dist
            ):
                accepted = candidate
                break

        if accepted is None:
            if sites_cfg.get("custom_region_strict"):
                raise ValueError("Custom region is too crowded for the requested cloud")
            # A crowded pocket can exhaust rejection attempts.  Grow from the
            # already accepted cloud as a last-resort valid-pocket proposal;
            # this path is not the normal distribution and is recorded in the
            # caller's geometry preflight if it becomes frequent.
            accepted = _grow_pocket_position(
                anchor, placed, local_protein, rng, sites_cfg
            )
            if (
                best is not None
                and _protein_clearance(accepted, local_protein) < min_protein_dist
            ):
                accepted = best
        placed.append(np.asarray(accepted, dtype=np.float64))
    return placed


def _interface_corridor_gaussian_site_positions(
    site: Dict[str, Any],
    count: int,
    protein_positions: Optional[np.ndarray],
    rng: np.random.Generator,
    sites_cfg: dict,
) -> List[np.ndarray]:
    """Place a scaffold-only cloud from the exit N into the interface pocket.

    Two connector points cover the bond-forming corridor immediately after the
    fixed terminal amine.  Small branches put all remaining points around the
    pocket center.  Large branches additionally fill a middle and distal zone.
    Only the scaffold anchor, an explicit pocket center, and protein clearance
    are used; target-side atom coordinates and graphs are never consumed.
    """
    from diffdynamic.chemistry.sites.pocket_geometry import _protein_clearance

    anchor = np.asarray(site["anchor_pos"], dtype=np.float64)
    raw_offset = site.get("pocket_center_offset")
    if raw_offset is None:
        raise ValueError(
            "interface_corridor_gaussian requires an explicit interface center"
        )
    offset = np.asarray(raw_offset, dtype=np.float64)
    center_distance = float(np.linalg.norm(offset))
    if center_distance <= 1e-8:
        raise ValueError("interface corridor center must differ from its anchor")
    # Far Glide centroids (IKZF2-40/50) must not dump extra atoms 4–9 Å
    # from the exit: reconstruction only recognizes covalent bonds, so the
    # pocket cloud is clamped onto a near-exit corridor.
    raw_max_center = sites_cfg.get("corridor_max_center_distance")
    if raw_max_center is not None:
        max_center = float(raw_max_center)
        if max_center > 1e-8 and center_distance > max_center:
            offset = offset * (max_center / center_distance)
            center_distance = max_center
    center_axis = offset / center_distance
    raw_exit = site.get("exit_direction")
    exit_axis = np.asarray(
        raw_exit if raw_exit is not None else center_axis,
        dtype=np.float64,
    )
    exit_axis /= max(float(np.linalg.norm(exit_axis)), 1e-8)
    axis_mode = str(sites_cfg.get("corridor_axis_mode", "pocket")).strip().lower()
    middle_exit_fraction = float(
        np.clip(
            sites_cfg.get("corridor_middle_exit_fraction", 0.65),
            0.0,
            1.0,
        )
    )

    def _normalise(vector: np.ndarray, fallback: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm > 1e-8 else fallback

    def _zone_axis(zone: str) -> np.ndarray:
        if axis_mode in ("exit", "exit_aligned"):
            return exit_axis
        if axis_mode in ("connector_exit", "exit_connectors"):
            return exit_axis if zone == "connector" else center_axis
        if axis_mode in ("exit_to_pocket", "exit_then_pocket"):
            if zone == "connector":
                return exit_axis
            if zone == "middle":
                blended = (
                    middle_exit_fraction * exit_axis
                    + (1.0 - middle_exit_fraction) * center_axis
                )
                return _normalise(blended, exit_axis)
        return center_axis

    def _axis_candidates(zone: str) -> list[np.ndarray]:
        primary = _zone_axis(zone)
        candidates = [primary]
        fallback_vectors = [center_axis, exit_axis]
        if axis_mode in ("exit_to_pocket", "exit_then_pocket"):
            fallback_vectors.extend(
                [
                    _normalise(exit_axis + center_axis, exit_axis),
                    _normalise(exit_axis - center_axis, exit_axis),
                ]
            )
        for candidate in fallback_vectors:
            if not any(
                np.linalg.norm(candidate - previous) <= 1e-8 for previous in candidates
            ):
                candidates.append(candidate)
        return candidates

    def _zone_frame(
        zone: str,
        axis_override: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        zone_axis = axis_override if axis_override is not None else _zone_axis(zone)
        basis = np.asarray(
            site.get("exit_direction") or [0.0, 1.0, 0.0],
            dtype=np.float64,
        )
        basis -= float(np.dot(basis, zone_axis)) * zone_axis
        if float(np.linalg.norm(basis)) <= 1e-8:
            basis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            if abs(float(np.dot(basis, zone_axis))) > 0.9:
                basis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            basis -= float(np.dot(basis, zone_axis)) * zone_axis
        tangent = basis / max(float(np.linalg.norm(basis)), 1e-8)
        binormal = np.cross(zone_axis, tangent)
        binormal /= max(float(np.linalg.norm(binormal)), 1e-8)
        return zone_axis, tangent, binormal

    connector_count = min(int(count), int(sites_cfg.get("corridor_connector_count", 2)))
    large_threshold = int(sites_cfg.get("corridor_large_count_threshold", 16))
    remaining = int(count) - connector_count
    if int(count) >= large_threshold:
        middle_count = min(
            remaining, int(sites_cfg.get("corridor_large_middle_count", 10))
        )
        pocket_count = 0
        distal_count = remaining - middle_count
    else:
        middle_count = 0
        pocket_count = remaining
        distal_count = 0

    connector_start = float(sites_cfg.get("corridor_connector_start", 1.45))
    connector_spacing = float(sites_cfg.get("corridor_connector_spacing", 0.90))
    middle_fraction = float(sites_cfg.get("corridor_middle_fraction", 0.58))
    distal_offset = float(sites_cfg.get("corridor_distal_offset", 1.25))
    max_anchor_dist = float(sites_cfg.get("pocket_max_anchor_dist", 8.0))
    min_protein_dist = float(sites_cfg.get("pocket_min_protein_dist", 1.8))
    min_point_dist = float(sites_cfg.get("pocket_min_point_dist", 0.8))
    attempts = max(int(sites_cfg.get("pocket_gaussian_attempts", 512)), 1)

    local_protein = protein_positions
    if protein_positions is not None and protein_positions.size > 0:
        local_radius = float(sites_cfg.get("pocket_local_radius", 12.0))
        local_mask = np.linalg.norm(protein_positions - anchor, axis=1) <= (
            local_radius + max_anchor_dist
        )
        if local_mask.any():
            local_protein = protein_positions[local_mask]

    zone_specs: list[tuple[str, float, float, float]] = []
    for connector_index in range(connector_count):
        zone_specs.append(
            (
                "connector",
                connector_start + connector_spacing * connector_index,
                float(sites_cfg.get("corridor_connector_axial_sigma", 0.12)),
                float(sites_cfg.get("corridor_connector_lateral_sigma", 0.24)),
            )
        )
    zone_specs.extend(
        [
            (
                "middle",
                min(center_distance * middle_fraction, max_anchor_dist - 0.5),
                float(sites_cfg.get("corridor_middle_axial_sigma", 0.55)),
                float(sites_cfg.get("corridor_middle_lateral_sigma", 0.75)),
            )
            for _ in range(middle_count)
        ]
    )
    zone_specs.extend(
        [
            (
                "pocket",
                min(center_distance, max_anchor_dist - 0.35),
                float(sites_cfg.get("corridor_pocket_axial_sigma", 0.65)),
                float(sites_cfg.get("corridor_pocket_lateral_sigma", 0.85)),
            )
            for _ in range(pocket_count)
        ]
    )
    zone_specs.extend(
        [
            (
                "distal",
                min(center_distance + distal_offset, max_anchor_dist - 0.35),
                float(sites_cfg.get("corridor_distal_axial_sigma", 0.70)),
                float(sites_cfg.get("corridor_distal_lateral_sigma", 0.95)),
            )
            for _ in range(distal_count)
        ]
    )

    placed: List[np.ndarray] = []
    for zone, axial_mean, axial_sigma, lateral_sigma in zone_specs:
        accepted = None
        for candidate_axis in _axis_candidates(zone):
            zone_axis, zone_tangent, zone_binormal = _zone_frame(zone, candidate_axis)
            for _attempt in range(attempts):
                candidate = (
                    anchor
                    + rng.normal(axial_mean, axial_sigma) * zone_axis
                    + rng.normal(0.0, lateral_sigma) * zone_tangent
                    + rng.normal(0.0, lateral_sigma) * zone_binormal
                )
                anchor_distance = float(np.linalg.norm(candidate - anchor))
                if zone == "connector" and anchor_distance < 1.20:
                    continue
                if anchor_distance > max_anchor_dist:
                    continue
                if _protein_clearance(candidate, local_protein) < min_protein_dist:
                    continue
                if (
                    placed
                    and float(
                        np.linalg.norm(np.asarray(placed) - candidate, axis=1).min()
                    )
                    < min_point_dist
                ):
                    continue
                accepted = candidate
                break
            if accepted is not None:
                break
        if accepted is None:
            raise ValueError(
                "interface corridor rejection sampling failed for "
                f"{zone} zone after {attempts} attempts"
            )
        placed.append(np.asarray(accepted, dtype=np.float64))

    return placed
