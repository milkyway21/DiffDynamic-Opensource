"""Placement responsibilities extracted from scaffold_sites.py."""

from __future__ import annotations

from diffdynamic.chemistry.sites.state import List, Optional, Tuple

from diffdynamic.chemistry.sites.state import (
    Any,
    DEFAULT_MURCKO_SITES_CFG,
    Dict,
    np,
    torch,
)


def init_extra_positions_at_sites(
    active_sites: List[Dict[str, Any]],
    counts: List[int],
    device,
    jitter_std: float = 1.0,
    jitter_mode: str = "gaussian",
    rng: Optional[np.random.Generator] = None,
    sites_cfg: Optional[dict] = None,
    protein_positions: Optional[np.ndarray] = None,
) -> torch.Tensor:
    """按位点分配生成世界坐标系 extra 位置 [n_extra, 3]。

    gaussian：在 site centroid 周围生成高斯云并沿方向做轻微径向展开。
    native_template：若有 removed_atom_positions，则使用 native target-side
    的空间构象作模板并加高斯扰动；缺少模板时沿方向外推。
    directional：沿 anchor→centroid 射线外推（旧离散方案）。
    hybrid：有 native 模板时使用模板，否则沿 anchor→exit 方向外推。
    """
    from diffdynamic.chemistry.sites.gaussian import (
        _directional_site_position,
        _gaussian_site_position,
        _interface_corridor_gaussian_site_positions,
        _pocket_center_gaussian_site_positions,
        _strict_anchor_gaussian_site_positions,
    )
    from diffdynamic.chemistry.sites.pocket_geometry import (
        _anchor_radial_site_position,
        _original_site_position,
        _pocket_aware_site_positions,
    )

    cfg = sites_cfg or {}
    radial_step = float(cfg.get("radial_step", DEFAULT_MURCKO_SITES_CFG["radial_step"]))
    radial_bulk = float(cfg.get("radial_bulk", DEFAULT_MURCKO_SITES_CFG["radial_bulk"]))
    radial_bulk_start = int(
        cfg.get("radial_bulk_start", DEFAULT_MURCKO_SITES_CFG["radial_bulk_start"])
    )
    directional_step = float(
        cfg.get("directional_step", DEFAULT_MURCKO_SITES_CFG["directional_step"])
    )
    min_base_dist = float(
        cfg.get(
            "directional_min_base_dist",
            DEFAULT_MURCKO_SITES_CFG["directional_min_base_dist"],
        )
    )
    rng = rng or np.random.default_rng()
    positions = []
    mode = str(jitter_mode or "gaussian").lower()
    use_gaussian = mode in (
        "gaussian",
        "isotropic",
        "clustered",
        "denovo",
        "random",
    )
    use_native = mode in ("native", "native_template")
    orig_min_dist = float(cfg.get("original_min_dist", 2.0))
    orig_max_dist = float(cfg.get("original_max_dist", 4.0))
    for site, cnt in zip(active_sites, counts):
        if cnt <= 0:
            continue
        if mode == "strict_anchor_gaussian":
            positions.extend(
                _strict_anchor_gaussian_site_positions(
                    site,
                    cnt,
                    jitter_std,
                    rng,
                    cfg,
                )
            )
            continue
        if mode == "pocket_center_gaussian":
            positions.extend(
                _pocket_center_gaussian_site_positions(
                    site,
                    cnt,
                    protein_positions,
                    rng,
                    cfg,
                )
            )
            continue
        if mode == "interface_corridor_gaussian":
            positions.extend(
                _interface_corridor_gaussian_site_positions(
                    site,
                    cnt,
                    protein_positions,
                    rng,
                    cfg,
                )
            )
            continue
        if mode in ("pocket_aware_template", "pocket_aware_growth"):
            positions.extend(
                _pocket_aware_site_positions(
                    site,
                    cnt,
                    protein_positions,
                    jitter_std,
                    rng,
                    cfg,
                )
            )
            continue
        for atom_i in range(cnt):
            if use_gaussian:
                positions.append(
                    _gaussian_site_position(
                        site,
                        jitter_std,
                        rng,
                        atom_idx=atom_i,
                        site_count=cnt,
                        radial_step=radial_step,
                        radial_bulk=radial_bulk,
                        radial_bulk_start=radial_bulk_start,
                    )
                )
            elif mode in ("anchor_radial", "legacy_radial"):
                positions.append(
                    _anchor_radial_site_position(
                        site,
                        jitter_std,
                        rng,
                        atom_idx=atom_i,
                        site_count=cnt,
                        radial_step=radial_step,
                        radial_bulk=radial_bulk,
                        radial_bulk_start=radial_bulk_start,
                    )
                )
            elif use_native:
                positions.append(
                    _original_site_position(
                        site,
                        jitter_std,
                        rng,
                        atom_idx=atom_i,
                        site_count=cnt,
                        radial_step=radial_step,
                        min_dist=orig_min_dist,
                        max_dist=orig_max_dist,
                    )
                )
            elif str(jitter_mode).lower() == "directional":
                positions.append(
                    _directional_site_position(
                        site,
                        atom_i,
                        jitter_std,
                        rng,
                        directional_step=directional_step,
                        min_base_dist=min_base_dist,
                    )
                )
            elif mode == "hybrid":
                if site.get("removed_atom_positions"):
                    positions.append(
                        _original_site_position(
                            site,
                            jitter_std,
                            rng,
                            atom_idx=atom_i,
                            site_count=cnt,
                            radial_step=radial_step,
                            min_dist=orig_min_dist,
                            max_dist=orig_max_dist,
                        )
                    )
                else:
                    positions.append(
                        _directional_site_position(
                            site,
                            atom_i,
                            jitter_std,
                            rng,
                            directional_step=directional_step,
                            min_base_dist=min_base_dist,
                        )
                    )
            else:
                positions.append(
                    _original_site_position(
                        site,
                        jitter_std,
                        rng,
                        atom_idx=atom_i,
                        site_count=cnt,
                        radial_step=radial_step,
                        min_dist=orig_min_dist,
                        max_dist=orig_max_dist,
                    )
                )

    if not positions:
        return torch.zeros(0, 3, device=device)

    arr = np.array(positions, dtype=np.float32)
    return torch.tensor(arr, dtype=torch.float32, device=device)


def build_custom_region_positions(attachment_sites, protein_positions, rng, device):
    """Exact region budgets, shared by CPU preview and GPU initialization.

    Validate the complete cloud after the legacy proposal generators: neither
    crowded-pocket fallback nor independently sampled regions may violate the
    clearance contract. Never drop atoms or substitute an unrelated pocket.
    """
    from diffdynamic.chemistry.sites.gaussian import (
        _interface_corridor_gaussian_site_positions,
        _pocket_center_gaussian_site_positions,
    )
    from diffdynamic.chemistry.sites.pocket_geometry import _protein_clearance

    protein = np.asarray(
        protein_positions.detach().cpu().numpy()
        if isinstance(protein_positions, torch.Tensor)
        else protein_positions,
        dtype=np.float64,
    ).reshape(-1, 3)
    placed, allocation = [], []
    for site in attachment_sites:
        region = site["custom_region"]
        count = int(region["atom_count"])
        cfg = {
            "custom_region_strict": True,
            "pocket_gaussian_axial_sigma": float(region["sigma_axial"]),
            "pocket_gaussian_lateral_sigma": float(region["sigma_lateral"]),
            "pocket_max_anchor_dist": float("inf"),
            "pocket_min_protein_dist": 1.8,
            "pocket_min_point_dist": 0.8,
            "pocket_gaussian_attempts": 256,
            "corridor_middle_axial_sigma": float(region["sigma_axial"]),
            "corridor_pocket_axial_sigma": float(region["sigma_axial"]),
            "corridor_distal_axial_sigma": float(region["sigma_axial"]),
            "corridor_middle_lateral_sigma": float(region["sigma_lateral"]),
            "corridor_pocket_lateral_sigma": float(region["sigma_lateral"]),
            "corridor_distal_lateral_sigma": float(region["sigma_lateral"]),
        }
        proposal = (
            _pocket_center_gaussian_site_positions
            if region["method"] == "gaussian"
            else _interface_corridor_gaussian_site_positions
        )
        accepted = None
        for _ in range(16):
            try:
                cloud = np.asarray(proposal(site, count, protein, rng, cfg))
            except ValueError:
                continue
            if cloud.shape != (count, 3) or not np.isfinite(cloud).all():
                continue
            if protein.size and any(
                _protein_clearance(p, protein) < 1.8 for p in cloud
            ):
                continue
            joined = np.asarray(placed + cloud.tolist())
            distances = np.linalg.norm(joined[:, None] - joined[None, :], axis=-1)
            np.fill_diagonal(distances, np.inf)
            if np.any(distances < 0.8):
                continue
            accepted = cloud
            break
        if accepted is None:
            raise ValueError(
                f"Region {int(site['site_id']) + 1}: cannot place {count} atoms "
                "with protein clearance 1.8 Å and point separation 0.8 Å; "
                "move the center or increase its width"
            )
        start = len(placed)
        placed.extend(accepted.tolist())
        allocation.append(
            {
                "site_id": site["site_id"],
                "count": count,
                "centroid_pos": site["centroid_pos"],
                "anchor_atom_index": region["anchor_atom_index"],
                "method": region["method"],
                "start": start,
                "end": len(placed),
            }
        )
    return torch.tensor(placed, dtype=torch.float32, device=device), {
        "placement": "custom_regions",
        "n_extra_effective": len(placed),
        "n_extra_placed": len(placed),
        "site_allocation": allocation,
        "preserved_atom_indices": [],
        "zero_allocation_site_ids": [],
        "custom_regions": [dict(s["custom_region"]) for s in attachment_sites],
    }


def _fallback_extra_positions(count, center, device, scale, rng, cfg, protein):
    """Use one CPU RNG/frame for web previews and sampling fallbacks."""
    if not cfg.get("deterministic_fallback"):
        return (
            center.unsqueeze(0).expand(count, -1)
            + torch.randn(count, 3, device=device) * scale
        )
    rng = rng if rng is not None else np.random.default_rng()
    if isinstance(protein, torch.Tensor):
        protein = protein.detach().cpu().numpy()
    if protein is not None and len(protein):
        origin = np.asarray(protein, dtype=np.float64).mean(axis=0)
    else:
        origin = center.detach().cpu().numpy()
    positions = origin + rng.normal(0.0, scale, size=(count, 3))
    return torch.tensor(positions, dtype=torch.float32, device=device)


def build_extra_atom_positions(
    n_extra: int,
    attachment_sites: List[Dict[str, Any]],
    sites_cfg: dict,
    fallback_center: torch.Tensor,
    device,
    fallback_noise_scale: float = 2.0,
    logger=None,
    rng: Optional[np.random.Generator] = None,
    protein_positions: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    为 grow / dynamic_locked / prudent 生成额外原子初始坐标。

    有位点时按 Murcko 侧链质心分配；不足或失败时回退口袋/蛋白中心随机放置。
    """
    from diffdynamic.chemistry.sites.allocation import (
        allocate_atoms_random_per_site,
        allocate_atoms_reference_joint,
        allocate_atoms_sequential_random,
        allocate_atoms_to_sites,
        allocate_atoms_weighted_single_site,
        cap_n_extra_for_sites,
    )
    from diffdynamic.chemistry.sites.configuration import (
        count_removed_sidechain_sites,
        resolve_max_active_sites,
        uses_site_budget_placement,
    )
    from diffdynamic.chemistry.sites.fragment_geometry import (
        _strict_fragment_gaussian_site_positions,
    )
    from diffdynamic.chemistry.sites.preservation import (
        _fill_zero_allocation_preserve_meta,
        concentrate_site_allocation,
    )

    if sites_cfg.get("custom_regions"):
        return build_custom_region_positions(
            attachment_sites,
            protein_positions,
            rng or np.random.default_rng(),
            device,
        )
    n_extra_requested = n_extra
    strict_anchor_gaussian = bool(sites_cfg.get("strict_anchor_gaussian", False))
    strict_fragment_gaussian = bool(sites_cfg.get("strict_fragment_gaussian", False))
    fragment_gaussian = bool(sites_cfg.get("fragment_gaussian", False))
    pocket_center_gaussian = bool(sites_cfg.get("pocket_center_gaussian", False))
    interface_corridor_gaussian = bool(
        sites_cfg.get("interface_corridor_gaussian", False)
    )
    fragment_geometry = strict_fragment_gaussian or fragment_gaussian
    if fragment_geometry and (pocket_center_gaussian or interface_corridor_gaussian):
        raise ValueError(
            "fragment Gaussian geometry cannot be combined with pocket or "
            "interface corridor Gaussian placement; choose one coordinate "
            "initializer explicitly"
        )
    if pocket_center_gaussian and interface_corridor_gaussian:
        raise ValueError(
            "pocket_center_gaussian and interface_corridor_gaussian are "
            "mutually exclusive coordinate initializers"
        )
    overflow_mode = str(sites_cfg.get("overflow_mode", "pocket_fallback"))
    meta: Dict[str, Any] = {
        "placement": "pocket_fallback",
        "site_allocation": None,
        "overflow_count": 0,
        "n_extra_requested": n_extra_requested,
        "n_extra_effective": n_extra_requested,
        "overflow_mode": overflow_mode,
        "strict_fragment_gaussian": strict_fragment_gaussian,
        "fragment_gaussian": fragment_gaussian,
        "pocket_center_gaussian": pocket_center_gaussian,
        "interface_corridor_gaussian": interface_corridor_gaussian,
        "fragment_type_hints": [],
        "fragment_layout": [],
    }

    if n_extra <= 0 and not uses_site_budget_placement(sites_cfg):
        meta["n_extra_effective"] = 0
        meta["site_allocation"] = []
        return torch.zeros(0, 3, device=device), _fill_zero_allocation_preserve_meta(
            meta,
            attachment_sites or [],
            [],
        )

    if not attachment_sites:
        if (
            strict_anchor_gaussian
            or fragment_geometry
            or pocket_center_gaussian
            or interface_corridor_gaussian
        ):
            raise ValueError(
                "strict scaffold Gaussian modes require scaffold exit anchor sites"
            )
        pos = _fallback_extra_positions(
            n_extra,
            fallback_center,
            device,
            fallback_noise_scale,
            rng,
            sites_cfg,
            protein_positions,
        )
        return pos, _fill_zero_allocation_preserve_meta(meta, [], [])

    p_active = float(sites_cfg.get("p_active", 0.5))
    max_per_site = int(sites_cfg.get("max_per_site", 20))
    jitter_std = float(sites_cfg.get("jitter_std", 1.0))
    jitter_mode = str(sites_cfg.get("jitter_mode", "gaussian"))
    count_mode = str(sites_cfg.get("per_site_count_mode", "split"))
    max_active_sites = resolve_max_active_sites(sites_cfg, attachment_sites)
    prefer_murcko = bool(sites_cfg.get("prefer_murcko_sites", True))
    meta["n_removed_sidechain_sites"] = count_removed_sidechain_sites(attachment_sites)

    if overflow_mode in ("cap", "drop"):
        p_active = float(sites_cfg.get("p_active", 0.5))

    joint_pattern = None
    if str(sites_cfg.get("site_selection_mode", "")).lower() == "reference_joint":
        active, counts, n_extra_eff, joint_pattern = allocate_atoms_reference_joint(
            n_extra,
            attachment_sites,
            sites_cfg,
            rng=rng,
        )
        overflow = 0
        meta["n_extra_effective"] = n_extra_eff
        meta["per_site_count_mode"] = "reference_joint"
        meta["site_selection_mode"] = "reference_joint"
        meta["reference_allocation_pattern"] = joint_pattern
    elif count_mode in ("random_per_site", "sequential_random"):
        min_c = int(sites_cfg.get("n_extra_min_clamp", 0))
        max_c = int(sites_cfg.get("n_extra_max_clamp", max(n_extra, 1)))
        cap_hi = cap_n_extra_for_sites(max_c, attachment_sites, sites_cfg)
        max_c = min(max_c, cap_hi) if cap_hi > 0 else max_c
        min_c = min(min_c, max_c)
        alloc_fn = (
            allocate_atoms_sequential_random
            if count_mode == "sequential_random"
            else allocate_atoms_random_per_site
        )
        active, counts, n_extra_eff = alloc_fn(
            attachment_sites,
            sites_cfg,
            n_extra_bounds=(min_c, max_c),
            rng=rng,
        )
        overflow = 0
        meta["n_extra_effective"] = n_extra_eff
        meta["per_site_count_mode"] = count_mode
        if n_extra_eff <= 0:
            meta["site_allocation"] = []
            return torch.zeros(
                0, 3, device=device
            ), _fill_zero_allocation_preserve_meta(
                meta,
                attachment_sites,
                [],
            )
        if logger:
            logger.info(
                f"[MurckoSites] {count_mode}: 各位点分配 {counts} → n_extra={n_extra_eff} "
                f"(budget∈[{min_c},{max_c}])"
            )
    elif str(sites_cfg.get("site_selection_mode", "")).lower() == "weighted_single":
        active, counts, n_extra_eff = allocate_atoms_weighted_single_site(
            n_extra,
            attachment_sites,
            sites_cfg,
            rng=rng,
        )
        overflow = max(0, int(n_extra) - int(n_extra_eff))
        meta["n_extra_effective"] = n_extra_eff
        meta["per_site_count_mode"] = "weighted_single"
        meta["site_selection_mode"] = "weighted_single"
        if logger and active:
            logger.info(
                f"[MurckoSites] weighted_single: site_id="
                f"{active[0].get('site_id')} slot="
                f"{active[0].get('profile_slot')} weight="
                f"{active[0].get('site_selection_weight')} "
                f"→ n_extra={n_extra_eff}"
            )
    else:
        if overflow_mode in ("cap", "drop"):
            n_extra = cap_n_extra_for_sites(n_extra, attachment_sites, sites_cfg)
            meta["n_extra_effective"] = n_extra
            if n_extra < n_extra_requested and logger:
                logger.info(
                    f"[MurckoSites] overflow_mode={overflow_mode}: "
                    f"n_extra {n_extra_requested} → {n_extra}（位点容量上限）"
                )
            if n_extra <= 0:
                meta["site_allocation"] = []
                return torch.zeros(
                    0, 3, device=device
                ), _fill_zero_allocation_preserve_meta(
                    meta,
                    attachment_sites,
                    [],
                )

        active, counts, overflow = allocate_atoms_to_sites(
            n_extra,
            attachment_sites,
            p_active=p_active,
            max_per_site=max_per_site,
            rng=rng,
        )

        drop_overflow = overflow_mode in ("cap", "drop")
        if overflow > 0:
            if drop_overflow:
                if logger:
                    logger.info(
                        f"[MurckoSites] overflow_mode={overflow_mode}: "
                        f"丢弃 {overflow} 个溢出原子（不口袋随机放置）"
                    )
                meta["overflow_dropped"] = overflow
                overflow = 0
            elif logger:
                logger.warning(
                    f"[MurckoSites] n_extra={n_extra} 超出位点容量，"
                    f"{overflow} 个原子回退口袋随机放置"
                )
        n_extra_eff = n_extra

    # Reference-joint patterns are already the intended complete allocation.
    n_before = len([c for c in counts if c > 0])
    if joint_pattern is None:
        active, counts = concentrate_site_allocation(
            active,
            counts,
            max_active_sites,
            rng=rng,
            prefer_murcko=prefer_murcko,
        )
    meta["max_active_sites"] = max_active_sites
    meta["n_sites_before_concentrate"] = n_before
    meta["n_sites_after_concentrate"] = len(active)
    if logger and n_before != len(active):
        logger.info(
            f"[MurckoSites] concentrate: {n_before} → {len(active)} 位点 "
            f"(max_active_sites={max_active_sites}, counts={counts})"
        )

    protein_np = None
    if protein_positions is not None:
        if isinstance(protein_positions, torch.Tensor):
            protein_np = protein_positions.detach().cpu().numpy().astype(np.float64)
        else:
            protein_np = np.asarray(protein_positions, dtype=np.float64)
    if fragment_geometry:
        if overflow > 0:
            if strict_fragment_gaussian:
                raise ValueError("strict_fragment_gaussian cannot place overflow atoms")
        place_rng = rng if rng is not None else np.random.default_rng()
        min_protein_dist = float(sites_cfg.get("pocket_min_protein_dist", 0.0) or 0.0)
        attempts = int(sites_cfg.get("fragment_clearance_attempts", 1) or 1)
        if protein_np is None or min_protein_dist <= 0.0:
            attempts = 1
        best = None
        accepted = None
        used = 0
        for used in range(max(1, attempts)):
            fragment_positions, fragment_type_hints, fragment_layout = (
                _strict_fragment_gaussian_site_positions(
                    active, counts, place_rng, sites_cfg
                )
            )
            if protein_np is None or min_protein_dist <= 0.0:
                accepted = (
                    fragment_positions,
                    fragment_type_hints,
                    fragment_layout,
                    None,
                )
                break
            arr = np.asarray(fragment_positions, dtype=np.float64)
            clearance = float(
                np.linalg.norm(arr[:, None, :] - protein_np[None, :, :], axis=2).min()
            )
            candidate = (
                fragment_positions,
                fragment_type_hints,
                fragment_layout,
                clearance,
            )
            if best is None or clearance > best[3]:
                best = candidate
            if clearance + 1e-5 >= min_protein_dist:
                accepted = candidate
                break
        if accepted is None:
            accepted = best
        fragment_positions, fragment_type_hints, fragment_layout, clearance = accepted
        site_pos = torch.tensor(
            np.asarray(fragment_positions, dtype=np.float32),
            dtype=torch.float32,
            device=device,
        )
        meta["fragment_type_hints"] = fragment_type_hints
        meta["fragment_layout"] = fragment_layout
        meta["fragment_protein_clearance"] = clearance
        meta["fragment_clearance_attempts_used"] = used + 1
    else:
        site_pos = init_extra_positions_at_sites(
            active,
            counts,
            device,
            jitter_std=jitter_std,
            jitter_mode=jitter_mode,
            rng=rng,
            sites_cfg=sites_cfg,
            protein_positions=protein_np,
        )
    n_site = site_pos.size(0)

    parts = []
    if n_site > 0:
        parts.append(site_pos)

    if overflow > 0:
        overflow_pos = _fallback_extra_positions(
            overflow,
            fallback_center,
            device,
            fallback_noise_scale,
            rng,
            sites_cfg,
            protein_positions,
        )
        parts.append(overflow_pos)

    if not parts:
        pos = _fallback_extra_positions(
            max(n_extra_eff, 1),
            fallback_center,
            device,
            fallback_noise_scale,
            rng,
            sites_cfg,
            protein_positions,
        )
        meta["site_allocation"] = [
            {
                "site_id": active[i].get("site_id"),
                "count": counts[i],
                "centroid_pos": active[i].get("centroid_pos"),
                "profile_slot": active[i].get("profile_slot"),
                "site_selection_weight": active[i].get("site_selection_weight"),
            }
            for i in range(len(active))
            if counts[i] > 0
        ]
        return pos, _fill_zero_allocation_preserve_meta(meta, attachment_sites)

    extra_pos = torch.cat(parts, dim=0)
    n_placed = extra_pos.size(0)
    meta["n_extra_effective"] = n_placed
    meta["n_extra_placed"] = n_placed

    drop_overflow = overflow_mode in ("cap", "drop")
    if (
        n_placed != n_extra_eff
        and not uses_site_budget_placement(sites_cfg)
        and not drop_overflow
    ):
        if logger:
            logger.warning(
                f"[MurckoSites] 分配异常 ({n_placed} != {n_extra_eff})，整批回退口袋放置"
            )
        pos = _fallback_extra_positions(
            n_extra,
            fallback_center,
            device,
            fallback_noise_scale,
            rng,
            sites_cfg,
            protein_positions,
        )
        meta["n_extra_effective"] = n_extra
        meta["site_allocation"] = [
            {
                "site_id": active[i].get("site_id"),
                "count": counts[i],
                "centroid_pos": active[i].get("centroid_pos"),
                "profile_slot": active[i].get("profile_slot"),
                "site_selection_weight": active[i].get("site_selection_weight"),
            }
            for i in range(len(active))
            if counts[i] > 0
        ]
        return pos, _fill_zero_allocation_preserve_meta(meta, attachment_sites)

    meta["placement"] = "murcko_sites"
    meta["overflow_count"] = overflow
    meta["site_allocation"] = [
        {
            "site_id": active[i].get("site_id"),
            "count": counts[i],
            "centroid_pos": active[i].get("centroid_pos"),
            "profile_slot": active[i].get("profile_slot"),
            "site_selection_weight": active[i].get("site_selection_weight"),
        }
        for i in range(len(active))
        if counts[i] > 0
    ]
    return extra_pos, _fill_zero_allocation_preserve_meta(meta, attachment_sites)
