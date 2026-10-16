"""Extraction responsibilities extracted from scaffold_sites.py."""

from __future__ import annotations

from diffdynamic.chemistry.sites.state import Chem, Iterable, Optional

from diffdynamic.chemistry.sites.state import (
    Any,
    CST,
    Dict,
    List,
    Path,
    SCAFFOLD_SOURCES_WITH_ATTACHMENT_SITES,
    datetime,
    deque,
    json,
    load_scaffold_profile,
    np,
)


def _atom_positions(mol, ref_pos_np: Optional[np.ndarray] = None) -> np.ndarray:
    """返回 [N, 3] 坐标；优先 conformer，否则用 ref_pos_np。"""
    n = mol.GetNumAtoms()
    if mol.GetNumConformers() > 0:
        conf = mol.GetConformer()
        return np.array([conf.GetAtomPosition(i) for i in range(n)], dtype=np.float64)
    if ref_pos_np is not None and len(ref_pos_np) >= n:
        return np.asarray(ref_pos_np[:n], dtype=np.float64)
    return np.zeros((n, 3), dtype=np.float64)


def _is_heavy(atom: Chem.Atom) -> bool:
    return atom.GetAtomicNum() > 1


def _connected_components(mol, atom_indices: List[int]) -> List[List[int]]:
    """对 atom_indices 子图做连通分量划分。"""
    idx_set = set(atom_indices)
    adj: Dict[int, List[int]] = {i: [] for i in atom_indices}
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if a in idx_set and b in idx_set:
            adj[a].append(b)
            adj[b].append(a)

    seen = set()
    components = []
    for start in atom_indices:
        if start in seen:
            continue
        comp = []
        queue = deque([start])
        seen.add(start)
        while queue:
            u = queue.popleft()
            comp.append(u)
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    queue.append(v)
        components.append(comp)
    return components


def extract_murcko_attachment_sites(
    mol,
    scaffold_indices: List[int],
    ref_pos_np: Optional[np.ndarray] = None,
    dedup_dist: float = 0.5,
) -> List[Dict[str, Any]]:
    """
    从完整分子 + Murcko 骨架索引提取侧链位点（被去掉片段的质心）。

    Returns:
        list of AttachmentSite dicts
    """
    if mol is None or not scaffold_indices:
        return []

    scaffold_set = set(int(i) for i in scaffold_indices)
    n_atoms = mol.GetNumAtoms()
    non_scaffold = [i for i in range(n_atoms) if i not in scaffold_set]
    if not non_scaffold:
        return []

    positions = _atom_positions(mol, ref_pos_np)
    components = _connected_components(mol, non_scaffold)
    sites: List[Dict[str, Any]] = []

    for comp in components:
        heavy = [i for i in comp if _is_heavy(mol.GetAtomWithIdx(i))]
        if not heavy:
            continue

        anchor_candidates = []
        for i in comp:
            atom = mol.GetAtomWithIdx(i)
            for nb in atom.GetNeighbors():
                j = nb.GetIdx()
                if j in scaffold_set:
                    anchor_candidates.append(j)

        if not anchor_candidates:
            continue

        anchor_idx = int(anchor_candidates[0])
        frag_pos = positions[comp]
        centroid = frag_pos.mean(axis=0)
        anchor_pos = positions[anchor_idx]

        sites.append(
            {
                "site_id": len(sites),
                "anchor_scaffold_idx": anchor_idx,
                "anchor_pos": anchor_pos.tolist(),
                "centroid_pos": centroid.tolist(),
                "removed_atom_indices": [int(x) for x in comp],
                "removed_atom_count": len(comp),
                "removed_atom_positions": frag_pos.tolist(),
            }
        )

    # 按质心距离去重
    if dedup_dist > 0 and len(sites) > 1:
        merged: List[Dict[str, Any]] = []
        for site in sites:
            c = np.array(site["centroid_pos"], dtype=np.float64)
            dup = False
            for kept in merged:
                kc = np.array(kept["centroid_pos"], dtype=np.float64)
                if np.linalg.norm(c - kc) < dedup_dist:
                    dup = True
                    kept["removed_atom_indices"] = list(
                        set(kept["removed_atom_indices"])
                        | set(site["removed_atom_indices"])
                    )
                    kept["removed_atom_count"] = len(kept["removed_atom_indices"])
                    kept_pos = kept.get("removed_atom_positions", [])
                    site_pos = site.get("removed_atom_positions", [])
                    kept["removed_atom_positions"] = kept_pos + site_pos
                    break
            if not dup:
                site["site_id"] = len(merged)
                merged.append(site)
        sites = merged

    return sites


def extract_exit_vector_sites(
    mol,
    scaffold_indices: List[int],
    ref_pos_np: Optional[np.ndarray] = None,
    mode: str = "aromatic_h",
    offset: float = 1.5,
    existing_sites: Optional[List[Dict[str, Any]]] = None,
    dedup_dist: float = 1.2,
) -> List[Dict[str, Any]]:
    """
    在骨架环原子上构造虚拟释放点（exit vector）。

    用于 Murcko 只剥掉极少侧链、真实位点过少的情况（如 X77 仅 1 个 tBu 位点）。
    质心沿「锚点 → 远离邻居质心」方向外推 offset Å。

    mode:
      - aromatic_h: 芳香 C/N 且有显式/隐式 H
      - ring_h: 任意环上 C/N 且有 H
      - all_h: 骨架上任意 C/N 且有 H
    """
    if mol is None or not scaffold_indices:
        return []

    positions = _atom_positions(mol, ref_pos_np)
    ri = mol.GetRingInfo()
    ring_atoms = set()
    for ring in ri.AtomRings():
        ring_atoms.update(ring)

    mode = str(mode or "aromatic_h").lower()
    virtual: List[Dict[str, Any]] = []

    for idx in scaffold_indices:
        atom = mol.GetAtomWithIdx(int(idx))
        z = atom.GetAtomicNum()
        if z not in (6, 7):
            continue
        if atom.GetTotalNumHs() < 1:
            continue
        if mode == "aromatic_h":
            if not atom.GetIsAromatic():
                continue
        elif mode == "ring_h":
            if int(idx) not in ring_atoms:
                continue
        elif mode == "all_h":
            pass
        else:
            if not atom.GetIsAromatic():
                continue

        anchor = positions[int(idx)]
        nbs = [n.GetIdx() for n in atom.GetNeighbors()]
        if nbs:
            nb_cent = positions[nbs].mean(axis=0)
            direction = anchor - nb_cent
        else:
            direction = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        nrm = float(np.linalg.norm(direction))
        if nrm < 1e-6:
            direction = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            nrm = 1.0
        direction = direction / nrm
        centroid = anchor + float(offset) * direction

        virtual.append(
            {
                "site_id": -1,
                "anchor_scaffold_idx": int(idx),
                "anchor_pos": anchor.tolist(),
                "centroid_pos": centroid.tolist(),
                "removed_atom_indices": [],
                "removed_atom_count": 0,
                "site_kind": "exit_vector",
                "exit_vector_mode": mode,
            }
        )

    # 与已有真实侧链位点去重
    kept_existing = list(existing_sites or [])
    existing_cent = [
        np.asarray(s["centroid_pos"], dtype=np.float64) for s in kept_existing
    ]
    merged_virtual: List[Dict[str, Any]] = []
    for site in virtual:
        c = np.asarray(site["centroid_pos"], dtype=np.float64)
        if any(np.linalg.norm(c - ec) < dedup_dist for ec in existing_cent):
            continue
        if any(
            np.linalg.norm(c - np.asarray(k["centroid_pos"], dtype=np.float64))
            < dedup_dist
            for k in merged_virtual
        ):
            continue
        merged_virtual.append(site)

    return merged_virtual


def merge_attachment_and_exit_vector_sites(
    murcko_sites: List[Dict[str, Any]],
    exit_sites: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """合并真实侧链位点与虚拟 exit-vector，重编号 site_id。"""
    merged: List[Dict[str, Any]] = []
    for s in murcko_sites:
        sc = dict(s)
        sc.setdefault("site_kind", "murcko_sidechain")
        sc["site_id"] = len(merged)
        merged.append(sc)
    for s in exit_sites:
        sc = dict(s)
        sc.setdefault("site_kind", "exit_vector")
        sc["site_id"] = len(merged)
        merged.append(sc)
    return merged


def extract_reference_exit_vector_sites(
    mol,
    scaffold_indices: List[int],
    ref_pos_np: Optional[np.ndarray],
    profile: Dict[str, Any],
    offset: float = 1.5,
    allowed_slots: Optional[List[int]] = None,
    allowed_anchor_symbols: Optional[Iterable[str]] = None,
    anchor_mode: str = "scaffold_exit",
    interface_center_world: Optional[Iterable[float]] = None,
) -> List[Dict[str, Any]]:
    """Create geometry-only virtual exits from a scaffold-local profile.

    ``profile`` contains only scaffold-local weights.  Coordinates come from
    the supplied scaffold pose and no reference target-side atom is copied.

    Exit directions use scaffold neighbors only.  A target-side neighbor must
    not influence the initialization cloud in strict scaffold mode.  The
    ``preserved_amine`` mode uses only the fixed C-N prefix: the terminal
    non-aromatic N is the placement reference and the direction is the vector
    from the preceding fixed prefix atom to that N.
    """
    if mol is None or not scaffold_indices:
        return []
    if int(profile.get("n_scaffold", 0)) != len(scaffold_indices):
        return []

    positions = _atom_positions(mol, ref_pos_np)
    scaffold_set = set(int(index) for index in scaffold_indices)
    anchor_mode = str(anchor_mode or "scaffold_exit").strip().lower()
    amine_frame = None
    if anchor_mode in ("preserved_amine", "amine", "amine_n"):
        preserved_count = int(profile.get("preserved_amine_path_length", 0))
        if preserved_count < 1 or preserved_count > len(scaffold_indices):
            raise ValueError(
                "preserved_amine anchor mode requires a fixed amine prefix"
            )
        prefix = [int(index) for index in scaffold_indices[-preserved_count:]]
        amine_positions = [
            position
            for position, atom_idx in enumerate(prefix)
            if mol.GetAtomWithIdx(atom_idx).GetSymbol() == "N"
            and not mol.GetAtomWithIdx(atom_idx).GetIsAromatic()
        ]
        if not amine_positions:
            raise ValueError(
                "preserved_amine anchor mode could not find a fixed non-aromatic N"
            )
        amine_position = amine_positions[-1]
        amine_idx = prefix[amine_position]
        amine_anchor = positions[amine_idx]
        if amine_position > 0:
            previous_idx = prefix[amine_position - 1]
            direction = amine_anchor - positions[previous_idx]
        else:
            previous_idx = None
            core_indices = scaffold_indices[:-preserved_count]
            core_center = positions[core_indices].mean(axis=0)
            direction = amine_anchor - core_center
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-8:
            raise ValueError(
                "preserved_amine anchor mode could not determine an exit "
                "direction from the fixed prefix"
            )
        amine_frame = {
            "anchor_idx": int(amine_idx),
            "previous_idx": (int(previous_idx) if previous_idx is not None else None),
            "anchor_pos": amine_anchor,
            "direction": direction / norm,
        }
    # The pocket mode uses only this single centroid vector as a spatial
    # prior.  Individual target-side atom coordinates, types, and bonds are
    # never attached to a site or passed to the sampler.
    explicit_interface_center = None
    if interface_center_world is not None:
        explicit_interface_center = np.asarray(
            list(interface_center_world), dtype=np.float64
        )
        if (
            explicit_interface_center.shape != (3,)
            or not np.isfinite(explicit_interface_center).all()
        ):
            raise ValueError(
                "interface_center_world must contain three finite coordinates"
            )

    target_center = None
    if explicit_interface_center is None:
        target_indices = [
            index
            for index in range(mol.GetNumAtoms())
            if index not in scaffold_set and _is_heavy(mol.GetAtomWithIdx(index))
        ]
        if target_indices:
            target_center = positions[target_indices].mean(axis=0)
    weights = profile.get("exit_site_weights") or {}
    allowed = (
        {int(slot) for slot in allowed_slots} if allowed_slots is not None else None
    )
    allowed_symbols = None
    if allowed_anchor_symbols is not None:
        allowed_symbols = {
            str(symbol).strip()
            for symbol in allowed_anchor_symbols
            if str(symbol).strip()
        }
    sites: List[Dict[str, Any]] = []
    for raw_slot, raw_weight in weights.items():
        slot = int(raw_slot)
        if allowed is not None and slot not in allowed:
            continue
        if slot < 0 or slot >= len(scaffold_indices):
            continue
        atom_idx = int(scaffold_indices[slot])
        atom = mol.GetAtomWithIdx(atom_idx)
        if allowed_symbols is not None and atom.GetSymbol() not in allowed_symbols:
            continue
        source_anchor = positions[atom_idx]
        neighbour_indices = [
            n.GetIdx() for n in atom.GetNeighbors() if n.GetIdx() in scaffold_set
        ]
        if amine_frame is not None:
            placement_anchor_idx = int(amine_frame["anchor_idx"])
            anchor = np.asarray(amine_frame["anchor_pos"])
            direction = np.asarray(amine_frame["direction"])
        else:
            placement_anchor_idx = atom_idx
            anchor = source_anchor
            if neighbour_indices:
                neighbour_centroid = positions[neighbour_indices].mean(axis=0)
                direction = anchor - neighbour_centroid
            else:
                direction = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if amine_frame is None and neighbour_indices:
            direction = anchor - neighbour_centroid
        else:
            direction = np.asarray(direction, dtype=np.float64)
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-6:
            direction = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            direction = direction / norm
        site_record = {
            "site_id": -1,
            "anchor_scaffold_idx": placement_anchor_idx,
            "source_anchor_scaffold_idx": atom_idx,
            "anchor_pos": anchor.tolist(),
            "centroid_pos": (anchor + float(offset) * direction).tolist(),
            "removed_atom_indices": [],
            "removed_atom_count": 0,
            "site_kind": "reference_exit_vector",
            "profile_slot": slot,
            "site_selection_weight": max(float(raw_weight), 0.0),
            "exit_direction": direction.tolist(),
            "anchor_symbol": mol.GetAtomWithIdx(placement_anchor_idx).GetSymbol(),
            "source_anchor_symbol": atom.GetSymbol(),
            "placement_anchor_mode": anchor_mode,
            "placement_distance_reference_atom_idx": placement_anchor_idx,
        }
        pocket_center = (
            explicit_interface_center
            if explicit_interface_center is not None
            else target_center
        )
        if pocket_center is not None:
            pocket_center_offset = pocket_center - anchor
            site_record["pocket_center_offset"] = pocket_center_offset.tolist()
            site_record["pocket_center_distance"] = float(
                np.linalg.norm(pocket_center_offset)
            )
            site_record["pocket_center_source"] = (
                "explicit_interface_center"
                if explicit_interface_center is not None
                else "reference_target_centroid_vector"
            )
        sites.append(site_record)
    return sites


def _transfer_template_geometry(
    source_site: Dict[str, Any],
    target_site: Dict[str, Any],
    attachment_first: bool = True,
) -> Optional[List[List[float]]]:
    """Move a native target-side coordinate cloud to another scaffold exit."""
    raw_positions = source_site.get("removed_atom_positions") or []
    if not raw_positions:
        return None
    source_anchor = np.asarray(source_site.get("anchor_pos"), dtype=np.float64)
    target_anchor = np.asarray(target_site.get("anchor_pos"), dtype=np.float64)
    source_positions = np.asarray(raw_positions, dtype=np.float64)
    if source_positions.ndim != 2 or source_positions.shape[1] != 3:
        return None

    # Connected-component traversal is not guaranteed to start at the atom
    # bonded to the scaffold. Use the closest target-side atom as the
    # attachment end for the exit direction. Reordering is explicit because
    # preserving native component order is useful as a geometry ablation.
    source_distances = np.linalg.norm(source_positions - source_anchor, axis=1)
    attachment_index = int(np.argmin(source_distances))
    if attachment_first:
        attachment_order = np.argsort(source_distances, kind="stable")
        source_positions = source_positions[attachment_order]
        attachment_index = 0
    source_direction = source_positions[attachment_index] - source_anchor
    target_direction = (
        np.asarray(target_site.get("centroid_pos"), dtype=np.float64) - target_anchor
    )
    source_norm = float(np.linalg.norm(source_direction))
    target_norm = float(np.linalg.norm(target_direction))
    if source_norm <= 1e-8 or target_norm <= 1e-8:
        return None
    source_direction /= source_norm
    target_direction /= target_norm

    cross = np.cross(source_direction, target_direction)
    dot = float(np.clip(np.dot(source_direction, target_direction), -1.0, 1.0))
    cross_norm = float(np.linalg.norm(cross))
    if cross_norm <= 1e-8:
        if dot >= 0.0:
            rotation = np.eye(3, dtype=np.float64)
        else:
            basis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            if abs(float(np.dot(basis, source_direction))) > 0.9:
                basis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            axis = np.cross(source_direction, basis)
            axis /= np.linalg.norm(axis)
            rotation = 2.0 * np.outer(axis, axis) - np.eye(3, dtype=np.float64)
    else:
        axis = cross / cross_norm
        skew = np.array(
            [
                [0.0, -axis[2], axis[1]],
                [axis[2], 0.0, -axis[0]],
                [-axis[1], axis[0], 0.0],
            ]
        )
        angle = float(np.arctan2(cross_norm, dot))
        rotation = (
            np.eye(3, dtype=np.float64)
            + np.sin(angle) * skew
            + (1.0 - np.cos(angle)) * (skew @ skew)
        )

    relative = source_positions - source_anchor
    transformed = relative @ rotation.T + target_anchor
    return transformed.astype(np.float64).tolist()


def merge_reference_exit_sites(
    attachment_sites: List[Dict[str, Any]],
    profile_sites: List[Dict[str, Any]],
    dedup_dist: float = 1.2,
    template_atom_order: str = "attachment_first",
) -> List[Dict[str, Any]]:
    """Merge profile exits, accumulating weight on an existing native site."""
    merged = [dict(site) for site in attachment_sites]
    for site in merged:
        weight = site.get("site_selection_weight")
        site["site_selection_weight"] = 1.0 if weight is None else weight

    for candidate in profile_sites:
        anchor_idx = candidate.get("anchor_scaffold_idx")
        same_anchor = next(
            (site for site in merged if site.get("anchor_scaffold_idx") == anchor_idx),
            None,
        )
        if same_anchor is not None:
            same_anchor["site_selection_weight"] = max(
                float(same_anchor.get("site_selection_weight", 1.0)),
                float(candidate.get("site_selection_weight", 0.0)),
            )
            same_anchor["profile_slot"] = candidate.get("profile_slot")
            same_anchor["reference_exit_profile_slot"] = candidate.get("profile_slot")
            continue

        centroid = np.asarray(candidate["centroid_pos"], dtype=np.float64)
        duplicate = any(
            np.linalg.norm(
                centroid - np.asarray(site["centroid_pos"], dtype=np.float64)
            )
            < float(dedup_dist)
            for site in merged
        )
        if duplicate:
            continue
        candidate = dict(candidate)
        template_site = next(
            (
                site
                for site in merged
                if site.get("removed_atom_positions")
                and site.get("removed_atom_count", 0) > 0
            ),
            None,
        )
        if template_site is not None:
            transferred = _transfer_template_geometry(
                template_site,
                candidate,
                attachment_first=str(template_atom_order).lower()
                in ("attachment_first", "attached_first", "sorted"),
            )
            if transferred is not None:
                candidate["removed_atom_positions"] = transferred
                candidate["removed_atom_count"] = len(transferred)
                candidate["template_source_site_id"] = template_site.get("site_id")
        candidate["site_id"] = len(merged)
        merged.append(candidate)

    for site_id, site in enumerate(merged):
        site["site_id"] = site_id
    return merged


def load_or_extract_attachment_sites(
    mol,
    scaffold_indices: List[int],
    scaffold_source: str,
    ref_pos_np: Optional[np.ndarray],
    output_dir: Optional[Path],
    ref_ligand_name: Optional[str],
    sites_cfg: dict,
    logger=None,
) -> List[Dict[str, Any]]:
    """从（全分子 − 骨架索引）提取侧链自由基位点；Murcko/自定义骨架共用。"""
    source = str(scaffold_source or "").strip().lower()
    if source not in SCAFFOLD_SOURCES_WITH_ATTACHMENT_SITES:
        if logger:
            logger.info(
                f"[MurckoSites] scaffold_source={scaffold_source}，跳过侧链位点提取，使用口袋随机放置"
            )
        return []

    if mol is None or not scaffold_indices:
        return []

    if sites_cfg.get("custom_regions"):
        from diffdynamic.chemistry.scaffold_inputs import custom_attachment_sites

        return custom_attachment_sites(
            mol, scaffold_indices, sites_cfg["custom_regions"]
        )

    include_ev = bool(sites_cfg.get("include_exit_vectors", False))
    ev_mode = str(sites_cfg.get("exit_vector_mode", "aromatic_h"))
    profile_path = sites_cfg.get("reference_exit_profile")
    use_profile = bool(profile_path)
    strict_anchor_gaussian = bool(sites_cfg.get("strict_anchor_gaussian", False))
    strict_fragment_gaussian = bool(sites_cfg.get("strict_fragment_gaussian", False))
    fragment_gaussian = bool(sites_cfg.get("fragment_gaussian", False))
    pocket_center_gaussian = bool(sites_cfg.get("pocket_center_gaussian", False))
    interface_corridor_gaussian = bool(
        sites_cfg.get("interface_corridor_gaussian", False)
    )
    strict_geometry_only = (
        strict_anchor_gaussian
        or strict_fragment_gaussian
        or fragment_gaussian
        or pocket_center_gaussian
        or interface_corridor_gaussian
    )
    profile_tag = "_profile_exit" if use_profile else ""
    cache_suffix = (f"_ev_{ev_mode}" if include_ev else "") + profile_tag

    json_path = None
    if sites_cfg.get("save_json", True) and output_dir and ref_ligand_name:
        scaffold_dir = Path(output_dir) / "scaffold"
        scaffold_dir.mkdir(parents=True, exist_ok=True)
        json_path = scaffold_dir / f"{ref_ligand_name}_murcko_sites{cache_suffix}.json"
        if (
            json_path.exists()
            and sites_cfg.get("reuse_cache", True)
            and not strict_geometry_only
        ):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                sites = data.get("attachment_sites", [])
                cache_ok = bool(sites)
                if include_ev:
                    cache_ok = cache_ok and any(
                        s.get("site_kind") == "exit_vector" for s in sites
                    )
                if use_profile:
                    cache_ok = cache_ok and any(
                        s.get("site_kind") == "reference_exit_vector"
                        or s.get("reference_exit_profile_slot") is not None
                        for s in sites
                    )
                if cache_ok:
                    if logger:
                        logger.info(
                            f"[MurckoSites] 从缓存加载 {len(sites)} 个位点: {json_path}"
                        )
                    return sites
            except Exception as e:
                if logger:
                    logger.warning(f"[MurckoSites] 读取缓存失败: {e}")

    if strict_geometry_only and use_profile:
        # Do not even construct target-side attachment templates in strict
        # mode.  Profile exits below are derived from scaffold atoms only.
        sites = []
    else:
        sites = extract_murcko_attachment_sites(
            mol,
            scaffold_indices,
            ref_pos_np,
            dedup_dist=float(sites_cfg.get("dedup_dist", 0.5)),
        )
    for s in sites:
        s.setdefault("site_kind", "murcko_sidechain")

    if logger:
        logger.info(f"[MurckoSites] 提取 {len(sites)} 个侧链质心位点")

    if include_ev:
        exit_sites = extract_exit_vector_sites(
            mol,
            scaffold_indices,
            ref_pos_np,
            mode=ev_mode,
            offset=float(sites_cfg.get("exit_vector_offset", 1.5)),
            existing_sites=sites,
            dedup_dist=float(sites_cfg.get("exit_vector_dedup_dist", 1.2)),
        )
        sites = merge_attachment_and_exit_vector_sites(sites, exit_sites)
        n_ev = sum(1 for s in sites if s.get("site_kind") == "exit_vector")
        if logger:
            logger.info(
                f"[MurckoSites] exit_vector({ev_mode}): +{n_ev} 虚拟位点 → 合计 {len(sites)}"
            )

    if use_profile:
        try:
            profile = load_scaffold_profile(profile_path)
            allowed_anchor_symbols = sites_cfg.get("reference_exit_anchor_symbols")
            if isinstance(allowed_anchor_symbols, str):
                allowed_anchor_symbols = [allowed_anchor_symbols]
            profile_sites = extract_reference_exit_vector_sites(
                mol,
                scaffold_indices,
                ref_pos_np,
                profile,
                offset=float(sites_cfg.get("reference_exit_offset", 1.5)),
                allowed_slots=sites_cfg.get("reference_exit_slots"),
                allowed_anchor_symbols=allowed_anchor_symbols,
                anchor_mode=str(
                    sites_cfg.get("reference_exit_anchor_mode", "scaffold_exit")
                ),
                interface_center_world=sites_cfg.get("interface_center_world"),
            )
            if strict_geometry_only:
                # Strict mode has a geometry-only contract: profile exits are
                # rebuilt from the scaffold pose and never merged with native
                # side-chain sites or their cached target-side coordinates.
                merged_sites = [dict(site) for site in profile_sites]
            else:
                merged_sites = merge_reference_exit_sites(
                    sites,
                    profile_sites,
                    dedup_dist=float(sites_cfg.get("reference_exit_dedup_dist", 1.2)),
                    template_atom_order=str(
                        sites_cfg.get(
                            "reference_exit_template_order", "attachment_first"
                        )
                    ),
                )
            if bool(sites_cfg.get("reference_exit_only", False)):
                sites = [
                    site
                    for site in merged_sites
                    if site.get("site_kind") == "reference_exit_vector"
                    or site.get("reference_exit_profile_slot") is not None
                ]
                for site_id, site in enumerate(sites):
                    site["site_id"] = site_id
            else:
                sites = merged_sites
            if logger:
                logger.info(
                    f"[MurckoSites] reference exit profile: "
                    f"{len(profile_sites)} exits, matched_refs="
                    f"{profile.get('matched_reference_records', 0)}"
                )
        except Exception as exc:
            if logger:
                logger.warning(
                    f"[MurckoSites] reference exit profile unavailable: {exc}"
                )

    if json_path and sites:
        payload = {
            "ligand_name": ref_ligand_name,
            "n_scaffold": len(scaffold_indices),
            "scaffold_indices": [int(i) for i in scaffold_indices],
            "scaffold_source": scaffold_source,
            "include_exit_vectors": include_ev,
            "exit_vector_mode": ev_mode if include_ev else None,
            "reference_exit_profile": str(profile_path) if use_profile else None,
            "attachment_sites": sites,
            "extracted_at": datetime.now(CST).isoformat(),
        }
        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            if logger:
                logger.info(f"[MurckoSites] 位点已保存: {json_path}")
        except Exception as e:
            if logger:
                logger.warning(f"[MurckoSites] 保存位点 JSON 失败: {e}")

    return sites
