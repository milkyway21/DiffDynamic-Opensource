"""Preservation responsibilities extracted from scaffold_sites.py."""

from __future__ import annotations

from diffdynamic.chemistry.sites.state import Optional, Tuple

from diffdynamic.chemistry.sites.state import (
    Any,
    Dict,
    List,
    np,
    torch,
)


def concentrate_site_allocation(
    active_sites: List[Dict[str, Any]],
    counts: List[int],
    max_active_sites: int,
    rng: Optional[np.random.Generator] = None,
    prefer_murcko: bool = True,
) -> Tuple[List[Dict[str, Any]], List[int]]:
    """
    将已分配的位点浓缩到 ≤ max_active_sites 个，原子数合并。

    默认优先真实侧链去除位点（murcko_sidechain），再按当前 count 加权抽样，
    使多样本间仍可换位点，但单样本内原子聚成一团（仿从头单中心）。
    """
    if not active_sites or not counts:
        return active_sites, counts
    max_k = int(max_active_sites)
    if max_k <= 0 or len(active_sites) <= max_k:
        return active_sites, counts

    rng = rng or np.random.default_rng()
    indices = list(range(len(active_sites)))
    if prefer_murcko:
        murcko_idx = [
            i
            for i in indices
            if str(active_sites[i].get("site_kind", "murcko_sidechain"))
            != "exit_vector"
        ]
        pool = murcko_idx if murcko_idx else indices
    else:
        pool = indices

    weights = np.array([max(int(counts[i]), 1) for i in pool], dtype=np.float64)
    weights = weights / weights.sum()

    k = min(max_k, len(pool))
    if k == 1:
        chosen = [int(rng.choice(pool, p=weights))]
    else:
        # 无放回加权：逐次选，去掉已选再归一化
        chosen = []
        rem = list(pool)
        rem_w = weights.copy()
        for _ in range(k):
            rem_w = rem_w / rem_w.sum()
            pick = int(rng.choice(len(rem), p=rem_w))
            chosen.append(rem[pick])
            rem.pop(pick)
            rem_w = np.delete(rem_w, pick)

    total = int(sum(counts))
    if k == 1:
        return [active_sites[chosen[0]]], [total]

    # 多焦点：把 total 随机切到 k 个选中位点
    cuts = sorted(rng.integers(0, total + 1, size=k - 1).tolist()) if total > 0 else []
    bounds = [0] + cuts + [total]
    new_counts = [bounds[i + 1] - bounds[i] for i in range(k)]
    # 去掉 0 计数
    out_sites, out_counts = [], []
    for i, c in zip(chosen, new_counts):
        if c > 0:
            out_sites.append(active_sites[i])
            out_counts.append(int(c))
    if not out_sites and total > 0:
        return [active_sites[chosen[0]]], [total]
    return out_sites, out_counts


def collect_zero_allocation_sites(
    attachment_sites: List[Dict[str, Any]],
    site_allocation: Optional[List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """返回 concentrate 后仍无新原子分配、且带有原侧链索引的位点。

    exit_vector 等无 removed_atom_indices 的位点不会进入结果。
    """
    allocated_ids = set()
    if site_allocation:
        for a in site_allocation:
            if int(a.get("count", 0) or 0) <= 0:
                continue
            sid = a.get("site_id")
            if sid is not None:
                allocated_ids.add(int(sid))

    zero: List[Dict[str, Any]] = []
    for site in attachment_sites or []:
        sid = site.get("site_id")
        if sid is None:
            continue
        if int(sid) in allocated_ids:
            continue
        removed = site.get("removed_atom_indices") or []
        if not removed:
            continue
        zero.append(site)
    return zero


def gather_preserved_sidechain_atoms(
    zero_sites: List[Dict[str, Any]],
    ref_pos: torch.Tensor,
    ref_log_v: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    """从参考配体取出零分配位点的原侧链坐标与类型 log-onehot。

    Returns:
        (pos [n_preserved, 3], log_v [n_preserved, C], indices 参考配体 0-based 索引)
    """
    device = ref_pos.device
    n_classes = (
        int(ref_log_v.shape[-1]) if ref_log_v is not None and ref_log_v.ndim >= 2 else 0
    )
    indices: List[int] = []
    seen = set()
    for site in zero_sites or []:
        for idx in site.get("removed_atom_indices") or []:
            i = int(idx)
            if i in seen:
                continue
            if i < 0 or i >= int(ref_pos.shape[0]):
                continue
            seen.add(i)
            indices.append(i)

    if not indices:
        return (
            torch.zeros(0, 3, device=device),
            torch.zeros(0, n_classes, device=device),
            [],
        )

    idx_t = torch.tensor(indices, dtype=torch.long, device=device)
    return ref_pos.index_select(0, idx_t), ref_log_v.index_select(0, idx_t), indices


def resolve_preserved_sidechains(
    attachment_sites: List[Dict[str, Any]],
    place_meta: Optional[Dict[str, Any]],
    sites_cfg: dict,
    ref_pos: torch.Tensor,
    ref_log_v: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, List[int], List[int]]:
    """按配置决定是否保留零分配位点原侧链。

    Returns:
        (preserved_pos, preserved_log_v, preserved_indices, zero_allocation_site_ids)
    """
    meta = place_meta or {}
    cfg = sites_cfg or {}
    enabled = bool(cfg.get("preserve_zero_allocation_sidechains", True))
    zero_ids = [int(x) for x in (meta.get("zero_allocation_site_ids") or [])]
    if not enabled:
        device = ref_pos.device
        n_classes = (
            int(ref_log_v.shape[-1])
            if ref_log_v is not None and ref_log_v.ndim >= 2
            else 0
        )
        return (
            torch.zeros(0, 3, device=device),
            torch.zeros(0, n_classes, device=device),
            [],
            zero_ids,
        )

    # 优先用 meta 已写好的索引（build_extra_atom_positions 填充）
    preserved_from_meta = meta.get("preserved_atom_indices")
    if preserved_from_meta is not None:
        indices = [int(i) for i in preserved_from_meta]
        if not indices:
            device = ref_pos.device
            n_classes = (
                int(ref_log_v.shape[-1])
                if ref_log_v is not None and ref_log_v.ndim >= 2
                else 0
            )
            return (
                torch.zeros(0, 3, device=device),
                torch.zeros(0, n_classes, device=device),
                [],
                zero_ids,
            )
        idx_t = torch.tensor(indices, dtype=torch.long, device=ref_pos.device)
        return (
            ref_pos.index_select(0, idx_t),
            ref_log_v.index_select(0, idx_t),
            indices,
            zero_ids,
        )

    zero_sites = collect_zero_allocation_sites(
        attachment_sites,
        meta.get("site_allocation"),
    )
    zero_ids = [
        int(s.get("site_id")) for s in zero_sites if s.get("site_id") is not None
    ]
    pos, log_v, indices = gather_preserved_sidechain_atoms(
        zero_sites, ref_pos, ref_log_v
    )
    return pos, log_v, indices, zero_ids


def _fill_zero_allocation_preserve_meta(
    meta: Dict[str, Any],
    attachment_sites: List[Dict[str, Any]],
    site_allocation: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """根据最终 site_allocation 填写零分配位点与可保留原子索引。"""
    alloc = (
        site_allocation if site_allocation is not None else meta.get("site_allocation")
    )
    zero_sites = collect_zero_allocation_sites(attachment_sites, alloc)
    zero_ids: List[int] = []
    preserved: List[int] = []
    seen = set()
    for site in zero_sites:
        sid = site.get("site_id")
        if sid is not None:
            zero_ids.append(int(sid))
        for idx in site.get("removed_atom_indices") or []:
            i = int(idx)
            if i in seen:
                continue
            seen.add(i)
            preserved.append(i)
    meta["zero_allocation_site_ids"] = zero_ids
    meta["preserved_atom_indices"] = preserved
    meta["n_preserved_atoms"] = len(preserved)
    return meta
