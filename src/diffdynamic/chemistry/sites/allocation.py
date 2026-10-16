"""Allocation responsibilities extracted from scaffold_sites.py."""

from __future__ import annotations

from diffdynamic.chemistry.sites.state import Optional, Tuple

from diffdynamic.chemistry.sites.state import (
    Any,
    DEFAULT_FRAGMENT_SIZES,
    DEFAULT_FRAGMENT_WEIGHTS,
    Dict,
    List,
    load_scaffold_profile,
    np,
)


def sample_fragment_atom_count(
    rng: np.random.Generator,
    sites_cfg: dict,
    room: int,
) -> int:
    """
    从常见/复杂自由基重原子数离散先验采样，并 clamp 到可用 room。

    默认目录含单苯(6)、共边双苯量级(11)、联苯(12) 等；不可达的 k 会被过滤。
    """
    if room <= 0:
        return 0
    max_per_site = int(sites_cfg.get("max_per_site", 20))
    min_per_site = int(sites_cfg.get("min_per_site", 0))
    lo = max(1, min_per_site) if min_per_site > 0 else 1
    hi = min(int(room), max_per_site)
    if hi < lo:
        return 0

    sizes = list(sites_cfg.get("fragment_sizes") or DEFAULT_FRAGMENT_SIZES)
    weights = list(sites_cfg.get("fragment_weights") or DEFAULT_FRAGMENT_WEIGHTS)
    if len(weights) != len(sizes):
        weights = list(DEFAULT_FRAGMENT_WEIGHTS[: len(sizes)])
        if len(weights) < len(sizes):
            weights.extend([1e-6] * (len(sizes) - len(weights)))

    kept_sizes: List[int] = []
    kept_w: List[float] = []
    for k, w in zip(sizes, weights):
        ki = int(k)
        if lo <= ki <= hi and float(w) > 0:
            kept_sizes.append(ki)
            kept_w.append(float(w))
    if not kept_sizes:
        return int(rng.integers(lo, hi + 1))
    w_arr = np.asarray(kept_w, dtype=np.float64)
    w_arr = w_arr / w_arr.sum()
    return int(rng.choice(kept_sizes, p=w_arr))


def sample_per_site_add(
    rng: np.random.Generator,
    sites_cfg: dict,
    room: int,
    *,
    allow_zero_uniform: bool = True,
) -> int:
    """按 per_site_add_mode 决定本轮向位点写入的原子数。"""
    if room <= 0:
        return 0
    mode = str(sites_cfg.get("per_site_add_mode", "fragment_prior")).lower()
    max_per_site = int(sites_cfg.get("max_per_site", 20))
    min_per_site = int(sites_cfg.get("min_per_site", 0))
    min_per_site = max(0, min(min_per_site, max_per_site))
    hi_add = min(max_per_site, int(room))
    if mode in ("uniform", "uniform_random"):
        lo_add = min(min_per_site, hi_add)
        if allow_zero_uniform:
            return int(rng.integers(lo_add, hi_add + 1))
        lo_add = max(1, lo_add) if hi_add >= 1 else 0
        if lo_add > hi_add:
            return 0
        return int(rng.integers(lo_add, hi_add + 1))
    return sample_fragment_atom_count(rng, sites_cfg, room)


def allocate_atoms_to_sites(
    n_extra: int,
    sites: List[Dict[str, Any]],
    p_active: float = 0.5,
    max_per_site: int = 20,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[List[Dict[str, Any]], List[int], int]:
    """
    将 n_extra 个原子分配到激活位点。

    Returns:
        (active_sites, counts_per_site, overflow_count)
        sum(counts) + overflow_count == n_extra
    """
    if n_extra <= 0 or not sites:
        return [], [], n_extra

    rng = rng or np.random.default_rng()
    active = [s for s in sites if rng.random() < p_active]
    if not active:
        active = [sites[int(rng.integers(0, len(sites)))]]

    n_active = len(active)
    capacity = n_active * max_per_site
    allocatable = min(n_extra, capacity)
    overflow = n_extra - allocatable

    counts = [0] * n_active
    if allocatable == 0:
        return active, counts, overflow

    if allocatable <= n_active:
        chosen = rng.choice(n_active, size=allocatable, replace=False)
        for idx in chosen:
            counts[int(idx)] += 1
        return active, counts, overflow

    # 随机切分 + clamp + 迭代调整
    cuts = sorted(rng.integers(0, allocatable + 1, size=max(n_active - 1, 0)).tolist())
    boundaries = [0] + cuts + [allocatable]
    for i in range(n_active):
        counts[i] = min(max_per_site, boundaries[i + 1] - boundaries[i])

    deficit = allocatable - sum(counts)
    for _ in range(100):
        if deficit == 0:
            break
        adjustable = [i for i, c in enumerate(counts) if c < max_per_site]
        if not adjustable:
            break
        idx = int(rng.choice(adjustable))
        counts[idx] += 1
        deficit -= 1

    if deficit > 0:
        overflow += deficit
        for i in range(n_active):
            counts[i] = min(counts[i], max_per_site)

    return active, counts, overflow


def allocate_atoms_random_per_site(
    sites: List[Dict[str, Any]],
    sites_cfg: dict,
    n_extra_bounds: Optional[Tuple[int, int]] = None,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[List[Dict[str, Any]], List[int], int]:
    """
    每位点独立随机原子数（增强样本间多样性）。

    默认 per_site_add_mode=fragment_prior：每激活位点采样一个自由基规模；
    uniform 时退回 Uniform[min_per_site, max_per_site]。
    总 n_extra = sum(counts)，再 clamp 到 n_extra_bounds（若提供）。

    Returns:
        (active_sites, counts_per_site, effective_n_extra)
    """
    if not sites:
        return [], [], 0

    rng = rng or np.random.default_rng()
    p_active = float(sites_cfg.get("p_active", 0.5))
    max_per_site = int(sites_cfg.get("max_per_site", 20))
    min_per_site = int(sites_cfg.get("min_per_site", 0))
    min_per_site = max(0, min(min_per_site, max_per_site))

    active = [s for s in sites if rng.random() < p_active]
    if not active:
        active = [sites[int(rng.integers(0, len(sites)))]]

    counts = [
        sample_per_site_add(rng, sites_cfg, max_per_site, allow_zero_uniform=True)
        for _ in active
    ]
    # fragment_prior 不会返回 0（除非 room=0）；若全 0 则强制至少 1
    if sum(counts) == 0 and active:
        counts[0] = (
            sample_per_site_add(
                rng,
                sites_cfg,
                max_per_site,
                allow_zero_uniform=False,
            )
            or 1
        )
    n_extra = sum(counts)

    if n_extra_bounds is not None:
        lo, hi = n_extra_bounds
        lo, hi = int(lo), int(hi)
        if n_extra < lo:
            deficit = lo - n_extra
            for _ in range(deficit):
                idx = int(rng.integers(0, len(counts)))
                if counts[idx] < max_per_site:
                    counts[idx] += 1
                    n_extra += 1
                elif n_extra >= lo:
                    break
            # 若仍不足，强制加到第一个位点（受 max_per_site 限制）
            while n_extra < lo and counts:
                for i in range(len(counts)):
                    if counts[i] < max_per_site:
                        counts[i] += 1
                        n_extra += 1
                        if n_extra >= lo:
                            break
                else:
                    break
        if n_extra > hi:
            surplus = n_extra - hi
            for _ in range(surplus):
                candidates = [i for i, c in enumerate(counts) if c > min_per_site]
                if not candidates:
                    candidates = [i for i, c in enumerate(counts) if c > 0]
                if not candidates:
                    break
                idx = int(rng.choice(candidates))
                counts[idx] -= 1
                n_extra -= 1

    return active, counts, n_extra


def allocate_atoms_sequential_random(
    sites: List[Dict[str, Any]],
    sites_cfg: dict,
    n_extra_bounds: Optional[Tuple[int, int]] = None,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[List[Dict[str, Any]], List[int], int]:
    """
    按位点顺序遍历：每位点以 p_active（默认 1/2）概率尝试加入，
    加入量默认按自由基规模先验（fragment_prior），上限 max_per_site（默认 20），
    多轮扫描直至达到本样本目标原子数（在 [lo, hi] 内随机，hi 为总数上限）。

    Returns:
        (active_sites, counts_per_site, effective_n_extra)
    """
    if not sites:
        return [], [], 0

    rng = rng or np.random.default_rng()
    p_active = float(sites_cfg.get("p_active", 0.5))
    max_per_site = int(sites_cfg.get("max_per_site", 20))
    min_per_site = int(sites_cfg.get("min_per_site", 0))
    min_per_site = max(0, min(min_per_site, max_per_site))

    lo, hi = 0, compute_site_capacity(sites, sites_cfg)
    if n_extra_bounds is not None:
        lo, hi = int(n_extra_bounds[0]), int(n_extra_bounds[1])
    lo = max(0, lo)
    hi = max(lo, hi)
    cap_hi = compute_site_capacity(sites, sites_cfg)
    hi = min(hi, cap_hi) if cap_hi > 0 else hi
    lo = min(lo, hi)

    if hi <= 0:
        return [], [], 0

    # 本样本目标总数：在 [lo, hi] 均匀随机，再按序分配至该目标
    budget = int(rng.integers(lo, hi + 1)) if hi > lo else hi
    if budget <= 0:
        return [], [], 0

    ordered = sorted(sites, key=lambda s: int(s.get("site_id", 0)))
    n_sites = len(ordered)
    counts_arr = [0] * n_sites
    remaining = budget

    stall = 0
    max_stall = max(n_sites * 4, 8)
    while remaining > 0 and stall < max_stall:
        added_round = False
        for idx in range(n_sites):
            if remaining <= 0:
                break
            room = min(max_per_site - counts_arr[idx], remaining)
            if room <= 0:
                continue
            if rng.random() >= p_active:
                continue
            add = sample_per_site_add(rng, sites_cfg, room, allow_zero_uniform=True)
            if add <= 0:
                continue
            counts_arr[idx] += add
            remaining -= add
            added_round = True
        if added_round:
            stall = 0
            continue
        stall += 1
        eligible = [i for i in range(n_sites) if counts_arr[i] < max_per_site]
        if not eligible or remaining <= 0:
            break
        idx = int(rng.choice(eligible))
        room = min(max_per_site - counts_arr[idx], remaining)
        # 打破停滞时仍走片段先验（至少 1），避免退回任意 Uniform
        add = sample_per_site_add(rng, sites_cfg, room, allow_zero_uniform=False)
        if add <= 0:
            add = int(rng.integers(1, room + 1)) if room >= 1 else 0
        if add <= 0:
            break
        counts_arr[idx] += add
        remaining -= add
        stall = 0

    active: List[Dict[str, Any]] = []
    counts: List[int] = []
    for idx, site in enumerate(ordered):
        if counts_arr[idx] > 0:
            active.append(site)
            counts.append(counts_arr[idx])

    return active, counts, budget - remaining


def compute_site_capacity(
    sites: List[Dict[str, Any]],
    sites_cfg: dict,
) -> int:
    """位点可放置原子的最大容量（全部位点激活时）。"""
    if not sites:
        return 0
    max_per_site = int(sites_cfg.get("max_per_site", 20))
    return len(sites) * max_per_site


def cap_n_extra_for_sites(
    n_extra: int,
    attachment_sites: List[Dict[str, Any]],
    sites_cfg: dict,
) -> int:
    """overflow_mode=cap/drop 时，将 n_extra 限制在位点总容量内。"""
    overflow_mode = str(sites_cfg.get("overflow_mode", "pocket_fallback"))
    if overflow_mode not in ("cap", "drop") or not attachment_sites:
        return n_extra
    return min(n_extra, compute_site_capacity(attachment_sites, sites_cfg))


def allocate_atoms_weighted_single_site(
    n_extra: int,
    sites: List[Dict[str, Any]],
    sites_cfg: dict,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[List[Dict[str, Any]], List[int], int]:
    """Choose one scaffold exit using profile weights and place the full budget."""
    if n_extra <= 0 or not sites:
        return [], [], 0
    rng = rng or np.random.default_rng()
    weights = np.asarray(
        [max(float(site.get("site_selection_weight", 1.0)), 0.0) for site in sites],
        dtype=np.float64,
    )
    if not np.isfinite(weights).all() or weights.sum() <= 0:
        weights = np.ones(len(sites), dtype=np.float64)
    weights /= weights.sum()
    selected_idx = int(rng.choice(len(sites), p=weights))
    selected = sites[selected_idx]

    min_c = int(sites_cfg.get("n_extra_min_clamp", n_extra))
    max_c = int(sites_cfg.get("n_extra_max_clamp", n_extra))
    min_c = max(0, min(min_c, int(n_extra)))
    max_c = max(min_c, max_c)
    budget_mode = str(sites_cfg.get("site_budget_mode", "requested")).lower()
    if budget_mode in ("random", "uniform"):
        budget = int(rng.integers(min_c, max_c + 1))
    else:
        budget = int(np.clip(n_extra, min_c, max_c))
    budget = min(budget, int(sites_cfg.get("max_per_site", budget)))
    return [selected], [budget], budget


def allocate_atoms_reference_joint(
    n_extra: int,
    sites: List[Dict[str, Any]],
    sites_cfg: dict,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[List[Dict[str, Any]], List[int], int, Dict[str, Any]]:
    """Sample one observed joint ``total + per-slot`` allocation pattern."""
    exact_site_counts = sites_cfg.get("reference_exact_site_counts") or {}
    if exact_site_counts:
        selected = {
            "n_extra": int(n_extra),
            "site_counts": {
                str(int(slot)): int(count)
                for slot, count in exact_site_counts.items()
                if int(count) > 0
            },
            "weight": 1.0,
            "source": "selected_exact_reference_record",
        }
        if sum(selected["site_counts"].values()) != int(n_extra):
            raise ValueError(
                "exact reference site allocation sums to "
                f"{sum(selected['site_counts'].values())}, expected {n_extra}"
            )
    else:
        profile_path = sites_cfg.get("reference_exit_profile")
        if not profile_path:
            raise ValueError("reference_joint requires reference_exit_profile")
        profile = load_scaffold_profile(profile_path)
        patterns = [
            pattern
            for pattern in profile.get("allocation_patterns", [])
            if int(pattern.get("n_extra", -1)) == int(n_extra)
        ]
        if not patterns:
            raise ValueError(f"no reference allocation pattern for n_extra={n_extra}")
        rng = rng or np.random.default_rng()
        weights = np.asarray(
            [max(float(pattern.get("weight", 1.0)), 0.0) for pattern in patterns],
            dtype=np.float64,
        )
        if not np.isfinite(weights).all() or weights.sum() <= 0:
            weights = np.ones(len(patterns), dtype=np.float64)
        selected = patterns[int(rng.choice(len(patterns), p=weights / weights.sum()))]

    site_by_slot = {}
    for site in sites:
        raw_slot = site.get("profile_slot", site.get("reference_exit_profile_slot"))
        if raw_slot is not None:
            site_by_slot[int(raw_slot)] = site
    active: List[Dict[str, Any]] = []
    counts: List[int] = []
    for raw_slot, raw_count in sorted(selected["site_counts"].items()):
        slot = int(raw_slot)
        count = int(raw_count)
        if slot not in site_by_slot:
            raise ValueError(f"reference allocation slot {slot} is absent from sites")
        active.append(site_by_slot[slot])
        counts.append(count)
    if sum(counts) != int(n_extra):
        raise ValueError(
            f"reference allocation sums to {sum(counts)}, expected {n_extra}"
        )
    return active, counts, int(n_extra), selected
