"""Configuration responsibilities extracted from scaffold_sites.py."""

from __future__ import annotations

from diffdynamic.chemistry.sites.state import Any, Dict, List, Optional

from diffdynamic.chemistry.sites.state import (
    DEFAULT_MURCKO_SITES_CFG,
)


def uses_site_budget_placement(sites_cfg: dict) -> bool:
    """grow/dynamic_locked 是否由位点分配逻辑决定 n_extra（非先验固定值）。"""
    return bool(sites_cfg.get("custom_regions")) or str(
        sites_cfg.get("per_site_count_mode", "split")
    ) in (
        "random_per_site",
        "sequential_random",
    )


def count_removed_sidechain_sites(
    attachment_sites: Optional[List[Dict[str, Any]]],
) -> int:
    """统计带有 removed_atom_indices 的真实侧链去除位点数（不含 exit_vector）。"""
    n = 0
    for site in attachment_sites or []:
        if str(site.get("site_kind", "murcko_sidechain")) == "exit_vector":
            continue
        if site.get("removed_atom_indices"):
            n += 1
    return n


def resolve_max_active_sites(
    sites_cfg: dict,
    attachment_sites: Optional[List[Dict[str, Any]]],
) -> int:
    """解析 max_active_sites：默认 / n_removed_sidechains = 去掉的侧链数量。"""
    raw = (sites_cfg or {}).get("max_active_sites", "n_removed_sidechains")
    n_removed = count_removed_sidechain_sites(attachment_sites)
    if raw is None or raw is True:
        return max(int(n_removed), 0)
    if isinstance(raw, str):
        key = raw.strip().lower()
        if key in (
            "",
            "auto",
            "all",
            "n_removed_sidechains",
            "removed_sidechains",
            "n_sidechains",
            "sidechains",
        ):
            return max(int(n_removed), 0)
        try:
            return max(int(raw), 0)
        except ValueError:
            return max(int(n_removed), 0)
    try:
        val = int(raw)
    except (TypeError, ValueError):
        return max(int(n_removed), 0)
    # 0 / 负数：视为「不限制浓缩」→ 用去掉的侧链数
    if val <= 0:
        return max(int(n_removed), 0)
    return val


def get_murcko_sites_cfg(scaffold_cfg: dict) -> dict:
    """合并 scaffold / grow 下的 murcko_sites 子配置。"""
    raw = {}
    if isinstance(scaffold_cfg.get("murcko_sites"), dict):
        raw.update(scaffold_cfg["murcko_sites"])
    grow = scaffold_cfg.get("grow")
    if isinstance(grow, dict) and isinstance(grow.get("murcko_sites"), dict):
        raw.update(grow["murcko_sites"])
    return {**DEFAULT_MURCKO_SITES_CFG, **raw}
