"""
Murcko 侧链位点：提取被去掉侧链的质心，并在骨架生长/动态锁骨架/Prudent 中
引导额外原子的初始放置（grow / dynamic_locked / prudent 共用）。

默认在去除位点质心附近用各向同性高斯云初始化（与从头生成 center+randn 同构），
单样本浓缩到少数去除位点，避免跨多位点/沿射线离散珠串。
"""

from __future__ import annotations

import json
from collections import deque
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from rdkit import Chem

from utils.gspt1_scaffold_prior import load_scaffold_profile

CST = timezone(timedelta(hours=8))

# 常见/复杂自由基重原子数先验（抬高单苯/稠环/联苯等）
DEFAULT_FRAGMENT_SIZES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 18, 20]
# 抬高 ≥6 原子片段概率（名义 P(≥6)≈0.89），降低 1–5 小侧链
DEFAULT_FRAGMENT_WEIGHTS = [
    0.02, 0.02, 0.02, 0.02, 0.03, 0.16, 0.10, 0.09, 0.08,
    0.12, 0.12, 0.10, 0.05, 0.04, 0.015, 0.01, 0.005,
]

DEFAULT_MURCKO_SITES_CFG = {
    'p_active': 0.5,
    'max_per_site': 20,
    'min_per_site': 0,
    'per_site_count_mode': 'split',
    # fragment_prior：按常见自由基规模一次写入；uniform：旧 Uniform[min,max]
    'per_site_add_mode': 'fragment_prior',
    'fragment_sizes': list(DEFAULT_FRAGMENT_SIZES),
    'fragment_weights': list(DEFAULT_FRAGMENT_WEIGHTS),
    # 与从头生成一致：质心 + N(0, σ)，σ≈1（sample_diffusion 里 center+randn）
    'jitter_std': 1.0,
    'jitter_mode': 'gaussian',  # gaussian|isotropic|clustered | directional（旧：沿位点射线离散外推）
    # 多原子时沿 anchor→centroid 径向外推（Å）：r = radial_step*atom_idx + radial_bulk*max(0, count-start)
    'radial_step': 0.25,
    'radial_bulk': 0.35,
    'radial_bulk_start': 4,
    # directional 模式：沿射线第 k 个原子再外推 directional_step Å
    'directional_step': 1.0,
    'directional_min_base_dist': 1.5,
    # 每个样本最多在几个去除位点上释放；默认 = 去掉的侧链数（n_removed_sidechains）
    'max_active_sites': 'n_removed_sidechains',
    'prefer_murcko_sites': True,  # 浓缩时优先真实侧链去除位点，而非 exit-vector
    'save_json': True,
    'overflow_mode': 'pocket_fallback',
    'dedup_dist': 0.5,
    # 虚拟 exit-vector：在环上可取代原子处补充释放点（缓解单侧链位点问题）
    'include_exit_vectors': False,
    'exit_vector_mode': 'aromatic_h',  # aromatic_h | ring_h | all_h
    'exit_vector_offset': 1.5,
    'exit_vector_dedup_dist': 1.2,
    # 位点分配为 0 时保留该位点原配体侧链（removed_atom_indices）；false=旧行为全剥侧链
    'preserve_zero_allocation_sidechains': True,
}


def uses_site_budget_placement(sites_cfg: dict) -> bool:
    """grow/dynamic_locked 是否由位点分配逻辑决定 n_extra（非先验固定值）。"""
    return str(sites_cfg.get('per_site_count_mode', 'split')) in (
        'random_per_site', 'sequential_random',
    )


def count_removed_sidechain_sites(attachment_sites: Optional[List[Dict[str, Any]]]) -> int:
    """统计带有 removed_atom_indices 的真实侧链去除位点数（不含 exit_vector）。"""
    n = 0
    for site in attachment_sites or []:
        if str(site.get('site_kind', 'murcko_sidechain')) == 'exit_vector':
            continue
        if site.get('removed_atom_indices'):
            n += 1
    return n


def resolve_max_active_sites(
    sites_cfg: dict,
    attachment_sites: Optional[List[Dict[str, Any]]],
) -> int:
    """解析 max_active_sites：默认 / n_removed_sidechains = 去掉的侧链数量。"""
    raw = (sites_cfg or {}).get('max_active_sites', 'n_removed_sidechains')
    n_removed = count_removed_sidechain_sites(attachment_sites)
    if raw is None or raw is True:
        return max(int(n_removed), 0)
    if isinstance(raw, str):
        key = raw.strip().lower()
        if key in (
            '', 'auto', 'all', 'n_removed_sidechains', 'removed_sidechains',
            'n_sidechains', 'sidechains',
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
    if isinstance(scaffold_cfg.get('murcko_sites'), dict):
        raw.update(scaffold_cfg['murcko_sites'])
    grow = scaffold_cfg.get('grow')
    if isinstance(grow, dict) and isinstance(grow.get('murcko_sites'), dict):
        raw.update(grow['murcko_sites'])
    return {**DEFAULT_MURCKO_SITES_CFG, **raw}


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

        sites.append({
            'site_id': len(sites),
            'anchor_scaffold_idx': anchor_idx,
            'anchor_pos': anchor_pos.tolist(),
            'centroid_pos': centroid.tolist(),
            'removed_atom_indices': [int(x) for x in comp],
            'removed_atom_count': len(comp),
            'removed_atom_positions': frag_pos.tolist(),
        })

    # 按质心距离去重
    if dedup_dist > 0 and len(sites) > 1:
        merged: List[Dict[str, Any]] = []
        for site in sites:
            c = np.array(site['centroid_pos'], dtype=np.float64)
            dup = False
            for kept in merged:
                kc = np.array(kept['centroid_pos'], dtype=np.float64)
                if np.linalg.norm(c - kc) < dedup_dist:
                    dup = True
                    kept['removed_atom_indices'] = list(
                        set(kept['removed_atom_indices']) | set(site['removed_atom_indices'])
                    )
                    kept['removed_atom_count'] = len(kept['removed_atom_indices'])
                    kept_pos = kept.get('removed_atom_positions', [])
                    site_pos = site.get('removed_atom_positions', [])
                    kept['removed_atom_positions'] = kept_pos + site_pos
                    break
            if not dup:
                site['site_id'] = len(merged)
                merged.append(site)
        sites = merged

    return sites


def extract_exit_vector_sites(
    mol,
    scaffold_indices: List[int],
    ref_pos_np: Optional[np.ndarray] = None,
    mode: str = 'aromatic_h',
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

    mode = str(mode or 'aromatic_h').lower()
    virtual: List[Dict[str, Any]] = []

    for idx in scaffold_indices:
        atom = mol.GetAtomWithIdx(int(idx))
        z = atom.GetAtomicNum()
        if z not in (6, 7):
            continue
        if atom.GetTotalNumHs() < 1:
            continue
        if mode == 'aromatic_h':
            if not atom.GetIsAromatic():
                continue
        elif mode == 'ring_h':
            if int(idx) not in ring_atoms:
                continue
        elif mode == 'all_h':
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

        virtual.append({
            'site_id': -1,
            'anchor_scaffold_idx': int(idx),
            'anchor_pos': anchor.tolist(),
            'centroid_pos': centroid.tolist(),
            'removed_atom_indices': [],
            'removed_atom_count': 0,
            'site_kind': 'exit_vector',
            'exit_vector_mode': mode,
        })

    # 与已有真实侧链位点去重
    kept_existing = list(existing_sites or [])
    existing_cent = [
        np.asarray(s['centroid_pos'], dtype=np.float64) for s in kept_existing
    ]
    merged_virtual: List[Dict[str, Any]] = []
    for site in virtual:
        c = np.asarray(site['centroid_pos'], dtype=np.float64)
        if any(np.linalg.norm(c - ec) < dedup_dist for ec in existing_cent):
            continue
        if any(
            np.linalg.norm(c - np.asarray(k['centroid_pos'], dtype=np.float64)) < dedup_dist
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
        sc.setdefault('site_kind', 'murcko_sidechain')
        sc['site_id'] = len(merged)
        merged.append(sc)
    for s in exit_sites:
        sc = dict(s)
        sc.setdefault('site_kind', 'exit_vector')
        sc['site_id'] = len(merged)
        merged.append(sc)
    return merged


def extract_reference_exit_vector_sites(
    mol,
    scaffold_indices: List[int],
    ref_pos_np: Optional[np.ndarray],
    profile: Dict[str, Any],
    offset: float = 1.5,
    allowed_slots: Optional[List[int]] = None,
) -> List[Dict[str, Any]]:
    """Create geometry-only virtual exits from a scaffold-local profile.

    ``profile`` contains only scaffold-local weights.  Coordinates come from
    the supplied scaffold pose and no reference target-side atom is copied.

    Exit directions use scaffold neighbors only.  A target-side neighbor must
    not influence the initialization cloud in strict scaffold mode.
    """
    if mol is None or not scaffold_indices:
        return []
    if int(profile.get('n_scaffold', 0)) != len(scaffold_indices):
        return []

    positions = _atom_positions(mol, ref_pos_np)
    scaffold_set = set(int(index) for index in scaffold_indices)
    weights = profile.get('exit_site_weights') or {}
    allowed = (
        {int(slot) for slot in allowed_slots}
        if allowed_slots is not None else None
    )
    sites: List[Dict[str, Any]] = []
    for raw_slot, raw_weight in weights.items():
        slot = int(raw_slot)
        if allowed is not None and slot not in allowed:
            continue
        if slot < 0 or slot >= len(scaffold_indices):
            continue
        atom_idx = int(scaffold_indices[slot])
        atom = mol.GetAtomWithIdx(atom_idx)
        anchor = positions[atom_idx]
        neighbour_indices = [
            n.GetIdx() for n in atom.GetNeighbors()
            if n.GetIdx() in scaffold_set
        ]
        if neighbour_indices:
            neighbour_centroid = positions[neighbour_indices].mean(axis=0)
            direction = anchor - neighbour_centroid
        else:
            direction = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-6:
            direction = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            direction = direction / norm
        sites.append({
            'site_id': -1,
            'anchor_scaffold_idx': atom_idx,
            'anchor_pos': anchor.tolist(),
            'centroid_pos': (anchor + float(offset) * direction).tolist(),
            'removed_atom_indices': [],
            'removed_atom_count': 0,
            'site_kind': 'reference_exit_vector',
            'profile_slot': slot,
            'site_selection_weight': max(float(raw_weight), 0.0),
            'exit_direction': direction.tolist(),
        })
    return sites


def _transfer_template_geometry(
    source_site: Dict[str, Any],
    target_site: Dict[str, Any],
    attachment_first: bool = True,
) -> Optional[List[List[float]]]:
    """Move a native target-side coordinate cloud to another scaffold exit."""
    raw_positions = source_site.get('removed_atom_positions') or []
    if not raw_positions:
        return None
    source_anchor = np.asarray(source_site.get('anchor_pos'), dtype=np.float64)
    target_anchor = np.asarray(target_site.get('anchor_pos'), dtype=np.float64)
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
        attachment_order = np.argsort(source_distances, kind='stable')
        source_positions = source_positions[attachment_order]
        attachment_index = 0
    source_direction = source_positions[attachment_index] - source_anchor
    target_direction = (
        np.asarray(target_site.get('centroid_pos'), dtype=np.float64)
        - target_anchor
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
        skew = np.array([
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ])
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
    template_atom_order: str = 'attachment_first',
) -> List[Dict[str, Any]]:
    """Merge profile exits, accumulating weight on an existing native site."""
    merged = [dict(site) for site in attachment_sites]
    for site in merged:
        weight = site.get('site_selection_weight')
        site['site_selection_weight'] = 1.0 if weight is None else weight

    for candidate in profile_sites:
        anchor_idx = candidate.get('anchor_scaffold_idx')
        same_anchor = next(
            (site for site in merged
             if site.get('anchor_scaffold_idx') == anchor_idx),
            None,
        )
        if same_anchor is not None:
            same_anchor['site_selection_weight'] = max(
                float(same_anchor.get('site_selection_weight', 1.0)),
                float(candidate.get('site_selection_weight', 0.0)),
            )
            same_anchor['profile_slot'] = candidate.get('profile_slot')
            same_anchor['reference_exit_profile_slot'] = candidate.get('profile_slot')
            continue

        centroid = np.asarray(candidate['centroid_pos'], dtype=np.float64)
        duplicate = any(
            np.linalg.norm(
                centroid - np.asarray(site['centroid_pos'], dtype=np.float64)
            ) < float(dedup_dist)
            for site in merged
        )
        if duplicate:
            continue
        candidate = dict(candidate)
        template_site = next(
            (
                site for site in merged
                if site.get('removed_atom_positions')
                and site.get('removed_atom_count', 0) > 0
            ),
            None,
        )
        if template_site is not None:
            transferred = _transfer_template_geometry(
                template_site,
                candidate,
                attachment_first=str(template_atom_order).lower()
                in ('attachment_first', 'attached_first', 'sorted'),
            )
            if transferred is not None:
                candidate['removed_atom_positions'] = transferred
                candidate['removed_atom_count'] = len(transferred)
                candidate['template_source_site_id'] = template_site.get('site_id')
        candidate['site_id'] = len(merged)
        merged.append(candidate)

    for site_id, site in enumerate(merged):
        site['site_id'] = site_id
    return merged


# 骨架选择入口（Murcko 与自定义并列）；选出 indices 后位点提取路径相同
SCAFFOLD_SOURCES_WITH_ATTACHMENT_SITES = (
    'auto_murcko',
    'auto_murcko_generic',
    'smarts',
    'custom',  # alias of smarts
    'atom_indices',
)


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
    source = str(scaffold_source or '').strip().lower()
    if source not in SCAFFOLD_SOURCES_WITH_ATTACHMENT_SITES:
        if logger:
            logger.info(
                f'[MurckoSites] scaffold_source={scaffold_source}，跳过侧链位点提取，使用口袋随机放置'
            )
        return []

    if mol is None or not scaffold_indices:
        return []

    include_ev = bool(sites_cfg.get('include_exit_vectors', False))
    ev_mode = str(sites_cfg.get('exit_vector_mode', 'aromatic_h'))
    profile_path = sites_cfg.get('reference_exit_profile')
    use_profile = bool(profile_path)
    strict_anchor_gaussian = bool(
        sites_cfg.get('strict_anchor_gaussian', False)
    )
    strict_fragment_gaussian = bool(
        sites_cfg.get('strict_fragment_gaussian', False)
    )
    fragment_gaussian = bool(sites_cfg.get('fragment_gaussian', False))
    strict_geometry_only = (
        strict_anchor_gaussian or strict_fragment_gaussian or fragment_gaussian
    )
    profile_tag = '_profile_exit' if use_profile else ''
    cache_suffix = (f'_ev_{ev_mode}' if include_ev else '') + profile_tag

    json_path = None
    if sites_cfg.get('save_json', True) and output_dir and ref_ligand_name:
        scaffold_dir = Path(output_dir) / 'scaffold'
        scaffold_dir.mkdir(parents=True, exist_ok=True)
        json_path = scaffold_dir / f'{ref_ligand_name}_murcko_sites{cache_suffix}.json'
        if json_path.exists() and not strict_geometry_only:
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                sites = data.get('attachment_sites', [])
                cache_ok = bool(sites)
                if include_ev:
                    cache_ok = cache_ok and any(
                        s.get('site_kind') == 'exit_vector' for s in sites
                    )
                if use_profile:
                    cache_ok = cache_ok and any(
                        s.get('site_kind') == 'reference_exit_vector'
                        or s.get('reference_exit_profile_slot') is not None
                        for s in sites
                    )
                if cache_ok:
                    if logger:
                        logger.info(f'[MurckoSites] 从缓存加载 {len(sites)} 个位点: {json_path}')
                    return sites
            except Exception as e:
                if logger:
                    logger.warning(f'[MurckoSites] 读取缓存失败: {e}')

    if strict_geometry_only and use_profile:
        # Do not even construct target-side attachment templates in strict
        # mode.  Profile exits below are derived from scaffold atoms only.
        sites = []
    else:
        sites = extract_murcko_attachment_sites(
            mol, scaffold_indices, ref_pos_np,
            dedup_dist=float(sites_cfg.get('dedup_dist', 0.5)),
        )
    for s in sites:
        s.setdefault('site_kind', 'murcko_sidechain')

    if logger:
        logger.info(f'[MurckoSites] 提取 {len(sites)} 个侧链质心位点')

    if include_ev:
        exit_sites = extract_exit_vector_sites(
            mol,
            scaffold_indices,
            ref_pos_np,
            mode=ev_mode,
            offset=float(sites_cfg.get('exit_vector_offset', 1.5)),
            existing_sites=sites,
            dedup_dist=float(sites_cfg.get('exit_vector_dedup_dist', 1.2)),
        )
        sites = merge_attachment_and_exit_vector_sites(sites, exit_sites)
        n_ev = sum(1 for s in sites if s.get('site_kind') == 'exit_vector')
        if logger:
            logger.info(
                f'[MurckoSites] exit_vector({ev_mode}): +{n_ev} 虚拟位点 → 合计 {len(sites)}'
            )

    if use_profile:
        try:
            profile = load_scaffold_profile(profile_path)
            profile_sites = extract_reference_exit_vector_sites(
                mol,
                scaffold_indices,
                ref_pos_np,
                profile,
                offset=float(sites_cfg.get('reference_exit_offset', 1.5)),
                allowed_slots=sites_cfg.get('reference_exit_slots'),
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
                    dedup_dist=float(
                        sites_cfg.get('reference_exit_dedup_dist', 1.2)
                    ),
                    template_atom_order=str(
                        sites_cfg.get(
                            'reference_exit_template_order', 'attachment_first'
                        )
                    ),
                )
            if bool(sites_cfg.get('reference_exit_only', False)):
                sites = [
                    site for site in merged_sites
                    if site.get('site_kind') == 'reference_exit_vector'
                    or site.get('reference_exit_profile_slot') is not None
                ]
                for site_id, site in enumerate(sites):
                    site['site_id'] = site_id
            else:
                sites = merged_sites
            if logger:
                logger.info(
                    f'[MurckoSites] reference exit profile: '
                    f'{len(profile_sites)} exits, matched_refs='
                    f'{profile.get("matched_reference_records", 0)}'
                )
        except Exception as exc:
            if logger:
                logger.warning(
                    f'[MurckoSites] reference exit profile unavailable: {exc}'
                )

    if json_path and sites:
        payload = {
            'ligand_name': ref_ligand_name,
            'n_scaffold': len(scaffold_indices),
            'scaffold_indices': [int(i) for i in scaffold_indices],
            'scaffold_source': scaffold_source,
            'include_exit_vectors': include_ev,
            'exit_vector_mode': ev_mode if include_ev else None,
            'reference_exit_profile': str(profile_path) if use_profile else None,
            'attachment_sites': sites,
            'extracted_at': datetime.now(CST).isoformat(),
        }
        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            if logger:
                logger.info(f'[MurckoSites] 位点已保存: {json_path}')
        except Exception as e:
            if logger:
                logger.warning(f'[MurckoSites] 保存位点 JSON 失败: {e}')

    return sites


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
    max_per_site = int(sites_cfg.get('max_per_site', 20))
    min_per_site = int(sites_cfg.get('min_per_site', 0))
    lo = max(1, min_per_site) if min_per_site > 0 else 1
    hi = min(int(room), max_per_site)
    if hi < lo:
        return 0

    sizes = list(sites_cfg.get('fragment_sizes') or DEFAULT_FRAGMENT_SIZES)
    weights = list(sites_cfg.get('fragment_weights') or DEFAULT_FRAGMENT_WEIGHTS)
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
    mode = str(sites_cfg.get('per_site_add_mode', 'fragment_prior')).lower()
    max_per_site = int(sites_cfg.get('max_per_site', 20))
    min_per_site = int(sites_cfg.get('min_per_site', 0))
    min_per_site = max(0, min(min_per_site, max_per_site))
    hi_add = min(max_per_site, int(room))
    if mode in ('uniform', 'uniform_random'):
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
    p_active = float(sites_cfg.get('p_active', 0.5))
    max_per_site = int(sites_cfg.get('max_per_site', 20))
    min_per_site = int(sites_cfg.get('min_per_site', 0))
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
        counts[0] = sample_per_site_add(
            rng, sites_cfg, max_per_site, allow_zero_uniform=False,
        ) or 1
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
    p_active = float(sites_cfg.get('p_active', 0.5))
    max_per_site = int(sites_cfg.get('max_per_site', 20))
    min_per_site = int(sites_cfg.get('min_per_site', 0))
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

    ordered = sorted(sites, key=lambda s: int(s.get('site_id', 0)))
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
    max_per_site = int(sites_cfg.get('max_per_site', 20))
    return len(sites) * max_per_site


def cap_n_extra_for_sites(
    n_extra: int,
    attachment_sites: List[Dict[str, Any]],
    sites_cfg: dict,
) -> int:
    """overflow_mode=cap/drop 时，将 n_extra 限制在位点总容量内。"""
    overflow_mode = str(sites_cfg.get('overflow_mode', 'pocket_fallback'))
    if overflow_mode not in ('cap', 'drop') or not attachment_sites:
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
        [max(float(site.get('site_selection_weight', 1.0)), 0.0) for site in sites],
        dtype=np.float64,
    )
    if not np.isfinite(weights).all() or weights.sum() <= 0:
        weights = np.ones(len(sites), dtype=np.float64)
    weights /= weights.sum()
    selected_idx = int(rng.choice(len(sites), p=weights))
    selected = sites[selected_idx]

    min_c = int(sites_cfg.get('n_extra_min_clamp', n_extra))
    max_c = int(sites_cfg.get('n_extra_max_clamp', n_extra))
    min_c = max(0, min(min_c, int(n_extra)))
    max_c = max(min_c, max_c)
    budget_mode = str(sites_cfg.get('site_budget_mode', 'requested')).lower()
    if budget_mode in ('random', 'uniform'):
        budget = int(rng.integers(min_c, max_c + 1))
    else:
        budget = int(np.clip(n_extra, min_c, max_c))
    budget = min(budget, int(sites_cfg.get('max_per_site', budget)))
    return [selected], [budget], budget


def allocate_atoms_reference_joint(
    n_extra: int,
    sites: List[Dict[str, Any]],
    sites_cfg: dict,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[List[Dict[str, Any]], List[int], int, Dict[str, Any]]:
    """Sample one observed joint ``total + per-slot`` allocation pattern."""
    profile_path = sites_cfg.get('reference_exit_profile')
    if not profile_path:
        raise ValueError('reference_joint requires reference_exit_profile')
    profile = load_scaffold_profile(profile_path)
    patterns = [
        pattern for pattern in profile.get('allocation_patterns', [])
        if int(pattern.get('n_extra', -1)) == int(n_extra)
    ]
    if not patterns:
        raise ValueError(
            f'no reference allocation pattern for n_extra={n_extra}'
        )
    rng = rng or np.random.default_rng()
    weights = np.asarray(
        [max(float(pattern.get('weight', 1.0)), 0.0) for pattern in patterns],
        dtype=np.float64,
    )
    if not np.isfinite(weights).all() or weights.sum() <= 0:
        weights = np.ones(len(patterns), dtype=np.float64)
    selected = patterns[int(rng.choice(len(patterns), p=weights / weights.sum()))]

    site_by_slot = {}
    for site in sites:
        raw_slot = site.get(
            'profile_slot', site.get('reference_exit_profile_slot')
        )
        if raw_slot is not None:
            site_by_slot[int(raw_slot)] = site
    active: List[Dict[str, Any]] = []
    counts: List[int] = []
    for raw_slot, raw_count in sorted(selected['site_counts'].items()):
        slot = int(raw_slot)
        count = int(raw_count)
        if slot not in site_by_slot:
            raise ValueError(
                f'reference allocation slot {slot} is absent from sites'
            )
        active.append(site_by_slot[slot])
        counts.append(count)
    if sum(counts) != int(n_extra):
        raise ValueError(
            f'reference allocation sums to {sum(counts)}, expected {n_extra}'
        )
    return active, counts, int(n_extra), selected


def _attachment_direction(site: Dict[str, Any]) -> Optional[np.ndarray]:
    """返回 normalize(centroid - anchor)；缺锚点或退化时返回 None。"""
    anchor = site.get('anchor_pos')
    centroid = site.get('centroid_pos')
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
    anchor = site.get('anchor_pos')
    centroid = site.get('centroid_pos')
    if anchor is None or centroid is None:
        centroid = np.array(site['centroid_pos'], dtype=np.float64)
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
    centroid = np.array(site['centroid_pos'], dtype=np.float64)
    r = float(radial_step) * float(atom_idx) + float(radial_bulk) * max(
        0.0, float(site_count) - float(radial_bulk_start),
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
    anchor = np.asarray(site['anchor_pos'], dtype=np.float64)
    raw_direction = site.get('exit_direction')
    if raw_direction is None:
        raw_direction = _attachment_direction(site)
    if raw_direction is None:
        raw_direction = [1.0, 0.0, 0.0]
    direction = np.asarray(raw_direction, dtype=np.float64)
    direction_norm = float(np.linalg.norm(direction))
    direction = direction / max(direction_norm, 1e-8)

    raw_tangent = site.get('tangent_direction')
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

    raw_binormal = site.get('binormal_direction')
    if raw_binormal is None:
        raw_binormal = np.cross(direction, tangent)
    binormal = np.asarray(raw_binormal, dtype=np.float64)
    binormal_norm = float(np.linalg.norm(binormal))
    binormal = binormal / max(binormal_norm, 1e-8)

    slot = str(site.get('profile_slot', ''))
    slot_offsets = sites_cfg.get('strict_gaussian_slot_offsets', {}) or {}
    slot_tangent_shifts = sites_cfg.get(
        'strict_gaussian_slot_tangent_shifts', {}
    ) or {}
    slot_binormal_shifts = sites_cfg.get(
        'strict_gaussian_slot_binormal_shifts', {}
    ) or {}
    offset = float(slot_offsets.get(
        slot, sites_cfg.get('strict_gaussian_center_offset', 1.5)
    ))
    tangent_shift = float(slot_tangent_shifts.get(
        slot, sites_cfg.get('strict_gaussian_tangent_shift', 0.0)
    ))
    binormal_shift = float(slot_binormal_shifts.get(
        slot, sites_cfg.get('strict_gaussian_binormal_shift', 0.0)
    ))
    sigma = float(sites_cfg.get('strict_gaussian_sigma', jitter_std))
    if sigma <= 0.0:
        raise ValueError('strict_gaussian_sigma must be positive')

    center = (
        anchor
        + offset * direction
        + tangent_shift * tangent
        + binormal_shift * binormal
    )
    return [
        center + rng.normal(0.0, sigma, size=3)
        for _ in range(int(count))
    ]


def _site_frame(site: Dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return a scaffold-derived exit frame without reading target-side atoms."""
    anchor = np.asarray(site['anchor_pos'], dtype=np.float64)
    raw_direction = site.get('exit_direction') or _attachment_direction(site)
    direction = np.asarray(raw_direction or [1.0, 0.0, 0.0], dtype=np.float64)
    direction /= max(float(np.linalg.norm(direction)), 1e-8)

    raw_tangent = site.get('tangent_direction') or [0.0, 1.0, 0.0]
    tangent = np.asarray(raw_tangent, dtype=np.float64)
    tangent -= np.dot(tangent, direction) * direction
    if float(np.linalg.norm(tangent)) <= 1e-8:
        basis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(float(np.dot(basis, direction))) > 0.9:
            basis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        tangent = basis - np.dot(basis, direction) * direction
    tangent /= max(float(np.linalg.norm(tangent)), 1e-8)

    raw_binormal = site.get('binormal_direction')
    binormal = np.asarray(
        raw_binormal or np.cross(direction, tangent), dtype=np.float64
    )
    binormal /= max(float(np.linalg.norm(binormal)), 1e-8)
    return anchor, direction, tangent, binormal


def _fragment_atom_keys(fragment: Dict[str, Any]) -> List[str]:
    keys: List[str] = []
    for key, count in sorted((fragment.get('atom_counts') or {}).items()):
        keys.extend([str(key)] * int(count))
    return keys


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
        values.append(
            radius * (
                np.cos(theta) * tangent + np.sin(theta) * binormal
            )
        )
    return np.asarray(values, dtype=np.float64)


def _fragment_ring_atom_count(
    fragment: Dict[str, Any],
    total: int,
    kind: str,
) -> int:
    ring_sizes = [
        int(value) for value in fragment.get('ring_sizes', [])
        if int(value) >= 3
    ]
    if kind in ('aromatic_ring', 'aliphatic_ring'):
        size = ring_sizes[0] if ring_sizes else total
        return max(3, min(int(total), int(size)))
    if kind in ('fused_aromatic', 'fused_ring'):
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
    aromatic = [key for key in atom_keys if key.endswith('|1')]
    non_aromatic = [key for key in atom_keys if not key.endswith('|1')]
    priority = {'C': 0, 'N': 1, 'S': 2, 'P': 3, 'O': 4, 'F': 5, 'Cl': 6}
    non_aromatic.sort(
        key=lambda key: priority.get(str(key).rsplit('|', 1)[0], 99)
    )
    ordered = aromatic + non_aromatic
    ring_keys = ordered[:int(ring_count)]
    extra_keys = ordered[int(ring_count):]
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
) -> tuple[list[np.ndarray], List[str]]:
    """Create randomized motif geometry and aligned type hints.

    These are generic motif clouds, not reference graphs: no bond list,
    reference atom index, or target-side coordinate enters this function.
    The resulting coordinates are still forward-noised before denoising.
    """
    kind = str(fragment.get('kind', 'chain'))
    atom_keys = _fragment_atom_keys(fragment)
    if not atom_keys:
        return [], []
    jitter = float(sites_cfg.get('fragment_geometry_sigma', 0.16))
    ring_radius = float(sites_cfg.get('fragment_ring_radius', 1.38))
    angle = float(rng.uniform(0.0, 2.0 * np.pi))
    rot_t = np.cos(angle) * tangent + np.sin(angle) * binormal
    rot_b = -np.sin(angle) * tangent + np.cos(angle) * binormal
    total = len(atom_keys)

    if kind in ('aromatic_ring', 'aliphatic_ring'):
        radius = ring_radius if kind == 'aromatic_ring' else ring_radius * 0.92
        ring_count = _fragment_ring_atom_count(fragment, total, kind)
        ring_keys, extra_keys = _split_fragment_ring_keys(
            atom_keys, ring_count, rng
        )
        local_ring = _regular_polygon(
            ring_count, radius, direction, rot_t, np.pi
        )
        local_points = [point for point in local_ring]
        for index, point in enumerate(extra_keys):
            local_points.append(
                local_ring[index % ring_count]
                + (1.35 + 0.20 * (index // ring_count)) * direction
                + 0.25 * ((-1) ** index) * rot_t
            )
        local = np.asarray(local_points, dtype=np.float64)
        atom_keys = ring_keys + extra_keys
    elif kind in ('fused_aromatic', 'fused_ring') and total >= 8:
        # Generic fused rings with shared positions.  The exact reference
        # fusion topology is intentionally not retained.
        ring_sizes = [
            int(value) for value in fragment.get('ring_sizes', [])
            if int(value) >= 3
        ] or [6, 5]
        first_size = ring_sizes[0]
        second_size = ring_sizes[1] if len(ring_sizes) > 1 else 5
        first = _regular_polygon(
            first_size, ring_radius, direction, rot_t, np.pi
        )
        second_center = 1.55 * rot_t
        second = second_center + _regular_polygon(
            second_size, ring_radius * 0.96, direction, rot_t, np.pi + 0.35
        )
        ring_count = _fragment_ring_atom_count(fragment, total, kind)
        ring_keys, extra_keys = _split_fragment_ring_keys(
            atom_keys, ring_count, rng
        )
        local_points = [first[0], first[1]]
        local_points.extend(first[2:])
        local_points.extend(second[2:])
        local_points = local_points[:ring_count]
        while len(local_points) < ring_count:
            extra_index = len(local_points)
            local_points.append(
                second_center
                + (extra_index + 1.0) * 0.75 * rot_t
                + 0.35 * direction
            )
        for index, point in enumerate(extra_keys):
            local_points.append(
                local_points[index % ring_count]
                + (1.30 + 0.20 * (index // ring_count)) * direction
                + 0.25 * ((-1) ** index) * rot_t
            )
        local = np.asarray(local_points[:total], dtype=np.float64)
        atom_keys = ring_keys + extra_keys
    elif kind == 'carbonyl' and total >= 2:
        carbonyl_axis = np.cos(angle) * rot_t + np.sin(angle) * rot_b
        local = np.zeros((total, 3), dtype=np.float64)
        local[0] = -0.615 * carbonyl_axis
        local[1] = 0.615 * carbonyl_axis
        for index in range(2, total):
            local[index] = (
                (index - 1) * 1.25 * direction
                + 0.35 * ((-1) ** index) * rot_t
            )
    elif kind == 'hetero_atom' or total == 1:
        local = np.zeros((total, 3), dtype=np.float64)
        for index in range(1, total):
            local[index] = 0.45 * index * rot_t
    else:
        local = np.zeros((total, 3), dtype=np.float64)
        for index in range(total):
            local[index] = (
                (index - 0.5 * (total - 1)) * 1.48 * direction
                + 0.42 * ((-1) ** index) * rot_t
            )

    positions = [
        center + offset + rng.normal(0.0, jitter, size=3)
        for offset in local
    ]
    if kind == 'chain':
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
    order = np.argsort(projections, kind='stable')
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
    profile_path = sites_cfg.get('fragment_prior_profile') or sites_cfg.get(
        'reference_exit_profile'
    )
    if not profile_path:
        raise ValueError(
            'fragment_gaussian requires fragment_prior_profile'
        )
    profile = load_scaffold_profile(profile_path)
    n_extra = int(sum(int(value) for value in counts))
    allocation = {
        str(site.get('profile_slot')): int(count)
        for site, count in zip(active_sites, counts)
        if int(count) > 0
    }
    candidates = []
    for pattern in profile.get('fragment_patterns', []):
        if int(pattern.get('n_extra', -1)) != n_extra:
            continue
        pattern_counts = {}
        for slot, fragments in (pattern.get('site_fragments') or {}).items():
            pattern_counts[str(slot)] = sum(
                sum(int(value) for value in (fragment.get('atom_counts') or {}).values())
                for fragment in fragments
            )
        if pattern_counts == allocation:
            candidates.append(pattern)
    if not candidates:
        raise ValueError(
            f'no fragment pattern matches n_extra={n_extra} '
            f'allocation={allocation}'
        )
    weights = np.asarray([
        max(float(pattern.get('weight', 0.0)), 0.0)
        for pattern in candidates
    ], dtype=np.float64)
    selected = candidates[int(rng.choice(
        len(candidates), p=weights / max(float(weights.sum()), 1e-12)
    ))]

    positions: List[np.ndarray] = []
    type_hints: List[str] = []
    layout: List[Dict[str, Any]] = []
    fragment_spacing = float(sites_cfg.get('fragment_center_spacing', 2.15))
    center_offset = float(sites_cfg.get('strict_fragment_center_offset', 1.55))
    lateral_sigma = float(sites_cfg.get('fragment_center_lateral_sigma', 0.45))
    attachment_distance = float(
        sites_cfg.get('fragment_attachment_distance', 1.55)
    )
    attachment_spacing = float(
        sites_cfg.get('fragment_attachment_spacing', fragment_spacing)
    )
    for site, count in zip(active_sites, counts):
        if int(count) <= 0:
            continue
        slot = str(site.get('profile_slot'))
        fragments = list(
            (selected.get('site_fragments') or {}).get(slot) or []
        )
        rng.shuffle(fragments)
        anchor, direction, tangent, binormal = _site_frame(site)
        site_center = anchor + center_offset * direction
        for fragment_index, fragment in enumerate(fragments):
            lateral = rng.normal(0.0, lateral_sigma, size=2)
            center = (
                site_center
                + fragment_index * fragment_spacing * direction
                + lateral[0] * tangent
                + lateral[1] * binormal
            )
            fragment_positions, fragment_types = _generic_fragment_geometry(
                fragment, center, direction, tangent, binormal, rng, sites_cfg
            )
            fragment_positions, fragment_types, projected_distance = (
                _align_fragment_to_anchor(
                    fragment_positions,
                    fragment_types,
                    anchor,
                    direction,
                    attachment_distance + fragment_index * attachment_spacing,
                )
            )
            positions.extend(fragment_positions)
            type_hints.extend(fragment_types)
            layout.append({
                'profile_slot': slot,
                'kind': str(fragment.get('kind', 'chain')),
                'ring_sizes': [int(v) for v in fragment.get('ring_sizes', [])],
                'atom_count': len(fragment_types),
                'atom_types': list(fragment_types),
                'anchor_projection_distance': projected_distance,
            })

    if len(positions) != n_extra or len(type_hints) != n_extra:
        raise ValueError(
            f'fragment placement produced {len(positions)} atoms, expected {n_extra}'
        )
    return positions, type_hints, layout


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
    anchor = np.asarray(site['anchor_pos'], dtype=np.float64)
    r = float(radial_step) * (float(atom_idx) + 1.0) + float(radial_bulk) * max(
        0.0, float(atom_idx) + 1.0 - float(radial_bulk_start),
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
    anchor = np.array(site['anchor_pos'], dtype=np.float64)
    orig_positions = site.get('removed_atom_positions')

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
        return float('inf')
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
    min_protein_dist = float(sites_cfg.get('pocket_min_protein_dist', 1.5))
    min_point_dist = float(sites_cfg.get('pocket_min_point_dist', 0.9))
    max_anchor_dist = float(sites_cfg.get('pocket_max_anchor_dist', 10.0))
    candidates = int(sites_cfg.get('pocket_growth_candidates', 128))
    bond_min = float(sites_cfg.get('pocket_growth_step_min', 1.25))
    bond_max = float(sites_cfg.get('pocket_growth_step_max', 1.55))
    parents = placed[-8:] if placed else [anchor]
    best = None
    best_score = -float('inf')
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
    direction = _attachment_direction({
        'anchor_pos': anchor,
        'centroid_pos': placed[-1] if placed else anchor + [1.0, 0.0, 0.0],
    })
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
    min_protein_dist = float(sites_cfg.get('pocket_min_protein_dist', 1.5))
    min_point_dist = float(sites_cfg.get('pocket_min_point_dist', 0.9))
    max_anchor_dist = float(sites_cfg.get('pocket_max_anchor_dist', 10.0))
    iterations = int(sites_cfg.get('pocket_relax_iterations', 120))
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
                    direction / norm if norm > 1e-8
                    else _random_unit_vector(rng)
                )
                relaxed[index] += direction * (
                    min_protein_dist - distance + 0.03
                )
                moved = True
        for left in range(len(relaxed)):
            for right in range(left + 1, len(relaxed)):
                delta = relaxed[left] - relaxed[right]
                distance = float(np.linalg.norm(delta))
                if distance >= min_point_dist:
                    continue
                direction = (
                    delta / distance if distance > 1e-8
                    else _random_unit_vector(rng)
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
    anchor = np.asarray(site['anchor_pos'], dtype=np.float64)
    local_protein = protein_positions
    if protein_positions is not None and protein_positions.size > 0:
        local_radius = float(sites_cfg.get('pocket_local_radius', 13.0))
        local_mask = np.linalg.norm(
            protein_positions - anchor, axis=1
        ) <= local_radius
        if local_mask.any():
            local_protein = protein_positions[local_mask]
    template = [
        np.asarray(position, dtype=np.float64)
        for position in (site.get('removed_atom_positions') or [])[:count]
    ]
    placed = [
        position + rng.normal(0.0, float(jitter_std), size=3)
        for position in template
    ]
    while len(placed) < count:
        placed.append(_grow_pocket_position(
            anchor, placed, local_protein, rng, sites_cfg
        ))
    array = _relax_positions_from_protein(
        np.asarray(placed), anchor, local_protein, rng, sites_cfg
    )
    return [array[index] for index in range(len(array))]


def _is_gaussian_jitter(jitter_mode: str) -> bool:
    mode = str(jitter_mode or 'gaussian').lower()
    return mode in ('gaussian', 'isotropic', 'clustered', 'denovo', 'random')


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
            i for i in indices
            if str(active_sites[i].get('site_kind', 'murcko_sidechain')) != 'exit_vector'
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
            if int(a.get('count', 0) or 0) <= 0:
                continue
            sid = a.get('site_id')
            if sid is not None:
                allocated_ids.add(int(sid))

    zero: List[Dict[str, Any]] = []
    for site in attachment_sites or []:
        sid = site.get('site_id')
        if sid is None:
            continue
        if int(sid) in allocated_ids:
            continue
        removed = site.get('removed_atom_indices') or []
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
    n_classes = int(ref_log_v.shape[-1]) if ref_log_v is not None and ref_log_v.ndim >= 2 else 0
    indices: List[int] = []
    seen = set()
    for site in zero_sites or []:
        for idx in site.get('removed_atom_indices') or []:
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
    enabled = bool(cfg.get('preserve_zero_allocation_sidechains', True))
    zero_ids = [int(x) for x in (meta.get('zero_allocation_site_ids') or [])]
    if not enabled:
        device = ref_pos.device
        n_classes = int(ref_log_v.shape[-1]) if ref_log_v is not None and ref_log_v.ndim >= 2 else 0
        return (
            torch.zeros(0, 3, device=device),
            torch.zeros(0, n_classes, device=device),
            [],
            zero_ids,
        )

    # 优先用 meta 已写好的索引（build_extra_atom_positions 填充）
    preserved_from_meta = meta.get('preserved_atom_indices')
    if preserved_from_meta is not None:
        indices = [int(i) for i in preserved_from_meta]
        if not indices:
            device = ref_pos.device
            n_classes = int(ref_log_v.shape[-1]) if ref_log_v is not None and ref_log_v.ndim >= 2 else 0
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
        attachment_sites, meta.get('site_allocation'),
    )
    zero_ids = [int(s.get('site_id')) for s in zero_sites if s.get('site_id') is not None]
    pos, log_v, indices = gather_preserved_sidechain_atoms(zero_sites, ref_pos, ref_log_v)
    return pos, log_v, indices, zero_ids


def _fill_zero_allocation_preserve_meta(
    meta: Dict[str, Any],
    attachment_sites: List[Dict[str, Any]],
    site_allocation: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """根据最终 site_allocation 填写零分配位点与可保留原子索引。"""
    alloc = site_allocation if site_allocation is not None else meta.get('site_allocation')
    zero_sites = collect_zero_allocation_sites(attachment_sites, alloc)
    zero_ids: List[int] = []
    preserved: List[int] = []
    seen = set()
    for site in zero_sites:
        sid = site.get('site_id')
        if sid is not None:
            zero_ids.append(int(sid))
        for idx in site.get('removed_atom_indices') or []:
            i = int(idx)
            if i in seen:
                continue
            seen.add(i)
            preserved.append(i)
    meta['zero_allocation_site_ids'] = zero_ids
    meta['preserved_atom_indices'] = preserved
    meta['n_preserved_atoms'] = len(preserved)
    return meta


def init_extra_positions_at_sites(
    active_sites: List[Dict[str, Any]],
    counts: List[int],
    device,
    jitter_std: float = 1.0,
    jitter_mode: str = 'gaussian',
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
    cfg = sites_cfg or {}
    radial_step = float(cfg.get('radial_step', DEFAULT_MURCKO_SITES_CFG['radial_step']))
    radial_bulk = float(cfg.get('radial_bulk', DEFAULT_MURCKO_SITES_CFG['radial_bulk']))
    radial_bulk_start = int(cfg.get('radial_bulk_start', DEFAULT_MURCKO_SITES_CFG['radial_bulk_start']))
    directional_step = float(cfg.get('directional_step', DEFAULT_MURCKO_SITES_CFG['directional_step']))
    min_base_dist = float(cfg.get(
        'directional_min_base_dist', DEFAULT_MURCKO_SITES_CFG['directional_min_base_dist'],
    ))
    rng = rng or np.random.default_rng()
    positions = []
    mode = str(jitter_mode or 'gaussian').lower()
    use_gaussian = mode in (
        'gaussian', 'isotropic', 'clustered', 'denovo', 'random',
    )
    use_native = mode in ('native', 'native_template')
    orig_min_dist = float(cfg.get('original_min_dist', 2.0))
    orig_max_dist = float(cfg.get('original_max_dist', 4.0))
    for site, cnt in zip(active_sites, counts):
        if cnt <= 0:
            continue
        if mode == 'strict_anchor_gaussian':
            positions.extend(_strict_anchor_gaussian_site_positions(
                site,
                cnt,
                jitter_std,
                rng,
                cfg,
            ))
            continue
        if mode in ('pocket_aware_template', 'pocket_aware_growth'):
            positions.extend(_pocket_aware_site_positions(
                site,
                cnt,
                protein_positions,
                jitter_std,
                rng,
                cfg,
            ))
            continue
        for atom_i in range(cnt):
            if use_gaussian:
                positions.append(_gaussian_site_position(
                    site, jitter_std, rng,
                    atom_idx=atom_i, site_count=cnt,
                    radial_step=radial_step,
                    radial_bulk=radial_bulk,
                    radial_bulk_start=radial_bulk_start,
                ))
            elif mode in ('anchor_radial', 'legacy_radial'):
                positions.append(_anchor_radial_site_position(
                    site, jitter_std, rng,
                    atom_idx=atom_i, site_count=cnt,
                    radial_step=radial_step,
                    radial_bulk=radial_bulk,
                    radial_bulk_start=radial_bulk_start,
                ))
            elif use_native:
                positions.append(_original_site_position(
                    site, jitter_std, rng,
                    atom_idx=atom_i, site_count=cnt,
                    radial_step=radial_step,
                    min_dist=orig_min_dist, max_dist=orig_max_dist,
                ))
            elif str(jitter_mode).lower() == 'directional':
                positions.append(_directional_site_position(
                    site, atom_i, jitter_std, rng,
                    directional_step=directional_step,
                    min_base_dist=min_base_dist,
                ))
            elif mode == 'hybrid':
                if site.get('removed_atom_positions'):
                    positions.append(_original_site_position(
                        site, jitter_std, rng,
                        atom_idx=atom_i, site_count=cnt,
                        radial_step=radial_step,
                        min_dist=orig_min_dist, max_dist=orig_max_dist,
                    ))
                else:
                    positions.append(_directional_site_position(
                        site, atom_i, jitter_std, rng,
                        directional_step=directional_step,
                        min_base_dist=min_base_dist,
                    ))
            else:
                positions.append(_original_site_position(
                    site, jitter_std, rng,
                    atom_idx=atom_i, site_count=cnt,
                    radial_step=radial_step,
                    min_dist=orig_min_dist, max_dist=orig_max_dist,
                ))

    if not positions:
        return torch.zeros(0, 3, device=device)

    arr = np.array(positions, dtype=np.float32)
    return torch.tensor(arr, dtype=torch.float32, device=device)


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
    n_extra_requested = n_extra
    strict_anchor_gaussian = bool(
        sites_cfg.get('strict_anchor_gaussian', False)
    )
    strict_fragment_gaussian = bool(
        sites_cfg.get('strict_fragment_gaussian', False)
    )
    fragment_gaussian = bool(sites_cfg.get('fragment_gaussian', False))
    fragment_geometry = strict_fragment_gaussian or fragment_gaussian
    overflow_mode = str(sites_cfg.get('overflow_mode', 'pocket_fallback'))
    meta: Dict[str, Any] = {
        'placement': 'pocket_fallback',
        'site_allocation': None,
        'overflow_count': 0,
        'n_extra_requested': n_extra_requested,
        'n_extra_effective': n_extra_requested,
        'overflow_mode': overflow_mode,
        'strict_fragment_gaussian': strict_fragment_gaussian,
        'fragment_gaussian': fragment_gaussian,
        'fragment_type_hints': [],
        'fragment_layout': [],
    }

    if n_extra <= 0 and not uses_site_budget_placement(sites_cfg):
        meta['n_extra_effective'] = 0
        meta['site_allocation'] = []
        return torch.zeros(0, 3, device=device), _fill_zero_allocation_preserve_meta(
            meta, attachment_sites or [], [],
        )

    if not attachment_sites:
        if strict_anchor_gaussian or fragment_geometry:
            raise ValueError(
                'strict scaffold Gaussian modes require scaffold exit anchor sites'
            )
        pos = fallback_center.unsqueeze(0).expand(n_extra, -1) + \
              torch.randn(n_extra, 3, device=device) * fallback_noise_scale
        return pos, _fill_zero_allocation_preserve_meta(meta, [], [])

    p_active = float(sites_cfg.get('p_active', 0.5))
    max_per_site = int(sites_cfg.get('max_per_site', 20))
    jitter_std = float(sites_cfg.get('jitter_std', 1.0))
    jitter_mode = str(sites_cfg.get('jitter_mode', 'gaussian'))
    count_mode = str(sites_cfg.get('per_site_count_mode', 'split'))
    max_active_sites = resolve_max_active_sites(sites_cfg, attachment_sites)
    prefer_murcko = bool(sites_cfg.get('prefer_murcko_sites', True))
    meta['n_removed_sidechain_sites'] = count_removed_sidechain_sites(attachment_sites)

    if overflow_mode in ('cap', 'drop'):
        p_active = float(sites_cfg.get('p_active', 0.5))

    joint_pattern = None
    if str(sites_cfg.get('site_selection_mode', '')).lower() == 'reference_joint':
        active, counts, n_extra_eff, joint_pattern = allocate_atoms_reference_joint(
            n_extra,
            attachment_sites,
            sites_cfg,
            rng=rng,
        )
        overflow = 0
        meta['n_extra_effective'] = n_extra_eff
        meta['per_site_count_mode'] = 'reference_joint'
        meta['site_selection_mode'] = 'reference_joint'
        meta['reference_allocation_pattern'] = joint_pattern
    elif count_mode in ('random_per_site', 'sequential_random'):
        min_c = int(sites_cfg.get('n_extra_min_clamp', 0))
        max_c = int(sites_cfg.get('n_extra_max_clamp', max(n_extra, 1)))
        cap_hi = cap_n_extra_for_sites(max_c, attachment_sites, sites_cfg)
        max_c = min(max_c, cap_hi) if cap_hi > 0 else max_c
        min_c = min(min_c, max_c)
        alloc_fn = (
            allocate_atoms_sequential_random
            if count_mode == 'sequential_random'
            else allocate_atoms_random_per_site
        )
        active, counts, n_extra_eff = alloc_fn(
            attachment_sites, sites_cfg,
            n_extra_bounds=(min_c, max_c),
            rng=rng,
        )
        overflow = 0
        meta['n_extra_effective'] = n_extra_eff
        meta['per_site_count_mode'] = count_mode
        if n_extra_eff <= 0:
            meta['site_allocation'] = []
            return torch.zeros(0, 3, device=device), _fill_zero_allocation_preserve_meta(
                meta, attachment_sites, [],
            )
        if logger:
            logger.info(
                f'[MurckoSites] {count_mode}: 各位点分配 {counts} → n_extra={n_extra_eff} '
                f'(budget∈[{min_c},{max_c}])'
            )
    elif str(sites_cfg.get('site_selection_mode', '')).lower() == 'weighted_single':
        active, counts, n_extra_eff = allocate_atoms_weighted_single_site(
            n_extra,
            attachment_sites,
            sites_cfg,
            rng=rng,
        )
        overflow = max(0, int(n_extra) - int(n_extra_eff))
        meta['n_extra_effective'] = n_extra_eff
        meta['per_site_count_mode'] = 'weighted_single'
        meta['site_selection_mode'] = 'weighted_single'
        if logger and active:
            logger.info(
                f'[MurckoSites] weighted_single: site_id='
                f'{active[0].get("site_id")} slot='
                f'{active[0].get("profile_slot")} weight='
                f'{active[0].get("site_selection_weight")} '
                f'→ n_extra={n_extra_eff}'
            )
    else:
        if overflow_mode in ('cap', 'drop'):
            n_extra = cap_n_extra_for_sites(n_extra, attachment_sites, sites_cfg)
            meta['n_extra_effective'] = n_extra
            if n_extra < n_extra_requested and logger:
                logger.info(
                    f'[MurckoSites] overflow_mode={overflow_mode}: '
                    f'n_extra {n_extra_requested} → {n_extra}（位点容量上限）'
                )
            if n_extra <= 0:
                meta['site_allocation'] = []
                return torch.zeros(0, 3, device=device), _fill_zero_allocation_preserve_meta(
                    meta, attachment_sites, [],
                )

        active, counts, overflow = allocate_atoms_to_sites(
            n_extra, attachment_sites, p_active=p_active, max_per_site=max_per_site,
            rng=rng,
        )

        drop_overflow = overflow_mode in ('cap', 'drop')
        if overflow > 0:
            if drop_overflow:
                if logger:
                    logger.info(
                        f'[MurckoSites] overflow_mode={overflow_mode}: '
                        f'丢弃 {overflow} 个溢出原子（不口袋随机放置）'
                    )
                meta['overflow_dropped'] = overflow
                overflow = 0
            elif logger:
                logger.warning(
                    f'[MurckoSites] n_extra={n_extra} 超出位点容量，'
                    f'{overflow} 个原子回退口袋随机放置'
                )
        n_extra_eff = n_extra

    # Reference-joint patterns are already the intended complete allocation.
    n_before = len([c for c in counts if c > 0])
    if joint_pattern is None:
        active, counts = concentrate_site_allocation(
            active, counts, max_active_sites, rng=rng,
            prefer_murcko=prefer_murcko,
        )
    meta['max_active_sites'] = max_active_sites
    meta['n_sites_before_concentrate'] = n_before
    meta['n_sites_after_concentrate'] = len(active)
    if logger and n_before != len(active):
        logger.info(
            f'[MurckoSites] concentrate: {n_before} → {len(active)} 位点 '
            f'(max_active_sites={max_active_sites}, counts={counts})'
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
                raise ValueError(
                    'strict_fragment_gaussian cannot place overflow atoms'
                )
        fragment_positions, fragment_type_hints, fragment_layout = (
            _strict_fragment_gaussian_site_positions(
                active, counts, rng, sites_cfg
            )
        )
        site_pos = torch.tensor(
            np.asarray(fragment_positions, dtype=np.float32),
            dtype=torch.float32,
            device=device,
        )
        meta['fragment_type_hints'] = fragment_type_hints
        meta['fragment_layout'] = fragment_layout
    else:
        site_pos = init_extra_positions_at_sites(
            active, counts, device,
            jitter_std=jitter_std, jitter_mode=jitter_mode,
            rng=rng, sites_cfg=sites_cfg, protein_positions=protein_np,
        )
    n_site = site_pos.size(0)

    parts = []
    if n_site > 0:
        parts.append(site_pos)

    if overflow > 0:
        overflow_pos = fallback_center.unsqueeze(0).expand(overflow, -1) + \
                         torch.randn(overflow, 3, device=device) * fallback_noise_scale
        parts.append(overflow_pos)

    if not parts:
        pos = fallback_center.unsqueeze(0).expand(max(n_extra_eff, 1), -1) + \
              torch.randn(max(n_extra_eff, 1), 3, device=device) * fallback_noise_scale
        meta['site_allocation'] = [
            {
                'site_id': active[i].get('site_id'),
                'count': counts[i],
                'centroid_pos': active[i].get('centroid_pos'),
                'profile_slot': active[i].get('profile_slot'),
                'site_selection_weight': active[i].get('site_selection_weight'),
            }
            for i in range(len(active)) if counts[i] > 0
        ]
        return pos, _fill_zero_allocation_preserve_meta(meta, attachment_sites)

    extra_pos = torch.cat(parts, dim=0)
    n_placed = extra_pos.size(0)
    meta['n_extra_effective'] = n_placed
    meta['n_extra_placed'] = n_placed

    drop_overflow = overflow_mode in ('cap', 'drop')
    if n_placed != n_extra_eff and not uses_site_budget_placement(sites_cfg) and not drop_overflow:
        if logger:
            logger.warning(
                f'[MurckoSites] 分配异常 ({n_placed} != {n_extra_eff})，整批回退口袋放置'
            )
        pos = fallback_center.unsqueeze(0).expand(n_extra, -1) + \
              torch.randn(n_extra, 3, device=device) * fallback_noise_scale
        meta['n_extra_effective'] = n_extra
        meta['site_allocation'] = [
            {
                'site_id': active[i].get('site_id'),
                'count': counts[i],
                'centroid_pos': active[i].get('centroid_pos'),
                'profile_slot': active[i].get('profile_slot'),
                'site_selection_weight': active[i].get('site_selection_weight'),
            }
            for i in range(len(active)) if counts[i] > 0
        ]
        return pos, _fill_zero_allocation_preserve_meta(meta, attachment_sites)

    meta['placement'] = 'murcko_sites'
    meta['overflow_count'] = overflow
    meta['site_allocation'] = [
        {
            'site_id': active[i].get('site_id'),
            'count': counts[i],
            'centroid_pos': active[i].get('centroid_pos'),
            'profile_slot': active[i].get('profile_slot'),
            'site_selection_weight': active[i].get('site_selection_weight'),
        }
        for i in range(len(active)) if counts[i] > 0
    ]
    return extra_pos, _fill_zero_allocation_preserve_meta(meta, attachment_sites)
