"""Sampling scaffold placement responsibilities for ligand sampling."""

from __future__ import annotations

from diffdynamic.models.probability import index_to_log_onehot
from rdkit import Chem
from diffdynamic.chemistry.scaffold_prior import load_scaffold_profile
from diffdynamic.chemistry.scaffold_sites import resolve_preserved_sidechains
import numpy as np
import torch
import diffdynamic.data.transforms as trans


def _murcko_placement_cfg(grow_cfg: dict, sites_cfg: dict) -> dict:
    """合并 grow 原子数 clamp 到 murcko_sites，供 random_per_site 等模式使用。"""
    cfg = dict(sites_cfg)
    if isinstance(grow_cfg, dict):
        for k in ('n_extra_min_clamp', 'n_extra_max_clamp'):
            if k in grow_cfg:
                cfg[k] = grow_cfg[k]
    return cfg


def _reference_extra_type_log_prior(
    sites_cfg: dict,
    ligand_atom_mode: str,
    n_classes: int,
    device,
    logger=None,
):
    """Build a weak aggregate extra-atom class prior for scaffold sampling.

    Only aggregate element/aromatic counts are used.  No reference atom
    ordering, bonds, coordinates, or target-side fragment is exposed to the
    diffusion model.  A zero strength keeps the historical uniform prior.
    """
    strength = float(
        sites_cfg.get('reference_extra_type_prior_strength', 0.0) or 0.0
    )
    profile_path = sites_cfg.get('reference_exit_profile')
    if strength <= 0.0 or not profile_path:
        return None
    try:
        profile = load_scaffold_profile(profile_path)
    except Exception as exc:
        if logger:
            logger.warning(f'[ScaffoldTypes] aggregate prior unavailable: {exc}')
        return None

    strength = min(max(strength, 0.0), 1.0)
    element_counts = profile.get('reference_extra_element_counts') or {}
    aromatic_counts = (
        profile.get('reference_extra_aromatic_element_counts') or {}
    )
    class_counts = np.full(int(n_classes), 0.25, dtype=np.float64)
    if ligand_atom_mode == 'add_aromatic':
        periodic_table = Chem.GetPeriodicTable()
        for (atomic_number, aromatic), class_idx in (
            trans.MAP_ATOM_TYPE_AROMATIC_TO_INDEX.items()
        ):
            class_idx = int(class_idx)
            if class_idx >= int(n_classes):
                continue
            symbol = periodic_table.GetElementSymbol(int(atomic_number))
            key = f'{symbol}|{int(bool(aromatic))}'
            class_counts[class_idx] += float(aromatic_counts.get(key, 0.0))
    elif ligand_atom_mode == 'basic':
        periodic_table = Chem.GetPeriodicTable()
        for atomic_number, class_idx in (
            trans.MAP_ATOM_TYPE_ONLY_TO_INDEX.items()
        ):
            class_idx = int(class_idx)
            if class_idx >= int(n_classes):
                continue
            symbol = periodic_table.GetElementSymbol(int(atomic_number))
            class_counts[class_idx] += float(element_counts.get(symbol, 0.0))
    else:
        return None

    # Extra atoms represent heavy atoms; leave H with a small legal mass so
    # the existing post-processing rule remains a last-resort guard only.
    if class_counts.size > 0:
        class_counts[0] = min(class_counts[0], 0.25)
    prior = class_counts / max(float(class_counts.sum()), 1e-12)
    uniform = np.full(int(n_classes), 1.0 / max(int(n_classes), 1))
    mixed = (1.0 - strength) * uniform + strength * prior
    log_prior = torch.tensor(
        np.log(np.clip(mixed, 1e-8, 1.0)),
        dtype=torch.float32,
        device=device,
    )
    if logger:
        logger.info(
            f'[ScaffoldTypes] aggregate reference class prior enabled: '
            f'strength={strength:.2f} mode={ligand_atom_mode}'
        )
    return log_prior


def _sample_reference_extra_type_quota(
    profile: dict,
    ligand_atom_mode: str,
    n_classes: int,
    n_extra: int,
    device,
    rng: np.random.Generator,
    type_hint_keys=None,
):
    """Sample a shuffled element-count multiset for one extra fragment.

    Profiles contain counts such as ``C|1: 6`` but no atom ordering, bonds,
    coordinates, or reference-side graph. Shuffling makes the assignment an
    exchangeable type prior rather than a copied target fragment.
    """
    if type_hint_keys is not None:
        class_indices = []
        periodic_table = Chem.GetPeriodicTable()
        for key in type_hint_keys:
            try:
                symbol, aromatic_text = str(key).rsplit('|', 1)
                atomic_number = int(periodic_table.GetAtomicNumber(symbol))
                aromatic = bool(int(aromatic_text))
            except (TypeError, ValueError):
                return None
            if ligand_atom_mode == 'add_aromatic':
                class_idx = trans.MAP_ATOM_TYPE_AROMATIC_TO_INDEX.get(
                    (atomic_number, aromatic)
                )
            elif ligand_atom_mode == 'basic':
                class_idx = trans.MAP_ATOM_TYPE_ONLY_TO_INDEX.get(atomic_number)
            else:
                return None
            if class_idx is None or int(class_idx) >= int(n_classes):
                return None
            class_indices.append(int(class_idx))
        if len(class_indices) != int(n_extra):
            return None
        type_indices = torch.tensor(class_indices, dtype=torch.long, device=device)
        return index_to_log_onehot(type_indices, int(n_classes))

    candidates = [
        pattern
        for pattern in profile.get('extra_type_patterns', [])
        if int(pattern.get('n_extra', -1)) == int(n_extra)
    ]
    if not candidates:
        return None
    weights = np.asarray(
        [max(float(pattern.get('weight', 0.0)), 0.0) for pattern in candidates],
        dtype=np.float64,
    )
    if not np.isfinite(weights).all() or float(weights.sum()) <= 0.0:
        return None
    selected = candidates[int(rng.choice(len(candidates), p=weights / weights.sum()))]

    periodic_table = Chem.GetPeriodicTable()
    class_indices = []
    for key, raw_count in sorted(selected.get('class_counts', {}).items()):
        try:
            symbol, aromatic_text = str(key).rsplit('|', 1)
            atomic_number = int(periodic_table.GetAtomicNumber(symbol))
            aromatic = bool(int(aromatic_text))
            count = int(raw_count)
        except (TypeError, ValueError):
            return None
        if ligand_atom_mode == 'add_aromatic':
            class_idx = trans.MAP_ATOM_TYPE_AROMATIC_TO_INDEX.get(
                (atomic_number, aromatic)
            )
        elif ligand_atom_mode == 'basic':
            class_idx = trans.MAP_ATOM_TYPE_ONLY_TO_INDEX.get(atomic_number)
        else:
            return None
        if class_idx is None or int(class_idx) >= int(n_classes) or count <= 0:
            return None
        class_indices.extend([int(class_idx)] * count)
    if len(class_indices) != int(n_extra):
        return None
    rng.shuffle(class_indices)
    type_indices = torch.tensor(class_indices, dtype=torch.long, device=device)
    return index_to_log_onehot(type_indices, int(n_classes))


def _resolve_init_preserved_atoms(
    attachment_sites,
    place_meta,
    place_cfg,
    ref_ligand_pos,
    log_ref_v,
    n_scaffold: int,
    n_extra: int,
    logger=None,
    log_tag: str = '[MurckoSites]',
    sample_idx: int = 0,
):
    """位点分配后解析零分配侧链保留，返回 (pos, log_v, indices, zero_ids, n_preserved, n_locked, n_total)。"""
    preserved_pos, preserved_log_v, preserved_indices, zero_site_ids = resolve_preserved_sidechains(
        attachment_sites, place_meta, place_cfg, ref_ligand_pos, log_ref_v,
    )
    n_preserved = int(preserved_pos.size(0))
    n_locked = int(n_scaffold) + n_preserved
    n_total = n_locked + int(n_extra)
    if logger and (sample_idx == 0 or sample_idx % 10 == 0):
        logger.info(
            f'{log_tag} init atoms: n_scaffold={n_scaffold} n_preserved={n_preserved} '
            f'n_extra={n_extra} n_total={n_total} zero_sites={zero_site_ids}'
        )
    return (
        preserved_pos, preserved_log_v, preserved_indices, zero_site_ids,
        n_preserved, n_locked, n_total,
    )


def _select_side_chain_context_anchor_indices(
    attachment_sites,
    place_meta,
    scaffold_indices,
    logger=None,
):
    """Return the fixed scaffold atoms that form the active exit context.

    The side-chain-only scaffold mode intentionally removes the rest of the
    CRBN-side scaffold from the model input.  It still needs the exact local
    interface seen by the trained network, so one real scaffold atom is kept
    for every realized active exit site.  Site metadata supplies only the
    scaffold atom index; no target-side atom, bond, order, or coordinate is
    copied into the generated tensor.
    """
    sites = list(attachment_sites or [])
    allocation = list((place_meta or {}).get('site_allocation') or [])
    site_by_id = {}
    site_by_slot = {}
    for site in sites:
        site_id = site.get('site_id')
        if site_id is not None:
            site_by_id[int(site_id)] = site
        slot = site.get('profile_slot')
        if slot is not None:
            site_by_slot[str(int(slot))] = site

    selected_sites = []
    if allocation:
        for item in allocation:
            try:
                count = int(item.get('count', 0))
            except (AttributeError, TypeError, ValueError):
                count = 0
            if count <= 0:
                continue
            site = None
            site_id = item.get('site_id')
            if site_id is not None:
                site = site_by_id.get(int(site_id))
            if site is None and item.get('profile_slot') is not None:
                site = site_by_slot.get(str(int(item['profile_slot'])))
            if site is not None:
                selected_sites.append(site)

    if not selected_sites:
        # This is only a defensive fallback for legacy placement metadata. A
        # strict exact-record run should always provide site_allocation.
        selected_sites = [
            site for site in sites
            if site.get('profile_slot') is not None
        ] or sites[:1]

    scaffold_set = {int(index) for index in (scaffold_indices or [])}
    indices = []
    for site in selected_sites:
        raw_index = site.get('anchor_scaffold_idx')
        if raw_index is None:
            raise ValueError(
                'side_chain_only requires anchor_scaffold_idx on every '
                f'active scaffold exit site: {site}'
            )
        index = int(raw_index)
        if index not in scaffold_set:
            raise ValueError(
                'side_chain_only exit anchor is outside the extracted '
                f'scaffold: anchor={index}, scaffold={sorted(scaffold_set)}'
            )
        if index not in indices:
            indices.append(index)

    if not indices:
        raise ValueError(
            'side_chain_only could not resolve an active fixed interface anchor'
        )
    if logger:
        logger.info(
            '[ScaffoldSideChainOnly] fixed interface context anchors: '
            f'{indices}'
        )
    return indices


def _murcko_rng_offset(sample_idx: int, retry: int = 0) -> int:
    """样本槽位 + 补足重试偏移（retry=0 与历史行为一致）。"""
    return int(sample_idx) * 9973 + int(retry) * 100003


def _murcko_sample_rng(
    sample_cfg: dict, sample_idx: int, retry: int = 0,
) -> np.random.Generator:
    base = int(sample_cfg.get('seed', 42)) if isinstance(sample_cfg, dict) else 42
    return np.random.default_rng(base + _murcko_rng_offset(sample_idx, retry))
