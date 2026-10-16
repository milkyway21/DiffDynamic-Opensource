from __future__ import annotations
from diffdynamic.io.paths import project_root

"""Shared imports, constants and process configuration for diffdynamic.chemistry.sites."""

import json


from collections import deque


from datetime import datetime, timezone, timedelta


from pathlib import Path


from typing import Any, Dict, Iterable, List, Optional, Tuple


import numpy as np


import torch


from rdkit import Chem


from diffdynamic.chemistry.scaffold_prior import load_scaffold_profile


CST = timezone(timedelta(hours=8))


DEFAULT_FRAGMENT_SIZES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 18, 20]


DEFAULT_FRAGMENT_WEIGHTS = [
    0.02,
    0.02,
    0.02,
    0.02,
    0.03,
    0.16,
    0.10,
    0.09,
    0.08,
    0.12,
    0.12,
    0.10,
    0.05,
    0.04,
    0.015,
    0.01,
    0.005,
]


DEFAULT_MURCKO_SITES_CFG = {
    "p_active": 0.5,
    "max_per_site": 20,
    "min_per_site": 0,
    "per_site_count_mode": "split",
    # fragment_prior：按常见自由基规模一次写入；uniform：旧 Uniform[min,max]
    "per_site_add_mode": "fragment_prior",
    "fragment_sizes": list(DEFAULT_FRAGMENT_SIZES),
    "fragment_weights": list(DEFAULT_FRAGMENT_WEIGHTS),
    # 与从头生成一致：质心 + N(0, σ)，σ≈1（sample_diffusion 里 center+randn）
    "jitter_std": 1.0,
    "jitter_mode": "gaussian",  # gaussian|pocket_center_gaussian|directional
    # 多原子时沿 anchor→centroid 径向外推（Å）：r = radial_step*atom_idx + radial_bulk*max(0, count-start)
    "radial_step": 0.25,
    "radial_bulk": 0.35,
    "radial_bulk_start": 4,
    # directional 模式：沿射线第 k 个原子再外推 directional_step Å
    "directional_step": 1.0,
    "directional_min_base_dist": 1.5,
    # 每个样本最多在几个去除位点上释放；默认 = 去掉的侧链数（n_removed_sidechains）
    "max_active_sites": "n_removed_sidechains",
    "prefer_murcko_sites": True,  # 浓缩时优先真实侧链去除位点，而非 exit-vector
    "save_json": True,
    "overflow_mode": "pocket_fallback",
    "dedup_dist": 0.5,
    # 虚拟 exit-vector：在环上可取代原子处补充释放点（缓解单侧链位点问题）
    "include_exit_vectors": False,
    "exit_vector_mode": "aromatic_h",  # aromatic_h | ring_h | all_h
    "exit_vector_offset": 1.5,
    "exit_vector_dedup_dist": 1.2,
    # 位点分配为 0 时保留该位点原配体侧链（removed_atom_indices）；false=旧行为全剥侧链
    "preserve_zero_allocation_sidechains": True,
}


SCAFFOLD_SOURCES_WITH_ATTACHMENT_SITES = (
    "auto_murcko",
    "auto_murcko_generic",
    "smarts",
    "custom",  # alias of smarts
    "atom_indices",
)


_ARYL_KINDS = frozenset({"aromatic_ring", "fused_aromatic", "fused_ring"})


_ALIPHATIC_RING_KINDS = frozenset({"aliphatic_ring"})


_LINKER_KINDS = frozenset({"chain", "carbonyl", "hetero_atom"})
