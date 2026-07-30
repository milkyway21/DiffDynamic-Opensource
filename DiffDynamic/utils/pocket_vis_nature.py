#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Nature-style pocket-quality visualization helpers (PNG only, text + _notext via caller).

Palette aligned with diffdynamic-paper-plots: iceblue / F56E1A.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Paper palette
ICEBLUE_FILL = "#C5DDF0"
ICEBLUE_EDGE = "#5A7FA0"
ACCENT = "#F56E1A"
ACCENT_SOFT = "#FBC9A6"
NEUTRAL = "#4A4A4A"
GRID = "#E8E8E8"
BAND_HIGH = "#A8C5A0"
BAND_MED = "#F0D9A8"

FONT = {
    "title": 11,
    "label": 9,
    "tick": 8,
    "annot": 7,
}


def apply_nature_axes(ax, grid_y: bool = True, grid_x: bool = False) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#888888")
    ax.spines["bottom"].set_color("#888888")
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(axis="both", labelsize=FONT["tick"], colors=NEUTRAL, length=3, width=0.6)
    if grid_y:
        ax.yaxis.grid(True, color=GRID, linewidth=0.6, zorder=0)
    if grid_x:
        ax.xaxis.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)


def morgan_fps_from_molecules(
    molecules_with_pos: Sequence[Tuple[Any, Any]],
    n_bits: int = 2048,
    radius: int = 2,
) -> Tuple[Optional[np.ndarray], List[int]]:
    """Return (N, n_bits) float array and list of original indices that succeeded."""
    try:
        from rdkit.Chem import AllChem, DataStructs
    except ImportError:
        return None, []
    rows = []
    keep = []
    for i, (mol, _pos) in enumerate(molecules_with_pos or []):
        if mol is None:
            continue
        try:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            arr = np.zeros((n_bits,), dtype=np.float64)
            DataStructs.ConvertToNumpyArray(fp, arr)
            rows.append(arr)
            keep.append(i)
        except Exception:
            continue
    if len(rows) < 1:
        return None, keep
    return np.vstack(rows), keep


def embed_2d(fps: np.ndarray, method: str = "auto", random_state: int = 42) -> Tuple[np.ndarray, str]:
    """
    2D embedding. Prefer UMAP; fall back to PCA.
    Returns (xy, method_used).
    """
    method = (method or "auto").lower()
    if method in ("auto", "umap"):
        try:
            import umap

            n = int(fps.shape[0])
            n_neighbors = int(np.clip(n // 5, 5, 30))
            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=n_neighbors,
                min_dist=0.1,
                metric="jaccard",
                random_state=random_state,
            )
            # binary Morgan; Jaccard matches fingerprint chemical distance
            xy = reducer.fit_transform(fps.astype(np.float64))
            return np.asarray(xy, dtype=np.float64), "UMAP"
        except Exception:
            if method == "umap":
                pass
    from sklearn.decomposition import PCA

    xy = PCA(n_components=2, random_state=random_state).fit_transform(fps)
    return np.asarray(xy, dtype=np.float64), "PCA"


def vina_colors_for_indices(
    idea_a: Dict[str, Any],
    indices: List[int],
) -> Optional[np.ndarray]:
    vs = idea_a.get("vina_scores") if idea_a else None
    if not vs:
        return None
    out = []
    for i in indices:
        if i < len(vs) and vs[i] is not None:
            try:
                out.append(float(vs[i]))
            except (TypeError, ValueError):
                out.append(np.nan)
        else:
            out.append(np.nan)
    arr = np.asarray(out, dtype=np.float64)
    if not np.isfinite(arr).any():
        return None
    return arr
