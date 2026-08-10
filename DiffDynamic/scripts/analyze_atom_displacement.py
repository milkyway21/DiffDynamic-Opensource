#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Atomic displacement analysis for SBDD diffusion trajectories.

Inventory (what existing .pt contain; none store true Gaussian init):
  - TargetDiff  (/data/ye/pt/targetdiff-main/outputs/result_*.pt)
      keys: data, pred_ligand_pos, pred_ligand_v, pred_ligand_pos_traj, pred_ligand_v_traj, time
      traj: (1000, N, 3); traj[0] = after first denoising step (NOT x_T)
  - IPDiff  (third_party/IPDiff/outputs_ipdiff_bench_100p/result_*.pt)
      same key family; traj (1000, N, 3)
  - MolForm (third_party/MolForm/outputs_molform_bench_100p/result_*.pt)
      traj (101, N, 3)
  - DecompDiff (third_party/DecompDiff/outputs_decompdiff_bench_100p/*/result.pt)
      list[dict] with pred_pos, pred_pos_traj (1000, N, 3)
  - DD-Fast (jsdpt3010): traj ~22–44 frames; phase split via time_indices
      baseline_refine: 有轨迹则逐步；无轨迹则用 refine末态→final 只记十步总位移
  - GlintDM: traj T=1 only -> skipped; DiffSBDD: SDF only -> skipped

Outputs under --out-dir (default outputs/displacement_stats/):
  stage10/{method}/stage10_*.parquet   — 10% progress stages (all methods)
  dd_phases/{method}/phase_steps_*.parquet / phase_end_to_end_*.parquet
  step_atom/{method}/steps_*.parquet   — every adjacent-frame per-atom d_step
  summary/*.csv                        — quantiles
  init_compare/*.csv                   — true_init vs traj0 when init_ligand_pos present
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# Ensure DiffDynamic root is on path so torch.load can unpickle ProteinLigandData
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
# Import modules referenced inside pickled ProteinLigandData graphs
for _mod in ("datasets", "utils", "utils.data", "datasets.pl_data"):
    try:
        __import__(_mod)
    except Exception:
        pass

import numpy as np
import pandas as pd
import torch

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

THIRD_PARTY = "/home/user/Desktop/Ye/DiffDynamic/third_party"

DEFAULT_SOURCES = {
    "targetdiff": "/data/ye/pt/targetdiff-main/outputs",
    "ipdiff": f"{THIRD_PARTY}/IPDiff/outputs_ipdiff_bench_100p",
    "molform": f"{THIRD_PARTY}/MolForm/outputs_molform_bench_100p",
    "decompdiff": f"{THIRD_PARTY}/DecompDiff/outputs_decompdiff_bench_100p",
    "dd_fast": "/home/user/Desktop/Ye/DiffDynamic/jsdpt3010",
}

DD_METHODS = {"dd_fast"}


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _as_np(x) -> np.ndarray:
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def _traj_to_TNC(traj_item) -> Optional[np.ndarray]:
    """Normalize one molecule trajectory to float64 (T, N, 3)."""
    if traj_item is None:
        return None
    if isinstance(traj_item, list):
        if len(traj_item) == 0:
            return None
        frames = [_as_np(f) for f in traj_item]
        try:
            arr = np.stack(frames, axis=0)
        except Exception:
            return None
    else:
        arr = _as_np(traj_item)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        return None
    if arr.shape[0] < 2:
        return None
    return arr.astype(np.float64, copy=False)


def _load_pt(path: str):
    return torch.load(path, map_location="cpu")


def _iter_std_molecules(obj: dict) -> Iterable[Tuple[int, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]]:
    """Yield (mol_idx, traj_TNC, final_pos, init_pos) for TargetDiff-family dicts."""
    trajs = obj.get("pred_ligand_pos_traj")
    finals = obj.get("pred_ligand_pos")
    inits = obj.get("init_ligand_pos")
    if trajs is None:
        return
    n = len(trajs)
    for i in range(n):
        traj = _traj_to_TNC(trajs[i])
        if traj is None:
            continue
        final = _as_np(finals[i]) if finals is not None and i < len(finals) else traj[-1]
        init = None
        if inits is not None and i < len(inits) and inits[i] is not None:
            init_arr = _as_np(inits[i])
            if init_arr.shape == traj[0].shape:
                init = init_arr.astype(np.float64, copy=False)
        yield i, traj, final.astype(np.float64, copy=False), init


def _iter_decomp_molecules(obj) -> Iterable[Tuple[int, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]]:
    if not isinstance(obj, list):
        return
    for i, item in enumerate(obj):
        if not isinstance(item, dict):
            continue
        traj = _traj_to_TNC(item.get("pred_pos_traj"))
        if traj is None:
            continue
        final = _as_np(item.get("pred_pos", traj[-1]))
        yield i, traj, final.astype(np.float64, copy=False), None


def discover_pt_files(method: str, root: str) -> List[str]:
    if not root or not os.path.exists(root):
        return []
    if method == "decompdiff":
        files = sorted(glob.glob(os.path.join(root, "*", "result.pt")))
        return files
    files = sorted(glob.glob(os.path.join(root, "result_*.pt")))
    if not files:
        files = sorted(glob.glob(os.path.join(root, "**", "result_*.pt"), recursive=True))
    # drop eval_results
    files = [f for f in files if "/eval_results/" not in f.replace("\\", "/")]
    return files


def parse_pocket_id(path: str) -> str:
    base = os.path.basename(path)
    m = re.search(r"result_(\d+)", base)
    if m:
        return m.group(1)
    parent = os.path.basename(os.path.dirname(path))
    m2 = re.search(r"sampling_\w+_(\d+)_", parent)
    if m2:
        return m2.group(1)
    return parent or base


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def per_atom_step_displacements(traj: np.ndarray) -> np.ndarray:
    """(T-1, N) Euclidean step distances."""
    return np.linalg.norm(traj[1:] - traj[:-1], axis=-1)


def stage10_displacements(traj: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      d_stage: (10, N)
      frame_lo, frame_hi: (10,)
    """
    T = traj.shape[0]
    qs = np.linspace(0.0, 1.0, 11)
    idxs = np.clip(np.rint(qs * (T - 1)).astype(np.int64), 0, T - 1)
    # ensure non-decreasing unique-ish anchors
    for i in range(1, len(idxs)):
        if idxs[i] < idxs[i - 1]:
            idxs[i] = idxs[i - 1]
    d = np.zeros((10, traj.shape[1]), dtype=np.float64)
    flo = np.zeros(10, dtype=np.int64)
    fhi = np.zeros(10, dtype=np.int64)
    for s in range(10):
        flo[s] = idxs[s]
        fhi[s] = idxs[s + 1]
        d[s] = np.linalg.norm(traj[fhi[s]] - traj[flo[s]], axis=-1)
    return d, flo, fhi


def quantiles(arr: np.ndarray, ps=(50, 90, 99)) -> Dict[str, float]:
    if arr.size == 0:
        return {f"p{p}": float("nan") for p in ps} | {"mean": float("nan"), "n": 0}
    out = {f"p{p}": float(np.percentile(arr, p)) for p in ps}
    out["mean"] = float(np.mean(arr))
    out["n"] = int(arr.size)
    return out


# ---------------------------------------------------------------------------
# DD phase split
# ---------------------------------------------------------------------------

def _cand_traj(cand: Optional[dict]) -> Optional[np.ndarray]:
    if not cand:
        return None
    return _traj_to_TNC(cand.get("pos_traj"))


def split_dd_phases(
    traj: np.ndarray,
    meta: Optional[dict],
    mol_idx: int,
) -> Dict[str, Optional[np.ndarray]]:
    """
    Return dict phase -> traj_slice (T_phase, N, 3) or None if phase_missing.
    Phases: large_step, refine, baseline_refine
    """
    meta = meta or {}
    br_ti = meta.get("baseline_refine_time_indices")
    n_br = len(br_ti) if br_ti is not None else 0

    large_cands = meta.get("large_step_candidates") or []
    refined_cands = meta.get("refined_candidates") or []

    ls_cand = large_cands[mol_idx] if mol_idx < len(large_cands) else None
    rf_cand = refined_cands[mol_idx] if mol_idx < len(refined_cands) else None

    ls_traj = _cand_traj(ls_cand if isinstance(ls_cand, dict) else None)
    rf_ti = None
    if isinstance(rf_cand, dict):
        rf_ti = rf_cand.get("time_indices")

    T = traj.shape[0]
    phases: Dict[str, Optional[np.ndarray]] = {
        "large_step": None,
        "refine": None,
        "baseline_refine": None,
    }

    # baseline: trailing frames
    if n_br > 0 and T >= n_br:
        phases["baseline_refine"] = traj[T - n_br :]
        body = traj[: T - n_br] if T > n_br else None
    else:
        body = traj
        phases["baseline_refine"] = None

    # large_step: prefer explicit candidate traj; else prefix matching len(large time_indices)
    n_ls = 0
    if ls_traj is not None and ls_traj.shape[0] >= 2:
        phases["large_step"] = ls_traj
        n_ls = ls_traj.shape[0]
    elif isinstance(ls_cand, dict) and ls_cand.get("time_indices") is not None:
        n_ls = len(ls_cand["time_indices"])
        if body is not None and n_ls >= 2 and body.shape[0] >= n_ls:
            # only treat as large_step prefix if refined time_indices suggest merge
            if rf_ti is not None and len(rf_ti) == (body.shape[0] if n_br == 0 else body.shape[0]):
                # if first time index looks like high-t large schedule
                try:
                    if int(rf_ti[0]) > 700 and n_ls < body.shape[0]:
                        phases["large_step"] = body[:n_ls]
                except Exception:
                    pass

    # refine: body after large_step frames when merged into top traj
    if body is None or body.shape[0] < 2:
        phases["refine"] = None
        return phases

    if phases["large_step"] is not None and ls_traj is not None:
        # large traj separate from top body -> body is refine (+ maybe empty large)
        # If top traj starts at refine boundary, entire body is refine
        if rf_ti is not None and len(rf_ti) == body.shape[0]:
            phases["refine"] = body
        elif n_ls > 0 and body.shape[0] > n_ls and phases.get("large_step") is not None and ls_traj is body[:n_ls]:
            phases["refine"] = body[n_ls - 1 :] if body.shape[0] - n_ls + 1 >= 2 else body[n_ls:]
            if phases["refine"] is not None and phases["refine"].shape[0] < 2:
                phases["refine"] = None
        else:
            phases["refine"] = body
    elif isinstance(ls_cand, dict) and ls_cand.get("time_indices") is not None and rf_ti is not None:
        n_ls_ti = len(ls_cand["time_indices"])
        # merged: len(rf_ti) includes large+refine
        if len(rf_ti) == body.shape[0] and n_ls_ti >= 2 and body.shape[0] > n_ls_ti:
            if phases["large_step"] is None:
                phases["large_step"] = body[:n_ls_ti]
            rest = body[n_ls_ti - 1 :]
            phases["refine"] = rest if rest.shape[0] >= 2 else None
        else:
            # refine-only top traj (common in archives)
            phases["refine"] = body
            if phases["large_step"] is None and ls_traj is None:
                phases["large_step"] = None  # phase_missing
    else:
        phases["refine"] = body

    return phases


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

class ShardWriter:
    def __init__(self, out_dir: str, prefix: str, fmt: str = "parquet", shard_rows: int = 2_000_000):
        self.out_dir = out_dir
        self.prefix = prefix
        self.fmt = fmt
        self.shard_rows = shard_rows
        self.rows: List[dict] = []
        self.shard_idx = 0
        self.total_rows = 0
        os.makedirs(out_dir, exist_ok=True)

    def extend(self, rows: List[dict]):
        if not rows:
            return
        self.rows.extend(rows)
        if len(self.rows) >= self.shard_rows:
            self.flush()

    def flush(self):
        if not self.rows:
            return
        df = pd.DataFrame(self.rows)
        path = os.path.join(self.out_dir, f"{self.prefix}_{self.shard_idx:04d}.{self.fmt}")
        if self.fmt == "csv":
            df.to_csv(path, index=False)
        else:
            df.to_parquet(path, index=False)
        self.total_rows += len(self.rows)
        self.rows = []
        self.shard_idx += 1

    def close(self):
        self.flush()


class StepAtomWriter:
    """Vectorized writer for per-atom step displacements (avoids Python dict per cell)."""

    def __init__(self, out_dir: str, method: str, shard_rows: int = 5_000_000):
        self.out_dir = out_dir
        self.method = method
        self.shard_rows = shard_rows
        self.shard_idx = 0
        self.total_rows = 0
        self._buf = {
            "source_file": [],
            "pocket_id": [],
            "mol_idx": [],
            "atom_idx": [],
            "step_k": [],
            "n_frames": [],
            "d_step": [],
        }
        self._buf_n = 0
        os.makedirs(out_dir, exist_ok=True)

    def add_traj(self, source_file: str, pocket_id: str, mol_idx: int, traj: np.ndarray):
        steps = per_atom_step_displacements(traj)  # (T-1, N)
        Tm1, N = steps.shape
        if Tm1 <= 0 or N <= 0:
            return
        # vectorized index grids
        step_k = np.repeat(np.arange(Tm1, dtype=np.int32), N)
        atom_idx = np.tile(np.arange(N, dtype=np.int32), Tm1)
        d_step = steps.reshape(-1).astype(np.float32)
        n = Tm1 * N
        self._buf["source_file"].append(np.full(n, source_file, dtype=object))
        self._buf["pocket_id"].append(np.full(n, pocket_id, dtype=object))
        self._buf["mol_idx"].append(np.full(n, mol_idx, dtype=np.int32))
        self._buf["atom_idx"].append(atom_idx)
        self._buf["step_k"].append(step_k)
        self._buf["n_frames"].append(np.full(n, traj.shape[0], dtype=np.int32))
        self._buf["d_step"].append(d_step)
        self._buf_n += n
        if self._buf_n >= self.shard_rows:
            self.flush()

    def flush(self):
        if self._buf_n == 0:
            return
        df = pd.DataFrame({
            "method": self.method,
            "source_file": np.concatenate(self._buf["source_file"]),
            "pocket_id": np.concatenate(self._buf["pocket_id"]),
            "mol_idx": np.concatenate(self._buf["mol_idx"]),
            "atom_idx": np.concatenate(self._buf["atom_idx"]),
            "step_k": np.concatenate(self._buf["step_k"]),
            "n_frames": np.concatenate(self._buf["n_frames"]),
            "d_step": np.concatenate(self._buf["d_step"]),
            "start_source": "traj0_proxy",
        })
        path = os.path.join(self.out_dir, f"steps_{self.shard_idx:04d}.parquet")
        df.to_parquet(path, index=False)
        self.total_rows += len(df)
        self.shard_idx += 1
        self._buf = {k: [] for k in self._buf}
        self._buf_n = 0
        del df

    def close(self):
        self.flush()


def write_quantile_table(records: List[dict], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame(records).to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Per-method processing
# ---------------------------------------------------------------------------

def process_molecule_stage10(
    method: str,
    source_file: str,
    pocket_id: str,
    mol_idx: int,
    traj: np.ndarray,
) -> List[dict]:
    d, flo, fhi = stage10_displacements(traj)
    T, N, _ = traj.shape
    rows = []
    for s in range(10):
        for a in range(N):
            rows.append({
                "method": method,
                "source_file": source_file,
                "pocket_id": pocket_id,
                "mol_idx": mol_idx,
                "atom_idx": a,
                "stage_s": s,
                "stage_pct_lo": s * 10,
                "stage_pct_hi": (s + 1) * 10,
                "frame_lo": int(flo[s]),
                "frame_hi": int(fhi[s]),
                "n_frames": T,
                "d_stage": float(d[s, a]),
                "start_source": "traj0_proxy",
            })
    return rows


def process_molecule_steps(
    method: str,
    source_file: str,
    pocket_id: str,
    mol_idx: int,
    traj: np.ndarray,
) -> List[dict]:
    steps = per_atom_step_displacements(traj)  # (T-1, N)
    T, N, _ = traj.shape
    rows = []
    for k in range(steps.shape[0]):
        for a in range(N):
            rows.append({
                "method": method,
                "source_file": source_file,
                "pocket_id": pocket_id,
                "mol_idx": mol_idx,
                "atom_idx": a,
                "step_k": k,
                "n_frames": T,
                "d_step": float(steps[k, a]),
                "start_source": "traj0_proxy",
            })
    return rows


def process_molecule_dd_phases(
    method: str,
    source_file: str,
    pocket_id: str,
    mol_idx: int,
    traj: np.ndarray,
    meta: dict,
    final: Optional[np.ndarray] = None,
) -> Tuple[List[dict], List[dict], Dict[str, int]]:
    """
    baseline_refine:
      - 有逐步轨迹 (T>=2)：逐步 d_step + 端到端
      - 无轨迹：用 refine 结束位姿 → final，只记十步总位移 d_phase_ee
        (phase_status=ee_only_no_traj；不写逐步表)
    """
    phases = split_dd_phases(traj, meta, mol_idx)
    step_rows: List[dict] = []
    ee_rows: List[dict] = []
    status: Dict[str, int] = {}

    refined_cands = (meta or {}).get("refined_candidates") or []
    rf_cand = refined_cands[mol_idx] if mol_idx < len(refined_cands) else None

    for phase, ptraj in phases.items():
        # baseline 无轨迹：只统计十步总位移（refine 末态 / cand.pos → final）
        if phase == "baseline_refine" and (ptraj is None or ptraj.shape[0] < 2):
            start = None
            if isinstance(rf_cand, dict) and rf_cand.get("pos") is not None:
                start = np.asarray(rf_cand["pos"], dtype=np.float64)
            elif phases.get("refine") is not None and phases["refine"].shape[0] >= 1:
                start = phases["refine"][-1]
            if start is not None and final is not None and start.shape == np.asarray(final).shape:
                end = np.asarray(final, dtype=np.float64)
                ee = np.linalg.norm(end - start, axis=-1)
                status[phase] = 1
                for a in range(ee.shape[0]):
                    ee_rows.append({
                        "method": method,
                        "phase": phase,
                        "source_file": source_file,
                        "pocket_id": pocket_id,
                        "mol_idx": mol_idx,
                        "atom_idx": a,
                        "d_phase_ee": float(ee[a]),
                        "path_length": float(ee[a]),  # 无逐步轨迹时路径长度≈端到端
                        "tortuosity": 1.0,
                        "n_frames_phase": 2,
                        "assumed_n_steps": 10,
                        "phase_status": "ee_only_no_traj",
                    })
            else:
                status[phase] = 0
                ee_rows.append({
                    "method": method,
                    "phase": phase,
                    "source_file": source_file,
                    "pocket_id": pocket_id,
                    "mol_idx": mol_idx,
                    "atom_idx": -1,
                    "d_phase_ee": float("nan"),
                    "path_length": float("nan"),
                    "tortuosity": float("nan"),
                    "n_frames_phase": 0,
                    "assumed_n_steps": 10,
                    "phase_status": "phase_missing",
                })
            continue

        if ptraj is None or ptraj.shape[0] < 2:
            status[phase] = 0
            ee_rows.append({
                "method": method,
                "phase": phase,
                "source_file": source_file,
                "pocket_id": pocket_id,
                "mol_idx": mol_idx,
                "atom_idx": -1,
                "d_phase_ee": float("nan"),
                "path_length": float("nan"),
                "tortuosity": float("nan"),
                "n_frames_phase": 0,
                "phase_status": "phase_missing",
            })
            continue

        status[phase] = 1
        steps = per_atom_step_displacements(ptraj)
        Tp, N, _ = ptraj.shape
        for k in range(steps.shape[0]):
            for a in range(N):
                step_rows.append({
                    "method": method,
                    "phase": phase,
                    "source_file": source_file,
                    "pocket_id": pocket_id,
                    "mol_idx": mol_idx,
                    "atom_idx": a,
                    "step_k_in_phase": k,
                    "d_step": float(steps[k, a]),
                    "n_frames_phase": Tp,
                    "phase_status": "ok",
                })
        ee = np.linalg.norm(ptraj[-1] - ptraj[0], axis=-1)
        path = steps.sum(axis=0)
        for a in range(N):
            tort = float(path[a] / ee[a]) if ee[a] > 1e-12 else float("nan")
            ee_rows.append({
                "method": method,
                "phase": phase,
                "source_file": source_file,
                "pocket_id": pocket_id,
                "mol_idx": mol_idx,
                "atom_idx": a,
                "d_phase_ee": float(ee[a]),
                "path_length": float(path[a]),
                "tortuosity": tort,
                "n_frames_phase": Tp,
                "phase_status": "ok",
            })
    return step_rows, ee_rows, status


def process_init_compare(
    method: str,
    source_file: str,
    pocket_id: str,
    mol_idx: int,
    traj: np.ndarray,
    final: np.ndarray,
    init: np.ndarray,
) -> List[dict]:
    rows = []
    N = init.shape[0]
    d_init = np.linalg.norm(final - init, axis=-1)
    d_traj0 = np.linalg.norm(final - traj[0], axis=-1)
    d_init_traj0 = np.linalg.norm(traj[0] - init, axis=-1)
    for a in range(N):
        rows.append({
            "method": method,
            "source_file": source_file,
            "pocket_id": pocket_id,
            "mol_idx": mol_idx,
            "atom_idx": a,
            "d_final_init": float(d_init[a]),
            "d_final_traj0": float(d_traj0[a]),
            "d_traj0_init": float(d_init_traj0[a]),
            "delta_proxy_error": float(d_init[a] - d_traj0[a]),
        })
    return rows


def process_method(
    method: str,
    root: str,
    out_root: str,
    fmt: str,
    max_files: Optional[int],
    max_mols_per_file: Optional[int],
    write_step_atom: bool,
    stage_values: Dict[Tuple[str, int], List[float]],
    phase_step_values: Dict[Tuple[str, str], List[float]],
    phase_ee_values: Dict[Tuple[str, str], List[float]],
    phase_missing_counts: Dict[Tuple[str, str], int],
    phase_ok_counts: Dict[Tuple[str, str], int],
) -> Dict[str, Any]:
    files = discover_pt_files(method, root)
    if max_files is not None:
        files = files[:max_files]
    stats = {"method": method, "n_files": len(files), "n_mols": 0, "n_skipped_traj": 0}

    stage_w = ShardWriter(os.path.join(out_root, "stage10", method), "stage10", fmt=fmt)
    step_w = StepAtomWriter(os.path.join(out_root, "step_atom", method), method) if write_step_atom else None
    phase_step_w = ShardWriter(os.path.join(out_root, "dd_phases", method), "phase_steps", fmt=fmt) if method in DD_METHODS else None
    phase_ee_w = ShardWriter(os.path.join(out_root, "dd_phases", method), "phase_end_to_end", fmt=fmt) if method in DD_METHODS else None
    init_w = ShardWriter(os.path.join(out_root, "init_compare", method), "init_compare", fmt="csv")

    for fi, fp in enumerate(files):
        try:
            obj = _load_pt(fp)
        except Exception as e:
            print(f"[warn] load fail {fp}: {e}", file=sys.stderr, flush=True)
            continue
        pocket_id = parse_pocket_id(fp)
        meta = obj.get("meta") if isinstance(obj, dict) else {}

        if method == "decompdiff":
            mol_iter = _iter_decomp_molecules(obj)
        elif isinstance(obj, dict):
            mol_iter = _iter_std_molecules(obj)
        else:
            continue

        n_mol_this = 0
        for mol_idx, traj, final, init in mol_iter:
            if max_mols_per_file is not None and n_mol_this >= max_mols_per_file:
                break
            n_mol_this += 1
            stats["n_mols"] += 1

            stage_rows = process_molecule_stage10(method, fp, pocket_id, mol_idx, traj)
            stage_w.extend(stage_rows)
            d_stage, _, _ = stage10_displacements(traj)
            for s in range(10):
                stage_values[(method, s)].extend(d_stage[s].tolist())

            if write_step_atom and step_w is not None:
                step_w.add_traj(fp, pocket_id, mol_idx, traj)

            if method in DD_METHODS and phase_step_w is not None and phase_ee_w is not None:
                srows, erows, status = process_molecule_dd_phases(
                    method, fp, pocket_id, mol_idx, traj, meta or {}, final=final
                )
                phase_step_w.extend(srows)
                phase_ee_w.extend(erows)
                for phase, ok in status.items():
                    key = (method, phase)
                    if ok:
                        phase_ok_counts[key] += 1
                    else:
                        phase_missing_counts[key] += 1
                for r in srows:
                    phase_step_values[(method, r["phase"])].append(r["d_step"])
                for r in erows:
                    if r.get("atom_idx", -1) >= 0 and r.get("phase_status") in (
                        "ok", "ee_only_no_traj"
                    ):
                        phase_ee_values[(method, r["phase"])].append(r["d_phase_ee"])

            if init is not None:
                init_w.extend(process_init_compare(method, fp, pocket_id, mol_idx, traj, final, init))

        if n_mol_this == 0:
            stats["n_skipped_traj"] += 1
        if (fi + 1) % 5 == 0 or fi == 0 or fi + 1 == len(files):
            print(
                f"  [{method}] files {fi + 1}/{len(files)} mols={stats['n_mols']} "
                f"step_rows={step_w.total_rows + (step_w._buf_n if step_w else 0)}",
                flush=True,
            )

    stage_w.close()
    if step_w:
        step_w.close()
    if phase_step_w:
        phase_step_w.close()
    if phase_ee_w:
        phase_ee_w.close()
    init_w.close()
    stats["stage_rows"] = stage_w.total_rows
    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_argparser():
    p = argparse.ArgumentParser(description="Atom displacement: 10% stages + DD phases")
    p.add_argument("--out-dir", type=str, default="outputs/displacement_stats")
    p.add_argument("--format", choices=["parquet", "csv"], default="parquet")
    p.add_argument("--methods", nargs="+", default=list(DEFAULT_SOURCES.keys()))
    p.add_argument("--source", action="append", default=[],
                   help="Override source as method=path (repeatable)")
    p.add_argument("--max-files", type=int, default=None)
    p.add_argument("--max-mols-per-file", type=int, default=None)
    p.add_argument("--write-step-atom", action="store_true", default=True)
    p.add_argument("--no-write-step-atom", action="store_false", dest="write_step_atom")
    return p


def main():
    args = build_argparser().parse_args()
    sources = dict(DEFAULT_SOURCES)
    for item in args.source:
        if "=" not in item:
            raise SystemExit(f"--source must be method=path, got {item}")
        m, path = item.split("=", 1)
        sources[m] = path

    out_root = os.path.abspath(args.out_dir)
    os.makedirs(out_root, exist_ok=True)

    stage_values: Dict[Tuple[str, int], List[float]] = defaultdict(list)
    phase_step_values: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    phase_ee_values: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    phase_missing_counts: Dict[Tuple[str, str], int] = defaultdict(int)
    phase_ok_counts: Dict[Tuple[str, str], int] = defaultdict(int)

    all_stats = []
    for method in args.methods:
        root = sources.get(method)
        if not root:
            print(f"[skip] no source for {method}")
            continue
        print(f"[run] {method} <- {root}")
        st = process_method(
            method=method,
            root=root,
            out_root=out_root,
            fmt=args.format,
            max_files=args.max_files,
            max_mols_per_file=args.max_mols_per_file,
            write_step_atom=args.write_step_atom,
            stage_values=stage_values,
            phase_step_values=phase_step_values,
            phase_ee_values=phase_ee_values,
            phase_missing_counts=phase_missing_counts,
            phase_ok_counts=phase_ok_counts,
        )
        all_stats.append(st)
        print(f"  -> files={st['n_files']} mols={st['n_mols']} stage_rows={st.get('stage_rows')}")

    # summary tables
    stage_q = []
    for (method, s), vals in sorted(stage_values.items()):
        q = quantiles(np.asarray(vals, dtype=np.float64))
        stage_q.append({
            "method": method,
            "stage_s": s,
            "stage_pct_lo": s * 10,
            "stage_pct_hi": (s + 1) * 10,
            **q,
        })
    write_quantile_table(stage_q, os.path.join(out_root, "summary", "stage10_quantiles_by_method.csv"))

    phase_q = []
    for (method, phase), vals in sorted(phase_step_values.items()):
        q = quantiles(np.asarray(vals, dtype=np.float64))
        phase_q.append({
            "method": method,
            "phase": phase,
            "metric": "d_step",
            **q,
            "n_mols_ok": phase_ok_counts[(method, phase)],
            "n_mols_missing": phase_missing_counts[(method, phase)],
        })
    for (method, phase), vals in sorted(phase_ee_values.items()):
        q = quantiles(np.asarray(vals, dtype=np.float64))
        phase_q.append({
            "method": method,
            "phase": phase,
            "metric": "d_phase_ee",
            **q,
            "n_mols_ok": phase_ok_counts[(method, phase)],
            "n_mols_missing": phase_missing_counts[(method, phase)],
        })
    write_quantile_table(phase_q, os.path.join(out_root, "summary", "dd_phase_quantiles.csv"))
    write_quantile_table(all_stats, os.path.join(out_root, "summary", "run_stats.csv"))

    # human-readable report snippet
    report_path = os.path.join(out_root, "summary", "report_numbers.txt")
    with open(report_path, "w") as f:
        f.write("=== stage10 d_stage P50/P90 by method×stage ===\n")
        for row in stage_q:
            f.write(
                f"{row['method']:12s} stage {row['stage_s']} "
                f"({row['stage_pct_lo']:3d}-{row['stage_pct_hi']:3d}%): "
                f"P50={row['p50']:.4f} P90={row['p90']:.4f} n={row['n']}\n"
            )
        f.write("\n=== DD phase d_step / d_phase_ee ===\n")
        for row in phase_q:
            f.write(
                f"{row['method']:10s} {row['phase']:18s} {row['metric']:12s}: "
                f"P50={row['p50']:.4f} P90={row['p90']:.4f} n={row['n']} "
                f"ok_mols={row['n_mols_ok']} missing_mols={row['n_mols_missing']}\n"
            )
    print(f"[done] wrote {report_path}")


if __name__ == "__main__":
    main()
