"""Structured access to DiffDynamic data: proteins, outputs, molecules, configs.

Replaces filesystem scanning with organized data access patterns.
"""

import glob
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import torch
import yaml


def get_config_snapshot(yml_path: str) -> dict:
    """Read and return the sampling.yml as a dict."""
    try:
        with open(yml_path) as f:
            return yaml.safe_load(f)
    except Exception:
        return {}


def write_config(yml_path: str, data: dict):
    """Write a dict back to sampling.yml."""
    with open(yml_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def _unique_dirs(paths) -> List[str]:
    seen = set()
    result = []
    for path in paths:
        if not path:
            continue
        normalized = os.path.realpath(path)
        if normalized not in seen and os.path.isdir(normalized):
            seen.add(normalized)
            result.append(normalized)
    return result


def scan_outputs(output_root: str) -> List[dict]:
    """Scan output directory for result_*.pt files with metadata."""
    results = []
    for scan_root in _unique_dirs((output_root, os.path.join(output_root, "pt"))):
        for entry in sorted(glob.glob(os.path.join(scan_root, "result_*.pt"))):
            stat = os.stat(entry)
            results.append({
                "path": entry,
                "filename": os.path.basename(entry),
                "size_mb": round(stat.st_size / 1024 / 1024, 2),
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "data_id": _extract_data_id(entry),
            })
    return results


def scan_eval_dirs(output_root: str) -> List[dict]:
    """Scan for eval directories containing evaluation results.

    Fast path: read the sidecar ``summary.json`` written by the evaluator.
    Legacy dirs without a sidecar get file metadata only — we never
    ``torch.load`` full result files here (they can be hundreds of MB).
    """
    results = []
    for eval_root in _unique_dirs((
        os.path.join(output_root, "eval"),
        output_root,
        os.path.join(output_root, "pt"),
    )):
        for entry in sorted(glob.glob(os.path.join(eval_root, "eval_*"))):
            if not os.path.isdir(entry):
                continue
            eval_pt = _find_eval_pt(entry)
            info = {
                "path": entry,
                "dirname": os.path.basename(entry),
                "has_eval_pt": eval_pt is not None,
            }
            summary = _read_summary_json(entry)
            if summary:
                info.update(summary)
            elif eval_pt:
                info.update(_eval_pt_file_meta(eval_pt))
            results.append(info)
    return results


def _read_summary_json(eval_dir: str) -> Optional[dict]:
    """Read evaluator sidecar summary (fast, no tensor loading)."""
    import json
    summary_path = os.path.join(eval_dir, "summary.json")
    try:
        with open(summary_path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return None
        out = {}
        for key in ("num_molecules", "mean_vina", "best_vina",
                    "n_reconstruct_success", "n_eval_success", "n_complete"):
            if data.get(key) is not None:
                out[key] = data[key]
        return out or None
    except (OSError, ValueError):
        return None


def _eval_pt_file_meta(eval_pt_path: str) -> dict:
    """Cheap file metadata for legacy eval dirs (no deserialization)."""
    try:
        stat = os.stat(eval_pt_path)
        return {
            "eval_pt_size_mb": round(stat.st_size / 1024 / 1024, 2),
            "eval_pt_modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        }
    except OSError:
        return {}


def load_pt_metadata(pt_path: str, owner: Optional[str] = None) -> dict:
    """Read sampling/evaluation metadata with the existing output trust policy."""
    try:
        from server.molecule_ingest import _load_eval_pt
        # Generated results embed ProteinLigandData; uploads remain strict.
        data = _load_eval_pt(pt_path, owner=owner)
        meta = {
            "path": pt_path,
            "num_molecules": 0,
            "fields": list(data.keys()) if isinstance(data, dict) else [],
        }
        if isinstance(data, dict):
            for key in ("pred_ligand_pos", "molecules", "molecules_with_pos", "atom_types"):
                if key in data and isinstance(data[key], (list, torch.Tensor)):
                    meta["num_molecules"] = len(data[key]) if isinstance(data[key], list) else data[key].shape[0]
                    break
            if "meta" in data and isinstance(data["meta"], dict):
                meta["meta"] = {k: str(v)[:200] for k, v in data["meta"].items()}
                records = data['meta'].get('records', [])
                generated = [r for r in records if not r.get('is_original')]
                meta['num_generated'] = len(generated)
                fields = ('status', 'n_scaffold', 'n_extra', 'initial_extra_positions',
                          'extra_placement', 'site_allocation', 'forward_noise_init_mode')
                meta['samples'] = [{k: r[k] for k in fields if k in r} for r in generated]
        return meta
    except Exception as e:
        return {"path": pt_path, "error": str(e)}


_GPU_CACHE: Dict[str, object] = {"ts": 0.0, "data": []}
_GPU_CACHE_TTL = 5.0


def get_gpu_info() -> List[dict]:
    """Query GPU status via nvidia-smi, cached for a few seconds.

    The UI polls /api/stats every few seconds; spawning nvidia-smi per
    request is wasteful, so serve from cache within the TTL.
    """
    import subprocess
    import time
    now = time.monotonic()
    cached = _GPU_CACHE["data"]
    if cached and now - _GPU_CACHE["ts"] < _GPU_CACHE_TTL:
        return list(cached)
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            timeout=5, stderr=subprocess.DEVNULL,
        ).decode().strip()
        gpus = []
        for line in out.splitlines():
            parts = [x.strip() for x in line.split(",")]
            if len(parts) >= 5:
                gpus.append({
                    "index": int(parts[0]),
                    "name": parts[1],
                    "memory_used_mb": int(parts[2]),
                    "memory_total_mb": int(parts[3]),
                    "utilization_pct": int(parts[4]),
                })
        if gpus:
            _GPU_CACHE["ts"] = now
            _GPU_CACHE["data"] = gpus
        return gpus
    except Exception:
        return list(cached)


def list_proteins(data_root: str) -> List[dict]:
    """List available protein pockets from the data directory."""
    pocket_dir = os.path.join(data_root, "crossdocked_v1.1_rmsd1.0_pocket10")
    if not os.path.isdir(pocket_dir):
        return []
    proteins = []
    for pdb in sorted(glob.glob(os.path.join(pocket_dir, "**", "*.pdb"), recursive=True)):
        proteins.append({
            "path": pdb,
            "name": os.path.basename(pdb).replace(".pdb", ""),
            "size_kb": round(os.path.getsize(pdb) / 1024, 1),
        })
    return proteins[:500]  # cap for performance


# ── Helpers ──────────────────────────────────────────────────────────────

def _extract_data_id(path: str) -> Optional[int]:
    m = re.search(r"result_(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else None


def _find_eval_pt(eval_dir: str) -> Optional[str]:
    for f in glob.glob(os.path.join(eval_dir, "eval_results_*final*.pt")):
        return f
    for f in glob.glob(os.path.join(eval_dir, "eval_results_*.pt")):
        return f
    return None


def _is_within_safe(path: str, root: str) -> bool:
    try:
        return os.path.commonpath([path, root]) == root
    except ValueError:
        return False


def browse_directory(root: str, subpath: str = "", extensions: Optional[tuple] = None,
                     extra_roots: Optional[List[str]] = None) -> List[dict]:
    """List files under an allowed root for UI file picker.

    ``extra_roots`` covers directories reached via top-level symlinks under
    root (their realpath escapes root); '..' is still rejected by callers.
    """
    abs_root = os.path.realpath(root)
    target = os.path.realpath(os.path.join(abs_root, subpath)) if subpath else abs_root
    if not _is_within_safe(target, abs_root):
        allowed = any(_is_within_safe(target, os.path.realpath(er))
                      for er in (extra_roots or []) if os.path.isdir(er))
        if not allowed:
            return []
    if not os.path.isdir(target):
        return []
    entries = []
    try:
        for name in sorted(os.listdir(target))[:300]:
            full = os.path.join(target, name)
            if os.path.isdir(full):
                entries.append({"name": name, "path": full, "type": "dir"})
            elif extensions is None or name.lower().endswith(extensions):
                entries.append({
                    "name": name, "path": full, "type": "file",
                    "size_kb": round(os.path.getsize(full) / 1024, 1),
                })
    except OSError:
        pass
    return entries
