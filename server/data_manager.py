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


def scan_outputs(output_root: str) -> List[dict]:
    """Scan output directory for result_*.pt files with metadata."""
    results = []
    if not os.path.isdir(output_root):
        return results
    # Scan outputs/pt/ subfolder (new structure)
    pt_dir = os.path.join(output_root, "pt")
    if os.path.isdir(pt_dir):
        for entry in sorted(glob.glob(os.path.join(pt_dir, "result_*.pt"))):
            stat = os.stat(entry)
            results.append({
                "path": entry,
                "filename": os.path.basename(entry),
                "size_mb": round(stat.st_size / 1024 / 1024, 2),
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "data_id": _extract_data_id(entry),
            })
    # Also scan outputs/ root for backwards compatibility
    for entry in sorted(glob.glob(os.path.join(output_root, "result_*.pt"))):
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
    """Scan for eval directories containing evaluation results."""
    results = []
    if not os.path.isdir(output_root):
        return results
    # Scan outputs/eval/ subfolder (new consolidated structure)
    eval_root = os.path.join(output_root, "eval")
    if os.path.isdir(eval_root):
        for entry in sorted(glob.glob(os.path.join(eval_root, "eval_*"))):
            if not os.path.isdir(entry):
                continue
            eval_pt = _find_eval_pt(entry)
            info = {
                "path": entry,
                "dirname": os.path.basename(entry),
                "has_eval_pt": eval_pt is not None,
            }
            if eval_pt:
                info.update(_load_eval_summary(eval_pt))
            results.append(info)
    # Also scan outputs/ root for backwards compatibility
    for entry in sorted(glob.glob(os.path.join(output_root, "eval_*"))):
        if not os.path.isdir(entry):
            continue
        eval_pt = _find_eval_pt(entry)
        info = {
            "path": entry,
            "dirname": os.path.basename(entry),
            "has_eval_pt": eval_pt is not None,
        }
        if eval_pt:
            info.update(_load_eval_summary(eval_pt))
        results.append(info)
    return results


def load_pt_metadata(pt_path: str) -> dict:
    """Load metadata from a result .pt file without loading full tensors."""
    try:
        try:
            data = torch.load(pt_path, map_location="cpu", weights_only=True)
        except Exception:
            data = torch.load(pt_path, map_location="cpu")
        meta = {
            "path": pt_path,
            "num_molecules": 0,
            "fields": list(data.keys()) if isinstance(data, dict) else [],
        }
        if isinstance(data, dict):
            for key in ("molecules", "molecules_with_pos", "atom_types"):
                if key in data and isinstance(data[key], (list, torch.Tensor)):
                    meta["num_molecules"] = len(data[key]) if isinstance(data[key], list) else data[key].shape[0]
                    break
            if "meta" in data and isinstance(data["meta"], dict):
                meta["meta"] = {k: str(v)[:200] for k, v in data["meta"].items()}
        return meta
    except Exception as e:
        return {"path": pt_path, "error": str(e)}


def get_gpu_info() -> List[dict]:
    """Query GPU status via nvidia-smi."""
    import subprocess
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
        return gpus
    except Exception:
        return []


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


def _load_eval_summary(eval_pt_path: str) -> dict:
    """Extract summary stats from an eval_results .pt file."""
    try:
        try:
            data = torch.load(eval_pt_path, map_location="cpu", weights_only=True)
        except Exception:
            data = torch.load(eval_pt_path, map_location="cpu")
        summary = {}
        if isinstance(data, dict):
            if "molecule_results" in data:
                mols = data["molecule_results"]
                summary["num_molecules"] = len(mols)
                vina_scores = [m.get("vina_score", {}).get("score_only", {}).get("affinity")
                               for m in mols if isinstance(m, dict)]
                vina_scores = [v for v in vina_scores if v is not None]
                if vina_scores:
                    summary["mean_vina"] = round(sum(vina_scores) / len(vina_scores), 2)
                    summary["best_vina"] = round(min(vina_scores), 2)
            if "summary" in data:
                summary["raw_summary"] = {k: str(v)[:100] for k, v in data["summary"].items()}
        return summary
    except Exception:
        return {}
