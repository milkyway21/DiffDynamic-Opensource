"""Shared helpers for job subprocess execution and result parsing."""

import copy
import csv
import os
import re
import subprocess
import threading
from typing import Callable, Dict, List, Optional

import yaml

from server.config import get_config
from server.database import add_evaluation, update_run


def release_gpu_safe(allocator, gpu_id: int):
    if gpu_id is not None and gpu_id >= 0:
        allocator.release(gpu_id)


def cleanup_temp_config(job_id: str):
    cfg = get_config()
    path = os.path.join(cfg.diffdynamic_root, "configs", f"run_{job_id}.yml")
    try:
        if os.path.isfile(path):
            os.remove(path)
    except OSError:
        pass


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base dict (mutates base)."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            deep_merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def write_patched_config(base_config_path: str, patches: dict, job_id: str) -> str:
    """Read base YAML, deep-merge patches, write to a temp config file."""
    with open(base_config_path) as f:
        base = yaml.safe_load(f) or {}
    deep_merge(base, patches)
    cfg = get_config()
    out_dir = os.path.join(cfg.diffdynamic_root, "configs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"run_{job_id}.yml")
    with open(out_path, "w") as f:
        yaml.dump(base, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    return out_path


def is_important_log(line: str) -> bool:
    """Filter: only keep progress, results, errors, key milestones."""
    if not line or line.strip() == "":
        return False
    if "%|" in line and "it/s" in line:
        return True
    if any(kw in line for kw in ("❌", "Error", "error:", "ERROR", "Traceback", "FileNotFoundError", "失败")):
        return True
    if any(kw in line for kw in ("✅", "Results saved", "Sample done", "Peak Memory", "完成", "成功")):
        return True
    if any(kw in line for kw in ("加载", "Loaded", "评估", "重建", "对接", "分子", "采样", "进度")):
        return True
    if re.search(r"\[\d+/\d+\]", line):
        return True
    return False


def make_job_env(extra: Optional[dict] = None) -> dict:
    """Standard env for DiffDynamic subprocesses."""
    cfg = get_config()
    env = os.environ.copy()
    env["PYTHONPATH"] = cfg.diffdynamic_root
    env["PYTHONUNBUFFERED"] = "1"
    if extra:
        env.update(extra)
    return env


def run_subprocess(
    rec,
    cmd: List[str],
    cwd: str,
    env: dict,
    timeout: Optional[int] = None,
    on_line: Optional[Callable[[str], None]] = None,
    sync_every: int = 20,
) -> int:
    """Run subprocess, stream stdout, enforce timeout. Returns exit code."""
    cfg = get_config()
    if timeout is None:
        timeout = cfg.job_timeout

    rec.process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, env=env, cwd=cwd, bufsize=1,
    )

    sync_counter = 0
    timed_out = [False]

    def _killer():
        if rec.process and rec.process.poll() is None:
            timed_out[0] = True
            rec.process.terminate()
            try:
                rec.process.wait(timeout=10)
            except Exception:
                rec.process.kill()
                rec.process.wait()

    timer = threading.Timer(timeout, _killer)
    timer.daemon = True
    timer.start()

    try:
        for line in rec.process.stdout:
            line = line.rstrip()
            if on_line:
                on_line(line)
            sync_counter += 1
            if sync_counter % sync_every == 0:
                update_run(rec.run_id, progress=rec.progress,
                           log_output="\n".join(rec.log_lines[-500:]))
        rec.process.wait()
    finally:
        timer.cancel()

    if timed_out[0]:
        rec.error = f"Job timed out after {timeout}s"
        return -9
    return rec.process.returncode if rec.process.returncode is not None else -1


def parse_pocket_scores_from_csv(
    csv_path: str,
    vis_dir: Optional[str] = None,
    pt_path: Optional[str] = None,
) -> Dict[str, str]:
    """Match CSV row by vis_dir folder name or pt_path; fallback to last row."""
    if not os.path.isfile(csv_path):
        return {}
    rows = []
    with open(csv_path, encoding="utf-8") as cf:
        rows = list(csv.DictReader(cf))
    if not rows:
        return {}

    target = None
    folder = os.path.basename(vis_dir.rstrip("/")) if vis_dir else ""
    pt_norm = os.path.realpath(pt_path) if pt_path else ""

    if folder or pt_norm:
        for row in reversed(rows):
            row_dir = (row.get("vis_dir") or row.get("output_dir") or row.get("folder") or "").strip()
            row_pt = (row.get("pt_path") or "").strip()
            if folder and row_dir:
                row_folder = os.path.basename(row_dir.rstrip("/"))
                if folder == row_folder or folder in row_dir or row_folder in (vis_dir or ""):
                    target = row
                    break
            if pt_norm and row_pt:
                try:
                    if os.path.realpath(row_pt) == pt_norm:
                        target = row
                        break
                except OSError:
                    if row_pt == pt_path:
                        target = row
                        break
    if target is None:
        target = rows[-1]

    keys = ("score_a", "score_b", "score_c", "score_d", "score_e", "score_f", "score_g", "score_h",
            "overall_score", "overall_label")
    return {k: target.get(k, "N/A") for k in keys if k in target or k.startswith("score") or k.startswith("overall")}


def store_pocket_evaluations(run_id: int, scores: dict):
    """Persist 8-dimension pocket scores to evaluations table."""
    dim_map = {
        "score_a": "vina_docking",
        "score_b": "clustering",
        "score_c": "ligand_efficiency",
        "score_d": "drug_likeness",
        "score_e": "completeness",
        "score_f": "diversity",
        "score_g": "size_consistency",
        "score_h": "pocket_volume",
    }
    for key, label in dim_map.items():
        val = scores.get(key)
        if val is not None and val != "N/A":
            try:
                add_evaluation(run_id, label, metric_value=float(val), details={"dim": key})
            except (TypeError, ValueError):
                add_evaluation(run_id, label, details={"raw": str(val), "dim": key})
    overall = scores.get("overall_score")
    if overall is not None and overall != "N/A":
        try:
            add_evaluation(run_id, "overall_pocket_score", metric_value=float(overall),
                           details={"label": scores.get("overall_label"), "dim": "overall_score"})
        except (TypeError, ValueError):
            add_evaluation(run_id, "overall_pocket_score", details=scores)
