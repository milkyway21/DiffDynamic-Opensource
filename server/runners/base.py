"""Shared runner utilities: progress parsing, finish helpers, config validation."""

from __future__ import annotations

import logging
import os
import re
import time
from datetime import datetime, timezone
from typing import Callable, Dict, Optional, Tuple

from server.config import get_config
from server.database import update_run, log_history
from server.job_helpers import is_important_log, make_job_env, run_subprocess

logger = logging.getLogger(__name__)

# ── Progress DB-write throttle ────────────────────────────────────────────
# Progress lines arrive per diffusion step / per molecule; persisting each one
# hammers SQLite. We coalesce: at most one UPDATE every ``_MIN_INTERVAL`` and
# only when progress actually moved by ``_MIN_DELTA``.
_PROGRESS_STATE: Dict[int, Tuple[float, float]] = {}
_MIN_INTERVAL = 2.0
_MIN_DELTA = 0.01


def _write_progress_throttled(rec, force: bool = False):
    now = time.monotonic()
    state = _PROGRESS_STATE.get(rec.run_id)
    if state is None:
        _PROGRESS_STATE[rec.run_id] = (now, rec.progress)
        try:
            update_run(rec.run_id, progress=rec.progress)
        except Exception:
            pass
        return
    last_t, last_p = state
    if force or (
        now - last_t >= _MIN_INTERVAL and abs(rec.progress - last_p) >= _MIN_DELTA
    ):
        _PROGRESS_STATE[rec.run_id] = (now, rec.progress)
        try:
            update_run(rec.run_id, progress=rec.progress)
        except Exception:
            pass


def _clear_progress_state(rec):
    _PROGRESS_STATE.pop(rec.run_id, None)


def validate_config_path(config_path: Optional[str], cfg=None) -> str:
    cfg = cfg or get_config()
    if not config_path:
        return cfg.sampling_config
    abs_path = os.path.realpath(config_path)
    allowed_dir = os.path.realpath(os.path.join(cfg.diffdynamic_root, "configs"))
    try:
        inside_allowed = os.path.commonpath([abs_path, allowed_dir]) == allowed_dir
    except ValueError:
        inside_allowed = False
    if not inside_allowed:
        raise ValueError(f"config_path must be under {allowed_dir}")
    if not os.path.isfile(abs_path):
        raise FileNotFoundError(f"Config file not found: {abs_path}")
    return abs_path


def read_config_snapshot(path: str) -> str:
    try:
        with open(path) as f:
            return f.read()
    except Exception:
        return ""


def append_log(rec, line: str, extra_keep: bool = False):
    if is_important_log(line) or extra_keep:
        rec.log_lines.append(line)


def mark_completed(rec, history_action: Optional[str] = None):
    rec.status = "completed"
    rec.finished_at = datetime.now(timezone.utc)
    _clear_progress_state(rec)
    update_run(
        rec.run_id,
        status="completed",
        finished_at=rec.finished_at,
        output_path=rec.output_path,
        progress=1.0,
        log_output="\n".join(rec.log_lines[-500:]),
    )
    if history_action:
        log_history(
            history_action,
            {"job_id": rec.job_id, "run_id": rec.run_id},
            user=getattr(rec, "owner", None) or "legacy",
        )


def mark_failed(rec, error: str, history_action: Optional[str] = None):
    rec.status = "failed"
    rec.error = error
    rec.finished_at = datetime.now(timezone.utc)
    _clear_progress_state(rec)
    update_run(
        rec.run_id,
        status="failed",
        finished_at=rec.finished_at,
        error_message=error,
        log_output="\n".join(rec.log_lines[-500:]),
    )
    if history_action:
        log_history(
            history_action,
            {"job_id": rec.job_id, "error": error},
            user=getattr(rec, "owner", None) or "legacy",
        )


def parse_generation_progress(rec, line: str):
    m = re.search(r"(\d+)/(\d+)\s+steps?", line)
    if m:
        current, total = int(m.group(1)), int(m.group(2))
        if total > 0:
            rec.progress = current / total
            _write_progress_throttled(rec)
    if ".pt" in line and ("saved" in line.lower() or "Results" in line):
        m2 = re.search(r"(/[\w./\-]+\.pt)", line)
        if m2:
            rec.output_path = m2.group(1)


def parse_eval_progress(rec, line: str):
    m = re.search(r"(\d+)/(\d+)", line)
    if m and ("molecule" in line.lower() or "评估分子" in line or "it/s" in line):
        current, total = int(m.group(1)), int(m.group(2))
        if total > 0:
            rec.progress = current / total
            _write_progress_throttled(rec)
    if "评估输出目录" in line:
        m2 = re.search(r"评估输出目录:\s*(.+)", line)
        if m2:
            rec.output_path = m2.group(1).strip()
    elif "输出目录" in line and "eval_" in line:
        m2 = re.search(r"(/[\w./\-]+eval_[\w./\-]+)", line)
        if m2:
            rec.output_path = m2.group(1)
    if "结果已保存至" in line or "Results saved" in line:
        m2 = re.search(r"(/[\w./\-]+\.pt)", line)
        if m2:
            pt_path = m2.group(1)
            eval_dir = os.path.dirname(pt_path)
            if "eval_" in eval_dir:
                rec.output_path = eval_dir


def parse_pocket_eval_progress(rec, line: str):
    # Prefer the concrete run subfolder (…/pocket_quality_vis/<id>_<ts>)
    if "Visualization dir ->" in line or "可视化目录" in line:
        m = re.search(r"(?:->|至)\s*(/\S+)", line)
        if m:
            path = m.group(1).strip().rstrip("/")
            if not path.endswith(".csv") and not path.endswith(".xlsx"):
                rec.output_path = path
    elif "Visualization output ->" in line or "可视化输出" in line:
        m = re.search(r"(?:->|至)\s*(/\S+)", line)
        if m:
            path = m.group(1).strip().rstrip("/")
            # Root only; keep if we do not yet have a concrete subfolder
            if not rec.output_path or rec.output_path.rstrip("/").endswith("pocket_quality_vis"):
                if not path.endswith(".csv"):
                    rec.output_path = path
    # Match …/pocket_quality_vis/<subdir> but never evaluation_records.csv
    if "pocket_quality_vis/" in line and "/" in line:
        m = re.search(r"(/[\w./\-]+pocket_quality_vis/[\w.\-]+)", line)
        if m:
            path = m.group(1).strip().rstrip("/")
            base = os.path.basename(path)
            if (
                base
                and not base.endswith(".csv")
                and not base.endswith(".png")
                and base != "evaluation_records.csv"
                and base != "pocket_quality_vis"
            ):
                rec.output_path = path
    if "overall" in line.lower() and ("score" in line.lower() or "分" in line):
        m = re.search(r"(\d+\.\d+)", line)
        if m:
            try:
                rec.progress = min(0.9, 0.5 + float(m.group(1)) * 0.4)
            except ValueError:
                pass
    if "=== Phase 2" in line:
        rec.progress = 0.4
        _write_progress_throttled(rec, force=True)


def strict_load_env_for_pt(pt_path: str) -> dict:
    """Env marking a .pt as untrusted for subprocess loaders.

    User-uploaded or otherwise external .pt must never trigger the
    weights_only=False pickle fallback in eval/pocket subprocesses (it would
    allow arbitrary code execution via a crafted pickle). Only files under
    the repo outputs tree are considered generated-and-trusted."""
    try:
        from server.molecule_ingest import OUTPUTS_ROOT
        real = os.path.realpath(str(pt_path or ""))
        if real and os.path.commonpath([real, OUTPUTS_ROOT]) == OUTPUTS_ROOT:
            return {}
    except (ValueError, OSError, ImportError):
        pass
    return {"DD_PT_STRICT_LOAD": "1"}


def run_cmd(rec, cmd, on_line: Optional[Callable[[str], None]] = None, env_extra=None) -> int:
    """Run a DiffDynamic subprocess with timeout and log streaming."""
    cfg = get_config()
    env = make_job_env(env_extra)

    def _wrapped(line: str):
        if on_line:
            on_line(line)
        else:
            append_log(rec, line)

    return run_subprocess(rec, cmd, cwd=cfg.diffdynamic_root, env=env, on_line=_wrapped)


def auto_chain(scheduler, rec, params: dict):
    """Run auto-evaluate and auto-extract after successful generation."""
    if params.get("medchem"):
        from server.runners.medchem import chain_medchem
        try:
            chain_medchem(scheduler, rec, params)
        except Exception as exc:
            import json
            update_run(rec.run_id, progress_detail=json.dumps({"medchem_error": str(exc)}))
        return
    cfg = get_config()
    auto_eval = params.get("auto_evaluate", False)
    auto_ext = params.get("auto_extract", False)
    vina_timeout = params.get("vina_timeout", 20)
    max_samples = params.get("max_samples", 5)
    owner = params.get("owner") or getattr(rec, "owner", None) or "legacy"
    if not ((auto_eval or auto_ext) and rec.output_path):
        return
    # 评估子进程强制要求 protein_root 存在（自定义模式的蛋白已以绝对路径
    # 存入 pt，此处只需保证目录有效）：自定义任务用上传蛋白所在目录，
    # 测试集任务用 pocket10 数据集根（脚本默认的 crossdocked_v1.1_rmsd1.0
    # 在本机不存在）。
    protein_root = params.get("protein_root") or (
        os.path.dirname(params["protein_path"]) if params.get("protein_path")
        else os.path.join(cfg.data_root, "crossdocked_pocket10_test_only")
    )
    logger.info("Auto-chain: eval=%s, extract=%s, pt=%s", auto_eval, auto_ext, rec.output_path)
    if auto_eval:
        eval_result = scheduler.submit_evaluation(
            pt_path=rec.output_path,
            protein_root=protein_root,
            max_samples=max_samples,
            vina_timeout=vina_timeout,
            pocket_name=params.get("pocket_name"),
            triggered_by=owner,
            skip_daily_quota=bool(params.get("_skip_daily_quota")),
        )
        if "error" not in eval_result:
            log_history(
                "auto_chain_evaluate",
                {"parent_job": rec.job_id, "eval_job": eval_result["job_id"]},
                user=owner,
            )
    if auto_ext:
        ext_result = scheduler.submit_extraction(
            pt_path=rec.output_path,
            protein_root=protein_root,
            remove_fragments=params.get("remove_fragments", True),
            max_samples=max_samples,
            vina_timeout=vina_timeout if auto_eval else None,
            triggered_by=owner,
            parent_run_id=rec.run_id,
        )
        if "error" not in ext_result:
            log_history(
                "auto_chain_extract",
                {"parent_job": rec.job_id, "extract_job": ext_result["job_id"]},
                user=owner,
            )
