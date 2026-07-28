"""Shared runner utilities: progress parsing, finish helpers, config validation."""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from typing import Callable, Optional

from server.config import get_config
from server.database import update_run, log_history
from server.job_helpers import is_important_log, make_job_env, run_subprocess

logger = logging.getLogger(__name__)


def validate_config_path(config_path: Optional[str], cfg=None) -> str:
    cfg = cfg or get_config()
    if not config_path:
        return cfg.sampling_config
    abs_path = os.path.realpath(config_path)
    allowed_dir = os.path.realpath(os.path.join(cfg.diffdynamic_root, "configs"))
    if not abs_path.startswith(allowed_dir):
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
    update_run(
        rec.run_id,
        status="completed",
        finished_at=rec.finished_at,
        output_path=rec.output_path,
        progress=1.0,
        log_output="\n".join(rec.log_lines[-500:]),
    )
    if history_action:
        log_history(history_action, {"job_id": rec.job_id, "run_id": rec.run_id})


def mark_failed(rec, error: str, history_action: Optional[str] = None):
    rec.status = "failed"
    rec.error = error
    rec.finished_at = datetime.now(timezone.utc)
    update_run(
        rec.run_id,
        status="failed",
        finished_at=rec.finished_at,
        error_message=error,
        log_output="\n".join(rec.log_lines[-500:]),
    )
    if history_action:
        log_history(history_action, {"job_id": rec.job_id, "error": error})


def parse_generation_progress(rec, line: str):
    m = re.search(r"(\d+)/(\d+)\s+steps?", line)
    if m:
        current, total = int(m.group(1)), int(m.group(2))
        if total > 0:
            rec.progress = current / total
            update_run(
                rec.run_id,
                progress=rec.progress,
                progress_detail=json.dumps({"step": current, "total": total}),
            )
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
            update_run(rec.run_id, progress=rec.progress)
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
        update_run(rec.run_id, progress=0.4)


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
    cfg = get_config()
    auto_eval = params.get("auto_evaluate", False)
    auto_ext = params.get("auto_extract", False)
    vina_timeout = params.get("vina_timeout", 20)
    max_samples = params.get("max_samples", 5)
    if not ((auto_eval or auto_ext) and rec.output_path):
        return
    logger.info("Auto-chain: eval=%s, extract=%s, pt=%s", auto_eval, auto_ext, rec.output_path)
    if auto_eval:
        eval_result = scheduler.submit_evaluation(
            pt_path=rec.output_path,
            max_samples=max_samples,
            vina_timeout=vina_timeout,
            pocket_name=params.get("pocket_name"),
            triggered_by="auto_chain",
        )
        if "error" not in eval_result:
            log_history(
                "auto_chain_evaluate",
                {"parent_job": rec.job_id, "eval_job": eval_result["job_id"]},
            )
    if auto_ext:
        ext_result = scheduler.submit_extraction(
            pt_path=rec.output_path,
            protein_root=params.get("protein_root")
            or os.path.join(cfg.data_root, "crossdocked_pocket10_test_only"),
            remove_fragments=params.get("remove_fragments", True),
            triggered_by="auto_chain",
        )
        if "error" not in ext_result:
            log_history(
                "auto_chain_extract",
                {"parent_job": rec.job_id, "extract_job": ext_result["job_id"]},
            )
