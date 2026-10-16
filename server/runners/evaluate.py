"""Evaluation and extraction runners (CPU-bound)."""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

from server.config import get_config
from server.database import update_run, log_history
from server.molecule_ingest import ingest_molecules_from_eval
from server.runners.base import (
    mark_completed,
    mark_failed,
    parse_eval_progress,
    run_cmd,
    append_log,
    strict_load_env_for_pt,
)

logger = logging.getLogger(__name__)


def _job_input_pt(rec, params):
    from server.safe_pt import prepare_job_pt
    return prepare_job_pt(params["pt_path"], rec.owner)


def run_evaluation(scheduler, rec, params: dict):
    cfg = get_config()
    try:
        update_run(rec.run_id, status="running", started_at=datetime.now(timezone.utc))
        input_pt = _job_input_pt(rec, params)
        cmd = [cfg.python_bin, cfg.evaluate_script(), input_pt]
        if params.get("protein_root"):
            cmd += ["--protein_root", params["protein_root"]]
        if params.get("max_samples"):
            cmd += ["--max_samples", str(params["max_samples"])]
        if params.get("vina_timeout"):
            cmd += ["--vina-timeout-seconds", str(params["vina_timeout"])]
        if params.get("vina_modes"):
            cmd += ["--vina-modes", str(params["vina_modes"])]
        if params.get("result_path"):
            output_dir = os.path.join(params["result_path"], f"eval_{rec.job_id}")
            cmd += ["--output_dir", output_dir]
        # 小样本任务无需中间快照，避免额外 JSON 写盘
        max_samples = params.get("max_samples")
        if max_samples is not None and 0 < int(max_samples) <= 16:
            cmd += ["--save_intermediate_interval", "0"]

        def on_line(line: str):
            append_log(rec, line)
            parse_eval_progress(rec, line)

        rc = run_cmd(
            rec, cmd, on_line=on_line,
            env_extra={
                "ADT_PYTHON": cfg.python_bin,
                "VINA_DOCK_TIMEOUT_SEC": str(params.get("vina_timeout", 20)),
                **strict_load_env_for_pt(input_pt),
            },
        )
        if rc == 0:
            mark_completed(rec, "evaluation_completed")
            if rec.output_path:
                try:
                    ingest_molecules_from_eval(
                        rec.run_id, rec.output_path, pocket_name=params.get("pocket_name"),
                    )
                    log_history(
                        "molecules_ingested",
                        {"run_id": rec.run_id, "eval_dir": rec.output_path},
                        user=getattr(rec, "owner", None) or "legacy",
                    )
                except Exception as e:
                    logger.warning("Failed to ingest molecules: %s", e)
        else:
            mark_failed(rec, rec.error or f"Exit code {rc}")
    except Exception as e:
        mark_failed(rec, str(e))
    finally:
        # CPU jobs don't hold GPU, but still dispatch in case mixed queue
        scheduler._release_and_dispatch(rec)


def run_extraction(scheduler, rec, params: dict):
    cfg = get_config()
    try:
        update_run(rec.run_id, status="running", started_at=datetime.now(timezone.utc))
        input_pt = _job_input_pt(rec, params)
        cmd = [cfg.python_bin, cfg.extract_script()]
        if params.get("protein_root"):
            cmd += ["--protein_root", params["protein_root"]]
        if params.get("remove_fragments"):
            cmd.append("--remove-fragments")
        cmd.append(input_pt)
        output_dir = None
        if params.get("eval_dir"):
            cmd += ["--eval_dir", params["eval_dir"]]
            output_dir = params["eval_dir"]
        elif params.get("result_path"):
            output_dir = os.path.join(params["result_path"], f"extract_{rec.job_id}")
            cmd += ["--output_dir", output_dir]
        # Forward eval-side limits to avoid full-length evaluation on every molecule
        max_samples = params.get("max_samples")
        if max_samples:
            cmd += ["--max_samples", str(max_samples)]
        vina_timeout = params.get("vina_timeout")
        if vina_timeout:
            cmd += ["--vina-timeout-seconds", str(vina_timeout)]
        # 提取内嵌重评估：同样默认仅 score_only；dock 需显式传入（慢，前端有独立按钮）
        if params.get("vina_modes"):
            cmd += ["--vina-modes", str(params["vina_modes"])]

        def on_line(line: str):
            append_log(rec, line)

        rc = run_cmd(rec, cmd, on_line=on_line, env_extra={
            "ADT_PYTHON": cfg.python_bin, **strict_load_env_for_pt(input_pt),
        })
        if rc == 0:
            if output_dir:
                rec.output_path = output_dir
            mark_completed(rec, "extraction_completed")
            ingest_target = output_dir or params.get("pt_path")
            if ingest_target:
                try:
                    ingest_molecules_from_eval(
                        rec.run_id, ingest_target, pocket_name=params.get("pocket_name"),
                    )
                    parent_id = params.get("parent_run_id")
                    if parent_id:
                        try:
                            parent_id = int(parent_id)
                        except (TypeError, ValueError):
                            parent_id = None
                    if parent_id and parent_id != rec.run_id:
                        ingest_molecules_from_eval(
                            parent_id, ingest_target, pocket_name=params.get("pocket_name"),
                        )
                    log_history(
                        "molecules_ingested",
                        {
                            "run_id": rec.run_id,
                            "parent_run_id": parent_id,
                            "eval_dir": ingest_target,
                        },
                        user=getattr(rec, "owner", None) or "legacy",
                    )
                except Exception as e:
                    logger.warning("Failed to ingest molecules after extract: %s", e)
        else:
            mark_failed(rec, rec.error or f"Exit code {rc}")
    except Exception as e:
        mark_failed(rec, str(e))
    finally:
        scheduler._release_and_dispatch(rec)
