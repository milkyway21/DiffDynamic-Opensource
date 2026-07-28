"""Evaluation and extraction runners (CPU-bound)."""

from __future__ import annotations

import logging
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
)

logger = logging.getLogger(__name__)


def run_evaluation(scheduler, rec, params: dict):
    cfg = get_config()
    try:
        update_run(rec.run_id, status="running", started_at=datetime.now(timezone.utc))
        cmd = [cfg.python_bin, cfg.evaluate_script(), params["pt_path"]]
        if params.get("protein_root"):
            cmd += ["--protein_root", params["protein_root"]]
        if params.get("max_samples"):
            cmd += ["--max_samples", str(params["max_samples"])]
        if params.get("vina_timeout"):
            cmd += ["--vina-timeout-seconds", str(params["vina_timeout"])]
        if params.get("vina_modes"):
            cmd += ["--vina-modes", str(params["vina_modes"])]

        def on_line(line: str):
            append_log(rec, line)
            parse_eval_progress(rec, line)

        rc = run_cmd(
            rec, cmd, on_line=on_line,
            env_extra={
                "ADT_PYTHON": cfg.python_bin,
                "VINA_DOCK_TIMEOUT_SEC": str(params.get("vina_timeout", 20)),
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
        cmd = [cfg.python_bin, cfg.extract_script()]
        if params.get("protein_root"):
            cmd += ["--protein_root", params["protein_root"]]
        if params.get("remove_fragments"):
            cmd.append("--remove-fragments")
        cmd.append(params["pt_path"])
        if params.get("eval_dir"):
            cmd += ["--eval_dir", params["eval_dir"]]

        def on_line(line: str):
            append_log(rec, line)

        rc = run_cmd(rec, cmd, on_line=on_line, env_extra={"ADT_PYTHON": cfg.python_bin})
        if rc == 0:
            mark_completed(rec, "extraction_completed")
        else:
            mark_failed(rec, rec.error or f"Exit code {rc}")
    except Exception as e:
        mark_failed(rec, str(e))
    finally:
        scheduler._release_and_dispatch(rec)
