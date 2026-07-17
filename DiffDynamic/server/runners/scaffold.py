"""Optimization, scaffold, and scaffold-cascade runners."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from server.config import get_config
from server.database import update_run
from server.runners.base import (
    mark_completed,
    mark_failed,
    parse_generation_progress,
    run_cmd,
    auto_chain,
    append_log,
)

logger = logging.getLogger(__name__)


def _build_sample_cmd(cfg, patched_config: str, params: dict) -> list:
    cmd = [
        cfg.python_bin, cfg.sample_script(),
        patched_config,
        "--device", f"cuda:{params['gpu_id']}",
    ]
    data_id = params.get("data_id")
    if data_id is not None:
        cmd += ["-i", str(data_id)]
    elif params.get("protein_path"):
        cmd += ["--protein_path", params["protein_path"]]
        if params.get("ligand_path"):
            cmd += ["--ligand_path", params["ligand_path"]]
    elif params.get("use_test_set"):
        cmd += ["--use_test_set"]
    if params.get("molecule_path"):
        cmd += ["--molecule_path", params["molecule_path"]]
    return cmd


def run_optimization(scheduler, rec, params: dict):
    cfg = get_config()
    try:
        update_run(rec.run_id, status="running", started_at=datetime.now(timezone.utc))
        patched_config = params.get("patched_config", cfg.sampling_config)
        cmd = _build_sample_cmd(cfg, patched_config, params)

        def on_line(line: str):
            append_log(rec, line)
            parse_generation_progress(rec, line)

        rc = run_cmd(rec, cmd, on_line=on_line)
        if rc == 0:
            mark_completed(rec, "optimization_completed")
            auto_chain(scheduler, rec, params)
        else:
            mark_failed(rec, rec.error or f"Exit code {rc}", "optimization_failed")
    except Exception as e:
        mark_failed(rec, str(e), "optimization_failed")
    finally:
        scheduler._release_and_dispatch(rec)


def run_scaffold(scheduler, rec, params: dict):
    cfg = get_config()
    try:
        update_run(rec.run_id, status="running", started_at=datetime.now(timezone.utc))
        patched_config = params.get("patched_config", cfg.sampling_config)
        cmd = _build_sample_cmd(cfg, patched_config, params)

        def on_line(line: str):
            append_log(rec, line)
            parse_generation_progress(rec, line)

        rc = run_cmd(rec, cmd, on_line=on_line)
        if rc == 0:
            mark_completed(rec, "scaffold_completed")
            auto_chain(scheduler, rec, params)
        else:
            mark_failed(rec, rec.error or f"Exit code {rc}", "scaffold_failed")
    except Exception as e:
        mark_failed(rec, str(e), "scaffold_failed")
    finally:
        scheduler._release_and_dispatch(rec)


def run_scaffold_cascade(scheduler, rec, params: dict):
    cfg = get_config()
    try:
        update_run(rec.run_id, status="running", started_at=datetime.now(timezone.utc))
        cmd = [
            cfg.python_bin, cfg.scaffold_cascade_script(),
            "--device", f"cuda:{params['gpu_id']}",
            "--samples_per_round", str(params.get("samples_per_round", 5)),
            "--rounds", str(params.get("rounds", 2)),
        ]
        if params.get("data_id") is not None:
            cmd += ["-i", str(params["data_id"])]
        if params.get("protein_path"):
            cmd += ["--protein_path", params["protein_path"]]
        if params.get("ligand_path"):
            cmd += ["--ligand_path", params["ligand_path"]]
        if params.get("config_path"):
            cmd += ["--config", params["config_path"]]

        def on_line(line: str):
            extra = "[R" in line or "Cascade" in line or "Merge" in line
            append_log(rec, line, extra_keep=extra)
            parse_generation_progress(rec, line)
            if "Cascade complete" in line and "Output:" in line:
                import re
                m = re.search(r"Output:\s*(/\S+\.pt)", line)
                if m:
                    rec.output_path = m.group(1)

        rc = run_cmd(rec, cmd, on_line=on_line)
        if rc == 0:
            mark_completed(rec, "scaffold_cascade_completed")
            auto_chain(scheduler, rec, params)
        else:
            mark_failed(rec, rec.error or f"Exit code {rc}", "scaffold_cascade_failed")
    except Exception as e:
        mark_failed(rec, str(e), "scaffold_cascade_failed")
    finally:
        scheduler._release_and_dispatch(rec)
