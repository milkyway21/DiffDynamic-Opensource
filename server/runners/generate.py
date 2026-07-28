"""Generation runners: single, batch, and custom pocket sampling."""

from __future__ import annotations

import logging
import os
import re
from datetime import datetime, timezone
from typing import Optional

from server.config import get_config
from server.database import update_run
from server.job_helpers import write_patched_config
from server.runners.base import (
    validate_config_path,
    mark_completed,
    mark_failed,
    parse_generation_progress,
    run_cmd,
    auto_chain,
    append_log,
)

logger = logging.getLogger(__name__)


def run_batch_generation(scheduler, rec, params: dict, config_path: Optional[str]):
    """Batch sampling via batch_sampleandeval_parallel.py.

    batch_size is applied via YAML patch (sample.large_step.batch_size),
    not as a CLI flag (the batch script does not accept --batch_size).
    """
    cfg = get_config()
    try:
        update_run(rec.run_id, status="running", started_at=datetime.now(timezone.utc))
        script = os.path.join(cfg.diffdynamic_root, "batch_sampleandeval_parallel.py")
        validated = validate_config_path(config_path, cfg)

        batch_size = params.get("batch_size")
        if batch_size:
            validated = write_patched_config(
                validated,
                {"sample": {"dynamic": {"large_step": {"batch_size": int(batch_size)}}}},
                rec.job_id,
            )
            params["patched_config"] = validated

        cmd = [
            cfg.python_bin, script,
            "--start", str(params.get("start_id", 0)),
            "--end", str(params.get("end_id", 0)),
        ]
        if params.get("gpus"):
            cmd += ["--gpus", str(params["gpus"])]
        elif params.get("gpu_id") is not None:
            cmd += ["--gpus", str(params["gpu_id"])]
        cmd += ["--config", validated]
        if params.get("sample_only") or not params.get("auto_evaluate"):
            cmd.append("--sample-only")

        def on_line(line: str):
            append_log(rec, line)
            if ".pt" in line:
                m = re.search(r"(/[\w./\-]+\.pt)", line)
                if m:
                    rec.output_path = m.group(1)

        rc = run_cmd(rec, cmd, on_line=on_line)
        if rc == 0:
            mark_completed(rec, "batch_generation_completed")
        else:
            mark_failed(rec, rec.error or f"Exit code {rc}", "batch_generation_failed")
    except Exception as e:
        mark_failed(rec, str(e), "batch_generation_failed")
    finally:
        scheduler._release_and_dispatch(rec)


def run_generation(scheduler, rec, params: dict, config_path: Optional[str]):
    cfg = get_config()
    try:
        update_run(rec.run_id, status="running", started_at=datetime.now(timezone.utc))
        validated_config = validate_config_path(config_path, cfg)
        cmd = [
            cfg.python_bin, cfg.sample_script(),
            validated_config,
            "--device", f"cuda:{params['gpu_id']}",
        ]
        data_id = params.get("data_id")
        if data_id is not None:
            cmd += ["-i", str(data_id)]
        elif params.get("use_test_set"):
            cmd += ["--use_test_set"]
        if params.get("batch_size"):
            cmd += ["--batch_size", str(params["batch_size"])]
        if params.get("mode"):
            cmd += ["--mode", params["mode"]]

        def on_line(line: str):
            append_log(rec, line)
            parse_generation_progress(rec, line)

        rc = run_cmd(rec, cmd, on_line=on_line)
        if rc == 0:
            mark_completed(rec, "generation_completed")
            auto_chain(scheduler, rec, params)
        else:
            mark_failed(rec, rec.error or f"Exit code {rc}", "generation_failed")
    except Exception as e:
        mark_failed(rec, str(e), "generation_failed")
    finally:
        scheduler._release_and_dispatch(rec)


def run_custom_generation(scheduler, rec, params: dict, config_path: Optional[str]):
    cfg = get_config()
    try:
        update_run(rec.run_id, status="running", started_at=datetime.now(timezone.utc))
        validated_config = validate_config_path(config_path, cfg)

        # Apply num_samples via YAML large_step.batch_size when provided
        num_samples = params.get("num_samples")
        if num_samples:
            validated_config = write_patched_config(
                validated_config,
                {"sample": {"dynamic": {"large_step": {"batch_size": int(num_samples)}}}},
                rec.job_id,
            )
            params["patched_config"] = validated_config

        cmd = [
            cfg.python_bin, cfg.sample_script(),
            validated_config,
            "--protein_path", params["protein_path"],
            "--device", f"cuda:{params['gpu_id']}",
        ]
        if params.get("ligand_path"):
            cmd += ["--ligand_path", params["ligand_path"]]
        cmd += ["--pocket_radius", str(params.get("pocket_radius", 10.0))]
        if num_samples:
            cmd += ["--batch_size", str(num_samples)]

        def on_line(line: str):
            append_log(rec, line)
            parse_generation_progress(rec, line)

        rc = run_cmd(rec, cmd, on_line=on_line)
        if rc == 0:
            mark_completed(rec, "custom_generation_completed")
            auto_chain(scheduler, rec, params)
        else:
            mark_failed(rec, rec.error or f"Exit code {rc}")
    except Exception as e:
        mark_failed(rec, str(e))
    finally:
        scheduler._release_and_dispatch(rec)
