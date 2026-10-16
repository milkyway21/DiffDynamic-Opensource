"""Generation runners: single, batch, and custom pocket sampling."""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

from server.config import get_config
from server.database import update_run
from server.job_helpers import dynamic_web_patches, write_patched_config
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
        validated = write_patched_config(validated, dynamic_web_patches(batch_size), rec.job_id)
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
        if params.get("sample_only") or not params.get("auto_evaluate") or params.get("medchem"):
            cmd.append("--sample-only")

        env_extra = {}
        if params.get("result_path"):
            batch_root = Path(params["result_path"])
            if params.get("medchem"):
                batch_root = batch_root / ("batch_" + rec.job_id)
                batch_root.mkdir(parents=True, exist_ok=True)
            env_extra["DD_RESULT_PATH"] = str(batch_root)

        completed_paths = []

        def on_line(line: str):
            append_log(rec, line)
            saved = re.search(r"(?:Results saved to:|采样成功:)\s*(.+?\.pt)\s*$", line)
            if saved and Path(saved.group(1)).is_file():
                completed_paths.append(saved.group(1))
                rec.output_path = saved.group(1)
            if ".pt" in line:
                m = re.search(r"(/[\w./\-]+\.pt)", line)
                if m:
                    rec.output_path = m.group(1)

        rc = run_cmd(rec, cmd, on_line=on_line, env_extra=env_extra or None)
        if rc == 0:
            paths = list(dict.fromkeys(completed_paths))
            if params.get("medchem") and not paths:
                # Checkpoints, trajectories and proposal pools are not final pockets.
                final_name = re.compile(r"result_\d+_(?:it\d+_)?\d{8}_\d{6}\.pt")
                paths = sorted(str(p) for p in batch_root.glob("result_*.pt") if final_name.fullmatch(p.name))
            if paths:
                import json
                update_run(rec.run_id, progress_detail=json.dumps({"medchem_source_pts": paths}))
            mark_completed(rec, "batch_generation_completed")
            if params.get("medchem"):
                from server.runners.medchem import chain_medchem

                try:
                    chain_medchem(scheduler, rec, params, pt_paths=paths)
                except Exception as exc:
                    import json
                    update_run(rec.run_id, progress_detail=json.dumps({"medchem_error": str(exc)}))
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
        # 优先用 sed 后的 patched config（prudent 参数已注入）
        use_cfg = params.get("patched_config") or validate_config_path(config_path, cfg)
        cmd = [
            cfg.python_bin, cfg.sample_script(),
            use_cfg,
            "--device", f"cuda:{params['gpu_id']}",
        ]
        data_id = params.get("data_id")
        if data_id is not None:
            cmd += ["-i", str(data_id)]
        elif params.get("use_test_set"):
            cmd += ["--use_test_set"]
        if params.get("mode") and params.get("mode") not in ("dynamic",):
            cmd += ["--mode", params["mode"]]
        if params.get("result_path"):
            cmd += ["--result_path", params["result_path"]]

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
        validated_config = write_patched_config(
            validated_config, dynamic_web_patches(num_samples), rec.job_id,
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
        if params.get("result_path"):
            cmd += ["--result_path", params["result_path"]]

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
