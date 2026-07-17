"""Pocket quality evaluation runner."""

from __future__ import annotations

import logging
import os
import re
from datetime import datetime, timezone

from server.config import get_config
from server.database import update_run, log_history
from server.job_helpers import parse_pocket_scores_from_csv, store_pocket_evaluations
from server.runners.base import (
    mark_completed,
    mark_failed,
    parse_pocket_eval_progress,
    parse_generation_progress,
    run_cmd,
    append_log,
)

logger = logging.getLogger(__name__)


def run_pocket_evaluation(scheduler, rec, params: dict):
    cfg = get_config()
    try:
        update_run(rec.run_id, status="running", started_at=datetime.now(timezone.utc))

        if params.get("generate_first") and not params.get("pt_path"):
            gen_cmd = [
                cfg.python_bin, cfg.sample_script(),
                cfg.sampling_config,
                "--device", f"cuda:{params['gpu_id']}",
                "--protein_path", params["protein_path"],
            ]
            if params.get("ligand_path"):
                gen_cmd += ["--ligand_path", params["ligand_path"]]
            gen_cmd += ["--pocket_radius", str(params.get("pocket_radius", 10.0))]
            if params.get("batch_size"):
                gen_cmd += ["--batch_size", str(params["batch_size"])]

            rec.log_lines.append("=== Phase 1: Generating molecules ===")

            def on_gen(line: str):
                append_log(rec, line)
                if ".pt" in line and ("saved" in line.lower() or "Results" in line):
                    m = re.search(r"(/[\w./\-]+\.pt)", line)
                    if m:
                        params["pt_path"] = m.group(1)

            rc = run_cmd(rec, gen_cmd, on_line=on_gen)
            if rc != 0:
                mark_failed(rec, rec.error or f"Generation failed with exit code {rc}")
                return
            rec.log_lines.append("=== Phase 2: Evaluating pocket quality ===")
            rec.progress = 0.3
            update_run(rec.run_id, progress=0.3)

        pt_path = params.get("pt_path")
        protein_path = params.get("protein_path")
        ligand_path = params.get("ligand_path")

        if pt_path:
            cmd = [
                cfg.python_bin, cfg.pocket_eval_script(),
                "--pt_file", pt_path,
                "--fpocket_protein_pdb", protein_path,
                "--visualize",
            ]
        else:
            cmd = [
                cfg.python_bin, cfg.pocket_eval_script(),
                "--eval_ligands", ligand_path,
                "--custom_pocket_pdb", protein_path,
                "--vina_outputs_dir", cfg.output_root,
                "--fpocket_protein_pdb", protein_path,
                "--visualize",
            ]

        if params.get("max_samples"):
            cmd += ["--idea_e_expected_n_molecules", str(params["max_samples"])]

        rec.log_lines.append(f"Running: {' '.join(cmd)}")

        def on_line(line: str):
            append_log(rec, line)
            parse_pocket_eval_progress(rec, line)

        rc = run_cmd(rec, cmd, on_line=on_line)
        if rc == 0:
            rec.progress = 1.0
            csv_path = os.path.join(
                cfg.diffdynamic_root, "pocket_quality_vis", "evaluation_records.csv",
            )
            # If progress parser missed the subfolder, recover from CSV by pt_path
            if (
                not rec.output_path
                or rec.output_path.endswith(".csv")
                or os.path.basename(rec.output_path.rstrip("/")) == "pocket_quality_vis"
            ):
                rec.output_path = None
            rec.scores = parse_pocket_scores_from_csv(
                csv_path, vis_dir=rec.output_path, pt_path=pt_path,
            )
            # Backfill vis_dir from latest matching CSV row if needed
            if not rec.output_path and os.path.isfile(csv_path):
                import csv as _csv
                with open(csv_path, encoding="utf-8") as cf:
                    rows = list(_csv.DictReader(cf))
                if rows:
                    last = rows[-1]
                    if pt_path and (last.get("pt_path") or "") == pt_path and last.get("vis_dir"):
                        rec.output_path = last["vis_dir"]
                    elif last.get("vis_dir"):
                        rec.output_path = last["vis_dir"]
            if rec.scores:
                store_pocket_evaluations(rec.run_id, rec.scores)
            mark_completed(rec)
            log_history(
                "pocket_eval_completed",
                {"job_id": rec.job_id, "vis_dir": rec.output_path, "scores": rec.scores},
            )
        else:
            mark_failed(rec, rec.error or f"Exit code {rc}")
    except Exception as e:
        mark_failed(rec, str(e))
    finally:
        scheduler._release_and_dispatch(rec)
