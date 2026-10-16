"""Run medchem in an interruptible CPU subprocess, preserving source snapshots."""

import json
import os
from pathlib import Path
import re

from server.config import get_config
from server.database import get_run, update_run
from server.runners.base import append_log, mark_completed, mark_failed, run_cmd


def _source_request(params, owner):
    request = dict(params)
    request["_protein_override"] = bool(params.get("protein_path"))
    cfg = get_config()
    from server.auth import user_upload_root, user_output_root

    request["context_roots"] = [cfg.data_root, str(user_output_root(cfg.output_root, owner)),
                                str(user_upload_root(cfg.upload_dir, owner))]
    if params.get("protein_root"):
        request["context_roots"].append(params["protein_root"])
    if params.get("source_run_id"):
        source = get_run(params["source_run_id"], owner=owner)
        if source is None:
            raise ValueError("Source run not found")
        source_params = json.loads(source.parameters or "{}")
        for destination, key in (("protein_path", "protein_path"),
                                 ("reference_ligand_path", "ligand_path")):
            if not request.get(destination):
                request[destination] = source_params.get(key)
        output = Path(source.output_path or "")
        if not request.get("source_pt") and not request.get("source_pts"):
            detail = json.loads(source.progress_detail or "{}")
            if source.run_type == "generate_batch":
                paths = detail.get("medchem_source_pts")
                if not paths:
                    raise ValueError("This historical batch has no complete pocket manifest; select an individual pocket result")
                request["source_pts"] = paths
            elif source.run_type in {"medchem", "medchem_dock"} and (output / "report.json").is_file():
                previous = json.loads((output / "report.json").read_text())
                if previous.get("pockets"):
                    request["source_reports"] = [str(path) for path in sorted(output.glob("pocket_*/report.json"))]
                else:
                    request["source_report"] = str(output / "report.json")
            elif output.is_file() and output.suffix == ".pt":
                request["source_pt"] = str(output)
            else:
                sdf = sorted(output.glob("reconstructed_molecules/*.sdf")) if output.is_dir() else []
                if not sdf:
                    raise ValueError("Source has no primary molecular snapshots; open its per-pocket report")
                request["snapshot_paths"] = [str(path) for path in sdf]
                request["input_aligned"] = True
        request["coordinate_source"] = "generated_pocket"
    if request.get("source_pt"):
        from server.safe_pt import prepare_job_pt

        request["source_pt"] = prepare_job_pt(request["source_pt"], owner)
        request["input_aligned"] = True
    if request.get("source_pts"):
        from server.safe_pt import prepare_job_pt
        request["source_pts"] = [prepare_job_pt(path, owner) for path in request["source_pts"]]
        request["input_aligned"] = True
    from server.safe_pt import is_generated_pt
    sources = request.get("source_pts") or ([request["source_pt"]] if request.get("source_pt") else [])
    request["_trusted_generated_sources"] = bool(sources) and all(is_generated_pt(path, owner) for path in sources)
    return request


def run_medchem(scheduler, rec, params):
    cfg = get_config()
    output = Path(params["result_path"]) / f"medchem_{rec.job_id}"
    output.mkdir(parents=True, exist_ok=True)
    rec.output_path = str(output)
    try:
        if rec.status == "cancelled":
            return
        update_run(rec.run_id, status="running", output_path=str(output))
        request = _source_request(params, rec.owner)
        if rec.status == "cancelled":
            return
        request["run_id"] = rec.run_id
        request_path = output / "request.json"
        request_path.write_text(json.dumps(request, ensure_ascii=False))
        cmd = [cfg.python_bin, "-m", "diffdynamic.chemistry.medchem.worker",
               "--request", str(request_path), "--output", str(output)]

        def on_line(line):
            append_log(rec, line)
            match = re.search(r"MEDCHEM_PROGRESS (\d+)/(\d+)", line)
            if match:
                rec.progress = int(match[1]) / max(int(match[2]), 1)
                update_run(rec.run_id, progress=rec.progress)

        env = {"PYTHONPATH": os.pathsep.join([str(Path(cfg.diffdynamic_root) / "src"), cfg.diffdynamic_root]),
               "PATH": str(Path(cfg.python_bin).resolve().parent) + os.pathsep + os.environ.get("PATH", ""),
               "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
               "DD_PT_STRICT_LOAD": "0" if request.get("_trusted_generated_sources") else "1"}
        rc = run_cmd(rec, cmd, on_line=on_line, env_extra=env)
        if rec.status == "cancelled":
            return
        if rc:
            report_path = output / "report.json"
            report = json.loads(report_path.read_text()) if report_path.exists() else {}
            mark_failed(rec, report.get("error") or "Medchem processing failed", "medchem_failed")
        else:
            from server.molecule_ingest import ingest_molecules_from_eval

            ingest_molecules_from_eval(rec.run_id, str(output))
            if params.get("source_run_id") and (params.get("auto_evaluate") or params.get("auto_extract")):
                ingest_molecules_from_eval(params["source_run_id"], str(output))
            if rec.status != "cancelled":
                mark_completed(rec, "medchem_completed")
    except Exception as exc:
        if rec.status != "cancelled":
            mark_failed(rec, str(exc), "medchem_failed")
    finally:
        scheduler._release_and_dispatch(rec)


def chain_medchem(scheduler, rec, params, pt_paths=None):
    """Queue one CPU job owning all pocket snapshots; skip ordinary eval/extract."""
    options = params.get("medchem")
    if not options:
        return []
    paths = list(dict.fromkeys(pt_paths or ([rec.output_path] if rec.output_path else [])))
    if not paths:
        raise ValueError("Generation did not report any molecular output for medchem")
    result = scheduler.submit_medchem(
            source_run_id=rec.run_id,
            **({"source_pt": paths[0]} if len(paths) == 1 else {"source_pts": paths}),
            source_label=Path(paths[0]).stem if len(paths) == 1 else "Batch pockets",
            protein_path=params.get("protein_path"),
            reference_ligand_path=params.get("ligand_path"),
            input_aligned=True, options=options,
            auto_evaluate=bool(params.get("auto_evaluate") and not params.get("sample_only")),
            auto_extract=params.get("auto_extract", True),
            max_records=max(100, int(params.get("num_samples") or params.get("batch_size") or 100)),
            protein_root=params.get("protein_root") or str(Path(get_config().data_root) / "crossdocked_pocket10_test_only"),
            triggered_by=rec.owner,
            skip_daily_quota=bool(params.get("_skip_daily_quota")),
        )
    children = [result]
    update_run(rec.run_id, progress_detail=json.dumps({"medchem_jobs": children,
                                                    "medchem_source_pts": paths}))
    return children
