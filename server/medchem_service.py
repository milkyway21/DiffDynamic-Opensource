"""Filesystem and authorization boundaries for medchem artifacts."""

import json
from pathlib import Path

from fastapi import HTTPException


def read_report(run):
    if run.run_type not in {"medchem", "medchem_dock"}:
        raise HTTPException(404, "This run has no medchem report")
    if not run.output_path:
        return {"run_id": run.id, "status": run.status, "records": [], "artifacts": []}
    root = Path(run.output_path).resolve()
    report_path = root / "report.json"
    if not report_path.is_file():
        return {"run_id": run.id, "status": run.status, "records": [], "artifacts": []}
    report = json.loads(report_path.read_text())
    report["run_id"] = run.id
    if run.status == "cancelled":
        report["status"] = "cancelled"
    elif run.status == "failed":
        report["status"] = "failed"
        report["error"] = report.get("error") or getattr(run, "error_message", None) or "Task failed"
    # Paths are private implementation details; public downloads use IDs.
    report["artifacts"] = [
        {key: val for key, val in item.items() if key != "path"}
        for item in report.get("artifacts", [])
    ]
    return report


def artifact_path(run, artifact_id):
    if run.run_type not in {"medchem", "medchem_dock"} or not run.output_path:
        raise HTTPException(404, "Artifact not found")
    root = Path(run.output_path).resolve()
    report_path = root / "report.json"
    if not report_path.is_file():
        raise HTTPException(404, "Report not ready")
    report = json.loads(report_path.read_text())
    item = next((a for a in report.get("artifacts", [])
                 if a["artifact_id"] == artifact_id), None)
    if not item:
        raise HTTPException(404, "Artifact not found")
    target = (root / item["path"]).resolve()
    if root not in target.parents or not target.is_file():
        raise HTTPException(404, "Artifact not found")
    return target, item


def child_reports(run, owner=None):
    from server.database import db_session, Run

    with db_session() as session:
        query = session.query(Run).filter(Run.run_type.in_(("medchem", "medchem_dock")))
        if owner:
            query = query.filter(Run.owner == owner)
        result = []
        for child in query.order_by(Run.id.desc()).all():
            params = json.loads(child.parameters or "{}")
            if params.get("source_run_id") == run.id:
                result.append({"run_id": child.id, "status": child.status,
                               "label": params.get("source_label") or str(child.id)})
        return result


def validate_medchem_request(req, request, deps):
    params = req.dict()
    for field in ("input_path", "protein_path", "reference_ligand_path"):
        if params.get(field):
            params[field] = deps.validate_path_for_user(params[field], request, field=field)
    if params.get("input_path") and Path(params["input_path"]).suffix.lower() not in {".sdf", ".mol"}:
        raise HTTPException(400, "Upload an SDF or MOL file")
    if req.source_run_id:
        source = deps._require_run(req.source_run_id, request)
        if source.status != "completed" or not source.output_path:
            raise HTTPException(400, "Choose a completed run with molecular output")
        if not (source.run_type.startswith("generate_") or source.run_type in {"medchem", "medchem_dock", "evaluate", "extract"}):
            raise HTTPException(400, "Unsupported source run")
    return params
