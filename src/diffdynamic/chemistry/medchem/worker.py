"""Subprocess entry point with per-record and per-docking hard deadlines."""

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys

from . import RULE_VERSION, ensure_repair_package
from .artifacts import export_summary, write_json, write_structure

_CHILDREN = set()


def _terminate(process):
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
    except ProcessLookupError:
        pass


def _cancel(signum, _frame):
    for child in list(_CHILDREN):
        _terminate(child)
    raise SystemExit(128 + signum)


def run_child(args, timeout, log_path):
    with Path(log_path).open("w") as log:
        process = subprocess.Popen([sys.executable, "-m", "diffdynamic.chemistry.medchem.worker", *args],
                                   stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        _CHILDREN.add(process)
        try:
            return process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            _terminate(process)
            return -9
        finally:
            _CHILDREN.discard(process)


def isolated_dock(variant, request, artifacts):
    identity = variant["variant_id"]
    item = next(a for a in artifacts.items if a["artifact_id"] == variant["structure_artifact_id"])
    scratch = artifacts.root / "scratch" / identity
    scratch.mkdir(parents=True, exist_ok=True)
    parameters = {"structure_path": str(artifacts.root / item["path"]),
                  "protein_path": request["protein_path"], "box": request["box"],
                  "evaluation": request["options"]["evaluation"],
                  "output": str(scratch / "docking.json"), "scratch": str(scratch)}
    input_path = scratch / "docking_request.json"
    write_json(input_path, parameters)
    rc = run_child(["--dock", str(input_path)], parameters["evaluation"]["timeout_seconds"], scratch / "dock.log")
    if rc == -9:
        return {"status": "timeout", "reason": "Vina hard deadline exceeded"}
    output = Path(parameters["output"])
    if rc or not output.exists():
        return {"status": "failed", "reason": "Docking process failed"}
    result = json.loads(output.read_text())
    pose = result.pop("pose", None)
    if result.get("status") == "completed" and pose:
        relative = f"poses/{identity}.pdbqt"
        target = artifacts.root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(pose)
        result["pose_artifact_id"] = artifacts.add(identity + "_pose", relative, "Docked pose", "pdbqt")
    return result


def _dock(input_path):
    from rdkit import Chem
    from diffdynamic.evaluation.docking_vina import VinaDockingTask

    parameters = json.loads(Path(input_path).read_text())
    try:
        mol = next(iter(Chem.SDMolSupplier(parameters["structure_path"], removeHs=False)), None)
        if mol is None:
            raise ValueError("Invalid structure snapshot")
        scratch = Path(parameters["scratch"])
        protein = scratch / "receptor.pdb"
        shutil.copyfile(parameters["protein_path"], protein)
        evaluation = parameters["evaluation"]
        task = VinaDockingTask(str(protein), mol, tmp_dir=str(scratch),
                               center=parameters["box"]["center"], box_size=parameters["box"]["size"])
        rows = task.run(mode="dock", exhaustiveness=evaluation["exhaustiveness"],
                        n_poses=1, seed=evaluation["seed"], cpu=1,
                        timeout_sec=evaluation["timeout_seconds"])
        if not rows or not math.isfinite(float(rows[0]["affinity"])) or not rows[0].get("pose"):
            raise ValueError("Vina returned no finite score and pose")
        result = {"status": "completed", "affinity": float(rows[0]["affinity"]),
                  "pose": rows[0]["pose"], "box": parameters["box"], "parameters": evaluation}
    except TimeoutError as exc:
        result = {"status": "timeout", "reason": str(exc)}
    except Exception as exc:
        result = {"status": "failed", "reason": str(exc)}
    write_json(parameters["output"], result)


def isolated_evaluate(variant, request, artifacts, index):
    item = next((a for a in artifacts.items if a["artifact_id"] == variant.get("structure_artifact_id")), None)
    if not item:
        return {"success": False, "error": "Primary structure snapshot unavailable"}
    scratch = artifacts.root / "scratch" / (variant["variant_id"] + "_eval")
    scratch.mkdir(parents=True, exist_ok=True)
    parameters = {"structure_path": str(artifacts.root / item["path"]),
                  "protein_path": request.get("protein_path"), "scratch": str(scratch),
                  "output": str(scratch / "evaluation.json"), "index": index,
                  "score_pose": bool(variant["geometry"].get("aligned") and variant["geometry"]["status"] == "passed" and request["options"]["evaluation"]["mode"] != "vina_dock")}
    input_path = scratch / "evaluation_request.json"
    write_json(input_path, parameters)
    rc = run_child(["--evaluate", str(input_path)], 120, scratch / "evaluate.log")
    if rc == -9:
        return {"success": False, "error": "Primary evaluation timed out"}
    output = Path(parameters["output"])
    if rc or not output.exists():
        return {"success": False, "error": "Primary evaluation process failed"}
    return json.loads(output.read_text())


def _evaluate(input_path):
    from rdkit import Chem
    from diffdynamic.evaluation.molecule.scoring import evaluate_single_molecule

    parameters = json.loads(Path(input_path).read_text())
    try:
        mol = next(iter(Chem.SDMolSupplier(parameters["structure_path"], removeHs=False)), None)
        if mol is None:
            raise ValueError("Invalid primary snapshot")
        if len(Chem.GetMolFrags(mol)) > 1:
            write_json(parameters["output"], {
                "success": False, "status": "skipped",
                "error": "The legacy evaluator does not support multi-fragment structures; all fragments and medchem properties are retained",
            })
            return
        protein = parameters.get("protein_path")
        if protein:
            target = Path(parameters["scratch"]) / "receptor.pdb"
            shutil.copyfile(protein, target)
            protein = str(target)
        result = evaluate_single_molecule(
            mol, None, str(Path(parameters["scratch"])), protein_path=protein,
            tmp_dir=parameters["scratch"], pre_docking_mmff_minimize=False,
            pre_docking_etkdg_reembed=False,
            vina_modes=["score_only"] if protein and parameters["score_pose"] else [],
        )
        result["mol_idx"] = parameters["index"]
        if result.get("conformer_energy") is not None:
            result["conformer_energy_context"] = "separately_generated_reference_conformer"
    except Exception as exc:
        result = {"success": False, "error": str(exc)}
    write_json(parameters["output"], result)


def _primary_results(report):
    results = []
    for i, record in enumerate(report["records"]):
        variant = record.get("primary")
        if not variant or not variant.get("properties"):
            continue
        properties = variant["properties"]
        result = {"mol_idx": record["input"]["index"], "molecule_id": variant["variant_id"],
                  "smiles": variant["smiles"], "chem": {"qed": properties.get("qed"), "sa": properties.get("sa")},
                  "logp": properties.get("logp"), "tpsa": properties.get("tpsa"),
                  "success": True, "rdkit_valid": True, "medchem_record_id": record["record_id"]}
        result.update({k: v for k, v in record.get("primary_evaluation", {}).items() if v is not None})
        result.update(mol_idx=record["input"]["index"], molecule_id=variant["variant_id"],
                      smiles=variant["smiles"], success=True, rdkit_valid=True,
                      evaluated_variant_id=variant["variant_id"])
        # Every score belongs to this snapshot; never restore a score from the source PT.
        docking = variant.get("evaluation", {})
        if docking.get("status") == "completed":
            result["vina_dock"] = [{"affinity": docking["affinity"]}]
        results.append(result)
    return results


def process_batch(request, output_dir):
    from .inputs import read_records
    from .processing import box_for_request

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    records = read_records(request)
    report = {"schema_version": 1, "run_id": request.get("run_id"),
              "source_run_id": request.get("source_run_id"), "status": "running",
              "options": request["options"], "rule_version": RULE_VERSION,
              "records": [{"record_id": r["record_id"], "name": r["name"], "input": r,
                           "status": "pending", "original": None, "primary": None,
                           "candidates": []} for r in records],
              "artifacts": [], "summary": {"total": len(records), "processed": 0}}
    write_json(root / "report.json", report)
    if request["options"]["evaluation"]["mode"] == "vina_dock" and not request.get("source_report"):
        if not request.get("protein_path"):
            raise ValueError("Docking requires a resolved protein")
        request["box"] = box_for_request(request)
        report["box"] = request["box"]
    if request.get("protein_path"):
        target = root / "protein.pdb"
        shutil.copyfile(request["protein_path"], target)
        request["protein_path"] = str(target)
        report["protein_artifact_id"] = "protein"
        report["artifacts"].append({"artifact_id": "protein", "path": "protein.pdb", "label": "Protein", "format": "pdb"})
    scratch = root / "scratch"
    scratch.mkdir(exist_ok=True)
    proteins = {}
    for record_index, record in enumerate(records):
        record_request = {**request, **record.get("context", {})}
        if request.get("_protein_override") or not record_request.get("protein_path"):
            record_request["protein_path"] = request.get("protein_path")
        protein = record_request.get("protein_path")
        protein_id = report.get("protein_artifact_id")
        if protein and protein != request.get("protein_path"):
            if protein not in proteins:
                protein_id = f"protein_{len(proteins) + 1}"
                target = root / (protein_id + ".pdb")
                shutil.copyfile(protein, target)
                proteins[protein] = (protein_id, str(target))
                report["artifacts"].append({"artifact_id": protein_id, "path": target.name,
                                             "label": "Protein", "format": "pdb"})
            protein_id, record_request["protein_path"] = proteins[protein]
        if request["options"]["evaluation"]["mode"] == "vina_dock" and request.get("source_report"):
            if not record_request.get("protein_path"):
                raise ValueError("Historical input has no protein for docking")
            if request["options"]["evaluation"].get("center_xyz") is not None or request.get("reference_ligand_path"):
                record_request["box"] = box_for_request(record_request)
            if not record_request.get("box"):
                raise ValueError("Historical input requires a reference ligand or explicit docking box")
        identity = record["record_id"]
        input_path = scratch / f"{identity}.request.json"
        output_path = scratch / f"{identity}.result.json"
        write_json(input_path, {"record": record, "request": record_request, "root": str(root), "output": str(output_path)})
        timeout = request["options"]["time_budget_seconds"] + 45
        if request["options"]["evaluation"]["mode"] == "vina_dock":
            timeout += (request["options"]["top_k"] + 2) * (request["options"]["evaluation"]["timeout_seconds"] + 5)
        if request.get("auto_evaluate"):
            timeout += 125
        rc = run_child(["--record", str(input_path)], timeout, scratch / f"{identity}.log")
        if not rc and output_path.exists():
            data = json.loads(output_path.read_text())
            result = data["record"]
            report["artifacts"].extend(data["artifacts"])
        else:
            result = {"record_id": identity, "name": record["name"], "status": "failed",
                      "error": "Record processing timed out" if rc == -9 else "Record processing failed",
                      "input": record, "original": None, "primary": None, "candidates": [], "findings": [], "changes": []}
            if output_path.exists():
                data = json.loads(output_path.read_text())
                result["error"] = data.get("error", result["error"])
        if protein_id:
            result["protein_artifact_id"] = protein_id
            for variant in [result.get("original"), result.get("primary"), *result.get("candidates", [])]:
                if variant:
                    variant["protein_artifact_id"] = protein_id
        result["box"] = record_request.get("box")
        report["records"][record_index] = result
        report["summary"]["processed"] = record_index + 1
        write_json(root / "report.json", report)
        if not request.get("_batch_child"):
            print(f"MEDCHEM_PROGRESS {record_index + 1}/{len(records)}", flush=True)
        else:
            done = request.get("_pocket_index", 0) + (record_index + 1) / max(len(records), 1)
            print(f"MEDCHEM_PROGRESS {int(done * 1000)}/{request.get('_pocket_total', 1) * 1000}", flush=True)
    failed = sum(r["status"] in {"failed", "rejected"} for r in report["records"])
    report["summary"].update(failed=failed, valid=sum(bool((r.get("primary") or {}).get("properties")) for r in report["records"]),
                             repaired=sum(r["status"] == "repaired" for r in report["records"]),
                             candidates=sum(len(r["candidates"]) for r in report["records"]))
    evaluation_failures = any(v.get("evaluation", {}).get("status") in {"failed", "timeout"} for r in report["records"] for v in [r.get("original"), r.get("primary"), *r.get("candidates", [])] if v)
    evaluation_failures = evaluation_failures or any(
        r.get("primary_evaluation") and not r["primary_evaluation"].get("success")
        for r in report["records"]
    )
    report["status"] = "failed" if failed == len(records) else "partial" if failed or evaluation_failures else "completed"
    export_summary(root, report)
    write_json(root / "report.json", report)
    # Existing molecule ingestion consumes only primary snapshots from this directory.
    import torch
    torch.save({"results": _primary_results(report), "medchem": True}, root / "eval_results_medchem_final.pt")
    return report


def process_pockets(request, output_dir):
    """One queued job owns a batch; each pocket keeps its own receptor and box."""
    import copy
    from rdkit import Chem
    import torch

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    report = {"schema_version": 1, "run_id": request.get("run_id"),
              "source_run_id": request.get("source_run_id"), "status": "running",
              "options": request["options"], "rule_version": RULE_VERSION,
              "records": [], "artifacts": [], "pockets": [], "summary": {}}
    write_json(root / "report.json", report)
    sources = request.get("source_pts") or request.get("source_reports") or []
    source_key = "source_pt" if request.get("source_pts") else "source_report"
    # Independently submitted historical batches share the 100-input limit.
    if not request.get("source_label"):
        from .inputs import read_records
        total = sum(len(read_records({**request, source_key: source})) for source in sources)
        if total > 100:
            raise ValueError("At most 100 input records are allowed for an independent task")
    for pocket_index, source in enumerate(sources):
        label = Path(source).parent.name if source_key == "source_report" else Path(source).stem
        prefix = f"p{pocket_index + 1:04d}_"
        directory = "pocket_" + str(pocket_index + 1).zfill(4)
        subrequest = {k: v for k, v in request.items() if k not in {"source_pts", "source_reports"}}
        subrequest[source_key] = source
        subrequest["_batch_child"] = True
        subrequest["_pocket_index"] = pocket_index
        subrequest["_pocket_total"] = len(sources)
        interrupted = None
        try:
            part = process_batch(subrequest, root / directory)
        except SystemExit as exc:
            interrupted = exc
            partial = root / directory / "report.json"
            part = json.loads(partial.read_text()) if partial.exists() else {"records": [], "artifacts": []}
            part["status"] = "cancelled"
        except Exception as exc:
            part = {"status": "failed", "error": str(exc), "records": [], "artifacts": [], "summary": {"failed": 1}}
        report["pockets"].append({"label": label, "status": part["status"],
                                  "box": part.get("box"), "error": part.get("error"),
                                  "protein_artifact_id": prefix + "protein" if part.get("protein_artifact_id") else None})
        identifiers = {item["artifact_id"]: prefix + item["artifact_id"] for item in part["artifacts"]}
        for artifact in part["artifacts"]:
            item = dict(artifact)
            item["artifact_id"] = identifiers[item["artifact_id"]]
            item["path"] = directory + "/" + item["path"]
            report["artifacts"].append(item)
        for row in part["records"]:
            row = copy.deepcopy(row)
            if row.get("protein_artifact_id"):
                row["protein_artifact_id"] = identifiers[row["protein_artifact_id"]]
            row["record_id"] = prefix + row["record_id"]
            row["input"]["record_id"] = row["record_id"]
            row["pocket_label"] = label
            row["name"] = row["pocket_label"] + " / " + row["name"]
            row["input"]["index"] = len(report["records"])
            for variant in [row.get("original"), row.get("primary"), *row.get("candidates", [])]:
                if not variant:
                    continue
                variant["variant_id"] = prefix + variant["variant_id"]
                variant["record_id"] = row["record_id"]
                if variant.get("parent_variant_id") and variant.get("stage") != "original":
                    variant["parent_variant_id"] = prefix + variant["parent_variant_id"]
                for key in ("structure_artifact_id", "depiction_artifact_id"):
                    if variant.get(key):
                        variant[key] = identifiers[variant[key]]
                if variant.get("evaluation", {}).get("pose_artifact_id"):
                    variant["evaluation"]["pose_artifact_id"] = identifiers[variant["evaluation"]["pose_artifact_id"]]
                for evaluation_key in ("evaluation", "main_evaluation"):
                    if variant.get(evaluation_key):
                        variant[evaluation_key]["evaluated_variant_id"] = variant["variant_id"]
                if part.get("protein_artifact_id"):
                    variant["protein_artifact_id"] = prefix + part["protein_artifact_id"]
                elif variant.get("protein_artifact_id"):
                    variant["protein_artifact_id"] = identifiers[variant["protein_artifact_id"]]
            for snapshot in [row.get("original"), row.get("primary"), *row.get("candidates", [])]:
                if not snapshot or not snapshot.get("structure_artifact_id"):
                    continue
                item = next(a for a in report["artifacts"] if a["artifact_id"] == snapshot["structure_artifact_id"])
                mol = next(iter(Chem.SDMolSupplier(str(root / item["path"]), sanitize=False, removeHs=False)), None)
                if mol is not None:
                    destination = "reconstructed_molecules" if snapshot["stage"] == "primary" else "structures"
                    target = root / destination / (snapshot["variant_id"] + ".sdf")
                    target.parent.mkdir(exist_ok=True)
                    mol.SetIntProp("Molecule_Index", row["input"]["index"])
                    mol.SetProp("Molecule_ID", snapshot["variant_id"])
                    write_structure(target, mol)
                    item["path"] = str(target.relative_to(root))
            if row.get("primary_evaluation"):
                row["primary_evaluation"]["mol_idx"] = row["input"]["index"]
            report["records"].append(row)
        report["summary"] = {"total": len(report["records"]),
                              "processed": sum(r["status"] != "pending" for r in report["records"])}
        write_json(root / "report.json", report)
        if interrupted is not None:
            report["status"] = "cancelled"
            export_summary(root, report)
            write_json(root / "report.json", report)
            raise interrupted
        print(f"MEDCHEM_PROGRESS {pocket_index + 1}/{len(sources)}", flush=True)
    report["summary"] = {"total": len(report["records"]), "processed": len(report["records"]),
                          "valid": sum(bool((r.get("primary") or {}).get("properties")) for r in report["records"]),
                          "failed": sum(r["status"] in {"failed", "rejected"} for r in report["records"]),
                          "repaired": sum(r["status"] == "repaired" for r in report["records"]),
                          "candidates": sum(len(r.get("candidates", [])) for r in report["records"])}
    statuses = [p["status"] for p in report["pockets"]]
    report["status"] = "failed" if all(s == "failed" for s in statuses) else "partial" if any(s != "completed" for s in statuses) else "completed"
    export_summary(root, report)
    write_json(root / "report.json", report)
    torch.save({"results": _primary_results(report), "medchem": True}, root / "eval_results_medchem_final.pt")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request")
    parser.add_argument("--output")
    parser.add_argument("--record")
    parser.add_argument("--dock")
    parser.add_argument("--evaluate")
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, _cancel)
    signal.signal(signal.SIGINT, _cancel)
    ensure_repair_package()
    if args.dock:
        _dock(args.dock)
    elif args.evaluate:
        _evaluate(args.evaluate)
    elif args.record:
        from .processing import process_record
        payload = json.loads(Path(args.record).read_text())
        try:
            result, artifacts = process_record(payload["record"], payload["request"], payload["root"])
            write_json(payload["output"], {"record": result, "artifacts": artifacts})
        except Exception as exc:
            write_json(payload["output"], {"error": str(exc)})
            raise
    else:
        request = json.loads(Path(args.request).read_text())
        try:
            report = process_pockets(request, args.output) if request.get("source_pts") or request.get("source_reports") else process_batch(request, args.output)
        except Exception as exc:
            path = Path(args.output) / "report.json"
            report = json.loads(path.read_text()) if path.exists() else {"records": [], "artifacts": []}
            report.update(status="failed", error=str(exc))
            write_json(path, report)
            raise
        if report["status"] == "failed":
            raise SystemExit(1)


if __name__ == "__main__":
    main()
