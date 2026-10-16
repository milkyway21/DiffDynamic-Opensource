"""Deterministic artifact identities, immutable structure snapshots and reports."""

import csv
import json
import math
from dataclasses import asdict, is_dataclass
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import Draw, rdDepictor


def plain(value):
    if is_dataclass(value):
        return plain(asdict(value))
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items() if not isinstance(item, Chem.Mol)}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "tolist"):
        return plain(value.tolist())
    if hasattr(value, "item"):
        return plain(value.item())
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(plain(value), ensure_ascii=False, allow_nan=False, indent=2))
    tmp.replace(path)


def write_structure(path, mol):
    """Keep aromatic [nH] in portable SDF; retain invalid graphs as raw evidence."""
    try:
        Chem.Kekulize(Chem.Mol(mol))
        kekulize = True
    except Exception:
        kekulize = False
    with Chem.SDWriter(str(path)) as writer:
        writer.SetKekulize(kekulize)
        writer.write(mol)


class Artifacts:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.items = []

    def add(self, artifact_id, relative, label, fmt):
        self.items.append({"artifact_id": artifact_id, "path": relative,
                           "label": label, "format": fmt})
        return artifact_id

    def structure(self, mol, variant_id, index, smiles, stage):
        directory = "reconstructed_molecules" if stage == "primary" else "structures"
        relative = f"{directory}/{variant_id}.sdf"
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        out = Chem.Mol(mol)
        # No inherited experimental or computational scores survive a new stage.
        for prop in list(out.GetPropNames(includePrivate=True, includeComputed=False)):
            if prop != "_Name":
                out.ClearProp(prop)
        out.SetProp("Molecule_ID", variant_id)
        out.SetIntProp("Molecule_Index", index)
        out.SetProp("Medchem_Stage", stage)
        if smiles:
            out.SetProp("SMILES", smiles)
        write_structure(path, out)
        return self.add(variant_id + "_sdf", relative, variant_id, "sdf")

    def depiction(self, mol, variant_id, highlight_atoms=()):
        copy = Chem.Mol(mol)
        copy.RemoveAllConformers()
        rdDepictor.Compute2DCoords(copy)
        drawer = Draw.rdMolDraw2D.MolDraw2DSVG(420, 300)
        drawer.DrawMolecule(copy, highlightAtoms=list(highlight_atoms))
        drawer.FinishDrawing()
        relative = f"depictions/{variant_id}.svg"
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(drawer.GetDrawingText())
        return self.add(variant_id + "_svg", relative, variant_id, "svg")


def export_summary(root, report):
    root = Path(root)
    with (root / "summary.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        fields = ["record_id", "name", "status", "variant_id", "parent_variant_id", "stage", "smiles",
                  "transform", "rationale", "qed", "sa", "mw", "logp", "tpsa", "similarity", "delta_heavy_atoms",
                  "geometry_status", "evaluation_status", "vina", "error"]
        writer.writerow(fields)
        for record in report["records"]:
            variants = [v for v in [record.get("original"), record.get("primary"), *record.get("candidates", [])] if v]
            if not variants:
                variants = [{}]
            for variant in variants:
                props = variant.get("properties", {})
                evaluation = variant.get("evaluation", {})
                if evaluation.get("status") == "skipped" and variant.get("main_evaluation"):
                    evaluation = variant["main_evaluation"]
                row = [record["record_id"], record["name"], record["status"], variant.get("variant_id"),
                       variant.get("parent_variant_id"), variant.get("stage"), variant.get("smiles"),
                       variant.get("transform_id"), variant.get("rationale"), props.get("qed"), props.get("sa"),
                       props.get("mw"), props.get("logp"), props.get("tpsa"), variant.get("similarity"),
                       variant.get("delta_heavy_atoms"), variant.get("geometry", {}).get("status"),
                       evaluation.get("status"), evaluation.get("affinity"),
                       record.get("error") or evaluation.get("reason", "")]
                # Spreadsheet applications must not execute user-supplied titles.
                writer.writerow(["'" + x if isinstance(x, str) and x.startswith(("=", "+", "-", "@")) else x for x in row])
    for stage in ("original", "primary", "candidate"):
        output = root / f"{stage}.sdf"
        with output.open("w") as handle:
            for record in report["records"]:
                variants = record.get("candidates", []) if stage == "candidate" else [record.get(stage)]
                for variant in variants:
                    if not variant:
                        continue
                    aid = variant.get("structure_artifact_id")
                    item = next((a for a in report["artifacts"] if a["artifact_id"] == aid), None)
                    if item:
                        handle.write((root / item["path"]).read_text())
        report["artifacts"].append({"artifact_id": f"download_{stage}", "path": output.name, "label": stage + " SDF", "format": "sdf"})
    report["artifacts"].extend([
        {"artifact_id": "download_summary", "path": "summary.csv", "label": "CSV", "format": "csv"},
        {"artifact_id": "download_audit", "path": "audit.json", "label": "Audit JSON", "format": "json"},
    ])
    write_json(root / "audit.json", report)
