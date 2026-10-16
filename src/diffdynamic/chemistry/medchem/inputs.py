"""Lossless input records and generated-structure adapters."""

import json
from pathlib import Path

from rdkit import Chem


def _record(index, name, raw, kind):
    return {"record_id": f"m{index + 1:05d}", "name": name or f"molecule_{index + 1}",
            "raw": raw, "kind": kind, "index": index}


def read_records(request):
    """Keep malformed records in their original order instead of dropping them."""
    if request.get("source_report"):
        return read_report_records(request)
    if request.get("snapshot_paths"):
        records = []
        for path in request["snapshot_paths"]:
            for record in read_records({"input_path": path, "max_records": request.get("max_records", 100)}):
                record["index"] = len(records)
                record["record_id"] = f"m{len(records) + 1:05d}"
                records.append(record)
        if len(records) > request.get("max_records", 100):
            raise ValueError("Too many input records")
        return records
    if request.get("source_pt"):
        return read_generated(request)
    if request.get("smiles") is not None:
        lines = [line for line in request["smiles"].splitlines() if line.strip()]
        records = []
        for i, line in enumerate(lines):
            parts = line.strip().split(maxsplit=1)
            records.append(_record(i, parts[1] if len(parts) > 1 else None,
                                   parts[0], "smiles"))
    else:
        path = Path(request["input_path"])
        raw = path.read_text(encoding="utf-8", errors="replace")
        blocks = raw.split("$$$$") if path.suffix.lower() == ".sdf" else [raw]
        # A delimiter terminates a record, including a malformed empty record.
        # Only its trailing whitespace is not itself another input record.
        if len(blocks) > 1 and not blocks[-1].strip():
            blocks.pop()
        records = []
        for block in blocks:
            # The delimiter consumes its own newline, not a molecule title.
            if records and block.startswith("\r\n"):
                block = block[2:]
            elif records and block.startswith("\n"):
                block = block[1:]
            name = block.splitlines()[0].strip() if block.splitlines() else ""
            records.append(_record(len(records), name, block, "molblock"))
    limit = request.get("max_records", 100)
    if not records or len(records) > limit:
        raise ValueError(f"Input must contain 1–{limit} records")
    return records


def _report_artifact_path(root, artifacts, artifact_id):
    """Resolve a report-owned snapshot without following paths outside its root."""
    item = artifacts.get(artifact_id)
    if not item or not item.get("path"):
        return None
    path = (root / item["path"]).resolve()
    if root not in path.parents:
        raise ValueError("Source artifact path is outside its report directory")
    return path if path.is_file() else None


def read_report_records(request):
    """Reprocess each prior primary snapshot while retaining failed input rows."""
    source = Path(request["source_report"]).resolve()
    previous = json.loads(source.read_text(encoding="utf-8"))
    rows = previous.get("records", [])
    limit = request.get("max_records", 100)
    if not rows or len(rows) > limit:
        raise ValueError(f"Input must contain 1–{limit} records")
    artifacts = {
        item["artifact_id"]: item for item in previous.get("artifacts", [])
    }
    records = []
    for index, row in enumerate(rows):
        primary = row.get("primary") or {}
        original = row.get("input") or {}
        record = _record(
            index, row.get("name") or original.get("name"),
            original.get("raw", ""), original.get("kind", "molblock"),
        )
        for field in ("raw_coordinates", "raw_atom_types"):
            if field in original:
                record[field] = original[field]
        snapshot = _report_artifact_path(
            source.parent, artifacts, primary.get("structure_artifact_id"),
        )
        if snapshot:
            # Read the raw snapshot: parsing and sanitization belong to the new
            # run, and must not mutate the historical molecule or coordinates.
            snapshots = read_records({"input_path": str(snapshot), "max_records": 1})
            record.update(raw=snapshots[0]["raw"], kind="molblock")
        else:
            record["error"] = (
                row.get("error") or original.get("error")
                or "Source primary structure snapshot is unavailable"
            )
        protein_id = (
            primary.get("protein_artifact_id") or row.get("protein_artifact_id")
            or previous.get("protein_artifact_id")
        )
        protein = _report_artifact_path(source.parent, artifacts, protein_id)
        geometry = primary.get("geometry") or {}
        box = row.get("box") or previous.get("box")
        if box is None and protein_id:
            pocket = next(
                (item for item in previous.get("pockets", [])
                 if item.get("protein_artifact_id") == protein_id),
                {},
            )
            box = pocket.get("box")
        record.update(
            source_record_id=row.get("record_id"),
            source_variant_id=primary.get("variant_id"),
            context={
                "input_aligned": bool(geometry.get("aligned", False)),
                "coordinate_source": geometry.get("coordinate_source", "external"),
                "protein_path": str(protein) if protein else None,
                "box": box,
            },
        )
        records.append(record)
    return records


def parse_record(record):
    if record.get("error"):
        return None
    if record["kind"] == "smiles":
        return Chem.MolFromSmiles(record["raw"], sanitize=False)
    return Chem.MolFromMolBlock(record["raw"], sanitize=False, removeHs=False)


def _context_file(value, request):
    if not value:
        return None
    path = Path(str(value))
    roots = [Path(root).resolve() for root in request.get("context_roots", [])]
    candidates = [path] if path.is_absolute() else [root / path for root in roots]
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.is_file() and any(root == candidate or root in candidate.parents for root in roots):
            return str(candidate)
    return None


def read_generated(request):
    """Reuse reconstruction without mutating the model's original PT output."""
    from diffdynamic.evaluation.molecule.loading import load_pt_file, decompdiff_list_to_targetdiff_dict
    from diffdynamic.evaluation.molecule.reconstruction import reconstruct_molecule

    data = load_pt_file(request["source_pt"])
    if isinstance(data, list):
        data, _ = decompdiff_list_to_targetdiff_dict(data, Path(request["source_pt"]))
    if not isinstance(data, dict) or "pred_ligand_pos" not in data:
        raise ValueError("Source is not a generated molecular result")
    raw_data = data.get("data")
    context_value = (
        raw_data.get if isinstance(raw_data, dict)
        else lambda key, default=None: getattr(raw_data, key, default)
    )
    if not request.get("protein_path"):
        request["protein_path"] = _context_file(
            context_value("protein_filename"), request,
        )
    if not request.get("reference_ligand_path"):
        request["reference_ligand_path"] = _context_file(
            context_value("ligand_filename"), request,
        )
    request["input_aligned"] = True
    request["coordinate_source"] = "generated_pocket"
    positions, atom_types = data["pred_ligand_pos"], data["pred_ligand_v"]
    if len(positions) != len(atom_types):
        raise ValueError("Generated atom types and coordinates have different lengths")
    limit = request.get("max_records", 100)
    if len(positions) == 0 or len(positions) > limit:
        raise ValueError(f"Input must contain 1–{limit} records")
    records = []
    for i, (pos, types) in enumerate(zip(positions, atom_types)):
        try:
            mol, error = reconstruct_molecule(
                pos, types, rdkit_structure_repair={},
                atom_mode=request.get("atom_mode", "add_aromatic"),
            )
            if mol is None:
                raise ValueError(str(error or "Reconstruction failed"))
            try:
                # Kekule bonds retain pyrrolic [nH] on an SDF round trip.
                block = Chem.MolToMolBlock(mol, kekulize=True)
            except Exception:
                # Invalid graphs still need their original structural record.
                block = Chem.MolToMolBlock(mol, kekulize=False)
            record = _record(i, None, block, "molblock")
        except Exception as exc:
            record = _record(i, None, "", "molblock")
            record["error"] = str(exc) or type(exc).__name__
            # Preserve the actual failed input for inspection.
            record["raw_coordinates"] = pos.tolist() if hasattr(pos, "tolist") else pos
            record["raw_atom_types"] = types.tolist() if hasattr(types, "tolist") else types
        records.append(record)
    return records
