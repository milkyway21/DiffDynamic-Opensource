#!/usr/bin/env python3
"""Migrate, reconstruct, and audit the historical GSPT1 PT corpus.

The command is deliberately PT-first.  Existing SDF files are moved into a
legacy archive for provenance, but the new canonical molecule library is
generated only by ``evaluate_pt_with_correct_reconstruct.py`` with Vina
disabled.  A source PT is selected once per generation job; eval chunk PTs
are used only when the original result PT is absent.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(
    "/data/zhang/Ye/DiffDynamic_outputs/hsvpol/"
    "molglue_ikzf2_gspt1/diffdynamic"
)
UNIFIED_NAME = "gspt1_unified"
DEFAULT_UNIFIED_ROOT = DATA_ROOT / UNIFIED_NAME
DEFAULT_PROTEIN_ROOT = Path(
    "/data/ye/e-drug-lab/Scientist_In_E-Drug-Lab/runs/"
    "molecular_glue_structures_ikzf2_gspt1_20260803_235811/"
    "02_gspt1_or_gstp1_molecular_glue/receptor"
)
DEFAULT_RECEPTOR = DEFAULT_PROTEIN_ROOT / "5HXB_receptor_clean.pdb"
DEFAULT_REFERENCE = Path("/data/ye/sdf/GSPT1.sdf")
EVALUATOR = REPO_ROOT / "evaluate_pt_with_correct_reconstruct.py"

RESULT_NAME = re.compile(r"^result_.*\.pt$", re.IGNORECASE)
EVAL_FINAL_NAME = re.compile(
    r"^eval_results_(?:.*_)?final(?:_.*)?\.pt$", re.IGNORECASE
)
CHUNK_INPUT_NAME = "input.pt"
JOB_MARKERS = {"jobs", "extract", "extract_cleaned", "extract_novina_v2"}
SCAFFOLD_SMARTS = (
    "O=C1CCC(N2Cc3ccccc3C2=O)C(=O)N1",
    "O=C1CCC(N2Cc3c(F)cccc3C2=O)C(=O)N1",
)


def _safe_slug(value: str, fallback: str = "unknown") -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._-")
    return value or fallback


def _sha1_text(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _jsonl_dump(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True))
            handle.write("\n")


def _jsonl_load(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def classify_campaign(path_parts: Sequence[str]) -> str:
    """Classify the known F/no-F campaign labels without guessing from SMILES."""
    text = "/".join(path_parts).lower()
    if re.search(r"(^|[/_.-])no[_-]?f([/_.-]|$)", text):
        return "no_f"
    if "f_main" in text or "fmain" in text:
        return "f_main"
    if "f_large" in text or "flarge" in text:
        return "f_large"
    return "unknown"


def job_context(path: Path, data_root: Path) -> dict[str, str]:
    """Return stable batch/campaign/job fields for result and eval PT paths."""
    relative = path.resolve().relative_to(data_root.resolve())
    parts = list(relative.parts)
    if not parts:
        raise ValueError(f"path is outside data root: {path}")
    batch = parts[0]
    try:
        target_index = next(
            index for index, part in enumerate(parts) if part.lower() == "gspt1"
        )
    except StopIteration as exc:
        raise ValueError(f"not a GSPT1 path: {path}") from exc

    before_target = parts[1:target_index]
    after_target = parts[target_index + 1 :]
    marker_index = next(
        (index for index, part in enumerate(after_target) if part in JOB_MARKERS),
        None,
    )
    if marker_index is None or marker_index + 1 >= len(after_target):
        job_name = "unassigned"
    else:
        job_name = after_target[marker_index + 1]
    campaign_parts = before_target or ["default"]
    campaign = _safe_slug("__".join(campaign_parts), "default")
    job = _safe_slug(job_name, "unassigned")
    return {
        "batch": _safe_slug(batch),
        "campaign": campaign,
        "job": job,
        "job_key": "/".join((*before_target, job_name)),
        "class_name": classify_campaign((*before_target, *after_target)),
    }


@dataclass
class Candidate:
    path: Path
    source_kind: str
    context: dict[str, str]
    relative_path: str
    source_id: str
    selected: bool = False
    selection_reason: str = ""
    sha256: str = ""
    legacy_path: str = ""
    raw_path: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "source_kind": self.source_kind,
            "original_path": str(self.path),
            "relative_path": self.relative_path,
            "batch": self.context["batch"],
            "campaign": self.context["campaign"],
            "job": self.context["job"],
            "job_key": self.context["job_key"],
            "class_name": self.context["class_name"],
            "selected": self.selected,
            "selection_reason": self.selection_reason,
            "sha256": self.sha256,
            "legacy_path": self.legacy_path,
            "raw_path": self.raw_path,
        }


def discover_candidates(data_root: Path) -> list[Candidate]:
    """Discover only GSPT1 result PTs and eval-final fallback PTs."""
    candidates: list[Candidate] = []
    unified = data_root / UNIFIED_NAME
    if not data_root.exists():
        raise FileNotFoundError(f"data root does not exist: {data_root}")
    for top in sorted(data_root.iterdir()):
        if top == unified or top.is_symlink():
            continue
        if not top.is_dir():
            continue
        for path in sorted(top.rglob("*.pt")):
            if not path.is_file():
                continue
            lower_parts = {part.lower() for part in path.relative_to(data_root).parts}
            if "gspt1" not in lower_parts:
                continue
            if RESULT_NAME.match(path.name):
                source_kind = "result"
            elif (
                path.name == CHUNK_INPUT_NAME
                and ".parallel_chunks" in path.parts
            ):
                source_kind = "chunk_input"
            elif EVAL_FINAL_NAME.match(path.name):
                source_kind = "eval_final_fallback"
            else:
                continue
            context = job_context(path, data_root)
            relative = path.relative_to(data_root).as_posix()
            source_id = "src_" + _sha1_text(f"{source_kind}:{relative}")
            candidates.append(
                Candidate(
                    path=path,
                    source_kind=source_kind,
                    context=context,
                    relative_path=relative,
                    source_id=source_id,
                )
            )
    return candidates


def select_candidates(candidates: Sequence[Candidate]) -> list[Candidate]:
    """Select result PTs, or reconstructible chunk inputs when result is absent."""
    grouped: dict[tuple[str, str], list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        grouped[
            (candidate.context["batch"], candidate.context["job_key"])
        ].append(candidate)

    selected: list[Candidate] = []
    for group in grouped.values():
        results = sorted(
            (item for item in group if item.source_kind == "result"),
            key=lambda item: (item.path.stat().st_mtime_ns, item.relative_path),
        )
        if results:
            chosen = results[-1]
            chosen.selected = True
            chosen.selection_reason = "preferred_result_pt"
            selected.append(chosen)
            for duplicate in results[:-1]:
                duplicate.selection_reason = "duplicate_result_excluded"
            for fallback in group:
                if fallback.source_kind != "result":
                    fallback.selection_reason = "result_pt_exists_excluded"
        else:
            fallbacks = sorted(
                (
                    item
                    for item in group
                    if item.source_kind == "chunk_input"
                ),
                key=lambda item: (item.source_kind, item.relative_path),
            )
            for fallback in fallbacks:
                fallback.selected = True
                fallback.selection_reason = "result_missing_chunk_input"
                selected.append(fallback)
            for item in group:
                if item.source_kind == "eval_final_fallback":
                    item.selection_reason = (
                        "non_reconstructible_eval_snapshot"
                        if fallbacks
                        else "no_reconstructible_pt"
                    )
    selected.sort(key=lambda item: item.source_id)
    return selected


def inventory_rows(data_root: Path) -> tuple[list[Candidate], list[Candidate]]:
    candidates = discover_candidates(data_root)
    selected = select_candidates(candidates)
    selected_ids = {item.source_id for item in selected}
    for candidate in candidates:
        if candidate.source_id in selected_ids:
            continue
        if not candidate.selection_reason:
            candidate.selection_reason = "not_selected"
    return candidates, selected


def inventory_summary(
    data_root: Path,
    candidates: Sequence[Candidate],
    selected: Sequence[Candidate],
) -> dict[str, Any]:
    return {
        "data_root": str(data_root),
        "candidate_count": len(candidates),
        "selected_count": len(selected),
        "candidate_by_kind": dict(Counter(item.source_kind for item in candidates)),
        "selected_by_kind": dict(Counter(item.source_kind for item in selected)),
        "selected_by_batch": dict(Counter(item.context["batch"] for item in selected)),
        "selected_by_class": dict(
            Counter(item.context["class_name"] for item in selected)
        ),
        "selected_by_batch_class": {
            f"{batch}:{class_name}": count
            for (batch, class_name), count in sorted(
                Counter(
                    (item.context["batch"], item.context["class_name"])
                    for item in selected
                ).items()
            )
        },
    }


def write_inventory(
    data_root: Path,
    unified_root: Path,
    candidates: Sequence[Candidate],
    selected: Sequence[Candidate],
    suffix: str = "",
) -> tuple[Path, Path, Path]:
    manifest = unified_root / "manifest"
    manifest.mkdir(parents=True, exist_ok=True)
    inventory_path = manifest / f"source_inventory{suffix}.jsonl"
    selected_path = manifest / f"selected_sources{suffix}.jsonl"
    summary_path = manifest / f"source_inventory{suffix}.json"
    _jsonl_dump(inventory_path, (item.as_dict() for item in candidates))
    _jsonl_dump(selected_path, (item.as_dict() for item in selected))
    summary = inventory_summary(data_root, candidates, selected)
    summary["inventory_path"] = str(inventory_path)
    summary["selected_path"] = str(selected_path)
    _json_dump(summary_path, summary)
    return inventory_path, selected_path, summary_path


def _candidate_from_dict(row: dict[str, Any]) -> Candidate:
    context = {
        key: str(row.get(key, ""))
        for key in ("batch", "campaign", "job", "job_key", "class_name")
    }
    return Candidate(
        path=Path(row["original_path"]),
        source_kind=str(row["source_kind"]),
        context=context,
        relative_path=str(row["relative_path"]),
        source_id=str(row["source_id"]),
        selected=bool(row.get("selected", False)),
        selection_reason=str(row.get("selection_reason", "")),
        sha256=str(row.get("sha256", "")),
        legacy_path=str(row.get("legacy_path", "")),
        raw_path=str(row.get("raw_path", "")),
    )


def _move_without_overwrite(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        if source.exists() or source.is_symlink():
            raise FileExistsError(
                f"refusing to overwrite during migration: {source} -> {destination}"
            )
        return
    shutil.move(str(source), str(destination))


def _relative_symlink(target: Path, link: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.exists() or link.is_symlink():
        if link.is_symlink() and link.resolve() == target.resolve():
            return
        raise FileExistsError(f"refusing to overwrite link: {link}")
    link.symlink_to(os.path.relpath(target, link.parent))


def migrate_corpus(
    data_root: Path,
    unified_root: Path,
    candidates: Sequence[Candidate],
    selected: Sequence[Candidate],
) -> list[dict[str, Any]]:
    """Move all top-level legacy outputs, then move selected PTs to raw/."""
    if data_root.resolve().stat().st_dev != unified_root.parent.resolve().stat().st_dev:
        raise OSError("data root and unified root must be on the same filesystem")
    legacy_root = unified_root / "legacy"
    raw_root = unified_root / "raw"
    legacy_root.mkdir(parents=True, exist_ok=True)
    raw_root.mkdir(parents=True, exist_ok=True)

    top_level_names = sorted(
        entry.name
        for entry in data_root.iterdir()
        if entry.name != unified_root.name
    )
    for entry in sorted(data_root.iterdir()):
        if entry.name == unified_root.name:
            continue
        destination = legacy_root / entry.name
        if entry.is_symlink():
            continue
        _move_without_overwrite(entry, destination)
        _relative_symlink(destination, data_root / entry.name)

    selected_by_id = {item.source_id: item for item in selected}
    migration_rows: list[dict[str, Any]] = []
    for candidate in candidates:
        original = data_root / candidate.relative_path
        legacy_path = legacy_root / candidate.relative_path
        if not legacy_path.exists() and not legacy_path.is_symlink():
            raise FileNotFoundError(f"migrated source is missing: {legacy_path}")
        row = candidate.as_dict()
        row["legacy_path"] = str(legacy_path)
        row["current_path"] = str(legacy_path)
        row["migration_status"] = "archived_only"
        if candidate.source_id in selected_by_id:
            raw_path = (
                raw_root
                / candidate.context["batch"]
                / candidate.context["campaign"]
                / candidate.context["job"]
                / candidate.source_id
                / "source.pt"
            )
            if raw_path.exists():
                digest = _sha256(raw_path)
            else:
                digest = _sha256(legacy_path)
                _move_without_overwrite(legacy_path, raw_path)
            _relative_symlink(raw_path, legacy_path)
            row["raw_path"] = str(raw_path)
            row["current_path"] = str(raw_path)
            row["sha256"] = digest
            row["migration_status"] = "selected_moved_to_raw"
        else:
            row["sha256"] = _sha256(legacy_path)
        migration_rows.append(row)

    _jsonl_dump(unified_root / "manifest" / "migration_manifest.jsonl", migration_rows)
    _json_dump(
        unified_root / "manifest" / "migration_summary.json",
        {
            "top_level_archived": top_level_names,
            "candidate_count": len(candidates),
            "selected_count": len(selected),
            "selected_moved_to_raw": sum(
                row["migration_status"] == "selected_moved_to_raw"
                for row in migration_rows
            ),
            "legacy_root": str(legacy_root),
            "raw_root": str(raw_root),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        },
    )
    return migration_rows


def _load_selected_after_migration(unified_root: Path) -> list[dict[str, Any]]:
    rows = _jsonl_load(unified_root / "manifest" / "migration_manifest.jsonl")
    selected = [
        row for row in rows if row.get("migration_status") == "selected_moved_to_raw"
    ]
    if not selected:
        raise RuntimeError("no migrated GSPT1 source PTs found")
    return sorted(selected, key=lambda row: str(row["source_id"]))


def _reconstruction_output(unified_root: Path, row: dict[str, Any]) -> Path:
    return (
        unified_root
        / "reconstructed"
        / str(row["batch"])
        / str(row["campaign"])
        / str(row["job"])
        / str(row["source_id"])
    )


def _flatten_sdf_outputs(evaluation_root: Path, molecules_root: Path) -> int:
    sdf_files = sorted(evaluation_root.rglob("*.sdf"))
    if not sdf_files:
        return 0
    molecules_root.mkdir(parents=True, exist_ok=True)
    moved = 0
    for source in sdf_files:
        if molecules_root in source.parents:
            continue
        destination = molecules_root / f"sample_{moved:06d}.sdf"
        while destination.exists():
            moved += 1
            destination = molecules_root / f"sample_{moved:06d}.sdf"
        source.rename(destination)
        moved += 1
    return moved


def reconstruct_one(
    row: dict[str, Any],
    unified_root: Path,
    protein_root: Path,
    receptor_pdb: Path,
) -> dict[str, Any]:
    source_id = str(row["source_id"])
    output_root = _reconstruction_output(unified_root, row)
    state_path = output_root / "reconstruction_summary.json"
    source_path = Path(str(row.get("raw_path") or row.get("current_path")))
    if not source_path.exists():
        return {
            **row,
            "reconstruction_status": "missing_source",
            "reconstruction_output": str(output_root),
        }
    expected_sha = str(row.get("sha256", ""))
    if state_path.exists():
        try:
            previous = json.loads(state_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            previous = {}
        if (
            previous.get("reconstruction_status") == "success"
            and previous.get("source_sha256") == expected_sha
            and previous.get("vina_modes") == "none"
        ):
            return previous | {"reconstruction_status": "skipped_existing"}

    evaluation_root = output_root / "evaluation"
    log_path = output_root / "evaluator.log"
    output_root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "DIFFDYNAMIC_SKIP_EVAL_RECORDS": "1",
            "DIFFDYNAMIC_SKIP_SCORE_CATEGORIES": "1",
            "EVAL_MOLECULE_ID_SUFFIX": source_id,
        }
    )
    command = [
        sys.executable,
        str(EVALUATOR),
        str(source_path),
        "--protein_root",
        str(protein_root),
        "--receptor_pdb",
        str(receptor_pdb),
        "--output_dir",
        str(evaluation_root),
        "--vina-modes",
        "none",
        "--enable_isolation",
        "--no-distribution-plots",
        "--save_intermediate_interval",
        "0",
    ]
    started = time.time()
    with log_path.open("w", encoding="utf-8") as log_handle:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    molecule_count = _flatten_sdf_outputs(
        evaluation_root, output_root / "molecules"
    )
    result = {
        **row,
        "reconstruction_status": (
            "success" if completed.returncode == 0 else "partial_or_failed"
        ),
        "returncode": completed.returncode,
        "elapsed_seconds": round(time.time() - started, 3),
        "source_sha256": expected_sha or _sha256(source_path),
        "reconstruction_output": str(output_root),
        "molecules_dir": str(output_root / "molecules"),
        "molecule_file_count": molecule_count,
        "evaluator_log": str(log_path),
        "vina_modes": "none",
        "reconstruction_command": command,
    }
    _json_dump(state_path, result)
    return result


def reconstruct_all(
    unified_root: Path,
    protein_root: Path,
    receptor_pdb: Path,
    workers: int,
    limit: int | None = None,
) -> dict[str, Any]:
    rows = _load_selected_after_migration(unified_root)
    if limit is not None:
        rows = rows[:limit]
    manifest_path = unified_root / "manifest" / "reconstruction_manifest.jsonl"
    existing = {
        row.get("source_id"): row
        for row in _jsonl_load(manifest_path)
        if row.get("source_id")
    }
    pending = []
    for row in rows:
        prior = existing.get(row["source_id"])
        if prior and prior.get("source_sha256") == row.get("sha256"):
            state = _reconstruction_output(unified_root, row) / "reconstruction_summary.json"
            if state.exists():
                continue
        pending.append(row)

    results = dict(existing)
    if pending:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            futures = {
                executor.submit(
                    reconstruct_one,
                    row,
                    unified_root,
                    protein_root,
                    receptor_pdb,
                ): row["source_id"]
                for row in pending
            }
            for future in as_completed(futures):
                result = future.result()
                results[result["source_id"]] = result
                print(
                    json.dumps(
                        {
                            "source_id": result["source_id"],
                            "status": result.get("reconstruction_status"),
                            "molecule_file_count": result.get("molecule_file_count", 0),
                        },
                        ensure_ascii=True,
                    ),
                    flush=True,
                )
                _jsonl_dump(
                    manifest_path,
                    (results[key] for key in sorted(results)),
                )
    _jsonl_dump(manifest_path, (results[key] for key in sorted(results)))
    summary = {
        "source_count": len(rows),
        "processed_count": len(pending),
        "success_count": sum(
            row.get("reconstruction_status") in {"success", "skipped_existing"}
            for row in results.values()
        ),
        "failed_count": sum(
            row.get("reconstruction_status") not in {"success", "skipped_existing"}
            for row in results.values()
        ),
        "molecule_file_count": sum(
            int(row.get("molecule_file_count", 0) or 0) for row in results.values()
        ),
        "workers": workers,
        "vina_modes": "none",
        "manifest": str(manifest_path),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    _json_dump(unified_root / "manifest" / "reconstruction_summary.json", summary)
    return summary


def _import_chemistry():
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import AllChem, rdMolDescriptors

    RDLogger.DisableLog("rdApp.*")
    return Chem, DataStructs, AllChem, rdMolDescriptors


def _iter_sdf(path: Path) -> Iterator[Any]:
    from utils.gspt1_scaffold_prior import iter_sdf_molecules

    yield from iter_sdf_molecules(path)


def _standardize(mol: Any, Chem: Any) -> Any | None:
    if mol is None:
        return None
    try:
        fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    except Exception:
        fragments = (mol,)
    if not fragments:
        return None
    molecule = max(
        fragments,
        key=lambda item: (item.GetNumHeavyAtoms(), item.GetNumAtoms()),
    )
    try:
        molecule = Chem.RemoveHs(molecule, sanitize=False)
    except Exception:
        pass
    for ops in (
        Chem.SanitizeFlags.SANITIZE_ALL,
        Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES,
    ):
        try:
            Chem.SanitizeMol(molecule, sanitizeOps=ops)
            return molecule
        except Exception:
            continue
    return None


def _outside_molecule(molecule: Any, pattern: Any, Chem: Any) -> Any | None:
    match = molecule.GetSubstructMatch(pattern)
    if not match:
        return None
    editable = Chem.RWMol(molecule)
    for atom_index in sorted(match, reverse=True):
        editable.RemoveAtom(int(atom_index))
    return _standardize(editable.GetMol(), Chem)


def _max_bond_length(molecule: Any, Chem: Any) -> float:
    if molecule.GetNumConformers() == 0:
        return 0.0
    conformer = molecule.GetConformer()
    maximum = 0.0
    for bond in molecule.GetBonds():
        begin = conformer.GetAtomPosition(bond.GetBeginAtomIdx())
        end = conformer.GetAtomPosition(bond.GetEndAtomIdx())
        distance = math.sqrt(
            (begin.x - end.x) ** 2
            + (begin.y - end.y) ** 2
            + (begin.z - end.z) ** 2
        )
        maximum = max(maximum, distance)
    return maximum


def _reference_records(reference_path: Path, Chem: Any, AllChem: Any) -> list[dict[str, Any]]:
    patterns = [Chem.MolFromSmarts(smarts) for smarts in SCAFFOLD_SMARTS]
    records = []
    seen = set()
    for index, raw in enumerate(_iter_sdf(reference_path)):
        molecule = _standardize(raw, Chem)
        if molecule is None:
            continue
        smiles = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False)
        if smiles in seen:
            continue
        seen.add(smiles)
        matched_pattern = next(
            (pattern for pattern in patterns if pattern and molecule.HasSubstructMatch(pattern)),
            None,
        )
        outside = (
            _outside_molecule(molecule, matched_pattern, Chem)
            if matched_pattern is not None
            else None
        )
        records.append(
            {
                "index": index,
                "smiles": smiles,
                "molecule": molecule,
                "fingerprint": AllChem.GetMorganFingerprintAsBitVect(
                    molecule, 2, nBits=2048
                ),
                "outside": outside,
                "outside_fingerprint": (
                    AllChem.GetMorganFingerprintAsBitVect(outside, 2, nBits=2048)
                    if outside is not None
                    else None
                ),
                "class_name": (
                    "f_main" if any(atom.GetAtomicNum() == 9 for atom in molecule.GetAtoms())
                    else "no_f"
                ),
                "pattern": matched_pattern,
            }
        )
    return records


def _source_for_molecule(path: Path, unified_root: Path, source_rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    try:
        relative = path.resolve().relative_to((unified_root / "reconstructed").resolve())
        source_id = relative.parts[-2]
    except (ValueError, IndexError):
        source_id = "unknown"
    return source_rows.get(source_id, {"source_id": source_id})


def _read_sdf_record(path: Path, index: int, Chem: Any) -> Any | None:
    for current_index, raw in enumerate(_iter_sdf(path)):
        if current_index == index:
            return _standardize(raw, Chem)
    return None


def audit_library(
    unified_root: Path,
    reference_path: Path,
    top_n: int = 500,
) -> dict[str, Any]:
    Chem, DataStructs, AllChem, rdMolDescriptors = _import_chemistry()
    analysis_root = unified_root / "analysis"
    analysis_root.mkdir(parents=True, exist_ok=True)
    reconstruction_manifest = _jsonl_load(
        unified_root / "manifest" / "reconstruction_manifest.jsonl"
    )
    source_rows = {row["source_id"]: row for row in reconstruction_manifest}
    references = _reference_records(reference_path, Chem, AllChem)
    reference_by_smiles = {row["smiles"]: row for row in references}
    reference_fps = [row for row in references if row["fingerprint"] is not None]

    molecule_paths = sorted(
        path
        for path in (unified_root / "reconstructed").rglob("*.sdf")
        if "molecules" in path.parts
    )
    all_records: list[dict[str, Any]] = []
    for path in molecule_paths:
        source = _source_for_molecule(path, unified_root, source_rows)
        for molecule_index, raw in enumerate(_iter_sdf(path)):
            try:
                fragments = Chem.GetMolFrags(raw, asMols=True, sanitizeFrags=False)
                fragment_count = len(fragments)
            except Exception:
                fragment_count = 0
            molecule = _standardize(raw, Chem)
            if molecule is None:
                all_records.append(
                    {
                        "source_id": source.get("source_id", "unknown"),
                        "source_path": str(path),
                        "molecule_index": molecule_index,
                        "valid": 0,
                    }
                )
                continue
            smiles = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False)
            fingerprint = AllChem.GetMorganFingerprintAsBitVect(
                molecule, 2, nBits=2048
            )
            best_full = max(
                (
                    (
                        float(DataStructs.TanimotoSimilarity(
                            fingerprint, reference["fingerprint"]
                        )),
                        reference,
                    )
                    for reference in reference_fps
                ),
                key=lambda item: item[0],
                default=(0.0, None),
            )
            exact_reference = reference_by_smiles.get(smiles)
            patterns = [Chem.MolFromSmarts(smarts) for smarts in SCAFFOLD_SMARTS]
            matched_pattern = next(
                (
                    pattern
                    for pattern in patterns
                    if pattern and molecule.HasSubstructMatch(pattern)
                ),
                None,
            )
            outside = (
                _outside_molecule(molecule, matched_pattern, Chem)
                if matched_pattern is not None
                else None
            )
            side_score = 0.0
            side_reference = None
            if outside is not None:
                outside_fp = AllChem.GetMorganFingerprintAsBitVect(
                    outside, 2, nBits=2048
                )
                side_score, side_reference = max(
                    (
                        (
                            float(DataStructs.TanimotoSimilarity(
                                outside_fp, reference["outside_fingerprint"]
                            )),
                            reference,
                        )
                        for reference in reference_fps
                        if reference["outside_fingerprint"] is not None
                    ),
                    key=lambda item: item[0],
                    default=(0.0, None),
                )
            row = {
                "source_id": source.get("source_id", "unknown"),
                "source_kind": source.get("source_kind", ""),
                "batch": source.get("batch", ""),
                "campaign": source.get("campaign", ""),
                "job": source.get("job", ""),
                "class_name": source.get("class_name", "unknown"),
                "source_path": str(path),
                "molecule_index": molecule_index,
                "valid": 1,
                "canonical_smiles": smiles,
                "heavy_atoms": molecule.GetNumHeavyAtoms(),
                "fragment_count_before_standardization": fragment_count,
                "is_pure_scaffold": int(
                    matched_pattern is not None
                    and molecule.GetNumHeavyAtoms() == matched_pattern.GetNumAtoms()
                ),
                "full_similarity": best_full[0],
                "full_reference_smiles": (
                    best_full[1]["smiles"] if best_full[1] else ""
                ),
                "side_similarity": side_score,
                "side_reference_smiles": (
                    side_reference["smiles"] if side_reference else ""
                ),
                "exact": int(exact_reference is not None),
                "reference_index": (
                    exact_reference["index"] if exact_reference else (
                        best_full[1]["index"] if best_full[1] else ""
                    )
                ),
                "formula": rdMolDescriptors.CalcMolFormula(molecule),
                "max_bond_length": round(_max_bond_length(molecule, Chem), 4),
            }
            row["long_bond_flag"] = int(row["max_bond_length"] > 2.2)
            all_records.append(row)

    priority = {"result": 0, "chunk_input": 1}
    unique: dict[str, dict[str, Any]] = {}
    duplicate_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in all_records:
        smiles = row.get("canonical_smiles")
        if not smiles:
            continue
        duplicate_groups[smiles].append(row)
        current = unique.get(smiles)
        if current is None or priority.get(row.get("source_kind", ""), 9) < priority.get(
            current.get("source_kind", ""), 9
        ):
            unique[smiles] = row

    ranked = sorted(
        unique.values(),
        key=lambda row: (
            -float(row.get("full_similarity", 0.0)),
            -float(row.get("side_similarity", 0.0)),
            row.get("canonical_smiles", ""),
        ),
    )
    fields = [
        "rank",
        *sorted({key for row in all_records for key in row}),
    ]
    records_path = analysis_root / "records.csv"
    with records_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for rank, row in enumerate(ranked, start=1):
            writer.writerow({**row, "rank": rank})

    top_fields = fields
    with (analysis_root / "top_similarity.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=top_fields, extrasaction="ignore")
        writer.writeheader()
        for rank, row in enumerate(ranked[:top_n], start=1):
            writer.writerow({**row, "rank": rank})

    exact_rows = [row for row in ranked if row.get("exact")]
    with (analysis_root / "exact_hits.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=top_fields, extrasaction="ignore")
        writer.writeheader()
        for rank, row in enumerate(exact_rows, start=1):
            writer.writerow({**row, "rank": rank})

    duplicate_fields = [
        "canonical_smiles",
        "duplicate_count",
        "source_ids",
        "source_paths",
    ]
    with (analysis_root / "duplicate_clusters.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=duplicate_fields)
        writer.writeheader()
        for smiles, rows in sorted(
            duplicate_groups.items(), key=lambda item: (-len(item[1]), item[0])
        ):
            if len(rows) < 2:
                continue
            writer.writerow(
                {
                    "canonical_smiles": smiles,
                    "duplicate_count": len(rows),
                    "source_ids": ";".join(sorted({str(row.get("source_id", "")) for row in rows})),
                    "source_paths": ";".join(sorted({str(row.get("source_path", "")) for row in rows})),
                }
            )

    needed_indices: dict[str, set[int]] = defaultdict(set)
    for row in ranked:
        needed_indices[str(row["source_path"])].add(int(row["molecule_index"]))
    molecule_cache: dict[tuple[str, int], Any] = {}
    for source_path, indices in needed_indices.items():
        for molecule_index, raw in enumerate(_iter_sdf(Path(source_path))):
            if molecule_index in indices:
                molecule_cache[(source_path, molecule_index)] = _standardize(raw, Chem)

    library_path = analysis_root / "unique_library.sdf"
    writer = Chem.SDWriter(str(library_path))
    for row in ranked:
        molecule = molecule_cache.get(
            (str(row["source_path"]), int(row["molecule_index"]))
        )
        if molecule is None:
            molecule = Chem.MolFromSmiles(row["canonical_smiles"])
        if molecule is None:
            continue
        for key, value in row.items():
            if key == "canonical_smiles" or value is None:
                continue
            molecule.SetProp(str(key), str(value))
        writer.write(molecule)
    writer.close()

    summary = {
        "reference_count": len(references),
        "sdf_file_count": len(molecule_paths),
        "raw_record_count": len(all_records),
        "valid_record_count": sum(bool(row.get("valid")) for row in all_records),
        "unique_record_count": len(unique),
        "exact_count": len(exact_rows),
        "best_full_similarity": ranked[0]["full_similarity"] if ranked else 0.0,
        "best_side_similarity": max(
            (float(row.get("side_similarity", 0.0)) for row in ranked),
            default=0.0,
        ),
        "long_bond_record_count": sum(
            bool(row.get("long_bond_flag")) for row in ranked
        ),
        "full_similarity_threshold_counts": {
            str(threshold): sum(
                float(row.get("full_similarity", 0.0)) >= threshold for row in ranked
            )
            for threshold in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
        },
        "class_counts": dict(Counter(row.get("class_name", "unknown") for row in ranked)),
        "records": str(records_path),
        "top_similarity": str(analysis_root / "top_similarity.csv"),
        "exact_hits": str(analysis_root / "exact_hits.csv"),
        "unique_library": str(library_path),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    _json_dump(analysis_root / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("inventory", "migrate", "reconstruct", "audit", "all")
    )
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--unified-root", type=Path, default=DEFAULT_UNIFIED_ROOT)
    parser.add_argument("--protein-root", type=Path, default=DEFAULT_PROTEIN_ROOT)
    parser.add_argument("--receptor-pdb", type=Path, default=DEFAULT_RECEPTOR)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--workers", type=int, default=90)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--top-n", type=int, default=500)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_root = args.data_root.resolve()
    unified_root = args.unified_root.resolve()
    if args.workers < 1:
        raise SystemExit("--workers must be positive")

    if args.command in {"inventory", "migrate", "all"}:
        candidates, selected = inventory_rows(data_root)
        suffix = ".dry_run" if args.dry_run else ""
        paths = write_inventory(
            data_root, unified_root, candidates, selected, suffix=suffix
        )
        print(
            json.dumps(
                inventory_summary(data_root, candidates, selected),
                ensure_ascii=True,
            )
        )
        print("inventory:", paths[0])
        if args.command == "inventory" or args.dry_run:
            return 0
        migrate_corpus(data_root, unified_root, candidates, selected)
        print("migration complete:", unified_root)

    if args.command in {"reconstruct", "all"}:
        if not args.protein_root.exists():
            raise SystemExit(f"protein root does not exist: {args.protein_root}")
        if not args.receptor_pdb.exists():
            raise SystemExit(f"receptor PDB does not exist: {args.receptor_pdb}")
        summary = reconstruct_all(
            unified_root,
            args.protein_root,
            args.receptor_pdb,
            args.workers,
            limit=args.limit,
        )
        print(json.dumps(summary, ensure_ascii=True))

    if args.command in {"audit", "all"}:
        if not args.reference.exists():
            raise SystemExit(f"reference SDF does not exist: {args.reference}")
        summary = audit_library(unified_root, args.reference, top_n=args.top_n)
        print(json.dumps(summary, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
