#!/usr/bin/env python3
"""Compare GSPT1 molecules after an in-memory all-carbon conversion.

This is an analysis-only view.  Input SDF files and the original audit records
are never rewritten.  Every non-hydrogen atom is represented as carbon while
bond connectivity, bond order, aromatic flags, and ring topology are kept.
The original canonical SMILES remains the deduplication key so distinct
elemental molecules are not silently collapsed by the analysis transform.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.gspt1_scaffold_prior import iter_sdf_molecules  # noqa: E402


RDLogger.DisableLog("rdApp.*")

SCAFFOLD_SMARTS = (
    "O=C1CCC(N2Cc3ccccc3C2=O)C(=O)N1",
    "O=C1CCC(N2Cc3c(F)cccc3C2=O)C(=O)N1",
)
FP_RADIUS = 2
FP_BITS = 2048
THRESHOLDS = (0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def _sanitize(molecule: Chem.Mol) -> Chem.Mol | None:
    for ops in (
        Chem.SanitizeFlags.SANITIZE_ALL,
        Chem.SanitizeFlags.SANITIZE_ALL
        ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES,
    ):
        try:
            Chem.SanitizeMol(molecule, sanitizeOps=ops)
            return molecule
        except Exception:
            continue
    return None


def _standardize(molecule: Chem.Mol | None) -> Chem.Mol | None:
    if molecule is None:
        return None
    try:
        fragments = Chem.GetMolFrags(
            molecule,
            asMols=True,
            sanitizeFrags=False,
        )
    except Exception:
        fragments = (molecule,)
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
    return _sanitize(molecule)


def _canonical_smiles(molecule: Chem.Mol | None) -> str | None:
    molecule = _standardize(molecule)
    if molecule is None:
        return None
    try:
        return Chem.MolToSmiles(
            molecule,
            canonical=True,
            isomericSmiles=False,
        )
    except Exception:
        return None


def _iter_reference(path: Path) -> Iterator[Chem.Mol]:
    for molecule in iter_sdf_molecules(path):
        standardized = _standardize(molecule)
        if standardized is not None:
            yield standardized


def _matched_scaffold_pattern(
    molecule: Chem.Mol,
    patterns: Sequence[Chem.Mol | None],
) -> Chem.Mol | None:
    has_fluorine = any(
        atom.GetAtomicNum() == 9 for atom in molecule.GetAtoms()
    )
    order = (1, 0) if has_fluorine else (0, 1)
    for index in order:
        pattern = patterns[index]
        if pattern is not None and molecule.HasSubstructMatch(pattern):
            return pattern
    return None


def _outside_molecule(
    molecule: Chem.Mol,
    pattern: Chem.Mol | None,
) -> Chem.Mol | None:
    if pattern is None:
        return None
    match = molecule.GetSubstructMatch(pattern)
    if not match:
        return None
    editable = Chem.RWMol(molecule)
    for atom_index in sorted(match, reverse=True):
        editable.RemoveAtom(int(atom_index))
    return _standardize(editable.GetMol())


def _to_all_carbon(molecule: Chem.Mol) -> Chem.Mol:
    """Map heavy atoms to neutral C without changing the molecular graph."""
    editable = Chem.RWMol(molecule)
    for atom in editable.GetAtoms():
        if atom.GetAtomicNum() <= 1:
            continue
        atom.SetAtomicNum(6)
        atom.SetFormalCharge(0)
        atom.SetIsotope(0)
        atom.SetNumExplicitHs(0)
        atom.SetNoImplicit(False)
        atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    result = editable.GetMol()
    result.UpdatePropertyCache(strict=False)
    return result


def _fingerprint(molecule: Chem.Mol) -> Any:
    return AllChem.GetMorganFingerprintAsBitVect(
        molecule,
        FP_RADIUS,
        nBits=FP_BITS,
    )


def _allc_view(molecule: Chem.Mol | None) -> tuple[str, Any] | None:
    if molecule is None:
        return None
    transformed = _to_all_carbon(molecule)
    try:
        smiles = Chem.MolToSmiles(
            transformed,
            canonical=True,
            isomericSmiles=False,
        )
        return smiles, _fingerprint(transformed)
    except Exception:
        return None


def _reference_records(reference_path: Path) -> list[dict[str, Any]]:
    patterns = [Chem.MolFromSmarts(smarts) for smarts in SCAFFOLD_SMARTS]
    records = []
    seen = set()
    for index, molecule in enumerate(_iter_reference(reference_path)):
        smiles = _canonical_smiles(molecule)
        if smiles is None or smiles in seen:
            continue
        seen.add(smiles)
        pattern = _matched_scaffold_pattern(molecule, patterns)
        outside = _outside_molecule(molecule, pattern)
        full_view = _allc_view(molecule)
        side_view = _allc_view(outside)
        if full_view is None:
            continue
        records.append(
            {
                "index": index,
                "smiles": smiles,
                "allc_smiles": full_view[0],
                "fingerprint": full_view[1],
                "outside_smiles": (
                    _canonical_smiles(outside) if outside is not None else ""
                ),
                "outside_allc_smiles": (
                    side_view[0] if side_view is not None else ""
                ),
                "outside_fingerprint": (
                    side_view[1] if side_view is not None else None
                ),
            }
        )
    return records


def _load_generated_records(records_path: Path) -> list[dict[str, str]]:
    with records_path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _score(
    fingerprint: Any,
    references: Iterable[dict[str, Any]],
    fingerprint_key: str,
) -> tuple[float, dict[str, Any] | None]:
    best_score = 0.0
    best_reference = None
    for reference in references:
        reference_fp = reference.get(fingerprint_key)
        if reference_fp is None:
            continue
        score = float(DataStructs.TanimotoSimilarity(fingerprint, reference_fp))
        if best_reference is None or score > best_score:
            best_score = score
            best_reference = reference
    return best_score, best_reference


def _rank_key(row: dict[str, Any]) -> tuple[float, float, str]:
    return (
        -float(row["allc_full_similarity"]),
        -float(row["allc_side_similarity"]),
        str(row["canonical_smiles"]),
    )


def _write_rows(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        return
    fields = [
        "rank",
        *sorted({
            key for row in rows for key in row if key != "rank"
        }),
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def audit(
    records_path: Path,
    reference_path: Path,
    output_dir: Path,
    top_n: int = 500,
) -> dict[str, Any]:
    generated = _load_generated_records(records_path)
    references = _reference_records(reference_path)
    reference_by_allc = {
        reference["allc_smiles"]: reference for reference in references
    }
    side_references = [
        reference
        for reference in references
        if reference.get("outside_fingerprint") is not None
    ]
    patterns = [Chem.MolFromSmarts(smarts) for smarts in SCAFFOLD_SMARTS]

    rows = []
    invalid_count = 0
    exact_count = 0
    side_exact_count = 0
    for source_row in generated:
        original_smiles = source_row.get("canonical_smiles", "")
        molecule = _standardize(Chem.MolFromSmiles(original_smiles))
        full_view = _allc_view(molecule)
        if molecule is None or full_view is None:
            invalid_count += 1
            continue

        pattern = _matched_scaffold_pattern(molecule, patterns)
        outside = _outside_molecule(molecule, pattern)
        side_view = _allc_view(outside)
        full_score, full_reference = _score(
            full_view[1], references, "fingerprint"
        )
        side_score = 0.0
        side_reference = None
        if side_view is not None:
            side_score, side_reference = _score(
                side_view[1], side_references, "outside_fingerprint"
            )

        allc_exact = full_view[0] in reference_by_allc
        allc_side_exact = bool(
            side_view is not None
            and any(
                side_view[0] == reference.get("outside_allc_smiles")
                for reference in side_references
            )
        )
        exact_count += int(allc_exact)
        side_exact_count += int(allc_side_exact)
        rows.append(
            {
                "batch": source_row.get("batch", ""),
                "campaign": source_row.get("campaign", ""),
                "class_name": source_row.get("class_name", "unknown"),
                "source_id": source_row.get("source_id", ""),
                "source_path": source_row.get("source_path", ""),
                "molecule_index": source_row.get("molecule_index", ""),
                "canonical_smiles": original_smiles,
                "allc_smiles": full_view[0],
                "outside_allc_smiles": (
                    side_view[0] if side_view is not None else ""
                ),
                "heavy_atoms": molecule.GetNumHeavyAtoms(),
                "allc_full_similarity": full_score,
                "allc_side_similarity": side_score,
                "allc_exact": int(allc_exact),
                "allc_side_exact": int(allc_side_exact),
                "full_reference_index": (
                    full_reference["index"] if full_reference else ""
                ),
                "full_reference_smiles": (
                    full_reference["smiles"] if full_reference else ""
                ),
                "full_reference_allc_smiles": (
                    full_reference["allc_smiles"] if full_reference else ""
                ),
                "side_reference_index": (
                    side_reference["index"] if side_reference else ""
                ),
                "side_reference_smiles": (
                    side_reference["smiles"] if side_reference else ""
                ),
                "side_reference_allc_smiles": (
                    side_reference["outside_allc_smiles"]
                    if side_reference else ""
                ),
                "original_full_similarity": source_row.get(
                    "full_similarity", ""
                ),
                "original_side_similarity": source_row.get(
                    "side_similarity", ""
                ),
            }
        )

    rows.sort(key=_rank_key)
    side_ranked = sorted(
        rows,
        key=lambda row: (
            -float(row["allc_side_similarity"]),
            -float(row["allc_full_similarity"]),
            str(row["canonical_smiles"]),
        ),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    ranked_rows = [{"rank": index, **row} for index, row in enumerate(rows, 1)]
    side_rows = [
        {"rank": index, **row}
        for index, row in enumerate(side_ranked, 1)
    ]
    _write_rows(output_dir / "records.csv", ranked_rows)
    _write_rows(output_dir / "top_similarity.csv", ranked_rows[:top_n])
    _write_rows(output_dir / "top_side_similarity.csv", side_rows[:top_n])

    class_summary = {}
    for class_name in sorted({row["class_name"] for row in rows}):
        class_rows = [row for row in rows if row["class_name"] == class_name]
        class_summary[class_name] = {
            "count": len(class_rows),
            "best_full": max(
                (float(row["allc_full_similarity"]) for row in class_rows),
                default=0.0,
            ),
            "best_side": max(
                (float(row["allc_side_similarity"]) for row in class_rows),
                default=0.0,
            ),
        }

    summary = {
        "mode": "all_carbon_graph_tanimoto",
        "description": (
            "All heavy atoms are mapped to neutral carbon in memory; input "
            "SDF and original records are unchanged."
        ),
        "records_path": str(records_path),
        "reference_path": str(reference_path),
        "output_dir": str(output_dir),
        "reference_count": len(references),
        "generated_input_count": len(generated),
        "generated_valid_count": len(rows),
        "invalid_count": invalid_count,
        "unique_key": "original_canonical_smiles",
        "allc_exact_count": exact_count,
        "allc_side_exact_count": side_exact_count,
        "best_full_similarity": (
            float(rows[0]["allc_full_similarity"]) if rows else 0.0
        ),
        "best_side_similarity": max(
            (float(row["allc_side_similarity"]) for row in rows),
            default=0.0,
        ),
        "full_similarity_threshold_counts": {
            str(threshold): sum(
                float(row["allc_full_similarity"]) >= threshold
                for row in rows
            )
            for threshold in THRESHOLDS
        },
        "side_similarity_threshold_counts": {
            str(threshold): sum(
                float(row["allc_side_similarity"]) >= threshold
                for row in rows
            )
            for threshold in THRESHOLDS
        },
        "class_summary": class_summary,
        "top_similarity": str(output_dir / "top_similarity.csv"),
        "top_side_similarity": str(output_dir / "top_side_similarity.csv"),
        "records": str(output_dir / "records.csv"),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    reference_path_out = output_dir / "reference_allc.csv"
    with reference_path_out.open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fields = (
            "index",
            "smiles",
            "allc_smiles",
            "outside_smiles",
            "outside_allc_smiles",
        )
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            {field: reference.get(field, "") for field in fields}
            for reference in references
        )
    summary["reference_allc"] = str(reference_path_out)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-csv", type=Path, required=True)
    parser.add_argument("--reference-sdf", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-n", type=int, default=500)
    args = parser.parse_args()
    summary = audit(
        args.records_csv.resolve(),
        args.reference_sdf.resolve(),
        args.output_dir.resolve(),
        top_n=args.top_n,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
