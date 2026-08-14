#!/usr/bin/env python3
"""Class-aware, no-docking audit for GSPT1 scaffold campaigns."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, Optional

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.gspt1_smiles_audit import canonical_smiles, standardize_mol  # noqa: E402
from utils.gspt1_scaffold_prior import (  # noqa: E402
    SUPPORTED_GENERATION_ELEMENTS,
    iter_sdf_molecules,
)


RDLogger.DisableLog("rdApp.*")


def _fingerprint(molecule: Chem.Mol):
    return AllChem.GetMorganFingerprintAsBitVect(molecule, 2, nBits=2048)


def _outside_molecule(
    molecule: Chem.Mol,
    scaffold_pattern: Chem.Mol,
) -> Optional[Chem.Mol]:
    match = molecule.GetSubstructMatch(scaffold_pattern)
    if not match:
        return None
    editable = Chem.RWMol(molecule)
    for atom_idx in sorted(match, reverse=True):
        editable.RemoveAtom(int(atom_idx))
    outside = editable.GetMol()
    if outside.GetNumHeavyAtoms() == 0:
        return None
    try:
        Chem.SanitizeMol(outside)
    except Exception:
        try:
            Chem.SanitizeMol(
                outside,
                sanitizeOps=(
                    Chem.SanitizeFlags.SANITIZE_ALL
                    ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES
                ),
            )
        except Exception:
            return None
    return outside


def _attachment_metrics(
    molecule: Chem.Mol,
    scaffold_match: tuple[int, ...],
) -> dict[str, object]:
    """Report reconstructed scaffold/extra connectivity without editing it."""
    scaffold_set = {int(index) for index in scaffold_match}
    extra_set = {
        int(atom.GetIdx())
        for atom in molecule.GetAtoms()
        if int(atom.GetIdx()) not in scaffold_set
        and atom.GetAtomicNum() > 1
    }
    cross_bonds = []
    bond_lengths = []
    conformer = None
    try:
        conformer = molecule.GetConformer()
    except Exception:
        pass
    for bond in molecule.GetBonds():
        begin = int(bond.GetBeginAtomIdx())
        end = int(bond.GetEndAtomIdx())
        crosses = (begin in scaffold_set) != (end in scaffold_set)
        if not crosses:
            continue
        cross_bonds.append((begin, end))
        if conformer is not None:
            begin_pos = conformer.GetAtomPosition(begin)
            end_pos = conformer.GetAtomPosition(end)
            distance = (
                (begin_pos.x - end_pos.x) ** 2
                + (begin_pos.y - end_pos.y) ** 2
                + (begin_pos.z - end_pos.z) ** 2
            ) ** 0.5
            bond_lengths.append(float(distance))

    components = 0
    unseen = set(extra_set)
    while unseen:
        components += 1
        stack = [unseen.pop()]
        while stack:
            atom_idx = stack.pop()
            for neighbour in molecule.GetAtomWithIdx(atom_idx).GetNeighbors():
                neighbour_idx = int(neighbour.GetIdx())
                if neighbour_idx in unseen:
                    unseen.remove(neighbour_idx)
                    stack.append(neighbour_idx)

    return {
        "scaffold_extra_cross_bonds": len(cross_bonds),
        "extra_components": components,
        "cross_bond_lengths": [round(value, 4) for value in bond_lengths],
        "cross_bond_lengths_valid": bool(
            bond_lengths and all(0.9 <= value <= 2.1 for value in bond_lengths)
        ),
    }


def _selected_references(
    reference_sdf: Path,
    reference_indices: Iterable[int],
    scaffold_pattern: Chem.Mol,
) -> list[dict]:
    selected = {int(index) for index in reference_indices}
    records = []
    for index, raw_molecule in enumerate(iter_sdf_molecules(reference_sdf)):
        if index not in selected:
            continue
        molecule = standardize_mol(raw_molecule)
        smiles = canonical_smiles(molecule) if molecule is not None else None
        outside = (
            _outside_molecule(molecule, scaffold_pattern)
            if molecule is not None else None
        )
        if not smiles or outside is None:
            continue
        symbols = {
            atom.GetSymbol()
            for atom in molecule.GetAtoms()
            if atom.GetAtomicNum() > 1
        }
        records.append({
            "index": index,
            "smiles": smiles,
            "molecule": molecule,
            "fingerprint": _fingerprint(molecule),
            "outside_fingerprint": _fingerprint(outside),
            "reachable": symbols <= SUPPORTED_GENERATION_ELEMENTS,
        })
    return records


def _sdf_paths(roots: Iterable[Path]) -> list[Path]:
    paths: set[Path] = set()
    for root in roots:
        if root.is_file() and root.suffix.lower() == ".sdf":
            paths.add(root)
        elif root.exists():
            paths.update(root.rglob("*.sdf"))
    return sorted(
        path for path in paths
        if "scaffold" not in path.parts
        and not path.name.endswith("_scaffold.sdf")
    )


def audit_class(
    roots: Iterable[Path],
    reference_sdf: Path,
    reference_indices: Iterable[int],
    scaffold_smarts: str,
    allowed_n_extra: Iterable[int],
    output_dir: Path,
    *,
    class_name: str,
    top_n: int = 300,
) -> dict:
    pattern = Chem.MolFromSmarts(scaffold_smarts)
    if pattern is None:
        raise ValueError(f"invalid scaffold SMARTS: {scaffold_smarts}")
    references = _selected_references(
        reference_sdf, reference_indices, pattern
    )
    if not references:
        raise ValueError(f"no usable references for {class_name}")
    reference_smiles = {record["smiles"] for record in references}
    reachable_smiles = {
        record["smiles"] for record in references if record["reachable"]
    }
    allowed = {int(value) for value in allowed_n_extra}

    unique: dict[str, tuple[Chem.Mol, str]] = {}
    generated_records = 0
    for path in _sdf_paths(roots):
        for raw_molecule in iter_sdf_molecules(path):
            molecule = standardize_mol(raw_molecule)
            smiles = canonical_smiles(molecule) if molecule is not None else None
            if not smiles or molecule is None:
                continue
            match = molecule.GetSubstructMatch(pattern)
            if match and molecule.GetNumHeavyAtoms() == len(match):
                continue
            generated_records += 1
            unique.setdefault(smiles, (molecule, str(path)))

    ranked = []
    core_missing = 0
    allowed_count = 0
    count_distribution: Counter[int] = Counter()
    outside_elements: Counter[str] = Counter()
    for smiles, (molecule, source_path) in unique.items():
        match = molecule.GetSubstructMatch(pattern)
        if not match:
            core_missing += 1
            continue
        n_extra = molecule.GetNumHeavyAtoms() - len(match)
        count_distribution[n_extra] += 1
        if n_extra in allowed:
            allowed_count += 1
        scaffold_set = set(match)
        for atom in molecule.GetAtoms():
            if atom.GetIdx() not in scaffold_set and atom.GetAtomicNum() > 1:
                outside_elements[atom.GetSymbol()] += 1
        outside = _outside_molecule(molecule, pattern)
        if outside is None:
            continue
        attachment = _attachment_metrics(molecule, tuple(int(i) for i in match))
        fingerprint = _fingerprint(molecule)
        outside_fingerprint = _fingerprint(outside)
        full_reference = max(
            references,
            key=lambda record: DataStructs.TanimotoSimilarity(
                fingerprint, record["fingerprint"]
            ),
        )
        side_reference = max(
            references,
            key=lambda record: DataStructs.TanimotoSimilarity(
                outside_fingerprint, record["outside_fingerprint"]
            ),
        )
        ranked.append({
            "canonical_smiles": smiles,
            "source_path": source_path,
            "n_extra": n_extra,
            "count_allowed": n_extra in allowed,
            "full_similarity": float(DataStructs.TanimotoSimilarity(
                fingerprint, full_reference["fingerprint"]
            )),
            "side_similarity": float(DataStructs.TanimotoSimilarity(
                outside_fingerprint, side_reference["outside_fingerprint"]
            )),
            "reference_index": int(full_reference["index"]),
            "side_reference_index": int(side_reference["index"]),
            "exact": smiles in reference_smiles,
            "exact_reachable": smiles in reachable_smiles,
            **attachment,
        })
    ranked.sort(
        key=lambda record: (
            -record["side_similarity"],
            -record["full_similarity"],
            record["canonical_smiles"],
        )
    )

    core_records = len(unique) - core_missing
    summary = {
        "class_name": class_name,
        "reference_indices": sorted(int(index) for index in reference_indices),
        "reachable_reference_indices": sorted(
            record["index"] for record in references if record["reachable"]
        ),
        "allowed_n_extra": sorted(allowed),
        "generated_records": generated_records,
        "generated_unique": len(unique),
        "core_records": core_records,
        "core_retention_rate": core_records / max(len(unique), 1),
        "allowed_count_records": allowed_count,
        "allowed_count_rate": allowed_count / max(core_records, 1),
        "n_extra_distribution": {
            str(value): count for value, count in sorted(count_distribution.items())
        },
        "outside_element_counts": dict(outside_elements),
        "exact_count": sum(record["exact"] for record in ranked),
        "exact_reachable_count": sum(
            record["exact_reachable"] for record in ranked
        ),
        "best_full_similarity": max(
            (record["full_similarity"] for record in ranked), default=0.0
        ),
        "best_side_similarity": max(
            (record["side_similarity"] for record in ranked), default=0.0
        ),
        "cross_bond_count_distribution": dict(Counter(
            record["scaffold_extra_cross_bonds"] for record in ranked
        )),
        "extra_component_distribution": dict(Counter(
            record["extra_components"] for record in ranked
        )),
        "valid_cross_bond_rate": sum(
            record["cross_bond_lengths_valid"] for record in ranked
        ) / max(len(ranked), 1),
        "side_threshold_counts": {
            str(threshold): sum(
                record["side_similarity"] >= threshold for record in ranked
            )
            for threshold in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
        },
        "best": ranked[0] if ranked else None,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "records.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fieldnames = [
            "rank", "canonical_smiles", "side_similarity",
            "full_similarity", "n_extra", "count_allowed", "exact",
            "exact_reachable", "reference_index", "side_reference_index",
            "scaffold_extra_cross_bonds", "extra_components",
            "cross_bond_lengths", "cross_bond_lengths_valid", "source_path",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for rank, record in enumerate(ranked[:top_n], start=1):
            writer.writerow({"rank": rank, **record})
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", action="append", required=True)
    parser.add_argument("--reference-sdf", required=True)
    parser.add_argument("--reference-indices", required=True)
    parser.add_argument("--scaffold-smarts", required=True)
    parser.add_argument("--allowed-n-extra", required=True)
    parser.add_argument("--class-name", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-n", type=int, default=300)
    args = parser.parse_args()
    summary = audit_class(
        [Path(root) for root in args.root],
        Path(args.reference_sdf),
        [int(value) for value in args.reference_indices.split(",")],
        args.scaffold_smarts,
        [int(value) for value in args.allowed_n_extra.split(",")],
        Path(args.output_dir),
        class_name=args.class_name,
        top_n=args.top_n,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
