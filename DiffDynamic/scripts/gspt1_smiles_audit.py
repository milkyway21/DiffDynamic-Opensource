#!/usr/bin/env python3
"""Audit generated GSPT1 SDFs against known references without docking."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.gspt1_scaffold_prior import iter_sdf_molecules

RDLogger.DisableLog("rdApp.*")


@dataclass
class MoleculeRecord:
    canonical_smiles: str
    source_path: str
    molecule_id: str
    heavy_atoms: int
    similarity: float = 0.0
    reference_smiles: str = ""
    exact: bool = False


def _sanitize(mol: Chem.Mol) -> Optional[Chem.Mol]:
    for ops in (
        Chem.SanitizeFlags.SANITIZE_ALL,
        Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES,
    ):
        try:
            Chem.SanitizeMol(mol, sanitizeOps=ops)
            return mol
        except Exception:
            continue
    return None


def standardize_mol(mol: Chem.Mol) -> Optional[Chem.Mol]:
    """Take the largest component, remove explicit H, and sanitize."""
    if mol is None:
        return None
    try:
        fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    except Exception:
        fragments = (mol,)
    if not fragments:
        return None
    mol = max(fragments, key=lambda item: (item.GetNumHeavyAtoms(), item.GetNumAtoms()))
    try:
        mol = Chem.RemoveHs(mol, sanitize=False)
    except Exception:
        pass
    return _sanitize(mol)


def canonical_smiles(mol: Chem.Mol) -> Optional[str]:
    mol = standardize_mol(mol)
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
    except Exception:
        return None


def load_reference(path: str | Path) -> dict[str, Chem.Mol]:
    references: dict[str, Chem.Mol] = {}
    for mol in iter_sdf_molecules(path):
        standardized = standardize_mol(mol)
        smiles = canonical_smiles(standardized) if standardized is not None else None
        if smiles and smiles not in references:
            references[smiles] = standardized
    return references


def _prop(mol: Chem.Mol, names: Iterable[str]) -> str:
    for name in names:
        if mol.HasProp(name):
            return mol.GetProp(name)
    return ""


def iter_generated_records(roots: list[Path], scaffold_atoms: int) -> Iterable[MoleculeRecord]:
    paths: set[Path] = set()
    for root in roots:
        if root.is_file() and root.suffix.lower() == ".sdf":
            paths.add(root)
        elif root.exists():
            paths.update(root.rglob("*.sdf"))

    for path in sorted(paths):
        if "scaffold" in path.parts or path.name.endswith("_scaffold.sdf"):
            continue
        try:
            molecules = list(iter_sdf_molecules(path))
        except Exception as exc:
            print(f"[WARN] failed to read {path}: {exc}", file=sys.stderr)
            continue
        for index, mol in enumerate(molecules):
            standardized = standardize_mol(mol)
            smiles = canonical_smiles(standardized) if standardized is not None else None
            if not smiles or standardized is None:
                continue
            heavy_atoms = standardized.GetNumHeavyAtoms()
            if heavy_atoms <= scaffold_atoms:
                continue
            yield MoleculeRecord(
                canonical_smiles=smiles,
                source_path=str(path),
                molecule_id=_prop(mol, ("Molecule_ID", "molecule_id", "_Name"))
                or f"{path.stem}:{index}",
                heavy_atoms=heavy_atoms,
            )


def audit(
    roots: list[Path],
    reference_sdf: Path,
    output_dir: Path,
    scaffold_atoms: int = 18,
    top_n: int = 500,
) -> dict:
    references = load_reference(reference_sdf)
    reference_smiles = set(references)
    reference_fps = {
        smiles: AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        for smiles, mol in references.items()
    }

    unique: dict[str, MoleculeRecord] = {}
    n_records = 0
    for record in iter_generated_records(roots, scaffold_atoms):
        n_records += 1
        unique.setdefault(record.canonical_smiles, record)

    ranked: list[MoleculeRecord] = []
    for record in unique.values():
        record.exact = record.canonical_smiles in reference_smiles
        if record.exact:
            record.similarity = 1.0
            record.reference_smiles = record.canonical_smiles
        else:
            mol = Chem.MolFromSmiles(record.canonical_smiles)
            if mol is None or not reference_fps:
                continue
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            best_smiles, best_score = max(
                (
                    (ref_smiles, DataStructs.TanimotoSimilarity(fp, ref_fp))
                    for ref_smiles, ref_fp in reference_fps.items()
                ),
                key=lambda item: item[1],
            )
            record.similarity = float(best_score)
            record.reference_smiles = best_smiles
        ranked.append(record)
    ranked.sort(key=lambda item: (-item.similarity, item.canonical_smiles))

    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "records.csv"
    with records_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "rank", "canonical_smiles", "similarity", "reference_smiles",
                "exact", "heavy_atoms", "molecule_id", "source_path",
            ],
        )
        writer.writeheader()
        for rank, record in enumerate(ranked[:top_n], start=1):
            writer.writerow({
                "rank": rank,
                "canonical_smiles": record.canonical_smiles,
                "similarity": f"{record.similarity:.6f}",
                "reference_smiles": record.reference_smiles,
                "exact": int(record.exact),
                "heavy_atoms": record.heavy_atoms,
                "molecule_id": record.molecule_id,
                "source_path": record.source_path,
            })

    hits = [record for record in ranked if record.exact]
    summary = {
        "reference_count": len(reference_smiles),
        "generated_records": n_records,
        "generated_unique": len(unique),
        "exact_count": len(hits),
        "best_similarity": ranked[0].similarity if ranked else 0.0,
        "best": (
            {
                "canonical_smiles": ranked[0].canonical_smiles,
                "similarity": ranked[0].similarity,
                "source_path": ranked[0].source_path,
            }
            if ranked
            else None
        ),
        "threshold_counts": {
            str(threshold): sum(record.similarity >= threshold for record in ranked)
            for threshold in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
        },
        "exact_hits": [
            {
                "canonical_smiles": record.canonical_smiles,
                "source_path": record.source_path,
                "molecule_id": record.molecule_id,
            }
            for record in hits
        ],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "exact_hits.json").write_text(
        json.dumps(summary["exact_hits"], indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", action="append", required=True)
    parser.add_argument("--reference-sdf", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--scaffold-atoms", type=int, default=18)
    parser.add_argument("--top-n", type=int, default=500)
    args = parser.parse_args()
    summary = audit(
        [Path(root) for root in args.root],
        Path(args.reference_sdf),
        Path(args.output_dir),
        scaffold_atoms=args.scaffold_atoms,
        top_n=args.top_n,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
