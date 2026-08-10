#!/usr/bin/env python3
"""Dedup generated molglue SDFs by canonical SMILES, then compare against reference SDF(s).

Performs two types of comparison:
  1. Exact match: canonical SMILES identical to a reference molecule
  2. Tanimoto similarity > threshold (default 0.7) using Morgan fingerprints

Outputs:
  reference_smiles.txt    - one canonical SMILES per line from all reference SDFs
  unique_ikzf2.csv        - IKZF2 first-wins unique molecules
  unique_gspt1.csv        - GSPT1 first-wins unique molecules
  unique_merged.csv       - global merged unique (ikzf2 priority)
  matches_vs_ref.csv      - exact SMILES matches
  similar_vs_ref.csv      - Tanimoto > threshold (best match per gen mol)
  summary.json            - counts
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")


def canonical_smiles(mol: Chem.Mol, *, isomeric: bool = False) -> Optional[str]:
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=isomeric)
    except Exception:
        return None


def load_reference(ref_sd_fs: List[Path]) -> List[Tuple[str, Chem.Mol]]:
    """Robustly parse multi-mol SDFs; return list of (canonical_smiles, mol) pairs."""
    result: List[Tuple[str, Chem.Mol]] = []
    seen: Set[str] = set()

    for ref_sdf in ref_sd_fs:
        raw = ref_sdf.read_bytes().decode("utf-8", errors="ignore")
        text = raw.replace("\r\n", "\n").replace("\r", "\n")
        # Try SDMolSupplier first
        mols_from_supplier: List[Chem.Mol] = []
        try:
            suppl = Chem.SDMolSupplier(str(ref_sdf), removeHs=False, sanitize=False)
            for mol in suppl:
                if mol is None:
                    continue
                for ops in (
                    Chem.SanitizeFlags.SANITIZE_ALL,
                    Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES,
                ):
                    try:
                        Chem.SanitizeMol(mol, sanitizeOps=ops)
                        break
                    except Exception:
                        continue
                mols_from_supplier.append(mol)
        except Exception:
            pass

        # If supplier yielded too few, fall back to manual block splitting
        if len(mols_from_supplier) < 2:
            mols_from_supplier = []
            blocks = [b for b in text.split("$$$$\n") if b.strip()]
            for blk in blocks:
                blk = blk.strip()
                if not blk:
                    continue
                mol = None
                try:
                    mol = Chem.MolFromMolBlock(blk, sanitize=True)
                except Exception:
                    mol = None
                if mol is None:
                    try:
                        mol = Chem.MolFromMolBlock(blk, sanitize=False)
                    except Exception:
                        mol = None
                    if mol is not None:
                        for ops in (
                            Chem.SanitizeFlags.SANITIZE_ALL,
                            Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES,
                        ):
                            try:
                                Chem.SanitizeMol(mol, sanitizeOps=ops)
                                break
                            except Exception:
                                continue
                if mol is not None:
                    mols_from_supplier.append(mol)

        for mol in mols_from_supplier:
            s = canonical_smiles(mol)
            if s and s not in seen:
                seen.add(s)
                result.append((s, mol))

    return result


def _process_sdf(args: Tuple[str, Path]) -> List[Dict]:
    """Read one SDF file; return list of molecule records."""
    target, sdf_path = args
    records: List[Dict] = []
    try:
        suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=False)
        for mol in suppl:
            if mol is None:
                continue
            try:
                Chem.SanitizeMol(mol)
            except Exception:
                continue
            s = canonical_smiles(mol)
            if not s:
                continue
            mid = None
            for prop in ("Molecule_ID", "molecule_id", "MoleculeID", "_Name"):
                if mol.HasProp(prop):
                    mid = mol.GetProp(prop)
                    break
            if mid is None:
                mid = mol.GetProp("_Name") if mol.HasProp("_Name") else sdf_path.stem
            records.append(
                {
                    "canonical_smiles": s,
                    "target": target,
                    "sdf_path": str(sdf_path),
                    "molecule_id": mid,
                }
            )
    except Exception as e:
        print(f"[WARN] failed to read {sdf_path}: {e}", file=sys.stderr)
    return records


def gather_sdf_files(root: Path, target: str) -> List[Path]:
    """Gather reconstructed SDFs from current and legacy extraction layouts."""
    patterns = (
        f"{target}/extract_cleaned/job_*/eval_*/reconstructed_molecules/*.sdf",
        f"{target}/extract_novina_v2/job_*/eval_*/reconstructed_molecules/*.sdf",
    )
    files = {path for pattern in patterns for path in root.glob(pattern)}
    return sorted(files)


def _job_id_from_path(sdf_path: str) -> str:
    for p in Path(sdf_path).parts:
        if p.startswith("job_"):
            return p
    return ""


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=str, required=True, help="campaign root")
    ap.add_argument(
        "--ref-sdf",
        type=str,
        required=True,
        nargs="+",
        help="one or more reference multi-mol SDF files",
    )
    ap.add_argument("--out-dir", type=str, required=True, help="output directory")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--tanimoto-threshold", type=float, default=0.7)
    args = ap.parse_args()

    root = Path(args.root)
    ref_paths = [Path(p) for p in args.ref_sdf]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. Load reference ----
    print(f"[1/5] Loading {len(ref_paths)} reference SDF(s) ...")
    for rp in ref_paths:
        print(f"      {rp}")
    ref_data = load_reference(ref_paths)
    ref_smiles: Set[str] = {s for s, _ in ref_data}
    print(f"      reference canonical SMILES count = {len(ref_smiles)}")

    ref_txt = out_dir / "reference_smiles.txt"
    with open(ref_txt, "w") as f:
        for s, _ in ref_data:
            f.write(s + "\n")
    print(f"      saved -> {ref_txt}")

    # Pre-compute reference Morgan fingerprints
    ref_fps: List[Tuple[str, object]] = []
    for s, mol in ref_data:
        try:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
            ref_fps.append((s, fp))
        except Exception:
            continue
    print(f"      reference fingerprints computed = {len(ref_fps)}")

    # ---- 2. Gather + read generated SDFs ----
    all_records: List[Dict] = []
    for target in ("ikzf2", "gspt1"):
        files = gather_sdf_files(root, target)
        print(f"[2/5] {target}: {len(files)} SDF files")
        if not files:
            continue
        tasks = [(target, f) for f in files]
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for i, result in enumerate(ex.map(_process_sdf, tasks, chunksize=32)):
                all_records.extend(result)
                if (i + 1) % 500 == 0:
                    print(f"      {target}: processed {i+1}/{len(files)} files, records={len(all_records)}")
    print(f"      total records (all molecules) = {len(all_records)}")

    # ---- 3. Dedup per-target + global ----
    print("[3/5] Deduplicating ...")
    unique_ik: Dict[str, Dict] = {}
    unique_gs: Dict[str, Dict] = {}
    for rec in all_records:
        s = rec["canonical_smiles"]
        t = rec["target"]
        if t == "ikzf2":
            if s not in unique_ik:
                unique_ik[s] = rec
        else:
            if s not in unique_gs:
                unique_gs[s] = rec

    unique_merged: Dict[str, Dict] = {}
    for s, rec in unique_ik.items():
        unique_merged[s] = rec
    for s, rec in unique_gs.items():
        if s not in unique_merged:
            unique_merged[s] = rec

    print(f"      unique IKZF2 = {len(unique_ik)}")
    print(f"      unique GSPT1 = {len(unique_gs)}")
    print(f"      unique merged = {len(unique_merged)}")

    def write_csv(path: Path, records: List[Dict], extra_cols: Optional[List[str]] = None):
        cols = ["molecule_id", "target", "job_id", "canonical_smiles", "sdf_path"]
        if extra_cols:
            cols.extend(extra_cols)
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in records:
                row = {k: r.get(k, "") for k in cols}
                w.writerow(row)

    # Write per-target unique
    ik_records = []
    for r in unique_ik.values():
        r2 = dict(r)
        r2["job_id"] = _job_id_from_path(r2["sdf_path"])
        ik_records.append(r2)
    gs_records = []
    for r in unique_gs.values():
        r2 = dict(r)
        r2["job_id"] = _job_id_from_path(r2["sdf_path"])
        gs_records.append(r2)
    write_csv(out_dir / "unique_ikzf2.csv", ik_records)
    write_csv(out_dir / "unique_gspt1.csv", gs_records)

    # ---- 4. Exact SMILES match vs reference ----
    print("[4/5] Exact SMILES matching vs reference ...")
    merged_records: List[Dict] = []
    exact_matches: List[Dict] = []
    for s, rec in unique_merged.items():
        r2 = dict(rec)
        r2["job_id"] = _job_id_from_path(r2["sdf_path"])
        r2["in_reference"] = s in ref_smiles
        merged_records.append(r2)
        if r2["in_reference"]:
            exact_matches.append(r2)

    write_csv(out_dir / "unique_merged.csv", merged_records, extra_cols=["in_reference"])
    write_csv(out_dir / "matches_vs_ref.csv", exact_matches, extra_cols=["in_reference"])

    # ---- 5. Tanimoto similarity > threshold ----
    print(f"[5/5] Tanimoto similarity > {args.tanimoto_threshold} ...")
    similar_records: List[Dict] = []
    n_checked = 0
    for s, rec in unique_merged.items():
        n_checked += 1
        if n_checked % 2000 == 0:
            print(f"      checked {n_checked}/{len(unique_merged)} ...")
        # Skip exact matches (already reported)
        if s in ref_smiles:
            continue
        try:
            mol = Chem.MolFromSmiles(s)
        except Exception:
            mol = None
        if mol is None:
            continue
        try:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        except Exception:
            continue

        best_sim = 0.0
        best_ref_smiles = ""
        for ref_s, ref_fp in ref_fps:
            try:
                sim = DataStructs.TanimotoSimilarity(fp, ref_fp)
            except Exception:
                continue
            if sim > best_sim:
                best_sim = sim
                best_ref_smiles = ref_s
            if best_sim >= 1.0:
                break

        if best_sim > args.tanimoto_threshold:
            r2 = dict(rec)
            r2["job_id"] = _job_id_from_path(r2["sdf_path"])
            r2["tanimoto"] = f"{best_sim:.4f}"
            r2["best_ref_smiles"] = best_ref_smiles
            similar_records.append(r2)

    # Sort by Tanimoto descending
    similar_records.sort(key=lambda x: float(x["tanimoto"]), reverse=True)
    write_csv(
        out_dir / "similar_vs_ref.csv",
        similar_records,
        extra_cols=["tanimoto", "best_ref_smiles"],
    )

    summary = {
        "total_sdf_read": len(all_records),
        "unique_ikzf2": len(unique_ik),
        "unique_gspt1": len(unique_gs),
        "unique_merged": len(unique_merged),
        "reference_smiles": len(ref_smiles),
        "exact_matches": len(exact_matches),
        "tanimoto_threshold": args.tanimoto_threshold,
        "similar_matches": len(similar_records),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"\nDone. Outputs in: {out_dir}")
    print(f"Exact matches: {out_dir / 'matches_vs_ref.csv'} ({len(exact_matches)} rows)")
    print(f"Similar (Tanimoto>{args.tanimoto_threshold}): {out_dir / 'similar_vs_ref.csv'} ({len(similar_records)} rows)")


if __name__ == "__main__":
    main()
