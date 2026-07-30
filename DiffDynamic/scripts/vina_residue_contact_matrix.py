#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从已有 Vina 评估结果 + reconstructed_molecules SDF + 蛋白 PDB，
统计每个分子与蛋白残基的最短重原子距离，写出 CSV：

  每行 = 一个有 Vina_Dock 分数的分子
  每列 = 一个接触残基（全数据集中至少有一次距离 ≤ cutoff）
  单元格 = 该分子到该残基的最短重原子距离（Å）；无接触则为空

说明：评估流水线未保存对接后 pose，仅保存了亲和力；
因此距离基于 reconstructed_molecules 中的构象（对接所用输入姿态）。
若需对接后姿态，需重新跑 Vina 并保存 pose。

示例：
  conda run -n diffdynamic python3 scripts/vina_residue_contact_matrix.py \\
    --pocket 8ri2 --cutoff 5.0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.PDB import PDBParser, Selection
from rdkit import Chem

REPO = Path(__file__).resolve().parents[1]

POCKETS = {
    "8ri2": {
        "root": REPO / "experiments/8ri2_scaffold_dl_gen1000_gpu0_20260719b",
        "protein": Path("/data/ye/protein-ligand/8RI2/8RI2_apo.pdb"),
        "tag": "8RI2",
    },
    "6w63": {
        "root": REPO / "experiments/6w63_scaffold_dl_gen1000_gpu5_20260719b",
        "protein": Path("/data/ye/protein-ligand/6W63_receptor_nowater_noligand.pdb"),
        "tag": "6W63",
    },
}


def load_residue_heavy_coords(pdb_path: Path):
    """Return list of (res_key, coords Nx3). res_key = CHAIN_RESnum_RESNAME."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("prot", str(pdb_path))
    residues = []
    for model in structure:
        for chain in model:
            cid = chain.id.strip() or "_"
            for res in chain:
                hetflag, resseq, icode = res.id
                if hetflag != " ":
                    continue  # skip hetero / water
                name = res.get_resname().strip()
                atoms = []
                for atom in res:
                    if atom.element == "H":
                        continue
                    atoms.append(atom.coord)
                if not atoms:
                    continue
                key = f"{cid}_{resseq}{icode.strip()}_{name}"
                residues.append((key, np.asarray(atoms, dtype=np.float64)))
        break
    return residues


def ligand_heavy_coords(mol) -> np.ndarray | None:
    if mol is None or mol.GetNumConformers() == 0:
        return None
    conf = mol.GetConformer()
    coords = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1:
            continue
        p = conf.GetAtomPosition(atom.GetIdx())
        coords.append([p.x, p.y, p.z])
    if not coords:
        return None
    return np.asarray(coords, dtype=np.float64)


def min_residue_distances(lig_xyz: np.ndarray, residues) -> dict:
    """res_key -> min heavy-atom distance (Å)."""
    out = {}
    for key, rxyz in residues:
        # (n_lig, n_res) distances
        d = np.linalg.norm(lig_xyz[:, None, :] - rxyz[None, :, :], axis=-1)
        out[key] = float(d.min())
    return out


def find_eval_xlsx(root: Path) -> Path:
    xs = sorted((root / "run").glob("evaluation_results_*.xlsx"))
    xs = [x for x in xs if "n100" not in x.name and "subset" not in x.name]
    if not xs:
        raise FileNotFoundError(f"no evaluation_results_*.xlsx under {root}/run")
    return xs[-1]


def process_pocket(
    pocket_key: str,
    cutoff: float,
    out_path: Path | None = None,
) -> Path:
    cfg = POCKETS[pocket_key]
    root: Path = cfg["root"]
    protein: Path = cfg["protein"]
    if not root.is_dir():
        raise FileNotFoundError(root)
    if not protein.is_file():
        raise FileNotFoundError(protein)

    xlsx = find_eval_xlsx(root)
    df = pd.read_excel(xlsx, sheet_name="评估结果")
    df["Vina_Dock_亲和力"] = pd.to_numeric(df["Vina_Dock_亲和力"], errors="coerce")
    df = df[df["Vina_Dock_亲和力"].notna()].copy()
    sdf_dir = root / "run" / "reconstructed_molecules"
    if not sdf_dir.is_dir():
        raise FileNotFoundError(sdf_dir)

    print(f"[{pocket_key}] protein={protein}")
    print(f"[{pocket_key}] xlsx={xlsx.name} vina_mols={len(df)}")
    print(f"[{pocket_key}] loading residues...")
    residues = load_residue_heavy_coords(protein)
    print(f"[{pocket_key}] residues={len(residues)}")

    rows = []
    contact_counts = {}
    n_ok = n_miss_sdf = n_fail_mol = 0
    for _, r in df.iterrows():
        mid = str(r["分子身份证"])
        sdf = sdf_dir / f"{mid}.sdf"
        if not sdf.is_file():
            cands = list(sdf_dir.glob(f"{mid}*.sdf"))
            sdf = cands[0] if cands else None
        if sdf is None or not sdf.is_file():
            n_miss_sdf += 1
            continue
        mol = Chem.SDMolSupplier(str(sdf), removeHs=False)[0]
        lig = ligand_heavy_coords(mol)
        if lig is None:
            n_fail_mol += 1
            continue
        dists = min_residue_distances(lig, residues)
        # contacts within cutoff
        contacts = {k: v for k, v in dists.items() if v <= cutoff}
        for k in contacts:
            contact_counts[k] = contact_counts.get(k, 0) + 1
        rows.append(
            {
                "pocket": cfg["tag"],
                "molecule_id": mid,
                "smiles": r.get("SMILES"),
                "vina_dock": float(r["Vina_Dock_亲和力"]),
                "_contacts": contacts,
            }
        )
        n_ok += 1
        if n_ok % 200 == 0:
            print(f"[{pocket_key}] processed {n_ok}/{len(df)}")

    # columns = residues that contact ≥1 molecule
    res_cols = sorted(contact_counts.keys(), key=lambda k: (-contact_counts[k], k))
    print(
        f"[{pocket_key}] ok={n_ok} miss_sdf={n_miss_sdf} fail_mol={n_fail_mol} "
        f"contact_residues(≤{cutoff}Å)={len(res_cols)}"
    )

    out_rows = []
    for item in rows:
        row = {
            "pocket": item["pocket"],
            "molecule_id": item["molecule_id"],
            "smiles": item["smiles"],
            "vina_dock": item["vina_dock"],
        }
        for rc in res_cols:
            row[rc] = item["_contacts"].get(rc, np.nan)
        out_rows.append(row)

    out_df = pd.DataFrame(out_rows)
    if out_path is None:
        out_path = root / f"vina_residue_contacts_cutoff{cutoff:.1f}A.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False, float_format="%.3f")
    print(f"[{pocket_key}] saved {out_path} shape={out_df.shape}")

    # also write residue frequency summary
    freq_path = out_path.with_name(out_path.stem + "_residue_freq.csv")
    freq = pd.DataFrame(
        [{"residue": k, "n_molecules_contact": contact_counts[k]} for k in res_cols]
    )
    freq.to_csv(freq_path, index=False)
    print(f"[{pocket_key}] saved {freq_path}")
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--pocket",
        choices=["8ri2", "6w63", "both"],
        default="both",
    )
    ap.add_argument(
        "--cutoff",
        type=float,
        default=5.0,
        help="接触距离阈值（Å）；仅 ≤cutoff 的残基作为列，单元格填最短距离",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="可选：统一输出目录（默认写到各 experiment 根目录）",
    )
    args = ap.parse_args()

    pockets = ["8ri2", "6w63"] if args.pocket == "both" else [args.pocket]
    paths = []
    for pk in pockets:
        out = None
        if args.out_dir is not None:
            args.out_dir.mkdir(parents=True, exist_ok=True)
            out = args.out_dir / f"{pk}_vina_residue_contacts_cutoff{args.cutoff:.1f}A.csv"
        paths.append(process_pocket(pk, args.cutoff, out_path=out))

    # combined optional
    if len(paths) == 2 and args.out_dir is not None:
        dfs = [pd.read_csv(p) for p in paths]
        # union of residue columns
        meta = ["pocket", "molecule_id", "smiles", "vina_dock"]
        res_cols = sorted({c for df in dfs for c in df.columns if c not in meta})
        aligned = []
        for df in dfs:
            for c in res_cols:
                if c not in df.columns:
                    df[c] = np.nan
            aligned.append(df[meta + res_cols])
        comb = pd.concat(aligned, ignore_index=True)
        comb_path = args.out_dir / f"both_vina_residue_contacts_cutoff{args.cutoff:.1f}A.csv"
        comb.to_csv(comb_path, index=False, float_format="%.3f")
        print(f"[both] saved {comb_path} shape={comb.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
