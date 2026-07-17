#!/usr/bin/env python3
"""Build n100 dock CSV for 6W63 ligandsize plots; fill missing Vina via score_only/minimize."""
from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, QED, RDConfig
from rdkit.Chem.MolStandardize import rdMolStandardize

sys.path.insert(0, "/data/ye/DiffDynamic")
sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
import sascorer  # noqa: E402

from utils.evaluation.docking_vina import VinaDockingTask  # noqa: E402

OUT = Path("/data/ye/DiffDynamic/experiments/6w63_4wi_ligandsize_normal_gen100_prudent_tbr10")
PROTEIN = "/data/ye/protein-ligand/6W63_receptor_nowater_noligand.pdb"
chooser = rdMolStandardize.LargestFragmentChooser()


def canon_mol(m):
    if m is None:
        return None
    try:
        m = chooser.choose(m)
        Chem.SanitizeMol(m)
        return Chem.MolToSmiles(m)
    except Exception:
        try:
            return Chem.MolToSmiles(m)
        except Exception:
            return None


def sa_score(m) -> float:
    # Project convention: higher better, ~0-1 scale via (10 - SA)/9
    return float((10.0 - sascorer.calculateScore(m)) / 9.0)


def props_from_mol(mol):
    mh = Chem.RemoveHs(mol)
    Chem.SanitizeMol(mh)
    return {
        "smiles": Chem.MolToSmiles(mh),
        "qed": float(QED.qed(mh)),
        "sa": sa_score(mh),
        "logp": float(Crippen.MolLogP(mh)),
        "molwt": float(Descriptors.MolWt(mh)),
    }


def dock_affinity(mol):
    mol_h = Chem.AddHs(mol, addCoords=True)
    task = VinaDockingTask(protein_path=PROTEIN, ligand_rdmol=mol_h)
    for mode in ("score_only", "minimize"):
        try:
            res = task.run(mode=mode)
            if res:
                aff = float(res[0]["affinity"])
                if np.isfinite(aff):
                    return aff
        except Exception:
            continue
    return None


def main():
    xlsx = sorted(OUT.glob("run/evaluation_results_*.xlsx"))[-1]
    df = pd.read_excel(xlsx, sheet_name="评估结果")
    full = pd.DataFrame(
        {
            "smiles": df["SMILES"],
            "vina_dock": df["Vina_Dock_亲和力"],
            "qed": df["QED评分"],
            "sa": df["SA评分"],
            "logp": df["logP"],
            "molwt": df["分子量"],
        }
    )
    full.to_csv(OUT / "generated_dock_qed_sa_full.csv", index=False)
    print(f"full CSV: {len(full)}")

    lookup = {}
    for _, r in full.iterrows():
        m = Chem.MolFromSmiles(str(r["smiles"])) if pd.notna(r["smiles"]) else None
        c = canon_mol(m)
        if c and c not in lookup:
            lookup[c] = r

    pairs = pd.read_csv(OUT / "pairs_manifest.csv")
    pairs_by_final = {}
    for _, r in pairs.iterrows():
        name = Path(str(r["final_sdf"])).name
        pairs_by_final[name] = r

    manifest = json.loads((OUT / "subset_n100_manifest.json").read_text())
    rows = []
    need_dock = []
    for item in manifest["picked"]:
        src = item["source"]
        sdf_path = OUT / "reconstructed_molecules_n100" / item["out"]
        mol = Chem.SDMolSupplier(str(sdf_path), removeHs=False)[0]
        row = {
            "smiles": None,
            "vina_dock": None,
            "qed": None,
            "sa": None,
            "logp": None,
            "molwt": None,
            "sdf": item["out"],
            "source": src,
        }
        if mol is not None:
            try:
                row.update(props_from_mol(mol))
            except Exception as e:
                print(f"props fail {item['out']}: {e}")
        # pairs_manifest first
        pr = pairs_by_final.get(src)
        if pr is not None and pd.notna(pr.get("vina_dock")):
            row["vina_dock"] = float(pr["vina_dock"])
            if pd.notna(pr.get("qed")):
                row["qed"] = float(pr["qed"])
            if pd.notna(pr.get("sa")):
                row["sa"] = float(pr["sa"])
            if pd.notna(pr.get("logp")):
                row["logp"] = float(pr["logp"])
            if pd.notna(pr.get("smiles")):
                row["smiles"] = pr["smiles"]
        # excel lookup
        c = canon_mol(Chem.MolFromSmiles(row["smiles"])) if row.get("smiles") else canon_mol(mol)
        if c and c in lookup:
            er = lookup[c]
            if row["vina_dock"] is None or (isinstance(row["vina_dock"], float) and np.isnan(row["vina_dock"])):
                row["vina_dock"] = er["vina_dock"]
            for k in ("qed", "sa", "logp", "molwt"):
                if row.get(k) is None or (isinstance(row.get(k), float) and np.isnan(row.get(k))):
                    row[k] = er[k]
            row["smiles"] = c
        rows.append(row)
        v = row["vina_dock"]
        if v is None or (isinstance(v, float) and np.isnan(v)):
            need_dock.append((len(rows) - 1, mol))

    print(f"n100={len(rows)} need_dock={len(need_dock)}")

    def _job(args):
        idx, mol = args
        if mol is None:
            return idx, None
        try:
            return idx, dock_affinity(mol)
        except Exception as e:
            print(f"dock fail idx={idx}: {e}")
            return idx, None

    done = 0
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(_job, a) for a in need_dock]
        for fut in as_completed(futs):
            idx, aff = fut.result()
            rows[idx]["vina_dock"] = aff
            done += 1
            with open(OUT / "build_n100_csv_progress.txt", "a") as pf:
                pf.write(f"{done}/{len(need_dock)} {rows[idx]['sdf']} vina={aff}\n")

    out = pd.DataFrame(
        [{k: r[k] for k in ["smiles", "vina_dock", "qed", "sa", "logp", "molwt"]} for r in rows]
    )
    out.to_csv(OUT / "generated_dock_qed_sa_n100.csv", index=False)
    out.to_csv(OUT / "generated_dock_qed_sa.csv", index=False)
    summary = out.describe().to_string() + f"\nvina_ok={int(out.vina_dock.notna().sum())}\nDONE\n"
    (OUT / "build_n100_csv_summary.txt").write_text(summary)
    print(summary, flush=True)


if __name__ == "__main__":
    main()
