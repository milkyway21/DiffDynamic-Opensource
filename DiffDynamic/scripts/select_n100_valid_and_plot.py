#!/usr/bin/env python3
"""从评估结果中随机抽取 n 个有效 Vina 分子，写 n100 子集 CSV/XLSX/SDF，并用新 palette 出图。

Usage:
  python3 scripts/select_n100_valid_and_plot.py \\
    --out experiments/6w63_..._preserve \\
    --protein /data/ye/protein-ligand/6W63_receptor_nowater_noligand.pdb \\
    --act /data/ye/protein-ligand/6W63/active_dock_qed_sa.csv \\
    --tag 6W63_4WI_ligandsize_preserve_n100 \\
    --prefix 6W63_preserve_n100 \\
    --seed 20260713 --n 100
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path, help="experiment root")
    ap.add_argument("--act", required=True, help="active_dock_qed_sa.csv")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--prefix", required=True, help="output SDF name prefix")
    ap.add_argument("--seed", type=int, default=20260713)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--skip_plot", action="store_true")
    args = ap.parse_args()

    out: Path = args.out
    run = out / "run"
    recon = run / "reconstructed_molecules"
    if not recon.is_dir():
        raise SystemExit(f"missing {recon}")

    xlsx_list = sorted(run.glob("evaluation_results_*.xlsx"))
    xlsx_list = [p for p in xlsx_list if "n100" not in p.name and "subset" not in p.name]
    if not xlsx_list:
        raise SystemExit(f"no evaluation_results_*.xlsx under {run}")
    xlsx = xlsx_list[-1]
    df = pd.read_excel(xlsx, sheet_name="评估结果")

    # valid Vina pool
    vina_col = "Vina_Dock_亲和力"
    id_col = "分子身份证"
    if vina_col not in df.columns or id_col not in df.columns:
        raise SystemExit(f"unexpected columns: {list(df.columns)[:20]}")

    pool = []
    for _, r in df.iterrows():
        mid = str(r[id_col]) if pd.notna(r[id_col]) else ""
        if not mid or mid.lower() in ("nan", "none"):
            continue
        vina = r[vina_col]
        if not (isinstance(vina, (int, float, np.floating)) and np.isfinite(float(vina))):
            continue
        sdf = recon / f"{mid}.sdf"
        if not sdf.exists():
            # try without extra suffix variants
            cands = list(recon.glob(f"{mid}*.sdf"))
            if not cands:
                continue
            sdf = cands[0]
        pool.append(
            {
                "mid": mid,
                "sdf": sdf,
                "vina": float(vina),
                "qed": float(r["QED评分"]) if pd.notna(r.get("QED评分")) else float("nan"),
                "sa": float(r["SA评分"]) if pd.notna(r.get("SA评分")) else float("nan"),
                "logp": float(r["logP"]) if pd.notna(r.get("logP")) else float("nan"),
                "molwt": float(r["分子量"]) if pd.notna(r.get("分子量")) else float("nan"),
                "smiles": str(r["SMILES"]) if pd.notna(r.get("SMILES")) else None,
                "row": r,
            }
        )

    if not pool:
        raise SystemExit("no valid-Vina candidates with matching SDF")

    rng = np.random.default_rng(args.seed)
    n_take = min(args.n, len(pool))
    idx = rng.choice(len(pool), size=n_take, replace=False)
    picked = [pool[i] for i in sorted(idx.tolist())]

    dst = out / "reconstructed_molecules_n100"
    if dst.exists():
        for f in dst.glob("*.sdf"):
            f.unlink()
    dst.mkdir(parents=True, exist_ok=True)

    rows = []
    subset_rows = []
    manifest_picked = []
    for i, item in enumerate(picked):
        name = f"{args.prefix}_{i:03d}"
        mol = Chem.SDMolSupplier(str(item["sdf"]), removeHs=False)[0]
        out_sdf = dst / f"{name}.sdf"
        if mol is not None:
            mol.SetProp("_Name", name)
            mol.SetProp("source_mid", item["mid"])
            mol.SetProp("source_sdf", item["sdf"].name)
            w = Chem.SDWriter(str(out_sdf))
            w.write(mol)
            w.close()
        else:
            shutil.copy2(item["sdf"], out_sdf)

        rows.append(
            {
                "smiles": item["smiles"],
                "vina_dock": item["vina"],
                "qed": item["qed"],
                "sa": item["sa"],
                "logp": item["logp"],
                "molwt": item["molwt"],
            }
        )
        subset_rows.append(item["row"])
        manifest_picked.append(
            {
                "idx": i,
                "out": out_sdf.name,
                "source": item["sdf"].name,
                "mid": item["mid"],
                "vina": item["vina"],
                "qed": item["qed"],
                "sa": item["sa"],
                "logp": item["logp"],
                "molwt": item["molwt"],
            }
        )

    gen_csv = out / "generated_dock_qed_sa_n100.csv"
    pd.DataFrame(rows).to_csv(gen_csv, index=False)
    pd.DataFrame(rows).to_csv(out / "generated_dock_qed_sa.csv", index=False)

    subset_df = pd.DataFrame(subset_rows)
    n100_xlsx = out / "evaluation_results_n100.xlsx"
    subset_df.to_excel(n100_xlsx, sheet_name="评估结果", index=False)
    # also under run/ for compatibility
    subset_df.to_excel(run / "evaluation_results_n100_subset.xlsx", sheet_name="评估结果", index=False)

    vinas = np.array([p["vina"] for p in picked], dtype=float)
    manifest = {
        "seed": args.seed,
        "mode": "uniform_random_valid_vina",
        "n_candidates": len(pool),
        "n": len(picked),
        "source_xlsx": str(xlsx),
        "stats": {
            "vina_mean": float(np.nanmean(vinas)),
            "qed_mean": float(np.nanmean([p["qed"] for p in picked])),
            "sa_mean": float(np.nanmean([p["sa"] for p in picked])),
            "logp_mean": float(np.nanmean([p["logp"] for p in picked])),
            "molwt_mean": float(np.nanmean([p["molwt"] for p in picked])),
        },
        "picked": manifest_picked,
    }
    (out / "subset_n100_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str)
    )
    print(
        f"n100: candidates={len(pool)} picked={len(picked)} "
        f"vina_mean={manifest['stats']['vina_mean']:.3f} → {dst}"
    )

    if args.skip_plot:
        return

    plot_script = Path(__file__).resolve().parent / "plot_violin_iceblue_f56e1a.py"
    plot_dir = out / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(plot_script),
        "--xlsx",
        str(n100_xlsx),
        "--gen_csv",
        str(gen_csv),
        "--act",
        args.act,
        "--out_dir",
        str(plot_dir),
        "--tag",
        args.tag,
    ]
    print("PLOT:", " ".join(cmd))
    subprocess.check_call(cmd)


if __name__ == "__main__":
    main()
