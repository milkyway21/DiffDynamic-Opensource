#!/usr/bin/env python3
"""Redraw pocket violin plots with ice-blue actives + F56E1A generated palette.

Palette (final paper style):
  Actives fill/line : #C5DDF0 / #5A7FA0
  Generated fill    : #FBC9A6  (pale orange background)
  Generated points  : #F56E1A  (and center line = mean)

Center line uses mean (not median), via plot_property_violins._violin_with_style.

Usage:
  conda activate diffdynamic
  python3 scripts/plot_violin_iceblue_f56e1a.py \\
    --xlsx experiments/.../evaluation_results_n100.xlsx \\
    --gen_csv experiments/.../generated_dock_qed_sa_n100.csv \\
    --act /data/ye/protein-ligand/6W63/active_dock_qed_sa.csv \\
    --out_dir experiments/.../plots \\
    --tag 6W63_4WI_ligandsize_n100
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, "/data/ye/protein-ligand")

import plot_generic_violin as pgv
import plot_generic_violin_expanded as pev
import plot_property_violins as ppv

# Final palette from earlier paper-demo iteration
PALETTE = {
    "actual": "#C5DDF0",
    "actual_center": "#5A7FA0",
    "generated_fill": "#FBC9A6",
    "generated_points": "#F56E1A",
}


def apply_palette() -> None:
    for mod in (ppv, pgv, pev):
        mod.COLOR_ACTUAL = PALETTE["actual"]
        mod.COLOR_ACTUAL_MEDIAN = PALETTE["actual_center"]
        mod.COLOR_GENERATED = PALETTE["generated_fill"]
        mod.COLOR_GENERATED_MEDIAN = PALETTE["generated_points"]


def enrich_xlsx_from_smiles(xlsx_path: Path, out_xlsx: Path) -> Path:
    """Ensure expanded panels have TPSA / atom count / Lipinski / Lilly from SMILES."""
    from rdkit import Chem

    df = pd.read_excel(xlsx_path, sheet_name="评估结果")
    rd = pev.compute_rdkit_props(df["SMILES"])
    if "TPSA" not in df.columns or df["TPSA"].isna().all():
        df["TPSA"] = rd["tpsa"]
    if "原子数" not in df.columns or df["原子数"].isna().all():
        df["原子数"] = rd["num_atoms"]
    if "环数" not in df.columns or df["环数"].isna().all():
        df["环数"] = rd["ring_count"]
    if "Lipinski规则得分" not in df.columns or df["Lipinski规则得分"].isna().all():
        df["Lipinski规则得分"] = rd["lipinski"]
    if "Lilly_Medchem_扣分" not in df.columns or df["Lilly_Medchem_扣分"].isna().all():
        try:
            from utils.evaluation.lilly_medchem_rules import evaluate_lilly_medchem_rules
        except ImportError:
            sys.path.insert(0, str(ROOT))
            from utils.evaluation.lilly_medchem_rules import evaluate_lilly_medchem_rules
        demerits = []
        for smi in df["SMILES"]:
            mol = Chem.MolFromSmiles(str(smi)) if pd.notna(smi) else None
            if mol is None:
                demerits.append(float("nan"))
                continue
            lr = evaluate_lilly_medchem_rules(mol, debug=False) or {}
            demerits.append(float(lr.get("demerit", 0) or 0))
        df["Lilly_Medchem_扣分"] = demerits
    out_xlsx.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(out_xlsx, sheet_name="评估结果", index=False)
    return out_xlsx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", required=True, help="evaluation_results xlsx (评估结果)")
    ap.add_argument("--gen_csv", required=True, help="generated_dock_qed_sa.csv for 3-panel")
    ap.add_argument("--act", required=True, help="active_dock_qed_sa.csv")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    apply_palette()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag

    xlsx = enrich_xlsx_from_smiles(Path(args.xlsx), out_dir.parent / "evaluation_results_n100_enriched.xlsx")

    gen_csv = pd.read_csv(args.gen_csv)
    act = pev.load_actives(args.act)

    print(f"Palette: actives {PALETTE['actual']}/{PALETTE['actual_center']} | "
          f"gen fill {PALETTE['generated_fill']} points {PALETTE['generated_points']}")
    print("Center line: MEAN")

    # 3-panel Vina/QED/SA
    pgv.plot_violins(gen_csv, act, out_dir, tag, show_text=True)
    pgv.plot_violins(gen_csv, act, out_dir, tag, show_text=False)
    # also save explicit F56E1A aliases
    for src_suffix, dst_suffix in [
        ("_violin.png", "_violin_final_F56E1A.png"),
        ("_violin_no_text.png", "_violin_final_F56E1A_no_text.png"),
    ]:
        src = out_dir / f"{tag}{src_suffix}"
        dst = out_dir / f"{tag}{dst_suffix}"
        if src.exists():
            dst.write_bytes(src.read_bytes())

    # 12-panel expanded
    gen_x = pev.load_generated(xlsx)
    pev.plot_expanded(gen_x, act, out_dir, tag, show_text=True)
    pev.plot_expanded(gen_x, act, out_dir, tag, show_text=False)
    for src_suffix, dst_suffix in [
        ("_violin_expanded.png", "_final_F56E1A_violin_expanded.png"),
        ("_violin_expanded_no_text.png", "_final_F56E1A_violin_expanded_no_text.png"),
        ("_violin_expanded.png", "_violin_final_iceblue_paleorange_expanded.png"),
        ("_violin_expanded_no_text.png", "_violin_final_iceblue_paleorange_expanded_no_text.png"),
    ]:
        src = out_dir / f"{tag}{src_suffix}"
        dst = out_dir / f"{tag}{dst_suffix}"
        if src.exists():
            dst.write_bytes(src.read_bytes())

    print("DONE", out_dir)


if __name__ == "__main__":
    main()
