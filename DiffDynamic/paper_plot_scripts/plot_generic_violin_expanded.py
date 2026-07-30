#!/usr/bin/env python3
"""Expanded 12-panel violin plot: Vina Dock / QED / SA / LogP / MW / TPSA /
HBA / HBD / Atom Count / Rotatable Bonds / Lipinski / Lilly Demerits.

Fixed y-axes: Vina -15~0 (inverted), QED 0-1, SA 0-1, Lipinski 0-5.
All generated data from xlsx; active data computed from SMILES via RDKit.
Vina filter (-15, 0) applied to both.

Usage:
  python3 plot_generic_violin_expanded.py --xlsx GEN_XLSX --act ACTIVE_CSV --out_dir OUT_DIR --tag TAG
"""
import sys, warnings, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# Reuse styling from existing violin module
sys.path.insert(0, str(Path(__file__).parent))
from plot_property_violins import (
    _violin_with_style, COLOR_ACTUAL, COLOR_GENERATED,
    COLOR_ACTUAL_MEDIAN, COLOR_GENERATED_MEDIAN, COLOR_VIOLIN_EDGE,
    COLOR_VIOLIN_SPINE, MEDIAN_LINEWIDTH,
)


def compute_rdkit_props(smiles_series):
    """Compute RDKit molecular properties from a Series of SMILES.
    Returns a DataFrame with columns: hba, hbd, rot_bonds, logp, molwt, tpsa,
    num_atoms, ring_count, lipinski.
    """
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Lipinski, Crippen

    results = {
        "hba": [], "hbd": [], "rot_bonds": [], "logp": [],
        "molwt": [], "tpsa": [], "num_atoms": [], "ring_count": [], "lipinski": [],
    }
    for smi in smiles_series:
        mol = Chem.MolFromSmiles(str(smi)) if pd.notna(smi) else None
        if mol is not None:
            results["hba"].append(Lipinski.NumHAcceptors(mol))
            results["hbd"].append(Lipinski.NumHDonors(mol))
            results["rot_bonds"].append(Lipinski.NumRotatableBonds(mol))
            results["logp"].append(Crippen.MolLogP(mol))
            results["molwt"].append(Descriptors.MolWt(mol))
            results["tpsa"].append(Descriptors.TPSA(mol))
            results["num_atoms"].append(mol.GetNumHeavyAtoms())
            results["ring_count"].append(Lipinski.RingCount(mol))
            # Lipinski rule of 5 violations (MolWt>500 count=1, LogP>5, HBD>5, HBA>10)
            lip = 4
            if Descriptors.MolWt(mol) > 500:
                lip -= 1
            if Crippen.MolLogP(mol) > 5:
                lip -= 1
            if Lipinski.NumHDonors(mol) > 5:
                lip -= 1
            if Lipinski.NumHAcceptors(mol) > 10:
                lip -= 1
            results["lipinski"].append(max(lip, 0))
        else:
            for k in results:
                results[k].append(np.nan)
    return pd.DataFrame(results)


def load_generated(xlsx_path):
    """Load generated molecule data from evaluation xlsx.
    Columns used: SMILES, Vina_Dock_亲和力, QED评分, SA评分, logP, 分子量, TPSA,
                  原子数, 环数, Lipinski规则得分, Lilly_Medchem_扣分
    RDKit-computed additions: HBA, HBD, Rotatable Bonds.
    """
    xlsx = Path(xlsx_path)
    if not xlsx.exists():
        raise FileNotFoundError(f"xlsx not found: {xlsx_path}")

    df = pd.read_excel(xlsx, sheet_name="评估结果")

    # Map Chinese column names to English
    col_map = {
        "SMILES": "smiles",
        "Vina_Dock_亲和力": "vina_dock",
        "QED评分": "qed",
        "SA评分": "sa",
        "logP": "logp",
        "分子量": "molwt",
        "TPSA": "tpsa",
        "原子数": "num_atoms",
        "环数": "ring_count",
        "Lipinski规则得分": "lipinski",
        "Lilly_Medchem_扣分": "lilly_demerits",
    }
    out = pd.DataFrame()
    for cn, en in col_map.items():
        if cn in df.columns:
            out[en] = pd.to_numeric(df[cn], errors="coerce") if en != "smiles" else df[cn].astype(str)

    # Compute RDKit properties for HBA/HBD/RotBonds (not in xlsx)
    rdkit_df = compute_rdkit_props(out["smiles"])
    out["hba"] = rdkit_df["hba"]
    out["hbd"] = rdkit_df["hbd"]
    out["rot_bonds"] = rdkit_df["rot_bonds"]

    # Use RDKit ring_count and lipinski as fallback if xlsx columns missing
    if "ring_count" not in out.columns or out["ring_count"].isna().all():
        out["ring_count"] = rdkit_df["ring_count"]
    if "lipinski" not in out.columns or out["lipinski"].isna().all():
        out["lipinski"] = rdkit_df["lipinski"]

    return out


def load_actives(csv_path):
    """Load active molecules from CSV (name,smiles,qed,sa,vina_dock).
    Compute all additional properties from SMILES via RDKit.
    Lilly demerits: use CSV column if present, else DiffDynamic Lilly Medchem Rules.
    """
    df = pd.read_csv(csv_path)
    out = pd.DataFrame({
        "smiles": df["smiles"].astype(str),
        "qed": pd.to_numeric(df["qed"], errors="coerce"),
        "sa": pd.to_numeric(df["sa"], errors="coerce"),
        "vina_dock": pd.to_numeric(df["vina_dock"], errors="coerce"),
    })
    # All other properties from RDKit
    rdkit_df = compute_rdkit_props(out["smiles"])
    for col in ["hba", "hbd", "rot_bonds", "logp", "molwt", "tpsa",
                "num_atoms", "ring_count", "lipinski"]:
        out[col] = rdkit_df[col]
    if "lilly_demerit" in df.columns:
        out["lilly_demerits"] = pd.to_numeric(df["lilly_demerit"], errors="coerce")
    else:
        # Same evaluator as DiffDynamic generation/eval pipeline
        try:
            from utils.evaluation.lilly_medchem_rules import evaluate_lilly_medchem_rules
        except ImportError:
            sys.path.insert(0, "/data/ye/DiffDynamic")
            from utils.evaluation.lilly_medchem_rules import evaluate_lilly_medchem_rules
        demerits = []
        for smi in out["smiles"]:
            mol = None
            try:
                from rdkit import Chem as _Chem
                mol = _Chem.MolFromSmiles(str(smi)) if pd.notna(smi) else None
            except Exception:
                mol = None
            if mol is None:
                demerits.append(np.nan)
                continue
            lr = evaluate_lilly_medchem_rules(mol, debug=False) or {}
            demerits.append(float(lr.get("demerit", 0) or 0))
        out["lilly_demerits"] = demerits
    return out


def plot_expanded(gen_df, active_df, out_dir, tag, show_text=True):
    """Generate 12-panel (4x3) violin plot with fixed y-axes where meaningful."""
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 10, "axes.labelsize": 11, "axes.titlesize": 12,
        "figure.dpi": 300, "savefig.dpi": 300, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.labelsize": 9, "ytick.labelsize": 8,
    })

    # Filter: Vina in (-15, 0)
    gen_filt = gen_df[(gen_df["vina_dock"].notna()) &
                       (gen_df["vina_dock"] < 0) &
                       (gen_df["vina_dock"] > -15)].copy()
    act_filt = active_df[(active_df["vina_dock"].notna()) &
                          (active_df["vina_dock"] < 0) &
                          (active_df["vina_dock"] > -15)].copy()
    excluded = len(gen_df) - len(gen_filt)
    act_excluded = len(active_df) - len(act_filt)
    print(f"Generated: total={len(gen_df)}, filtered(Vina -15~0)={len(gen_filt)}, excluded={excluded}")
    print(f"Known actives: total={len(active_df)}, filtered={len(act_filt)}, excluded={act_excluded}")

    # Panel definitions: (col_name, display_title, y_label, (y_lo, y_hi), invert)
    # Use None for y_lo/hi to auto-scale; pass a fixed tuple to lock axes.
    metrics = [
        # Row 1 — Docking Quality
        ("vina_dock",   "Vina Dock",        "Vina Dock (kcal/mol)", (-15, 0),    True),
        ("qed",         "QED",              "QED",                  (0, 1),      False),
        ("sa",          "SA Score",         "SA Score",             (0, 1),      False),
        # Row 2 — Physicochemical
        ("logp",        "LogP",             "LogP",                 (None, None), False),
        ("molwt",       "Molecular Weight", "MW (Da)",              (None, None), False),
        ("tpsa",        "TPSA",             "TPSA (Å²)",   (None, None), False),
        # Row 3 — Structural
        ("hba",         "H-Bond Acceptors", "HBA Count",           (None, None), False),
        ("hbd",         "H-Bond Donors",    "HBD Count",           (None, None), False),
        ("num_atoms",   "Atom Count",       "Heavy Atom Count",    (None, None), False),
        # Row 4 — Drug-likeness
        ("rot_bonds",   "Rotatable Bonds",  "Rotatable Bonds",     (None, None), False),
        ("lipinski",    "Lipinski Score",   "Lipinski (RO5)",       (0, 5),      False),
        ("lilly_demerits", "Lilly Demerits","Lilly Medchem Demerits",(None, None), False),
    ]

    ncols = 3
    nrows = 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 22), sharey=False,
                              gridspec_kw={"wspace": 0.45, "hspace": 0.42})

    for idx, (col, display, y_label, (y_fixed_lo, y_fixed_hi), invert) in enumerate(metrics):
        row, col_idx = divmod(idx, ncols)
        ax = axes[row][col_idx]

        # Gather data arrays
        a = pd.to_numeric(act_filt.get(col, pd.Series(dtype=float)), errors="coerce").dropna().to_numpy()
        g = pd.to_numeric(gen_filt.get(col, pd.Series(dtype=float)), errors="coerce").dropna().to_numpy()

        if a.size == 0 and g.size == 0:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center", fontsize=9, color="#999")
            ax.set_xlim(0.35, 2.15)
            if show_text:
                ax.set_title(display, fontsize=11, fontweight="medium", pad=6)
            continue

        # Determine y-limits
        if y_fixed_lo is not None and y_fixed_hi is not None:
            y_min, y_max = y_fixed_lo, y_fixed_hi
        else:
            all_vals = np.concatenate([a, g]) if a.size > 0 else g
            if all_vals.size == 0:
                y_min, y_max = 0, 1
            else:
                data_range = float(np.max(all_vals) - np.min(all_vals)) or 1.0
                pad = 0.12 * data_range
                y_min = float(np.min(all_vals)) - pad
                y_max = float(np.max(all_vals)) + pad
                # Ensure non-negative axes start at 0
                if np.min(all_vals) >= 0:
                    y_min = max(y_min, -pad * 0.3)

        # Draw violins for actives (left) and generated (right)
        for arr, pos, color, med_col in [
            (a, 0.85, COLOR_ACTUAL, COLOR_ACTUAL_MEDIAN),
            (g, 1.65, COLOR_GENERATED, COLOR_GENERATED_MEDIAN),
        ]:
            if arr.size >= 2:
                _violin_with_style(ax, arr, pos, color, y_min, y_max, median_color=med_col)
            elif arr.size == 1:
                ax.scatter([pos], [arr[0]], color=color, s=28, zorder=5,
                          edgecolors=COLOR_VIOLIN_EDGE, linewidth=0.6)

        ax.set_xlim(0.35, 2.15)
        ax.set_xticks([0.85, 1.65])

        if show_text:
            ax.set_xticklabels(["Known\nActives", "Generated"], rotation=16, ha="right", fontsize=8)
            ax.set_title(display, fontsize=11, fontweight="medium", pad=6)
            ax.set_ylabel(y_label, fontsize=9)
        else:
            ax.set_xticklabels([])
            ax.set_title("")
            ax.set_ylabel("")
            ax.tick_params(axis="y", labelleft=False)

        ax.set_ylim(y_min, y_max)
        if invert:
            ax.invert_yaxis()
        ax.yaxis.grid(True, linestyle="--", alpha=0.2, color="#AAAAAA")
        ax.set_axisbelow(True)

    if show_text:
        from matplotlib.patches import Patch
        fig.legend(handles=[
            Patch(facecolor=COLOR_ACTUAL, edgecolor=COLOR_VIOLIN_EDGE, alpha=1.0,
                  label=f"Known Actives (n={len(act_filt)})"),
            Patch(facecolor=COLOR_GENERATED, edgecolor=COLOR_VIOLIN_EDGE, alpha=1.0,
                  label=f"Generated, Vina -15~0 (n={len(gen_filt)})"),
        ], loc="upper center", ncol=2, frameon=True, fancybox=False, edgecolor="#CCCCCC",
           bbox_to_anchor=(0.5, -0.01), framealpha=0.95, fontsize=10)

    plt.tight_layout(rect=(0, 0.03, 1, 0.97))
    suffix = "" if show_text else "_no_text"
    out_png = out_dir / f"{tag}_violin_expanded{suffix}.png"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_png}")


def main():
    parser = argparse.ArgumentParser(description="Expanded 12-panel violin plot")
    parser.add_argument("--xlsx", required=True, help="Generated molecules xlsx (evaluation_results_*.xlsx)")
    parser.add_argument("--act", required=True, help="Active molecules CSV (active_dock_qed_sa.csv)")
    parser.add_argument("--out_dir", required=True, help="Output directory for plots")
    parser.add_argument("--tag", required=True, help="Plot filename prefix (e.g. 8RI2_logp5_v2)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading generated from: {args.xlsx}")
    gen_df = load_generated(args.xlsx)
    print(f"  {len(gen_df)} molecules loaded")

    print(f"Loading actives from: {args.act}")
    active_df = load_actives(args.act)
    print(f"  {len(active_df)} actives loaded")

    print(f"\nTag: {args.tag}")
    plot_expanded(gen_df, active_df, out_dir, args.tag, show_text=True)
    plot_expanded(gen_df, active_df, out_dir, args.tag, show_text=False)


if __name__ == "__main__":
    main()
