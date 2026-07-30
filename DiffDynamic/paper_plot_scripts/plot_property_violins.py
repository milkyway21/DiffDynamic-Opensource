#!/usr/bin/env python3
"""
Comparative violin plots: Actual (PDB active ligands) vs Generated molecules.
Adapted from ccp_roco4_pipeline/plot_actual_vs_generated_properties.py.
Nature journal styling applied.

Data sources:
  - Actual:  batch_sdf_eval_*.csv (from batch docking)
  - Generated: {target}_all.sdf (QED, SA, SMILES; LogP computed from SMILES)

Usage:
    conda activate diffdynamic
    python plot_property_violins.py
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from scipy.interpolate import interp1d

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent

# Nature-style colour palette
COLOR_ACTUAL = "#a7d8e9"
COLOR_ACTUAL_MEDIAN = "#5a9ab5"
COLOR_GENERATED = "#fff0f0"
COLOR_GENERATED_MEDIAN = "#c97a7a"
COLOR_VIOLIN_SPINE = "#DDDDDD"
COLOR_VIOLIN_EDGE = "#444444"
MEDIAN_LINEWIDTH = 1.8


def latest_batch_csv(subdir):
    files = sorted(subdir.glob("batch_sdf_eval_*.csv"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"No batch_sdf_eval_*.csv in {subdir}")
    return files[-1]


def load_actual_from_eval_csv(csv_path):
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    df = pd.read_csv(csv_path)
    ok = df["success"].astype(str).str.lower().isin(("true", "1", "yes"))
    df = df.loc[ok].copy()
    out = pd.DataFrame({
        "vina_dock": pd.to_numeric(df["vina_dock"], errors="coerce"),
        "qed": pd.to_numeric(df["qed"], errors="coerce"),
        "sa": pd.to_numeric(df["sa"], errors="coerce"),
        "score": pd.to_numeric(df["comprehensive_score"], errors="coerce"),
        "logp": pd.to_numeric(df["logp"], errors="coerce"),
    })
    # Compute molecular weight from SMILES
    molwts = []
    for smi in df["smiles"].astype(str):
        try:
            m = Chem.MolFromSmiles(smi)
            molwts.append(float(Descriptors.MolWt(m)) if m else np.nan)
        except Exception:
            molwts.append(np.nan)
    out["molwt"] = molwts
    # Filter unreasonable Vina scores (positive = docking failure)
    out.loc[out["vina_dock"] > 0, "vina_dock"] = np.nan
    return out


def load_generated_from_sdf_file(sdf_path):
    """Load generated molecule properties from multi-molecule SDF.
    Handles SDFs with QED/SA/SMILES (no Vina_Dock/Comprehensive_Score).
    LogP computed from SMILES via RDKit Crippen.MolLogP.
    """
    from rdkit import Chem
    from rdkit.Chem import Crippen, Descriptors

    suppl = Chem.SDMolSupplier(str(sdf_path))
    rows = []
    for mol in suppl:
        if mol is None:
            continue
        try:
            smiles = mol.GetProp("SMILES") if mol.HasProp("SMILES") else ""
            qed_val = float(mol.GetProp("QED")) if mol.HasProp("QED") else np.nan
            sa_val = float(mol.GetProp("SA")) if mol.HasProp("SA") else np.nan
            vina = float(mol.GetProp("Vina_Dock")) if mol.HasProp("Vina_Dock") else np.nan
            cs = float(mol.GetProp("Comprehensive_Score")) if mol.HasProp("Comprehensive_Score") else np.nan
            rows.append({
                "smiles": smiles, "vina_dock": vina, "qed": qed_val,
                "sa": sa_val, "score": cs,
            })
        except (KeyError, ValueError):
            continue

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    logps = []
    molwts = []
    for smi in df["smiles"].astype(str):
        try:
            m = Chem.MolFromSmiles(smi)
            logps.append(float(Crippen.MolLogP(m)) if m else np.nan)
            molwts.append(float(Descriptors.MolWt(m)) if m else np.nan)
        except Exception:
            logps.append(np.nan)
            molwts.append(np.nan)
    df["logp"] = logps
    df["molwt"] = molwts
    return df


def _violin_with_style(ax, data, position, color, y_min, y_max, *, median_color,
                       max_half_width=0.23, grid_n=256, center_stat="mean"):
    """KDE violin pinched to zero width at data min/max, with jittered scatter points.

    center_stat: \"mean\" (default) or \"median\" — horizontal center line statistic.
    median_color: color for scatter points and the center line (name kept for compat).
    """
    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]
    if data.size < 2:
        return

    def _center(arr: np.ndarray) -> float:
        if str(center_stat).lower() == "median":
            return float(np.median(arr))
        return float(np.mean(arr))

    # Cap scatter points to avoid visual clutter; KDE still uses full dataset
    MAX_SCATTER_POINTS = 500
    scatter_data = data
    if data.size > MAX_SCATTER_POINTS:
        rng = np.random.default_rng(42)
        scatter_data = rng.choice(data, MAX_SCATTER_POINTS, replace=False)

    y_range = y_max - y_min if y_max > y_min else 1.0
    y_axis_top = y_max + 0.06 * y_range
    y_axis_bot = y_min - 0.06 * y_range

    yd_lo = float(np.min(data))
    yd_hi = float(np.max(data))
    span = yd_hi - yd_lo

    ax.plot(
        [position, position], [y_axis_bot, y_axis_top],
        color=COLOR_VIOLIN_SPINE, linewidth=1.8, zorder=1, solid_capstyle="round",
    )

    if span < 1e-14:
        cen = _center(data)
        ax.plot(
            [position - 0.08125, position + 0.08125], [cen, cen],  # +30% wider
            color=median_color, solid_capstyle="round",
            linewidth=MEDIAN_LINEWIDTH, zorder=10,  # zorder 10 = topmost
        )
        # Jittered scatter for identical values: use fixed width
        rng = np.random.default_rng(42)
        jitter = rng.uniform(-0.70 * max_half_width, 0.70 * max_half_width, scatter_data.size)
        ax.scatter(position + jitter, scatter_data,
                   s=14, alpha=0.8, color=median_color,
                   edgecolors='none', linewidth=0, zorder=4)
        return

    y_grid = np.linspace(yd_lo, yd_hi, grid_n)
    kde = gaussian_kde(data, bw_method=0.35)  # Reduced from 0.7 for less smoothing
    dens = kde.evaluate(y_grid)
    dens = np.clip(dens, 0.0, None)
    dmax = float(np.max(dens))
    if dmax <= 0:
        # Jittered scatter for zero-density edge case
        rng = np.random.default_rng(42)
        jitter = rng.uniform(-0.70 * max_half_width, 0.70 * max_half_width, scatter_data.size)
        ax.scatter(position + jitter, scatter_data,
                   s=14, alpha=0.8, color=median_color,
                   edgecolors='none', linewidth=0, zorder=4)
        return

    dens_norm = dens / dmax
    t = (y_grid - yd_lo) / span
    pinch = 4.0 * t * (1.0 - t)
    half_w = dens_norm * pinch * max_half_width

    ax.fill_betweenx(
        y_grid, position - half_w, position + half_w,
        facecolor=color, edgecolor=COLOR_VIOLIN_EDGE, linewidth=1.0,
        alpha=1.0, zorder=3,
    )

    # --- Jittered strip scatter: individual molecule positions within violin ---
    # Interpolate violin half-width at each data point's y-value
    hw_interp = interp1d(y_grid, half_w, kind='linear',
                         bounds_error=False, fill_value=0.0)
    pt_hw = hw_interp(scatter_data)
    rng = np.random.default_rng(42)
    # Keep points inside violin: use 80% of available width for margin
    jitter = rng.uniform(-0.80, 0.80, scatter_data.size) * np.maximum(pt_hw, 0.0)
    ax.scatter(position + jitter, scatter_data,
               s=14, alpha=0.8, color=median_color,
               edgecolors='none', linewidth=0, zorder=4)

    cen = _center(data)
    ax.plot(
        [position - 0.08125, position + 0.08125], [cen, cen],  # +30% wider
        color=median_color, linewidth=MEDIAN_LINEWIDTH, zorder=10,  # zorder 10 = topmost
        solid_capstyle="round",
    )


def _metric_values(df, col):
    return pd.to_numeric(df[col], errors="coerce").dropna().to_numpy(dtype=float)


def plot_property_grid(actual, generated, title, out_png, *, show_text=True):
    """Create 1x5 violin grid comparing actual vs generated across 5 properties."""
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "axes.linewidth": 1.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })

    metrics = [
        ("vina_dock", "Vina Dock", "VINA DOCK"),
        ("qed", "QED", "QED"),
        ("sa", "SA", "SA"),
        ("score", "Comprehensive Score", "COMPREHENSIVE SCORE"),
        ("logp", "LogP", "LOGP"),
    ]

    fig, axes = plt.subplots(
        1, 5, figsize=(18.5, 5.2), sharey=False,
        gridspec_kw={"wspace": 0.36},
    )

    panel_labels = ["a", "b", "c", "d", "e"]

    for ax_idx, (ax, (col, display, y_label)) in enumerate(zip(axes, metrics)):
        a = _metric_values(actual, col)
        g = _metric_values(generated, col) if not generated.empty else np.array([])
        pool = np.concatenate([a, g]) if (a.size and g.size) else (a if a.size else g)

        if pool.size == 0:
            if show_text:
                ax.set_title(display, fontsize=10)
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=9)
            continue

        y_min, y_max = float(np.min(pool)), float(np.max(pool))
        y_range = y_max - y_min if y_max > y_min else 1.0
        y_lo = y_min - 0.08 * y_range
        y_hi = y_max + 0.08 * y_range

        for arr, pos, color, med_col in (
            (a, 0.85, COLOR_ACTUAL, COLOR_ACTUAL_MEDIAN),
            (g, 1.65, COLOR_GENERATED, COLOR_GENERATED_MEDIAN),
        ):
            if arr.size >= 2:
                _violin_with_style(ax, arr, pos, color, y_min, y_max, median_color=med_col)
            elif arr.size == 1:
                ax.scatter([pos], [arr[0]], color=color, s=32, zorder=5,
                           edgecolors=COLOR_VIOLIN_EDGE, linewidth=0.8)

        ax.set_xlim(0.35, 2.15)
        ax.set_xticks([0.85, 1.65])

        if show_text:
            ax.set_xticklabels(["Actual", "Generated"], rotation=16, ha="right", fontsize=10)
            ax.set_title(display, fontsize=11, fontweight="medium", pad=8)
            ax.set_ylabel(y_label, fontsize=10)
            ax.text(
                -0.12, 1.02, panel_labels[ax_idx],
                transform=ax.transAxes, fontsize=12, fontweight="bold",
                va="bottom", ha="left",
            )
        else:
            ax.set_xticklabels([])
            ax.set_title("")
            ax.set_ylabel("")
            ax.tick_params(axis="y", labelleft=False)

        ax.set_ylim(y_lo, y_hi)
        if col == "vina_dock":
            ax.invert_yaxis()
        ax.yaxis.grid(True, linestyle="--", alpha=0.25, color="#AAAAAA")
        ax.set_axisbelow(True)

    if show_text:
        from matplotlib.patches import Patch
        legend_handles = [
            Patch(facecolor=COLOR_ACTUAL, edgecolor=COLOR_VIOLIN_EDGE, alpha=1.0,
                  label="Known actives"),
            Patch(facecolor=COLOR_GENERATED, edgecolor=COLOR_VIOLIN_EDGE, alpha=1.0,
                  label="Generated"),
        ]
        fig.legend(
            handles=legend_handles, loc="upper center", ncol=2,
            frameon=True, fancybox=False, edgecolor="#CCCCCC",
            bbox_to_anchor=(0.5, -0.02), framealpha=0.95, fontsize=9,
        )

    plt.tight_layout(rect=(0, 0.04, 1, 0.96))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"Saved: {out_png}")


def _load_background_molecules(n=1000, seed=42):
    """Load background molecules from RDKit NCI first_5K.smi."""
    from rdkit import Chem
    from rdkit.Chem import Crippen, Descriptors
    import random

    bg_path = Path("/home/user/Desktop/Ye/ccp_roco4_pipeline/.venv/lib/python3.12/site-packages/rdkit/Data/NCI/first_5K.smi")
    if not bg_path.is_file():
        return None

    smiles_list = []
    with open(bg_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            smi = line.split()[0]
            smiles_list.append(smi)

    random.seed(seed)
    selected = random.sample(smiles_list, min(n, len(smiles_list)))

    rows = []
    for smi in selected:
        try:
            m = Chem.MolFromSmiles(smi)
            if m:
                rows.append({
                    "logp": float(Crippen.MolLogP(m)),
                    "molwt": float(Descriptors.MolWt(m)),
                })
        except Exception:
            continue
    print(f"  Background loaded: {len(rows)} molecules")
    return pd.DataFrame(rows)


def plot_logp_vs_molweight(actual, generated, out_png, *, show_text=True, with_background=True):
    """2D scatter plot: LogP vs Molecular Weight, Actual vs Generated.
    Color scheme matches t-SNE plot: gray = background, red = actual, blue = generated.
    """
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "axes.linewidth": 1.2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })

    # Drop NaN values
    actual_valid = actual[["logp", "molwt"]].dropna()
    generated_valid = generated[["logp", "molwt"]].dropna()

    # Load background molecules
    bg_df = _load_background_molecules(n=5000) if with_background else None

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    # Color scheme matching t-SNE plot
    COLOR_BACKGROUND = "#9a9a9a"  # Gray for background
    COLOR_ACTUAL = "#C51610"      # Red for reference ligands
    COLOR_GENERATED = "#2563eb"   # Blue for generated

    # Background molecules (lowest z-order)
    if bg_df is not None and len(bg_df) > 0:
        ax.scatter(
            bg_df["molwt"], bg_df["logp"],
            s=14, marker="o", c=COLOR_BACKGROUND, edgecolors="none",
            alpha=0.3, label=f"Background ({len(bg_df)})", zorder=1
        )

    # Generated molecules (middle z-order)
    if len(generated_valid) > 0:
        ax.scatter(
            generated_valid["molwt"], generated_valid["logp"],
            s=18, marker="o", c=COLOR_GENERATED, edgecolors="none",
            alpha=0.6, label=f"Generated ({len(generated_valid)})", zorder=3
        )

    # Actual molecules (highest z-order)
    if len(actual_valid) > 0:
        ax.scatter(
            actual_valid["molwt"], actual_valid["logp"],
            s=18, marker="o", c=COLOR_ACTUAL, edgecolors="none",
            alpha=1.0, label=f"Known actives ({len(actual_valid)})", zorder=6
        )

    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.tick_params(axis="both", which="major", width=1.2, length=6, labelsize=12)

    if show_text:
        ax.set_xlabel("Molecular Weight", fontsize=12)
        ax.set_ylabel("LogP (Crippen)", fontsize=12)
        ax.set_title("LogP vs Molecular Weight - 8R12", fontsize=13, pad=10)
        ax.legend(loc="best", framealpha=0.92, prop={"size": 9}, markerscale=2)
    else:
        ax.tick_params(axis="both", which="major", labelbottom=False, labelleft=False)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title("")

    ax.grid(True, linestyle="--", alpha=0.25, color="#AAAAAA")
    ax.set_axisbelow(True)

    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"Saved: {out_png}")


def main():
    try:
        import rdkit  # noqa: F401
    except ImportError:
        raise SystemExit("RDKit required. Run: conda activate diffdynamic")

    out_dir = ROOT / "plots"

    # --- 8R12 ---
    print("=== 8R12 ===")
    eval_dir = ROOT / "8R12_eval"
    csv_path = latest_batch_csv(eval_dir)
    actual = load_actual_from_eval_csv(csv_path)
    print(f"Actual CSV: {csv_path.name} | n={len(actual)}")

    gen_sdf = ROOT / "8R12_results" / "8R12_top50.sdf"
    gen = load_generated_from_sdf_file(gen_sdf)
    print(f"Generated SDF: {gen_sdf.name} | n={len(gen)}")
    for col in ["vina_dock", "qed", "sa", "score", "logp", "molwt"]:
        print(f"  {col}: {gen[col].notna().sum()} valid")

    plot_property_grid(actual, gen, title="8R12",
                       out_png=out_dir / "8R12_property_violin.png", show_text=True)
    plot_property_grid(actual, gen, title="",
                       out_png=out_dir / "8R12_property_violin_no_text.png", show_text=False)

    # --- LogP vs Molecular Weight scatter ---
    plot_logp_vs_molweight(actual, gen,
                           out_png=out_dir / "8R12_logp_vs_molweight.png", show_text=True)
    plot_logp_vs_molweight(actual, gen,
                           out_png=out_dir / "8R12_logp_vs_molweight_no_text.png", show_text=False)

    print("\nDone.")


if __name__ == "__main__":
    main()
