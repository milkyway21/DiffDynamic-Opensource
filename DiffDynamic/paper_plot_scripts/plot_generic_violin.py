#!/usr/bin/env python3
"""Generic violin plot for pocket comparison: Vina Dock / QED / SA.
Fixed y-axes: Vina 0~-15 (-15 top), QED 0-1, SA 0-1.
Filters generated molecules to Vina in (-15, 0).

Usage:
  python3 plot_generic_violin.py --gen GEN_CSV --act ACTIVE_CSV --out_dir OUT_DIR --tag TAG
"""
import sys, warnings, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

warnings.filterwarnings("ignore")

# Same visual constants as original 8RI2 plot_property_violins.py
sys.path.insert(0, str(Path(__file__).parent))
from plot_property_violins import (_violin_with_style, COLOR_ACTUAL, COLOR_GENERATED,
    COLOR_ACTUAL_MEDIAN, COLOR_GENERATED_MEDIAN, COLOR_VIOLIN_EDGE, COLOR_VIOLIN_SPINE,
    MEDIAN_LINEWIDTH)


def plot_violins(gen_df, active_df, out_dir, tag, show_text=True):
    """Generate the 3-panel violin plot with fixed y-axes."""
    matplotlib.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 11, "axes.labelsize": 12, "axes.titlesize": 13,
        "figure.dpi": 300, "savefig.dpi": 300, "axes.linewidth": 1.0,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.labelsize": 10, "ytick.labelsize": 10,
    })

    # Filter generated: only Vina in (-15, 0)
    gen_filtered = gen_df[(gen_df["vina_dock"].notna()) &
                           (gen_df["vina_dock"] < 0) &
                           (gen_df["vina_dock"] > -15)].copy()
    vina_filtered_out = len(gen_df) - len(gen_filtered)
    print(f"Generated: total={len(gen_df)}, filtered(Vina -15~0)={len(gen_filtered)}, excluded={vina_filtered_out}")
    print(f"Known actives: {len(active_df)}")

    # Fixed y-axis ranges
    metrics = [
        ("vina_dock", "Vina Dock", "Vina Dock (kcal/mol)", (-15, 0), True),
        ("qed", "QED", "QED", (0, 1), False),
        ("sa", "SA Score", "SA Score", (0, 1), False),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12, 5), sharey=False, gridspec_kw={"wspace": 0.4})

    for ax, (col, display, y_label, (y_fixed_lo, y_fixed_hi), invert) in zip(axes, metrics):
        a = pd.to_numeric(active_df[col], errors="coerce").dropna().to_numpy()
        g = pd.to_numeric(gen_filtered[col], errors="coerce").dropna().to_numpy()

        # Use fixed axis bounds
        y_min, y_max = y_fixed_lo, y_fixed_hi

        for arr, pos, color, med_col in [
            (a, 0.85, COLOR_ACTUAL, COLOR_ACTUAL_MEDIAN),
            (g, 1.65, COLOR_GENERATED, COLOR_GENERATED_MEDIAN),
        ]:
            if arr.size >= 2:
                _violin_with_style(ax, arr, pos, color, y_min, y_max, median_color=med_col)
            elif arr.size == 1:
                ax.scatter([pos], [arr[0]], color=color, s=32, zorder=5,
                          edgecolors=COLOR_VIOLIN_EDGE, linewidth=0.8)

        ax.set_xlim(0.35, 2.15)
        ax.set_xticks([0.85, 1.65])

        if show_text:
            ax.set_xticklabels(["Known\nActives", "Generated"], rotation=16, ha="right", fontsize=9)
            ax.set_title(display, fontsize=11, fontweight="medium", pad=8)
            ax.set_ylabel(y_label, fontsize=10)
        else:
            ax.set_xticklabels([])
            ax.set_title("")
            ax.set_ylabel("")
            ax.tick_params(axis="y", labelleft=False)

        ax.set_ylim(y_min, y_max)
        if invert:
            ax.invert_yaxis()
        ax.yaxis.grid(True, linestyle="--", alpha=0.25, color="#AAAAAA")
        ax.set_axisbelow(True)

    if show_text:
        from matplotlib.patches import Patch
        fig.legend(handles=[
            Patch(facecolor=COLOR_ACTUAL, edgecolor=COLOR_VIOLIN_EDGE, alpha=1.0,
                  label=f"Known Actives (n={len(active_df)})"),
            Patch(facecolor=COLOR_GENERATED, edgecolor=COLOR_VIOLIN_EDGE, alpha=1.0,
                  label=f"Generated, Vina -15~0 (n={len(gen_filtered)})"),
        ], loc="upper center", ncol=2, frameon=True, fancybox=False, edgecolor="#CCCCCC",
           bbox_to_anchor=(0.5, -0.02), framealpha=0.95, fontsize=9)

    plt.tight_layout(rect=(0, 0.06, 1, 0.96))
    suffix = "" if show_text else "_no_text"
    out_png = out_dir / f"{tag}_violin{suffix}.png"
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_png}")


def main():
    parser = argparse.ArgumentParser(description="Generic violin plot for pocket comparison")
    parser.add_argument("--gen", required=True, help="Generated molecules CSV (smiles,vina_dock,qed,sa)")
    parser.add_argument("--act", required=True, help="Active molecules CSV (name,smiles,qed,sa,vina_dock)")
    parser.add_argument("--out_dir", required=True, help="Output directory for plots")
    parser.add_argument("--tag", required=True, help="Plot filename prefix (e.g. 8RI2, 6W63)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gen_df = pd.read_csv(args.gen)
    active_df = pd.read_csv(args.act)

    print(f"Tag: {args.tag}")
    plot_violins(gen_df, active_df, out_dir, args.tag, show_text=True)
    plot_violins(gen_df, active_df, out_dir, args.tag, show_text=False)


if __name__ == "__main__":
    main()
