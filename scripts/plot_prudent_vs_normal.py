#!/usr/bin/env python3
"""Prudent vs Normal (dynamic_locked) comparison: 3-way violin plots.
Fixed y-axes: Vina 0~-15 (-15 top), QED 0-1, SA 0-1.

Usage:
  python3 plot_prudent_vs_normal.py \
    --prudent PRUDENT_CSV --normal NORMAL_CSV --act ACTIVE_CSV \
    --out_dir OUT_DIR --tag TAG
"""
import sys, warnings, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# Import shared styling from project
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "protein-ligand"))
try:
    from plot_property_violins import (
        _violin_with_style, COLOR_ACTUAL, COLOR_GENERATED,
        COLOR_ACTUAL_MEDIAN, COLOR_GENERATED_MEDIAN,
        COLOR_VIOLIN_EDGE, COLOR_VIOLIN_SPINE, MEDIAN_LINEWIDTH,
    )
except ImportError:
    # Fallback colors
    COLOR_ACTUAL = "#C51610"
    COLOR_GENERATED = "#2563eb"
    COLOR_PRUDENT = "#059669"  # Green for prudent
    COLOR_ACTUAL_MEDIAN = "#8B0000"
    COLOR_GENERATED_MEDIAN = "#1e40af"
    COLOR_PRUDENT_MEDIAN = "#064e3b"
    COLOR_VIOLIN_EDGE = "#333333"
    COLOR_VIOLIN_SPINE = "#666666"
    MEDIAN_LINEWIDTH = 1.5


def plot_comparison_violins(prudent_df, normal_df, active_df, out_dir, tag, show_text=True):
    """Generate 3-panel violin plot: Known Actives vs Normal vs Prudent."""
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 11, "axes.labelsize": 12, "axes.titlesize": 13,
        "figure.dpi": 300, "savefig.dpi": 300, "axes.linewidth": 1.0,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.labelsize": 10, "ytick.labelsize": 10,
    })

    # Filter: only Vina in (-15, 0)
    def filter_vina(df):
        if df is None or len(df) == 0:
            return pd.DataFrame()
        return df[(df["vina_dock"].notna()) &
                  (df["vina_dock"] < 0) &
                  (df["vina_dock"] > -15)].copy()

    p_filt = filter_vina(prudent_df)
    n_filt = filter_vina(normal_df)

    print(f"Prudent: total={len(prudent_df)}, filtered Vina -15~0={len(p_filt)}")
    print(f"Normal:  total={len(normal_df) if normal_df is not None else 0}, filtered Vina -15~0={len(n_filt)}")
    print(f"Actives: {len(active_df) if active_df is not None else 0}")

    metrics = [
        ("vina_dock", "Vina Dock", "Vina Dock (kcal/mol)", (-15, 0), True),
        ("qed", "QED", "QED", (0, 1), False),
        ("sa", "SA Score", "SA Score", (0, 1), False),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False,
                            gridspec_kw={"wspace": 0.4})

    for ax, (col, display, y_label, (y_fixed_lo, y_fixed_hi), invert) in zip(axes, metrics):
        a_data = pd.to_numeric(active_df[col], errors="coerce").dropna().to_numpy() if active_df is not None else np.array([])
        n_data = pd.to_numeric(n_filt[col], errors="coerce").dropna().to_numpy()
        p_data = pd.to_numeric(p_filt[col], errors="coerce").dropna().to_numpy()

        y_min, y_max = y_fixed_lo, y_fixed_hi

        # Three violins: Known Actives (left), Normal (middle), Prudent (right)
        datasets = [
            (a_data, 0.65, "#C51610", "#8B0000", "Known\nActives"),
            (n_data, 1.50, "#2563eb", "#1e40af", "Normal\n(dyn_locked)"),
            (p_data, 2.35, "#059669", "#064e3b", "Prudent"),
        ]

        for arr, pos, color, med_col, label in datasets:
            if arr.size >= 2:
                try:
                    _violin_with_style(ax, arr, pos, color, y_min, y_max,
                                      median_color=med_col)
                except Exception:
                    # Fallback: simple boxplot
                    parts = ax.violinplot(arr, positions=[pos], showmeans=True,
                                         showmedians=True, showextrema=True)
                    for body in parts["bodies"]:
                        body.set_facecolor(color)
                        body.set_alpha(0.7)
            elif arr.size == 1:
                ax.scatter([pos], [arr[0]], color=color, s=32, zorder=5,
                          edgecolors=COLOR_VIOLIN_EDGE, linewidth=0.8)

        ax.set_xlim(0.15, 2.85)
        ax.set_xticks([0.65, 1.50, 2.35])

        if show_text:
            ax.set_xticklabels(["Known\nActives", "Normal\n(dyn_locked)", "Prudent"],
                              rotation=16, ha="right", fontsize=8)
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
            Patch(facecolor="#C51610", edgecolor=COLOR_VIOLIN_EDGE, alpha=1.0,
                  label=f"Known Actives (n={len(active_df) if active_df is not None else 0})"),
            Patch(facecolor="#2563eb", edgecolor=COLOR_VIOLIN_EDGE, alpha=1.0,
                  label=f"Normal dyn_locked (n={len(n_filt)})"),
            Patch(facecolor="#059669", edgecolor=COLOR_VIOLIN_EDGE, alpha=1.0,
                  label=f"Prudent (n={len(p_filt)})"),
        ], loc="upper center", ncol=3, frameon=True, fancybox=False,
           edgecolor="#CCCCCC", bbox_to_anchor=(0.5, -0.02), framealpha=0.95,
           fontsize=8)

    plt.tight_layout(rect=(0, 0.08, 1, 0.96))
    suffix = "" if show_text else "_no_text"
    out_png = out_dir / f"{tag}_comparison{suffix}.png"
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_png}")


def main():
    parser = argparse.ArgumentParser(
        description="Prudent vs Normal comparison violin plots")
    parser.add_argument("--prudent", required=True,
                       help="Prudent generated molecules CSV (smiles,vina_dock,qed,sa)")
    parser.add_argument("--normal", required=True,
                       help="Normal mode generated molecules CSV")
    parser.add_argument("--act", required=True,
                       help="Active molecules CSV (name,smiles,qed,sa,vina_dock)")
    parser.add_argument("--out_dir", required=True, help="Output directory for plots")
    parser.add_argument("--tag", required=True,
                       help="Plot filename prefix (e.g. 6W63, 8RI2)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prudent_df = pd.read_csv(args.prudent)
    normal_df = pd.read_csv(args.normal)
    active_df = pd.read_csv(args.act)

    print(f"Tag: {args.tag}")
    plot_comparison_violins(prudent_df, normal_df, active_df, out_dir,
                           args.tag, show_text=True)
    plot_comparison_violins(prudent_df, normal_df, active_df, out_dir,
                           args.tag, show_text=False)

    # Also print summary stats
    def summarize(df, label):
        if df is None or len(df) == 0:
            print(f"\n{label}: NO DATA")
            return
        v = pd.to_numeric(df["vina_dock"], errors="coerce")
        v_filt = v[(v < 0) & (v > -15)]
        q = pd.to_numeric(df["qed"], errors="coerce")
        s = pd.to_numeric(df["sa"], errors="coerce")
        print(f"\n{label} (n={len(v_filt)}):")
        print(f"  Vina: mean={v_filt.mean():.2f} median={v_filt.median():.2f} "
              f"min={v_filt.min():.2f} top10%={v_filt.nsmallest(max(1,int(len(v_filt)*0.1))).mean():.2f}")
        print(f"  QED:  mean={q.mean():.3f} median={q.median():.3f}")
        print(f"  SA:   mean={s.mean():.3f} median={s.median():.3f}")

    summarize(prudent_df, "PRUDENT")
    summarize(normal_df, "NORMAL (dyn_locked)")
    summarize(active_df, "KNOWN ACTIVES")


if __name__ == "__main__":
    main()
