#!/usr/bin/env python3
"""Re-color 3-panel violin plots with 10 publication-oriented palettes.

Does NOT overwrite original plots. Writes to <exp>/plots/palette_variants/.

Usage:
  conda run -n diffdynamic python3 scripts/plot_violin_palette_variants.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, "/data/ye/protein-ligand")
from plot_property_violins import _violin_with_style  # noqa: E402

# ---------------------------------------------------------------------------
# 10 publication-oriented palettes
# Fill = soft pastel (print-friendly); median = darker companion for contrast.
# References: Okabe–Ito, Paul Tol, Nature/Cell muted pairs, Wong colorblind.
# ---------------------------------------------------------------------------
PALETTES = [
    {
        "id": "v01_skyblue_coral",
        "name": "Sky Blue / Soft Coral",
        "note": "经典冷暖对比：淡天蓝 vs 浅珊瑚（接近现用风格的精修版）",
        "actual": "#A8D5E5",
        "actual_med": "#3D7A96",
        "generated": "#F6C6C1",
        "generated_med": "#B85C55",
        "edge": "#4A4A4A",
    },
    {
        "id": "v02_okabe_ito",
        "name": "Okabe–Ito Blue / Orange",
        "note": "色盲友好 Okabe–Ito：天蓝 vs 橙黄，Nature Methods 常用",
        "actual": "#9FD4F0",
        "actual_med": "#0072B2",
        "generated": "#F5D08A",
        "generated_med": "#E69F00",
        "edge": "#333333",
    },
    {
        "id": "v03_tol_muted",
        "name": "Paul Tol Muted Cyan / Wine",
        "note": "Paul Tol muted：青灰蓝 vs 酒红，适合印刷灰度仍可辨",
        "actual": "#B3CDE3",
        "actual_med": "#4C78A8",
        "generated": "#E8C4C8",
        "generated_med": "#AC252A",
        "edge": "#3D3D3D",
    },
    {
        "id": "v04_nature_blue_rose",
        "name": "Nature Soft Blue / Rose",
        "note": "Nature 风格低饱和：雾蓝 vs 雾玫瑰",
        "actual": "#BDD7EE",
        "actual_med": "#5B8DB8",
        "generated": "#F2C6D0",
        "generated_med": "#C45C74",
        "edge": "#404040",
    },
    {
        "id": "v05_wong_cb",
        "name": "Wong CB Blue / Vermillion",
        "note": "Wong 色盲安全：蓝 vs 朱红（对比强，适合主图）",
        "actual": "#A6CEE3",
        "actual_med": "#1F78B4",
        "generated": "#FDBF6F",
        "generated_med": "#D55E00",
        "edge": "#2F2F2F",
    },
    {
        "id": "v06_teal_sand",
        "name": "Teal / Warm Sand",
        "note": "冷青绿 vs 暖沙色，常见于 Cell / JACS SI",
        "actual": "#A8DADC",
        "actual_med": "#2A6F6F",
        "generated": "#F0D9B5",
        "generated_med": "#A67C52",
        "edge": "#3A3A3A",
    },
    {
        "id": "v07_slate_terracotta",
        "name": "Slate Blue / Terracotta",
        "note": "石板蓝 vs 陶土橙，偏严肃论文主图气质",
        "actual": "#B8C5D6",
        "actual_med": "#4A5E7A",
        "generated": "#E8B4A0",
        "generated_med": "#C05A3C",
        "edge": "#333333",
    },
    {
        "id": "v08_ice_lavender",
        "name": "Ice Blue / Soft Lavender",
        "note": "极淡冰蓝 vs 淡紫灰，打印友好、不抢结构图注意力",
        "actual": "#C5DDF0",
        "actual_med": "#5A7FA0",
        "generated": "#D9D0E6",
        "generated_med": "#7A6A9A",
        "edge": "#555555",
    },
    {
        "id": "v09_sage_periwinkle",
        "name": "Sage Green / Periwinkle",
        "note": "鼠尾草绿 vs 长春花蓝，双冷色但色相差足够",
        "actual": "#C5D9B8",
        "actual_med": "#5A7D4A",
        "generated": "#C4C9E8",
        "generated_med": "#5C64A8",
        "edge": "#3F3F3F",
    },
    {
        "id": "v10_lancet_duo",
        "name": "Lancet Soft Cyan / Magenta",
        "note": "Lancet 系软青 vs 品红，对比清晰且不过艳",
        "actual": "#A9D6E5",
        "actual_med": "#1B6CA8",
        "generated": "#E8B4D0",
        "generated_med": "#B03A6E",
        "edge": "#3A3A3A",
    },
]

EXPERIMENTS = [
    {
        "dir": ROOT / "experiments/8ri2_ligand_instant_gen100_prudent_tbr10",
        "tag": "8RI2_ligand_prudent_tbr10",
        "act": Path("/data/ye/protein-ligand/8RI2/active_dock_qed_sa.csv"),
    },
    {
        "dir": ROOT / "experiments/6w63_4wi_instant_gen100_prudent_tbr10",
        "tag": "6W63_4WI_prudent_tbr10",
        "act": Path("/data/ye/protein-ligand/6W63/active_dock_qed_sa.csv"),
    },
]


def plot_three_panel(gen_df, active_df, out_png, palette, show_text=True):
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 11, "axes.labelsize": 12, "axes.titlesize": 13,
        "figure.dpi": 300, "savefig.dpi": 300, "axes.linewidth": 1.0,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.labelsize": 10, "ytick.labelsize": 10,
    })

    gen_filtered = gen_df[
        (gen_df["vina_dock"].notna())
        & (gen_df["vina_dock"] < 0)
        & (gen_df["vina_dock"] > -15)
    ].copy()

    metrics = [
        ("vina_dock", "Vina Dock", "Vina Dock (kcal/mol)", (-15, 0), True),
        ("qed", "QED", "QED", (0, 1), False),
        ("sa", "SA Score", "SA Score", (0, 1), False),
    ]

    c_a, c_am = palette["actual"], palette["actual_med"]
    c_g, c_gm = palette["generated"], palette["generated_med"]
    c_edge = palette["edge"]

    fig, axes = plt.subplots(1, 3, figsize=(12, 5), sharey=False, gridspec_kw={"wspace": 0.4})
    for ax, (col, display, y_label, (y_lo, y_hi), invert) in zip(axes, metrics):
        a = pd.to_numeric(active_df[col], errors="coerce").dropna().to_numpy()
        g = pd.to_numeric(gen_filtered[col], errors="coerce").dropna().to_numpy()
        for arr, pos, color, med_col in [
            (a, 0.85, c_a, c_am),
            (g, 1.65, c_g, c_gm),
        ]:
            if arr.size >= 2:
                # temporarily patch edge via local draw: _violin_with_style uses global edge
                _violin_with_style(ax, arr, pos, color, y_lo, y_hi, median_color=med_col)
            elif arr.size == 1:
                ax.scatter([pos], [arr[0]], color=color, s=32, zorder=5,
                           edgecolors=c_edge, linewidth=0.8)

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
        ax.set_ylim(y_lo, y_hi)
        if invert:
            ax.invert_yaxis()
        ax.yaxis.grid(True, linestyle="--", alpha=0.25, color="#AAAAAA")
        ax.set_axisbelow(True)

    if show_text:
        fig.legend(
            handles=[
                Patch(facecolor=c_a, edgecolor=c_edge, label=f"Known Actives (n={len(active_df)})"),
                Patch(facecolor=c_g, edgecolor=c_edge, label=f"Generated, Vina -15~0 (n={len(gen_filtered)})"),
            ],
            loc="upper center", ncol=2, frameon=True, fancybox=False, edgecolor="#CCCCCC",
            bbox_to_anchor=(0.5, -0.02), framealpha=0.95, fontsize=9,
        )
        fig.suptitle(f"{palette['name']}", fontsize=11, y=1.02, color="#333333")

    plt.tight_layout(rect=(0, 0.06, 1, 0.96))
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_swatch_sheet(out_png: Path):
    """One-page overview of all 10 palettes for quick selection."""
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 10, "figure.dpi": 200, "savefig.dpi": 250,
    })
    fig, axes = plt.subplots(5, 2, figsize=(11, 13))
    axes = axes.ravel()
    for ax, p in zip(axes, PALETTES):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 3)
        ax.add_patch(Rectangle((0.4, 0.7), 3.8, 1.6, facecolor=p["actual"],
                               edgecolor=p["edge"], linewidth=1.0))
        ax.add_patch(Rectangle((5.0, 0.7), 3.8, 1.6, facecolor=p["generated"],
                               edgecolor=p["edge"], linewidth=1.0))
        # median markers
        ax.plot([0.7, 3.9], [1.5, 1.5], color=p["actual_med"], lw=3.0, solid_capstyle="round")
        ax.plot([5.3, 8.5], [1.5, 1.5], color=p["generated_med"], lw=3.0, solid_capstyle="round")
        ax.text(2.3, 2.55, "Known Actives", ha="center", va="bottom", fontsize=9)
        ax.text(6.9, 2.55, "Generated", ha="center", va="bottom", fontsize=9)
        ax.set_title(f"{p['id']}: {p['name']}", fontsize=11, fontweight="medium", loc="left", pad=6)
        ax.text(0.4, 0.15, p["note"], fontsize=7.5, color="#555555", ha="left", va="bottom")
        ax.axis("off")
    fig.suptitle("Violin palette candidates (paper-oriented)", fontsize=14, y=0.995)
    plt.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out_png, dpi=250, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved swatch sheet: {out_png}")


def main():
    # Shared swatch sheet under DiffDynamic/experiments for browsing
    overview_dir = ROOT / "experiments/_palette_overview"
    overview_dir.mkdir(parents=True, exist_ok=True)
    make_swatch_sheet(overview_dir / "violin_palette_10_swatches.png")
    readme = overview_dir / "README_palette_variants.md"
    lines = [
        "# Violin palette variants (10 options)",
        "",
        "Original plots are untouched. Variants live in each experiment's",
        "`plots/palette_variants/`.",
        "",
        "| ID | Name | Notes |",
        "|----|------|-------|",
    ]
    for p in PALETTES:
        lines.append(f"| `{p['id']}` | {p['name']} | {p['note']} |")
    lines += [
        "",
        "Pick one `id` and tell me which to promote as the final paper style.",
        "",
        "![swatches](violin_palette_10_swatches.png)",
        "",
    ]
    readme.write_text("\n".join(lines), encoding="utf-8")

    for exp in EXPERIMENTS:
        gen_csv = exp["dir"] / "generated_dock_qed_sa.csv"
        if not gen_csv.exists():
            print(f"SKIP missing CSV: {gen_csv}")
            continue
        out_dir = exp["dir"] / "plots" / "palette_variants"
        out_dir.mkdir(parents=True, exist_ok=True)
        gen_df = pd.read_csv(gen_csv)
        act_df = pd.read_csv(exp["act"])
        print(f"\n=== {exp['tag']} ===")
        for p in PALETTES:
            out = out_dir / f"{exp['tag']}_violin_{p['id']}.png"
            plot_three_panel(gen_df, act_df, out, p, show_text=True)
            print(f"  Saved {out.name}")
        # also a mini legend strip for this experiment
        make_swatch_sheet(out_dir / f"{exp['tag']}_palette_swatches.png")

    print("\nDone. Browse experiments/_palette_overview/ and */plots/palette_variants/")


if __name__ == "__main__":
    main()
