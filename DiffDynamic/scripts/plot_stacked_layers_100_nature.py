#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Horizontal stacked bars for 100-pocket 3-layer scores (blue–orange diverging).

Palette: ColorBrewer RdBu / user heatmap (cool blue ↔ warm orange).
- Ligand A/B/C: light → dark blue
- Compat D/E: peach → soft orange
- Pocket F/G/H: mid → deep burnt orange

Optional overlay from fig7 four-line CSV (row i = pocket_id i):
  SiteMap=circle, P2Rank=triangle, FPocket=diamond.

Saves text + ``_notext`` PNGs/PDFs (strip logic as evaluate_pocket_quality).

Usage:
  /home/user/anaconda3/envs/diffdynamic/bin/python3 \\
    scripts/plot_stacked_layers_100_nature.py \\
    --csv pocket_quality_vis/jsdpt3010_run8_latest_cavity/evaluation_records.csv \\
    --tool-csv fig7/pocketeval1.csv

For a SiteMap-only overlay, add ``--tools sitemap``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from utils.pocket_vis_nature import FONT, GRID, NEUTRAL, apply_nature_axes  # noqa: E402
from evaluate_pocket_quality import _strip_figure_text  # noqa: E402

# ColorBrewer RdBu diverging — muted (bar = background, low chroma)
RDBU = {
    "blue_pale": "#EAF2F8",
    "blue_soft": "#B8D4E8",
    "blue_mid": "#7AAFD0",
    "blue_dark": "#4A86B8",
    "orange_pale": "#FBE8D8",
    "orange_soft": "#F2C8A8",
    "orange_mid": "#E8A878",
    "orange_deep": "#D88458",
    "orange_dark": "#C06040",
}

SHADE_HEX = {
    "score_a": RDBU["blue_pale"],
    "score_b": RDBU["blue_soft"],
    "score_c": RDBU["blue_dark"],
    "score_d": RDBU["orange_pale"],
    "score_e": RDBU["orange_soft"],
    "score_f": RDBU["orange_mid"],
    "score_g": RDBU["orange_deep"],
    "score_h": RDBU["orange_dark"],
}

# Four-line tool markers — deeper, saturated (pop above muted bars)
TOOL_STYLE = {
    "sitemap": {"marker": "o", "color": "#1F7A1F", "label": "SiteMap"},
    "p2rank": {"marker": "^", "color": "#E8950A", "label": "P2Rank"},
    "fpocket": {"marker": "D", "color": "#7A4FB0", "label": "FPocket"},
}


def _hex_to_rgb(h: str):
    h = h.lstrip("#")
    return tuple(int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4))


def load_tool_scores(tool_csv: Path) -> pd.DataFrame:
    """Load fig7/pocketeval1.csv; pocket_id = row index 0..n-1."""
    raw = pd.read_csv(tool_csv)
    # flexible column names
    colmap = {}
    lower = {c.lower(): c for c in raw.columns}
    for key, aliases in {
        "p2rank": ("p2rank", "p2", "p2_rank", "p2rank_raw"),
        "fpocket": ("fpocket", "fp", "fp_score", "fpocket_raw"),
        "sitemap": ("sitemap", "sitemap_dscore", "sitemap_combined_raw", "sitemap_score"),
    }.items():
        for a in aliases:
            if a in lower:
                colmap[key] = lower[a]
                break
        if key not in colmap:
            raise ValueError(f"{tool_csv}: missing column for {key} (tried {aliases})")

    out = pd.DataFrame({
        "pocket_id": np.arange(len(raw), dtype=int),
        "p2rank": pd.to_numeric(raw[colmap["p2rank"]], errors="coerce"),
        "fpocket": pd.to_numeric(raw[colmap["fpocket"]], errors="coerce"),
        "sitemap": pd.to_numeric(raw[colmap["sitemap"]], errors="coerce"),
    })
    return out


def align_tools_to_sorted(df_sorted: pd.DataFrame, tools: pd.DataFrame) -> pd.DataFrame:
    """Reindex tool scores to match df_sorted pocket_id order."""
    t = tools.set_index("pocket_id")
    aligned = t.reindex(df_sorted["pocket_id"].astype(int).values)
    return aligned.reset_index(drop=True)


def build_figure(
    df: pd.DataFrame,
    title: str,
    tools_aligned: Optional[pd.DataFrame] = None,
    tool_keys: tuple[str, ...] = ("sitemap", "p2rank", "fpocket"),
    raw_tool_scores: bool = False,
):
    layers = [
        ("Ligand (A/B/C)", ["score_a", "score_b", "score_c"], "s_ligand"),
        ("Compat (D/E)", ["score_d", "score_e"], "s_compatibility"),
        ("Pocket (F/G/H)", ["score_f", "score_g", "score_h"], "s_pocket"),
    ]
    dim_labels = {f"score_{x}": x.upper() for x in "abcdefgh"}
    shade_map = {c: _hex_to_rgb(h) for c, h in SHADE_HEX.items()}

    n = len(df)
    fig_h = max(12.0, 0.20 * n + 2.2)
    fig, ax = plt.subplots(figsize=(9.2, fig_h), dpi=180)
    y = np.arange(n)
    lefts = np.zeros(n, dtype=float)
    legend_handles, seen = [], set()

    ssum = np.maximum(
        (df["s_ligand"] + df["s_compatibility"] + df["s_pocket"]).astype(float).values,
        1e-12,
    )
    overall = df["overall_score"].astype(float).values

    for layer_name, cols, s_col in layers:
        S = df[s_col].astype(float).values
        layer_width = overall * (S / ssum)
        sub = df[cols].astype(float).fillna(0.0).values
        sub_sum = sub.sum(axis=1)
        sub_sum = np.where(sub_sum > 1e-12, sub_sum, 1.0)
        for j, c in enumerate(cols):
            widths = layer_width * (sub[:, j] / sub_sum)
            ax.barh(
                y,
                widths,
                left=lefts,
                height=0.72,
                color=shade_map[c],
                edgecolor="white",
                linewidth=0.25,
                align="center",
                zorder=2,
            )
            lefts = lefts + widths
            if c not in seen:
                legend_handles.append(
                    Patch(
                        facecolor=shade_map[c],
                        edgecolor="#B0B0B0",
                        linewidth=0.4,
                        label=f"{dim_labels[c]} · {layer_name.split()[0]}",
                    )
                )
                seen.add(c)

    tool_legend = []
    marker_values = []
    if tools_aligned is not None and len(tools_aligned) == n:
        scatter_kw = dict(
            s=160,
            alpha=1.0,
            zorder=6,
            edgecolors="white",
            linewidths=1.4,
        )
        for key in tool_keys:
            st = TOOL_STYLE[key]
            vals = tools_aligned[key].astype(float).values
            mask = np.isfinite(vals)
            if not np.any(mask):
                continue
            marker_values.extend(vals[mask].tolist())
            ax.scatter(
                vals[mask],
                y[mask],
                marker=st["marker"],
                c=st["color"],
                label=st["label"],
                **scatter_kw,
            )
            tool_legend.append(
                Line2D(
                    [0],
                    [0],
                    marker=st["marker"],
                    color="w",
                    markerfacecolor=st["color"],
                    markeredgecolor="white",
                    markeredgewidth=1.4,
                    markersize=14,
                    linestyle="None",
                    label=st["label"],
                )
            )

    ylabels = [
        f"{int(pid)}  ({ov:.3f})"
        for pid, ov in zip(df["pocket_id"], df["overall_score"])
    ]
    ax.set_yticks(y)
    ax.set_yticklabels(ylabels, fontsize=FONT["annot"], color=NEUTRAL)
    ax.invert_yaxis()
    marker_mode = "raw marker values" if raw_tool_scores else "marker scores"
    ax.set_xlabel(
        "DD-eval bar (0–1); "
        + marker_mode
        + " = "
        + " / ".join(TOOL_STYLE[key]["label"] for key in tool_keys),
        fontsize=FONT["label"],
        color=NEUTRAL,
    )
    if marker_values:
        marker_min = min(marker_values)
        marker_max = max(marker_values)
        x_min = min(0.0, marker_min)
        x_max = max(1.0, marker_max)
        pad = max(0.02, 0.04 * (x_max - x_min))
        ax.set_xlim(x_min - pad, x_max + pad)
    else:
        ax.set_xlim(0, 1.02)
    if marker_values and min(marker_values) < 0:
        ax.axvline(0.0, color="#777777", ls="-", lw=0.7, zorder=1, alpha=0.8)
    ax.set_ylim(n - 0.5, -0.5)
    ax.axvline(0.70, color="#A0A0A0", ls="--", lw=0.7, zorder=1, alpha=0.75)
    ax.axvline(0.45, color="#B8B8B8", ls=":", lw=0.65, zorder=1, alpha=0.7)
    ax.text(
        0.70, -1.15, "high ≥ 0.70", ha="center", va="bottom",
        fontsize=FONT["annot"], color="#777777",
    )
    ax.text(
        0.45, -1.15, "med ≥ 0.45", ha="center", va="bottom",
        fontsize=FONT["annot"], color="#999999",
    )
    ax.set_title(title, fontsize=FONT["title"], color=NEUTRAL, pad=8)
    apply_nature_axes(ax, grid_y=False, grid_x=True)
    ax.xaxis.grid(True, color=GRID, linewidth=0.55, zorder=0)

    leg = ax.legend(
        handles=legend_handles,
        loc="lower right",
        fontsize=FONT["annot"],
        frameon=True,
        fancybox=False,
        edgecolor="#D0D0D0",
        framealpha=0.95,
        title="Sub-scores",
        title_fontsize=FONT["annot"],
        bbox_to_anchor=(1.0, 0.0),
    )
    layer_patches = [
        Patch(facecolor=_hex_to_rgb(RDBU["blue_mid"]), label="Ligand (A/B/C)"),
        Patch(facecolor=_hex_to_rgb(RDBU["orange_soft"]), label="Compatibility (D/E)"),
        Patch(facecolor=_hex_to_rgb(RDBU["orange_deep"]), label="Pocket (F/G/H)"),
    ]
    if tool_legend:
        layer_patches = layer_patches + tool_legend
    ax.legend(
        handles=layer_patches,
        loc="upper right",
        fontsize=FONT["annot"],
        frameon=True,
        fancybox=False,
        edgecolor="#D0D0D0",
        title="Layers / tools",
        title_fontsize=FONT["annot"],
        bbox_to_anchor=(1.0, 1.0),
    )
    ax.add_artist(leg)
    fig.patch.set_facecolor("white")
    fig.tight_layout()
    return fig


def _strip_all_legends(fig):
    """Remove every legend, including those kept via ``ax.add_artist``."""
    from matplotlib.legend import Legend

    for lg in list(getattr(fig, "legends", []) or []):
        try:
            lg.remove()
        except Exception:
            pass
    for ax in fig.axes:
        leg = ax.get_legend()
        if leg is not None:
            try:
                leg.remove()
            except Exception:
                pass
        for child in list(ax.get_children()):
            if isinstance(child, Legend):
                try:
                    child.remove()
                except Exception:
                    pass


def save_with_notext(fig, path: Path, dpi: int = 200):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    print(f"Saved: {path}")

    notext = path.parent / f"{path.stem}_notext{path.suffix}"
    pos_bounds = [tuple(ax.get_position().bounds) for ax in fig.axes]
    _strip_figure_text(fig)
    _strip_all_legends(fig)
    if len(fig.axes) == len(pos_bounds):
        for ax, b in zip(fig.axes, pos_bounds):
            ax.set_position(b)
    try:
        fig.tight_layout()
    except Exception:
        pass
    if len(fig.axes) == len(pos_bounds):
        for ax, b in zip(fig.axes, pos_bounds):
            ax.set_position(b)
    fig.savefig(notext, dpi=dpi, bbox_inches="tight", facecolor="white")
    print(f"Saved: {notext}")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv",
        type=Path,
        default=REPO / "pocket_quality_vis/jsdpt3010_run8_latest_cavity/evaluation_records.csv",
    )
    ap.add_argument(
        "--tool-csv",
        type=Path,
        default=REPO / "fig7" / "pocketeval1.csv",
        help="Four-line CSV (row i = pocket_id i); SiteMap/P2Rank/FPocket overlay",
    )
    ap.add_argument(
        "--no-tools",
        action="store_true",
        help="Skip tool marker overlay",
    )
    ap.add_argument(
        "--tools",
        type=str,
        default="sitemap,p2rank,fpocket",
        help="Comma-separated marker tools to draw (default: sitemap,p2rank,fpocket)",
    )
    ap.add_argument(
        "--raw-tool-scores",
        action="store_true",
        help="Label marker values as raw scores and expand x-axis beyond [0, 1]",
    )
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--stem", type=str, default="stacked_layers_100_sorted_nature")
    args = ap.parse_args()

    out_dir = (args.out_dir or args.csv.parent).resolve()
    df = pd.read_csv(args.csv)
    df["pocket_id"] = df["pocket_id"].astype(int)
    df = df.sort_values("overall_score", ascending=True).reset_index(drop=True)

    try:
        tool_keys = tuple(
            key.strip().lower()
            for key in args.tools.split(",")
            if key.strip()
        )
    except AttributeError:
        tool_keys = ("sitemap", "p2rank", "fpocket")
    unknown_tools = set(tool_keys) - set(TOOL_STYLE)
    if unknown_tools:
        print(f"❌ unknown tools: {sorted(unknown_tools)}", file=sys.stderr)
        return 2

    tools_aligned = None
    if not args.no_tools:
        tool_path = args.tool_csv if args.tool_csv.is_absolute() else (REPO / args.tool_csv)
        if not tool_path.is_file():
            print(f"❌ tool CSV not found: {tool_path}", file=sys.stderr)
            return 2
        tools = load_tool_scores(tool_path)
        tools_aligned = align_tools_to_sorted(df, tools)
        print(
            f"Tools from {tool_path.name}: "
            f"SiteMap n={tools_aligned['sitemap'].notna().sum()}, "
            f"P2Rank n={tools_aligned['p2rank'].notna().sum()}, "
            f"FPocket n={tools_aligned['fpocket'].notna().sum()}"
        )

    marker_text = " / ".join(TOOL_STYLE[key]["label"] for key in tool_keys) or "none"
    title = (
        "jsdpt3010 · 100 pockets · 3-layer + tools (blue–orange)\n"
        f"Bar = overall; marker values = {marker_text}"
        + (" (raw)" if args.raw_tool_scores else "")
        + "  |  low → high (top → bottom)"
    )

    fig = build_figure(
        df,
        title=title,
        tools_aligned=tools_aligned,
        tool_keys=tool_keys,
        raw_tool_scores=args.raw_tool_scores,
    )
    save_with_notext(fig, out_dir / f"{args.stem}.png", dpi=200)

    fig2 = build_figure(
        df,
        title=title,
        tools_aligned=tools_aligned,
        tool_keys=tool_keys,
        raw_tool_scores=args.raw_tool_scores,
    )
    pdf = out_dir / f"{args.stem}.pdf"
    fig2.savefig(pdf, bbox_inches="tight", facecolor="white")
    print(f"Saved: {pdf}")
    pos_bounds = [tuple(ax.get_position().bounds) for ax in fig2.axes]
    _strip_figure_text(fig2)
    _strip_all_legends(fig2)
    if len(fig2.axes) == len(pos_bounds):
        for ax, b in zip(fig2.axes, pos_bounds):
            ax.set_position(b)
    notext_pdf = out_dir / f"{args.stem}_notext.pdf"
    fig2.savefig(notext_pdf, bbox_inches="tight", facecolor="white")
    print(f"Saved: {notext_pdf}")
    plt.close(fig2)

    fig3 = build_figure(
        df,
        title=title,
        tools_aligned=tools_aligned,
        tool_keys=tool_keys,
        raw_tool_scores=args.raw_tool_scores,
    )
    save_with_notext(fig3, out_dir / "stacked_layers_100_sorted.png", dpi=200)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
