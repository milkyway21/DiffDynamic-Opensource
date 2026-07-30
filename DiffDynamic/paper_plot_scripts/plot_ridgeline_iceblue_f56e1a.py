#!/usr/bin/env python3
"""3D perspective ridgeline (joy) plots — Known actives + Generated.

Matches paper-style F/G/H panels: X=property, Y=group depth, Z=density,
with a 3D box + wall grids. Does NOT modify violin scripts.

Palette:
  Known (front / inner) : #C5DDF0 / #5A7FA0
  Generated (back / outer): #FBC9A6 / #F56E1A

Usage:
  conda activate diffdynamic
  python3 paper_plot_scripts/plot_ridgeline_iceblue_f56e1a.py \\
    --gen_csv .../generated_dock_qed_sa_n100.csv \\
    --act .../active_dock_qed_sa.csv \\
    --xlsx .../evaluation_results_n100_enriched.xlsx \\
    --out_dir .../plots --tag TAG
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from scipy.stats import gaussian_kde

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent  # DiffDynamic/
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore")

COLOR_KNOWN = "#C5DDF0"
COLOR_KNOWN_EDGE = "#5A7FA0"
COLOR_GEN = "#FBC9A6"
COLOR_GEN_EDGE = "#F56E1A"

# F/G/H-style primary trio (as in reference figure)
FGH_METRICS = [
    ("molwt", "MW distribution", "MW (Da)", (None, None)),
    ("logp", "LogP distribution", "LogP", (None, None)),
    ("tpsa", "TPSA distribution", "TPSA (Å²)", (None, None)),
]

CORE_METRICS = [
    ("vina_dock", "Vina Dock", "Vina Dock (kcal/mol)", (-15.0, 0.0)),
    ("qed", "QED", "QED", (0.0, 1.0)),
    ("sa", "SA Score", "SA Score", (0.0, 1.0)),
]

# 4×2 (ncols×nrows): row1 Vina/QED/SA/TPSA, row2 HBA/HBD/Atom/Lilly
GRID_4X2_METRICS = [
    ("vina_dock", "Vina Dock", "Vina Dock (kcal/mol)", (-15.0, 0.0)),
    ("qed", "QED", "QED", (0.0, 1.0)),
    ("sa", "SA Score", "SA Score", (0.0, 1.0)),
    ("tpsa", "TPSA", "TPSA (Å²)", (None, None)),
    ("hba", "H-Bond Acceptors", "HBA Count", (None, None)),
    ("hbd", "H-Bond Donors", "HBD Count", (None, None)),
    ("num_atoms", "Atom Count", "Heavy Atom Count", (None, None)),
    ("lilly_demerits", "Lilly Demerits", "Lilly Medchem Demerits", (None, None)),
]

EXPANDED_METRICS = [
    ("vina_dock", "Vina Dock", "Vina Dock (kcal/mol)", (-15.0, 0.0)),
    ("qed", "QED", "QED", (0.0, 1.0)),
    ("sa", "SA Score", "SA Score", (0.0, 1.0)),
    ("logp", "LogP", "LogP", (None, None)),
    ("molwt", "Molecular Weight", "MW (Da)", (None, None)),
    ("tpsa", "TPSA", "TPSA (Å²)", (None, None)),
    ("hba", "H-Bond Acceptors", "HBA Count", (None, None)),
    ("hbd", "H-Bond Donors", "HBD Count", (None, None)),
    ("num_atoms", "Atom Count", "Heavy Atom Count", (None, None)),
    ("rot_bonds", "Rotatable Bonds", "Rotatable Bonds", (None, None)),
    ("lipinski", "Lipinski Score", "Lipinski (RO5)", (0.0, 5.0)),
    ("lilly_demerits", "Lilly Demerits", "Lilly Medchem Demerits", (None, None)),
]


def _filter_vina(df: pd.DataFrame) -> pd.DataFrame:
    if "vina_dock" not in df.columns:
        return df.copy()
    v = pd.to_numeric(df["vina_dock"], errors="coerce")
    return df[(v.notna()) & (v < 0) & (v > -15)].copy()


def _kde_density(vals: np.ndarray, x_grid: np.ndarray) -> np.ndarray:
    """Return KDE density (not peak-normalized) on x_grid."""
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.zeros_like(x_grid)
    if vals.size == 1 or np.allclose(vals, vals[0]):
        bw = max(1e-3, 0.03 * (float(x_grid.max()) - float(x_grid.min()) + 1e-9))
        y = np.exp(-0.5 * ((x_grid - vals[0]) / bw) ** 2)
        # approximate density scale
        y = y / (bw * np.sqrt(2 * np.pi))
        return y
    try:
        kde = gaussian_kde(vals)
        return np.maximum(kde(x_grid), 0.0)
    except Exception:
        return np.zeros_like(x_grid)


def _resolve_xlim(known, generated, xlim):
    """Axis x-range (= floor width). Ridges stay inside this range."""
    x_lo, x_hi = xlim
    parts = []
    if known.size:
        parts.append(known)
    if generated.size:
        parts.append(generated)
    if not parts:
        return 0.0, 1.0
    concat = np.concatenate(parts)
    span = float(concat.max()) - float(concat.min()) + 1e-9
    pad = 0.10 * span
    auto_lo, auto_hi = float(concat.min()) - pad, float(concat.max()) + pad
    if x_lo is None:
        x_lo = auto_lo
    if x_hi is None:
        x_hi = auto_hi
    x_lo, x_hi = float(x_lo), float(x_hi)
    if x_hi <= x_lo:
        x_hi = x_lo + 1.0
    # Small pad so KDE tails die inside the axis box (ridges never leave it).
    s = x_hi - x_lo
    return x_lo - 0.08 * s, x_hi + 0.08 * s


def _add_ridge_3d(ax, x, z, y_pos, facecolor, edgecolor, *, alpha=0.78, zorder=1):
    """Filled density ridge on plane y=y_pos, strictly inside axis box."""
    x = np.asarray(x, dtype=float)
    z = np.asarray(z, dtype=float)
    verts = [
        list(
            zip(
                np.concatenate([x, x[::-1]]),
                np.full(2 * len(x), float(y_pos)),
                np.concatenate([np.zeros(len(x)), z[::-1]]),
            )
        )
    ]
    poly = Poly3DCollection(
        verts,
        facecolors=facecolor,
        edgecolors="none",
        alpha=alpha,
        linewidths=0,
        zorder=zorder,
    )
    ax.add_collection3d(poly)
    ax.plot(x, np.full_like(x, y_pos), z, color=edgecolor, lw=1.5, zorder=zorder + 1)
    ax.plot([x[0], x[-1]], [y_pos, y_pos], [0.0, 0.0], color=edgecolor, lw=1.2, zorder=zorder + 1)


def _draw_3d_box_grid(ax, x_lo, x_hi, y_lo, y_hi, z_hi, *, wall_y=None):
    """XY floor + XZ back wall only (two planes)."""
    if wall_y is None:
        wall_y = y_lo

    floor_edges = [
        [(x_lo, y_lo, 0), (x_hi, y_lo, 0)],
        [(x_hi, y_lo, 0), (x_hi, y_hi, 0)],
        [(x_hi, y_hi, 0), (x_lo, y_hi, 0)],
        [(x_lo, y_hi, 0), (x_lo, y_lo, 0)],
    ]
    ax.add_collection3d(Line3DCollection(floor_edges, colors="#888888", linewidths=0.9, alpha=0.95))
    nx, ny = 6, 4
    for i in range(nx + 1):
        xx = x_lo + (x_hi - x_lo) * i / nx
        ax.plot([xx, xx], [y_lo, y_hi], [0, 0], color="#CCCCCC", lw=0.45, alpha=0.75)
    for j in range(ny + 1):
        yy = y_lo + (y_hi - y_lo) * j / ny
        ax.plot([x_lo, x_hi], [yy, yy], [0, 0], color="#CCCCCC", lw=0.45, alpha=0.75)

    wall_edges = [
        [(x_lo, wall_y, 0), (x_hi, wall_y, 0)],
        [(x_hi, wall_y, 0), (x_hi, wall_y, z_hi)],
        [(x_hi, wall_y, z_hi), (x_lo, wall_y, z_hi)],
        [(x_lo, wall_y, z_hi), (x_lo, wall_y, 0)],
    ]
    ax.add_collection3d(Line3DCollection(wall_edges, colors="#888888", linewidths=0.9, alpha=0.95))
    nz = 4
    for i in range(nx + 1):
        xx = x_lo + (x_hi - x_lo) * i / nx
        ax.plot([xx, xx], [wall_y, wall_y], [0, z_hi], color="#DDDDDD", lw=0.45, alpha=0.75)
    for k in range(nz + 1):
        zz = z_hi * k / nz
        ax.plot([x_lo, x_hi], [wall_y, wall_y], [zz, zz], color="#DDDDDD", lw=0.45, alpha=0.75)


def _draw_panel_3d(
    ax,
    known: np.ndarray,
    generated: np.ndarray,
    xlim,
    *,
    show_text: bool,
    xlabel: str,
    title: str,
    panel_letter: str | None = None,
    fig=None,
):
    """Ridges inside the axis box; XY floor + XZ back wall fully visible."""
    known = np.asarray(known, dtype=float)
    generated = np.asarray(generated, dtype=float)
    known = known[np.isfinite(known)]
    generated = generated[np.isfinite(generated)]

    if known.size == 0 and generated.size == 0:
        ax.text2D(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
        return

    # --- Axis box (= floor + wall extent). Everything stays inside. ---
    x_lo, x_hi = _resolve_xlim(known, generated, xlim)
    x_grid = np.linspace(x_lo, x_hi, 220)
    z_k = _kde_density(known, x_grid) if known.size else np.zeros_like(x_grid)
    z_g = _kde_density(generated, x_grid) if generated.size else np.zeros_like(x_grid)
    z_hi = float(max(float(z_k.max()), float(z_g.max()), 1e-9)) * 1.28

    # Depth: both ridges strictly inside [y_lo, y_hi], inset from near/far edges
    # so mplot3d does not shear the near-side ridge off the viewport.
    y_lo, y_hi = 0.0, 1.0
    y_gen, y_known = 0.28, 0.62
    wall_y = y_lo  # back wall at far side

    # Limits EXACTLY match the two planes
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_zlim(0.0, z_hi)
    ax.view_init(elev=24, azim=75)
    # zoom out so all floor corners + full back wall are visible
    ax.set_box_aspect((1.35, 1.0, 0.72), zoom=0.65)

    _draw_3d_box_grid(ax, x_lo, x_hi, y_lo, y_hi, z_hi, wall_y=wall_y)

    # Invisible corner anchors = the 8 corners of the axis box
    _sc = ax.scatter(
        [x_lo, x_hi, x_lo, x_hi, x_lo, x_hi, x_lo, x_hi],
        [y_lo, y_lo, y_hi, y_hi, y_lo, y_lo, y_hi, y_hi],
        [0.0, 0.0, 0.0, 0.0, z_hi, z_hi, z_hi, z_hi],
        s=8,
        alpha=0.0,
        depthshade=False,
    )
    _sc.set_clip_on(False)

    if generated.size:
        _add_ridge_3d(ax, x_grid, z_g, y_gen, COLOR_GEN, COLOR_GEN_EDGE, alpha=0.80, zorder=2)
    if known.size:
        _add_ridge_3d(ax, x_grid, z_k, y_known, COLOR_KNOWN, COLOR_KNOWN_EDGE, alpha=0.82, zorder=3)

    # Do not let the 2D axes patch shear near-side 3D artists
    ax.set_clip_on(False)
    for art in list(ax.collections) + list(ax.lines):
        try:
            art.set_clip_on(False)
        except Exception:
            pass

    xticks = [float(v) for v in ax.get_xticks() if x_lo - 1e-9 <= float(v) <= x_hi + 1e-9]
    zticks = [float(v) for v in ax.get_zticks() if 0.0 <= float(v) <= z_hi + 1e-9]
    ax.set_axis_off()

    if show_text:
        x_mid = 0.5 * (x_lo + x_hi)
        ax.text(x_mid, y_hi + 0.02, -0.06 * z_hi, xlabel, fontsize=8, ha="center", va="top")
        ax.text(x_hi - 0.02 * (x_hi - x_lo), y_known, 0.04 * z_hi, "Known", fontsize=7, ha="right", va="bottom")
        ax.text(x_hi - 0.02 * (x_hi - x_lo), y_gen, 0.04 * z_hi, "Generated", fontsize=7, ha="right", va="bottom")
        ax.text(x_lo - 0.01 * (x_hi - x_lo), wall_y, 0.55 * z_hi, "Density", fontsize=8, ha="right", va="center")
        for xv in xticks:
            ax.text(xv, y_hi, -0.03 * z_hi, f"{xv:g}", fontsize=6.5, ha="center", va="top")
        for zv in zticks:
            ax.text(x_lo, wall_y, zv, f"{zv:g}", fontsize=6.5, ha="right", va="center")
        ax.text2D(0.5, 1.03, title, transform=ax.transAxes, fontsize=11, fontweight="medium", ha="center", va="bottom")
        if panel_letter:
            ax.text2D(-0.08, 0.98, panel_letter, transform=ax.transAxes, fontsize=13, fontweight="bold", va="top")



def _prepare_gen_df(gen_csv: Path, xlsx: Path | None) -> pd.DataFrame:
    import plot_generic_violin_expanded as pev

    if xlsx is not None and Path(xlsx).is_file():
        gen = pev.load_generated(xlsx)
        print(f"Loaded generated from xlsx: {xlsx} (n={len(gen)})")
        return gen

    df = pd.read_csv(gen_csv)
    out = pd.DataFrame(
        {
            "smiles": df["smiles"].astype(str) if "smiles" in df.columns else "",
            "vina_dock": pd.to_numeric(df.get("vina_dock"), errors="coerce"),
            "qed": pd.to_numeric(df.get("qed"), errors="coerce"),
            "sa": pd.to_numeric(df.get("sa"), errors="coerce"),
        }
    )
    for col in ("logp", "molwt"):
        if col in df.columns:
            out[col] = pd.to_numeric(df[col], errors="coerce")
    rd = pev.compute_rdkit_props(out["smiles"])
    for col in ("hba", "hbd", "rot_bonds", "logp", "molwt", "tpsa", "num_atoms", "ring_count", "lipinski"):
        if col not in out.columns or out[col].isna().all():
            out[col] = rd[col]
    if "lilly_demerit" in df.columns:
        out["lilly_demerits"] = pd.to_numeric(df["lilly_demerit"], errors="coerce")
    else:
        out["lilly_demerits"] = np.nan
    print(f"Loaded generated from csv: {gen_csv} (n={len(out)})")
    return out


def plot_ridgeline_3d_row(
    gen_df: pd.DataFrame,
    act_df: pd.DataFrame,
    metrics: list,
    out_path: Path,
    *,
    show_text: bool,
    figsize: tuple[float, float],
    n_known: int,
    n_gen: int,
    panel_letters: list[str] | None = None,
):
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 10,
            "figure.dpi": 200,
            "savefig.dpi": 200,
        }
    )
    n = len(metrics)
    fig = plt.figure(figsize=figsize)
    for i, (col, title, xlabel, xlim) in enumerate(metrics):
        ax = fig.add_subplot(1, n, i + 1, projection="3d")
        a = pd.to_numeric(act_df.get(col, pd.Series(dtype=float)), errors="coerce").dropna().to_numpy()
        g = pd.to_numeric(gen_df.get(col, pd.Series(dtype=float)), errors="coerce").dropna().to_numpy()
        letter = None
        if panel_letters and i < len(panel_letters):
            letter = panel_letters[i]
        _draw_panel_3d(
            ax,
            a,
            g,
            xlim,
            show_text=show_text,
            xlabel=xlabel,
            title=title,
            panel_letter=letter if show_text else None,
            fig=fig,
        )

    if show_text:
        fig.legend(
            handles=[
                Patch(
                    facecolor=COLOR_KNOWN,
                    edgecolor=COLOR_KNOWN_EDGE,
                    label=f"Known Actives (n={n_known})",
                ),
                Patch(
                    facecolor=COLOR_GEN,
                    edgecolor=COLOR_GEN_EDGE,
                    label=f"Generated, Vina -15~0 (n={n_gen})",
                ),
            ],
            loc="lower center",
            ncol=2,
            frameon=True,
            fancybox=False,
            edgecolor="#CCCCCC",
            bbox_to_anchor=(0.5, -0.02),
            fontsize=9,
        )
        fig.tight_layout(rect=(0, 0.05, 1, 1))
    else:
        fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {out_path}")


def plot_ridgeline_3d_grid(
    gen_df: pd.DataFrame,
    act_df: pd.DataFrame,
    metrics: list,
    out_path: Path,
    *,
    show_text: bool,
    nrows: int,
    ncols: int,
    figsize: tuple[float, float],
    n_known: int,
    n_gen: int,
):
    """Same panel size for text and no_text; no_text just omits glyphs.

    Fixed per-panel figsize (no bbox_inches='tight') keeps both versions
    the same overall size, with label-area margins reserved either way.
    """
    import io

    from matplotlib.ticker import NullFormatter
    from PIL import Image as PILImage

    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 9,
            "figure.dpi": 180,
            "savefig.dpi": 180,
        }
    )

    # Roomy panels + zoomed-out camera so near-right corner stays inside clip box.
    panel_w = min(4.4, float(figsize[0]) / ncols + 0.55)
    panel_h = min(3.8, float(figsize[1]) / nrows + 0.25)
    tiles: list[PILImage.Image] = []

    def _content_crop(im: PILImage.Image, pad: int = 18) -> PILImage.Image:
        """Crop white margins but keep a uniform pad around real pixels."""
        arr = np.asarray(im.convert("RGBA"))
        mask = (arr[:, :, :3] < 250).any(axis=2) | (arr[:, :, 3] < 250)
        if not mask.any():
            return im
        ys, xs = np.where(mask)
        x0 = max(0, int(xs.min()) - pad)
        y0 = max(0, int(ys.min()) - pad)
        x1 = min(im.width, int(xs.max()) + 1 + pad)
        y1 = min(im.height, int(ys.max()) + 1 + pad)
        return im.crop((x0, y0, x1, y1))

    for col, title, xlabel, xlim in metrics:
        fig = plt.figure(figsize=(panel_w, panel_h), facecolor="white")
        # Leave margin around axes so the full floor+wall projection stays visible
        ax = fig.add_axes([0.08, 0.08, 0.84, 0.84], projection="3d")
        a = pd.to_numeric(act_df.get(col, pd.Series(dtype=float)), errors="coerce").dropna().to_numpy()
        g = pd.to_numeric(gen_df.get(col, pd.Series(dtype=float)), errors="coerce").dropna().to_numpy()
        _draw_panel_3d(
            ax,
            a,
            g,
            xlim,
            show_text=show_text,
            xlabel=xlabel,
            title=title,
            panel_letter=None,
            fig=fig,
        )

        if not show_text:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            ax.xaxis.set_major_formatter(NullFormatter())
            ax.yaxis.set_major_formatter(NullFormatter())
            ax.zaxis.set_major_formatter(NullFormatter())
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])
            for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
                axis._axinfo["tick"]["inward_factor"] = 0
                axis._axinfo["tick"]["outward_factor"] = 0

        buf = io.BytesIO()
        # pad so the full floor+wall projection is not cropped at the figure edge
        fig.savefig(buf, format="png", dpi=180, facecolor="white", bbox_inches="tight", pad_inches=0.45)
        plt.close(fig)
        buf.seek(0)
        tiles.append(_content_crop(PILImage.open(buf).convert("RGBA"), pad=16))
    # Uniform tile size; scale-to-fit + center so smaller panels do not leave
    # a blank bottom-right corner from top-left alignment.
    tw = max(im.width for im in tiles)
    th = max(im.height for im in tiles)
    gap = 12
    canvas_w = ncols * tw + (ncols - 1) * gap
    legend_h = 64 if show_text else 0
    canvas_h = nrows * th + (nrows - 1) * gap + legend_h
    canvas = PILImage.new("RGBA", (canvas_w, canvas_h), (255, 255, 255, 255))

    for idx, im in enumerate(tiles):
        r, c = divmod(idx, ncols)
        scale = min(tw / max(im.width, 1), th / max(im.height, 1))
        nw = max(1, int(round(im.width * scale)))
        nh = max(1, int(round(im.height * scale)))
        im2 = im.resize((nw, nh), PILImage.Resampling.LANCZOS)
        pad = PILImage.new("RGBA", (tw, th), (255, 255, 255, 255))
        pad.paste(im2, ((tw - nw) // 2, (th - nh) // 2), im2)
        canvas.paste(pad, (c * (tw + gap), r * (th + gap)))

    if show_text:
        fig_l = plt.figure(figsize=(max(4.0, canvas_w / 180.0), legend_h / 180.0))
        fig_l.legend(
            handles=[
                Patch(
                    facecolor=COLOR_KNOWN,
                    edgecolor=COLOR_KNOWN_EDGE,
                    label=f"Known Actives (n={n_known})",
                ),
                Patch(
                    facecolor=COLOR_GEN,
                    edgecolor=COLOR_GEN_EDGE,
                    label=f"Generated (n={n_gen})",
                ),
            ],
            loc="center",
            ncol=2,
            frameon=True,
            fontsize=11,
        )
        buf = io.BytesIO()
        fig_l.savefig(buf, format="png", dpi=180, facecolor="white", bbox_inches="tight", pad_inches=0.10)
        plt.close(fig_l)
        buf.seek(0)
        leg = PILImage.open(buf).convert("RGBA")
        need_w = max(canvas_w, leg.width + 20)
        need_h = max(canvas_h, nrows * th + (nrows - 1) * gap + leg.height + 8)
        if need_w != canvas_w or need_h != canvas_h:
            new_c = PILImage.new("RGBA", (need_w, need_h), (255, 255, 255, 255))
            new_c.paste(canvas, ((need_w - canvas_w) // 2, 0))
            canvas = new_c
            canvas_w, canvas_h = need_w, need_h
        lx = (canvas_w - leg.width) // 2
        ly = nrows * th + (nrows - 1) * gap + 4
        canvas.paste(leg, (lx, ly), leg)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(out_path, format="PNG")
    print(f"saved {out_path}")



def main() -> int:
    ap = argparse.ArgumentParser(description="3D perspective ridgeline plots (Known + Generated)")
    ap.add_argument("--gen_csv", required=True)
    ap.add_argument("--act", required=True)
    ap.add_argument("--xlsx", default=None)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    import plot_generic_violin_expanded as pev

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag

    gen_raw = _prepare_gen_df(Path(args.gen_csv), Path(args.xlsx) if args.xlsx else None)
    act_raw = pev.load_actives(args.act)
    gen = _filter_vina(gen_raw)
    act = _filter_vina(act_raw)
    print(f"Filtered Vina (-15,0): gen {len(gen_raw)}→{len(gen)}, act {len(act_raw)}→{len(act)}")

    for show_text in (True, False):
        suffix = "" if show_text else "_no_text"

        # Primary: 4×2 ridgeline
        # row1: Vina / QED / SA / TPSA
        # row2: HBA / HBD / Atom Count / Lilly Demerits
        plot_ridgeline_3d_grid(
            gen,
            act,
            GRID_4X2_METRICS,
            out_dir / f"{tag}_ridgeline{suffix}.png",
            show_text=show_text,
            nrows=2,
            ncols=4,
            figsize=(13.5, 7.0),
            n_known=len(act),
            n_gen=len(gen),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
