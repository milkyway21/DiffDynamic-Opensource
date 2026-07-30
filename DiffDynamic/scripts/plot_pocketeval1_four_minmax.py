#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pocketeval1 四列散点：仅作图时按列 min–max → [0,1]，不改 CSV/XLSX 原始数据。

相对 scripts/plot_triple_scatter_from_csv.py 的独立入口（不覆盖旧脚本/旧图）。
默认读 fig7/pocketeval1.csv，写出 pics/pocketeval1_four_minmax.png。

示例：

  conda run -n diffdynamic python3 scripts/plot_pocketeval1_four_minmax.py

  python3 scripts/plot_pocketeval1_four_minmax.py \\
    --csv fig7/pocketeval1.csv \\
    --out pics/pocketeval1_four_minmax.png
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import warnings
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_plot_mod():
    path = REPO_ROOT / "scripts" / "plot_triple_scatter_from_csv.py"
    spec = importlib.util.spec_from_file_location("plot_triple_scatter_from_csv", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser(
        description="四列 min–max 归一化展示（只影响作图，不写回数据文件）"
    )
    ap.add_argument(
        "--csv",
        type=Path,
        default=REPO_ROOT / "fig7" / "pocketeval1.csv",
        help="输入 CSV（只读）",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "pics" / "pocketeval1_four_minmax.png",
        help="输出图路径（默认新文件名，不覆盖 pocketeval1_four.png）",
    )
    ap.add_argument(
        "--sort-by",
        choices=("ddeval", "sitemap"),
        default="ddeval",
        help="横轴排序键（默认按 this_work/ddeval 升序）",
    )
    ap.add_argument("--title", type=str, default="", help="可选图标题")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument(
        "--no-bare-out",
        action="store_true",
        help="不额外保存无标题的 _bare 图",
    )
    args = ap.parse_args()

    csv_path = args.csv if args.csv.is_absolute() else (REPO_ROOT / args.csv)
    out_path = args.out if args.out.is_absolute() else (REPO_ROOT / args.out)
    if not csv_path.is_file():
        print(f"❌ 找不到 CSV: {csv_path}", file=sys.stderr)
        return 2

    mod = _load_plot_mod()
    if mod.plt is None or mod.pd is None:
        print("❌ 需要 matplotlib 与 pandas", file=sys.stderr)
        return 2

    df = mod.pd.read_csv(csv_path, encoding="utf-8")
    ddeval, p2, fp, sm, pocket_ids = mod.load_quad_scores_dataframe(df)
    n = len(pocket_ids)
    if n == 0:
        print("❌ CSV 无有效行", file=sys.stderr)
        return 2
    if not np.any(np.isfinite(fp)):
        print("❌ 需要 FPocket 列（如 fpocket）", file=sys.stderr)
        return 2

    # 横轴顺序仍按原始 ddeval（归一化前）升序，与旧四列图一致
    sort_key = sm if args.sort_by == "sitemap" else ddeval
    sort_label = "SiteMap ↑" if args.sort_by == "sitemap" else "ddeval ↑"
    ddeval, p2, fp, sm, pocket_ids = mod.order_rows_by_ddeval_ascending_quad(
        ddeval, p2, fp, sm, pocket_ids, sort_key=sort_key
    )

    # 仅展示：各列独立 min–max → [0,1]；NaN 保持缺失
    ddeval_n = mod._minmax_01(np.where(np.isfinite(ddeval), ddeval, np.nan))
    p2_n = mod._minmax_01(np.where(np.isfinite(p2), p2, np.nan))
    fp_n = mod._minmax_01(np.where(np.isfinite(fp), fp, np.nan))
    sm_n = mod._minmax_01(np.where(np.isfinite(sm), sm, np.nan))

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message=r"Glyph .+ missing from (current )?font",
        )
    except ImportError:
        print("❌ 需要 matplotlib", file=sys.stderr)
        return 2

    x = np.arange(n, dtype=np.float64)
    colors = {
        "ddeval": "#1f77b4",
        "p2": "#FFC107",
        "sm": "#2ca02c",
        "fp": "#9467bd",
    }

    fig_w = max(14, n * 0.12) * 1.2
    fig_h = 5.5 * 0.92
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.subplots_adjust(top=0.86)
    mod.add_quad_scatter_vlines_and_smooth_trends(ax, x, ddeval_n, p2_n, fp_n, sm_n, colors)
    pt_kw = dict(s=52, alpha=0.55, zorder=3, edgecolors="white", linewidths=0.9)
    h_de = ax.scatter(x, ddeval_n, c=colors["ddeval"], label="ddeval", **pt_kw)
    h_p2 = ax.scatter(x, p2_n, c=colors["p2"], label="P2Rank", **pt_kw)
    h_fp = ax.scatter(x, fp_n, c=colors["fp"], label="FPocket", **pt_kw)
    h_sm = ax.scatter(x, sm_n, c=colors["sm"], label="SiteMap", **pt_kw)

    ax.set_xlabel(f"Pocket id (row order or data_id; sorted by {sort_label})")
    ax.set_ylabel("Min-max over pockets → [0,1] (1 = batch max per series)")
    ax.set_xticks(x)
    if n <= 40:
        ax.set_xticklabels(pocket_ids, rotation=90, fontsize=8)
    else:
        step = max(1, n // 20)
        ax.set_xticks(x[::step])
        ax.set_xticklabels([pocket_ids[i] for i in range(0, n, step)], rotation=0, fontsize=8)

    ax.grid(True, axis="y", alpha=0.28)
    ax.legend(
        [h_de, h_p2, h_fp, h_sm],
        ["ddeval", "P2Rank", "FPocket", "SiteMap"],
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
        fontsize=8,
        frameon=True,
    )
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(0.0, 1.0)

    title = args.title.strip() or (
        f"CSV ({n} pockets): ddeval / P2Rank / FPocket / SiteMap"
        " (y = batch min-max per series; CSV unchanged)"
    )
    ax.set_title(title)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"📁 读取(只读): {csv_path}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=int(args.dpi), bbox_inches="tight")
    print(f"✅ 已保存: {out_path}")

    if not args.no_bare_out:
        bare_path = out_path.parent / f"{out_path.stem}_bare{out_path.suffix}"
        ax.set_title("")
        fig.tight_layout()
        fig.savefig(bare_path, dpi=int(args.dpi), bbox_inches="tight")
        print(f"✅ 已保存: {bare_path}")

    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
