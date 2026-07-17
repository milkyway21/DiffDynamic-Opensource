#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
仅根据表格绘制散点图（样式与 plot_testset_pocket_triple_scatter 一致），**不跑评估、不读 split/pt**。
数据源二选一：**--csv** 或 **--xlsx**。默认三列：**ddeval / P2Rank / SiteMap**；加 **--four-metrics** 时四列：
**ddeval（含 this_work）/ P2Rank / FPocket / SiteMap**，横轴均按 ddeval 升序重排。

列名（不区分大小写；无 data_id 时用行号 0..n-1；空单元格 = 缺失）：

- **data_id** 或 **pocket_id**：横轴口袋编号。
- **ddeval**：``ddeval_raw`` / ``ddeval`` / **this_work** / ``overall_score`` 等。
- **p2rank**：``p2rank_raw``、``p2rank`` / ``p2`` 等。
- **SiteMap**：``sitemap_combined_raw`` / ``sitemap_dscore`` / ``sitemap`` 等。
- **FPocket**（仅 ``--four-metrics``）：``fpocket_raw`` / ``fpocket`` / ``fp_score`` 等。

示例：

  python3 third_party/plot_triple_scatter_from_csv.py \\
    --csv data/triple_scatter_100pockets_prefilled.csv \\
    --out-base pocket_quality_vis --out triple_from_csv.png

  python3 third_party/plot_triple_scatter_from_csv.py \\
    --xlsx outputs/triple_74_valid_sites_scores.xlsx \\
    --out-base pocket_quality_vis --out triple_from_xlsx.png

  # prefilled CSV 常无 sitemap 列：用汇总表并入 combined_plot（绿色点）+ 仅 74 个 pocket_id：
  python3 third_party/plot_triple_scatter_from_csv.py \\
    --csv data/triple_scatter_100pockets_prefilled.csv \\
    --merge-sitemap-xlsx outputs/sitemap_valid_sites_min1.xlsx \\
    --only-data-ids-file outputs/only_data_ids_74.txt \\
    --out-base outputs --no-tps-dir --out triple_74_with_sitemap.png

  python3 third_party/plot_triple_scatter_from_csv.py \\
    --csv my_scores.csv --minmax-pockets \\
    --out /tmp/triple.png --no-tps-dir

  # 四列（ddeval / P2Rank / FPocket / SiteMap），仍按 ddeval 升序排横轴；无 data_id 时用行号 0..n-1：
  python3 third_party/plot_triple_scatter_from_csv.py \\
    --xlsx docs/pocketeval1.xlsx --scores-sheet Sheet1 --four-metrics \\
    --out-base pics --no-tps-dir --out pocketeval1_four.png
"""

from __future__ import annotations

import argparse
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

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
    plt = None

try:
    import pandas as pd
except ImportError:
    pd = None


def _scale_theoretical_01(arr: np.ndarray, cap: float) -> np.ndarray:
    if cap <= 0 or not np.isfinite(cap):
        return np.full(np.asarray(arr).shape, np.nan, dtype=np.float64)
    a = np.asarray(arr, dtype=np.float64)
    out = np.full(a.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(a)
    out[finite] = np.clip(a[finite] / float(cap), 0.0, 1.0)
    return out


def order_rows_by_ddeval_ascending(
    ddeval: np.ndarray,
    p2: np.ndarray,
    sm: np.ndarray,
    pocket_ids: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """按 ddeval 原始分从低到高重排（NaN 排在最后）。"""
    key = np.asarray(ddeval, dtype=np.float64)
    order = np.argsort(np.nan_to_num(key, nan=np.inf), kind="stable")
    ddeval = np.asarray(ddeval, dtype=np.float64)[order]
    p2 = np.asarray(p2, dtype=np.float64)[order]
    sm = np.asarray(sm, dtype=np.float64)[order]
    pocket_ids = [pocket_ids[i] for i in order]
    return ddeval, p2, sm, pocket_ids


def order_rows_by_ddeval_ascending_quad(
    ddeval: np.ndarray,
    p2: np.ndarray,
    fp: np.ndarray,
    sm: np.ndarray,
    pocket_ids: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """按 ddeval 升序重排四列：ddeval → p2rank → fpocket → sitemap + 横轴标签。"""
    key = np.asarray(ddeval, dtype=np.float64)
    order = np.argsort(np.nan_to_num(key, nan=np.inf), kind="stable")
    ddeval = np.asarray(ddeval, dtype=np.float64)[order]
    p2 = np.asarray(p2, dtype=np.float64)[order]
    fp = np.asarray(fp, dtype=np.float64)[order]
    sm = np.asarray(sm, dtype=np.float64)[order]
    pocket_ids = [pocket_ids[i] for i in order]
    return ddeval, p2, fp, sm, pocket_ids


def _minmax_01(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float64)
    out = np.full(a.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(a)
    if not np.any(finite):
        return out
    lo = float(np.nanmin(a))
    hi = float(np.nanmax(a))
    if not np.isfinite(lo) or not np.isfinite(hi):
        return out
    if hi - lo < 1e-15:
        out[finite] = 0.5
        return out
    out[finite] = (a[finite] - lo) / (hi - lo)
    return out


def _smooth_trend_curve_xy(
    y: np.ndarray,
    x: np.ndarray,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    沿横轴顺序的柔和趋势线：先对 y 做高斯平滑，再三次样条在密网格上取样，避免折线过尖。
    需 scipy；失败时返回 (None, None)。
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    x = np.asarray(x, dtype=np.float64).ravel()
    n = len(x)
    if n < 3:
        return None, None
    good = np.isfinite(y)
    if np.count_nonzero(good) < 2:
        return None, None
    y_fill = y.copy()
    if not np.all(good):
        y_fill = np.interp(x, x[good], y[good])
    try:
        from scipy.interpolate import CubicSpline
        from scipy.ndimage import gaussian_filter1d
    except ImportError:
        return None, None
    sigma = float(np.clip(n / 18.0, 1.15, 5.0))
    y_g = gaussian_filter1d(y_fill, sigma=sigma, mode="nearest")
    cs = CubicSpline(x, y_g, bc_type="natural")
    n_dense = max(200, n * 5)
    xd = np.linspace(float(x[0]), float(x[-1]), n_dense)
    return xd, cs(xd)


def add_triple_scatter_vlines_and_smooth_trends(
    ax,
    x: np.ndarray,
    ddeval_n: np.ndarray,
    p2_n: np.ndarray,
    sm_n: np.ndarray,
    colors: dict[str, str],
) -> None:
    """每个横轴位置一条浅灰竖线；三组各一条沿口袋顺序的平滑趋势线（zorder 低于散点）。"""
    xv = np.asarray(x, dtype=np.float64).ravel()
    for xi in xv:
        ax.axvline(
            float(xi),
            color="#c8c8c8",
            linewidth=0.75,
            alpha=0.75,
            zorder=0,
        )
    series = (
        (ddeval_n, colors["ddeval"], "ddeval trend"),
        (p2_n, colors["p2"], "P2Rank trend"),
        (sm_n, colors["sm"], "SiteMap trend"),
    )
    for y_arr, c, lab in series:
        xd, yd = _smooth_trend_curve_xy(y_arr, x)
        if xd is None or yd is None:
            continue
        ax.plot(
            xd,
            yd,
            color=c,
            linestyle="-",
            linewidth=2.0,
            alpha=0.88,
            zorder=2,
            label=lab,
        )


def add_quad_scatter_vlines_and_smooth_trends(
    ax,
    x: np.ndarray,
    y_ddeval: np.ndarray,
    y_p2: np.ndarray,
    y_fp: np.ndarray,
    y_sm: np.ndarray,
    colors: dict[str, str],
) -> None:
    """竖线 + 四条平滑趋势线，顺序：ddeval → P2Rank → FPocket → SiteMap。"""
    xv = np.asarray(x, dtype=np.float64).ravel()
    for xi in xv:
        ax.axvline(
            float(xi),
            color="#c8c8c8",
            linewidth=0.75,
            alpha=0.75,
            zorder=0,
        )
    series = (
        (y_ddeval, colors["ddeval"]),
        (y_p2, colors["p2"]),
        (y_fp, colors["fp"]),
        (y_sm, colors["sm"]),
    )
    for y_arr, c in series:
        xd, yd = _smooth_trend_curve_xy(y_arr, x)
        if xd is None or yd is None:
            continue
        ax.plot(
            xd,
            yd,
            color=c,
            linestyle="-",
            linewidth=2.0,
            alpha=0.88,
            zorder=2,
            label="_nolegend_",
        )


def _pick_column(df: "pd.DataFrame", names: tuple[str, ...]) -> Optional[str]:
    for n in names:
        if n in df.columns:
            return n
    return None


def _series_numeric(df: "pd.DataFrame", col: Optional[str]) -> np.ndarray:
    n = len(df)
    if col is None:
        return np.full(n, np.nan, dtype=np.float64)
    s = pd.to_numeric(df[col], errors="coerce")
    return s.astype(np.float64).values


def _normalize_columns(df: "pd.DataFrame") -> "pd.DataFrame":
    out = df.copy()
    out.columns = [str(c).strip().lower().lstrip("\ufeff") for c in out.columns]
    return out


def load_scores_dataframe(df: "pd.DataFrame") -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    if pd is None:
        raise RuntimeError("需要 pandas")
    df = _normalize_columns(df)

    id_col = _pick_column(df, ("data_id", "pocket_id", "id", "pocket"))
    if id_col is None:
        # 无口袋 ID 时按行序 0..n-1（与仅含分数列的汇总表一致；缺失分数仍保留横轴位）
        df = df.copy()
        df["data_id"] = np.arange(len(df), dtype=int)
        id_col = "data_id"

    ddeval_c = _pick_column(
        df,
        (
            "ddeval_raw",
            "ddeval",
            "this_work",
            "overall_score",
            "mine",
            "thiswork",
        ),
    )
    p2_c = _pick_column(df, ("p2rank_raw", "p2rank", "p2", "p2_rank"))
    sm_c = _pick_column(
        df,
        (
            "sitemap_combined_raw",
            "combined_plot",
            "sitemap_dscore",
            "dscore",
            "sitemap",
            "sitemap_score",
        ),
    )

    ddeval = _series_numeric(df, ddeval_c)
    p2 = _series_numeric(df, p2_c)
    sm = _series_numeric(df, sm_c)

    pocket_ids = [str(x).strip() for x in df[id_col].tolist()]
    return ddeval, p2, sm, pocket_ids


def load_quad_scores_dataframe(df: "pd.DataFrame") -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """返回顺序：ddeval, p2rank, fpocket, sitemap（与表列顺序一致）。"""
    if pd is None:
        raise RuntimeError("需要 pandas")
    df = _normalize_columns(df)

    id_col = _pick_column(df, ("data_id", "pocket_id", "id", "pocket"))
    if id_col is None:
        df = df.copy()
        df["data_id"] = np.arange(len(df), dtype=int)
        id_col = "data_id"

    ddeval_c = _pick_column(
        df,
        (
            "ddeval_raw",
            "ddeval",
            "this_work",
            "overall_score",
            "mine",
            "thiswork",
        ),
    )
    p2_c = _pick_column(df, ("p2rank_raw", "p2rank", "p2", "p2_rank"))
    sm_c = _pick_column(
        df,
        (
            "sitemap_combined_raw",
            "combined_plot",
            "sitemap_dscore",
            "dscore",
            "sitemap",
            "sitemap_score",
        ),
    )
    fp_c = _pick_column(
        df,
        (
            "fpocket_raw",
            "fpocket_plot_y",
            "fpocket",
            "fp_score_raw",
            "fp_score",
            "fp_plot_y",
            "fp",
        ),
    )

    ddeval = _series_numeric(df, ddeval_c)
    p2 = _series_numeric(df, p2_c)
    fp = _series_numeric(df, fp_c)
    sm = _series_numeric(df, sm_c)

    pocket_ids = [str(x).strip() for x in df[id_col].tolist()]
    return ddeval, p2, fp, sm, pocket_ids


def load_scores_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    if pd is None:
        raise RuntimeError("需要 pandas")
    df = pd.read_csv(path, encoding="utf-8")
    return load_scores_dataframe(df)


def load_scores_xlsx(path: Path, sheet: str = "scores") -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    if pd is None:
        raise RuntimeError("需要 pandas")
    try:
        import openpyxl  # noqa: F401
    except ImportError as e:
        raise RuntimeError("读取 xlsx 需要 openpyxl：pip install openpyxl") from e
    try:
        df = pd.read_excel(path, sheet_name=sheet, engine="openpyxl")
    except ValueError:
        # 常见：表名为 Sheet1 而非 scores
        if str(sheet).lower() == "scores":
            df = pd.read_excel(path, sheet_name=0, engine="openpyxl")
        else:
            raise
    return load_scores_dataframe(df)


def load_quad_scores_xlsx(path: Path, sheet: str = "scores") -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    if pd is None:
        raise RuntimeError("需要 pandas")
    try:
        import openpyxl  # noqa: F401
    except ImportError as e:
        raise RuntimeError("读取 xlsx 需要 openpyxl：pip install openpyxl") from e
    try:
        df = pd.read_excel(path, sheet_name=sheet, engine="openpyxl")
    except ValueError:
        if str(sheet).lower() == "scores":
            df = pd.read_excel(path, sheet_name=0, engine="openpyxl")
        else:
            raise
    return load_quad_scores_dataframe(df)


def _parse_data_ids_file(path: Path) -> set[int]:
    raw = path.read_text(encoding="utf-8")
    out: set[int] = set()
    for part in raw.replace("\n", ",").split(","):
        part = part.strip()
        if not part:
            continue
        out.add(int(part))
    return out


def _read_csv_dataframe(path: Path) -> "pd.DataFrame":
    if pd is None:
        raise RuntimeError("需要 pandas")
    return _normalize_columns(pd.read_csv(path, encoding="utf-8"))


def apply_sitemap_merge_and_id_filter(
    df: "pd.DataFrame",
    *,
    merge_sitemap_xlsx: Optional[Path] = None,
    merge_sitemap_sheet: Optional[str] = None,
    only_data_ids_file: Optional[Path] = None,
) -> "pd.DataFrame":
    """
    在已规范列名的 DataFrame 上：可选按 data_id 子集筛选；可选并入
    ``export_sitemap_summaries_to_xlsx`` / ``sitemap_valid_sites_min1`` 类表的 SiteMap 数值列。
    """
    if pd is None:
        raise RuntimeError("需要 pandas")
    df = df.copy()
    id_col = _pick_column(df, ("data_id", "pocket_id", "id", "pocket"))
    if id_col is None:
        raise ValueError("表格需含 data_id 或 pocket_id 列")
    if id_col != "data_id":
        df = df.rename(columns={id_col: "data_id"})
    df["data_id"] = pd.to_numeric(df["data_id"], errors="coerce")

    if only_data_ids_file is not None:
        if not only_data_ids_file.is_file():
            raise FileNotFoundError(f"未找到 --only-data-ids-file: {only_data_ids_file}")
        want = _parse_data_ids_file(only_data_ids_file)
        df = df[df["data_id"].isin(want)].copy()

    if merge_sitemap_xlsx is not None:
        try:
            import openpyxl  # noqa: F401
        except ImportError as e:
            raise RuntimeError("读取 xlsx 需要 openpyxl：pip install openpyxl") from e
        sheet = merge_sitemap_sheet if merge_sitemap_sheet is not None else 0
        smdf = pd.read_excel(merge_sitemap_xlsx, sheet_name=sheet, engine="openpyxl")
        smdf = _normalize_columns(smdf)
        if "pocket_id" not in smdf.columns:
            raise ValueError("合并用 SiteMap 表需含 pocket_id 列")
        smcol = None
        for cand in ("combined_plot", "sitemap_dscore"):
            if cand in smdf.columns:
                smcol = cand
                break
        if smcol is None:
            raise ValueError("合并用 SiteMap 表需含 combined_plot 或 sitemap_dscore")
        smdf["pocket_id"] = pd.to_numeric(smdf["pocket_id"], errors="coerce")
        sub = smdf[["pocket_id", smcol]].drop_duplicates(subset=["pocket_id"])
        sub = sub.rename(columns={smcol: "sitemap_dscore"})
        if "sitemap_dscore" in df.columns:
            df = df.drop(columns=["sitemap_dscore"])
        df = df.merge(sub, left_on="data_id", right_on="pocket_id", how="inner")
        df = df.drop(columns=["pocket_id"], errors="ignore")

    return df


def main() -> int:
    parser = argparse.ArgumentParser(
        description="从 CSV 或 xlsx 绘制三/四指标散点图（不跑评估、不读 split/pt）；"
        "四列模式加 --four-metrics。",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="含 data_id 与三列分数（列名别名见脚本文档）",
    )
    parser.add_argument(
        "--xlsx",
        type=Path,
        default=None,
        metavar="PATH.xlsx",
        help=(
            "plot_testset_pocket_triple_scatter.py --scores-xlsx 导出的表；"
            "默认读工作表「scores」"
        ),
    )
    parser.add_argument(
        "--scores-sheet",
        type=str,
        default="scores",
        metavar="NAME",
        help="与 --xlsx 联用：工作表名（默认 scores）",
    )
    parser.add_argument(
        "--merge-sitemap-xlsx",
        type=Path,
        default=None,
        metavar="PATH.xlsx",
        help=(
            "仅与 --csv 联用：从汇总表（如 outputs/sitemap_valid_sites_min1.xlsx）"
            "按 pocket_id 并入 combined_plot 或 sitemap_dscore 到 sitemap_dscore 列；"
            "inner 合并，只保留表中有 SiteMap 数值的行。"
        ),
    )
    parser.add_argument(
        "--merge-sitemap-sheet",
        type=str,
        default=None,
        metavar="NAME",
        help="与 --merge-sitemap-xlsx 联用：工作表名；省略则读第一个表",
    )
    parser.add_argument(
        "--only-data-ids-file",
        type=Path,
        default=None,
        metavar="PATH.txt",
        help="一行或多行逗号分隔的 data_id；先筛 CSV 再与 SiteMap 表合并（可选）",
    )
    parser.add_argument(
        "--out-base",
        type=Path,
        default=REPO_ROOT / "pocket_quality_vis",
        help="相对输出根目录（与主脚本一致）",
    )
    parser.add_argument(
        "--no-tps-dir",
        action="store_true",
        help="不创建 tps_<时间戳> 子目录",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("triple_scatter_from_csv.png"),
        help="输出 PNG 文件名或路径",
    )
    parser.add_argument("--minmax-pockets", action="store_true")
    parser.add_argument("--scale-theoretical-max", action="store_true")
    parser.add_argument(
        "--cap-ddeval",
        "--cap-this-work",
        type=float,
        default=1.0,
        dest="cap_ddeval",
    )
    parser.add_argument("--cap-p2", type=float, default=1.0)
    parser.add_argument("--cap-sitemap", type=float, default=1.0)
    parser.add_argument(
        "--four-metrics",
        action="store_true",
        help="绘制四列：ddeval（含 this_work）/ P2Rank / FPocket / SiteMap；横轴仍按 ddeval 升序",
    )
    parser.add_argument(
        "--cap-fpocket",
        type=float,
        default=1.0,
        help="与 --scale-theoretical-max 联用：FPocket 列理论上限（默认 1）",
    )
    parser.add_argument("--title", type=str, default="")
    parser.add_argument("--no-bare-out", action="store_true")

    args = parser.parse_args()

    if plt is None:
        print("❌ 需要 matplotlib")
        return 2

    if (args.csv is None) == (args.xlsx is None):
        print("❌ 请只指定其一：--csv PATH 或 --xlsx PATH（无需再跑评估）")
        return 2

    if args.merge_sitemap_xlsx is not None and args.csv is None:
        print("❌ --merge-sitemap-xlsx 仅能与 --csv 联用")
        return 2
    if args.only_data_ids_file is not None and args.csv is None:
        print("❌ --only-data-ids-file 仅能与 --csv 联用")
        return 2
    if args.four_metrics and args.merge_sitemap_xlsx is not None:
        print("❌ --four-metrics 当前不与 --merge-sitemap-xlsx 联用（请先在表里备好四列）")
        return 2

    merged_sitemap = False
    if args.csv is not None:
        src_path = args.csv.resolve()
        if not src_path.is_file():
            print(f"❌ 未找到 CSV: {src_path}")
            return 2
        try:
            df_in = _read_csv_dataframe(src_path)
            if args.merge_sitemap_xlsx is not None or args.only_data_ids_file is not None:
                mpath = args.merge_sitemap_xlsx.resolve() if args.merge_sitemap_xlsx else None
                df_in = apply_sitemap_merge_and_id_filter(
                    df_in,
                    merge_sitemap_xlsx=mpath,
                    merge_sitemap_sheet=args.merge_sitemap_sheet,
                    only_data_ids_file=(
                        args.only_data_ids_file.resolve()
                        if args.only_data_ids_file
                        else None
                    ),
                )
                merged_sitemap = mpath is not None
            if args.four_metrics:
                ddeval, p2, fp, sm, pocket_ids = load_quad_scores_dataframe(df_in)
            else:
                ddeval, p2, sm, pocket_ids = load_scores_dataframe(df_in)
        except Exception as e:
            print(f"❌ 读取/合并 CSV 失败: {e}")
            return 2
    else:
        src_path = args.xlsx.resolve()
        if not src_path.is_file():
            print(f"❌ 未找到 xlsx: {src_path}")
            return 2
        try:
            if args.four_metrics:
                ddeval, p2, fp, sm, pocket_ids = load_quad_scores_xlsx(src_path, args.scores_sheet)
            else:
                ddeval, p2, sm, pocket_ids = load_scores_xlsx(src_path, args.scores_sheet)
        except Exception as e:
            print(f"❌ 读取 xlsx 失败: {e}")
            return 2
    n = len(pocket_ids)

    if args.four_metrics:
        if not np.any(np.isfinite(fp)):
            print("❌ --four-metrics 需要表中存在 FPocket 数值列（如 fpocket）", file=sys.stderr)
            return 2
        ddeval, p2, fp, sm, pocket_ids = order_rows_by_ddeval_ascending_quad(
            ddeval, p2, fp, sm, pocket_ids
        )
    else:
        ddeval, p2, sm, pocket_ids = order_rows_by_ddeval_ascending(ddeval, p2, sm, pocket_ids)

    if args.four_metrics:
        if args.scale_theoretical_max:
            ddeval_n = _scale_theoretical_01(
                np.where(np.isfinite(ddeval), ddeval, np.nan), args.cap_ddeval
            )
            p2_n = _scale_theoretical_01(np.where(np.isfinite(p2), p2, np.nan), args.cap_p2)
            sm_n = _scale_theoretical_01(np.where(np.isfinite(sm), sm, np.nan), args.cap_sitemap)
            fp_n = _scale_theoretical_01(np.where(np.isfinite(fp), fp, np.nan), args.cap_fpocket)
            y_mode = "theory"
        elif args.minmax_pockets:
            ddeval_n = _minmax_01(np.where(np.isfinite(ddeval), ddeval, np.nan))
            p2_n = _minmax_01(np.where(np.isfinite(p2), p2, np.nan))
            sm_n = _minmax_01(np.where(np.isfinite(sm), sm, np.nan))
            fp_n = _minmax_01(np.where(np.isfinite(fp), fp, np.nan))
            y_mode = "minmax"
        else:
            ddeval_n, p2_n, fp_n, sm_n = ddeval, p2, fp, sm
            y_mode = "raw"
    elif args.scale_theoretical_max:
        ddeval_n = _scale_theoretical_01(
            np.where(np.isfinite(ddeval), ddeval, np.nan), args.cap_ddeval
        )
        p2_n = _scale_theoretical_01(np.where(np.isfinite(p2), p2, np.nan), args.cap_p2)
        sm_n = _scale_theoretical_01(np.where(np.isfinite(sm), sm, np.nan), args.cap_sitemap)
        y_mode = "theory"
    elif args.minmax_pockets:
        ddeval_n = _minmax_01(np.where(np.isfinite(ddeval), ddeval, np.nan))
        p2_n = _minmax_01(np.where(np.isfinite(p2), p2, np.nan))
        sm_n = _minmax_01(np.where(np.isfinite(sm), sm, np.nan))
        y_mode = "minmax"
    else:
        ddeval_n, p2_n, sm_n = ddeval, p2, sm
        y_mode = "raw"

    out_base = args.out_base.resolve()
    if args.out.is_absolute():
        args_out = args.out
    else:
        if args.no_tps_dir:
            args_out = out_base / args.out
        else:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            args_out = out_base / f"tps_{stamp}" / args.out

    x = np.arange(n, dtype=np.float64)
    colors = {
        "ddeval": "#1f77b4",
        "p2": "#FFC107",
        "sm": "#2ca02c",
        "fp": "#9467bd",
    }

    fig_w = max(14, n * 0.12) * 1.2
    fig_h = 5.5 * (0.92 if args.four_metrics else 0.85)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    if args.four_metrics:
        fig.subplots_adjust(top=0.86)
    if args.four_metrics:
        add_quad_scatter_vlines_and_smooth_trends(ax, x, ddeval_n, p2_n, fp_n, sm_n, colors)
    else:
        add_triple_scatter_vlines_and_smooth_trends(ax, x, ddeval_n, p2_n, sm_n, colors)
    pt_kw = dict(s=52, alpha=0.55, zorder=3, edgecolors="white", linewidths=0.9)
    sm_lbl = (
        "SiteMap ((SiteScore+Dscore)/div from table)"
        if merged_sitemap
        else "SiteMap (sitemap_dscore / combined)"
    )
    if args.four_metrics:
        h_de = ax.scatter(x, ddeval_n, c=colors["ddeval"], label="ddeval", **pt_kw)
        h_p2 = ax.scatter(x, p2_n, c=colors["p2"], label="P2Rank", **pt_kw)
        h_fp = ax.scatter(x, fp_n, c=colors["fp"], label="FPocket", **pt_kw)
        h_sm = ax.scatter(x, sm_n, c=colors["sm"], label="SiteMap", **pt_kw)
    else:
        ax.scatter(x, ddeval_n, c=colors["ddeval"], label="ddeval (overall_score)", **pt_kw)
        ax.scatter(x, p2_n, c=colors["p2"], label="P2Rank (ref. ligand pocket)", **pt_kw)
        ax.scatter(x, sm_n, c=colors["sm"], label=sm_lbl, **pt_kw)

    ax.set_xlabel(
        "Pocket id (row order or data_id; sorted by ddeval ↑)"
        if args.four_metrics
        else "Pocket data_id (from table, sorted by ddeval ↑)"
    )
    if args.four_metrics:
        if y_mode == "raw":
            ylab = "Raw score (ddeval / P2Rank / FPocket / SiteMap)"
        elif y_mode == "minmax":
            ylab = "Min-max over pockets → [0,1] (1 = batch max per series)"
        else:
            ylab = (
                f"raw ÷ cap (1 = cap: "
                f"{args.cap_ddeval:g} / {args.cap_p2:g} / {args.cap_fpocket:g} / {args.cap_sitemap:g})"
            )
    elif y_mode == "raw":
        ylab = "Raw score (units differ: ddeval / P2Rank / SiteMap)"
    elif y_mode == "minmax":
        ylab = "Min-max over pockets → [0,1] (1 = batch max per series)"
    else:
        ylab = (
            f"raw ÷ theoretical max (1 = cap: "
            f"{args.cap_ddeval:g} / {args.cap_p2:g} / {args.cap_sitemap:g})"
        )
    ax.set_ylabel(ylab)
    ax.set_xticks(x)
    if n <= 40:
        ax.set_xticklabels(pocket_ids, rotation=90, fontsize=8)
    else:
        step = max(1, n // 20)
        ax.set_xticks(x[::step])
        ax.set_xticklabels([pocket_ids[i] for i in range(0, n, step)], rotation=0, fontsize=8)

    ax.grid(True, axis="y", alpha=0.28)
    if args.four_metrics:
        ax.legend(
            [h_de, h_p2, h_fp, h_sm],
            ["ddeval", "P2Rank", "FPocket", "SiteMap"],
            ncol=4,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.08),
            fontsize=8,
            frameon=True,
        )
    else:
        ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim(-0.5, n - 0.5)
    if y_mode in ("minmax", "theory"):
        ax.set_ylim(0.0, 1.0)

    if args.title:
        ax.set_title(args.title)
    else:
        if y_mode == "raw":
            sub = " (raw scores)"
        elif y_mode == "minmax":
            sub = " (y = batch min-max per series)"
        else:
            sub = " (y = raw / theoretical max)"
        if args.xlsx is not None:
            kind = "XLSX"
        elif merged_sitemap:
            kind = "CSV+SiteMap"
        else:
            kind = "CSV"
        if args.four_metrics:
            ax.set_title(f"{kind} ({n} pockets): ddeval / P2Rank / FPocket / SiteMap{sub}")
        else:
            ax.set_title(f"{kind} ({n} pockets): ddeval / P2Rank / SiteMap{sub}")

    args_out.parent.mkdir(parents=True, exist_ok=True)
    print(f"📁 读取: {src_path}")
    print(f"📁 输出目录: {args_out.parent}")
    fig.tight_layout()
    fig.savefig(args_out, dpi=200, bbox_inches="tight")
    print(f"✅ 已保存: {args_out}")

    if not args.no_bare_out:
        bare_path = args_out.parent / f"{args_out.stem}_bare{args_out.suffix}"
        ax.set_title("")
        ax.set_xlabel("")
        ax.set_ylabel("")
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()
        ax.tick_params(axis="both", which="both", labelbottom=False, labelleft=False)
        ax.grid(False)
        ax.grid(True, axis="y", alpha=0.12)
        fig.tight_layout()
        fig.savefig(bare_path, dpi=200, bbox_inches="tight", pad_inches=0.02)
        print(f"✅ 已保存（无文字）: {bare_path}")

    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
