#!/usr/bin/env python3
"""
三方法散点图：横向对比口袋评估方法

方法:
  ddeval (蓝)  — 本项目 evaluate_pocket_quality 的 overall_score
  P2Rank (黄)  — 口袋预测概率
  SiteMap (绿) — Schrödinger SiteMap 的 Dscore / combined 分

用法:
  python3 scripts/plot_testset_pocket_triple_scatter.py \
    --scores-xlsx docs/pocketeval1.xlsx \
    --output-dir pocket_quality_vis/tps

输入 Excel 格式 (pocketeval1.xlsx):
  必须列: data_id, ddeval
  可选列: p2rank, sitemap (缺失时该方法不绘制)
"""

import argparse
import sys
import os
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ── 样式常量 ──────────────────────────────────────────────────────────────────
COLOR_DDEVAL = '#2196F3'   # 蓝
COLOR_P2RANK = '#FFC107'   # 黄
COLOR_FPOCKET = '#FF7043'  # 橙
COLOR_SITEMAP = '#4CAF50'  # 绿

LABEL_DDEVAL = 'DiffDynamic (this work)'
LABEL_P2RANK = 'P2Rank'
LABEL_FPOCKET = 'FPocket'
LABEL_SITEMAP = 'SiteMap'

MARKER_SIZE = 36
MARKER_ALPHA = 0.75
MARKER_EDGE = '#333333'
MARKER_EDGE_WIDTH = 0.5


def load_scores(xlsx_path):
    """加载评分 Excel，返回 DataFrame。"""
    df = pd.read_excel(xlsx_path)
    required = ['data_id', 'ddeval']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"缺少必要列: {col}")
    return df.sort_values('data_id').reset_index(drop=True)


def make_scatter(df, output_dir, show_text=False):
    """
    生成三方法散点图。

    X 轴: data_id (口袋编号 0-99)
    Y 轴: 评分 (0-1)

    每个方法用不同颜色的散点，同一 data_id 的三个点略微偏移避免重叠。
    """
    n = len(df)
    x = df['data_id'].values.astype(float)

    # 方法数据
    methods = []
    if 'ddeval' in df.columns:
        methods.append(('ddeval', df['ddeval'].values, COLOR_DDEVAL, LABEL_DDEVAL))
    if 'p2rank' in df.columns:
        methods.append(('p2rank', df['p2rank'].values, COLOR_P2RANK, LABEL_P2RANK))
    if 'fpocket' in df.columns:
        methods.append(('fpocket', df['fpocket'].values, COLOR_FPOCKET, LABEL_FPOCKET))
    if 'sitemap' in df.columns:
        methods.append(('sitemap', df['sitemap'].values, COLOR_SITEMAP, LABEL_SITEMAP))

    if not methods:
        print("没有可绘制的方法数据")
        return

    n_methods = len(methods)
    offsets = np.linspace(-0.25, 0.25, n_methods) if n_methods > 1 else [0.0]

    # ── 主图 ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(16, 6))

    for i, (key, scores, color, label) in enumerate(methods):
        x_off = x + offsets[i]
        valid = ~np.isnan(scores)
        ax.scatter(
            x_off[valid], scores[valid],
            s=MARKER_SIZE, c=color, alpha=MARKER_ALPHA,
            edgecolors=MARKER_EDGE, linewidths=MARKER_EDGE_WIDTH,
            marker='o', label=label, zorder=3,
        )
        if show_text:
            for xi, yi in zip(x_off[valid], scores[valid]):
                ax.annotate(f'{yi:.2f}', (xi, yi), fontsize=4, ha='center', va='bottom')

    # 均值线
    for key, scores, color, label in methods:
        mean_val = np.nanmean(scores)
        ax.axhline(y=mean_val, color=color, linestyle='--', alpha=0.4, linewidth=0.8)
        ax.text(n - 0.5, mean_val + 0.005, f'{label} μ={mean_val:.3f}',
                fontsize=7, color=color, ha='right', va='bottom')

    ax.set_xlabel('Pocket ID (data_id)', fontsize=11)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title('Triple Method Comparison: Pocket Quality Scores', fontsize=13, fontweight='bold')
    ax.set_xlim(-1, n)
    ax.set_ylim(-0.05, 1.08)
    ax.set_xticks(np.arange(0, n, 5))
    ax.tick_params(axis='both', labelsize=9)
    ax.grid(axis='y', alpha=0.3)
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)

    fig.tight_layout()

    # 保存
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    path_full = os.path.join(output_dir, f'triple_pocket_scatter_{ts}.png')
    fig.savefig(path_full, dpi=200, bbox_inches='tight')
    print(f'保存: {path_full}')

    # 无文字版 (bare)
    ax.set_title('')
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.legend_.remove() if ax.legend_ else None
    # 移除均值标注
    for txt in ax.texts:
        txt.remove()

    path_bare = os.path.join(output_dir, f'triple_pocket_scatter_{ts}_bare.png')
    fig.savefig(path_bare, dpi=200, bbox_inches='tight')
    print(f'保存: {path_bare}')

    plt.close(fig)

    # ── 统计表 ────────────────────────────────────────────────────────────────
    print(f'\n{"="*60}')
    print(f'  方法对比统计 (n={n} pockets)')
    print(f'{"="*60}')
    print(f'  {"Method":<20} {"Mean":>6} {"Std":>6} {"Min":>6} {"Max":>6} {"Median":>6}')
    print(f'  {"-"*52}')
    for key, scores, color, label in methods:
        valid = scores[~np.isnan(scores)]
        if len(valid) > 0:
            print(f'  {label:<20} {np.mean(valid):>6.3f} {np.std(valid):>6.3f} '
                  f'{np.min(valid):>6.3f} {np.max(valid):>6.3f} {np.median(valid):>6.3f}')

    # 相关性 (如果有多方法)
    if n_methods >= 2:
        print(f'\n  方法间相关性 (Pearson r):')
        for i in range(n_methods):
            for j in range(i + 1, n_methods):
                key_i, scores_i, _, label_i = methods[i]
                key_j, scores_j, _, label_j = methods[j]
                valid = ~(np.isnan(scores_i) | np.isnan(scores_j))
                if valid.sum() > 2:
                    r = np.corrcoef(scores_i[valid], scores_j[valid])[0, 1]
                    print(f'    {label_i} vs {label_j}: r={r:.3f}')


def main():
    parser = argparse.ArgumentParser(description='三方法散点图：口袋评估横向对比')
    parser.add_argument('--scores-xlsx', type=str, required=True,
                        help='评分 Excel 文件路径 (含 data_id, ddeval 列)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='输出目录 (默认: pocket_quality_vis/tps_<timestamp>/)')
    parser.add_argument('--show-text', action='store_true',
                        help='在散点旁显示分数值')
    args = parser.parse_args()

    df = load_scores(args.scores_xlsx)
    print(f'加载 {len(df)} 个口袋评分 from {args.scores_xlsx}')

    output_dir = args.output_dir
    if output_dir is None:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = f'pocket_quality_vis/tps_{ts}'

    make_scatter(df, output_dir, show_text=args.show_text)


if __name__ == '__main__':
    main()
