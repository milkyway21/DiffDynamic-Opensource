#!/usr/bin/env python3
"""
口袋评估综合可视化：分布图 + 维度分解 + 相关性热图

输入: docs/pocketeval1.xlsx (含 data_id, ddeval, idea_a~h 列)
输出: pocket_quality_vis/eval_summary_<timestamp>/
  - score_distribution.png: ddeval 分布直方图
  - dimension_boxplot.png: 各维度箱线图
  - correlation_heatmap.png: 维度间相关性热图
  - dimension_radar.png: 雷达图 (各维度均值)
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
from matplotlib.patches import FancyBboxPatch
import matplotlib.gridspec as gridspec


# ── 颜色方案 ──────────────────────────────────────────────────────────────────
DIM_COLORS = {
    'idea_a': '#E53935',  # 红
    'idea_b': '#FB8C00',  # 橙
    'idea_c': '#FDD835',  # 黄
    'idea_d': '#43A047',  # 绿
    'idea_e': '#1E88E5',  # 蓝
    'idea_f': '#8E24AA',  # 紫
    'idea_g': '#00ACC1',  # 青
    'idea_h': '#6D4C41',  # 棕
}
DIM_LABELS = {
    'idea_a': 'A: Vina',
    'idea_b': 'B: Cluster',
    'idea_c': 'C: LE',
    'idea_d': 'D: Drug',
    'idea_e': 'E: Recon',
    'idea_f': 'F: Uniq',
    'idea_g': 'G: Size',
    'idea_h': 'H: Volume',
}


def load_data(xlsx_path):
    df = pd.read_excel(xlsx_path)
    return df.sort_values('data_id').reset_index(drop=True)


def plot_distribution(df, output_dir):
    """ddeval 分布直方图 + KDE"""
    fig, ax = plt.subplots(figsize=(8, 5))

    scores = df['ddeval'].values
    n = len(scores)

    # 直方图
    bins = np.linspace(0, 1, 25)
    ax.hist(scores, bins=bins, color='#2196F3', alpha=0.7, edgecolor='white', linewidth=0.5)

    # KDE
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(scores)
    x_kde = np.linspace(0, 1, 200)
    ax.plot(x_kde, kde(x_kde) * n * (bins[1] - bins[0]),
            color='#1565C0', linewidth=2, label='KDE')

    # 统计线
    mean_val = np.mean(scores)
    median_val = np.median(scores)
    ax.axvline(mean_val, color='#E53935', linestyle='--', linewidth=1.5, label=f'Mean={mean_val:.3f}')
    ax.axvline(median_val, color='#FB8C00', linestyle=':', linewidth=1.5, label=f'Median={median_val:.3f}')

    # 质量区间
    ax.axvspan(0.6, 1.0, alpha=0.08, color='green', label='High (≥0.6)')
    ax.axvspan(0.3, 0.6, alpha=0.08, color='orange', label='Medium (0.3-0.6)')
    ax.axvspan(0.0, 0.3, alpha=0.08, color='red', label='Low (<0.3)')

    ax.set_xlabel('Overall Score (ddeval)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(f'Pocket Quality Distribution (n={n})', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, loc='upper left')
    ax.set_xlim(0, 1)
    ax.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    path = os.path.join(output_dir, 'score_distribution.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    print(f'保存: {path}')
    plt.close(fig)


def plot_dimension_boxplot(df, output_dir):
    """各维度箱线图"""
    dim_keys = [k for k in DIM_COLORS if k in df.columns]
    if not dim_keys:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    data = [df[k].dropna().values for k in dim_keys]
    labels = [DIM_LABELS.get(k, k) for k in dim_keys]
    colors = [DIM_COLORS[k] for k in dim_keys]

    bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.6,
                    medianprops=dict(color='black', linewidth=1.5))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # 叠加散点
    for i, d in enumerate(data):
        x_jitter = np.random.normal(i + 1, 0.04, size=len(d))
        ax.scatter(x_jitter, d, s=12, alpha=0.5, color='black', zorder=3)

    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Per-Dimension Score Distribution', fontsize=14, fontweight='bold')
    ax.set_ylim(-0.05, 1.08)
    ax.axhline(0.6, color='green', linestyle='--', alpha=0.3, linewidth=0.8)
    ax.axhline(0.3, color='orange', linestyle='--', alpha=0.3, linewidth=0.8)
    ax.grid(axis='y', alpha=0.3)
    ax.tick_params(axis='x', rotation=15)

    fig.tight_layout()
    path = os.path.join(output_dir, 'dimension_boxplot.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    print(f'保存: {path}')
    plt.close(fig)


def plot_correlation_heatmap(df, output_dir):
    """维度间相关性热图"""
    dim_keys = [k for k in DIM_COLORS if k in df.columns]
    if len(dim_keys) < 2:
        return

    labels = [DIM_LABELS.get(k, k) for k in dim_keys]
    corr = df[dim_keys].corr()

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')

    # 标注
    for i in range(len(dim_keys)):
        for j in range(len(dim_keys)):
            val = corr.iloc[i, j]
            color = 'white' if abs(val) > 0.6 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=9, color=color, fontweight='bold' if i == j else 'normal')

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_title('Dimension Correlation Matrix', fontsize=14, fontweight='bold')

    fig.colorbar(im, ax=ax, shrink=0.8, label='Pearson r')
    fig.tight_layout()
    path = os.path.join(output_dir, 'correlation_heatmap.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    print(f'保存: {path}')
    plt.close(fig)


def plot_dimension_radar(df, output_dir):
    """雷达图：各维度均值"""
    dim_keys = [k for k in DIM_COLORS if k in df.columns]
    if not dim_keys:
        return

    labels = [DIM_LABELS.get(k, k) for k in dim_keys]
    means = [df[k].mean() for k in dim_keys]
    colors = [DIM_COLORS[k] for k in dim_keys]

    n_dims = len(dim_keys)
    angles = np.linspace(0, 2 * np.pi, n_dims, endpoint=False).tolist()
    means_plot = means + [means[0]]
    angles += [angles[0]]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    ax.fill(angles, means_plot, alpha=0.15, color='#2196F3')
    ax.plot(angles, means_plot, 'o-', color='#2196F3', linewidth=2, markersize=6)

    # 标注均值
    for i, (angle, mean) in enumerate(zip(angles[:-1], means)):
        ax.annotate(f'{mean:.2f}', xy=(angle, mean), fontsize=8,
                    ha='center', va='bottom', color=colors[i], fontweight='bold')

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=7, alpha=0.5)
    ax.set_title('Mean Dimension Scores (Radar)', fontsize=14, fontweight='bold', y=1.08)

    fig.tight_layout()
    path = os.path.join(output_dir, 'dimension_radar.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    print(f'保存: {path}')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='口袋评估综合可视化')
    parser.add_argument('--scores-xlsx', type=str, default='docs/pocketeval1.xlsx',
                        help='评分 Excel 文件路径')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='输出目录')
    args = parser.parse_args()

    df = load_data(args.scores_xlsx)
    print(f'加载 {len(df)} 个口袋评分')

    if args.output_dir is None:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output_dir = f'pocket_quality_vis/eval_summary_{ts}'
    os.makedirs(args.output_dir, exist_ok=True)

    plot_distribution(df, args.output_dir)
    plot_dimension_boxplot(df, args.output_dir)
    plot_correlation_heatmap(df, args.output_dir)
    plot_dimension_radar(df, args.output_dir)

    print(f'\n所有图表已保存到: {args.output_dir}')


if __name__ == '__main__':
    main()
