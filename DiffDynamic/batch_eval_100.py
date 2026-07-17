#!/usr/bin/env python3
"""批量评估 jsdpt3010 测试集 100 个口袋，输出综合统计。"""

import sys
import os
import json
import glob
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from evaluate_pocket_quality import evaluate_pocket_quality

DATA_DIR = Path('/home/user/Desktop/Ye/DiffDynamic/jsdpt3010')
OUTPUT_CSV = Path('/data/ye/DiffDynamic/eval_100_results.csv')
OUTPUT_JSON = Path('/data/ye/DiffDynamic/eval_100_results.json')


def find_result_files(data_dir):
    """找到所有 result_{id}_*.pt 文件，返回 {data_id: path} 字典。"""
    files = sorted(data_dir.glob('result_*_*.pt'))
    result = {}
    for f in files:
        parts = f.stem.split('_')
        if len(parts) >= 3:
            try:
                data_id = int(parts[1])
                result[data_id] = str(f)
            except ValueError:
                pass
    return result


def evaluate_single(data_id, pt_path, vina_dir):
    """评估单个口袋，返回结果 dict。"""
    try:
        result = evaluate_pocket_quality(
            pt_path=pt_path,
            vina_outputs_dir=str(vina_dir),
            data_id=data_id,
        )
        return result
    except Exception as e:
        return {
            'data_id': data_id,
            'error': str(e),
            'overall_score': 0.0,
            'overall_label': 'error',
        }


def format_row(data_id, result):
    """提取一行 CSV 数据。"""
    row = {'data_id': data_id}
    row['overall_score'] = f"{result.get('overall_score', 0):.4f}"
    row['overall_label'] = result.get('overall_label', 'unknown')

    for key in ['idea_a', 'idea_b', 'idea_c', 'idea_d', 'idea_e', 'idea_f', 'idea_g', 'idea_h']:
        r = result.get(key, {})
        row[f'{key}_score'] = f"{r.get('score', 0):.4f}" if r.get('success') else 'N/A'
        row[f'{key}_label'] = r.get('quality_label', 'unknown') if r.get('success') else 'FAIL'

    # 关键细节
    idea_a = result.get('idea_a', {})
    row['vina_mean'] = f"{idea_a.get('vina_mean', 0):.2f}" if idea_a.get('success') else 'N/A'
    row['vina_best'] = f"{idea_a.get('vina_best', 0):.2f}" if idea_a.get('success') else 'N/A'
    row['vina_pct_good'] = f"{idea_a.get('vina_pct_good', 0)*100:.1f}%" if idea_a.get('success') else 'N/A'

    idea_d = result.get('idea_d', {})
    row['qed_mean'] = f"{idea_d.get('qed_mean', 0):.3f}" if idea_d.get('success') else 'N/A'
    row['sa_mean'] = f"{idea_d.get('sa_mean', 0):.3f}" if idea_d.get('success') else 'N/A'
    row['pains_ratio'] = f"{idea_d.get('pains_ratio', 0)*100:.1f}%" if idea_d.get('success') else 'N/A'
    row['veber_ratio'] = f"{idea_d.get('veber_ratio', 0)*100:.1f}%" if idea_d.get('success') else 'N/A'

    idea_h = result.get('idea_h', {})
    row['volume_ang3'] = f"{idea_h.get('volume_ang3', 0):.0f}" if idea_h.get('success') else 'N/A'

    return row


def print_summary(all_results):
    """打印统计摘要。"""
    scores = [r.get('overall_score', 0) for r in all_results if not r.get('error')]
    if not scores:
        print("No valid results!")
        return

    arr = np.array(scores)
    print(f"\n{'='*60}")
    print(f"  评估完成: {len(arr)} 个口袋")
    print(f"{'='*60}")
    print(f"  Overall Score:  mean={arr.mean():.3f}, std={arr.std():.3f}")
    print(f"                  min={arr.min():.3f}, max={arr.max():.3f}")
    print(f"                  median={np.median(arr):.3f}")

    # 各维度统计
    dim_names = ['idea_a', 'idea_b', 'idea_c', 'idea_d', 'idea_e', 'idea_f', 'idea_g', 'idea_h']
    dim_labels = ['A-Vina', 'B-Cluster', 'C-LE', 'D-Drug', 'E-Recon', 'F-Uniq', 'G-Size', 'H-Volume']

    print(f"\n  {'Dimension':<12} {'Mean':>6} {'Std':>6} {'Min':>6} {'Max':>6} {'High%':>6} {'Fail%':>6}")
    print(f"  {'-'*54}")
    for key, label in zip(dim_names, dim_labels):
        dim_scores = []
        n_high = 0
        n_fail = 0
        for r in all_results:
            if r.get('error'):
                n_fail += 1
                continue
            d = r.get(key, {})
            if d.get('success'):
                dim_scores.append(d['score'])
                if d.get('quality_label') == 'high':
                    n_high += 1
            else:
                n_fail += 1
        if dim_scores:
            a = np.array(dim_scores)
            print(f"  {label:<12} {a.mean():>6.3f} {a.std():>6.3f} {a.min():>6.3f} {a.max():>6.3f} {n_high/len(all_results)*100:>5.1f}% {n_fail/len(all_results)*100:>5.1f}%")
        else:
            print(f"  {label:<12}   N/A")

    # 质量分布
    n_high = sum(1 for s in scores if s >= 0.6)
    n_medium = sum(1 for s in scores if 0.3 <= s < 0.6)
    n_low = sum(1 for s in scores if s < 0.3)
    print(f"\n  质量分布: high={n_high} ({n_high/len(scores)*100:.1f}%), "
          f"medium={n_medium} ({n_medium/len(scores)*100:.1f}%), "
          f"low={n_low} ({n_low/len(scores)*100:.1f}%)")

    errors = sum(1 for r in all_results if r.get('error'))
    if errors:
        print(f"  错误: {errors} 个口袋评估失败")


def main():
    print(f"开始批量评估 jsdpt3010 测试集...")
    print(f"数据目录: {DATA_DIR}")
    print(f"输出: {OUTPUT_CSV}")
    print()

    result_files = find_result_files(DATA_DIR)
    print(f"找到 {len(result_files)} 个 result_*.pt 文件")

    all_results = []
    all_rows = []

    sorted_ids = sorted(result_files.keys())
    for i, data_id in enumerate(sorted_ids):
        pt_path = result_files[data_id]
        print(f"[{i+1}/{len(sorted_ids)}] 评估 data_id={data_id} ...", end=' ', flush=True)

        result = evaluate_single(data_id, pt_path, DATA_DIR)
        result['data_id'] = data_id
        all_results.append(result)

        row = format_row(data_id, result)
        all_rows.append(row)

        if result.get('error'):
            print(f"ERROR: {result['error'][:60]}")
        else:
            print(f"score={result['overall_score']:.3f} ({result['overall_label']})")

    # 写 CSV
    if all_rows:
        import csv
        with open(OUTPUT_CSV, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\nCSV 已保存: {OUTPUT_CSV}")

    # 写 JSON（完整结果）
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"JSON 已保存: {OUTPUT_JSON}")

    print_summary(all_results)


if __name__ == '__main__':
    main()
