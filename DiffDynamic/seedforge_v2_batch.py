#!/usr/bin/env python3
"""SeedForge v2: Batch test strategies on 100 pockets (5 mols each, Vina score_only).

Usage:
    python3 seedforge_v2_batch.py --phase 1          # Test top 25 strategies
    python3 seedforge_v2_batch.py --phase 2 --rounds 50  # Bayesian optimization
    python3 seedforge_v2_batch.py --phase 1 --strategy 5  # Test single strategy
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROTEIN_ROOT = './data/crossdocked_pocket10_test_only'
N_POCKETS = 100
MOLS_PER_POCKET = 5
VINA_TIMEOUT = 20


def run_batch_eval(config_path, strategy_idx, output_root):
    """Run batch_sampleandeval_parallel.py for one strategy on 100 pockets."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    batch_csv = os.path.join(output_root, f"v2_s{strategy_idx:02d}_{timestamp}.csv")

    cmd = [
        sys.executable, os.path.join(SCRIPT_DIR, "batch_sampleandeval_parallel.py"),
        "--start", "0", "--end", str(N_POCKETS - 1),
        "--gpus", "0,1,2,3,4,5",
        "--num_cpu_cores", "60", "--cores_per_task", "6",
        "--config", config_path,
        "--protein_root", PROTEIN_ROOT,
        "--eval-vina-modes", "score_only",
        "--excel_file", batch_csv,
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = SCRIPT_DIR
    env["PYTHONUNBUFFERED"] = "1"

    print(f"\n{'='*70}")
    print(f"  Strategy {strategy_idx}: {config_path}")
    print(f"  Output: {batch_csv}")
    print(f"{'='*70}")

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, env=env, cwd=SCRIPT_DIR, bufsize=1,
    )

    for line in proc.stdout:
        line = line.rstrip()
        if line:
            print(f"  {line}")

    proc.wait()
    if proc.returncode != 0:
        print(f"  ERROR: batch_sampleandeval exited with code {proc.returncode}")
        return None

    return batch_csv


def parse_batch_results(csv_path):
    """Parse batch CSV to extract per-pocket and global metrics."""
    import pandas as pd

    if not os.path.exists(csv_path):
        return None

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"  Warning: could not read CSV: {e}")
        return None

    if len(df) == 0:
        return None

    data_id_col = '数据ID'
    vina_col = 'Vina_ScoreOnly_亲和力'

    # Per-pocket stats
    pocket_stats = []
    for did in sorted(df[data_id_col].unique()):
        pocket_df = df[df[data_id_col] == did]
        valid = pocket_df[(pocket_df[vina_col].notna()) & (pocket_df[vina_col] <= 0)]
        pocket_stats.append({
            'data_id': did,
            'total_mols': len(pocket_df),
            'valid_mols': len(valid),
            'vina_mean': float(valid[vina_col].mean()) if len(valid) > 0 else np.nan,
            'vina_best': float(valid[vina_col].min()) if len(valid) > 0 else np.nan,
        })

    stats_df = pd.DataFrame(pocket_stats)

    # Global metrics
    total_mols = len(df)
    valid_mols = len(df[(df[vina_col].notna()) & (df[vina_col] <= 0)])
    vina_values = df[vina_col].dropna()
    valid_vina = vina_values[vina_values <= 0]

    metrics = {
        'total_mols': total_mols,
        'valid_mols': valid_mols,
        'reconstruct_rate': valid_mols / total_mols if total_mols > 0 else 0,
        'vina_mean': float(valid_vina.mean()) if len(valid_vina) > 0 else 0,
        'vina_median': float(valid_vina.median()) if len(valid_vina) > 0 else 0,
        'vina_std': float(valid_vina.std()) if len(valid_vina) > 0 else 0,
        'vina_best_mean': float(stats_df['vina_best'].mean()) if len(stats_df) > 0 else 0,
        'vina_best_std': float(stats_df['vina_best'].std()) if len(stats_df) > 0 else 0,
        'pocket_stats': pocket_stats,
    }

    # Composite score: 70% Vina + 30% reconstruction rate
    vina_norm = max(-15, min(0, metrics['vina_mean'])) / -15  # 0-1, higher better
    metrics['composite'] = 0.70 * vina_norm + 0.30 * metrics['reconstruct_rate']

    return metrics


def phase1_top25(args):
    """Phase 1: Test top 25 strategies on 100 pockets."""
    strategies_path = os.path.join(SCRIPT_DIR, 'outputs/seedforge_bayesian/v2_strategies.json')
    with open(strategies_path) as f:
        strategies = json.load(f)

    output_root = os.path.join(SCRIPT_DIR, 'outputs/seedforge_v2')
    os.makedirs(output_root, exist_ok=True)

    if args.strategy is not None:
        # Test single strategy
        strategies = [s for s in strategies if s['index'] == args.strategy]
        if not strategies:
            print(f"Strategy {args.strategy} not found")
            return

    results = []
    for s in strategies:
        idx = s['index']
        config_path = os.path.join(SCRIPT_DIR, s['config_path'])

        csv_path = run_batch_eval(config_path, idx, output_root)
        if csv_path is None:
            print(f"  Strategy {idx}: FAILED")
            continue

        metrics = parse_batch_results(csv_path)
        if metrics is None:
            print(f"  Strategy {idx}: No results")
            continue

        metrics['strategy_idx'] = idx
        metrics['original_round'] = s['round']
        metrics['params'] = s['params']
        metrics['original_composite'] = s['original_composite']
        metrics['csv_path'] = csv_path
        results.append(metrics)

        print(f"\n  Strategy {idx} Results:")
        print(f"    Vina mean: {metrics['vina_mean']:.3f}")
        print(f"    Vina best mean: {metrics['vina_best_mean']:.3f}")
        print(f"    Reconstruct rate: {metrics['reconstruct_rate']:.1%}")
        print(f"    Composite: {metrics['composite']:.4f}")

    # Save results
    report_path = os.path.join(output_root, f'phase1_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    # Print ranking
    ranked = sorted(results, key=lambda x: x['composite'], reverse=True)
    print(f"\n{'='*70}")
    print(f"  Phase 1 Results — Top 25 Strategies on 100 Pockets")
    print(f"{'='*70}")
    print(f"{'Rank':<5} {'Idx':<5} {'Vina均值':<10} {'最佳均值':<10} {'重建率':<8} {'Composite':<10} {'原Composite':<12}")
    print(f"{'-'*60}")
    for rank, r in enumerate(ranked, 1):
        print(f"{rank:<5} {r['strategy_idx']:<5} {r['vina_mean']:<10.3f} {r['vina_best_mean']:<10.3f} {r['reconstruct_rate']:<8.1%} {r['composite']:<10.4f} {r['original_composite']:<12.4f}")

    print(f"\n  Report: {report_path}")
    return results


def phase2_bayesian(args):
    """Phase 2: Bayesian optimization on 100 pockets."""
    from bayes_opt import BayesianOptimization, UtilityFunction

    output_root = os.path.join(SCRIPT_DIR, 'outputs/seedforge_v2')
    os.makedirs(output_root, exist_ok=True)

    # Load Phase 1 results as initial observations
    phase1_results = []
    phase1_reports = sorted(Path(output_root).glob('phase1_report_*.json'))
    if phase1_reports:
        with open(phase1_reports[-1]) as f:
            phase1_results = json.load(f)

    def objective(start_t, step_size, lambda_coeff_a, lambda_coeff_b):
        """Objective: composite score on 100 pockets."""
        # Ensure a > b
        if lambda_coeff_a <= lambda_coeff_b:
            lambda_coeff_b = max(3, lambda_coeff_a - 5)

        params = {
            'start_t': int(round(float(start_t))),
            'step_size': float(step_size),
            'lambda_coeff_a': int(round(float(lambda_coeff_a))),
            'lambda_coeff_b': int(round(float(lambda_coeff_b))),
        }

        # Generate config
        with open(os.path.join(SCRIPT_DIR, 'configs/sampling_scaffold_bayesian.yml')) as f:
            cfg = yaml.safe_load(f)
        grow = cfg['sample']['scaffold']['grow']
        grow['num_samples'] = MOLS_PER_POCKET
        grow['start_t'] = params['start_t']
        grow['step_size'] = params['step_size']
        grow['lambda_coeff_a'] = params['lambda_coeff_a']
        grow['lambda_coeff_b'] = params['lambda_coeff_b']
        grow['n_extra_mode'] = 'pocket_prior'

        config_path = os.path.join(output_root, f'bayesian_r{round_counter[0]:03d}.yml')
        with open(config_path, 'w') as f:
            yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

        idx = round_counter[0]
        round_counter[0] += 1

        csv_path = run_batch_eval(config_path, 100 + idx, output_root)
        if csv_path is None:
            return 0.0

        metrics = parse_batch_results(csv_path)
        if metrics is None:
            return 0.0

        metrics['strategy_idx'] = 100 + idx
        metrics['params'] = params
        metrics['csv_path'] = csv_path
        all_results.append(metrics)

        print(f"\n  Bayesian Round {idx}: composite={metrics['composite']:.4f}")
        return metrics['composite']

    optimizer = BayesianOptimization(
        f=objective,
        pbounds={
            'start_t': (350, 550),
            'step_size': (0.15, 0.55),
            'lambda_coeff_a': (25, 60),
            'lambda_coeff_b': (3, 15),
        },
        random_state=42,
        verbose=2,
    )

    # Register Phase 1 results as initial observations
    all_results = list(phase1_results)
    round_counter = [0]

    if phase1_results:
        print(f"\n  Registering {len(phase1_results)} Phase 1 results as initial observations")
        for r in phase1_results:
            p = r['params']
            try:
                optimizer.register(
                    params={
                        'start_t': float(p['start_t']),
                        'step_size': float(p['step_size']),
                        'lambda_coeff_a': float(p['lambda_coeff_a']),
                        'lambda_coeff_b': float(p['lambda_coeff_b']),
                    },
                    target=r['composite'],
                )
            except Exception:
                pass

    utility = UtilityFunction(kind="ucb", kappa=2.5, xi=0.0)

    try:
        n_init = max(0, args.init_points - len(phase1_results))
        if n_init > 0:
            print(f"\n  Random exploration: {n_init} rounds")
            optimizer.maximize(init_points=n_init, n_iter=0)

        remaining = args.rounds - len(phase1_results)
        print(f"\n  GP-guided optimization: {remaining} rounds")
        for i in range(remaining):
            next_point = optimizer.suggest(utility)
            target = objective(**next_point)
            optimizer.register(params=next_point, target=target)

            if (i + 1) % 10 == 0:
                new_kappa = max(0.5, 2.5 - (i + 1) / remaining * 2.0)
                utility = UtilityFunction(kind="ucb", kappa=new_kappa, xi=0.0)
    except KeyboardInterrupt:
        print("\n  Interrupted! Saving results...")
    except Exception as e:
        print(f"\n  Error: {e}")

    # Save report
    report_path = os.path.join(output_root, f'bayesian_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    ranked = sorted(all_results, key=lambda x: x.get('composite', 0), reverse=True)
    report = {
        'total_rounds': len(all_results),
        'best': ranked[0] if ranked else None,
        'all_rounds': ranked,
    }
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    if ranked:
        print(f"\n  Best: composite={ranked[0]['composite']:.4f}, params={ranked[0]['params']}")
    print(f"  Report: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="SeedForge v2: 100-pocket evaluation")
    parser.add_argument("--phase", type=int, required=True, choices=[1, 2],
                        help="Phase 1: test top 25, Phase 2: Bayesian optimization")
    parser.add_argument("--strategy", type=int, default=None,
                        help="Test single strategy index (Phase 1 only)")
    parser.add_argument("--rounds", type=int, default=50,
                        help="Number of Bayesian optimization rounds (Phase 2)")
    parser.add_argument("--init_points", type=int, default=5,
                        help="Random exploration rounds (Phase 2)")
    args = parser.parse_args()

    if args.phase == 1:
        phase1_top25(args)
    elif args.phase == 2:
        phase2_bayesian(args)


if __name__ == "__main__":
    main()
