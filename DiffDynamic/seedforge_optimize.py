#!/usr/bin/env python3
"""SeedForge — Iterative Parameter Optimization for Scaffold-Cascade Generation.

Runs scaffold-constrained generation with different parameters each round,
evaluates with VinaScore, tracks metrics, and picks the best configuration.

Designed for systematic method optimization on a target pocket.

Usage:
    python3 seedforge_optimize.py -i 10 --device cuda:1 --rounds 10 --batch_size 50
"""

import argparse
import copy
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = Path(SCRIPT_DIR)


def get_round_configs():
    """Define parameter configurations for each optimization round.

    Each round varies scaffold grow parameters to explore different
    generation strategies. Returns list of (name, grow_params) tuples.
    """
    configs = [
        # Round 1: Baseline — Bayesian-optimized scaffold grow params
        ("baseline", {
            "start_t": 357, "stride": 15, "step_size": 0.3262,
            "lambda_coeff_a": 47, "lambda_coeff_b": 11,
            "n_extra_mode": "pocket_prior",
        }),
        # Round 2: Lower start_t — start denoising earlier
        ("low_start_t", {
            "start_t": 350, "stride": 15, "step_size": 0.33,
            "lambda_coeff_a": 40, "lambda_coeff_b": 5,
            "n_extra_mode": "prior_minus_scaffold",
        }),
        # Round 3: Higher start_t — more noise, more diversity
        ("high_start_t", {
            "start_t": 550, "stride": 15, "step_size": 0.33,
            "lambda_coeff_a": 40, "lambda_coeff_b": 5,
            "n_extra_mode": "prior_minus_scaffold",
        }),
        # Round 4: Fine stride — smaller denoising steps
        ("fine_stride", {
            "start_t": 450, "stride": 8, "step_size": 0.33,
            "lambda_coeff_a": 40, "lambda_coeff_b": 5,
            "n_extra_mode": "prior_minus_scaffold",
        }),
        # Round 5: Pocket prior mode
        ("pocket_prior", {
            "start_t": 450, "stride": 15, "step_size": 0.33,
            "lambda_coeff_a": 40, "lambda_coeff_b": 5,
            "n_extra_mode": "pocket_prior",
        }),
        # Round 6: Strong guidance — higher lambda
        ("strong_guidance", {
            "start_t": 450, "stride": 15, "step_size": 0.33,
            "lambda_coeff_a": 60, "lambda_coeff_b": 10,
            "n_extra_mode": "prior_minus_scaffold",
        }),
        # Round 7: Gentle guidance — lower lambda
        ("gentle_guidance", {
            "start_t": 450, "stride": 15, "step_size": 0.33,
            "lambda_coeff_a": 25, "lambda_coeff_b": 3,
            "n_extra_mode": "prior_minus_scaffold",
        }),
        # Round 8: High start + fine stride combo
        ("high_fine", {
            "start_t": 550, "stride": 10, "step_size": 0.33,
            "lambda_coeff_a": 40, "lambda_coeff_b": 5,
            "n_extra_mode": "prior_minus_scaffold",
        }),
        # Round 9: More atoms to grow
        ("more_atoms", {
            "start_t": 450, "stride": 15, "step_size": 0.33,
            "lambda_coeff_a": 40, "lambda_coeff_b": 5,
            "n_extra_mode": "prior_minus_scaffold",
            "n_extra_fixed": 12,
            "n_extra_min": 5,
            "n_extra_max": 25,
        }),
        # Round 10: Best-of synthesis (updated dynamically)
        ("best_rerun", None),  # Will be filled with best params
    ]
    return configs


def write_patched_config(base_config, out_path, num_samples, seed, grow_params):
    """Create a patched YAML config for scaffold grow mode."""
    with open(base_config) as f:
        cfg = yaml.safe_load(f)

    sc = cfg.setdefault("sample", {}).setdefault("scaffold", {})
    sc["enable"] = True
    sc["mode"] = "grow"
    sc["after_dynamic"] = False
    sc["scaffold_source"] = "auto_murcko"
    sc["fix_scaffold_pos"] = True
    sc["fix_scaffold_type"] = True
    sc["schedule"] = "lambda"
    sc["use_with_noise"] = True
    sc["use_adaptive_step"] = True
    sc["qed_weight"] = 1.0
    sc["sa_weight"] = 1.0
    sc["diversity_weight"] = 0.5
    sc["min_qed"] = 0.15
    sc["min_sa"] = 0.15
    sc["keep_original"] = True
    sc["filter_incomplete"] = True
    sc["diversity_filter"] = {
        "enable": True, "max_tanimoto": 0.9,
        "fingerprint_radius": 2, "fingerprint_nbits": 2048,
    }

    grow = sc.get("grow", {})
    grow["num_samples"] = num_samples
    grow["filter_incomplete"] = False
    # Apply round-specific params
    for k, v in grow_params.items():
        grow[k] = v
    sc["grow"] = grow
    cfg["sample"]["seed"] = seed

    # Disable targetdiff refine for cleaner output
    if "targetdiff_baseline_refine" in cfg.get("sample", {}):
        cfg["sample"]["targetdiff_baseline_refine"]["enable"] = False

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    return out_path


def run_generation(config_path, data_id, device, extra_args=None):
    """Run sample_diffusion.py and return the output .pt path."""
    cmd = [
        sys.executable, os.path.join(SCRIPT_DIR, "scripts", "sample_diffusion.py"),
        config_path, "-i", str(data_id), "--device", device,
    ]
    if extra_args:
        cmd += extra_args

    env = os.environ.copy()
    env["PYTHONPATH"] = SCRIPT_DIR
    env["PYTHONUNBUFFERED"] = "1"

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, env=env, cwd=SCRIPT_DIR, bufsize=1,
    )

    output_pt = None
    for line in proc.stdout:
        line = line.rstrip()
        if line:
            print(f"  [gen] {line}")
        if "Results saved to:" in line:
            m = re.search(r"Results saved to:\s*(/\S+\.pt)", line)
            if m:
                output_pt = m.group(1)

    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"sample_diffusion.py exited with code {proc.returncode}")
    return output_pt


def run_evaluation(pt_path, data_id, output_dir, max_samples=50, vina_timeout=20):
    """Run evaluate_pt_with_correct_reconstruct.py with VinaScore only."""
    protein_root = os.path.join(SCRIPT_DIR, "data", "crossdocked_pocket10_test_only")

    cmd = [
        sys.executable, os.path.join(SCRIPT_DIR, "evaluate_pt_with_correct_reconstruct.py"),
        pt_path,
        "--protein_root", protein_root,
        "--data_id", str(data_id),
        "--output_dir", output_dir,
        "--max_samples", str(max_samples),
        "--vina-timeout-seconds", str(vina_timeout),
        "--vina-modes", "score_only",
        "--no_sdf",
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = SCRIPT_DIR
    env["PYTHONUNBUFFERED"] = "1"
    # Set ADT_PYTHON so prepare_receptor4.py works (it's called by VinaDockingTask)
    env["ADT_PYTHON"] = sys.executable

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, env=env, cwd=SCRIPT_DIR, bufsize=1,
    )

    for line in proc.stdout:
        line = line.rstrip()
        if line:
            print(f"  [eval] {line}")

    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"Evaluation exited with code {proc.returncode}")
    return output_dir


def parse_eval_results(eval_dir):
    """Parse evaluation Excel to extract key metrics."""
    import pandas as pd

    xlsx_files = list(Path(eval_dir).rglob("evaluation_results_*.xlsx"))
    if not xlsx_files:
        return None

    xlsx = str(xlsx_files[0])
    try:
        results = pd.read_excel(xlsx, sheet_name="评估结果")
        stats = pd.read_excel(xlsx, sheet_name="统计信息")
    except Exception as e:
        print(f"  Warning: could not parse Excel: {e}")
        return None

    # Build stats dict
    stats_dict = {}
    for _, row in stats.iterrows():
        stats_dict[row["统计项目"]] = row["数值"]

    # Extract SMILES to check fragments
    smiles_list = results["SMILES"].dropna().tolist()
    total_mols = len(smiles_list)
    complete_mols = sum(1 for s in smiles_list if "." not in str(s))
    fragment_mols = total_mols - complete_mols

    # Get Vina scores (score_only mode)
    vina_scores = results["Vina_ScoreOnly_亲和力"].dropna().tolist()

    metrics = {
        "total_molecules": total_mols,
        "complete_molecules": complete_mols,
        "fragment_molecules": fragment_mols,
        "completeness_rate": complete_mols / total_mols if total_mols > 0 else 0,
        "vina_score_mean": float(stats_dict.get("Vina_ScoreOnly_平均亲和力", 0)),
        "vina_score_best": float(stats_dict.get("Vina_ScoreOnly_最佳亲和力", 0)),
        "vina_score_std": float(stats_dict.get("Vina_ScoreOnly_亲和力标准差", 0)),
        "mean_qed": float(stats_dict.get("平均QED", 0)),
        "mean_sa": float(stats_dict.get("平均SA", 0)),
        "reconstruction_rate": float(stats_dict.get("重建成功百分比(%)", 0)),
        "docking_rate": float(stats_dict.get("对接成功百分比(%)", 0)),
        "smiles": smiles_list,
    }

    # Compute composite score: lower (more negative) Vina is better
    # Weight: 60% Vina + 20% QED + 10% SA + 10% completeness
    vina_norm = max(-15, min(0, metrics["vina_score_mean"])) / -15  # 0-1, higher better
    qed_norm = metrics["mean_qed"]  # 0-1
    sa_norm = metrics["mean_sa"]  # 0-1
    comp_norm = metrics["completeness_rate"]  # 0-1

    metrics["composite_score"] = (
        0.60 * vina_norm +
        0.20 * qed_norm +
        0.10 * sa_norm +
        0.10 * comp_norm
    )

    return metrics


def run_single_round(round_idx, round_name, grow_params, args, base_config):
    """Run one optimization round: generate → evaluate → parse metrics."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    round_dir = os.path.join(SCRIPT_DIR, "outputs", "seedforge", f"r{round_idx:02d}_{round_name}_{timestamp}")
    os.makedirs(round_dir, exist_ok=True)

    seeds = [2021, 42, 123, 456, 789]
    seed = seeds[round_idx % len(seeds)]

    print(f"\n{'='*70}")
    print(f"  SeedForge Round {round_idx}: {round_name}")
    print(f"  Params: {json.dumps(grow_params, indent=2)}")
    print(f"  Seed: {seed} | Batch: {args.batch_size} | Device: {args.device}")
    print(f"{'='*70}")

    # ── Step 1: Generate ──
    config_path = os.path.join(round_dir, "sampling.yml")
    write_patched_config(base_config, config_path, args.batch_size, seed, grow_params)

    gen_start = time.time()
    try:
        pt_path = run_generation(config_path, args.data_id, args.device)
    except RuntimeError as e:
        print(f"  ERROR: Generation failed: {e}")
        return None
    gen_time = time.time() - gen_start

    if not pt_path:
        print("  ERROR: No output .pt file detected")
        return None

    print(f"  Generated: {pt_path} ({gen_time:.0f}s)")

    # Copy .pt to round dir for records
    import shutil
    pt_copy = os.path.join(round_dir, os.path.basename(pt_path))
    shutil.copy2(pt_path, pt_copy)

    # ── Step 2: Evaluate ──
    eval_dir = os.path.join(round_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)

    eval_start = time.time()
    try:
        run_evaluation(pt_path, args.data_id, eval_dir,
                       max_samples=args.batch_size, vina_timeout=args.vina_timeout)
    except RuntimeError as e:
        print(f"  ERROR: Evaluation failed: {e}")
        return None
    eval_time = time.time() - eval_start

    # ── Step 3: Parse metrics ──
    metrics = parse_eval_results(eval_dir)
    if not metrics:
        print("  ERROR: Could not parse evaluation results")
        return None

    metrics["round"] = round_idx
    metrics["name"] = round_name
    metrics["params"] = grow_params
    metrics["gen_time"] = gen_time
    metrics["eval_time"] = eval_time
    metrics["pt_path"] = pt_path
    metrics["eval_dir"] = eval_dir
    metrics["timestamp"] = timestamp

    # Save metrics
    metrics_path = os.path.join(round_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n  Round {round_idx} Results:")
    print(f"    Molecules: {metrics['total_molecules']} total, {metrics['complete_molecules']} complete, {metrics['fragment_molecules']} fragments")
    print(f"    Completeness: {metrics['completeness_rate']:.1%}")
    print(f"    VinaScore: mean={metrics['vina_score_mean']:.3f}, best={metrics['vina_score_best']:.3f}")
    print(f"    QED: {metrics['mean_qed']:.3f} | SA: {metrics['mean_sa']:.3f}")
    print(f"    Composite: {metrics['composite_score']:.4f}")
    print(f"    Time: gen={gen_time:.0f}s, eval={eval_time:.0f}s")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="SeedForge: Iterative scaffold parameter optimization")
    parser.add_argument("-i", "--data_id", type=int, required=True,
                        help="Test set pocket index (e.g. 10 = CARM1)")
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--vina_timeout", type=int, default=20)
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    base_config = args.config or os.path.join(SCRIPT_DIR, "configs", "sampling.yml")
    round_configs = get_round_configs()[:args.rounds]

    # Track all results
    all_results = []
    best_result = None
    best_composite = -1

    # Main optimization loop
    for round_idx, (round_name, grow_params) in enumerate(round_configs):
        # For the last round, use best params if available
        if grow_params is None and best_result:
            grow_params = best_result["params"]
            round_name = f"best_rerun"
            print(f"\n>>> Round {round_idx}: Using best params from round {best_result['round']}")

        if grow_params is None:
            print(f"\n>>> Round {round_idx}: No params available, skipping")
            continue

        metrics = run_single_round(round_idx, round_name, grow_params, args, base_config)
        if metrics:
            all_results.append(metrics)
            if metrics["composite_score"] > best_composite:
                best_composite = metrics["composite_score"]
                best_result = metrics

    # ── Final Report ──
    print(f"\n{'='*70}")
    print(f"  SeedForge Optimization Complete")
    print(f"{'='*70}")

    if not all_results:
        print("  No successful rounds!")
        return 1

    # Sort by composite score
    ranked = sorted(all_results, key=lambda x: x["composite_score"], reverse=True)

    print(f"\n  Rankings (by composite score):")
    print(f"  {'Rank':<5} {'Round':<5} {'Name':<20} {'Vina':<10} {'QED':<8} {'SA':<8} {'Complete':<10} {'Composite':<10}")
    print(f"  {'-'*76}")
    for rank, r in enumerate(ranked, 1):
        print(f"  {rank:<5} {r['round']:<5} {r['name']:<20} {r['vina_score_mean']:<10.3f} {r['mean_qed']:<8.3f} {r['mean_sa']:<8.3f} {r['completeness_rate']:<10.1%} {r['composite_score']:<10.4f}")

    best = ranked[0]
    print(f"\n  Best Configuration: Round {best['round']} ({best['name']})")
    print(f"    Params: {json.dumps(best['params'], indent=4)}")
    print(f"    VinaScore: {best['vina_score_mean']:.3f} (best: {best['vina_score_best']:.3f})")
    print(f"    QED: {best['mean_qed']:.3f} | SA: {best['mean_sa']:.3f}")
    print(f"    Completeness: {best['completeness_rate']:.1%}")
    print(f"    Composite: {best['composite_score']:.4f}")

    # Save full report
    report_dir = os.path.join(SCRIPT_DIR, "outputs", "seedforge")
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, f"optimization_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    report = {
        "data_id": args.data_id,
        "device": args.device,
        "batch_size": args.batch_size,
        "vina_timeout": args.vina_timeout,
        "total_rounds": len(all_results),
        "best_round": best["round"],
        "best_name": best["name"],
        "best_params": best["params"],
        "best_metrics": {k: v for k, v in best.items() if k not in ("smiles", "params")},
        "all_rounds": [{k: v for k, v in r.items() if k != "smiles"} for r in ranked],
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  Full report: {report_path}")

    # Print best params for easy copy
    print(f"\n  Best params YAML snippet:")
    print(f"  sample:")
    print(f"    scaffold:")
    print(f"      grow:")
    for k, v in best["params"].items():
        print(f"        {k}: {v}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
