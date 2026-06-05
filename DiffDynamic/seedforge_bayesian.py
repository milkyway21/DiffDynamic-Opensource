#!/usr/bin/env python3
"""SeedForge Bayesian — GP-based parameter optimization for scaffold-constrained generation.

Uses Bayesian Optimization (Gaussian Process) to find optimal scaffold grow parameters
across two protein pockets simultaneously. Each round generates molecules on both pockets,
evaluates them, and feeds the composite score back to the GP surrogate.

Optimized parameters:
    - start_t:          [350, 550]  — starting noise timestep
    - step_size:        [0.15, 0.55] — denoising step size
    - lambda_coeff_a:   [25, 60]    — guidance strength (numerator)
    - lambda_coeff_b:   [3, 15]     — guidance strength (denominator), must be < a

Usage:
    conda activate diffdynamic
    python3 seedforge_bayesian.py --rounds 100 --batch_size 30 --device cuda:0
"""

import argparse
import copy
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = Path(SCRIPT_DIR)

# ─── Pocket definitions ───────────────────────────────────────────────────────
POCKETS = {
    "8R12": {
        "protein_path": "/data/ye/protein-ligand/8R12.pdbqt",
        "ligand_path": "/data/ye/protein-ligand/8R12_ligand.sdf",
        "label": "8R12",
    },
    "7RPZ_KRAS": {
        "protein_path": "/data/ye/protein-ligand/7PRZ_new/7PRZ.pdbqt",
        "ligand_path": "/data/ye/protein-ligand/7PRZ_new/7PRZ_ligand.sdf",
        "label": "7RPZ_KRAS_G12D",
    },
}

# ─── Parameter search space ───────────────────────────────────────────────────
PARAM_BOUNDS = {
    "start_t":        (350, 550),
    "step_size":      (0.15, 0.55),
    "lambda_coeff_a": (25, 60),
    "lambda_coeff_b": (3, 15),
}


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
    grow["n_extra_mode"] = "pocket_prior"
    grow["n_extra_fixed"] = 8
    grow["n_extra_min"] = 3
    grow["n_extra_max"] = 20
    grow["min_qed"] = 0.2
    grow["min_sa"] = 0.2
    # Apply Bayesian-suggested params (convert numpy types to native Python)
    for k, v in grow_params.items():
        if k in ("start_t", "stride"):
            grow[k] = int(round(float(v)))
        elif isinstance(v, (np.floating, np.integer)):
            grow[k] = float(v)
        else:
            grow[k] = v
    # stride is derived from step_size for compatibility
    # (stride controls denoising frequency, step_size controls magnitude)
    # Keep stride at 15 as default — step_size is the primary tunable
    grow.setdefault("stride", 15)
    sc["grow"] = grow
    cfg["sample"]["seed"] = seed

    if "targetdiff_baseline_refine" in cfg.get("sample", {}):
        cfg["sample"]["targetdiff_baseline_refine"]["enable"] = False

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    return out_path


def run_generation(config_path, protein_path, ligand_path, device, extra_args=None):
    """Run sample_diffusion.py with custom pocket and return output .pt path."""
    cmd = [
        sys.executable, os.path.join(SCRIPT_DIR, "scripts", "sample_diffusion.py"),
        config_path,
        "--protein_path", protein_path,
        "--ligand_path", ligand_path,
        "--device", device,
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
            print(f"    [gen] {line}")
        if "Results saved to:" in line:
            m = re.search(r"Results saved to:\s*(/\S+\.pt)", line)
            if m:
                output_pt = m.group(1)

    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"sample_diffusion.py exited with code {proc.returncode}")
    return output_pt


def run_evaluation(pt_path, protein_path, ligand_path, output_dir, max_samples=30, vina_timeout=20):
    """Run evaluate_pt_with_correct_reconstruct.py with custom pocket."""
    cmd = [
        sys.executable, os.path.join(SCRIPT_DIR, "evaluate_pt_with_correct_reconstruct.py"),
        pt_path,
        "--output_dir", output_dir,
        "--max_samples", str(max_samples),
        "--vina-timeout-seconds", str(vina_timeout),
        "--vina-modes", "score_only",
        "--no_sdf",
        "--receptor_pdb", protein_path,
        "--reference_ligand", ligand_path,
        "--protein_root", os.path.dirname(protein_path),
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = SCRIPT_DIR
    env["PYTHONUNBUFFERED"] = "1"
    env["ADT_PYTHON"] = sys.executable

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, env=env, cwd=SCRIPT_DIR, bufsize=1,
    )

    for line in proc.stdout:
        line = line.rstrip()
        if line:
            print(f"    [eval] {line}")

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
        print(f"    Warning: could not parse Excel: {e}")
        return None

    stats_dict = {}
    for _, row in stats.iterrows():
        stats_dict[row["统计项目"]] = row["数值"]

    smiles_list = results["SMILES"].dropna().tolist()
    total_mols = len(smiles_list)
    complete_mols = sum(1 for s in smiles_list if "." not in str(s))
    fragment_mols = total_mols - complete_mols

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

    # Composite score: 60% Vina + 20% QED + 10% SA + 10% completeness
    vina_norm = max(-15, min(0, metrics["vina_score_mean"])) / -15
    qed_norm = metrics["mean_qed"]
    sa_norm = metrics["mean_sa"]
    comp_norm = metrics["completeness_rate"]

    metrics["composite_score"] = (
        0.60 * vina_norm +
        0.20 * qed_norm +
        0.10 * sa_norm +
        0.10 * comp_norm
    )

    return metrics


def run_single_round(round_idx, grow_params, args, base_config, output_root):
    """Run one optimization round on BOTH pockets, return averaged metrics."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    round_dir = os.path.join(output_root, f"r{round_idx:03d}_{timestamp}")
    os.makedirs(round_dir, exist_ok=True)

    seeds = [2021, 42, 123, 456, 789, 2024, 314, 618, 999, 1337]
    seed = seeds[round_idx % len(seeds)]

    # Ensure lambda_coeff_a > lambda_coeff_b
    a = grow_params["lambda_coeff_a"]
    b = grow_params["lambda_coeff_b"]
    if a <= b:
        grow_params["lambda_coeff_b"] = max(3, a - 5)
        b = grow_params["lambda_coeff_b"]

    # Round integers
    grow_params["start_t"] = int(round(grow_params["start_t"]))
    grow_params["lambda_coeff_a"] = int(round(a))
    grow_params["lambda_coeff_b"] = int(round(b))

    print(f"\n{'='*70}")
    print(f"  SeedForge Bayesian Round {round_idx}")
    print(f"  Params: {json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in grow_params.items()})}")
    print(f"  Seed: {seed} | Batch: {args.batch_size} | Device: {args.device}")
    print(f"{'='*70}")

    pocket_metrics = {}

    for pocket_name, pocket_info in POCKETS.items():
        print(f"\n  ── Pocket: {pocket_info['label']} ──")

        pocket_dir = os.path.join(round_dir, pocket_name)
        os.makedirs(pocket_dir, exist_ok=True)

        # Generate
        config_path = os.path.join(pocket_dir, "sampling.yml")
        write_patched_config(base_config, config_path, args.batch_size, seed, grow_params)

        gen_start = time.time()
        try:
            pt_path = run_generation(
                config_path,
                pocket_info["protein_path"],
                pocket_info["ligand_path"],
                args.device,
            )
        except RuntimeError as e:
            print(f"    ERROR: Generation failed for {pocket_name}: {e}")
            pocket_metrics[pocket_name] = None
            continue
        gen_time = time.time() - gen_start

        if not pt_path:
            print(f"    ERROR: No output .pt file for {pocket_name}")
            pocket_metrics[pocket_name] = None
            continue

        # Copy pt for records
        pt_copy = os.path.join(pocket_dir, os.path.basename(pt_path))
        shutil.copy2(pt_path, pt_copy)
        print(f"    Generated: {pt_path} ({gen_time:.0f}s)")

        # Evaluate
        eval_dir = os.path.join(pocket_dir, "eval")
        os.makedirs(eval_dir, exist_ok=True)

        eval_start = time.time()
        try:
            run_evaluation(
                pt_path,
                pocket_info["protein_path"],
                pocket_info["ligand_path"],
                eval_dir,
                max_samples=args.batch_size,
                vina_timeout=args.vina_timeout,
            )
        except RuntimeError as e:
            print(f"    ERROR: Evaluation failed for {pocket_name}: {e}")
            pocket_metrics[pocket_name] = None
            continue
        eval_time = time.time() - eval_start

        metrics = parse_eval_results(eval_dir)
        if not metrics:
            print(f"    ERROR: Could not parse results for {pocket_name}")
            pocket_metrics[pocket_name] = None
            continue

        metrics["gen_time"] = gen_time
        metrics["eval_time"] = eval_time
        metrics["pt_path"] = pt_path

        # Save per-pocket metrics
        with open(os.path.join(pocket_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False, default=str)

        print(f"    Vina: {metrics['vina_score_mean']:.3f} | QED: {metrics['mean_qed']:.3f} | SA: {metrics['mean_sa']:.3f} | Composite: {metrics['composite_score']:.4f}")

        pocket_metrics[pocket_name] = metrics

    # Average composite scores across pockets
    valid = [m for m in pocket_metrics.values() if m is not None]
    if not valid:
        print(f"\n  Round {round_idx}: ALL pockets failed!")
        return None

    avg_composite = np.mean([m["composite_score"] for m in valid])
    avg_vina = np.mean([m["vina_score_mean"] for m in valid])
    avg_qed = np.mean([m["mean_qed"] for m in valid])
    avg_sa = np.mean([m["mean_sa"] for m in valid])
    avg_completeness = np.mean([m["completeness_rate"] for m in valid])

    round_result = {
        "round": round_idx,
        "params": grow_params,
        "avg_composite": float(avg_composite),
        "avg_vina": float(avg_vina),
        "avg_qed": float(avg_qed),
        "avg_sa": float(avg_sa),
        "avg_completeness": float(avg_completeness),
        "pocket_details": {},
        "timestamp": timestamp,
    }

    for pn, pm in pocket_metrics.items():
        if pm is not None:
            round_result["pocket_details"][pn] = {
                "composite": pm["composite_score"],
                "vina_mean": pm["vina_score_mean"],
                "vina_best": pm["vina_score_best"],
                "qed": pm["mean_qed"],
                "sa": pm["mean_sa"],
                "completeness": pm["completeness_rate"],
                "total_mols": pm["total_molecules"],
                "gen_time": pm.get("gen_time", 0),
                "eval_time": pm.get("eval_time", 0),
            }

    # Save round summary
    with open(os.path.join(round_dir, "round_summary.json"), "w") as f:
        json.dump(round_result, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n  Round {round_idx} Summary:")
    print(f"    Avg Composite: {avg_composite:.4f}")
    print(f"    Avg Vina: {avg_vina:.3f} | Avg QED: {avg_qed:.3f} | Avg SA: {avg_sa:.3f}")

    return round_result


def main():
    parser = argparse.ArgumentParser(description="SeedForge Bayesian: GP-based scaffold parameter optimization")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=30)
    parser.add_argument("--vina_timeout", type=int, default=20)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--init_points", type=int, default=5,
                        help="Random exploration rounds before GP kicks in")
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    base_config = args.config or os.path.join(SCRIPT_DIR, "configs", "sampling.yml")
    output_root = args.output_dir or os.path.join(SCRIPT_DIR, "outputs", "seedforge_bayesian")
    os.makedirs(output_root, exist_ok=True)

    # Check pockets exist
    for name, info in POCKETS.items():
        if not os.path.exists(info["protein_path"]):
            print(f"ERROR: Protein file not found for {name}: {info['protein_path']}")
            return 1
        if not os.path.exists(info["ligand_path"]):
            print(f"ERROR: Ligand file not found for {name}: {info['ligand_path']}")
            return 1

    print(f"\n{'='*70}")
    print(f"  SeedForge Bayesian Optimization")
    print(f"  Rounds: {args.rounds} | Batch/pocket: {args.batch_size} | Device: {args.device}")
    print(f"  Pockets: {', '.join(POCKETS.keys())}")
    print(f"  Init points: {args.init_points}")
    print(f"  Output: {output_root}")
    print(f"{'='*70}\n")

    # ─── Bayesian Optimization ────────────────────────────────────────────────
    from bayes_opt import BayesianOptimization, UtilityFunction

    def black_box_function(start_t, step_size, lambda_coeff_a, lambda_coeff_b):
        """Objective function: average composite score across both pockets."""
        # Ensure a > b
        if lambda_coeff_a <= lambda_coeff_b:
            lambda_coeff_b = lambda_coeff_a - 5
            if lambda_coeff_b < 3:
                lambda_coeff_b = 3
                lambda_coeff_a = lambda_coeff_b + 5

        grow_params = {
            "start_t": float(start_t),
            "step_size": float(step_size),
            "lambda_coeff_a": float(lambda_coeff_a),
            "lambda_coeff_b": float(lambda_coeff_b),
        }

        nonlocal round_counter
        round_idx = round_counter
        round_counter += 1

        result = run_single_round(round_idx, grow_params, args, base_config, output_root)
        if result is None:
            return 0.0  # Failed round

        all_results.append(result)

        # Update best
        nonlocal best_result
        if best_result is None or result["avg_composite"] > best_result["avg_composite"]:
            best_result = result

        return result["avg_composite"]

    optimizer = BayesianOptimization(
        f=black_box_function,
        pbounds=PARAM_BOUNDS,
        random_state=42,
        verbose=2,
    )

    # Use UCB (Upper Confidence Bound) acquisition function
    # kappa controls exploration vs exploitation:
    #   - High kappa = more exploration (early rounds)
    #   - Low kappa = more exploitation (later rounds)
    utility = UtilityFunction(kind="ucb", kappa=2.5, xi=0.0)

    all_results = []
    best_result = None
    round_counter = 0

    # Handle graceful shutdown
    def save_and_exit(signum, frame):
        print(f"\n\n  Interrupted! Saving results...")
        save_report(all_results, best_result, output_root, args)
        sys.exit(0)

    signal.signal(signal.SIGINT, save_and_exit)
    signal.signal(signal.SIGTERM, save_and_exit)

    try:
        # Phase 1: Random exploration
        print(f"\n  Phase 1: Random exploration ({args.init_points} rounds)")
        optimizer.maximize(
            init_points=args.init_points,
            n_iter=0,
        )

        # Phase 2: GP-guided optimization
        remaining = args.rounds - args.init_points
        print(f"\n  Phase 2: GP-guided optimization ({remaining} rounds)")
        for i in range(remaining):
            next_point = optimizer.suggest(utility)
            target = black_box_function(**next_point)
            optimizer.register(params=next_point, target=target)

            # Decay kappa over time (more exploitation later)
            if (i + 1) % 20 == 0:
                new_kappa = max(0.5, 2.5 - (i + 1) / remaining * 2.0)
                utility = UtilityFunction(kind="ucb", kappa=new_kappa, xi=0.0)
                print(f"  [kappa decay] New kappa: {new_kappa:.2f}")

    except Exception as e:
        print(f"\n  ERROR: {e}")
        traceback.print_exc()
    finally:
        save_report(all_results, best_result, output_root, args)

    return 0


def save_report(all_results, best_result, output_root, args):
    """Save final optimization report."""
    if not all_results:
        print("  No results to report!")
        return

    # Rank by composite
    ranked = sorted(all_results, key=lambda x: x["avg_composite"], reverse=True)

    print(f"\n{'='*70}")
    print(f"  SeedForge Bayesian Optimization Complete")
    print(f"{'='*70}")
    print(f"  Total rounds: {len(all_results)}")
    print(f"  Successful: {sum(1 for r in all_results if r['avg_composite'] > 0)}")

    print(f"\n  Top 10 Configurations:")
    print(f"  {'Rank':<5} {'Round':<6} {'Vina':<10} {'QED':<8} {'SA':<8} {'Comp':<8} {'Composite':<10} {'Params'}")
    print(f"  {'-'*90}")
    for rank, r in enumerate(ranked[:10], 1):
        p = r["params"]
        pstr = f"t={p['start_t']} s={p['step_size']:.2f} a={p['lambda_coeff_a']} b={p['lambda_coeff_b']}"
        print(f"  {rank:<5} {r['round']:<6} {r['avg_vina']:<10.3f} {r['avg_qed']:<8.3f} {r['avg_sa']:<8.3f} {r['avg_completeness']:<8.1%} {r['avg_composite']:<10.4f} {pstr}")

    if best_result:
        bp = best_result["params"]
        print(f"\n  Best Configuration (Round {best_result['round']}):")
        print(f"    start_t:        {bp['start_t']}")
        print(f"    step_size:      {bp['step_size']:.4f}")
        print(f"    lambda_coeff_a: {bp['lambda_coeff_a']}")
        print(f"    lambda_coeff_b: {bp['lambda_coeff_b']}")
        print(f"    Composite:      {best_result['avg_composite']:.4f}")
        print(f"    Vina:           {best_result['avg_vina']:.3f}")
        print(f"    QED:            {best_result['avg_qed']:.3f}")
        print(f"    SA:             {best_result['avg_sa']:.3f}")

        print(f"\n  Best params YAML:")
        print(f"  sample:")
        print(f"    scaffold:")
        print(f"      grow:")
        for k, v in bp.items():
            if isinstance(v, float):
                print(f"        {k}: {v:.4f}")
            else:
                print(f"        {k}: {v}")

    # Save JSON report
    report = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "args": {
            "rounds": args.rounds,
            "batch_size": args.batch_size,
            "device": args.device,
            "vina_timeout": args.vina_timeout,
        },
        "total_rounds": len(all_results),
        "best_round": best_result["round"] if best_result else None,
        "best_params": best_result["params"] if best_result else None,
        "best_composite": best_result["avg_composite"] if best_result else None,
        "all_rounds": [{k: v for k, v in r.items()} for r in ranked],
    }

    report_path = os.path.join(output_root, f"bayesian_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  Report: {report_path}")

    # Save best params as YAML snippet
    if best_result:
        best_yaml_path = os.path.join(output_root, "best_params.yml")
        with open(best_yaml_path, "w") as f:
            f.write("# SeedForge Bayesian Optimization — Best Parameters\n")
            f.write(f"# Composite: {best_result['avg_composite']:.4f}\n")
            f.write(f"# Round: {best_result['round']}\n")
            f.write("sample:\n")
            f.write("  scaffold:\n")
            f.write("    grow:\n")
            for k, v in best_result["params"].items():
                if isinstance(v, float):
                    f.write(f"      {k}: {v:.4f}\n")
                else:
                    f.write(f"      {k}: {v}\n")
        print(f"  Best YAML: {best_yaml_path}")


if __name__ == "__main__":
    sys.exit(main())
