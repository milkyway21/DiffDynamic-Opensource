#!/usr/bin/env python3
"""ScaffoldCascade Pipeline — DualSeed Scaffold Grow.

Runs scaffold-constrained generation twice with different random seeds,
then merges the outputs into a single .pt file for evaluation.

Usage:
    python3 scaffold_cascade_pipeline.py -i 0 --device cuda:0 --samples_per_round 5
    python3 scaffold_cascade_pipeline.py --protein_path x.pdb --ligand_path y.sdf --device cuda:0
"""

import argparse
import copy
import os
import subprocess
import sys
import tempfile
import time

import torch
import yaml


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def write_patched_config(base_config, out_path, scaffold_num_samples, seed, extra_grow_params=None):
    """Create a patched YAML config for scaffold grow mode."""
    with open(base_config) as f:
        cfg = yaml.safe_load(f)

    cfg.setdefault("sample", {}).setdefault("scaffold", {})
    cfg["sample"]["scaffold"]["enable"] = True
    cfg["sample"]["scaffold"]["mode"] = "grow"
    cfg["sample"]["scaffold"]["after_dynamic"] = False
    cfg["sample"]["scaffold"]["scaffold_source"] = "auto_murcko"
    cfg["sample"]["scaffold"]["fix_scaffold_pos"] = True
    cfg["sample"]["scaffold"]["fix_scaffold_type"] = True
    cfg["sample"]["scaffold"]["schedule"] = "lambda"
    cfg["sample"]["scaffold"]["use_with_noise"] = True
    cfg["sample"]["scaffold"]["use_adaptive_step"] = True
    cfg["sample"]["scaffold"]["qed_weight"] = 1.0
    cfg["sample"]["scaffold"]["sa_weight"] = 1.0
    cfg["sample"]["scaffold"]["diversity_weight"] = 0.5
    cfg["sample"]["scaffold"]["min_qed"] = 0.15
    cfg["sample"]["scaffold"]["min_sa"] = 0.15
    cfg["sample"]["scaffold"]["keep_original"] = True
    cfg["sample"]["scaffold"]["filter_incomplete"] = True
    cfg["sample"]["scaffold"]["diversity_filter"] = {
        "enable": True,
        "max_tanimoto": 0.9,
        "fingerprint_radius": 2,
        "fingerprint_nbits": 2048,
    }

    grow_cfg = cfg["sample"]["scaffold"].get("grow", {})
    grow_cfg["num_samples"] = scaffold_num_samples
    grow_cfg.setdefault("start_t", 357)
    grow_cfg.setdefault("stride", 15)
    grow_cfg.setdefault("step_size", 0.3262)
    grow_cfg.setdefault("lambda_coeff_a", 47)
    grow_cfg.setdefault("lambda_coeff_b", 11)
    grow_cfg.setdefault("n_extra_mode", "pocket_prior")
    grow_cfg.setdefault("n_extra_fixed", 8)
    grow_cfg.setdefault("n_extra_min", 3)
    grow_cfg.setdefault("n_extra_max", 20)
    grow_cfg.setdefault("n_extra_min_clamp", 1)
    grow_cfg.setdefault("n_extra_max_clamp", 45)
    grow_cfg.setdefault("min_qed", 0.2)
    grow_cfg.setdefault("min_sa", 0.2)
    grow_cfg.setdefault("filter_incomplete", False)

    if extra_grow_params:
        grow_cfg.update(extra_grow_params)

    cfg["sample"]["scaffold"]["grow"] = grow_cfg
    cfg["sample"]["seed"] = seed

    # Disable targetdiff baseline refine for cleaner output
    if "targetdiff_baseline_refine" in cfg.get("sample", {}):
        cfg["sample"]["targetdiff_baseline_refine"]["enable"] = False

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    return out_path


def run_sample(config_path, args, log_prefix="[Round]"):
    """Run sample_diffusion.py and return the output .pt path."""
    cmd = [
        sys.executable,
        os.path.join(SCRIPT_DIR, "scripts", "sample_diffusion.py"),
        config_path,
        "--device", args.device,
    ]

    if args.data_id is not None:
        cmd += ["-i", str(args.data_id)]
    if args.protein_path:
        cmd += ["--protein_path", args.protein_path]
    if args.ligand_path:
        cmd += ["--ligand_path", args.ligand_path]
    if args.pocket_radius != 10.0:
        cmd += ["--pocket_radius", str(args.pocket_radius)]

    print(f"{log_prefix} Running: {' '.join(cmd)}")

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
            print(f"{log_prefix} {line}")
        if "Results saved to:" in line:
            import re
            m = re.search(r"Results saved to:\s*(/\S+\.pt)", line)
            if m:
                output_pt = m.group(1)

    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"{log_prefix} sample_diffusion.py exited with code {proc.returncode}")

    return output_pt


def merge_pt_files(pt_paths, output_path):
    """Merge multiple .pt result files into one."""
    all_results = []
    merged_meta = {}

    for pt_path in pt_paths:
        if not pt_path or not os.path.isfile(pt_path):
            print(f"  Warning: skipping missing file {pt_path}")
            continue
        data = torch.load(pt_path, map_location="cpu")
        if isinstance(data, dict):
            all_results.append(data)
            if "meta" in data:
                merged_meta.update(data.get("meta", {}))
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    all_results.append(item)

    if not all_results:
        raise RuntimeError("No valid results to merge")

    if len(all_results) == 1:
        merged = all_results[0]
    else:
        import numpy as np
        merged = {}
        first = all_results[0]
        for key in first:
            val = first[key]
            if key == "meta":
                merged[key] = merged_meta
                continue
            if key == "mode":
                merged[key] = "scaffold_cascade"
                continue
            if isinstance(val, list) and len(val) > 0 and isinstance(val[0], (torch.Tensor, np.ndarray)):
                # List of arrays/tensors — extend from all rounds
                combined = []
                for r in all_results:
                    if key in r and isinstance(r[key], list):
                        combined.extend(r[key])
                merged[key] = combined
            elif isinstance(val, (torch.Tensor, np.ndarray)):
                # Single array — concatenate along dim 0
                pieces = []
                for r in all_results:
                    if key in r and isinstance(r[key], (torch.Tensor, np.ndarray)):
                        pieces.append(r[key])
                if pieces:
                    merged[key] = np.concatenate(pieces, axis=0) if isinstance(pieces[0], np.ndarray) else torch.cat(pieces, dim=0)
            else:
                merged[key] = val

        # Copy keys only in later results
        for r in all_results[1:]:
            for key in r:
                if key not in merged:
                    merged[key] = r[key]

    merged["mode"] = "scaffold_cascade"
    if "meta" not in merged:
        merged["meta"] = {}
    merged["meta"]["method"] = "scaffold_cascade_dual_seed"
    merged["meta"]["num_rounds"] = len(all_results)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(merged, output_path)
    print(f"  Merged {len(all_results)} results → {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="ScaffoldCascade: Dual-seed scaffold grow pipeline")
    parser.add_argument("-i", "--data_id", type=int, default=None)
    parser.add_argument("--protein_path", type=str, default=None)
    parser.add_argument("--ligand_path", type=str, default=None)
    parser.add_argument("--pocket_radius", type=float, default=10.0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--samples_per_round", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if args.data_id is None and args.protein_path is None:
        parser.error("Either --data_id or --protein_path is required")

    base_config = args.config or os.path.join(SCRIPT_DIR, "configs", "sampling.yml")
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    pocket_id = str(args.data_id) if args.data_id is not None else "custom"
    output_dir = os.path.join(SCRIPT_DIR, "outputs", "pt")

    # Run multiple rounds with different seeds
    seeds = [2021, 42, 123, 456, 789][:args.rounds]
    # Vary grow parameters slightly for diversity across rounds
    grow_variants = [
        {"n_extra_mode": "prior_minus_scaffold"},
        {"n_extra_mode": "prior_minus_scaffold", "start_t": 350},
        {"n_extra_mode": "prior_minus_scaffold", "stride": 20},
        {"n_extra_mode": "pocket_prior"},
        {"n_extra_mode": "prior_minus_scaffold", "start_t": 500, "stride": 10},
    ][:args.rounds]

    round_outputs = []
    for round_idx, seed in enumerate(seeds):
        print(f"\n{'='*60}")
        print(f"  Round {round_idx + 1}/{len(seeds)} | seed={seed} | samples={args.samples_per_round}")
        print(f"{'='*60}")

        config_path = os.path.join(SCRIPT_DIR, "configs", f"cascade_r{round_idx + 1}_{timestamp}.yml")
        extra = grow_variants[round_idx] if round_idx < len(grow_variants) else None
        write_patched_config(base_config, config_path, args.samples_per_round, seed, extra)

        try:
            pt_path = run_sample(config_path, args, log_prefix=f"[R{round_idx + 1}]")
            if pt_path:
                round_outputs.append(pt_path)
                print(f"[R{round_idx + 1}] Output: {pt_path}")
            else:
                print(f"[R{round_idx + 1}] Warning: no output file detected")
        except Exception as e:
            print(f"[R{round_idx + 1}] Error: {e}")

    if not round_outputs:
        print("ERROR: No rounds produced output files")
        return 1

    # Merge outputs
    if args.output:
        merged_path = args.output
    else:
        merged_path = os.path.join(output_dir, f"cascade_{pocket_id}_{timestamp}.pt")

    print(f"\n{'='*60}")
    print(f"  Merging {len(round_outputs)} round outputs")
    print(f"{'='*60}")

    merge_pt_files(round_outputs, merged_path)
    print(f"\nCascade complete! Output: {merged_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
