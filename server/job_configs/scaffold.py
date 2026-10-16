"""Build scaffold job YAML and replay metadata without scheduling a job."""

import logging
import os
from typing import Any, Dict, Mapping, Tuple

logger = logging.getLogger(__name__)


def build_scaffold_patches(
    options: Mapping[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Keep legacy mode, placement, gate and refinement values unchanged."""
    metadata: Dict[str, Any] = {}
    scaffold_cfg = {
        "web_input_contract": True,
        "workflow": options["workflow"],
        "enable": True,
        "mode": options["scaffold_mode"],
        "after_dynamic": options["after_dynamic"],
        "save_dynamic_before_scaffold": True,
        "optimization_style_naming": True,
        "scaffold_source": options["scaffold_source"],
        "scaffold_atom_indices": options["scaffold_atom_indices"],
        "scaffold_smarts": options["scaffold_smarts"],
        "fix_scaffold_pos": options["fix_scaffold_pos"],
        "fix_scaffold_type": options["fix_scaffold_type"],
        "schedule": "lambda",
        "use_with_noise": True,
        "use_adaptive_step": True,
        "use_time_scale": False,
        "qed_weight": options["qed_weight"],
        "sa_weight": options["sa_weight"],
        "diversity_weight": options["diversity_weight"],
        "rmsd_penalty_weight": options["rmsd_penalty_weight"],
        "min_qed": options["min_qed"],
        "min_sa": options["min_sa"],
        "keep_original": True,
        "filter_incomplete": True,
        "diversity_filter": {
            "enable": options["diversity_filter_enable"],
            "max_tanimoto": options["max_tanimoto"],
            "fingerprint_radius": 2,
            "fingerprint_nbits": 2048,
        },
        "murcko_sites": {
            "p_active": 0.5,
            "max_per_site": 20,
            "min_per_site": 0,
            "per_site_count_mode": "sequential_random",
            "per_site_add_mode": "fragment_prior",
            "fragment_sizes": [
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                16,
                18,
                20,
            ],
            "fragment_weights": [
                0.02,
                0.02,
                0.02,
                0.02,
                0.03,
                0.16,
                0.10,
                0.09,
                0.08,
                0.12,
                0.12,
                0.10,
                0.05,
                0.04,
                0.015,
                0.01,
                0.005,
            ],
            "jitter_std": 1.0,
            "jitter_mode": "gaussian",
            "max_active_sites": "n_removed_sidechains",
            "preserve_zero_allocation_sidechains": True,
            "prefer_murcko_sites": True,
            "save_json": True,
            "reuse_cache": False,
            "deterministic_fallback": True,
            "overflow_mode": "cap",
            "dedup_dist": 0.5,
        },
        "adaptive_gates": {
            "enable": bool(options["adaptive_gates_enable"]),
            "ligand": options["ligand_path"] or options["molecule_path"],
            "actives": options["adaptive_gates_actives"],
        },
    }

    grow_block = {
        "seed": options["seed"],
        "n_extra_mode": options["grow_n_extra_mode"],
        "n_extra_fixed": options["grow_n_extra_fixed"],
        "n_extra_min": options["grow_n_extra_min"],
        "n_extra_max": options["grow_n_extra_max"],
        "n_extra_min_clamp": options["grow_n_extra_min_clamp"],
        "n_extra_max_clamp": options["grow_n_extra_max_clamp"],
        "num_samples": options["grow_num_samples"],
        "start_t": options["grow_start_t"],
        "stride": options["grow_stride"],
        "step_size": options["grow_step_size"],
        "lambda_coeff_a": options["grow_lambda_a"],
        "lambda_coeff_b": options["grow_lambda_b"],
        "min_qed": options["grow_min_qed"],
        "min_sa": options["grow_min_sa"],
        "filter_incomplete": options["grow_filter_incomplete"],
    }
    if options["grow_n_extra_mode"] == "fixed":
        # The explicit UI count must survive both global and site caps.
        budget = max(1, min(50, int(options["grow_n_extra_fixed"])))
        grow_block.update(
            {
                "n_extra_fixed": budget,
                "n_extra_min_clamp": budget,
                "n_extra_max_clamp": budget,
            }
        )
        scaffold_cfg["murcko_sites"]["max_per_site"] = max(20, budget)

    if options["scaffold_mode"] == "grow":
        scaffold_cfg["grow"] = grow_block
    elif options["scaffold_mode"] == "dynamic_locked":
        scaffold_cfg["fix_scaffold_pos"] = True
        scaffold_cfg["fix_scaffold_type"] = False
        scaffold_cfg["num_samples"] = options["grow_num_samples"]
        scaffold_cfg["grow"] = grow_block
    elif options["scaffold_mode"] == "prudent":
        # Gen0 uses dynamic_locked-style grow; then iterative renoise/refine
        scaffold_cfg["fix_scaffold_pos"] = True
        scaffold_cfg["fix_scaffold_type"] = False
        scaffold_cfg["num_samples"] = options["grow_num_samples"]
        scaffold_cfg["grow"] = grow_block
        scaffold_cfg["prudent"] = {
            "n_generations": options["prudent_n_generations"],
            "n_chains_per_seed": options["prudent_n_chains_per_seed"],
            "advance_top_k": options["prudent_advance_top_k"],
            "renoise_t": options["prudent_renoise_t"],
            # null → 用 scaffold.num_samples（= grow_num_samples / API N）
            "target_final_count": None,
            "max_final_fill_attempts": None,
            "qed_weight": options["prudent_qed_weight"],
            "sa_weight": options["prudent_sa_weight"],
            "vina_weight": options["prudent_vina_weight"],
            "lipinski_weight": options["prudent_lipinski_weight"],
            "lilly_demerit_weight": options["prudent_lilly_weight"],
            "vina_exhaustiveness": options["prudent_vina_exhaustiveness"],
            "min_qed_for_docking": options["prudent_min_qed_for_docking"],
            "min_sa_for_docking": options["prudent_min_sa_for_docking"],
            "advance_top_k_mode": "cap",
            "continue_when_no_advance": True,
            "fallback_keep_top_k": 1,
        }
        # 人工门控：写入固定阈值，并关闭自适应
        if (options["gate_mode"] or "adaptive").lower() == "manual" or not options[
            "adaptive_gates_enable"
        ]:
            scaffold_cfg["adaptive_gates"]["enable"] = False
            if options["manual_max_logp"] is not None:
                scaffold_cfg["prudent"]["max_logp"] = float(options["manual_max_logp"])
            if options["manual_max_molwt"] is not None:
                scaffold_cfg["prudent"]["max_molwt"] = float(
                    options["manual_max_molwt"]
                )
            if options["manual_max_heavy_atoms"] is not None:
                scaffold_cfg["prudent"]["max_heavy_atoms"] = int(
                    options["manual_max_heavy_atoms"]
                )
            if options["manual_max_rings"] is not None:
                scaffold_cfg["prudent"]["max_rings"] = int(options["manual_max_rings"])
            metadata["manual_gates"] = {
                "max_logp": options["manual_max_logp"],
                "max_molwt": options["manual_max_molwt"],
                "max_heavy_atoms": options["manual_max_heavy_atoms"],
                "max_rings": options["manual_max_rings"],
            }
        # 自适应：任务创建时按原始参考配体写入门控
        _ref_lig = options["ligand_path"] or options["molecule_path"]
        if (
            options["adaptive_gates_enable"]
            and (options["gate_mode"] or "adaptive").lower() != "manual"
            and _ref_lig
            and os.path.isfile(_ref_lig)
        ):
            try:
                from diffdynamic.sampling.gates import (
                    apply_gates_to_prudent_cfg,
                    compute_adaptive_gates,
                )

                _gates = compute_adaptive_gates(
                    _ref_lig,
                    actives_csv=options["adaptive_gates_actives"],
                    grow_cfg=grow_block,
                    base_prudent=scaffold_cfg["prudent"],
                )
                scaffold_cfg["prudent"] = apply_gates_to_prudent_cfg(
                    scaffold_cfg["prudent"],
                    _gates,
                )
                scaffold_cfg["prudent"].pop("_adaptive_gates", None)
                scaffold_cfg["adaptive_gates"]["ligand"] = _ref_lig
                scaffold_cfg["adaptive_gates"]["preview"] = _gates.as_prudent_dict()
                metadata["adaptive_gates"] = _gates.as_prudent_dict()
            except Exception as e:
                logger.warning("adaptive_gates pre-apply failed: %s", e)
    else:  # evolve
        scaffold_cfg["evolve"] = {
            "population_size": options["evolve_population_size"],
            "n_generations": options["evolve_n_generations"],
            "children_per_parent": options["evolve_children_per_parent"],
            "start_t_high": options["evolve_start_t_high"],
            "start_t_low": options["evolve_start_t_low"],
            "noise_anneal": True,
            "stride": options["evolve_stride"],
            "step_size": options["evolve_step_size"],
            "lambda_coeff_a": options["evolve_lambda_a"],
            "lambda_coeff_b": options["evolve_lambda_b"],
        }

    if options["placement"] and options["placement"].get("mode") == "custom":
        regions = options["placement"]["regions"]
        budget = sum(int(r["atom_count"]) for r in regions)
        scaffold_cfg["murcko_sites"]["custom_regions"] = regions
        scaffold_cfg["murcko_sites"]["preserve_zero_allocation_sidechains"] = False
        grow_block.update(
            {
                "n_extra_mode": "fixed",
                "n_extra_fixed": budget,
                "n_extra_min_clamp": budget,
                "n_extra_max_clamp": budget,
            }
        )
        scaffold_cfg["extra_atom_type_mode"] = "diffuse"
        scaffold_cfg["lock_extra_atom_types"] = False
    if options["workflow"] == "repaint":
        scaffold_cfg.update(
            {
                "fix_scaffold_pos": True,
                "fix_scaffold_type": True,
                "extra_atom_type_mode": "diffuse",
                "lock_extra_atom_types": False,
                "allow_soft_extra_anchor": False,
            }
        )
        grow_block.update(
            {
                "start_t": 999,
                "forward_noise_init": True,
                "forward_noise_init_mode": "full_q",
                "extra_anchor_strength": 0.0,
                "extra_type_anchor_strength": 0.0,
            }
        )
    sc_skip_refine = (
        False
        if options["scaffold_mode"] in ("dynamic_locked", "prudent")
        else options["scaffold_skip_refine"]
    )
    sc_patches = {
        "sample": {
            "seed": options["seed"],
            "scaffold": scaffold_cfg,
            "dynamic": {"skip_refine": sc_skip_refine},
            "targetdiff_baseline_refine": {
                "enable": options["scaffold_enable_refine"],
                "start_t": 9,
                "after_scaffold_init_only": False,
            },
        },
    }
    if options["workflow"] == "repaint":
        sc_patches["sample"]["dynamic"].update(
            {
                "refine": {
                    "max_grad_fusion_iterations": 30,
                    "preserve_schedule_endpoint": True,
                },
                "large_step": {"batch_size": min(options["grow_num_samples"], 5)},
            }
        )
        sc_patches["sample"]["targetdiff_baseline_refine"].update(
            {
                "start_t": 29,
                "lock_prefix": "types_and_pos",
            }
        )
    return sc_patches, metadata
