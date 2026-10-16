"""Build the existing optimization schedule without queue or database access."""

from typing import Any, Dict, Mapping


def build_optimization_patches(options: Mapping[str, Any]) -> Dict[str, Any]:
    """Translate the persisted optimization options to their original YAML."""
    return {
        "sample": {
            "optimization": {
                "enable": True,
                "after_dynamic": options["after_dynamic"],
                "molecule_path": options["molecule_path"],
                "num_samples": options["num_samples"],
                "keep_original": True,
                "start_t": options["start_t"],
                "stride": options["stride"],
                "step_size": options["step_size"],
                "noise_scale": 0,
                "schedule": options["schedule"],
                "cycles": options["cycles"],
                "use_with_noise": True,
                "use_adaptive_step": True,
                "use_time_scale": False,
                "lambda_coeff_a": 1,
                "lambda_coeff_b": 1,
                "selector": {
                    "enable_filter": True,
                    "filter_incomplete": True,
                    "max_survivors_per_cycle": options["max_survivors_per_cycle"],
                    "max_survivors_total": options["max_survivors_total"],
                    "min_qed": options["min_qed"],
                    "min_sa": options["min_sa"],
                    "qed_weight": 1,
                    "sa_weight": 1,
                },
            },
        },
    }
