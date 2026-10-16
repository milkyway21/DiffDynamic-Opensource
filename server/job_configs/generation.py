"""Build default Dynamic and legacy Prudent generation patches."""

from typing import Any, Dict, Mapping


def build_generation_patches(options: Mapping[str, Any]) -> Dict[str, Any]:
    """Preserve the fast Dynamic override and legacy CLI Prudent defaults."""
    patches = {
        "sample": {
            "mode": options["mode"],
            "dynamic": {
                "prudent": {"enable": options["mode"] == "prudent"},
                "large_step": {"batch_size": int(options["batch_size"])},
            },
        }
    }
    if options["mode"] == "dynamic":
        from server.job_helpers import dynamic_web_patches

        patches = dynamic_web_patches(options["batch_size"])
    if options["mode"] == "prudent":
        patches["sample"]["dynamic"]["prudent"]["n_sampling"] = (
            int(options["prudent_n_sampling"]) if options["prudent_n_sampling"] else 10
        )
        patches["sample"]["dynamic"]["prudent"]["advance_top_k"] = (
            int(options["prudent_advance_top_k"])
            if options["prudent_advance_top_k"]
            else 2
        )
        patches["sample"]["dynamic"]["prudent"]["max_checkpoint_rounds"] = (
            int(options["prudent_max_checkpoint_rounds"])
            if options["prudent_max_checkpoint_rounds"]
            else 1
        )
        patches["sample"]["dynamic"]["prudent"]["refine_batch_size"] = (
            int(options["prudent_refine_batch_size"])
            if options["prudent_refine_batch_size"]
            else 250
        )
        patches["sample"]["dynamic"]["prudent"]["scoring_workers"] = (
            int(options["prudent_scoring_workers"])
            if options["prudent_scoring_workers"]
            else 100
        )
    return patches
