from pathlib import Path

from scripts.gspt1_scaffold_exact_loop import _variant_config


def _base_config():
    return {
        "sample": {
            "scaffold": {
                "murcko_sites": {
                    "per_site_count_mode": "sequential_random",
                },
                "grow": {},
            },
            "targetdiff_baseline_refine": {},
        }
    }


def _profile():
    return {
        "n_extra_values": [13, 14],
        "n_extra_weights": [1, 1],
    }


def test_weighted_single_disables_sequential_site_split():
    config = _variant_config(
        _base_config(),
        Path("profile.json"),
        "native_template",
        0.2,
        0.25,
        True,
        True,
        "prior_minus_scaffold",
        "weighted_single",
        _profile(),
    )
    sites = config["sample"]["scaffold"]["murcko_sites"]
    assert sites["site_selection_mode"] == "weighted_single"
    assert sites["per_site_count_mode"] == "split"


def test_legacy_site_mode_preserves_sequential_allocation():
    config = _variant_config(
        _base_config(),
        Path("profile.json"),
        "native_template",
        0.2,
        0.25,
        False,
        True,
        "prior_minus_scaffold",
        "legacy",
        _profile(),
    )
    sites = config["sample"]["scaffold"]["murcko_sites"]
    assert "site_selection_mode" not in sites
    assert sites["per_site_count_mode"] == "sequential_random"


def test_aggregate_type_prior_is_opt_in_on_scaffold_sites():
    config = _variant_config(
        _base_config(),
        Path("profile.json"),
        "native_template",
        0.2,
        0.25,
        True,
        True,
        "prior_minus_scaffold",
        "weighted_single",
        _profile(),
        extra_type_prior_strength=0.35,
    )
    sites = config["sample"]["scaffold"]["murcko_sites"]
    assert sites["reference_extra_type_prior_strength"] == 0.35


def test_targetdiff_baseline_override_sets_steps_and_position_only_lock():
    config = _variant_config(
        _base_config(),
        Path("profile.json"),
        "native_template",
        0.2,
        0.25,
        False,
        True,
        "prior_minus_scaffold",
        "legacy",
        _profile(),
        targetdiff_start_t=19,
        targetdiff_lock_prefix="pos_only",
    )
    refine = config["sample"]["targetdiff_baseline_refine"]
    assert refine["start_t"] == 19
    assert refine["lock_prefix"] == "pos_only"
