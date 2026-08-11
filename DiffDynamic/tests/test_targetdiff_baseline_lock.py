from scripts.sample_diffusion import _baseline_refine_lock_restore_flags


def _config(lock_prefix):
    return {
        "sample": {
            "targetdiff_baseline_refine": {
                "lock_prefix": lock_prefix,
            },
        },
    }


def test_position_only_does_not_restore_atom_types():
    assert _baseline_refine_lock_restore_flags(_config("pos_only")) == (
        True,
        False,
    )


def test_existing_lock_modes_remain_unchanged():
    assert _baseline_refine_lock_restore_flags(_config("types_only")) == (
        False,
        True,
    )
    assert _baseline_refine_lock_restore_flags(_config("types_and_pos")) == (
        True,
        True,
    )
    assert _baseline_refine_lock_restore_flags(_config("none")) == (
        False,
        False,
    )
