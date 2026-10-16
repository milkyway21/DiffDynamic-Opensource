"""Explicit operations consumed by fixed and terminal Prudent rounds."""


def round_helpers():
    from diffdynamic.sampling.prudent.state import _prudent_copy_candidate
    from diffdynamic.sampling.prudent.scoring import _prudent_parse_size_gates
    from diffdynamic.sampling.prudent.refinement import _prudent_refine_batch
    from diffdynamic.sampling.prudent.scoring import _prudent_score_all_chains
    from diffdynamic.sampling.prudent.scoring import _prudent_select_survivors
    from diffdynamic.sampling.refinement import apply_targetdiff_baseline_refinement

    return {
        '_prudent_copy_candidate': _prudent_copy_candidate,
        '_prudent_parse_size_gates': _prudent_parse_size_gates,
        '_prudent_refine_batch': _prudent_refine_batch,
        '_prudent_score_all_chains': _prudent_score_all_chains,
        '_prudent_select_survivors': _prudent_select_survivors,
        'apply_targetdiff_baseline_refinement': apply_targetdiff_baseline_refinement,
    }
