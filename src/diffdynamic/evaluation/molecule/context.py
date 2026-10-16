"""Explicit state passed between evaluation pipeline stages."""

from dataclasses import dataclass


@dataclass
class PreparedEvaluation:
    _vina_modes: object
    atom_mode: object
    buffer: object
    data: object
    data_id: object
    debug: object
    eval_output_dir: object
    eval_timestamp: object
    exhaustiveness: object
    keep_preopt_sdf: object
    ligand_filename: object
    meta_records: object
    n_poses: object
    n_scaffold_pt: object
    num_samples: object
    pre_docking_etkdg_reembed: object
    pre_docking_mmff_max_iters: object
    pre_docking_mmff_minimize: object
    pre_docking_use_uff_fallback: object
    pred_atom_types: object
    pred_positions: object
    protein_filename: object
    protein_root: object
    pt_mode: object
    pt_path: object
    ref_vina_score: object
    refined_for_pt_vina: object
    remove_fragments: object
    resolved_protein_path: object
    save_distribution_plots: object
    save_excel: object
    save_intermediate_interval: object
    save_sdf: object
    scaffold_bonds_pt: object
    scaffold_cfg_meta: object
    scaffold_substruct_mol: object
    score_category_dirs: object
    sdf_dir: object
    sdf_dir_optimized: object
    sdf_dir_original: object
    size_factor: object
    tmp_dir: object
    use_isolation: object
    use_split_sdf_dirs: object


@dataclass
class EvaluatedMolecules:
    failure_stats: object
    n_complete: object
    n_eval_success: object
    n_medchem_optimized: object
    n_reconstruct_success: object
    optimize_results: object
    reconstruct_failures: object
    results: object
