"""Batch responsibilities extracted from evaluate_pocket_quality.py."""

from diffdynamic.analysis.pocket.state import (
    BATCH_SCRIPT,
    DEFAULT_NUM_CPU_CORES,
    OUTPUT_DIR,
    REPO_ROOT,
    cpu_count,
    glob,
    os,
    subprocess,
    sys,
)


def run_batch_sampleandeval(
    start=1,
    end=99,
    gpus="0",
    num_cpu_cores=None,
    cores_per_task=1,
    protein_path=None,
    ligand_path=None,
    protein_root=None,
    sample_only=False,
):
    """调用 batch_sampleandeval_parallel.py 执行采样和评估"""
    if num_cpu_cores is None:
        num_cpu_cores = min(DEFAULT_NUM_CPU_CORES, cpu_count())
    if not BATCH_SCRIPT.exists():
        return False, [], f"脚本不存在: {BATCH_SCRIPT}"

    cmd = [sys.executable, str(BATCH_SCRIPT)]

    if protein_path:
        cmd.extend(["--protein_path", str(protein_path)])
        if ligand_path:
            cmd.extend(["--ligand_path", str(ligand_path)])
    else:
        cmd.extend(["--start", str(start), "--end", str(end)])

    cmd.extend(
        [
            "--gpus",
            gpus,
            "--num_cpu_cores",
            str(num_cpu_cores),
            "--cores_per_task",
            str(cores_per_task),
        ]
    )
    if protein_root:
        cmd.extend(["--protein_root", str(protein_root)])
    if sample_only:
        cmd.append("--sample-only")

    print(f"执行: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))
    except subprocess.CalledProcessError as e:
        return False, [], f"batch 脚本执行失败: {e}"

    if protein_path:
        pattern = str(OUTPUT_DIR / "result_custom_*.pt")
    else:
        pattern = str(OUTPUT_DIR / "result_*_*.pt")

    pt_files = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    return True, pt_files, "完成"


def _eval_single_pt_task(args):
    """Parallel evaluation worker for Pool.map (receives single tuple)"""
    from diffdynamic.analysis.pocket.pipeline import evaluate_pocket_quality

    (
        pt,
        protein_root,
        data_id,
        atom_mode,
        weight_a,
        weight_b,
        weight_c,
        weight_d,
        weight_e,
        weight_f,
        weight_g,
        weight_h,
        visualize,
        vis_dir,
        use_tsne,
        idea_b_atom_coord_subset,
        idea_b_edge_max_heavy_neighbors,
        idea_b_combined_focus_weight,
        vina_outputs_dir,
        vina_pocket_id,
        custom_pocket_pdb,
        fpocket_protein_pdb,
        fpocket_cmd,
        fpocket_pocket_index,
        fpocket_max_ligand_models,
        fpocket_timeout,
        fpocket_optimal_min,
        fpocket_optimal_max,
        fpocket_zero_below,
        fpocket_zero_above,
        idea_h_ligand_path,
        idea_e_expected_n_molecules,
    ) = args
    return evaluate_pocket_quality(
        pt_path=pt,
        protein_root=protein_root,
        data_id=data_id,
        atom_mode=atom_mode,
        weight_a=weight_a,
        weight_b=weight_b,
        weight_c=weight_c,
        weight_d=weight_d,
        weight_e=weight_e,
        weight_f=weight_f,
        weight_g=weight_g,
        weight_h=weight_h,
        idea_b_atom_coord_subset=idea_b_atom_coord_subset,
        idea_b_edge_max_heavy_neighbors=idea_b_edge_max_heavy_neighbors,
        idea_b_combined_focus_weight=idea_b_combined_focus_weight,
        vina_outputs_dir=vina_outputs_dir,
        vina_pocket_id=vina_pocket_id,
        custom_pocket_pdb=custom_pocket_pdb,
        visualize=visualize,
        vis_dir=vis_dir,
        use_tsne=use_tsne,
        fpocket_protein_pdb=fpocket_protein_pdb,
        fpocket_cmd=fpocket_cmd,
        fpocket_pocket_index=fpocket_pocket_index,
        fpocket_max_ligand_models=fpocket_max_ligand_models,
        fpocket_timeout=fpocket_timeout,
        fpocket_optimal_min=fpocket_optimal_min,
        fpocket_optimal_max=fpocket_optimal_max,
        fpocket_zero_below=fpocket_zero_below,
        fpocket_zero_above=fpocket_zero_above,
        idea_h_ligand_path=idea_h_ligand_path,
        idea_e_expected_n_molecules=idea_e_expected_n_molecules,
    )


def find_pt_files_for_range(start, end, output_dir=None):
    """根据 data_id 范围查找 .pt 文件"""
    output_dir = output_dir or OUTPUT_DIR
    found = []
    for i in range(start, end + 1):
        pattern = str(output_dir / f"result_{i}_*.pt")
        matches = glob.glob(pattern)
        if matches:
            found.append(max(matches, key=os.path.getmtime))
    return found
