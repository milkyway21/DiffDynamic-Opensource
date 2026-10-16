"""Single-file evaluation orchestration; stages carry explicit typed state."""


def evaluate_pt_file(
    pt_path,
    protein_root,
    output_dir=None,
    atom_mode="add_aromatic",
    exhaustiveness=8,
    n_poses=9,
    size_factor=1.0,
    buffer=5.0,
    max_samples=None,
    save_sdf=True,
    debug=False,
    tmp_dir=None,
    use_isolation=False,
    save_intermediate_interval=64,
    eval_parallel_workers=None,
    force_pre_docking_mmff=None,
    pre_docking_mmff_max_iters_override=None,
    receptor_pdb=None,
    index_pkl=None,
    data_id_override=None,
    benchmark_ligands_csv=None,
    reference_ligand_rel=None,
    vina_modes=None,
    save_distribution_plots=True,
    remove_fragments=True,
    keep_preopt_sdf="structure_only",
    save_excel=True,
):
    """
    评估整个.pt文件

    Args:
        save_sdf: 是否保存重建成功的分子为SDF文件（默认True）
        save_intermediate_interval: 每处理多少个分子保存一次中间结果；0=禁用中间保存（默认64）
        eval_parallel_workers: 对接阶段并行线程数；None 时读环境变量 EVAL_PARALLEL_WORKERS，否则默认 min(8,max(2,cpu))
        use_isolation: 是否为每个分子启子进程隔离（默认 False，同进程顺序评估）
        force_pre_docking_mmff: True/False 时覆盖 sampling.yml 中 sample.pre_docking.mmff_minimize；None 表示完全按配置
        pre_docking_mmff_max_iters_override: 非 None 时覆盖 MMFF 最大迭代次数
        receptor_pdb: 显式受体 PDB（绝对或相对路径）。无配体文件名的 .pt 可与本参数联用；若同时提供 index 解析结果，本参数优先覆盖受体路径。
        index_pkl: CrossDocked 的 index.pkl；与 data_id 联用，在 .pt 缺少 ligand_filename/protein_filename 时补全相对路径（相对 protein_root）。
        data_id_override: 强制指定 data_id（覆盖 .pt 内 extra_info 及文件名推断），用于 index 查表或记录。
        benchmark_ligands_csv: 如 docs/100ligands.csv：按**测试集** data_id 取 ligand_rel（与 test_set[data_id] 一致）。优先于 index.pkl（全库行号≠测试编号）。
        reference_ligand_rel: 自定义口袋评测时，参考配体相对 ``protein_root`` 的路径（如 shoc2ligand.sdf）；与 ``--receptor_pdb`` 联用，并跳过 CSV/index 补全配体。
        vina_modes: 可选 frozenset/set，元素为 dock / score_only / minimize；None 表示三种均运行（与历史默认一致）。
        save_distribution_plots: False 时不写入 bond_length_hist.png / pair_dist_hist.png 等聚合分布图。
        keep_preopt_sdf: 仅在 medchem 优化开启时生效，控制优化前分子如何保留：
            ``none`` 只留优化后分子；``structure_only`` 额外写 reconstructed_molecules_preopt/
            仅含 3D 结构（零额外计算）；``evaluated`` 优化前分子也跑一遍完整评估，
            Excel 出双份指标可直接对照（计算成本翻倍，含 Vina）。
        save_excel: 是否在评估目录写 evaluation_results_*.xlsx / complete_molecules_*.xlsx。
            Web 服务端默认关闭（数据已在 .pt/DB 中，导出走 /api/runs/{id}/export）；
            离线 CLI 默认开启，保持旧行为。

    Returns:
        dict: 包含所有评估结果和统计信息
    """
    from .preparation import prepare_evaluation
    from .execution import evaluate_molecules
    from .finalization import finalize_evaluation

    context = prepare_evaluation(
        pt_path=pt_path,
        protein_root=protein_root,
        output_dir=output_dir,
        atom_mode=atom_mode,
        exhaustiveness=exhaustiveness,
        n_poses=n_poses,
        size_factor=size_factor,
        buffer=buffer,
        max_samples=max_samples,
        save_sdf=save_sdf,
        debug=debug,
        tmp_dir=tmp_dir,
        use_isolation=use_isolation,
        save_intermediate_interval=save_intermediate_interval,
        eval_parallel_workers=eval_parallel_workers,
        force_pre_docking_mmff=force_pre_docking_mmff,
        pre_docking_mmff_max_iters_override=pre_docking_mmff_max_iters_override,
        receptor_pdb=receptor_pdb,
        index_pkl=index_pkl,
        data_id_override=data_id_override,
        benchmark_ligands_csv=benchmark_ligands_csv,
        reference_ligand_rel=reference_ligand_rel,
        vina_modes=vina_modes,
        save_distribution_plots=save_distribution_plots,
        remove_fragments=remove_fragments,
        keep_preopt_sdf=keep_preopt_sdf,
        save_excel=save_excel,
    )
    if context is None:
        return None
    evaluated = evaluate_molecules(context)
    return finalize_evaluation(context, evaluated)
