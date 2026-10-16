"""Sampling pipelines dynamic_then_optimization responsibilities for ligand sampling."""

from __future__ import annotations

from torch_geometric.transforms import Compose
import diffdynamic.data.transforms as trans


def run_dynamic_then_optimization(args, model, data, config, ligand_atom_mode, ligand_featurizer, logger):
    """Run the dynamic then optimization pipeline and return its result bundle."""
    from diffdynamic.sampling.dynamic import sample_dynamic_diffusion_ligand
    from diffdynamic.sampling.optimization import _create_data_from_generated_mol
    from diffdynamic.sampling.optimization import optimize_molecule
    from diffdynamic.sampling.refinement import apply_targetdiff_baseline_refine_to_sampling_result

    config.sample.setdefault('dynamic', {})
    config.sample.dynamic.setdefault('large_step', {})
    config.sample.dynamic['large_step'].setdefault('batch_size', args.batch_size)
    logger.info('[Dynamic+Optimization] 第一阶段：执行 dynamic 两阶段扩散（强制 legacy 模式，分子数由 large_step/refine 控制）')
    dynamic_output = sample_dynamic_diffusion_ligand(
        model=model,
        data=data,
        config=config,
        ligand_atom_mode=ligand_atom_mode,
        device=args.device,
        logger=logger,
        force_method='legacy',  # 避免使用 unified 的 num_samples 循环，确保 n_dyn = batch_size * n_repeat * n_sampling
        skip_targetdiff_baseline_refine=True,  # baseline refine 在优化结束后对所有输出分子统一执行
    )
    pos_list_dyn = dynamic_output['pos_list']
    v_list_dyn = dynamic_output['v_list']
    n_dyn_raw = len(pos_list_dyn)
    # sample.num_samples 在 baseline 下表示「每口袋分子数」；此处对齐为「最多参与优化的 dynamic 条数」
    num_samples_target = max(1, int(config.sample.get('num_samples', 100)))
    if n_dyn_raw > num_samples_target:
        logger.info(
            f'[Dynamic+Optimization] dynamic 阶段产出 {n_dyn_raw} 条，按 sample.num_samples={num_samples_target} '
            f'截断（legacy 原始条数由 large_step.batch_size×n_repeat×refine.n_sampling 决定，与 sample.num_samples 无关）'
        )
        pos_list_dyn = pos_list_dyn[:num_samples_target]
        v_list_dyn = v_list_dyn[:num_samples_target]
    n_dyn = len(pos_list_dyn)
    logger.info(f'[Dynamic+Optimization] 第一阶段完成: {n_dyn} 个分子（截断后上限 sample.num_samples={num_samples_target}）')
    logger.info('[Dynamic+Optimization] 第二阶段：对每个「RDKit 重建成功」的 dynamic 分子做 optimization')
    all_pos, all_v, all_pos_traj, all_v_traj, all_log_v_traj, all_time, all_meta = [], [], [], [], [], [], []
    optimization_time_indices = None
    mol_transform = Compose([ligand_featurizer, trans.FeaturizeLigandBond()])
    n_recon_ok = 0
    n_recon_fail = 0
    for i in range(n_dyn):
        data_i = _create_data_from_generated_mol(
            data, pos_list_dyn[i], v_list_dyn[i],
            ligand_atom_mode, mol_transform, logger
        )
        if data_i is None:
            n_recon_fail += 1
            if logger:
                logger.warning(f'[Dynamic+Optimization] 分子 {i+1}/{n_dyn} RDKit 重建失败，跳过优化')
            continue
        n_recon_ok += 1
        opt_out = optimize_molecule(model, data_i, config, ligand_atom_mode, device=args.device, logger=logger, opt_batch_size=1, n_dynamic_molecules=n_dyn)
        if optimization_time_indices is None:
            oti = opt_out['meta'].get('optimization_time_indices')
            if oti is not None:
                optimization_time_indices = oti
        all_pos.extend(opt_out['pos_list'])
        all_v.extend(opt_out['v_list'])
        all_pos_traj.extend(opt_out['pos_traj'])
        all_v_traj.extend(opt_out['v_traj'])
        all_log_v_traj.extend(opt_out['log_v_traj'])
        all_time.extend(opt_out['time_list'])
        for r in opt_out['meta']['records']:
            r['source_dynamic_idx'] = i
        all_meta.extend(opt_out['meta']['records'])
    n_final = len(all_pos)
    if logger:
        logger.info(
            f'[Dynamic+Optimization] 汇总: 最终写入 {n_final} 个分子 | '
            f'dynamic 条数={n_dyn} | RDKit 重建成功={n_recon_ok} | 重建失败跳过={n_recon_fail}'
        )
    if n_final < num_samples_target:
        logger.warning(
            f'[Dynamic+Optimization] 最终分子数 {n_final} < sample.num_samples={num_samples_target}。'
            f' 常见原因：① dynamic 后 RDKit 成键/重建失败过多；② optimization.selector 筛选过严；③ 仅蛋白时 prior 原子数不稳定。'
            f' 可尝试：临时设 optimization.enable=false 仅用 dynamic；或提供配体并设 sample_num_atoms=ref；'
            f'或放宽 optimization.selector（min_qed/min_sa、enable_filter）。'
        )
    meta_do = {
        'method': 'dynamic_then_optimization',
        'records': all_meta,
        'dynamic_then_optimization_stats': {
            'sample_num_samples_cap': num_samples_target,
            'dynamic_molecules_after_truncate': n_dyn,
            'reconstruct_success': n_recon_ok,
            'reconstruct_fail': n_recon_fail,
            'final_molecule_count': n_final,
        },
    }
    if optimization_time_indices is not None:
        meta_do['optimization_time_indices'] = optimization_time_indices
    result = {
        'data': data,
        'pred_ligand_pos': all_pos,
        'pred_ligand_v': all_v,
        'pred_ligand_pos_traj': all_pos_traj,
        'pred_ligand_v_traj': all_v_traj,
        'pred_ligand_log_v_traj': all_log_v_traj,
        'time': all_time,
        'meta': meta_do,
        'mode': 'dynamic_then_optimization'
    }
    apply_targetdiff_baseline_refine_to_sampling_result(
        model, data, result, config, device=args.device, logger=logger
    )

    return result
