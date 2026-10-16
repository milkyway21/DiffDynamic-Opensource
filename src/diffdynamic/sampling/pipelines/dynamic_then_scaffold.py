"""Sampling pipelines dynamic_then_scaffold responsibilities for ligand sampling."""

from __future__ import annotations

from torch_geometric.transforms import Compose
import numpy as np
import diffdynamic.data.transforms as trans


def run_dynamic_then_scaffold(args, model, data, config, ligand_atom_mode, ligand_featurizer, logger):
    """Run the dynamic then scaffold pipeline and return its result bundle."""
    from diffdynamic.sampling.candidates import evaluate_candidate
    from diffdynamic.sampling.dynamic import sample_dynamic_diffusion_ligand
    from diffdynamic.sampling.optimization import _create_data_from_generated_mol
    from diffdynamic.sampling.refinement import apply_targetdiff_baseline_refine_to_sampling_result
    from diffdynamic.sampling.scaffold.evolve import scaffold_optimize_molecule
    from diffdynamic.sampling.scaffold.inputs import _get_scaffold_mode_cfg

    config.sample.setdefault('dynamic', {})
    config.sample.dynamic.setdefault('large_step', {})
    config.sample.dynamic['large_step'].setdefault('batch_size', args.batch_size)
    logger.info('[Dynamic+ScaffoldEvolve] 第一阶段：执行 dynamic 两阶段扩散...')
    dynamic_output = sample_dynamic_diffusion_ligand(
        model=model,
        data=data,
        config=config,
        ligand_atom_mode=ligand_atom_mode,
        device=args.device,
        logger=logger,
        force_method='legacy',
        skip_targetdiff_baseline_refine=True,
    )
    pos_list_dyn = dynamic_output['pos_list']
    v_list_dyn = dynamic_output['v_list']
    n_dyn = len(pos_list_dyn)
    logger.info(
        f'[Dynamic+ScaffoldEvolve] 第一阶段完成: {n_dyn} 个分子 '
        f'（由 large_step.batch_size × n_repeat × refine.n_sampling 决定）'
    )
    # 第二阶段：对每个 RDKit 重建成功的分子做骨架进化优化
    logger.info('[Dynamic+ScaffoldEvolve] 第二阶段：对每个分子进行骨架约束进化优化...')
    config.sample.setdefault('scaffold', {})
    sc_merged, _ = _get_scaffold_mode_cfg(config)
    save_dyn_pre = bool(sc_merged.get('save_dynamic_before_scaffold', False))
    opt_style_name = bool(sc_merged.get('optimization_style_naming', True))
    dyn_sel = {
        'min_qed': sc_merged.get('min_qed'),
        'min_sa': sc_merged.get('min_sa'),
        'qed_weight': sc_merged.get('qed_weight', 1.0),
        'sa_weight': sc_merged.get('sa_weight', 1.0),
        'filter_incomplete': False,
    }
    mol_transform = Compose([ligand_featurizer, trans.FeaturizeLigandBond()])
    all_pos, all_v, all_pos_traj, all_v_traj, all_log_v_traj, all_time, all_meta = [], [], [], [], [], [], []
    prepend_meta, prepend_pos, prepend_v = [], [], []
    if save_dyn_pre:
        for di in range(n_dyn):
            pos_np = np.asarray(pos_list_dyn[di], dtype=np.float64)
            v_np = np.asarray(v_list_dyn[di], dtype=np.int64)
            metric_info = evaluate_candidate(pos_np, v_np, ligand_atom_mode, dyn_sel)
            prepend_pos.append(pos_np)
            prepend_v.append(v_np)
            prepend_meta.append({
                'method': 'dynamic_then_scaffold',
                'stage': 'dynamic_pre_scaffold',
                'ligand_num_atoms': int(len(pos_np)),
                'time': 0.0,
                'rmsd_from_ref': 0.0,
                'smiles': metric_info.get('smiles'),
                'qed': metric_info.get('metrics', {}).get('qed'),
                'sa': metric_info.get('metrics', {}).get('sa'),
                'status': metric_info.get('status'),
                'is_original': True,
                'is_dynamic_raw': True,
                'source_dynamic_idx': di,
            })
        if logger:
            logger.info(
                f'[Dynamic+ScaffoldEvolve] 已按 save_dynamic_before_scaffold 追加 {len(prepend_meta)} 条 '
                f'dynamic 原始分子（拼在 pred 最前）；骨架阶段临时关闭 keep_original 以免与「原分子」身份证重复'
            )
    orig_keep = None
    if save_dyn_pre:
        orig_keep = config.sample['scaffold'].get('keep_original', True)
        config.sample['scaffold']['keep_original'] = False
    n_recon_ok = 0
    n_recon_fail = 0
    try:
        for i in range(n_dyn):
            data_i = _create_data_from_generated_mol(
                data, pos_list_dyn[i], v_list_dyn[i],
                ligand_atom_mode, mol_transform, logger
            )
            if data_i is None:
                n_recon_fail += 1
                if logger:
                    logger.warning(
                        f'[Dynamic+ScaffoldEvolve] 分子 {i+1}/{n_dyn} RDKit 重建失败，跳过'
                    )
                continue
            n_recon_ok += 1
            logger.info(
                f'[Dynamic+ScaffoldEvolve] 正在优化第 {n_recon_ok} 个分子 '
                f'（dynamic 第 {i+1}/{n_dyn} 条）...'
            )
            scaff_out = scaffold_optimize_molecule(
                model=model,
                data=data_i,
                config=config,
                ligand_atom_mode=ligand_atom_mode,
                device=args.device,
                logger=logger,
            )
            all_pos.extend(scaff_out['pos_list'])
            all_v.extend(scaff_out['v_list'])
            all_pos_traj.extend(scaff_out['pos_traj'])
            all_v_traj.extend(scaff_out['v_traj'])
            all_log_v_traj.extend(scaff_out['log_v_traj'])
            all_time.extend(scaff_out['time_list'])
            for r in scaff_out['meta']['records']:
                r['source_dynamic_idx'] = i
                if opt_style_name:
                    r['optimization_style_naming'] = True
            all_meta.extend(scaff_out['meta']['records'])
    finally:
        if save_dyn_pre and orig_keep is not None:
            config.sample['scaffold']['keep_original'] = orig_keep

    if save_dyn_pre and prepend_pos:
        z = len(prepend_pos)
        all_pos = prepend_pos + all_pos
        all_v = prepend_v + all_v
        all_pos_traj = [[] for _ in range(z)] + all_pos_traj
        all_v_traj = [[] for _ in range(z)] + all_v_traj
        all_log_v_traj = [[] for _ in range(z)] + all_log_v_traj
        all_time = [0.0] * z + all_time
        all_meta = prepend_meta + all_meta

    n_final = len(all_pos)
    if logger:
        logger.info(
            f'[Dynamic+ScaffoldEvolve] 汇总: 最终 {n_final} 个分子（含 dynamic 前置条数）| '
            f'dynamic 条数={n_dyn} | 重建成功={n_recon_ok} | 重建失败={n_recon_fail}'
        )
    meta_ds = {
        'method': 'dynamic_then_scaffold',
        'records': all_meta,
        'optimization_style_naming': opt_style_name,
        'excel_only_dynamic_raw': save_dyn_pre,
        'dynamic_then_scaffold_stats': {
            'dynamic_molecules': n_dyn,
            'reconstruct_success': n_recon_ok,
            'reconstruct_fail': n_recon_fail,
            'final_molecule_count': n_final,
            'prepended_dynamic_raw': len(prepend_meta),
        },
    }
    result = {
        'data': data,
        'pred_ligand_pos': all_pos,
        'pred_ligand_v': all_v,
        'pred_ligand_pos_traj': all_pos_traj,
        'pred_ligand_v_traj': all_v_traj,
        'pred_ligand_log_v_traj': all_log_v_traj,
        'time': all_time,
        'meta': meta_ds,
        'mode': 'dynamic_then_scaffold',
    }
    apply_targetdiff_baseline_refine_to_sampling_result(
        model, data, result, config, device=args.device, logger=logger
    )

    return result
