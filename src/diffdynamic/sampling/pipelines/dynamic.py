"""Sampling pipelines dynamic responsibilities for ligand sampling."""

from __future__ import annotations

import os


def run_dynamic(args, model, data, config, ligand_atom_mode, logger, pocket_id, result_path_abs, run_timestamp_cst, sampling_mode):
    """Run the dynamic pipeline and return its result bundle."""
    from diffdynamic.sampling.dynamic import sample_dynamic_diffusion_ligand
    from diffdynamic.sampling.prudent.state import _prudent_max_rounds_filename_tag

    config.sample.setdefault('dynamic', {})  # 如果不存在则创建空字典。
    config.sample.dynamic.setdefault('large_step', {})  # 确保大步配置存在。
    config.sample.dynamic['large_step'].setdefault('batch_size', args.batch_size)  # 设置大步批量大小。
    if sampling_mode == 'prudent':
        config.sample['mode'] = 'prudent'
        config.sample.dynamic.setdefault('prudent', {})
        config.sample.dynamic['prudent'].setdefault('enable', True)
    pr_dyn = config.sample.get('dynamic', {}).get('prudent', {})
    prudent_like = (
        sampling_mode == 'prudent'
        or config.sample.get('mode') == 'prudent'
        or bool(pr_dyn.get('enable'))
    )
    segment_ck_ctx = None
    _pr_seg_ev = prudent_like and bool(pr_dyn.get('segment_evolution_mode', False))
    _legacy_seg_ck = bool(pr_dyn.get('save_checkpoint_every_segment'))
    # OmegaConf.DictConfig 不是 dict，不可用 isinstance(..., dict) 判断「键是否存在」
    _ex_bp = pr_dyn is not None and ('save_segment_breakpoint_pt' in pr_dyn)
    _ex_gen = pr_dyn is not None and ('save_segment_generation_pool_pt' in pr_dyn)
    _want_bp = (
        bool(pr_dyn.get('save_segment_breakpoint_pt')) if _ex_bp else _legacy_seg_ck
    )
    # 分段演化：未显式配置时默认写每代全链 pool（与 save_checkpoint_every_segment 解耦）
    _want_gen = (
        bool(pr_dyn.get('save_segment_generation_pool_pt'))
        if _ex_gen
        else (_legacy_seg_ck or _pr_seg_ev)
    )
    if _pr_seg_ev and (_want_bp or _want_gen):
        ck_sub_raw = pr_dyn.get('segment_checkpoint_subdirectory', 'prudent_segment_ckpt')
        ck_sub = None if ck_sub_raw in (None, '', '~') else str(ck_sub_raw)
        ck_root = (
            os.path.join(result_path_abs, ck_sub)
            if ck_sub
            else result_path_abs
        )
        os.makedirs(ck_root, exist_ok=True)
        segment_ck_ctx = {
            'checkpoint_dir': ck_root,
            'pocket_id': str(pocket_id),
            'timestamp': run_timestamp_cst,
            'it_rounds_tag': _prudent_max_rounds_filename_tag(config),
        }
        if logger:
            logger.info(
                f'[Prudent] segment 落盘启用 → {ck_root} '
                f'(breakpoint_pt={_want_bp}, generation_pool_pt={_want_gen})'
            )
    # 执行动态采样（prudent 在 sample_dynamic_diffusion_ligand 内分支）。
    dynamic_output = sample_dynamic_diffusion_ligand(
        model=model,
        data=data,
        config=config,
        ligand_atom_mode=ligand_atom_mode,
        device=args.device,
        logger=logger,
        segment_checkpoint_context=segment_ck_ctx,
    )
    result = {
        'data': data,
        'pred_ligand_pos': dynamic_output['pos_list'],
        'pred_ligand_v': dynamic_output['v_list'],
        'pred_ligand_pos_traj': dynamic_output['pos_traj'],
        'pred_ligand_v_traj': dynamic_output['v_traj'],
        'pred_ligand_log_v_traj': dynamic_output['log_v_traj'],
        'time': dynamic_output['time_list'],
        'meta': dynamic_output['meta'],
        'init_ligand_pos': dynamic_output.get('init_list'),
        'mode': (
            'prudent' if (
                sampling_mode == 'prudent'
                or config.sample.get('mode') == 'prudent'
                or config.sample.get('dynamic', {}).get('prudent', {}).get('enable')
            ) else 'dynamic'
        ),
    }

    return result
