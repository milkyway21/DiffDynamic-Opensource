"""Sampling prudent evolution responsibilities for ligand sampling."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import os


def _run_prudent_segment_evolution_dynamic(
    model,
    data,
    config,
    ligand_atom_mode,
    ls_out,
    total_candidates,
    *,
    device='cuda:0',
    logger=None,
    skip_targetdiff_baseline_refine=False,
    segment_checkpoint_context=None,
):
    """Prudent Cumulative：跨代累积分段演化

    每代流程：checkpoint_frame → 剩余融合至本代终点；可选 ``save_segment_generation_pool_pt`` 将
    本代 refine + baseline 之后 **pool 内全部并行链**（``n_sampling`` 条）落盘；
    ``save_segment_breakpoint_pt`` 控制是否额外保存中段断点 ``d`` 的 tensor 快照。
    """
    from diffdynamic.sampling.candidates import evaluate_candidate
    from diffdynamic.sampling.configuration import get_time_boundary
    from diffdynamic.sampling.prudent.scoring import _prudent_score_one_round
    from diffdynamic.sampling.prudent.state import _prudent_candidate_traj_cleared
    from diffdynamic.sampling.prudent.state import _prudent_checkpoint_resume_t_upper
    from diffdynamic.sampling.prudent.state import _prudent_copy_candidate
    from diffdynamic.sampling.prudent.state import _prudent_forward_diffuse_to_t
    from diffdynamic.sampling.prudent.state import _prudent_save_iteration_torch_checkpoint
    from diffdynamic.sampling.prudent.state import _prudent_segment_generation_pt_seed_allowed
    from diffdynamic.sampling.prudent.state import _prudent_stride_gpu_chunk
    from diffdynamic.sampling.prudent.state import _prudent_terminal_time_upper
    from diffdynamic.sampling.prudent.state import _snapshot_prudent_generation
    from diffdynamic.sampling.refinement import _prudent_apply_targetdiff_baseline_to_pool
    from diffdynamic.sampling.trajectory import _coerce_time_indices_seq

    dynamic_cfg = config.sample.get('dynamic', {})
    refine_cfg = dynamic_cfg.get('refine', {})
    prudent_cfg = dynamic_cfg.get('prudent', {})

    time_boundary = get_time_boundary(dynamic_cfg, 750)
    time_lower_ref = int(refine_cfg.get('time_lower', 0))
    renoise_t = int(prudent_cfg.get('renoise_t', time_boundary))
    renoise_t = int(np.clip(renoise_t, 0, getattr(model, 'num_timesteps', 1000) - 1))

    # Cumulative 核心参数（只保留这些）
    n_sampling = max(1, int(prudent_cfg.get('n_sampling', 4)))
    max_segments = max(1, int(prudent_cfg.get('max_checkpoint_rounds', 5)))
    advance_top_k = max(1, int(prudent_cfg.get('advance_top_k', 1)))
    checkpoint_frame = max(1, int(prudent_cfg.get('checkpoint_frame', 2)))
    total_frames = max(1, int(prudent_cfg.get('total_frames', 24)))

    legacy_seg_disk = bool(prudent_cfg.get('save_checkpoint_every_segment'))
    explicit_bp_key = 'save_segment_breakpoint_pt' in prudent_cfg
    explicit_gen_key = 'save_segment_generation_pool_pt' in prudent_cfg
    save_bp_pt = (
        bool(prudent_cfg['save_segment_breakpoint_pt'])
        if explicit_bp_key
        else legacy_seg_disk
    )
    save_gen_pool_pt = (
        bool(prudent_cfg['save_segment_generation_pool_pt'])
        if explicit_gen_key
        else (legacy_seg_disk or True)
    )

    # 打分权重
    vina_weight = float(prudent_cfg.get('vina_weight', 0.86))
    qed_weight = float(prudent_cfg.get('qed_weight', 0.07))
    sa_weight = float(prudent_cfg.get('sa_weight', 0.07))
    vina_exhaustiveness = int(prudent_cfg.get('vina_exhaustiveness', 1))

    # 其他配置
    vina_advance_thr = prudent_cfg.get('max_vina_affinity_for_advance', None)
    if vina_advance_thr is not None:
        try:
            vina_advance_thr = float(vina_advance_thr)
        except (TypeError, ValueError):
            vina_advance_thr = None

    center_pos_mode = config.sample.get('center_pos_mode', 'protein')
    pos_only = config.sample.get('pos_only', False)

    # protein_root 用于 Vina
    protein_path = getattr(data, 'protein_filename', None)
    lig_fn = getattr(data, 'ligand_filename', None)
    vina_protein_root = prudent_cfg.get('protein_root') or os.environ.get('PROTEIN_ROOT', '').strip() or None
    if vina_protein_root:
        try:
            vina_protein_root = str(Path(vina_protein_root).expanduser().resolve())
        except Exception:
            pass

    protein_path = getattr(data, 'protein_filename', None)
    lig_fn = getattr(data, 'ligand_filename', None)
    vina_protein_root = prudent_cfg.get('protein_root') or os.environ.get('PROTEIN_ROOT', '').strip() or None
    if vina_protein_root:
        try:
            vina_protein_root = str(Path(vina_protein_root).expanduser().resolve())
        except Exception:
            pass
    vina_advance_thr = prudent_cfg.get('max_vina_affinity_for_advance', 4.0)
    if vina_advance_thr is not None:
        try:
            vina_advance_thr = float(vina_advance_thr)
        except (TypeError, ValueError):
            vina_advance_thr = 4.0

    center_pos_mode = config.sample.get('center_pos_mode', 'protein')
    pos_only = config.sample.get('pos_only', False)

    nti0 = total_candidates[0].get('time_indices') if total_candidates else None
    seq_ls = _coerce_time_indices_seq(nti0) if nti0 is not None else []
    n_ls_fusion = len(seq_ls)

    refined_records = []
    pending_analysis_hits = []
    segment_summaries_by_seed = []
    prudent_segment_ck_paths = []

    def _append_refined(seed_idx_local, pool_local):
        for ri, c in enumerate(pool_local):
            pr_det = c.pop('prudent_score_detail', {})
            c.pop('_seg_lineage_checkpoint', None)
            c.pop('_lineage_for_next', None)
            metric_info = evaluate_candidate(
                c['pos'], c['v'], ligand_atom_mode, dynamic_cfg.get('selector', {})
            )
            refined_records.append({
                'pos': c['pos'],
                'v': c['v'],
                'log_v': c['log_v'],
                'pos_traj': c.get('pos_traj') or [],
                'v_traj': c.get('v_traj') or [],
                'log_v_traj': c.get('log_v_traj') or [],
                'num_atoms': c['num_atoms'],
                'source_index': seed_idx_local,
                'repeat_index': ri,
                'time_indices': c.get('time_indices'),
                'prudent_composite_detail': pr_det,
                **metric_info,
            })

    # Cumulative 核心循环
    for seed_idx, seed_cand in enumerate(total_candidates):
        seg_log = []
        
        # 记录初始 seed_cand 状态
        if logger:
            logger.debug(
                f'[Prudent] SEED_INIT s{seed_idx} | '
                f'seed_cand pos_shape={seed_cand.get("pos", []).shape if hasattr(seed_cand.get("pos"), "shape") else "N/A"} | '
                f'v_dtype={seed_cand.get("v").dtype if hasattr(seed_cand.get("v"), "dtype") else "N/A"}'
            )
        
        pool = [
            _prudent_candidate_traj_cleared(
                _prudent_copy_candidate(seed_cand, logger=logger, context=f'g0_init'), 
                logger=logger, 
                context=f'g0_init_c{ci}'
            )
            for ci in range(n_sampling)
        ]
        current_t = int(time_boundary)
        final_pool_out = []

        for seg_idx in range(max_segments):
            # 全局时间轴：g0=0~26，g1=2~26，g2=4~26...
            # 每一代都固定先跑 checkpoint_frame 次融合并保存断点 d，下一代从「本代 winner 对应链」的 d 续跑。
            global_fusion_offset = seg_idx * checkpoint_frame
            remaining_steps = max(1, total_frames - global_fusion_offset)
            phase1_cap = max(1, min(checkpoint_frame, remaining_steps))

            if logger:
                logger.info(
                    f'[Prudent] g{seg_idx}s{seed_idx} t={current_t} '
                    f'global_offset={global_fusion_offset} remain={remaining_steps} phase1={phase1_cap}'
                )

            # 阶段1：先跑到断点 d（默认 2 次融合）
            pool_checkpoint = _prudent_stride_gpu_chunk(
                model, data, pool,
                time_upper=int(current_t),
                time_lower=time_lower_ref,
                stride_cap=phase1_cap,
                grad_anchor=None,
                renoise_t=renoise_t,
                refine_cfg=refine_cfg,
                dynamic_cfg=dynamic_cfg,
                center_pos_mode=center_pos_mode,
                pos_only=pos_only,
                device=device,
                logger=logger,
            )
            ck_tt = _prudent_terminal_time_upper(pool_checkpoint)

            # 保存断点 d（给下一代起点）
            if logger:
                logger.info(f'[Prudent] 存断点 s{seed_idx}g{seg_idx}f{phase1_cap} | {phase1_cap}次融合 | 下代续')
            if segment_checkpoint_context is not None and save_bp_pt:
                ck_path_frame = _prudent_save_iteration_torch_checkpoint(
                    segment_checkpoint_context,
                    data=data,
                    pool=pool_checkpoint,
                    seed_idx=seed_idx,
                    fname_middle_tail=f'seed{int(seed_idx)}_gen{int(seg_idx)}_frame{phase1_cap}',
                    checkpoint_kind='prudent_checkpoint',
                    scoring_snap=None,
                    seg_log_row={'gen': seg_idx, 'frame': phase1_cap},
                    prudent_max_iterations=max_segments,
                    ligand_atom_mode=ligand_atom_mode,
                    selector_cfg=dynamic_cfg.get('selector') or {},
                    logger=logger,
                )
                if ck_path_frame:
                    prudent_segment_ck_paths.append(os.path.abspath(ck_path_frame))

            # 阶段2：从断点 d 继续跑到本代终点
            remaining_to_end = max(0, remaining_steps - phase1_cap)
            if remaining_to_end > 0:
                pool_final = _prudent_stride_gpu_chunk(
                    model, data, pool_checkpoint,
                    time_upper=int(ck_tt) if ck_tt is not None else current_t,
                    time_lower=time_lower_ref,
                    stride_cap=remaining_to_end,
                    grad_anchor=None,
                    renoise_t=renoise_t,
                    refine_cfg=refine_cfg,
                    dynamic_cfg=dynamic_cfg,
                    center_pos_mode=center_pos_mode,
                    pos_only=pos_only,
                    device=device,
                    logger=logger,
                )
            else:
                pool_final = pool_checkpoint

            last_tt = _prudent_terminal_time_upper(pool_final)
            if last_tt is not None:
                last_tt = int(last_tt)

            _prudent_apply_targetdiff_baseline_to_pool(
                model,
                data,
                pool_final,
                config,
                device=device,
                logger=logger,
                skip=skip_targetdiff_baseline_refine,
                log_prefix=f's{seed_idx} gen{seg_idx}',
                round_idx=seg_idx,
                n_rounds=max_segments,
            )

            # 打分 & 选 winner
            pool_out, pend, broke = _prudent_score_one_round(
                pool_final,
                seed_idx=seed_idx,
                rnd=seg_idx,
                n_rounds=max_segments,
                advance_top_k=advance_top_k,
                prudent_cfg=prudent_cfg,
                protein_path=protein_path,
                lig_fn=lig_fn,
                vina_protein_root=vina_protein_root,
                ligand_atom_mode=ligand_atom_mode,
                vina_advance_thr=vina_advance_thr,
                logger=logger,
            )
            pending_analysis_hits.extend(pend)

            # 生成 scoring snapshot
            scoring_snap_full = None
            if pool_final:
                scoring_snap_full = _snapshot_prudent_generation(
                    seed_idx=seed_idx,
                    generation_idx=seg_idx,
                    restart_t_upper=int(current_t),
                    terminal_t_upper=last_tt,
                    pool=pool_final,
                    ligand_atom_mode=ligand_atom_mode,
                    selector_cfg=dynamic_cfg.get('selector') or {},
                )

            seg_log.append({
                'generation': seg_idx,
                'global_fusion_offset': global_fusion_offset,
                'checkpoint_frame': phase1_cap,
                'total_frames': total_frames,
                'remaining_steps': remaining_steps,
                'terminal_t': last_tt,
                'scoring_snapshot': scoring_snap_full,
            })

            # 每代末尾：写入本代全部并行链（refine + baseline 后，打分前）；见 save_segment_generation_pool_pt
            if (
                segment_checkpoint_context is not None
                and save_gen_pool_pt
                and _prudent_segment_generation_pt_seed_allowed(prudent_cfg, seed_idx)
            ):
                ck_path_gen = _prudent_save_iteration_torch_checkpoint(
                    segment_checkpoint_context,
                    data=data,
                    pool=pool_final,
                    seed_idx=seed_idx,
                    fname_middle_tail=f'gen{int(seg_idx)}_seed{int(seed_idx)}_chains{len(pool_final)}',
                    checkpoint_kind='prudent_generation_pool',
                    scoring_snap=scoring_snap_full,
                    seg_log_row={'gen': seg_idx, 'checkpoint_kind': 'prudent_generation_pool'},
                    prudent_max_iterations=max_segments,
                    ligand_atom_mode=ligand_atom_mode,
                    selector_cfg=dynamic_cfg.get('selector') or {},
                    logger=logger,
                )
                if ck_path_gen:
                    prudent_segment_ck_paths.append(os.path.abspath(ck_path_gen))
            elif segment_checkpoint_context is not None:
                ck_path_full = _prudent_save_iteration_torch_checkpoint(
                    segment_checkpoint_context,
                    data=data,
                    pool=pool_final,
                    seed_idx=seed_idx,
                    fname_middle_tail=f'seed{int(seed_idx)}_gen{int(seg_idx)}_full',
                    checkpoint_kind='prudent_full',
                    scoring_snap=scoring_snap_full,
                    seg_log_row={'gen': seg_idx, 'frame': 'full'},
                    prudent_max_iterations=max_segments,
                    ligand_atom_mode=ligand_atom_mode,
                    selector_cfg=dynamic_cfg.get('selector') or {},
                    logger=logger,
                )
                if ck_path_full:
                    prudent_segment_ck_paths.append(os.path.abspath(ck_path_full))

            if broke or not pool_out:
                final_pool_out = pool_out if pool_out else []
                break

            final_pool_out = pool_out

            # 准备下一代：读取 winner 对应链在断点 d 的状态并续跑
            winner = pool_out[0]
            winner_ana = winner.get('prudent_round_analysis') if isinstance(winner, dict) else None
            winner_chain_idx = 0
            if isinstance(winner_ana, dict):
                try:
                    winner_chain_idx = int(winner_ana.get('chain_idx', 0))
                except (TypeError, ValueError):
                    winner_chain_idx = 0
            
            # 记录调试信息
            if logger:
                logger.debug(
                    f'[Prudent] GEN_HANDOVER s{seed_idx} g{seg_idx}→g{seg_idx+1} | '
                    f'winner_chain_idx={winner_chain_idx} (from analysis) | '
                    f'pool_checkpoint_len={len(pool_checkpoint) if pool_checkpoint else 0} | '
                    f'winner pos_shape={winner.get("pos", []).shape if hasattr(winner.get("pos"), "shape") else "N/A"} | '
                    f'winner v_dtype={winner.get("v").dtype if hasattr(winner.get("v"), "dtype") else "N/A"}'
                )
            
            if pool_checkpoint:
                # 确保索引在有效范围内
                winner_chain_idx = int(np.clip(winner_chain_idx, 0, len(pool_checkpoint) - 1))
                checkpoint_cand = pool_checkpoint[winner_chain_idx]
                
                if logger:
                    logger.debug(
                        f'[Prudent] CHECKPOINT_SELECT s{seed_idx} g{seg_idx} | '
                        f'selected_idx={winner_chain_idx} | '
                        f'ck_pos_shape={checkpoint_cand.get("pos", []).shape if hasattr(checkpoint_cand.get("pos"), "shape") else "N/A"} | '
                        f'ck_v_dtype={checkpoint_cand.get("v").dtype if hasattr(checkpoint_cand.get("v"), "dtype") else "N/A"} | '
                        f'ck_v_range=[{checkpoint_cand.get("v").min() if hasattr(checkpoint_cand.get("v"), "min") else "N/A"}, '
                        f'{checkpoint_cand.get("v").max() if hasattr(checkpoint_cand.get("v"), "max") else "N/A"}]'
                    )
                
                winner_lineage = _prudent_copy_candidate(
                    checkpoint_cand, 
                    logger=logger, 
                    context=f'g{seg_idx}_winner_ck'
                )
            else:
                winner_lineage = _prudent_copy_candidate(
                    winner, 
                    logger=logger, 
                    context=f'g{seg_idx}_winner_final'
                )

            if logger:
                rt_next = _prudent_checkpoint_resume_t_upper(winner_lineage)
                logger.info(f'[Prudent] g{seg_idx} winner_chain={winner_chain_idx} @d(frame={phase1_cap}) → t{rt_next}')

            next_parent = _prudent_copy_candidate(
                winner_lineage, 
                logger=logger, 
                context=f'g{seg_idx+1}_parent'
            )
            next_parent.pop('_seg_lineage_checkpoint', None)
            next_parent.pop('_lineage_for_next', None)

            rt_next = _prudent_checkpoint_resume_t_upper(next_parent)
            if rt_next is None:
                rt_next = int(ck_tt) if ck_tt is not None else int(time_boundary)

            # 下一代起点：从断点 d 直接启动（取消显式加噪，噪声由refine配置处理）
            # 若需显式加噪，设置 generation_restart_renoise_t 或 generation_restart_noise_fusions
            t_max = int(getattr(model, 'num_timesteps', 1000) - 1)
            restart_renoise_t = prudent_cfg.get('generation_restart_renoise_t')
            restart_noise_fusions = prudent_cfg.get('generation_restart_noise_fusions')
            restart_t_upper = None
            if restart_renoise_t is not None:
                try:
                    restart_t_upper = int(np.clip(int(restart_renoise_t), 0, t_max))
                except (TypeError, ValueError):
                    restart_t_upper = None
            elif restart_noise_fusions is not None:
                try:
                    restart_t_upper = int(np.clip(int(rt_next) + max(0, int(restart_noise_fusions)), 0, t_max))
                except (TypeError, ValueError):
                    restart_t_upper = None
            if restart_t_upper is not None and restart_t_upper > int(rt_next):
                next_parent = _prudent_forward_diffuse_to_t(
                    model,
                    data,
                    next_parent,
                    restart_t_upper,
                    center_pos_mode,
                    pos_only,
                    device,
                    logger=logger,
                )
                current_t = int(restart_t_upper)
                if logger:
                    logger.info(
                        f'[Prudent] g{seg_idx + 1} 重启前加噪: t={rt_next} -> t={restart_t_upper}'
                    )
            else:
                current_t = int(rt_next)
                if logger and seg_idx < max_segments - 1:
                    logger.info(f'[Prudent] g{seg_idx + 1} 直接从断点 t={rt_next} 启动（无显式加噪）')

            if logger:
                logger.info(f'[Prudent] 续断点 s{seed_idx}g{seg_idx + 1} | f{phase1_cap}→余{remaining_steps - phase1_cap}融合')
            
            # 生成下一代 pool，记录详细调试
            pool = []
            for pi in range(n_sampling):
                copied = _prudent_copy_candidate(
                    next_parent, 
                    logger=logger, 
                    context=f'g{seg_idx+1}_pool_p{pi}'
                )
                pool.append(copied)
            
            # 记录 pool 状态摘要
            if logger:
                logger.debug(
                    f'[Prudent] POOL_READY s{seed_idx} g{seg_idx+1} | '
                    f'pool_size={len(pool)} | '
                    f'first_pos_shape={pool[0]["pos"].shape if pool else "N/A"} | '
                    f'first_v_dtype={pool[0]["v"].dtype if pool else "N/A"} | '
                    f'first_v_range=[{pool[0]["v"].min() if pool else "N/A"}, {pool[0]["v"].max() if pool else "N/A"}]'
                )

        _append_refined(seed_idx, final_pool_out if final_pool_out else [])
        segment_summaries_by_seed.append({'seed_idx': seed_idx, 'segments': seg_log})

    refined_pos_list = [rec['pos'] for rec in refined_records]
    refined_v_list = [rec['v'] for rec in refined_records]
    refined_pos_traj = [rec['pos_traj'] for rec in refined_records]
    refined_v_traj = [rec['v_traj'] for rec in refined_records]

    # meta_dict 精简版
    meta_dict = dict(ls_out.get('meta', {}))
    meta_dict['refined_candidates'] = refined_records
    prev_pr = meta_dict.get('prudent')
    pr_meta = dict(prev_pr) if isinstance(prev_pr, dict) else {}
    pr_meta.update({
        'cumulative_mode': True,
        'n_sampling': n_sampling,
        'max_generations': max_segments,
        'checkpoint_frame': checkpoint_frame,
        'total_frames': total_frames,
        'advance_top_k': advance_top_k,
        'weights': {
            'vina': vina_weight,
            'qed': qed_weight,
            'sa': sa_weight,
            'lipinski': float(prudent_cfg.get('lipinski_weight', 0.0)),
            'lilly_demerit': float(prudent_cfg.get('lilly_demerit_weight', 0.0)),
        },
        'vina_exhaustiveness': vina_exhaustiveness,
        'segment_summaries_by_seed': segment_summaries_by_seed,
        'pending_analysis_hits': pending_analysis_hits,
        'time_boundary': time_boundary,
        'renoise_t': renoise_t,
        'generation_restart_renoise_t': prudent_cfg.get('generation_restart_renoise_t'),
        'generation_restart_noise_fusions': prudent_cfg.get('generation_restart_noise_fusions'),
    })
    if prudent_segment_ck_paths:
        pr_meta['checkpoint_paths'] = prudent_segment_ck_paths
    meta_dict['prudent'] = pr_meta

    time_list = list(ls_out.get('time_list', []))

    return {
        'pos_list': refined_pos_list,
        'v_list': refined_v_list,
        'pos_traj': refined_pos_traj,
        'v_traj': refined_v_traj,
        'log_v_traj': [rec['log_v_traj'] for rec in refined_records],
        'time_list': time_list,
        'meta': meta_dict,
    }
