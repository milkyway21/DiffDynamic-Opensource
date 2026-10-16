"""Sampling prudent legacy responsibilities for ligand sampling."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import os
import time
import torch


def _run_prudent_legacy_dynamic(
    model, data, config, ligand_atom_mode, device='cuda:0', logger=None,
    skip_targetdiff_baseline_refine=False,
    segment_checkpoint_context=None,
):
    """Prudent：large_step 后多链 refine。

    - **默认**（``prudent.full_generation_mode=false``）：自 ``time_boundary`` 向下连续 refine，每 ``checkpoint_every_grad_fusions``
      梯度融合后打分/晋级至多 ``max_checkpoint_rounds`` 次；用尽或到达 ``time_lower`` 后仍会视情况跑满末尾 refine。
    - **线程分段演化**（``segment_evolution_mode=true``）：第 0 段 reverse 上限 = ``refine.max_grad_fusion_iterations``（或 ``generation_max_grad_fusion_iterations``）；
      第 ``k`` 段为 ``max(1, 上限 − k × checkpoint_every_grad_fusions)``；lineage 取自本段轨迹第 ``segment_lineage_traj_index`` 帧；段末可选 ``segment_pre_baseline_forward_t``，再 ``targetdiff_baseline_refine``，再 Vina/QED/SA。``max_checkpoint_rounds`` 为分段轮数。
    """
    from diffdynamic.sampling.candidates import evaluate_candidate
    from diffdynamic.sampling.common import _log
    from diffdynamic.sampling.configuration import get_time_boundary
    from diffdynamic.sampling.dynamic import _run_legacy_dynamic
    from diffdynamic.sampling.prudent.evolution import _run_prudent_segment_evolution_dynamic
    from diffdynamic.sampling.prudent.scoring import _prudent_parse_size_gates
    from diffdynamic.sampling.prudent.scoring import _prudent_score_all_chains
    from diffdynamic.sampling.prudent.scoring import _prudent_score_one_round
    from diffdynamic.sampling.prudent.scoring import _prudent_select_survivors
    from diffdynamic.sampling.prudent.state import _prudent_candidate_traj_cleared
    from diffdynamic.sampling.prudent.state import _prudent_copy_candidate
    from diffdynamic.sampling.prudent.state import _prudent_expand_survivors_for_stride
    from diffdynamic.sampling.prudent.state import _prudent_forward_reset_pool
    from diffdynamic.sampling.prudent.state import _prudent_full_generation_grad_cap
    from diffdynamic.sampling.prudent.state import _prudent_generation_restart_upper
    from diffdynamic.sampling.prudent.state import _prudent_save_iteration_torch_checkpoint
    from diffdynamic.sampling.prudent.state import _prudent_stride_gpu_chunk
    from diffdynamic.sampling.prudent.state import _prudent_terminal_time_upper
    from diffdynamic.sampling.prudent.state import _snapshot_prudent_generation
    from diffdynamic.sampling.refinement import _prudent_apply_targetdiff_baseline_to_pool
    from diffdynamic.sampling.refinement import apply_targetdiff_baseline_refinement

    from diffdynamic.sampling.prudent.registry import round_helpers

    dynamic_cfg = config.sample.get('dynamic', {})
    refine_cfg = dynamic_cfg.get('refine', {})
    prudent_cfg = dynamic_cfg.get('prudent', {})
    time_boundary = get_time_boundary(dynamic_cfg, 750)
    renoise_t = int(prudent_cfg.get('renoise_t', time_boundary))
    renoise_t = int(np.clip(renoise_t, 0, getattr(model, 'num_timesteps', 1000) - 1))

    n_sampling = max(1, int(prudent_cfg.get('n_sampling', refine_cfg.get('n_sampling', 4))))
    advance_top_k = max(1, int(prudent_cfg.get('advance_top_k', min(2, n_sampling))))
    advance_top_k = min(advance_top_k, n_sampling)
    advance_top_k_mode = str(prudent_cfg.get('advance_top_k_mode', 'cap')).strip().lower()
    if advance_top_k_mode == 'cap' and prudent_cfg.get('advance_top_k_strict') is True:
        advance_top_k_mode = 'strict'
    if advance_top_k_mode in ('require_full_k', 'full_k'):
        advance_top_k_mode = 'strict'
    if advance_top_k_mode != 'strict':
        advance_top_k_mode = 'cap'

    center_pos_mode = config.sample.get('center_pos_mode', 'protein')
    pos_only = config.sample.get('pos_only', False)

    protein_path = getattr(data, 'protein_filename', None)
    lig_fn = getattr(data, 'ligand_filename', None)
    vina_protein_root = prudent_cfg.get('protein_root') or os.environ.get('PROTEIN_ROOT', '').strip() or None
    if vina_protein_root:
        try:
            vina_protein_root = str(Path(vina_protein_root).expanduser().resolve())
        except Exception:
            pass

    # 1) 仅 large_step：临时打开 skip_refine
    fixed_rounds = prudent_cfg.get('round_grad_fusion_steps') is not None
    if fixed_rounds and str(device).startswith('cuda'):
        torch.cuda.synchronize(device)
    fixed_ls_started = time.perf_counter()
    prev_skip = dynamic_cfg.get('skip_refine', False)
    dynamic_cfg['skip_refine'] = True
    try:
        ls_out = _run_legacy_dynamic(
            model, data, config, ligand_atom_mode,
            device=device, logger=logger, skip_targetdiff_baseline_refine=True,
        )
    finally:
        dynamic_cfg['skip_refine'] = prev_skip

    if fixed_rounds:
        if str(device).startswith('cuda'):
            torch.cuda.synchronize(device)
        ls_out['meta']['prudent_large_step_wall_s'] = (
            time.perf_counter() - fixed_ls_started
        )

    total_candidates = ls_out['meta'].get('large_step_candidates', [])
    if logger:
        logger.debug(f"[DEBUG] large_step_candidates count: {len(total_candidates)}")
    if logger:
        logger.info(f'[Prudent] large_step输出 {len(total_candidates)} 个候选，准备refine到t=0')
    if not total_candidates:
        _log(logger, 'warning', '[Prudent] large_step 无候选，回退为空结果')
        return {
            'pos_list': [],
            'v_list': [],
            'pos_traj': [],
            'v_traj': [],
            'log_v_traj': [],
            'time_list': ls_out.get('time_list', []),
            'meta': {'prudent': {'error': 'no_large_step_candidates'}, **ls_out.get('meta', {})},
        }

    if fixed_rounds:
        if prudent_cfg.get('gating_mode') == 'terminal_rollout_backtrack':
            from diffdynamic.sampling.prudent.terminal_rollout import run_terminal_rounds
            run_rounds = run_terminal_rounds
        else:
            from diffdynamic.sampling.prudent.round_sampling import run_fixed_rounds
            run_rounds = run_fixed_rounds

        return run_rounds(
            model, data, config, ligand_atom_mode, ls_out, total_candidates,
            helpers=round_helpers(), device=device, logger=logger,
            skip_targetdiff_baseline_refine=skip_targetdiff_baseline_refine,
        )

    seg_flag = prudent_cfg.get('segment_evolution_mode')
    if seg_flag is None:
        # 兼容：若配置显式给出 cumulative 关键键，则默认启用分段累计模式
        seg_flag = prudent_cfg.get('cumulative_mode')
        if seg_flag is None and ('checkpoint_frame' in prudent_cfg or 'total_frames' in prudent_cfg):
            seg_flag = True
    segment_evolution_mode = bool(seg_flag)
    if segment_evolution_mode:
        if logger:
            logger.info(
                '[Prudent] segment_evolution_mode=true：每段 reverse 上限 = '
                'max(1, refine.max_grad_fusion_iterations − seg×checkpoint_every_grad_fusions)→ 可选 '
                'segment_pre_baseline_forward_t → baseline_refine → Vina/QED/SA；下一段 lineage 取自轨迹帧'
            )
        return _run_prudent_segment_evolution_dynamic(
            model, data, config, ligand_atom_mode, ls_out, total_candidates,
            device=device, logger=logger,
            skip_targetdiff_baseline_refine=skip_targetdiff_baseline_refine,
            segment_checkpoint_context=segment_checkpoint_context,
        )

    try:
        stride_fusions = max(1, int(prudent_cfg.get('checkpoint_every_grad_fusions', 3)))
    except (TypeError, ValueError):
        stride_fusions = 3
    try:
        max_checkpoint_rounds = max(
            1,
            int(prudent_cfg.get('max_checkpoint_rounds', prudent_cfg.get('n_rounds', 5))),
        )
    except (TypeError, ValueError):
        max_checkpoint_rounds = 5
    time_lower_ref = int(refine_cfg.get('time_lower', 0))
    selector_cfg_dyn = dict(dynamic_cfg.get('selector') or {})

    full_generation_mode = bool(prudent_cfg.get('full_generation_mode', False))
    generation_tables_by_seed = []
    full_gen_grad_cap = (
        _prudent_full_generation_grad_cap(refine_cfg, prudent_cfg)
        if full_generation_mode
        else None
    )

    time_records = {'large_step': ls_out['meta'].get('time_records', {}).get('large_step', []), 'refine': []}

    refined_records = []
    pending_analysis_hits = []
    prudent_iter_ck_paths = []

    vina_advance_thr = prudent_cfg.get('max_vina_affinity_for_advance', 4.0)
    if vina_advance_thr is not None:
        try:
            vina_advance_thr = float(vina_advance_thr)
        except (TypeError, ValueError):
            vina_advance_thr = 4.0

    use_async_vina = (
        len(total_candidates) > 1
        and prudent_cfg.get('async_vina_across_seeds', True)
    )
    _mw = prudent_cfg.get('async_vina_max_workers')
    async_max_workers = (
        min(32, max(4, len(total_candidates)))
        if _mw is None
        else max(1, int(_mw))
    )

    def _append_refined_for_seed(seed_idx_local, pool_local):
        for ri, c in enumerate(pool_local):
            pr_det = c.pop('prudent_score_detail', {})
            metric_info = evaluate_candidate(
                c['pos'], c['v'], ligand_atom_mode, dynamic_cfg.get('selector', {}))
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

    if full_generation_mode and use_async_vina and logger:
        logger.warning(
            '[Prudent] full_generation_mode 与 async_vina_across_seeds 互不兼容；本运行改为按 seed 顺序执行（不走跨 seed Vina 线程池）。'
        )
    use_async_effective = bool(use_async_vina and not full_generation_mode)

    if segment_checkpoint_context is not None and (use_async_effective or full_generation_mode):
        if logger:
            logger.warning(
                '[Prudent] save_checkpoint_every_segment：当前分支为 '
                + ('async_vina_across_seeds' if use_async_effective else 'full_generation_mode')
                + '，暂不写入按迭代 .pt（仅 sequential 与 segment_evolution 支持）。'
            )

    if use_async_effective:
        n_seeds = len(total_candidates)
        pools = [
            [_prudent_copy_candidate(total_candidates[s]) for _ in range(n_sampling)]
            for s in range(n_seeds)
        ]
        current_ts = [int(time_boundary) for _ in range(n_seeds)]
        dead = [False] * n_seeds
        if logger:
            logger.info(
                f'[Prudent] async_vina_across_seeds：{n_seeds} 个 large_step 候选 | '
                f'checkpoint：每 {stride_fusions} 次梯度融合打分一次，至多 {max_checkpoint_rounds} 次 '
                f'（max_workers={async_max_workers}）'
            )

        def _active_seed_idx():
            return [s for s in range(n_seeds) if not dead[s]]

        for chk in range(max_checkpoint_rounds):
            actives = _active_seed_idx()
            if not actives:
                break
            # ---- 合并所有活 seed 的 pool 链为一个大 batch（GPU 一次前向，内部按 refine_batch_size 分块）----
            merged_pool = []
            seed_chain_counts = []
            for seed_idx in actives:
                pool = pools[seed_idx]
                merged_pool.extend(pool)
                seed_chain_counts.append(len(pool))
            if merged_pool:
                # 完整 refine 到 t=0：打分必须基于采样完成的 t=0 分子（QED/SA/Vina 才算得准）
                is_last = chk == max_checkpoint_rounds - 1
                _t_gpu0 = time.time()
                merged_refined = _prudent_stride_gpu_chunk(
                    model,
                    data,
                    merged_pool,
                    time_upper=int(time_boundary),
                    time_lower=time_lower_ref,
                    stride_cap=(refine_cfg.get('max_grad_fusion_iterations')
                                if max_checkpoint_rounds == 1 else None)
                    if is_last else stride_fusions,
                    grad_anchor=None,
                    renoise_t=renoise_t,
                    refine_cfg=refine_cfg,
                    dynamic_cfg=dynamic_cfg,
                    center_pos_mode=center_pos_mode,
                    pos_only=pos_only,
                    device=device,
                    logger=logger,
                )
                time_records.setdefault('refine', []).append(time.time() - _t_gpu0)
                # 切分回填到各 seed（保持顺序）
                _off = 0
                for k, seed_idx in enumerate(actives):
                    cnt = seed_chain_counts[k]
                    pools[seed_idx] = merged_refined[_off:_off + cnt]
                    _off += cnt
                for seed_idx in actives:
                    tt = _prudent_terminal_time_upper(pools[seed_idx])
                    if tt is not None:
                        current_ts[seed_idx] = int(tt)

            # 注：打分筛选前不做 targetdiff baseline；仅对筛选后的分子做一次结构修复（见函数末尾统一 baseline）
            # 打分：全链一次并行（scoring_workers，>=32 时用进程池绕开 GIL），再按 seed 分组选 survivor
            _sw = int(prudent_cfg.get('scoring_workers', 16) or 16)
            _t_cpu0 = time.time()
            pools_by_seed = {s: pools[s] for s in actives}
            scored_by_seed = _prudent_score_all_chains(
                pools_by_seed,
                chk=chk,
                n_rounds=max_checkpoint_rounds,
                prudent_cfg=prudent_cfg,
                protein_path=protein_path,
                lig_fn=lig_fn,
                vina_protein_root=vina_protein_root,
                ligand_atom_mode=ligand_atom_mode,
                size_gates=_prudent_parse_size_gates(prudent_cfg),
                min_q_dock=float(prudent_cfg.get('min_qed_for_docking', 0.3)),
                min_s_dock=float(prudent_cfg.get('min_sa_for_docking', 0.3)),
                scoring_workers=_sw,
                logger=logger,
                log_chains=prudent_cfg.get('log_per_chain_scores', True),
                use_process_pool=_sw >= 32,
            )
            for seed_idx in actives:
                scored_rows = scored_by_seed.get(seed_idx, [])
                pool_out, pend, broke = _prudent_select_survivors(
                    scored_rows,
                    seed_idx=seed_idx,
                    rnd=chk,
                    n_rounds=max_checkpoint_rounds,
                    advance_top_k=advance_top_k,
                    prudent_cfg=prudent_cfg,
                    vina_advance_thr=vina_advance_thr,
                    ligand_atom_mode=ligand_atom_mode,
                    logger=logger,
                )
                pending_analysis_hits.extend(pend)
                if broke:
                    dead[seed_idx] = True
                    pools[seed_idx] = []
                else:
                    pools[seed_idx] = pool_out
            time_records.setdefault('scoring', []).append(time.time() - _t_cpu0)

            for seed_idx in _active_seed_idx():
                ct = int(current_ts[seed_idx])
                last_chk = chk == max_checkpoint_rounds - 1
                at_end = ct <= time_lower_ref
                if last_chk or at_end:
                    continue
                pools[seed_idx] = _prudent_expand_survivors_for_stride(
                    pools[seed_idx],
                    n_sampling,
                    seed_idx=seed_idx,
                    chk=chk,
                    max_chk=max_checkpoint_rounds,
                    logger=logger,
                )

        # 注：chk 循环内最后一轮已完整 refine 到 t=0，此处无需重复 refine。
        for seed_idx in range(n_seeds):
            if dead[seed_idx]:
                continue
            _append_refined_for_seed(seed_idx, pools[seed_idx])
    elif full_generation_mode:
        if logger:
            logger.info(
                f'[Prudent] 世代模式：每代一次 refine，梯度融合迭代上限={full_gen_grad_cap} '
                f'（prudent.generation_max_grad_fusion_iterations 优先，否则 refine.max_grad_fusion_iterations）；'
                f'末端 Vina/QED/SA→晋级→扩链；代数>0 时前向加噪再 refine。'
                f'每代全链写入 meta.prudent.generation_tables_by_seed（至多 {max_checkpoint_rounds} 代）。'
            )
        for seed_idx, seed_cand in enumerate(total_candidates):
            gen_table_seed = []
            pool = [
                _prudent_candidate_traj_cleared(_prudent_copy_candidate(seed_cand))
                for _ in range(n_sampling)
            ]
            for gen_idx in range(max_checkpoint_rounds):
                t_upper = _prudent_generation_restart_upper(
                    gen_idx, time_boundary, renoise_t, time_lower_ref, model, prudent_cfg,
                )
                if gen_idx > 0:
                    pool = _prudent_forward_reset_pool(
                        model, data, pool, t_upper,
                        center_pos_mode, pos_only, device, logger=logger,
                    )
                pool = _prudent_stride_gpu_chunk(
                    model,
                    data,
                    pool,
                    time_upper=int(t_upper),
                    time_lower=time_lower_ref,
                    stride_cap=full_gen_grad_cap,
                    grad_anchor=None,
                    renoise_t=renoise_t,
                    refine_cfg=refine_cfg,
                    dynamic_cfg=dynamic_cfg,
                    center_pos_mode=center_pos_mode,
                    pos_only=pos_only,
                    device=device,
                    logger=logger,
                )
                tt = _prudent_terminal_time_upper(pool)
                _prudent_apply_targetdiff_baseline_to_pool(
                    model,
                    data,
                    pool,
                    config,
                    device=device,
                    logger=logger,
                    skip=skip_targetdiff_baseline_refine,
                    log_prefix=f'fullgen s{seed_idx} g{gen_idx}',
                    round_idx=gen_idx,
                    n_rounds=max_checkpoint_rounds,
                )
                pool_ann = pool
                pool_out, pend, broke = _prudent_score_one_round(
                    pool_ann,
                    seed_idx=seed_idx,
                    rnd=gen_idx,
                    n_rounds=max_checkpoint_rounds,
                    advance_top_k=advance_top_k,
                    prudent_cfg=prudent_cfg,
                    protein_path=protein_path,
                    lig_fn=lig_fn,
                    vina_protein_root=vina_protein_root,
                    ligand_atom_mode=ligand_atom_mode,
                    vina_advance_thr=vina_advance_thr,
                    logger=logger,
                )
                gen_table_seed.append(
                    _snapshot_prudent_generation(
                        seed_idx=seed_idx,
                        generation_idx=gen_idx,
                        restart_t_upper=t_upper,
                        terminal_t_upper=tt,
                        pool=pool_ann,
                        ligand_atom_mode=ligand_atom_mode,
                        selector_cfg=selector_cfg_dyn,
                    )
                )
                pending_analysis_hits.extend(pend)
                if broke or not pool_out:
                    pool = list(pool_out) if pool_out else []
                    break
                if gen_idx == max_checkpoint_rounds - 1:
                    pool = pool_out
                    break
                pool = _prudent_expand_survivors_for_stride(
                    pool_out,
                    n_sampling,
                    seed_idx=seed_idx,
                    chk=gen_idx,
                    max_chk=max_checkpoint_rounds,
                    logger=logger,
                )
            _append_refined_for_seed(seed_idx, pool)
            generation_tables_by_seed.append(gen_table_seed)
    else:
        for seed_idx, seed_cand in enumerate(total_candidates):
            pool = [_prudent_copy_candidate(seed_cand) for _ in range(n_sampling)]
            current_t = int(time_boundary)
            for chk in range(max_checkpoint_rounds):
                restart_for_chk = int(current_t)
                last_chk = chk == max_checkpoint_rounds - 1
                grad_anchor = None if chk == 0 else int(current_t)
                # 最后一轮完整 refine 到 t=0，保证 QED/SA/Vina 基于采样完成的分子打分
                pool = _prudent_stride_gpu_chunk(
                    model,
                    data,
                    pool,
                    time_upper=int(current_t),
                    time_lower=time_lower_ref,
                    stride_cap=(refine_cfg.get('max_grad_fusion_iterations')
                                if max_checkpoint_rounds == 1 else None)
                    if last_chk else stride_fusions,
                    grad_anchor=grad_anchor,
                    renoise_t=renoise_t,
                    refine_cfg=refine_cfg,
                    dynamic_cfg=dynamic_cfg,
                    center_pos_mode=center_pos_mode,
                    pos_only=pos_only,
                    device=device,
                    logger=logger,
                )
                tt = _prudent_terminal_time_upper(pool)
                if tt is not None:
                    current_t = int(tt)

                # 注：打分筛选前不做 targetdiff baseline；仅对筛选后的分子做一次结构修复（见函数末尾统一 baseline）
                pool, pend, broke = _prudent_score_one_round(
                    pool,
                    seed_idx=seed_idx,
                    rnd=chk,
                    n_rounds=max_checkpoint_rounds,
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
                if segment_checkpoint_context is not None and not use_async_effective and not full_generation_mode:
                    legacy_snap = _snapshot_prudent_generation(
                        seed_idx=seed_idx,
                        generation_idx=chk,
                        restart_t_upper=restart_for_chk,
                        terminal_t_upper=tt,
                        pool=pool,
                        ligand_atom_mode=ligand_atom_mode,
                        selector_cfg=selector_cfg_dyn,
                    )
                    ck_path = _prudent_save_iteration_torch_checkpoint(
                        segment_checkpoint_context,
                        data=data,
                        pool=pool,
                        seed_idx=seed_idx,
                        fname_middle_tail=f'seed{int(seed_idx)}_rnd{int(chk)}',
                        checkpoint_kind='prudent_legacy_sequential_post_scoring',
                        scoring_snap=legacy_snap,
                        seg_log_row={'rnd': chk, 'scoring_snapshot': legacy_snap},
                        prudent_max_iterations=max_checkpoint_rounds,
                        ligand_atom_mode=ligand_atom_mode,
                        selector_cfg=selector_cfg_dyn,
                        logger=logger,
                    )
                    if ck_path:
                        prudent_iter_ck_paths.append(os.path.abspath(ck_path))
                if broke:
                    pool = []
                    break

                at_end = int(current_t) <= int(time_lower_ref)
                if last_chk or at_end:
                    break

                pool = _prudent_expand_survivors_for_stride(
                    pool,
                    n_sampling,
                    seed_idx=seed_idx,
                    chk=chk,
                    max_chk=max_checkpoint_rounds,
                    logger=logger,
                )

            # 注：最后一轮已完整 refine 到 t=0，无需重复 refine。
            _append_refined_for_seed(seed_idx, pool)

    refined_pos_list = [rec['pos'] for rec in refined_records]
    refined_v_list = [rec['v'] for rec in refined_records]
    refined_pos_traj = [rec['pos_traj'] for rec in refined_records]
    refined_v_traj = [rec['v_traj'] for rec in refined_records]

    meta_baseline_refine_ti = None
    _t_bs0 = time.time()
    if not skip_targetdiff_baseline_refine:
        refine_out = apply_targetdiff_baseline_refinement(
            model, data, refined_pos_list, refined_v_list, config, device=device, logger=logger
        )
        if len(refine_out) == 5:
            refined_pos_list, refined_v_list, refine_pos_traj_add, refine_v_traj_add, refine_time_indices = refine_out
            refined_pos_traj = [list(pt) + list(rpt) for pt, rpt in zip(refined_pos_traj, refine_pos_traj_add)]
            refined_v_traj = [list(vt) + list(rvt) for vt, rvt in zip(refined_v_traj, refine_v_traj_add)]
            meta_baseline_refine_ti = refine_time_indices
        else:
            refined_pos_list, refined_v_list = refine_out
    time_records.setdefault('baseline', []).append(time.time() - _t_bs0)

    meta_dict = dict(ls_out.get('meta', {}))
    meta_dict['refined_candidates'] = refined_records
    pr_legacy_meta = {
        'n_sampling': n_sampling,
        'checkpoint_every_grad_fusions': stride_fusions,
        'max_checkpoint_rounds': max_checkpoint_rounds,
        'advance_top_k': advance_top_k,
        'advance_top_k_mode': advance_top_k_mode,
        'time_boundary': time_boundary,
        'renoise_t': renoise_t,
        'first_round_max_grad_fusion_iterations': prudent_cfg.get('first_round_max_grad_fusion_iterations'),
        'max_grad_fusion_iterations': prudent_cfg.get('max_grad_fusion_iterations'),
        'grad_fusion_total_budget': prudent_cfg.get('grad_fusion_total_budget'),
        'weights': {
            'vina': prudent_cfg.get('vina_weight', 0.6),
            'qed': prudent_cfg.get('qed_weight', 0.2),
            'sa': prudent_cfg.get('sa_weight', 0.2),
        },
        'min_qed_for_docking': float(prudent_cfg.get('min_qed_for_docking', 0.3)),
        'min_sa_for_docking': float(prudent_cfg.get('min_sa_for_docking', 0.3)),
        'max_vina_affinity_for_advance': vina_advance_thr,
        'pending_analysis_hits': pending_analysis_hits,
        'async_vina_across_seeds': bool(use_async_effective),
        'async_vina_max_workers': async_max_workers if use_async_effective else None,
        'full_generation_mode': bool(full_generation_mode),
        'generation_tables_by_seed': generation_tables_by_seed,
        'generation_resume_base_t': prudent_cfg.get('generation_resume_base_t'),
        'generation_resume_t_stride': prudent_cfg.get(
            'generation_resume_t_stride', prudent_cfg.get('generation_t_stride', 0)
        ),
        'generation_grad_fusion_cap': full_gen_grad_cap,
    }
    if prudent_iter_ck_paths:
        pr_legacy_meta['iteration_checkpoint_paths'] = prudent_iter_ck_paths
    meta_dict['prudent'] = pr_legacy_meta
    if meta_baseline_refine_ti is not None:
        meta_dict['baseline_refine_time_indices'] = meta_baseline_refine_ti

    time_list = list(ls_out.get('time_list', [])) + time_records.get('refine', [])

    # 汇总三项时间（GPU refine / CPU 打分 / baseline refine）并写入 meta + 日志
    _gpu_s = float(sum(time_records.get('refine', [])) or 0.0)
    _cpu_s = float(sum(time_records.get('scoring', [])) or 0.0)
    _bs_s = float(sum(time_records.get('baseline', [])) or 0.0)
    meta_dict['timing'] = {
        'gpu_s': round(_gpu_s, 2),
        'cpu_s': round(_cpu_s, 2),
        'baseline_s': round(_bs_s, 2),
        'total_s': round(_gpu_s + _cpu_s + _bs_s, 2),
    }
    if logger:
        logger.info(
            f'[Timing] gpu={_gpu_s:.1f}s cpu={_cpu_s:.1f}s baseline={_bs_s:.1f}s '
            f'total={_gpu_s + _cpu_s + _bs_s:.1f}s n_mol={len(refined_pos_list)}'
        )

    if logger:
        logger.info(f'[Prudent] refine完成 {len(refined_pos_list)} 个分子，准备保存PT')
    else:
        print(f"[DEBUG] prudent final refined_pos_list count: {len(refined_pos_list)}")
    return {
        'pos_list': refined_pos_list,
        'v_list': refined_v_list,
        'pos_traj': refined_pos_traj,
        'v_traj': refined_v_traj,
        'log_v_traj': [rec['log_v_traj'] for rec in refined_records],
        'time_list': time_list,
        'meta': meta_dict,
    }
