"""Sampling prudent refinement responsibilities for ligand sampling."""

from __future__ import annotations

from diffdynamic.data.graph import FOLLOW_BATCH
from torch_geometric.data import Batch
import numpy as np
import torch


def _batch_refinement_forward_split(
    model,
    data,
    cands,
    *,
    center_pos_mode,
    pos_only,
    device,
    time_upper,
    time_lower,
    refine_cfg,
    refinement_grad_kwargs,
    error_prefix='[Refine batch]',
    repaint_cfg=None,
):
    """与 large_step 同类：多候选拼 ``Batch``，一次 ``sample_diffusion_refinement``，再按配体原子数拆包。

    ``refinement_grad_kwargs`` 会并入 ``sample_diffusion_refinement``（如
    ``_dynamic_subcfg_grad_cap_kwargs(refine_cfg)`` 或 Prudent 的截断参数）。

    ``repaint_cfg``（可选）：传入 sample_diffusion_refinement 用于 scaffold_prudent 模式下的骨架锁定。

    Returns:
        list[dict]: 每条链 ``pos`` / ``v`` / ``log_v`` / ``num_atoms`` / 本段 ``pos_traj_ref`` 等（未与历史轨迹合并）。
    """
    from diffdynamic.sampling.common import safe_repeat_interleave

    if not cands:
        return []
    n = len(cands)
    batch = Batch.from_data_list(
        [data.clone() for _ in range(n)], follow_batch=FOLLOW_BATCH
    ).to(device)
    batch_protein = batch.protein_element_batch
    ligand_num_atoms = []
    for c in cands:
        na = int(np.asarray(c['pos']).shape[0])
        ligand_num_atoms.append(max(1, na))
    repeats_val = torch.tensor(ligand_num_atoms, device=device, dtype=torch.long)
    batch_ligand = safe_repeat_interleave(
        torch.arange(n, dtype=torch.long), repeats_val, device=device
    )
    init_pos = torch.cat(
        [
            torch.tensor(c['pos'], dtype=torch.float32, device=device)
            for c in cands
        ],
        dim=0,
    )
    init_log_v = torch.cat(
        [
            torch.tensor(c['log_v'], dtype=torch.float32, device=device)
            for c in cands
        ],
        dim=0,
    )
    if int(init_pos.shape[0]) != int(sum(ligand_num_atoms)):
        raise ValueError(
            f'{error_prefix} pos 总原子数与 ligand_num_atoms 之和不一致 '
            f'({init_pos.shape[0]} vs {sum(ligand_num_atoms)})'
        )

    if (getattr(model, 'ligand_v_input', 'onehot') == 'log_prob'
            or refinement_grad_kwargs.get('continuation', False)):
        init_input = init_log_v
        log_mode = 'log_prob'
    else:
        init_input = init_log_v.argmax(dim=-1)
        log_mode = 'auto'

    _refine_kwargs = dict(refinement_grad_kwargs)
    if repaint_cfg is not None:
        _refine_kwargs['repaint_cfg'] = repaint_cfg
    res = model.sample_diffusion_refinement(
        protein_pos=batch.protein_pos,
        protein_v=batch.protein_atom_feature.float(),
        batch_protein=batch_protein,
        init_ligand_pos=init_pos,
        init_ligand_v=init_input,
        batch_ligand=batch_ligand,
        center_pos_mode=center_pos_mode,
        pos_only=pos_only,
        step_stride=refine_cfg.get('stride'),
        step_size=refine_cfg.get('step_size'),
        add_noise=refine_cfg.get('noise_scale'),
        pos_clip=refine_cfg.get('pos_clip'),
        v_clip=refine_cfg.get('v_clip'),
        time_upper=time_upper,
        time_lower=time_lower,
        num_cycles=refine_cfg.get('cycles', 1),
        log_ligand_input_mode='log_prob' if log_mode == 'log_prob' else 'auto',
        **_refine_kwargs,
    )

    pos_all = res['pos'].detach().cpu().numpy().astype(np.float64)
    v_all = res['v'].detach().cpu().numpy()
    log_v_all = res['log_v'].detach().cpu().numpy().astype(np.float32)
    split_idx = np.cumsum(ligand_num_atoms[:-1])
    pos_chunks = np.split(pos_all, split_idx, axis=0)
    v_chunks = np.split(v_all, split_idx, axis=0)
    log_v_chunks = np.split(log_v_all, split_idx, axis=0)

    pos_traj_ref_per = [[] for _ in range(n)]
    log_v_traj_ref_per = [[] for _ in range(n)]
    for traj in res.get('pos_traj', []):
        parts = torch.split(traj, ligand_num_atoms, dim=0)
        for i in range(n):
            pos_traj_ref_per[i].append(
                parts[i].detach().cpu().numpy().astype(np.float64)
            )
    for traj in res.get('log_v_traj', []):
        parts = torch.split(traj, ligand_num_atoms, dim=0)
        for i in range(n):
            log_v_traj_ref_per[i].append(
                parts[i].detach().cpu().numpy().astype(np.float32)
            )
    v_traj_ref_per = [
        [x.argmax(axis=-1).astype(np.int64) for x in lv]
        for lv in log_v_traj_ref_per
    ]

    rti = res.get('time_indices')
    n_steps = len(rti) if rti is not None else 0

    out = []
    for i in range(n):
        out.append({
            'pos': pos_chunks[i],
            'v': v_chunks[i],
            'log_v': log_v_chunks[i],
            'num_atoms': ligand_num_atoms[i],
            'pos_traj_ref': pos_traj_ref_per[i],
            'v_traj_ref': v_traj_ref_per[i],
            'log_v_traj_ref': log_v_traj_ref_per[i],
            'rti': rti,
            'n_steps': n_steps,
        })
    return out


def _batch_refinement_forward_split_oom(
    model,
    data,
    cands,
    *,
    center_pos_mode,
    pos_only,
    device,
    time_upper,
    time_lower,
    refine_cfg,
    refinement_grad_kwargs,
    error_prefix='[Refine batch]',
    repaint_cfg=None,
    max_chunk=100,
    logger=None,
):
    """``_batch_refinement_forward_split`` + 分块 + CUDA OOM 减半重试。

    Fast 默认一块 100 链；大口袋超显存时自动对半拆分直至 1 链。
    """
    if not cands:
        return []
    try:
        chunk_size = int(max_chunk) if max_chunk is not None else len(cands)
    except (TypeError, ValueError):
        chunk_size = len(cands)
    if chunk_size <= 0:
        chunk_size = len(cands)

    def _once(chunk):
        return _batch_refinement_forward_split(
            model,
            data,
            chunk,
            center_pos_mode=center_pos_mode,
            pos_only=pos_only,
            device=device,
            time_upper=time_upper,
            time_lower=time_lower,
            refine_cfg=refine_cfg,
            refinement_grad_kwargs=refinement_grad_kwargs,
            error_prefix=error_prefix,
            repaint_cfg=repaint_cfg,
        )

    def _retry(cands_chunk, size):
        if size <= 0:
            size = 1
        if len(cands_chunk) <= size:
            try:
                return _once(cands_chunk)
            except torch.cuda.OutOfMemoryError:
                if len(cands_chunk) <= 1:
                    raise
                torch.cuda.empty_cache()
                mid = max(1, len(cands_chunk) // 2)
                if logger:
                    logger.warning(
                        f'{error_prefix} OOM：{len(cands_chunk)} 条链超显存，分块减半重试（→{mid}）'
                    )
                return _retry(cands_chunk[:mid], mid) + _retry(cands_chunk[mid:], mid)
        out = []
        for start in range(0, len(cands_chunk), size):
            out.extend(_retry(cands_chunk[start:start + size], size))
        return out

    return _retry(cands, chunk_size)


def _prudent_refine_chunk_size(dynamic_cfg, refine_cfg, n_total: int) -> int:
    """Prudent 单次 refine 前向的链数上限（显存保护）。

    ``refine_cfg.refine_batch_size`` 优先；否则取 ``dynamic.prudent.refine_batch_size``；
    未配置时全量一批（历史行为）。
    """
    if n_total <= 0:
        return 1
    prudent_cfg = (dynamic_cfg or {}).get('prudent', {}) or {}
    raw = refine_cfg.get('refine_batch_size', prudent_cfg.get('refine_batch_size'))
    if raw is None:
        return n_total
    if isinstance(raw, str) and raw.strip().lower() in ('', 'none', 'null'):
        return n_total
    try:
        v = int(raw)
    except (TypeError, ValueError):
        return n_total
    return max(1, min(n_total, v))


def _prudent_refine_batch(
    model, data, cands, dynamic_cfg, refine_cfg, center_pos_mode, pos_only, device,
    time_upper, time_lower, max_grad_cap, logger=None,
    grad_fusion_anchor_t=None,
    explicit_time_indices=None, continuation=False,
):
    """多链 refine：与 large_step 同类，``Batch.from_data_list`` 拼批后一次 ``sample_diffusion_refinement`` 前向。

    各链可具有不同配体原子数（``batch_ligand`` + ``safe_repeat_interleave``），再按原子数拆回每条链。

    支持 ``refine_cfg.refine_batch_size``（或 ``dynamic.prudent.refine_batch_size``）分块：
    批链数超过阈值时按该大小拆分为多次前向，避免 >50 链并行 refine 的显存峰值（OOM）。
    """
    from diffdynamic.sampling.trajectory import _coerce_time_indices_seq

    if not cands:
        return []
    if logger:
        logger.info(f'[Prudent] refine_exec t={time_upper}→{time_lower} cap={max_grad_cap} n={len(cands)}')
    gfa = grad_fusion_anchor_t if max_grad_cap is not None else None
    refinement_grad_kwargs = {
        'max_grad_fusion_iterations': max_grad_cap,
        'max_gradient_steps': refine_cfg.get('max_gradient_steps'),
        'grad_fusion_anchor_t': gfa,
    }
    if explicit_time_indices is not None:
        refinement_grad_kwargs['explicit_time_indices'] = explicit_time_indices
        refinement_grad_kwargs['continuation'] = bool(continuation)

    _chunk = _prudent_refine_chunk_size(dynamic_cfg, refine_cfg, len(cands))

    def _refine_once(cands_chunk):
        raw_list = _batch_refinement_forward_split(
            model,
            data,
            cands_chunk,
            center_pos_mode=center_pos_mode,
            pos_only=pos_only,
            device=device,
            time_upper=time_upper,
            time_lower=time_lower,
            refine_cfg=refine_cfg,
            refinement_grad_kwargs=refinement_grad_kwargs,
            error_prefix='[Prudent] refine batch',
        )
        chunk_out = []
        for i, cand in enumerate(cands_chunk):
            raw = raw_list[i]
            pos_traj_ref = raw['pos_traj_ref']
            v_traj_ref = raw['v_traj_ref']
            log_v_traj_ref = raw['log_v_traj_ref']
            rti = raw['rti']
            n_steps = raw['n_steps']

            lpt, lvt, llv, lti = (
                cand.get('pos_traj') or [],
                cand.get('v_traj') or [],
                cand.get('log_v_traj') or [],
                cand.get('time_indices'),
            )
            merged_pos_traj = list(lpt) + pos_traj_ref
            merged_v_traj = list(lvt) + v_traj_ref
            merged_log_v_traj = list(llv) + log_v_traj_ref
            lt_part = [] if lti is None else (_coerce_time_indices_seq(lti) or [])
            rt_part = [] if rti is None else (_coerce_time_indices_seq(rti) or [])
            merged_time_indices = lt_part + rt_part

            chunk_out.append({
                'pos': raw['pos'],
                'v': raw['v'],
                'log_v': raw['log_v'],
                'num_atoms': raw['num_atoms'],
                'pos_traj': merged_pos_traj,
                'v_traj': merged_v_traj,
                'log_v_traj': merged_log_v_traj,
                'time_indices': merged_time_indices if merged_time_indices else rti,
                'grad_fusion_steps': n_steps,
            })
        return chunk_out

    def _refine_with_oom_retry(cands_chunk, chunk_size):
        """单次 refine 前向；遇 CUDA OOM 时把分块减半重试，直至最小分块（1 链）。"""
        if chunk_size <= 0:
            chunk_size = 1
        if len(cands_chunk) <= chunk_size:
            try:
                return _refine_once(cands_chunk)
            except torch.cuda.OutOfMemoryError:
                if len(cands_chunk) <= 1:
                    raise
                torch.cuda.empty_cache()
                if logger:
                    logger.warning(
                        f'[Prudent] refine OOM：{len(cands_chunk)} 条链超显存，分块减半重试（→{max(1, len(cands_chunk) // 2)}）'
                    )
                mid = len(cands_chunk) // 2
                return _refine_with_oom_retry(cands_chunk[:mid], len(cands_chunk) // 2) + \
                       _refine_with_oom_retry(cands_chunk[mid:], len(cands_chunk) // 2)
        # 需要按 chunk_size 拆分后再跑（递归）
        out = []
        for start in range(0, len(cands_chunk), chunk_size):
            out.extend(_refine_with_oom_retry(
                cands_chunk[start:start + chunk_size], chunk_size))
        return out

    out = _refine_with_oom_retry(cands, _chunk)
    return out


def _prudent_refine_one(
    model, data, cand, dynamic_cfg, refine_cfg, center_pos_mode, pos_only, device,
    time_upper, time_lower, max_grad_cap, logger=None,
    grad_fusion_anchor_t=None,
):
    """单次 refine：等价于 ``_prudent_refine_batch(..., [cand])``。"""
    return _prudent_refine_batch(
        model, data, [cand], dynamic_cfg, refine_cfg,
        center_pos_mode, pos_only, device,
        time_upper=time_upper,
        time_lower=time_lower,
        max_grad_cap=max_grad_cap,
        logger=logger,
        grad_fusion_anchor_t=grad_fusion_anchor_t,
    )[0]
