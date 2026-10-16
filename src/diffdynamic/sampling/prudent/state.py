"""Sampling prudent state responsibilities for ligand sampling."""

from __future__ import annotations

from diffdynamic.data.graph import FOLLOW_BATCH
from diffdynamic.models.constants import GRAD_FUSION_CAP_UNSPECIFIED
from diffdynamic.models.probability import center_pos
from diffdynamic.models.probability import ensure_log_ligand
from diffdynamic.models.probability import index_to_log_onehot
from torch_geometric.data import Batch
import numpy as np
import os
import torch


def _prudent_copy_candidate(cand, logger=None, context=""):
    """深拷贝 large_step / 中间候选，供 prudent 多链并行。"""
    if logger:
        v_orig = cand.get('v')
        pos_orig = cand.get('pos')
        v_dtype = v_orig.dtype if hasattr(v_orig, 'dtype') else type(v_orig)
        v_min = v_orig.min() if hasattr(v_orig, 'min') else 'N/A'
        v_max = v_orig.max() if hasattr(v_orig, 'max') else 'N/A'
        pos_shape = pos_orig.shape if hasattr(pos_orig, 'shape') else 'N/A'
        logger.debug(f'[_prudent_copy_candidate] {context} | v_dtype={v_dtype}, v_range=[{v_min}, {v_max}], pos_shape={pos_shape}')
    
    v_array = np.asarray(cand['v'])
    if v_array.dtype != np.int64 and v_array.dtype != np.int32:
        v_array = v_array.astype(np.int64)
    
    out = {
        'pos': np.array(cand['pos'], copy=True, dtype=np.float64),
        'v': np.array(v_array, copy=True, dtype=np.int64),
        'num_atoms': int(cand.get('num_atoms', cand['pos'].shape[0])),
        'repeat': cand.get('repeat', 0),
        'time_indices': cand.get('time_indices'),
        'pos_traj': cand.get('pos_traj') or [],
        'v_traj': cand.get('v_traj') or [],
        'log_v_traj': cand.get('log_v_traj') or [],
    }
    for key in (
        'n_scaffold', 'n_preserved', 'n_locked', 'n_extra',
        'preserved_atom_indices', 'scaffold_bond_pairs',
    ):
        if key in cand and cand[key] is not None:
            out[key] = cand[key]
    if cand.get('log_v') is not None:
        out['log_v'] = np.array(cand['log_v'], copy=True, dtype=np.float32)
    sp = cand.get('prudent_renoise_snapshot_pos')
    slv = cand.get('prudent_renoise_snapshot_log_v')
    if sp is not None:
        out['prudent_renoise_snapshot_pos'] = np.array(sp, copy=True, dtype=np.float64)
    if slv is not None:
        out['prudent_renoise_snapshot_log_v'] = np.array(slv, copy=True, dtype=np.float32)
    
    if logger:
        logger.debug(f'[_prudent_copy_candidate] {context} OUTPUT | v_dtype={out["v"].dtype}, v_range=[{out["v"].min()}, {out["v"].max()}]')
    
    return out


def _prudent_attach_renoise_snapshot(cand, renoise_t, logger=None):
    """在 refine 轨迹上取最接近 renoise_t 的 (pos, log_v)，供 renoise_mode=checkpoint 轮次起点。"""
    from diffdynamic.sampling.trajectory import _coerce_time_indices_seq

    cand.pop('prudent_renoise_snapshot_pos', None)
    cand.pop('prudent_renoise_snapshot_log_v', None)
    ti = cand.get('time_indices')
    pt = cand.get('pos_traj') or []
    lvt = cand.get('log_v_traj') or []
    if not ti or not pt or not lvt:
        return
    try:
        tis = _coerce_time_indices_seq(ti)
    except Exception:
        return
    n = min(len(tis), len(pt), len(lvt))
    if n < 1:
        return
    tis, pt, lvt = tis[:n], pt[:n], lvt[:n]
    rt = int(renoise_t)
    best_i = min(range(n), key=lambda i: abs(int(tis[i]) - rt))
    cand['prudent_renoise_snapshot_pos'] = np.array(pt[best_i], copy=True, dtype=np.float64)
    cand['prudent_renoise_snapshot_log_v'] = np.array(lvt[best_i], copy=True, dtype=np.float32)


def _prudent_generation_restart_upper(gen_idx, time_boundary, renoise_t, time_lower_ref, model, prudent_cfg):
    """世代 g 的起扩散时间 t_upper：第 0 代自 time_boundary（large_step）；之后代默认自 renoise_t + g × stride。"""
    t_lim = getattr(model, 'num_timesteps', 1000) - 1
    if gen_idx <= 0:
        return int(np.clip(int(time_boundary), int(time_lower_ref) + 1, int(t_lim)))
    bt = prudent_cfg.get('generation_resume_base_t')
    base = int(renoise_t if bt is None else int(bt))
    try:
        stride = int(prudent_cfg.get('generation_resume_t_stride', prudent_cfg.get('generation_t_stride', 0)))
    except (TypeError, ValueError):
        stride = 0
    t = int(base + int(gen_idx) * stride)
    return int(np.clip(t, int(time_lower_ref) + 1, int(t_lim)))


def _prudent_full_generation_grad_cap(refine_cfg, prudent_cfg):
    """世代模式下每代单次 refine 的梯度融合迭代上限（与模型 ``_truncate_schedule_to_grad_fusion_iterations`` 一致）。"""
    from diffdynamic.sampling.configuration import _dynamic_subcfg_grad_cap_kwargs

    ov = prudent_cfg.get('generation_max_grad_fusion_iterations')
    if ov is not None:
        return max(1, int(ov))
    kv = refine_cfg.get('max_grad_fusion_iterations')
    if kv is not None:
        return max(1, int(kv))
    merged = _dynamic_subcfg_grad_cap_kwargs(refine_cfg)['max_grad_fusion_iterations']
    if merged is not GRAD_FUSION_CAP_UNSPECIFIED:
        try:
            return max(1, int(merged))
        except (TypeError, ValueError):
            pass
    return 24


def _prudent_candidate_traj_cleared(cand, logger=None, context=''):
    """新一世代 refine 的起点：清空历史轨迹与分析字段（坐标/类别保留）。"""
    o = _prudent_copy_candidate(cand, logger=logger, context=f'{context}->traj_cleared')
    o['pos_traj'] = []
    o['v_traj'] = []
    o['log_v_traj'] = []
    o['time_indices'] = None
    o.pop('prudent_score_detail', None)
    o.pop('prudent_round_analysis', None)
    o.pop('prudent_renoise_snapshot_pos', None)
    o.pop('prudent_renoise_snapshot_log_v', None)
    return o


def _prudent_forward_reset_pool(model, data, pool, t_upper, center_pos_mode, pos_only, device, logger=None):
    """上一世代幸存者列表 → 各自前向噪声到 ``t_upper``，作为下一轮 reverse refine 的起点。"""
    reset = []
    for c in pool:
        stripped = _prudent_candidate_traj_cleared(c)
        fwd = _prudent_forward_diffuse_to_t(
            model, data, stripped, int(t_upper),
            center_pos_mode, pos_only, device, logger=logger,
        )
        fwd['pos_traj'] = []
        fwd['v_traj'] = []
        fwd['log_v_traj'] = []
        fwd['time_indices'] = None
        fwd.pop('prudent_score_detail', None)
        fwd.pop('prudent_round_analysis', None)
        reset.append(fwd)
    return reset


def _snapshot_prudent_generation(
    *,
    seed_idx,
    generation_idx,
    restart_t_upper,
    terminal_t_upper,
    pool,
    ligand_atom_mode,
    selector_cfg,
):
    """将一代内各链打分摘要（可为后续导出「五代分析表」）序列化为纯 Python 字典。"""
    from diffdynamic.sampling.candidates import evaluate_candidate

    chains = []
    for c in pool:
        ev = evaluate_candidate(
            c['pos'], c['v'], ligand_atom_mode,
            selector_cfg if selector_cfg is not None else {},
        )
        chain_row = {
            'smiles': ev.get('smiles') or '',
            'status': ev.get('status'),
            'metrics': dict(ev.get('metrics') or {}),
            'score': ev.get('score'),
        }
        ra = c.get('prudent_round_analysis')
        if ra is not None:
            chain_row['prudent'] = dict(ra)
        chains.append({
            'chain': chain_row,
            'num_atoms': int(c.get('num_atoms', np.asarray(c['pos']).shape[0])),
        })
    return {
        'seed_idx': int(seed_idx),
        'generation': int(generation_idx),
        'restart_t_upper': int(restart_t_upper),
        'terminal_t_upper': None if terminal_t_upper is None else int(terminal_t_upper),
        'chains': chains,
    }


def _prudent_max_rounds_filename_tag(config):
    """与脚本末尾保存 ``result_<pocket>_it{N}_*.pt`` 时使用的 Prudent 轮数后缀一致。"""
    try:
        samp = getattr(config, 'sample', None)
        if samp is None:
            return None
        mode = samp.get('mode')
        dyn = samp.get('dynamic') or {}
        pr = dyn.get('prudent', {}) if isinstance(dyn, dict) else {}
        prudent_on = mode == 'prudent' or bool(pr.get('enable'))
        if not prudent_on:
            return None
        rnd = pr.get('max_checkpoint_rounds', pr.get('n_rounds'))
        if rnd is None:
            return None
        return max(1, int(rnd))
    except Exception:
        return None


def _prudent_ckpt_records_from_annotated_pool(pool, seed_idx, ligand_atom_mode, selector_cfg):
    """将打分后的 pool（未 pop 修改）转成与 ``refined_candidates`` 条目近似的可存档结构。"""
    from diffdynamic.sampling.candidates import evaluate_candidate

    sel = selector_cfg if selector_cfg is not None else {}
    rows = []
    for ri, cand in enumerate(pool):
        cd = cand
        pr_det = dict(cd.get('prudent_score_detail') or {})
        metric_info = evaluate_candidate(
            cd['pos'], cd['v'], ligand_atom_mode,
            sel,
        )
        log_v_np = cd.get('log_v')
        rows.append({
            'pos': np.asarray(cd['pos'], dtype=np.float64),
            # NumPy<2 不支持 np.asarray(..., copy=)；用 .copy() 保持独立副本
            'v': np.asarray(cd['v'], dtype=np.int64).copy(),
            'log_v': (
                np.asarray(log_v_np, dtype=np.float32).copy()
                if log_v_np is not None
                else None
            ),
            'pos_traj': list(cd.get('pos_traj') or []),
            'v_traj': list(cd.get('v_traj') or []),
            'log_v_traj': list(cd.get('log_v_traj') or []),
            'num_atoms': int(cd.get('num_atoms', np.asarray(cd['pos']).shape[0])),
            'source_index': int(seed_idx),
            'repeat_index': int(ri),
            'time_indices': cd.get('time_indices'),
            'prudent_composite_detail': pr_det,
            **metric_info,
        })
    return rows


def _prudent_save_iteration_torch_checkpoint(
    ctx,
    *,
    data,
    pool,
    seed_idx,
    fname_middle_tail,
    checkpoint_kind,
    scoring_snap,
    seg_log_row,
    prudent_max_iterations,
    ligand_atom_mode,
    selector_cfg,
    logger=None,
):
    """在单次 Prudent 「迭代」（分段演化的一段或 legacy 的一轮打分）结束时落盘一个独立 .pt。

    fname_middle_tail 例： ``seed0_seg2`` / ``seed0_rnd3`` ，拼入 ``result_<p>_it<R>_<ts>_<tail>.pt``。
    """
    if ctx is None:
        return None
    try:
        ck_dir = ctx['checkpoint_dir']
        pid = str(ctx['pocket_id'])
        ts = ctx['timestamp']
        itag = ctx.get('it_rounds_tag')
        if itag is not None:
            stem = f'result_{pid}_it{int(itag)}_{ts}_{fname_middle_tail}'
        else:
            stem = f'result_{pid}_{ts}_{fname_middle_tail}'
        fname = os.path.join(ck_dir, f'{stem}.pt')

        recs = _prudent_ckpt_records_from_annotated_pool(pool, seed_idx, ligand_atom_mode, selector_cfg)

        prudent_extra = {'checkpoint_max_iterations_hint': prudent_max_iterations}
        if isinstance(seg_log_row, dict):
            prudent_extra['segment_log_row_echo'] = seg_log_row

        bundle = {
            'checkpoint_kind': checkpoint_kind,
            'large_step_seed_idx': int(seed_idx),
            'data': data,
            'pred_ligand_pos': [np.asarray(r['pos']) for r in recs],
            'pred_ligand_v': [np.asarray(r['v']) for r in recs],
            'pred_ligand_pos_traj': [r.get('pos_traj') or [] for r in recs],
            'pred_ligand_v_traj': [r.get('v_traj') or [] for r in recs],
            'pred_ligand_log_v_traj': [r.get('log_v_traj') or [] for r in recs],
            'time': [0.0] * len(recs),
            'meta': {
                'refined_candidates': recs,
                'prudent': prudent_extra,
            },
            'scoring_snapshot_echo': scoring_snap,
            'mode': 'prudent_iteration_checkpoint',
        }
        torch.save(bundle, fname)
        if logger:
            logger.info(f'[Prudent][checkpoint] 迭代结束已写入 {fname}（kind={checkpoint_kind}）')
        return fname
    except Exception as e:
        if logger:
            logger.warning(f'[Prudent][checkpoint] 写入失败: {e}')
        return None


@torch.no_grad()
def _prudent_forward_diffuse_to_t(
    model, data, cand, t_upper, center_pos_mode, pos_only, device, logger=None,
):
    """将当前坐标/类型前向扩散到 t=t_upper（与训练 q 分布一致），用于轮次间「回到 time_boundary」。"""
    from diffdynamic.sampling.common import safe_repeat_interleave

    batch = Batch.from_data_list([data.clone()], follow_batch=FOLLOW_BATCH).to(device)
    batch_protein = batch.protein_element_batch
    num_atoms = max(1, int(cand.get('num_atoms', 10)))
    repeats_val = torch.tensor([num_atoms], device=device, dtype=torch.long)
    batch_ligand = safe_repeat_interleave(
        torch.arange(1, dtype=torch.long), repeats_val, device=device
    )
    init_pos = torch.tensor(cand['pos'], dtype=torch.float32, device=device)
    v_np = cand['v']
    init_v = torch.tensor(np.asarray(v_np), dtype=torch.long, device=device)
    log_mode = (
        'log_prob' if getattr(model, 'ligand_v_input', 'onehot') == 'log_prob' else 'auto'
    )
    if cand.get('log_v') is not None:
        init_log_v = torch.tensor(cand['log_v'], dtype=torch.float32, device=device)
        log_v0 = ensure_log_ligand(init_log_v, model.num_classes, mode=log_mode)
    else:
        log_v0 = index_to_log_onehot(init_v, model.num_classes)

    protein_pos, ligand_pos, offset = center_pos(
        batch.protein_pos, init_pos, batch_protein, batch_ligand, mode=center_pos_mode
    )
    num_graphs = batch_protein.max().item() + 1
    t = torch.full((num_graphs,), int(t_upper), dtype=torch.long, device=device)
    xt = model.q_pos_sample(ligand_pos, t, batch_ligand)
    v_idx, log_v_sample = model.q_v_sample(log_v0, t, batch_ligand)

    pos_unc = (xt + offset[batch_ligand]).detach().cpu().numpy().astype(np.float64)
    v_out = v_idx.detach().cpu().numpy()
    log_v_out = log_v_sample.detach().cpu().numpy().astype(np.float32)
    if logger:
        logger.debug(f'[Prudent] forward diffuse → t={t_upper} | atoms={num_atoms}')
    return {
        'pos': pos_unc,
        'v': v_out,
        'log_v': log_v_out,
        'num_atoms': num_atoms,
        'pos_traj': cand.get('pos_traj') or [],
        'v_traj': cand.get('v_traj') or [],
        'log_v_traj': cand.get('log_v_traj') or [],
        'time_indices': cand.get('time_indices'),
    }


def _prudent_terminal_time_upper(pool):
    """同批 refine 后各链 time_indices 末项应一致；取首链当前扩散时间下标作为下一段 ``time_upper``。"""
    from diffdynamic.sampling.trajectory import _coerce_time_indices_seq

    if not pool:
        return None
    ti = pool[0].get('time_indices')
    seq = _coerce_time_indices_seq(ti)
    if not seq:
        return None
    return int(seq[-1])


def _prudent_stride_gpu_chunk(
    model,
    data,
    pool,
    *,
    time_upper,
    time_lower,
    stride_cap,
    grad_anchor,
    renoise_t,
    refine_cfg,
    dynamic_cfg,
    center_pos_mode,
    pos_only,
    device,
    logger,
):
    """沿当前坐标连续 refine：``stride_cap`` 为单次梯度融合迭代上限；``None`` 表示跑满剩余调度。"""
    from diffdynamic.sampling.prudent.refinement import _prudent_refine_batch

    refined_list = _prudent_refine_batch(
        model,
        data,
        pool,
        dynamic_cfg,
        refine_cfg,
        center_pos_mode,
        pos_only,
        device,
        time_upper=int(time_upper),
        time_lower=int(time_lower),
        max_grad_cap=stride_cap,
        logger=logger,
        grad_fusion_anchor_t=grad_anchor,
    )
    pool_out = []
    for refined in refined_list:
        r = dict(refined)
        r.pop('grad_fusion_steps', None)
        pool_out.append(r)
    rt = int(np.clip(int(renoise_t), 0, getattr(model, 'num_timesteps', 1000) - 1))
    for ref in pool_out:
        _prudent_attach_renoise_snapshot(ref, rt, logger=logger)
    return pool_out


def _prudent_expand_survivors_for_stride(pool_survivors, n_sampling, *, seed_idx, chk, max_chk, logger):
    k_surv = len(pool_survivors)
    expanded = []
    for c in pool_survivors:
        for _ in range(n_sampling):
            expanded.append(_prudent_copy_candidate(c))
    if logger:
        logger.debug(
            f'[Prudent] seed={seed_idx} chk={chk + 1}/{max_chk}: '
            f'expand {k_surv} survivors × {n_sampling} = {len(expanded)} chains'
        )
    return expanded


def _prudent_candidate_from_pv(pos, v_np, model):
    """由坐标与离散类型构造 Prudent/refine 用候选字典（含 ``log_v``）。"""
    v_t = torch.tensor(np.asarray(v_np), dtype=torch.long)
    lv = index_to_log_onehot(v_t, model.num_classes).numpy().astype(np.float32)
    p = np.asarray(pos, dtype=np.float64)
    return {
        'pos': p,
        'v': np.asarray(v_np, dtype=np.int64),
        'log_v': lv,
        'num_atoms': int(p.shape[0]),
        'pos_traj': [],
        'v_traj': [],
        'log_v_traj': [],
        'time_indices': None,
    }


def _prudent_checkpoint_resume_t_upper(cand):
    """从候选 trajectory 读出当前离散扩散步下标（最后一条 ``time_indices``）。"""
    from diffdynamic.sampling.trajectory import _coerce_time_indices_seq

    ti = cand.get('time_indices')
    seq = _coerce_time_indices_seq(ti) if ti is not None else None
    if not seq:
        return None
    return int(seq[-1])


def _prudent_lineage_snapshot_from_refine_traj(cand, fusion_idx=0):
    """从单次 ``sample_diffusion_refinement``（含梯度融合截断）合并轨迹中取第 ``fusion_idx`` 帧作为下一段起点。

    与「650→606 记下一步」一致时取 ``fusion_idx=0``（轨迹首帧通常为第一步融合后的状态，视模型 ``record_traj`` 语义而定）。
    若轨迹不足则退回为当前 ``cand`` 深拷贝。"""
    from diffdynamic.sampling.trajectory import _coerce_time_indices_seq

    pt = cand.get('pos_traj') or []
    lvt = cand.get('log_v_traj') or []
    ti = cand.get('time_indices')
    if fusion_idx >= len(pt) or fusion_idx >= len(lvt):
        return _prudent_copy_candidate(cand)
    lv = np.asarray(lvt[fusion_idx], dtype=np.float32)
    if lv.ndim >= 2:
        v_flat = lv.argmax(axis=-1).astype(np.int64)
    else:
        v_flat = np.asarray(cand['v'], dtype=np.int64)
    ti_seq = _coerce_time_indices_seq(ti) if ti is not None else []
    ti_prefix = ti_seq[: fusion_idx + 1] if ti_seq else []
    pt_prefix = [np.asarray(p, dtype=np.float64).copy() for p in pt[: fusion_idx + 1]]
    lvt_prefix = [np.asarray(x, dtype=np.float32).copy() for x in lvt[: fusion_idx + 1]]
    vt_prefix = []
    for x in lvt_prefix:
        xa = np.asarray(x, dtype=np.float32)
        if xa.ndim >= 2:
            vt_prefix.append(xa.argmax(axis=-1).astype(np.int64))
        else:
            vt_prefix.append(np.asarray(cand['v'], dtype=np.int64))
    return {
        'pos': np.asarray(pt[fusion_idx], dtype=np.float64).copy(),
        'v': v_flat.copy(),
        'log_v': lv.copy(),
        'num_atoms': int(cand.get('num_atoms', np.asarray(pt[fusion_idx]).shape[0])),
        'pos_traj': pt_prefix,
        'v_traj': vt_prefix,
        'log_v_traj': lvt_prefix,
        'time_indices': ti_prefix if ti_prefix else cand.get('time_indices'),
    }


def _prudent_segment_generation_pt_seed_allowed(prudent_cfg, seed_idx):
    """是否与应写入「每代全链 pool」.pt（多 large_step seed 时可只写其中一部分）。"""
    raw = prudent_cfg.get('segment_generation_pt_seeds', 'all')
    if raw is None or raw in ('all', '*', ''):
        return True
    if isinstance(raw, str) and raw.strip().lower() in ('first', '0'):
        return int(seed_idx) == 0
    if isinstance(raw, (list, tuple, set)):
        try:
            allowed = {int(x) for x in raw}
        except (TypeError, ValueError):
            return True
        return int(seed_idx) in allowed
    return True
