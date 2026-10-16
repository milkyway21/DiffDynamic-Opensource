"""Sampling refinement responsibilities for ligand sampling."""

from __future__ import annotations

from diffdynamic.models.probability import index_to_log_onehot
import numpy as np
import torch


def _targetdiff_baseline_refine_chunk_size(refine_cfg, n_total: int) -> int:
    """TargetDiff baseline 单次 ``sample_diffusion`` 堆叠的分子数。

    ``refine_batch_size``：``None`` / 省略 / 0 / ``null`` → 全量 ``n_total`` 一批；正整数则每批至多该条（省显存）。
    """
    if n_total <= 0:
        return 0
    raw = refine_cfg.get('refine_batch_size', refine_cfg.get('batch_size', None))
    if raw is None:
        return n_total
    if isinstance(raw, str) and raw.strip().lower() in ('', 'none', 'null'):
        return n_total
    try:
        b = int(raw)
    except (TypeError, ValueError):
        return n_total
    if b <= 0:
        return n_total
    return min(b, n_total)


def _targetdiff_run_baseline_this_prudent_round(refine_cfg, round_idx: int, n_rounds: int) -> bool:
    """Prudent 多轮中是否在本轮执行 baseline（``run_every_prudent_round``）。"""
    v = refine_cfg.get('run_every_prudent_round', True)
    if v is False or v == 0:
        return int(round_idx) == int(n_rounds) - 1
    if isinstance(v, str) and v.strip().lower() in ('false', '0', 'no', 'off'):
        return int(round_idx) == int(n_rounds) - 1
    return True


def apply_targetdiff_baseline_refinement(model, data, pos_list, v_list, config, device='cuda:0', logger=None):
    """在 dynamic 扩散后，用 TargetDiff 基准 DDPM 再跑若干步，修复原子位置与键角。

    将生成分子 x0 前向扩散到 t=999，再运行 sample_diffusion 反扩散 num_steps 步，
    利用 TargetDiff 标准扩散过程改善梯度融合模型可能产生的几何不合理问题。

    当 step_trajectory_save.enable=true 时，会记录 refine 阶段的轨迹供可视化。

    Args:
        model: 扩散模型（需实现 sample_diffusion 与 q_pos_sample）。
        data: 蛋白口袋数据（ProteinLigandData）。
        pos_list: 动态采样输出的位置列表 [n_mol][n_atoms_i, 3]。
        v_list: 动态采样输出的类别索引列表 [n_mol][n_atoms_i]。
        config: 采样配置，需包含 sample.targetdiff_baseline_refine。
        device: 运行设备。
        logger: 可选日志器。

    Returns:
        tuple: (refined_pos_list, refined_v_list) 或 (refined_pos_list, refined_v_list, pos_traj_list, v_traj_list, time_indices)
        默认返回五元组（含轨迹），便于位移/阶段分析；仅当无分子或未启用时返回原列表。
    """
    from diffdynamic.sampling.common import safe_repeat_interleave

    refine_cfg = config.sample.get('targetdiff_baseline_refine', {})
    if not refine_cfg.get('enable', False):
        return pos_list, v_list

    if not pos_list or len(pos_list) == 0:
        if logger:
            logger.info('[TargetDiff Refine] 无分子输出，跳过 baseline refine')
        return pos_list, v_list

    if not hasattr(model, 'sample_diffusion') or not hasattr(model, 'q_pos_sample'):
        if logger:
            logger.warning('[TargetDiff Refine] Model lacks sample_diffusion or q_pos_sample, skip refinement.')
        return pos_list, v_list

    # 始终记录轨迹供 .pt meta / 位移分析（与 step_trajectory_save SDF 可视化解耦）
    record_traj = True
    start_t = int(refine_cfg.get('start_t', 100))  # 低 t 区间，训练时学习修复键角/位置；步数=start_t+1
    pos_only = config.sample.get('pos_only', False)
    center_pos_mode = config.sample.get('center_pos_mode', 'protein')
    batch_size = _targetdiff_baseline_refine_chunk_size(refine_cfg, len(pos_list))

    t_start = min(start_t, model.num_timesteps - 1)  # 前向扩散到此 t，再反扩散到 t=0
    refined_pos_list, refined_v_list = [], []
    all_pos_traj, all_v_traj = [], []  # 仅当 record_traj 时填充

    def _to_tensor(x):
        if hasattr(x, 'cpu'):
            t = x if isinstance(x, torch.Tensor) else torch.tensor(x)
            return t.to(device)
        return torch.tensor(x, device=device)

    for batch_start in range(0, len(pos_list), batch_size):
        batch_end = min(batch_start + batch_size, len(pos_list))
        batch_pos = [np.asarray(pos_list[i], dtype=np.float64) for i in range(batch_start, batch_end)]
        batch_v = [np.asarray(v_list[i], dtype=np.int64) for i in range(batch_start, batch_end)]
        batch_counts = [len(p) for p in batch_pos]
        # 跳过空分子
        if sum(batch_counts) == 0:
            refined_pos_list.extend(batch_pos)
            refined_v_list.extend(batch_v)
            if record_traj:
                for _ in batch_counts:
                    all_pos_traj.append([])
                    all_v_traj.append([])
            continue

        # 直接构建批次张量，避免创建完整 ProteinLigandData
        n_mol = len(batch_counts)
        protein_pos = _to_tensor(data.protein_pos).float()
        protein_v = _to_tensor(data.protein_atom_feature).float()
        n_protein = protein_pos.shape[0]

        # 重复蛋白以匹配批次（每个分子共享同一蛋白）
        batch_protein = safe_repeat_interleave(
            torch.arange(n_mol, dtype=torch.long, device=device),
            [n_protein] * n_mol,
            device=device
        )
        protein_pos_batch = protein_pos.unsqueeze(0).expand(n_mol, -1, -1).reshape(-1, 3)
        feat_dim = protein_v.shape[-1] if protein_v.dim() > 1 else 1
        protein_v_batch = protein_v.unsqueeze(0).expand(n_mol, -1, feat_dim).reshape(-1, feat_dim)

        ligand_pos_list = [torch.tensor(p, dtype=torch.float32, device=device) for p in batch_pos]
        ligand_v_list = [torch.tensor(v, dtype=torch.long, device=device) for v in batch_v]
        batch_ligand = safe_repeat_interleave(
            torch.arange(n_mol, dtype=torch.long, device=device),
            batch_counts,
            device=device
        )
        ligand_pos = torch.cat(ligand_pos_list, dim=0)
        log_ligand_v = index_to_log_onehot(torch.cat(ligand_v_list, dim=0), model.num_classes)

        # 前向扩散到 t=t_start
        t_tensor = torch.full((n_mol,), t_start, dtype=torch.long, device=device)
        with torch.no_grad():
            xt = model.q_pos_sample(ligand_pos, t_tensor, batch_ligand)
            _, log_vt = model.q_v_sample(log_ligand_v, t_tensor, batch_ligand)

        # 运行 TargetDiff 基准扩散（从 start_t 反扩散到 t=0，利用低 t 区间的修复能力）
        with torch.no_grad():
            r = model.sample_diffusion(
                protein_pos=protein_pos_batch,
                protein_v=protein_v_batch,
                batch_protein=batch_protein,
                init_ligand_pos=xt,
                init_ligand_v=log_vt,
                batch_ligand=batch_ligand,
                num_steps=None,  # 使用 start_t 时由 start_t 决定步数
                center_pos_mode=center_pos_mode,
                pos_only=pos_only,
                start_t=t_start  # 从低 t 反扩散到 t=0，训练时最后几步学习修复
            )

        refined_pos = r['pos'].cpu().numpy().astype(np.float64)
        refined_v = r['v'].cpu().numpy().astype(np.int64)
        cum = np.cumsum([0] + batch_counts)
        for k in range(len(batch_counts)):
            refined_pos_list.append(refined_pos[cum[k]:cum[k + 1]])
            refined_v_list.append(refined_v[cum[k]:cum[k + 1]])

        if record_traj and r.get('pos_traj') and r.get('v_traj'):
            pos_traj_batch = r['pos_traj']
            v_traj_batch = r['v_traj']
            for k in range(len(batch_counts)):
                pk, pk1 = cum[k], cum[k + 1]
                mol_pos_traj = [p[pk:pk1].numpy().astype(np.float64) for p in pos_traj_batch]
                mol_v_traj = [v[pk:pk1].numpy().astype(np.int64) for v in v_traj_batch]
                all_pos_traj.append(mol_pos_traj)
                all_v_traj.append(mol_v_traj)

    if logger:
        logger.info(f'[TargetDiff Refine] Applied refinement from t={t_start} to t=0 ({t_start + 1} steps) for {len(pos_list)} molecules')
    if record_traj and all_pos_traj:
        time_indices = list(range(t_start, -1, -1))
        return refined_pos_list, refined_v_list, all_pos_traj, all_v_traj, time_indices
    return refined_pos_list, refined_v_list


def _restore_locked_prefix_after_refine(
    refined_pos_list,
    refined_v_list,
    orig_pos_list,
    orig_v_list,
    lock_counts,
    logger=None,
    tag='[TargetDiff Refine]',
    restore_pos=False,
    restore_type=True,
    type_lock_counts=None,
):
    """TargetDiff baseline 无 RePaint：按位置/类型各自的锁定长度还原。

    DiffDynamic 阶段用 pos_mask/type_mask 锁死骨架+零分配保留侧链；baseline
    ~10 步会改动全部原子。默认只还原原子类型（``restore_type=True``），
    **不还原坐标**（``restore_pos=False``）：允许 refine 微调位置/键角，
    但禁止骨架/原自由基段改成别的元素。严格 scaffold 模式可通过
    ``type_lock_counts`` 恢复全部新增原子类型，同时仍让新增坐标自由。

    说明：扩散状态无显式键；键在 OpenBabel/RDKit 重构时由坐标推断。
    锁类型 ≠ 锁键拓扑；位置微移后个别键连通可能变化。
    """
    if not lock_counts or not refined_pos_list:
        return refined_pos_list, refined_v_list
    if not restore_pos and not restore_type:
        return refined_pos_list, refined_v_list
    n_restored = 0
    out_pos, out_v = [], []
    for i, (rp, rv) in enumerate(zip(refined_pos_list, refined_v_list)):
        n_pos_lock = int(lock_counts[i]) if i < len(lock_counts) else 0
        n_type_lock = (
            int(type_lock_counts[i])
            if type_lock_counts is not None and i < len(type_lock_counts)
            else n_pos_lock
        )
        if n_pos_lock <= 0 and n_type_lock <= 0:
            out_pos.append(rp)
            out_v.append(rv)
            continue
        op = np.asarray(orig_pos_list[i], dtype=np.float64)
        ov = np.asarray(orig_v_list[i], dtype=np.int64)
        rp = np.asarray(rp, dtype=np.float64).copy()
        rv = np.asarray(rv, dtype=np.int64).copy()
        n_pos_lock = min(n_pos_lock, rp.shape[0], op.shape[0])
        n_type_lock = min(n_type_lock, rv.shape[0], ov.shape[0])
        if n_pos_lock > 0 or n_type_lock > 0:
            if restore_pos:
                rp[:n_pos_lock] = op[:n_pos_lock]
            if restore_type:
                rv[:n_type_lock] = ov[:n_type_lock]
            n_restored += 1
        out_pos.append(rp)
        out_v.append(rv)
    if logger and n_restored:
        mode = []
        if restore_type:
            mode.append('类型')
        if restore_pos:
            mode.append('坐标')
        logger.info(
            f'{tag} 已还原 {n_restored}/{len(refined_pos_list)} 个分子 locked 前缀的'
            f'{"+".join(mode)}（骨架+可选保留侧链；位置{"未" if not restore_pos else "已"}锁）'
        )
    return out_pos, out_v


def _baseline_refine_lock_restore_flags(config):
    """读 ``sample.targetdiff_baseline_refine.lock_prefix``。

    - ``types_only``（默认）：只锁原子类型，位置交给 refine
    - ``types_and_pos``：类型+坐标都还原（旧行为）
    - ``pos_only``：只还原坐标，允许 refine 改变原子类型
    - ``none``：不还原
    """
    tbr = {}
    if config is not None:
        sample = getattr(config, 'sample', None)
        if sample is not None:
            tbr = sample.get('targetdiff_baseline_refine', {}) or {}
        elif isinstance(config, dict):
            tbr = (config.get('sample') or {}).get('targetdiff_baseline_refine', {}) or {}
    mode = str(tbr.get('lock_prefix', 'types_only')).strip().lower()
    if mode in ('none', 'off', 'false', '0'):
        return False, False
    if mode in ('types_and_pos', 'pos_and_type', 'both', 'all'):
        return True, True
    if mode in ('pos_only', 'position_only', 'positions_only'):
        return True, False
    # types_only / type / types / default
    return False, True


def _prudent_apply_targetdiff_baseline_to_pool(
    model,
    data,
    pool,
    config,
    device='cuda:0',
    logger=None,
    *,
    skip=False,
    log_prefix='',
    round_idx=None,
    n_rounds=None,
):
    """Prudent：每代 GPU refine 结束后、Vina/QED/SA 前，按 ``sample.targetdiff_baseline_refine`` 更新 pool。

    ``skip_targetdiff_baseline_refine=True``（如 chained optimization 前置）时不执行；仅 ``enable: true``
    时调用 ``apply_targetdiff_baseline_refinement``。

    ``round_idx`` / ``n_rounds``：若配置 ``run_every_prudent_round: false``，仅在 ``round_idx == n_rounds - 1`` 时执行。
    """
    if skip or not pool:
        return pool
    tbr = config.sample.get('targetdiff_baseline_refine', {})
    if not tbr.get('enable', False):
        return pool
    if round_idx is not None and n_rounds is not None:
        if not _targetdiff_run_baseline_this_prudent_round(tbr, int(round_idx), int(n_rounds)):
            return pool
    prefix = f'[{log_prefix}] ' if log_prefix else ''
    if logger:
        logger.info(
            f'[Prudent][TargetDiff Refine] {prefix}打分前 baseline（{len(pool)} 条链）：'
            f'start_t={int(tbr.get("start_t", 100))}'
        )
    pos_list = [np.asarray(c['pos'], dtype=np.float64) for c in pool]
    v_list = [np.asarray(c['v'], dtype=np.int64) for c in pool]
    lock_counts = [
        int(c.get('n_locked', int(c.get('n_scaffold', 0)) + int(c.get('n_preserved', 0))))
        for c in pool
    ]
    refine_out = apply_targetdiff_baseline_refinement(
        model, data, pos_list, v_list, config, device=device, logger=logger
    )
    if len(refine_out) >= 2:
        refined_pos_list, refined_v_list = refine_out[0], refine_out[1]
    else:
        refined_pos_list, refined_v_list = refine_out
    restore_pos, restore_type = _baseline_refine_lock_restore_flags(config)
    refined_pos_list, refined_v_list = _restore_locked_prefix_after_refine(
        refined_pos_list, refined_v_list, pos_list, v_list, lock_counts,
        logger=logger, tag='[Prudent][TargetDiff Refine]',
        restore_pos=restore_pos, restore_type=restore_type,
    )
    for i, c in enumerate(pool):
        c['pos'] = np.asarray(refined_pos_list[i], dtype=np.float64).copy()
        c['v'] = np.asarray(refined_v_list[i], dtype=np.int64).copy()
        if c.get('log_v') is not None:
            c['log_v'] = (
                index_to_log_onehot(torch.tensor(c['v'], dtype=torch.long), model.num_classes)
                .numpy()
                .astype(np.float32)
            )
    return pool


def apply_targetdiff_baseline_refine_to_sampling_result(model, data, result, config, device='cuda:0', logger=None):
    """对采样 result 中的全部最终配体执行 TargetDiff baseline refine（阶段 2b）。

    在「下游阶段」全部完成后调用：单独 optimization、单独 scaffold（evolve/grow）、
    dynamic_then_optimization、dynamic_then_scaffold 等均在构建 result 之后调用本函数，
    确保 2b 晚于 2c/2d/2e。dynamic 作为前置阶段时，sample_dynamic_diffusion_ligand 须传
    skip_targetdiff_baseline_refine=True，避免在 dynamic 结束时就提前做 2b。"""
    refine_cfg = config.sample.get('targetdiff_baseline_refine', {})
    if not refine_cfg.get('enable', False):
        return
    pos_list = result.get('pred_ligand_pos')
    v_list = result.get('pred_ligand_v')
    if not pos_list or not v_list or len(pos_list) != len(v_list):
        if logger:
            logger.warning('[TargetDiff Refine] result 配体列表为空或长度不一致，跳过优化后 refine')
        return
    orig_pos = [np.asarray(p, dtype=np.float64).copy() for p in pos_list]
    orig_v = [np.asarray(v, dtype=np.int64).copy() for v in v_list]
    meta0 = result.get('meta') if isinstance(result.get('meta'), dict) else {}
    records = meta0.get('meta_records') or meta0.get('records') or []
    lock_counts = []
    type_lock_counts = []
    for i in range(len(pos_list)):
        n_lock = 0
        n_type_lock = 0
        if i < len(records) and isinstance(records[i], dict):
            rec = records[i]
            n_lock = int(rec.get(
                'n_locked',
                int(rec.get('n_scaffold', 0)) + int(rec.get('n_preserved', 0)),
            ))
            n_type_lock = (
                len(orig_v[i])
                if bool(rec.get('extra_atom_types_locked', False))
                else n_lock
            )
        lock_counts.append(n_lock)
        type_lock_counts.append(n_type_lock)
    refine_out = apply_targetdiff_baseline_refinement(
        model, data, pos_list, v_list, config, device=device, logger=logger
    )
    meta = result.get('meta')
    if not isinstance(meta, dict):
        meta = {}
        result['meta'] = meta
    restore_pos, restore_type = _baseline_refine_lock_restore_flags(config)
    if len(refine_out) == 5:
        pos_list, v_list, refine_pos_traj, refine_v_traj, refine_time_indices = refine_out
        pos_list, v_list = _restore_locked_prefix_after_refine(
            pos_list, v_list, orig_pos, orig_v, lock_counts, logger=logger,
            restore_pos=restore_pos, restore_type=restore_type,
            type_lock_counts=type_lock_counts,
        )
        result['pred_ligand_pos'] = pos_list
        result['pred_ligand_v'] = v_list
        pos_traj = result.get('pred_ligand_pos_traj') or []
        v_traj = result.get('pred_ligand_v_traj') or []
        n = len(pos_list)
        if (
            len(pos_traj) == n
            and len(v_traj) == n
            and len(refine_pos_traj) == n
            and len(refine_v_traj) == n
        ):
            result['pred_ligand_pos_traj'] = [
                list(pt) + list(rpt) for pt, rpt in zip(pos_traj, refine_pos_traj)
            ]
            result['pred_ligand_v_traj'] = [
                list(vt) + list(rvt) for vt, rvt in zip(v_traj, refine_v_traj)
            ]
        meta['baseline_refine_time_indices'] = refine_time_indices
        if logger:
            logger.info(f'[TargetDiff Refine] 已在优化阶段之后对 {n} 个分子应用 refine')
    else:
        pos_list, v_list = refine_out
        pos_list, v_list = _restore_locked_prefix_after_refine(
            pos_list, v_list, orig_pos, orig_v, lock_counts, logger=logger,
            restore_pos=restore_pos, restore_type=restore_type,
            type_lock_counts=type_lock_counts,
        )
        result['pred_ligand_pos'] = pos_list
        result['pred_ligand_v'] = v_list
