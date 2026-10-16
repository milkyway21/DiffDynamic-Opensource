"""Sampling scaffold prudent responsibilities for ligand sampling."""

from __future__ import annotations

from diffdynamic.data.graph import FOLLOW_BATCH
from diffdynamic.models.probability import center_pos
from diffdynamic.models.probability import ensure_log_ligand
from diffdynamic.models.probability import index_to_log_onehot
from diffdynamic.models.probability import log_sample_categorical
from pathlib import Path
from rdkit import Chem
from torch_geometric.data import Batch
from torch_scatter import scatter_mean
from diffdynamic.chemistry.scaffold_sites import build_extra_atom_positions
from diffdynamic.chemistry.scaffold_sites import cap_n_extra_for_sites
from diffdynamic.chemistry.scaffold_sites import get_murcko_sites_cfg
from diffdynamic.chemistry.scaffold_sites import load_or_extract_attachment_sites
from diffdynamic.chemistry.scaffold_sites import uses_site_budget_placement
import numpy as np
import os
import time
import torch
import torch.nn.functional as F


def _prudent_locked_sample_once(
    *,
    model,
    data,
    device,
    grow_cfg,
    _attachment_sites,
    _murcko_sites_cfg,
    protein_centroid,
    ref_ligand_pos,
    log_ref_v,
    scaffold_indices,
    n_scaffold,
    scaffold_bond_pairs,
    fix_scaffold_pos,
    fix_scaffold_type,
    log_mode_is_log_prob,
    large_cfg,
    refine_cfg,
    time_boundary,
    center_pos_mode,
    pos_only,
    ligand_atom_mode,
    selector_cfg,
    sample_idx,
    slot_retry,
    logger=None,
    log_tag='[ScaffoldPrudent]',
):
    """生成一条骨架锁定分子（Gen0 / final_fill 共用）。返回 cand dict 或 None。"""
    from diffdynamic.sampling.candidates import evaluate_candidate
    from diffdynamic.sampling.configuration import _dynamic_subcfg_grad_cap_kwargs
    from diffdynamic.sampling.optimization import _forward_diffuse_molecule
    from diffdynamic.sampling.scaffold.inputs import _pin_place_cfg_to_n_extra
    from diffdynamic.sampling.scaffold.inputs import _sample_n_extra_atoms
    from diffdynamic.sampling.scaffold.placement import _murcko_placement_cfg
    from diffdynamic.sampling.scaffold.placement import _murcko_sample_rng
    from diffdynamic.sampling.scaffold.placement import _resolve_init_preserved_atoms

    n_extra = _sample_n_extra_atoms(
        grow_cfg, data.protein_pos, n_scaffold, model.num_timesteps,
        sample_idx=sample_idx, retry=slot_retry,
    )
    _place_cfg = _pin_place_cfg_to_n_extra(
        _murcko_placement_cfg(grow_cfg, _murcko_sites_cfg), n_extra, grow_cfg,
    )
    if not uses_site_budget_placement(_place_cfg):
        n_extra = cap_n_extra_for_sites(n_extra, _attachment_sites, _murcko_sites_cfg)

    batch = Batch.from_data_list([data.clone()], follow_batch=FOLLOW_BATCH).to(device)
    batch_protein = batch.protein_element_batch
    extra_pos_world, _site_place_meta = build_extra_atom_positions(
        n_extra, _attachment_sites, _place_cfg,
        protein_centroid, device, fallback_noise_scale=1.0, logger=logger,
        rng=_murcko_sample_rng(grow_cfg, sample_idx, retry=slot_retry),
        protein_positions=batch.protein_pos,
    )
    n_extra = int(extra_pos_world.size(0))
    (
        preserved_pos, preserved_log_v, preserved_indices, zero_site_ids,
        n_preserved, n_locked, n_total,
    ) = _resolve_init_preserved_atoms(
        _attachment_sites, _site_place_meta, _place_cfg,
        ref_ligand_pos, log_ref_v, n_scaffold, n_extra,
        logger=logger, log_tag=log_tag, sample_idx=sample_idx,
    )

    init_ligand_pos = protein_centroid.unsqueeze(0).expand(n_total, -1) + \
                      torch.randn(n_total, 3, device=device)
    if n_scaffold > 0 and fix_scaffold_pos:
        init_ligand_pos[:n_scaffold] = ref_ligand_pos[scaffold_indices]
    if n_preserved > 0 and fix_scaffold_pos:
        init_ligand_pos[n_scaffold:n_locked] = preserved_pos
    if n_extra > 0:
        init_ligand_pos[n_locked:] = extra_pos_world
    batch_ligand = torch.zeros(n_total, dtype=torch.long, device=device)

    if fix_scaffold_type and n_locked > 0:
        scaffold_log_part = (
            log_ref_v[scaffold_indices] if n_scaffold > 0
            else torch.zeros(0, model.num_classes, device=device)
        )
        extra_log_v = F.log_softmax(
            torch.zeros(n_extra, model.num_classes, device=device), dim=-1
        ) if n_extra > 0 else torch.zeros(0, model.num_classes, device=device)
        init_ligand_v_input = torch.cat(
            [scaffold_log_part, preserved_log_v, extra_log_v], dim=0
        )
    else:
        uniform_logits = torch.zeros(n_total, model.num_classes, device=device)
        if log_mode_is_log_prob:
            init_ligand_v_input = F.log_softmax(uniform_logits, dim=-1)
        else:
            init_ligand_v_input = log_sample_categorical(uniform_logits)

    repaint_cfg = None
    if n_locked > 0:
        scaffold_pos_orig = (
            ref_ligand_pos[scaffold_indices] if n_scaffold > 0
            else torch.zeros(0, 3, device=device)
        )
        locked_pos_world = torch.cat([scaffold_pos_orig, preserved_pos], dim=0)
        locked_pos_centered = locked_pos_world - protein_centroid.unsqueeze(0)
        if n_extra > 0:
            extra_pos_centered = extra_pos_world - protein_centroid.unsqueeze(0)
        else:
            extra_pos_centered = torch.zeros(0, 3, device=device)
        x0_pos_ref = torch.cat([locked_pos_centered, extra_pos_centered], dim=0)
        scaffold_log_v = (
            log_ref_v[scaffold_indices] if n_scaffold > 0
            else torch.zeros(0, model.num_classes, device=device)
        )
        extra_log_v_pad = F.log_softmax(
            torch.zeros(n_extra, model.num_classes, device=device), dim=-1
        ) if n_extra > 0 else torch.zeros(0, model.num_classes, device=device)
        x0_log_v_ref = torch.cat([scaffold_log_v, preserved_log_v, extra_log_v_pad], dim=0)
        pos_mask_local = torch.zeros(n_total, device=device)
        if fix_scaffold_pos:
            pos_mask_local[:n_locked] = 1.0
        type_mask_local = None
        if fix_scaffold_type:
            type_mask_local = torch.zeros(n_total, device=device)
            type_mask_local[:n_locked] = 1.0
        repaint_cfg = {
            'x0_pos': x0_pos_ref,
            'x0_log_v': x0_log_v_ref,
            'pos_mask': pos_mask_local,
            'type_mask': type_mask_local,
            'use_mean_for_discrete': True,
            '_use_dual_mask': True,
        }

    if grow_cfg.get('forward_noise_init', False):
        _, centered_init, init_offset = center_pos(
            batch.protein_pos, init_ligand_pos, batch_protein, batch_ligand,
            mode=center_pos_mode,
        )
        noised_pos, noised_log_v = _forward_diffuse_molecule(
            model, centered_init,
            ensure_log_ligand(init_ligand_v_input, model.num_classes, mode='auto'),
            batch_ligand,
            min(int(grow_cfg.get('start_t', 999)), model.num_timesteps - 1),
            device,
        )
        init_ligand_pos = noised_pos + init_offset[batch_ligand]
        init_ligand_v_input = noised_log_v

    with torch.no_grad():
        res_large = model.sample_diffusion_large_step(
            protein_pos=batch.protein_pos,
            protein_v=batch.protein_atom_feature.float(),
            batch_protein=batch_protein,
            init_ligand_pos=init_ligand_pos,
            init_ligand_v=init_ligand_v_input,
            batch_ligand=batch_ligand,
            num_steps=large_cfg.get('num_steps'),
            center_pos_mode=center_pos_mode,
            pos_only=pos_only,
            step_stride=large_cfg.get('stride'),
            step_size=large_cfg.get('step_size'),
            add_noise=large_cfg.get('noise_scale'),
            pos_clip=large_cfg.get('pos_clip'),
            v_clip=large_cfg.get('v_clip'),
            log_ligand_input_mode='log_prob' if log_mode_is_log_prob else 'auto',
            repaint_cfg=repaint_cfg,
            **_dynamic_subcfg_grad_cap_kwargs(large_cfg),
        )
        refine_init_pos = res_large['pos']
        refine_init_v = res_large['log_v'] if log_mode_is_log_prob else res_large['v']
        res_refine = model.sample_diffusion_refinement(
            protein_pos=batch.protein_pos,
            protein_v=batch.protein_atom_feature.float(),
            batch_protein=batch_protein,
            init_ligand_pos=refine_init_pos,
            init_ligand_v=refine_init_v,
            batch_ligand=batch_ligand,
            center_pos_mode=center_pos_mode,
            pos_only=pos_only,
            step_stride=refine_cfg.get('stride'),
            step_size=refine_cfg.get('step_size'),
            add_noise=refine_cfg.get('noise_scale'),
            pos_clip=refine_cfg.get('pos_clip'),
            v_clip=refine_cfg.get('v_clip'),
            time_upper=refine_cfg.get('time_upper', time_boundary),
            time_lower=refine_cfg.get('time_lower', 0),
            num_cycles=refine_cfg.get('cycles', 1),
            log_ligand_input_mode='log_prob' if log_mode_is_log_prob else 'auto',
            repaint_cfg=repaint_cfg,
            **_dynamic_subcfg_grad_cap_kwargs(refine_cfg),
        )
        final_pos_t = res_refine['pos']
        log_v_out = res_refine['log_v']

    pos_np = final_pos_t.detach().cpu().numpy().astype(np.float64)
    v_np = log_v_out.argmax(dim=-1).detach().cpu().numpy().astype(np.int64)
    log_v_np = log_v_out.detach().cpu().numpy().astype(np.float32)
    if n_extra > 0:
        extra_v = v_np[n_locked:]
        to_carbon = (extra_v == 0)
        if to_carbon.any():
            extra_v[to_carbon] = 1
            v_np[n_locked:] = extra_v

    metric_info = evaluate_candidate(
        pos_np, v_np, ligand_atom_mode, selector_cfg,
        scaffold_bonds=scaffold_bond_pairs, n_scaffold=n_scaffold,
    )
    return {
        'pos': pos_np,
        'v': v_np,
        'log_v': log_v_np,
        'num_atoms': n_total,
        'n_scaffold': n_scaffold,
        'n_preserved': n_preserved,
        'n_extra': n_extra,
        'n_locked': n_locked,
        'preserved_atom_indices': list(preserved_indices),
        'scaffold_bond_pairs': scaffold_bond_pairs,
        'zero_allocation_site_ids': zero_site_ids,
        'pos_traj': [],
        'v_traj': [],
        'log_v_traj': [],
        'time_indices': None,
        'metric_info': metric_info,
        'site_place_meta': _site_place_meta,
        'initial_extra_positions': extra_pos_world.detach().cpu().tolist(),
        'sample_idx': sample_idx,
        'slot_retry': slot_retry,
    }


def scaffold_prudent_molecule(
    model,
    data,
    config,
    ligand_atom_mode: str,
    device: str = 'cuda:0',
    logger=None,
    output_dir: str = None,
) -> dict:
    """Scaffold-Prudent：骨架锁定 + 迭代重筛选（回原点再跑）。

    核心流程：
      Gen 0: scaffold_dynamic_locked 生成初始分子池（large_step + refine，锁骨架位置）
        → QED/SA/Vina 打分 → 选 top-K 幸存者
      Gen 1..N: forward-diffuse 幸存者到 renoise_t → refine(renoise_t→0, 带骨架 repaint)
        → 打分 → 选 top-K → 循环

    与 dynamic_locked 的区别：dynamic_locked 一次性生成所有分子后即结束；
    prudent 在每代末尾选取最佳分子并「回到原点」加噪后重新 refine，实现迭代优化。
    """
    from diffdynamic.sampling.candidates import evaluate_candidate
    from diffdynamic.sampling.configuration import _dynamic_subcfg_grad_cap_kwargs
    from diffdynamic.sampling.configuration import get_time_boundary
    from diffdynamic.sampling.optimization import _forward_diffuse_molecule
    from diffdynamic.sampling.prudent.reconstruction import _prudent_canonical_smiles
    from diffdynamic.sampling.prudent.reconstruction import _prudent_reconstruct_mol
    from diffdynamic.sampling.prudent.refinement import _batch_refinement_forward_split
    from diffdynamic.sampling.prudent.scoring import _prudent_chem_scores
    from diffdynamic.sampling.prudent.scoring import _prudent_composite_score
    from diffdynamic.sampling.prudent.scoring import _prudent_eligible_for_advance
    from diffdynamic.sampling.prudent.scoring import _prudent_parse_size_gates
    from diffdynamic.sampling.prudent.scoring import _prudent_size_gate_ok
    from diffdynamic.sampling.prudent.state import _prudent_candidate_traj_cleared
    from diffdynamic.sampling.prudent.state import _prudent_copy_candidate
    from diffdynamic.sampling.prudent.state import _prudent_forward_diffuse_to_t
    from diffdynamic.sampling.scaffold.inputs import _extract_scaffold_bond_pairs
    from diffdynamic.sampling.scaffold.inputs import _extract_scaffold_from_reference
    from diffdynamic.sampling.scaffold.inputs import _get_scaffold_mode_cfg
    from diffdynamic.sampling.scaffold.inputs import _pin_place_cfg_to_n_extra
    from diffdynamic.sampling.scaffold.inputs import _sample_n_extra_atoms
    from diffdynamic.sampling.scaffold.placement import _murcko_placement_cfg
    from diffdynamic.sampling.scaffold.placement import _murcko_sample_rng
    from diffdynamic.sampling.scaffold.placement import _resolve_init_preserved_atoms

    sc_cfg, _ = _get_scaffold_mode_cfg(config)
    prudent_cfg = sc_cfg.get('prudent', {})
    grow_cfg = dict(config.sample.get('scaffold', {}).get('grow', {}) or {})
    # 与 sample.seed 对齐（launcher/API 改的是 sample.seed）
    try:
        _sample_seed = int(config.sample.get('seed', grow_cfg.get('seed', 42)))
    except (TypeError, ValueError):
        _sample_seed = 42
    grow_cfg['seed'] = _sample_seed
    try:
        config.sample.setdefault('scaffold', {})
        config.sample['scaffold'].setdefault('grow', {})
        config.sample['scaffold']['grow']['seed'] = _sample_seed
    except Exception:
        pass

    n_generations = max(1, int(prudent_cfg.get('n_generations', 5)))
    n_chains_per_seed = max(1, int(prudent_cfg.get('n_chains_per_seed', 4)))
    advance_top_k = max(1, int(prudent_cfg.get('advance_top_k', 2)))
    _re_t = int(prudent_cfg.get('renoise_t', 600))
    renoise_t = int(np.clip(_re_t, 0, getattr(model, 'num_timesteps', 1000) - 1))

    dynamic_cfg = config.sample.get('dynamic', {})
    large_cfg = dynamic_cfg.get('large_step', {})
    refine_cfg = dict(dynamic_cfg.get('refine', {}))
    time_boundary = get_time_boundary(dynamic_cfg, 750)
    if 'time_upper' not in refine_cfg:
        refine_cfg['time_upper'] = time_boundary
    center_pos_mode = config.sample.get('center_pos_mode', 'protein')
    pos_only = config.sample.get('pos_only', False)
    log_mode_is_log_prob = (getattr(model, 'ligand_v_input', 'onehot') == 'log_prob')

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

    # ---- 参考配体 & 骨架提取（复用共享逻辑）---------------------------
    if not hasattr(data, 'ligand_pos') or data.ligand_pos is None or data.ligand_pos.size(0) == 0:
        raise ValueError(
            '骨架 Prudent 模式需要参考配体（含 3D 坐标 SDF）以提取骨架，请通过 --ligand_path 指定。'
        )

    ref_ligand_pos = data.ligand_pos.to(device)
    ref_ligand_v = data.ligand_atom_feature_full.to(device)
    n_scaffold_ref = ref_ligand_pos.size(0)
    log_ref_v = index_to_log_onehot(ref_ligand_v, model.num_classes)

    scaffold_indices, ref_mol, ref_ligand_name, n_scaffold = _extract_scaffold_from_reference(
        data, sc_cfg, ligand_atom_mode, ref_ligand_pos, ref_ligand_v, log_ref_v,
        n_scaffold_ref, device, config=config, logger=logger,
    )
    fix_scaffold_pos = bool(sc_cfg.get('fix_scaffold_pos', True))
    fix_scaffold_type = bool(sc_cfg.get('fix_scaffold_type', False))
    scaffold_bond_pairs = _extract_scaffold_bond_pairs(ref_mol, scaffold_indices, logger=logger)
    if logger:
        logger.info(
            f'[ScaffoldPrudent] 骨架原子: {n_scaffold}/{n_scaffold_ref} | '
            f'锁定位置: {fix_scaffold_pos} | 锁定类型: {fix_scaffold_type} | '
            f'骨架键: {len(scaffold_bond_pairs)} | '
            f'代际: {n_generations} | renoise_t: {renoise_t} | advance_top_k: {advance_top_k}'
        )

    # ---- 蛋白中心（用于 forward-diffuse 和 repaint_cfg 重建）---------------
    batch_center = Batch.from_data_list([data.clone()], follow_batch=FOLLOW_BATCH).to(device)
    protein_centroid = scatter_mean(batch_center.protein_pos, batch_center.protein_element_batch, dim=0)[0]

    ref_pos_np_prudent = ref_ligand_pos.detach().cpu().numpy().astype(np.float64)
    if output_dir is not None:
        _pr_base_output_dir = Path(output_dir)
    else:
        _pr_base_output_dir = Path(config.sample.get('output_dir', './outputs'))
    _murcko_sites_cfg = get_murcko_sites_cfg(sc_cfg)
    _attachment_sites = load_or_extract_attachment_sites(
        ref_mol, scaffold_indices,
        sc_cfg.get('scaffold_source', 'auto_murcko'),
        ref_pos_np_prudent, _pr_base_output_dir, ref_ligand_name,
        _murcko_sites_cfg, logger=logger,
    )

    # ---- 自适应门控：有参考配体时默认启用（可由 adaptive_gates.enable=false 关闭）----
    # 门控阈值由「任务原始参考配体」性质给出；actives CSV 仅作可选补充。
    _ag_cfg = sc_cfg.get('adaptive_gates') or prudent_cfg.get('adaptive_gates') or {}
    if isinstance(_ag_cfg, bool):
        _ag_cfg = {'enable': _ag_cfg}
    _ag_enable = bool(_ag_cfg.get('enable', True))  # 默认开
    if _ag_enable:
        try:
            from diffdynamic.sampling.gates import (
                apply_gates_to_prudent_cfg,
                compute_adaptive_gates,
                write_gates_json,
            )
            # 优先：任务传入的原始配体路径（data.ligand_filename）
            _lig_path = None
            for cand in (
                getattr(data, 'ligand_filename', None),
                _ag_cfg.get('ligand'),
                sc_cfg.get('ligand_path'),
                prudent_cfg.get('ligand_path'),
            ):
                if cand and str(cand) not in ('N/A', 'None', '') and Path(str(cand)).exists():
                    _lig_path = str(Path(str(cand)).expanduser().resolve())
                    break
            _act_path = _ag_cfg.get('actives') or _ag_cfg.get('actives_csv')
            if _act_path and not Path(str(_act_path)).exists():
                _act_path = None
            if _lig_path:
                _margins = _ag_cfg.get('margins') if isinstance(_ag_cfg.get('margins'), dict) else None
                _gates = compute_adaptive_gates(
                    _lig_path,
                    actives_csv=_act_path,
                    margins=_margins,
                    grow_cfg=grow_cfg,
                    base_prudent=prudent_cfg,
                )
                prudent_cfg = apply_gates_to_prudent_cfg(prudent_cfg, _gates)
                try:
                    write_gates_json(_gates, _pr_base_output_dir / 'adaptive_gates.json')
                except Exception:
                    pass
                if logger:
                    logger.info(
                        f'[ScaffoldPrudent] adaptive_gates from ligand={_lig_path}: '
                        f'{_gates.as_prudent_dict()}'
                    )
                    for _r in _gates.rationale:
                        logger.info(f'[ScaffoldPrudent] adaptive_gates: {_r}')
            elif logger:
                logger.info(
                    '[ScaffoldPrudent] adaptive_gates 跳过：无可用参考配体路径'
                )
        except Exception as _ag_err:
            if logger:
                logger.warning(f'[ScaffoldPrudent] adaptive_gates 失败，沿用原门控: {_ag_err}')

    # ---- Gen 0: scaffold_dynamic_locked 生成初始分子池 -----------------
    num_samples = max(1, int(sc_cfg.get('num_samples', grow_cfg.get('num_samples', 200))))
    if logger:
        logger.info(f'[ScaffoldPrudent] Gen 0: 调用 scaffold_dynamic_locked 生成 {num_samples} 个初始分子')

    selector_cfg = {
        'min_qed': grow_cfg.get('min_qed', 0.2),
        'min_sa': grow_cfg.get('min_sa', 0.2),
        'filter_incomplete': grow_cfg.get('filter_incomplete', False),
        'qed_weight': 1.0, 'sa_weight': 1.0,
    }

    # ---- Gen 0 逐样本循环（直接内联，以捕获 log_v 供后续 forward-diffuse）----
    pool = []
    meta_records_all = []
    total_time = 0.0
    # 累积所有分子的坐标
    all_pos_accum: list = []
    all_v_accum: list = []
    all_meta_accum: list = []

    # ---- Gate params for target-driven Gen0 accumulation ----
    _size_gates = _prudent_parse_size_gates(prudent_cfg)
    _hard_gate = bool(_size_gates)
    # 允许 gate 失败 / SMILES 去重后同槽位补采（含无 hard_gate 时的唯一 SMILES 重试）
    _max_gen0_attempts = num_samples * 200
    _gen0_attempt = 0
    _gen0_seen_smiles = set()
    _gen0_dedup_reject = 0
    _slot_retry = 0

    while True:
        # ---- Exit conditions ----
        if _gen0_attempt >= _max_gen0_attempts:
            if logger:
                logger.warning(
                    f'[ScaffoldPrudent] Gen0 达到最大尝试次数 {_max_gen0_attempts}, '
                    f'仅积累 {len(pool)}/{num_samples} gate-passing 分子'
                )
            break
        if len(pool) >= num_samples:
            break

        _gen0_attempt += 1
        # 槽位 = 当前已积累数；同槽位门控失败补足用 slot_retry 偏移 RNG
        sample_idx = len(pool)
        n_extra = _sample_n_extra_atoms(
            grow_cfg, data.protein_pos, n_scaffold, model.num_timesteps,
            sample_idx=sample_idx, retry=_slot_retry,
        )
        _place_cfg = _pin_place_cfg_to_n_extra(
            _murcko_placement_cfg(grow_cfg, _murcko_sites_cfg), n_extra, grow_cfg,
        )
        if not uses_site_budget_placement(_place_cfg):
            n_extra = cap_n_extra_for_sites(n_extra, _attachment_sites, _murcko_sites_cfg)
        n_total = n_scaffold + n_extra

        if logger and (_gen0_attempt == 1 or _gen0_attempt % 10 == 0 or len(pool) <= 5 or _slot_retry > 0):
            if _hard_gate:
                logger.info(
                    f'[ScaffoldPrudent] Gen0 尝试#{_gen0_attempt} | '
                    f'sample_idx={sample_idx} slot_retry={_slot_retry} | '
                    f'已积累 {len(pool)}/{num_samples} gate-passing | '
                    f'骨架: {n_scaffold} + 新原子: {n_extra} = 总计: {n_total}'
                )
            else:
                logger.info(
                    f'[ScaffoldPrudent] Gen0 样本 {sample_idx + 1}/{num_samples} | '
                    f'骨架: {n_scaffold} + 新原子: {n_extra} = 总计: {n_total}'
                )

        batch = Batch.from_data_list([data.clone()], follow_batch=FOLLOW_BATCH).to(device)
        batch_protein = batch.protein_element_batch

        # 初始化：骨架 + 零分配保留侧链 + Murcko 位点新原子
        extra_pos_world, _site_place_meta = build_extra_atom_positions(
            n_extra, _attachment_sites, _place_cfg,
            protein_centroid, device, fallback_noise_scale=1.0, logger=logger,
            rng=_murcko_sample_rng(grow_cfg, sample_idx, retry=_slot_retry),
            protein_positions=batch.protein_pos,
        )
        n_extra = int(extra_pos_world.size(0))
        (
            preserved_pos, preserved_log_v, preserved_indices, zero_site_ids,
            n_preserved, n_locked, n_total,
        ) = _resolve_init_preserved_atoms(
            _attachment_sites, _site_place_meta, _place_cfg,
            ref_ligand_pos, log_ref_v, n_scaffold, n_extra,
            logger=logger, log_tag='[ScaffoldPrudent]', sample_idx=sample_idx,
        )

        init_ligand_pos = protein_centroid.unsqueeze(0).expand(n_total, -1) + \
                          torch.randn(n_total, 3, device=device)
        if n_scaffold > 0 and fix_scaffold_pos:
            init_ligand_pos[:n_scaffold] = ref_ligand_pos[scaffold_indices]
        if n_preserved > 0 and fix_scaffold_pos:
            init_ligand_pos[n_scaffold:n_locked] = preserved_pos
        if n_extra > 0:
            init_ligand_pos[n_locked:] = extra_pos_world
        batch_ligand = torch.zeros(n_total, dtype=torch.long, device=device)

        if fix_scaffold_type and n_locked > 0:
            scaffold_log_part = (
                log_ref_v[scaffold_indices] if n_scaffold > 0
                else torch.zeros(0, model.num_classes, device=device)
            )
            extra_log_v = F.log_softmax(
                torch.zeros(n_extra, model.num_classes, device=device), dim=-1
            ) if n_extra > 0 else torch.zeros(0, model.num_classes, device=device)
            init_ligand_v_input = torch.cat(
                [scaffold_log_part, preserved_log_v, extra_log_v], dim=0
            )
        else:
            uniform_logits = torch.zeros(n_total, model.num_classes, device=device)
            if log_mode_is_log_prob:
                init_ligand_v_input = F.log_softmax(uniform_logits, dim=-1)
            else:
                init_ligand_v_input = log_sample_categorical(uniform_logits)

        # 构建 repaint_cfg（锁骨架 + 保留侧链位置）
        repaint_cfg = None
        if n_locked > 0:
            scaffold_pos_orig = (
                ref_ligand_pos[scaffold_indices] if n_scaffold > 0
                else torch.zeros(0, 3, device=device)
            )
            locked_pos_world = torch.cat([scaffold_pos_orig, preserved_pos], dim=0)
            locked_pos_centered = locked_pos_world - protein_centroid.unsqueeze(0)
            if n_extra > 0:
                extra_pos_centered = extra_pos_world - protein_centroid.unsqueeze(0)
            else:
                extra_pos_centered = torch.zeros(0, 3, device=device)
            x0_pos_ref = torch.cat([locked_pos_centered, extra_pos_centered], dim=0)

            scaffold_log_v = (
                log_ref_v[scaffold_indices] if n_scaffold > 0
                else torch.zeros(0, model.num_classes, device=device)
            )
            extra_log_v_pad = F.log_softmax(
                torch.zeros(n_extra, model.num_classes, device=device), dim=-1
            ) if n_extra > 0 else torch.zeros(0, model.num_classes, device=device)
            x0_log_v_ref = torch.cat(
                [scaffold_log_v, preserved_log_v, extra_log_v_pad], dim=0
            )

            extra_anchor_strength = float(grow_cfg.get('extra_anchor_strength', 0.0))
            if extra_anchor_strength > 0.0:
                pos_mask_local = torch.full((n_total,), extra_anchor_strength, device=device)
            else:
                pos_mask_local = torch.zeros(n_total, device=device)
            if fix_scaffold_pos:
                pos_mask_local[:n_locked] = 1.0

            type_mask_local = None
            if fix_scaffold_type:
                type_mask_local = torch.zeros(n_total, device=device)
                type_mask_local[:n_locked] = 1.0

            repaint_cfg = {
                'x0_pos': x0_pos_ref,
                'x0_log_v': x0_log_v_ref,
                'pos_mask': pos_mask_local,
                'type_mask': type_mask_local,
                'use_mean_for_discrete': True,
                '_use_dual_mask': True,
            }

        if grow_cfg.get('forward_noise_init', False):
            _, centered_init, init_offset = center_pos(
                batch.protein_pos, init_ligand_pos, batch_protein, batch_ligand,
                mode=center_pos_mode,
            )
            noised_pos, noised_log_v = _forward_diffuse_molecule(
                model, centered_init,
                ensure_log_ligand(init_ligand_v_input, model.num_classes, mode='auto'),
                batch_ligand, min(int(grow_cfg.get('start_t', 999)), model.num_timesteps - 1),
                device,
            )
            init_ligand_pos = noised_pos + init_offset[batch_ligand]
            init_ligand_v_input = noised_log_v
        t_start_wall = time.time()
        with torch.no_grad():
            res_large = model.sample_diffusion_large_step(
                protein_pos=batch.protein_pos,
                protein_v=batch.protein_atom_feature.float(),
                batch_protein=batch_protein,
                init_ligand_pos=init_ligand_pos,
                init_ligand_v=init_ligand_v_input,
                batch_ligand=batch_ligand,
                num_steps=large_cfg.get('num_steps'),
                center_pos_mode=center_pos_mode,
                pos_only=pos_only,
                step_stride=large_cfg.get('stride'),
                step_size=large_cfg.get('step_size'),
                add_noise=large_cfg.get('noise_scale'),
                pos_clip=large_cfg.get('pos_clip'),
                v_clip=large_cfg.get('v_clip'),
                log_ligand_input_mode='log_prob' if log_mode_is_log_prob else 'auto',
                repaint_cfg=repaint_cfg,
                **_dynamic_subcfg_grad_cap_kwargs(large_cfg),
            )
            # refine（始终跑，scaffold dynamic_locked 契约）
            refine_init_pos = res_large['pos']
            if log_mode_is_log_prob:
                refine_init_v = res_large['log_v']
            else:
                refine_init_v = res_large['v']
            res_refine = model.sample_diffusion_refinement(
                protein_pos=batch.protein_pos,
                protein_v=batch.protein_atom_feature.float(),
                batch_protein=batch_protein,
                init_ligand_pos=refine_init_pos,
                init_ligand_v=refine_init_v,
                batch_ligand=batch_ligand,
                center_pos_mode=center_pos_mode,
                pos_only=pos_only,
                step_stride=refine_cfg.get('stride'),
                step_size=refine_cfg.get('step_size'),
                add_noise=refine_cfg.get('noise_scale'),
                pos_clip=refine_cfg.get('pos_clip'),
                v_clip=refine_cfg.get('v_clip'),
                time_upper=refine_cfg.get('time_upper', time_boundary),
                time_lower=refine_cfg.get('time_lower', 0),
                num_cycles=refine_cfg.get('cycles', 1),
                log_ligand_input_mode='log_prob' if log_mode_is_log_prob else 'auto',
                repaint_cfg=repaint_cfg,
                **_dynamic_subcfg_grad_cap_kwargs(refine_cfg),
            )
            final_pos_t = res_refine['pos']
            log_v_out = res_refine['log_v']
        t_end_wall = time.time()
        total_time += t_end_wall - t_start_wall

        pos_np = final_pos_t.detach().cpu().numpy().astype(np.float64)
        v_np = log_v_out.argmax(dim=-1).detach().cpu().numpy().astype(np.int64)
        log_v_np = log_v_out.detach().cpu().numpy().astype(np.float32)

        # 后处理：修复额外原子中价态不兼容元素 → C
        if n_extra > 0:
            extra_v = v_np[n_locked:]
            to_carbon = (extra_v == 0)
            if to_carbon.any():
                extra_v[to_carbon] = 1
                v_np[n_locked:] = extra_v

        # 骨架 RMSD
        if n_scaffold > 0:
            scaffold_pos_ref_np = ref_ligand_pos[scaffold_indices].detach().cpu().numpy()
            scaffold_pos_gen = pos_np[:n_scaffold]
            scaffold_rmsd = float(np.sqrt(np.mean((scaffold_pos_gen - scaffold_pos_ref_np) ** 2)))
        else:
            scaffold_rmsd = 0.0

        metric_info = evaluate_candidate(
            pos_np, v_np, ligand_atom_mode, selector_cfg,
            scaffold_bonds=scaffold_bond_pairs, n_scaffold=n_scaffold,
        )

        # ---- Gate check: 仅当 _hard_gate 时在 Gen0 就过滤，保证最终输出全是 gate-passing 分子 ----
        gen0_mol = None
        if _hard_gate:
            gen0_mol = _prudent_reconstruct_mol(
                pos_np, v_np, ligand_atom_mode,
                logger=logger, context=f'SPg0a{_gen0_attempt}',
                scaffold_bonds=scaffold_bond_pairs, n_scaffold=n_scaffold,
            )
            qed, sa, logp, chem_err = _prudent_chem_scores(gen0_mol, logger=logger)
            gate_ok, reasons = _prudent_size_gate_ok(gen0_mol, qed, sa, logp, prudent_cfg, gates=_size_gates)

            if not gate_ok:
                if logger and (_gen0_attempt <= 5 or _gen0_attempt % 50 == 0):
                    if not reasons:
                        reasons = ['unknown_gate_fail']
                    logger.info(
                        f'[ScaffoldPrudent] Gen0 gate-reject #{_gen0_attempt} '
                        f'(sample_idx={sample_idx} slot_retry={_slot_retry}): '
                        + ', '.join(reasons)
                        + f' | pool={len(pool)}/{num_samples}'
                    )
                # 定期警告：如果长时间没有 gate-passing 分子
                if _hard_gate and _gen0_attempt % 1000 == 0 and len(pool) == 0:
                    if logger:
                        logger.warning(
                            f'[ScaffoldPrudent] Gen0 已尝试 {_gen0_attempt} 次，尚无 gate-passing 分子。'
                            f'请考虑放宽尺寸门控 {_size_gates} 或降低 min_qed/min_sa 阈值。'
                        )
                _slot_retry += 1
                continue  # 丢弃此分子，同槽位补足（RNG 已偏移）

        # Canonical SMILES 去重：池中已有相同结构则同槽位重采
        gen0_smi = _prudent_canonical_smiles(
            gen0_mol if gen0_mol is not None else None,
            pos=pos_np, v=v_np, ligand_atom_mode=ligand_atom_mode,
            scaffold_bonds=scaffold_bond_pairs, n_scaffold=n_scaffold,
            logger=None, context=f'SPg0dedup{_gen0_attempt}',
        )
        if gen0_smi is None:
            gen0_smi = metric_info.get('smiles')
            if gen0_smi:
                try:
                    m_tmp = Chem.MolFromSmiles(str(gen0_smi))
                    gen0_smi = Chem.MolToSmiles(m_tmp) if m_tmp is not None else None
                except Exception:
                    gen0_smi = None
        if gen0_smi and gen0_smi in _gen0_seen_smiles:
            _gen0_dedup_reject += 1
            if logger and (_gen0_dedup_reject <= 5 or _gen0_dedup_reject % 50 == 0):
                logger.info(
                    f'[ScaffoldPrudent] Gen0 SMILES-dedup-reject #{_gen0_dedup_reject} '
                    f'(sample_idx={sample_idx} slot_retry={_slot_retry}) smi={gen0_smi[:80]} '
                    f'| pool={len(pool)}/{num_samples}'
                )
            _slot_retry += 1
            continue
        if gen0_smi:
            _gen0_seen_smiles.add(gen0_smi)

        pool.append({
            'pos': pos_np,
            'v': v_np,
            'log_v': log_v_np,
            'num_atoms': n_total,
            'n_scaffold': n_scaffold,
            'n_preserved': n_preserved,
            'n_extra': n_extra,
            'n_locked': n_locked,
            'preserved_atom_indices': list(preserved_indices),
            'scaffold_bond_pairs': scaffold_bond_pairs,
            'smiles': gen0_smi,
            'pos_traj': [],
            'v_traj': [],
            'log_v_traj': [],
            'time_indices': None,
        })
        meta_records_all.append({
            'method': 'scaffold_prudent',
            'generation': 0,
            'sample_idx': sample_idx,
            'slot_retry': _slot_retry,
            'ligand_num_atoms': n_total,
            'n_scaffold': n_scaffold,
            'n_preserved': n_preserved,
            'n_extra': n_extra,
            'n_locked': n_locked,
            'zero_allocation_site_ids': zero_site_ids,
            'preserved_atom_indices': preserved_indices,
            'n_scaffold_bonds': len(scaffold_bond_pairs),
            'scaffold_rmsd': scaffold_rmsd,
            'time': t_end_wall - t_start_wall,
            'smiles': gen0_smi or metric_info.get('smiles'),
            'qed': metric_info.get('metrics', {}).get('qed'),
            'sa': metric_info.get('metrics', {}).get('sa'),
            'status': metric_info.get('status'),
            'is_original': False,
            'initial_extra_positions': extra_pos_world.detach().cpu().tolist(),
            'forward_noise_init': bool(grow_cfg.get('forward_noise_init', False)),
            'extra_placement': _site_place_meta.get('placement'),
            'site_allocation': _site_place_meta.get('site_allocation'),
        })
        _slot_retry = 0

    if logger:
        valid_gen0 = sum(1 for m in meta_records_all if m.get('status') == 'ok')
        if _hard_gate:
            logger.info(
                f'[ScaffoldPrudent] Gen 0 完成: {len(pool)}/{num_samples} gate-passing 分子 | '
                f'共 {_gen0_attempt} 次生成尝试 | 有效分子 {valid_gen0} | '
                f'SMILES-dedup-reject={_gen0_dedup_reject} n_unique_smiles={len(_gen0_seen_smiles)} | '
                f'耗时 {total_time:.2f}s | '
                f'通过率 {len(pool)/max(1,_gen0_attempt)*100:.1f}%'
            )
        else:
            logger.info(
                f'[ScaffoldPrudent] Gen 0 完成: {valid_gen0}/{num_samples} 有效分子 | '
                f'SMILES-dedup-reject={_gen0_dedup_reject} n_unique_smiles={len(_gen0_seen_smiles)} | '
                f'耗时 {total_time:.2f}s'
            )

    # ---- 多代迭代 ------------------------------------------------------------
    protein_path = getattr(data, 'protein_filename', None)
    lig_fn = getattr(data, 'ligand_filename', None)
    min_q_dock = float(prudent_cfg.get('min_qed_for_docking', 0.15))
    min_s_dock = float(prudent_cfg.get('min_sa_for_docking', 0.15))
    max_logp = prudent_cfg.get('max_logp', None)
    if max_logp is not None:
        max_logp = float(max_logp)

    generation_stats = []  # 每代统计

    for gen_idx in range(n_generations):
        if logger:
            logger.info(
                f'[ScaffoldPrudent] === Gen {gen_idx}/{n_generations} | pool={len(pool)} chains ==='
            )

        # ---- 打分：QED/SA/LogP + 条件 Vina → composite ----
        scored = []
        for ci, c in enumerate(pool):
            mol = _prudent_reconstruct_mol(
                c['pos'], c['v'], ligand_atom_mode,
                logger=logger, context=f'SPg{gen_idx}c{ci}',
                scaffold_bonds=c.get('scaffold_bond_pairs', scaffold_bond_pairs),
                n_scaffold=int(c.get('n_scaffold', n_scaffold)),
            )
            qed, sa, logp, chem_err = _prudent_chem_scores(mol, logger=logger)
            if mol is not None and not c.get('smiles'):
                try:
                    c['smiles'] = Chem.MolToSmiles(mol)
                except Exception:
                    pass

            gate_ok = (
                mol is not None
                and np.isfinite(qed) and np.isfinite(sa)
                and qed >= min_q_dock and sa >= min_s_dock
                and (max_logp is None or not np.isfinite(logp) or logp <= max_logp)
            )

            if not gate_ok:
                detail = {
                    'qed': qed, 'sa': sa, 'logp': logp,
                    'vina_affinity': float('nan'), 'vina_norm': 0.0,
                    'vina_skipped': True, 'reconstruct_error': chem_err,
                }
                comp = float('-inf')
            else:
                comp, detail = _prudent_composite_score(
                    mol, prudent_cfg, protein_path, lig_fn,
                    logger=logger, protein_root=vina_protein_root,
                    prefetched_chem=(qed, sa, logp, chem_err), skip_vina=False,
                )

            if logger and gen_idx == 0 and ci < 5:
                aff = detail.get('vina_affinity')
                aff_s = f'{float(aff):.3f}' if aff is not None and np.isfinite(float(aff)) else 'nan'
                logger.info(
                    f'[ScaffoldPrudent][score] g{gen_idx} c{ci} | '
                    f'QED={qed:.4f} SA={sa:.4f} LogP={logp:.2f} Vina={aff_s} composite={comp:.4f}'
                )

            c['prudent_score_detail'] = detail
            scored.append((comp, detail, c, ci))

        scored.sort(key=lambda x: x[0], reverse=True)

        # 记录本代统计
        comps = [s[0] for s in scored if np.isfinite(s[0])]
        gen_stat = {
            'generation': gen_idx,
            'pool_size': len(pool),
            'valid_scored': len(comps),
            'best_composite': float(max(comps)) if comps else float('nan'),
            'mean_composite': float(np.mean(comps)) if comps else float('nan'),
        }
        generation_stats.append(gen_stat)
        if logger:
            logger.info(
                f'[ScaffoldPrudent] Gen {gen_idx} stats: '
                f'best_comp={gen_stat["best_composite"]:.4f} '
                f'mean_comp={gen_stat["mean_composite"]:.4f} '
                f'valid={gen_stat["valid_scored"]}/{gen_stat["pool_size"]}'
            )

        # 最后一轮不打分不选（输出最终 pool）
        if gen_idx == n_generations - 1:
            break

        # ---- 选择 survivors ----
        filtered = []
        for comp, detail, c, ci in scored:
            mol = _prudent_reconstruct_mol(
                c['pos'], c['v'], ligand_atom_mode,
                logger=None, context=f'SPg{gen_idx}sel{ci}',
                scaffold_bonds=c.get('scaffold_bond_pairs', scaffold_bond_pairs),
                n_scaffold=int(c.get('n_scaffold', n_scaffold)),
            )
            qed_v = detail.get('qed', 0.0)
            sa_v = detail.get('sa', 0.0)
            if mol is not None and _prudent_eligible_for_advance(
                mol, qed_v, sa_v, detail, prudent_cfg, vina_advance_thr,
            ):
                filtered.append((comp, detail, c, ci))

        _atk_mode = str(prudent_cfg.get('advance_top_k_mode', 'cap')).strip().lower()
        if _atk_mode == 'cap' and prudent_cfg.get('advance_top_k_strict') is True:
            _atk_mode = 'strict'
        continue_when_no_advance = bool(prudent_cfg.get('continue_when_no_advance', True))
        fallback_keep_top_k = max(1, int(prudent_cfg.get('fallback_keep_top_k', 1)))

        if _atk_mode in ('strict', 'require_full_k', 'full_k'):
            if len(filtered) < advance_top_k:
                if continue_when_no_advance and scored:
                    survivors = [s[2] for s in scored[:min(fallback_keep_top_k, len(scored))]]
                    if logger:
                        logger.warning(
                            f'[ScaffoldPrudent] Gen {gen_idx}: strict 模式下仅 {len(filtered)} 条合格，'
                            f'回退保留 top-{len(survivors)}'
                        )
                else:
                    survivors = []
            else:
                survivors = [f[2] for f in filtered[:advance_top_k]]
        else:
            if not filtered:
                if continue_when_no_advance and scored:
                    survivors = [s[2] for s in scored[:min(fallback_keep_top_k, len(scored))]]
                else:
                    survivors = []
            else:
                survivors = [f[2] for f in filtered[:advance_top_k]]

        if not survivors:
            if logger:
                logger.warning(f'[ScaffoldPrudent] Gen {gen_idx}: 无幸存者，提前结束迭代')
            break

        if logger:
            logger.info(
                f'[ScaffoldPrudent] Gen {gen_idx}: '
                f'{len(survivors)} survivors → 扩链 ×{n_chains_per_seed} → {len(survivors) * n_chains_per_seed} chains'
            )

        # ---- 将当前 pool 中 gate-passing 分子累积到 all_* 列表 ----
        for pc in pool:
            # 当 _hard_gate 时，跳过门控失败分子（vina_skipped=True）
            if _hard_gate:
                detail = pc.get('prudent_score_detail', {})
                if detail.get('vina_skipped', False):
                    continue
            all_pos_accum.append(np.asarray(pc['pos'], dtype=np.float64))
            all_v_accum.append(np.asarray(pc['v'], dtype=np.int64))
            # 记录每个分子来自哪一代
            pc_meta = dict(pc.get('prudent_score_detail', {}))
            pc_meta['method'] = 'scaffold_prudent'
            pc_meta['generation'] = gen_idx
            pc_meta['num_atoms'] = int(pc.get('num_atoms', 0))
            pc_meta['n_scaffold'] = int(pc.get('n_scaffold', n_scaffold))
            pc_meta['n_extra'] = int(pc.get('n_extra', 0))
            all_meta_accum.append(pc_meta)

        # ---- 扩链 + forward-diffuse + refine ----
        new_pool = []
        for surv in survivors:
            for _ in range(n_chains_per_seed):
                copied = _prudent_candidate_traj_cleared(_prudent_copy_candidate(surv))
                # Forward-diffuse 到 renoise_t
                fwd = _prudent_forward_diffuse_to_t(
                    model, data, copied, int(renoise_t),
                    center_pos_mode, pos_only, device, logger=None,
                )
                fwd['pos_traj'] = []
                fwd['v_traj'] = []
                fwd['log_v_traj'] = []
                fwd['time_indices'] = None
                fwd.pop('prudent_score_detail', None)
                fwd.pop('prudent_round_analysis', None)

                n_total_c = int(fwd['num_atoms'])
                n_scaffold_c = int(surv.get('n_scaffold', n_scaffold))
                n_preserved_c = int(surv.get('n_preserved', 0))
                n_locked_c = int(surv.get('n_locked', n_scaffold_c + n_preserved_c))
                n_extra_c = max(0, n_total_c - n_locked_c)

                # 重建 repaint_cfg：锁骨架 + Gen0 保留侧链
                # x0 必须用 survivor 干净态，不可用 fwd 加噪坐标（否则多代骨架漂）
                repaint_cfg_c = None
                if n_locked_c > 0:
                    pos_clean = torch.as_tensor(
                        np.asarray(surv['pos'], dtype=np.float64),
                        dtype=torch.float32, device=device,
                    )
                    if pos_clean.shape[0] != n_total_c:
                        # 原子数异常时回退到 fwd（极少见）
                        pos_clean = torch.as_tensor(
                            np.asarray(fwd['pos'], dtype=np.float64),
                            dtype=torch.float32, device=device,
                        )
                    x0_pos_ref_c = pos_clean - protein_centroid.unsqueeze(0)

                    if isinstance(surv.get('log_v'), np.ndarray) and np.asarray(surv['log_v']).ndim == 2 \
                            and np.asarray(surv['log_v']).shape[0] == n_total_c:
                        x0_log_v_ref_c = torch.as_tensor(
                            surv['log_v'], dtype=torch.float32, device=device,
                        )
                    else:
                        v_clean = torch.as_tensor(
                            np.asarray(surv['v'], dtype=np.int64),
                            dtype=torch.long, device=device,
                        )
                        if v_clean.shape[0] != n_total_c:
                            v_clean = torch.as_tensor(
                                np.asarray(fwd['v'], dtype=np.int64),
                                dtype=torch.long, device=device,
                            )
                        x0_log_v_ref_c = index_to_log_onehot(v_clean, model.num_classes)

                    extra_anchor_strength_c = float(grow_cfg.get('extra_anchor_strength', 0.0))
                    if extra_anchor_strength_c > 0.0:
                        pos_mask_c = torch.full((n_total_c,), extra_anchor_strength_c, device=device)
                    else:
                        pos_mask_c = torch.zeros(n_total_c, device=device)
                    if fix_scaffold_pos:
                        pos_mask_c[:n_locked_c] = 1.0
                    type_mask_c = None
                    if fix_scaffold_type:
                        type_mask_c = torch.zeros(n_total_c, device=device)
                        type_mask_c[:n_locked_c] = 1.0

                    repaint_cfg_c = {
                        'x0_pos': x0_pos_ref_c,
                        'x0_log_v': x0_log_v_ref_c,
                        'pos_mask': pos_mask_c,
                        'type_mask': type_mask_c,
                        'use_mean_for_discrete': True,
                        '_use_dual_mask': True,
                    }

                # 单链 refine with repaint_cfg
                refined_list = _batch_refinement_forward_split(
                    model, data, [fwd],
                    center_pos_mode=center_pos_mode,
                    pos_only=pos_only,
                    device=device,
                    time_upper=int(renoise_t),
                    time_lower=int(refine_cfg.get('time_lower', 0)),
                    refine_cfg=refine_cfg,
                    refinement_grad_kwargs=_dynamic_subcfg_grad_cap_kwargs(refine_cfg),
                    error_prefix='[ScaffoldPrudent] refine',
                    repaint_cfg=repaint_cfg_c,
                )
                refined = refined_list[0]
                refined['n_scaffold'] = n_scaffold_c
                refined['n_preserved'] = n_preserved_c
                refined['n_locked'] = n_locked_c
                refined['n_extra'] = n_extra_c
                refined['scaffold_bond_pairs'] = surv.get(
                    'scaffold_bond_pairs', scaffold_bond_pairs,
                )
                if surv.get('preserved_atom_indices') is not None:
                    refined['preserved_atom_indices'] = surv.get('preserved_atom_indices')
                refined.pop('grad_fusion_steps', None)
                new_pool.append(refined)

        pool = new_pool

    # ---- 最后一轮凑满 target_n（不足则 final_fill 多轮补足）----------------
    _tfc = prudent_cfg.get('target_final_count', None)
    if _tfc is None:
        target_n = int(num_samples)
    else:
        try:
            target_n = max(1, int(_tfc))
        except (TypeError, ValueError):
            target_n = int(num_samples)
    _mffa = prudent_cfg.get('max_final_fill_attempts', None)
    if _mffa is None:
        max_final_fill_attempts = int(target_n) * 200
    else:
        try:
            max_final_fill_attempts = max(1, int(_mffa))
        except (TypeError, ValueError):
            max_final_fill_attempts = int(target_n) * 200

    def _pool_gate_ok(c):
        if not _hard_gate:
            return True
        detail = c.get('prudent_score_detail', {})
        return not detail.get('vina_skipped', False)

    def _comp_key(c):
        d = c.get('prudent_score_detail') or {}
        q = d.get('qed', 0.0) or 0.0
        s = d.get('sa', 0.0) or 0.0
        v = d.get('vina_norm', 0.0) or 0.0
        return float(q) + float(s) + float(v)

    gated = [c for c in pool if _pool_gate_ok(c)]
    # 按 canonical SMILES 去重：同结构保留 composite 更高者
    _final_dedup_reject = 0
    _best_by_smi = {}
    _no_smi = []
    for c in gated:
        smi = _prudent_canonical_smiles(
            c, ligand_atom_mode=ligand_atom_mode,
            scaffold_bonds=c.get('scaffold_bond_pairs', scaffold_bond_pairs),
            n_scaffold=int(c.get('n_scaffold', n_scaffold)),
            logger=None, context='final_pool_dedup',
        )
        if smi:
            c['smiles'] = smi
            prev = _best_by_smi.get(smi)
            if prev is None:
                _best_by_smi[smi] = c
            else:
                _final_dedup_reject += 1
                if _comp_key(c) > _comp_key(prev):
                    _best_by_smi[smi] = c
        else:
            _no_smi.append(c)
    final_pool = list(_best_by_smi.values()) + _no_smi
    if logger and _final_dedup_reject:
        logger.info(
            f'[ScaffoldPrudent] final_pool SMILES 去重: gated={len(gated)} → '
            f'unique={len(_best_by_smi)} (+{len(_no_smi)} no-smiles) '
            f'dedup_reject={_final_dedup_reject}'
        )
    if len(final_pool) > target_n:
        final_pool = sorted(final_pool, key=_comp_key, reverse=True)[:target_n]
        if logger:
            logger.info(
                f'[ScaffoldPrudent] final pool 去重后仍 > target_n={target_n}，按分数截断为 {len(final_pool)}'
            )

    fill_attempt = 0
    fill_slot_retry = 0
    _fill_dedup_reject = 0
    _final_seen_smiles = {
        c.get('smiles') for c in final_pool
        if c.get('smiles')
    }
    while len(final_pool) < target_n and fill_attempt < max_final_fill_attempts:
        need = target_n - len(final_pool)
        sample_idx = len(final_pool)
        if logger and (fill_attempt == 0 or fill_attempt % 10 == 0 or fill_slot_retry > 0):
            logger.info(
                f'[ScaffoldPrudent] final_fill need={need} pool={len(final_pool)}/{target_n} '
                f'attempt={fill_attempt} sample_idx={sample_idx} slot_retry={fill_slot_retry}'
            )
        cand = _prudent_locked_sample_once(
            model=model, data=data, device=device, grow_cfg=grow_cfg,
            _attachment_sites=_attachment_sites, _murcko_sites_cfg=_murcko_sites_cfg,
            protein_centroid=protein_centroid, ref_ligand_pos=ref_ligand_pos,
            log_ref_v=log_ref_v, scaffold_indices=scaffold_indices, n_scaffold=n_scaffold,
            scaffold_bond_pairs=scaffold_bond_pairs, fix_scaffold_pos=fix_scaffold_pos,
            fix_scaffold_type=fix_scaffold_type, log_mode_is_log_prob=log_mode_is_log_prob,
            large_cfg=large_cfg, refine_cfg=refine_cfg, time_boundary=time_boundary,
            center_pos_mode=center_pos_mode, pos_only=pos_only,
            ligand_atom_mode=ligand_atom_mode, selector_cfg=selector_cfg,
            sample_idx=sample_idx, slot_retry=fill_slot_retry, logger=logger,
            log_tag='[ScaffoldPrudent][final_fill]',
        )
        fill_attempt += 1
        if cand is None:
            fill_slot_retry += 1
            continue
        if _hard_gate:
            mol = _prudent_reconstruct_mol(
                cand['pos'], cand['v'], ligand_atom_mode,
                logger=logger, context=f'SPfill{fill_attempt}',
                scaffold_bonds=scaffold_bond_pairs, n_scaffold=n_scaffold,
            )
            qed, sa, logp, chem_err = _prudent_chem_scores(mol, logger=logger)
            gate_ok, reasons = _prudent_size_gate_ok(
                mol, qed, sa, logp, prudent_cfg, gates=_size_gates,
            )
            if not gate_ok:
                if logger and (fill_attempt <= 5 or fill_attempt % 50 == 0):
                    logger.info(
                        f'[ScaffoldPrudent] final_fill gate-reject #{fill_attempt}: '
                        + ', '.join(reasons or ['unknown'])
                        + f' | pool={len(final_pool)}/{target_n}'
                    )
                fill_slot_retry += 1
                continue
        else:
            mol = None
        fill_smi = _prudent_canonical_smiles(
            mol if mol is not None else cand,
            ligand_atom_mode=ligand_atom_mode,
            scaffold_bonds=scaffold_bond_pairs, n_scaffold=n_scaffold,
            logger=None, context=f'SPfill_dedup{fill_attempt}',
        )
        if fill_smi and fill_smi in _final_seen_smiles:
            _fill_dedup_reject += 1
            if logger and (_fill_dedup_reject <= 5 or _fill_dedup_reject % 50 == 0):
                logger.info(
                    f'[ScaffoldPrudent] final_fill SMILES-dedup-reject #{_fill_dedup_reject} '
                    f'smi={fill_smi[:80]} | pool={len(final_pool)}/{target_n}'
                )
            fill_slot_retry += 1
            continue
        # strip helper-only keys
        cand.pop('site_place_meta', None)
        cand.pop('metric_info', None)
        cand.pop('zero_allocation_site_ids', None)
        cand['from_final_fill'] = True
        cand['fill_attempt'] = fill_attempt
        if fill_smi:
            cand['smiles'] = fill_smi
            _final_seen_smiles.add(fill_smi)
        final_pool.append(cand)
        fill_slot_retry = 0
        if logger and len(final_pool) % 10 == 0:
            logger.info(
                f'[ScaffoldPrudent] final_fill 已凑 {len(final_pool)}/{target_n}'
            )

    if len(final_pool) < target_n and logger:
        logger.warning(
            f'[ScaffoldPrudent] final_fill 结束仍不足: {len(final_pool)}/{target_n} '
            f'(attempts={fill_attempt}/{max_final_fill_attempts} '
            f'fill_dedup_reject={_fill_dedup_reject})'
        )
    pool = final_pool
    n_final_pool = len(pool)
    _n_unique_final = len({c.get('smiles') for c in pool if c.get('smiles')})
    if logger:
        logger.info(
            f'[ScaffoldPrudent] 最后一轮最终分子数 n_final_pool={n_final_pool} '
            f'(target_n={target_n}, n_unique_smiles={_n_unique_final}, '
            f'final_dedup_reject={_final_dedup_reject}, fill_dedup_reject={_fill_dedup_reject})'
        )

    # ---- 输出最终分子池（含所有累积分子 + 最终 pool）----------------------
    pos_list, v_list = [], []
    pos_traj_list, v_traj_list, log_v_traj_list = [], [], []
    time_list = []
    out_meta_records = list(meta_records_all)  # Gen 0 records

    # 1) 先添加所有累积的分子（Gen 0 + Gen 1..N-1 的完整 pool）
    for ai, (apos, av) in enumerate(zip(all_pos_accum, all_v_accum)):
        pos_list.append(apos)
        v_list.append(av)
        pos_traj_list.append([])
        v_traj_list.append([])
        log_v_traj_list.append([])
        time_list.append(0.0)
        if ai < len(all_meta_accum):
            am = dict(all_meta_accum[ai])
            am['is_original'] = False
            am['time'] = 0.0
            out_meta_records.append(am)

    final_pool_start = len(pos_list)

    # 2) 再添加最终 pool（最后一代 + final_fill 凑满的分子）
    for ci, c in enumerate(pool):
        # 当 _hard_gate 时，跳过门控失败分子
        if _hard_gate:
            detail = c.get('prudent_score_detail', {})
            if detail.get('vina_skipped', False):
                continue
        pos_list.append(np.asarray(c['pos'], dtype=np.float64))
        v_list.append(np.asarray(c['v'], dtype=np.int64))
        pos_traj_list.append(c.get('pos_traj') or [])
        v_traj_list.append(c.get('v_traj') or [])
        log_v_traj_list.append(c.get('log_v_traj') or [])
        time_list.append(0.0)

        metric_info = evaluate_candidate(
            np.asarray(c['pos']), np.asarray(c['v']),
            ligand_atom_mode, selector_cfg,
            scaffold_bonds=c.get('scaffold_bond_pairs', scaffold_bond_pairs),
            n_scaffold=int(c.get('n_scaffold', n_scaffold)),
        )
        final_gen = n_generations - 1
        out_meta_records.append({
            'method': 'scaffold_prudent',
            'generation': final_gen,
            'final_pool': True,
            'from_final_fill': bool(c.get('from_final_fill', False)),
            'ligand_num_atoms': c.get('num_atoms', 0),
            'n_scaffold': c.get('n_scaffold', n_scaffold),
            'n_extra': c.get('n_extra', 0),
            'time': 0.0,
            'smiles': metric_info.get('smiles'),
            'qed': metric_info.get('metrics', {}).get('qed'),
            'sa': metric_info.get('metrics', {}).get('sa'),
            'status': metric_info.get('status'),
            'is_original': False,
        })

    n_final_pool = len(pos_list) - final_pool_start
    if logger:
        final_valid = sum(1 for m in out_meta_records if not m.get('is_original') and m.get('status') == 'ok')
        gate_note = ' [gate-filtered]' if _hard_gate else ''
        logger.info(
            f'[ScaffoldPrudent] 最终输出{gate_note}: {final_valid}/{len(pos_list)} 有效分子 | '
            f'累积 {len(all_pos_accum)} + 最终 pool {n_final_pool} = {len(pos_list)} 个坐标 | '
            f'final_pool_start={final_pool_start} | 总 meta 记录 {len(out_meta_records)}'
        )

    return {
        'pos_list': pos_list,
        'v_list': v_list,
        'pos_traj': pos_traj_list,
        'v_traj': v_traj_list,
        'log_v_traj': log_v_traj_list,
        'time_list': time_list,
        'meta': {
            'method': 'scaffold_prudent',
            'records': out_meta_records,
            'n_final_pool': n_final_pool,
            'final_pool_start': final_pool_start,
            'target_final_count': target_n,
            'scaffold_cfg': {
                'scaffold_source': sc_cfg.get('scaffold_source'),
                'n_scaffold_atoms': n_scaffold,
                'scaffold_indices': scaffold_indices,
                'fix_scaffold_pos': fix_scaffold_pos,
                'fix_scaffold_type': fix_scaffold_type,
                'time_boundary': time_boundary,
                'n_generations': n_generations,
                'renoise_t': renoise_t,
                'advance_top_k': advance_top_k,
                'n_chains_per_seed': n_chains_per_seed,
                'num_samples': num_samples,
                'target_final_count': target_n,
            },
            'generation_stats': generation_stats,
            'total_time': total_time,
        },
    }
