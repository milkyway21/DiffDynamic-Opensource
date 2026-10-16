"""Sampling scaffold evolve responsibilities for ligand sampling."""

from __future__ import annotations

from diffdynamic.data.graph import FOLLOW_BATCH
from diffdynamic.models.probability import center_pos
from diffdynamic.models.probability import index_to_log_onehot
from torch_geometric.data import Batch
import numpy as np
import time
import torch


def scaffold_optimize_molecule(
    model,
    data,
    config,
    ligand_atom_mode: str,
    device: str = 'cuda:0',
    logger=None,
) -> dict:
    """
    骨架约束进化优化（主入口）。

    流水线：
        1. 解析骨架掩码（pos_mask / type_mask）
        2. 进化循环（n_generations 代）：
           a. 对当前种群每个成员：前向扩散到 start_t（自适应退火）→ 双掩码反扩散
           b. 评估所有子代（QED/SA/多样性/骨架RMSD）
           c. 多样性感知种群选择，保留前 population_size 个
        3. 收集最终种群，可选保留原始参考分子

    Args:
        model: 扩散模型实例。
        data: ProteinLigandData（含蛋白口袋 + 参考配体）。
        config: 采样配置。
        ligand_atom_mode: 原子编码模式。
        device: 推理设备。
        logger: 可选日志器。

    Returns:
        dict: 与 optimize_molecule 格式兼容的结果字典。
    """
    from diffdynamic.sampling.optimization import _forward_diffuse_molecule
    from diffdynamic.sampling.scaffold.fitness import _compute_scaffold_fitness
    from diffdynamic.sampling.scaffold.fitness import _diversity_filter_population
    from diffdynamic.sampling.scaffold.inputs import _extract_scaffold_from_reference
    from diffdynamic.sampling.scaffold.inputs import _get_scaffold_mode_cfg
    from diffdynamic.sampling.scaffold.inputs import _resolve_scaffold_mask

    from diffdynamic.sampling.guidance import DualMaskedDiffusionSampler

    scaffold_cfg, _ = _get_scaffold_mode_cfg(config)  # 读统一 scaffold 键，evolve 子节已覆盖顶层
    center_pos_mode = config.sample.get('center_pos_mode', 'protein')
    pos_only = config.sample.get('pos_only', False)

    if not hasattr(data, 'ligand_pos') or data.ligand_pos is None or data.ligand_pos.size(0) == 0:
        raise ValueError(
            '骨架约束优化需要参考配体（含 3D 坐标 SDF），请通过 --ligand_path 指定。'
        )

    ref_ligand_pos = data.ligand_pos.to(device)
    ref_ligand_v = data.ligand_atom_feature_full.to(device)
    ref_num_atoms = ref_ligand_pos.size(0)
    log_ref_v = index_to_log_onehot(ref_ligand_v, model.num_classes)

    ref_pos_np = ref_ligand_pos.detach().cpu().numpy().astype(np.float64)
    ref_v_np = np.argmax(log_ref_v.detach().cpu().numpy(), axis=-1).astype(np.int64)

    # ---- 解析骨架掩码 ---------------------------------------------------------
    if scaffold_cfg.get('web_input_contract'):
        indices, _, _, _ = _extract_scaffold_from_reference(
            data, scaffold_cfg, ligand_atom_mode, ref_ligand_pos,
            ref_ligand_v, log_ref_v, ref_num_atoms, device, config=config,
            logger=logger,
        )
        scaffold_mask = torch.zeros(ref_num_atoms, device=device)
        scaffold_mask[indices] = 1.0
    else:
        scaffold_mask = _resolve_scaffold_mask(
            scaffold_cfg, ref_ligand_pos, ref_ligand_v,
            ligand_atom_mode, ref_num_atoms, device,
        )
    scaffold_mask_np = scaffold_mask.cpu().numpy()
    n_scaffold = int(scaffold_mask_np.sum())
    if logger:
        logger.info(
            f'[ScaffoldOpt] 骨架原子: {n_scaffold}/{ref_num_atoms} '
            f'(来源: {scaffold_cfg.get("scaffold_source", "none")}) | '
            f'固定位置={scaffold_cfg.get("fix_scaffold_pos", True)}, '
            f'固定类型={scaffold_cfg.get("fix_scaffold_type", False)}'
        )

    fix_pos = scaffold_cfg.get('fix_scaffold_pos', True)
    fix_type = scaffold_cfg.get('fix_scaffold_type', False)
    pos_mask_tensor = scaffold_mask if fix_pos else torch.zeros(ref_num_atoms, device=device)
    type_mask_tensor = scaffold_mask if fix_type else None

    # ---- 进化参数 -------------------------------------------------------------
    population_size = int(scaffold_cfg.get('population_size', 20))
    n_generations = int(scaffold_cfg.get('n_generations', 8))
    children_per_parent = int(scaffold_cfg.get('children_per_parent', 5))
    start_t_high = int(scaffold_cfg.get('start_t_high', 20))
    start_t_low = int(scaffold_cfg.get('start_t_low', 6))
    noise_anneal = bool(scaffold_cfg.get('noise_anneal', True))
    stride = int(scaffold_cfg.get('stride', 2))
    step_size = float(scaffold_cfg.get('step_size', 0.2))
    schedule = scaffold_cfg.get('schedule', 'linear')
    use_with_noise = bool(scaffold_cfg.get('use_with_noise', True))
    use_adaptive_step = bool(scaffold_cfg.get('use_adaptive_step', True))
    use_time_scale = bool(scaffold_cfg.get('use_time_scale', False))
    keep_original = bool(scaffold_cfg.get('keep_original', True))

    if logger:
        logger.info(
            f'[ScaffoldOpt] 进化参数: population={population_size}, '
            f'generations={n_generations}, children/parent={children_per_parent} | '
            f'噪声退火: t {start_t_high}→{start_t_low}'
        )

    # ---- 初始种群（单个参考分子）----------------------------------------------
    population = [{'pos': ref_pos_np.copy(), 'v': ref_v_np.copy(), 'fitness': 0.0, 'smiles': None, 'status': 'seed'}]

    pos_list, v_list = [], []
    pos_traj_list, v_traj_list, log_v_traj_list = [], [], []
    time_list = []
    meta_records = []
    total_time = 0.0

    # ---- 进化循环 -------------------------------------------------------------
    for gen_idx in range(n_generations):
        # 自适应噪声退火
        if noise_anneal and n_generations > 1:
            frac = gen_idx / (n_generations - 1)
            start_t = int(start_t_high - frac * (start_t_high - start_t_low))
        else:
            start_t = start_t_high
        start_t = max(start_t, 2)

        if logger:
            logger.info(
                f'[ScaffoldOpt] 第 {gen_idx + 1}/{n_generations} 代 | '
                f'种群: {len(population)} 个 | start_t={start_t}'
            )

        # 构建去噪时间索引
        if schedule == 'lambda':
            lambda_a = float(scaffold_cfg.get('lambda_coeff_a', 1.0))
            lambda_b = float(scaffold_cfg.get('lambda_coeff_b', 1.0))
            time_indices = model._build_lambda_schedule(start_t=start_t, end_t=0, coeff_a=lambda_a, coeff_b=lambda_b)
        elif schedule == 'linear':
            total_steps = max(start_t // max(stride, 1), 10)
            time_indices = np.linspace(start_t, 0, total_steps + 1, dtype=int).tolist()
            time_indices = sorted(set(time_indices), reverse=True)
        else:
            time_indices = list(range(start_t, -1, -max(stride, 1)))
        if time_indices and time_indices[-1] != 0:
            time_indices.append(0)

        all_children = []

        for parent in population:
            pos_t = torch.tensor(parent['pos'], dtype=torch.float32, device=device)
            log_v_t = torch.tensor(
                np.log(np.clip(np.eye(model.num_classes, dtype=np.float32)[parent['v']], 1e-8, 1.0)),
                device=device,
            )

            # 批量化：每个父代生成 children_per_parent 个子代
            data_list = [data.clone() for _ in range(children_per_parent)]
            batch = Batch.from_data_list(data_list, follow_batch=FOLLOW_BATCH).to(device)
            batch_protein = batch.protein_element_batch
            batch_ligand = batch.ligand_element_batch

            pos_centered, ref_centered, offset = center_pos(
                batch.protein_pos,
                pos_t.unsqueeze(0).expand(children_per_parent, -1, -1).reshape(-1, 3),
                batch_protein, batch_ligand, mode=center_pos_mode,
            )
            log_v_batch = log_v_t.unsqueeze(0).expand(children_per_parent, -1, -1).reshape(-1, model.num_classes)

            # 骨架掩码扩展（与批次 batch_ligand 对齐）
            pos_mask_batch = pos_mask_tensor.unsqueeze(0).expand(children_per_parent, -1).reshape(-1)
            type_mask_batch = (
                type_mask_tensor.unsqueeze(0).expand(children_per_parent, -1).reshape(-1)
                if type_mask_tensor is not None else None
            )

            # 构建双掩码 RePaint 配置
            repaint_cfg = {
                'x0_pos': ref_centered,          # 参考坐标（已居中）
                'x0_log_v': log_v_batch,
                'pos_mask': pos_mask_batch,       # 位置掩码（骨架）
                'type_mask': type_mask_batch,     # 类型掩码（可为 None）
                'use_mean_for_discrete': True,
                '_use_dual_mask': True,           # 标记使用双掩码模式
            }

            t_start_wall = time.time()
            with torch.no_grad():
                noised_pos, noised_log_v = _forward_diffuse_molecule(
                    model, ref_centered, log_v_batch, batch_ligand, start_t, device
                )
                ligand_pos_out, log_v_out, pos_traj, log_v_traj = model._dynamic_diffusion(
                    protein_pos=batch.protein_pos,
                    protein_v=batch.protein_atom_feature.float(),
                    batch_protein=batch_protein,
                    ligand_pos=noised_pos,
                    log_ligand_v=noised_log_v,
                    batch_ligand=batch_ligand,
                    time_indices=time_indices,
                    step_size=step_size,
                    add_noise=0.0,
                    record_traj=False,
                    pos_only=pos_only,
                    use_with_noise=use_with_noise,
                    use_adaptive_step=use_adaptive_step,
                    use_time_scale=use_time_scale,
                    repaint_cfg=repaint_cfg,
                )
            t_end_wall = time.time()
            total_time += t_end_wall - t_start_wall

            ligand_pos_final = ligand_pos_out + offset[batch_ligand]

            # 收集子代
            existing_smiles = [p.get('smiles') for p in population if p.get('smiles')]
            for ci in range(children_per_parent):
                mask = (batch_ligand == ci)
                pos_c = ligand_pos_final[mask].detach().cpu().numpy().astype(np.float64)
                v_c = log_v_out[mask].argmax(dim=-1).detach().cpu().numpy().astype(np.int64)
                fitness_info = _compute_scaffold_fitness(
                    pos_c, v_c, ligand_atom_mode,
                    scaffold_cfg, ref_pos_np, scaffold_mask_np,
                    existing_smiles,
                )
                all_children.append({
                    'pos': pos_c,
                    'v': v_c,
                    'pos_traj': [],
                    'v_traj': [],
                    **fitness_info,
                })

        # 种群选择（多样性感知）
        candidates = all_children
        if gen_idx > 0:
            candidates = candidates + [
                c for c in population if c.get('status') == 'ok'
            ]
        population = _diversity_filter_population(candidates, population_size, scaffold_cfg)

        if not population:
            if logger:
                logger.warning(f'[ScaffoldOpt] 第 {gen_idx + 1} 代种群全部被过滤，保留父代')
            population = [{'pos': ref_pos_np.copy(), 'v': ref_v_np.copy(), 'fitness': 0.0, 'smiles': None, 'status': 'seed'}]
        elif logger:
            best = population[0]
            logger.info(
                f'[ScaffoldOpt] 第 {gen_idx + 1} 代筛选后: {len(population)} 个 | '
                f'最优 fitness={best.get("fitness", 0.0):.3f} '
                f'(QED={best.get("qed", 0.0):.3f}, SA={best.get("sa", 0.0):.3f})'
            )

    # ---- 收集最终种群结果 -----------------------------------------------------
    if keep_original:
        metric_orig = _compute_scaffold_fitness(
            ref_pos_np, ref_v_np, ligand_atom_mode,
            scaffold_cfg, ref_pos_np, scaffold_mask_np, [],
        )
        pos_list.append(ref_pos_np)
        v_list.append(ref_v_np)
        pos_traj_list.append([])
        v_traj_list.append([])
        log_v_traj_list.append([])
        time_list.append(0.0)
        meta_records.append({
            'method': 'scaffold_optimization',
            'ligand_num_atoms': ref_num_atoms,
            'time': 0.0,
            'rmsd_from_ref': 0.0,
            'scaffold_rmsd': 0.0,
            'smiles': metric_orig.get('smiles'),
            'qed': metric_orig.get('qed'),
            'sa': metric_orig.get('sa'),
            'fitness': metric_orig.get('fitness'),
            'status': metric_orig.get('status'),
            'is_original': True,
            'generation': -1,
        })

    for rank, cand in enumerate(population):
        pos_list.append(cand['pos'])
        v_list.append(cand['v'])
        pos_traj_list.append(cand.get('pos_traj', []))
        v_traj_list.append(cand.get('v_traj', []))
        log_v_traj_list.append([])
        time_list.append(0.0)
        rmsd_all = float(np.sqrt(np.mean((cand['pos'] - ref_pos_np) ** 2)))
        meta_records.append({
            'method': 'scaffold_optimization',
            'ligand_num_atoms': ref_num_atoms,
            'time': 0.0,
            'rmsd_from_ref': rmsd_all,
            'scaffold_rmsd': cand.get('scaffold_rmsd', float('nan')),
            'smiles': cand.get('smiles'),
            'qed': cand.get('qed'),
            'sa': cand.get('sa'),
            'fitness': cand.get('fitness'),
            'diversity_bonus': cand.get('diversity_bonus'),
            'status': cand.get('status'),
            'is_original': False,
            'rank': rank,
        })

    if logger:
        valid = sum(1 for m in meta_records if m.get('status') == 'ok' and not m.get('is_original'))
        logger.info(
            f'[ScaffoldOpt] 完成: {valid}/{len(population)} 有效优化分子 | '
            f'总耗时 {total_time:.2f}s'
        )

    return {
        'pos_list': pos_list,
        'v_list': v_list,
        'pos_traj': pos_traj_list,
        'v_traj': v_traj_list,
        'log_v_traj': log_v_traj_list,
        'time_list': time_list,
        'meta': {
            'method': 'scaffold_optimization',
            'records': meta_records,
            'scaffold_cfg': {
                'scaffold_source': scaffold_cfg.get('scaffold_source'),
                'n_scaffold_atoms': n_scaffold,
                'fix_scaffold_pos': fix_pos,
                'fix_scaffold_type': fix_type,
                'population_size': population_size,
                'n_generations': n_generations,
                'children_per_parent': children_per_parent,
                'start_t_high': start_t_high,
                'start_t_low': start_t_low,
                'noise_anneal': noise_anneal,
            },
            'total_time': total_time,
        },
    }
