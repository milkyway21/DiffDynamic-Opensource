"""Sampling scaffold grow responsibilities for ligand sampling."""

from __future__ import annotations

from datetime import datetime
from datetime import timedelta
from datetime import timezone
from diffdynamic.data.graph import FOLLOW_BATCH
from diffdynamic.models.probability import center_pos
from diffdynamic.models.probability import index_to_log_onehot
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
import time
import torch
import torch.nn.functional as F


def scaffold_grow_molecule(
    model,
    data,
    config,
    ligand_atom_mode: str,
    device: str = 'cuda:0',
    logger=None,
    output_dir: str = None,
) -> dict:
    """
    骨架生长模式：固定骨架，从零生成骨架以外的原子（原子数可变）。

    与 scaffold_optimize_molecule 的核心区别：
    - optimize：输入完整分子，在保持骨架位置的同时"优化"已有原子（原子总数不变）
    - grow    ：骨架以外的原子完全从零生成（原子数由口袋先验决定，可增可减）

    流水线：
        1. 解析骨架掩码（支持 auto_murcko / smarts / atom_indices）
        2. 对每个样本：
           a. 采样 N_extra（口袋先验 - 骨架原子数，或固定/范围模式）
           b. 构建混合初始态：
              · 骨架原子：原始位置（中心化后）+ 原始类型
              · N_extra 原子：口袋中心附近的随机高斯噪声 + 均匀类别分布
           c. 运行 _dynamic_diffusion（从 start_t≈T 开始全程去噪）
              · 双掩码 RePaint：骨架位置和类型全程锚定
              · 新原子自由演化以适应口袋形状
        3. 评估和记录（QED/SA/骨架RMSD）

    Args:
        model: 扩散模型实例。
        data: ProteinLigandData（含蛋白口袋 + 参考配体，参考配体用于提供骨架）。
        config: 采样配置。
        ligand_atom_mode: 原子编码模式。
        device: 推理设备。
        logger: 可选日志器。

    Returns:
        dict: 与 optimize_molecule / scaffold_optimize_molecule 格式兼容的结果字典。
    """
    from diffdynamic.sampling.candidates import evaluate_candidate
    from diffdynamic.sampling.optimization import _forward_diffuse_molecule
    from diffdynamic.sampling.scaffold.inputs import _extract_scaffold_from_reference
    from diffdynamic.sampling.scaffold.inputs import _extract_scaffold_submol
    from diffdynamic.sampling.scaffold.inputs import _get_scaffold_mode_cfg
    from diffdynamic.sampling.scaffold.inputs import _pin_place_cfg_to_n_extra
    from diffdynamic.sampling.scaffold.inputs import _sample_n_extra_atoms
    from diffdynamic.sampling.scaffold.placement import _murcko_placement_cfg
    from diffdynamic.sampling.scaffold.placement import _murcko_sample_rng
    from diffdynamic.sampling.scaffold.placement import _resolve_init_preserved_atoms

    from diffdynamic.sampling.guidance import DualMaskedDiffusionSampler

    grow_cfg, _ = _get_scaffold_mode_cfg(config)   # 读统一 scaffold 键，grow 子节已覆盖顶层
    center_pos_mode = config.sample.get('center_pos_mode', 'protein')
    pos_only = config.sample.get('pos_only', False)

    if not hasattr(data, 'ligand_pos') or data.ligand_pos is None or data.ligand_pos.size(0) == 0:
        raise ValueError(
            '骨架生长模式需要参考配体（含 3D 坐标 SDF）以提取骨架，请通过 --ligand_path 指定。'
        )

    ref_ligand_pos = data.ligand_pos.to(device)
    ref_ligand_v = data.ligand_atom_feature_full.to(device)
    n_scaffold_ref = ref_ligand_pos.size(0)
    log_ref_v = index_to_log_onehot(ref_ligand_v, model.num_classes)

    ref_pos_np = ref_ligand_pos.detach().cpu().numpy().astype(np.float64)
    ref_v_np = np.argmax(log_ref_v.detach().cpu().numpy(), axis=-1).astype(np.int64)

    # ---- 解析骨架掩码（复用共享提取逻辑）-------------------------------------
    scaffold_indices, ref_mol, ref_ligand_name, n_scaffold = _extract_scaffold_from_reference(
        data, grow_cfg, ligand_atom_mode, ref_ligand_pos, ref_ligand_v, log_ref_v,
        n_scaffold_ref, device, config=config, logger=logger,
    )

    if logger:
        logger.info(
            f'[ScaffoldGrow] 骨架原子: {n_scaffold}/{n_scaffold_ref} '
            f'(来源: {grow_cfg.get("scaffold_source", "auto_murcko")}) | '
            f'额外原子策略: {grow_cfg.get("n_extra_mode", "prior_minus_scaffold")}'
        )

    if output_dir is not None:
        _grow_base_output_dir = Path(output_dir)
    else:
        _grow_base_output_dir = Path(config.sample.get('output_dir', './outputs'))

    _murcko_sites_cfg = get_murcko_sites_cfg(grow_cfg)
    _attachment_sites = load_or_extract_attachment_sites(
        ref_mol, scaffold_indices,
        grow_cfg.get('scaffold_source', 'auto_murcko'),
        ref_pos_np, _grow_base_output_dir, ref_ligand_name,
        _murcko_sites_cfg, logger=logger,
    )

    # ---- 保存骨架 SDF 结构 ------------------------------------------------------
    # 从原始参考配体中提取并保存骨架结构（使用前面已加载的 ref_mol）
    try:
        # 如果前面未成功加载 ref_mol，再次尝试从 ligand_mol 属性获取
        if ref_mol is None and hasattr(data, 'ligand_mol') and data.ligand_mol is not None:
            ref_mol = data.ligand_mol
            if hasattr(data, 'ligand_filename') and data.ligand_filename:
                ref_ligand_name = Path(data.ligand_filename).stem
        
        # 如果成功获取参考分子且找到骨架，保存骨架 SDF
        if ref_mol is not None and n_scaffold > 0:
            scaffold_mol = _extract_scaffold_submol(ref_mol, scaffold_indices, logger)
            if scaffold_mol is not None:
                # 确定输出路径（使用传入的 output_dir 或默认 outputs）
                base_output_dir = _grow_base_output_dir
                
                # 创建 scaffold 子目录
                scaffold_output_dir = base_output_dir / 'scaffold'
                scaffold_output_dir.mkdir(parents=True, exist_ok=True)
                
                # 生成骨架 SDF 文件名：原始配体名 + _scaffold
                if ref_ligand_name:
                    scaffold_filename = f'{ref_ligand_name}_scaffold.sdf'
                else:
                    # 备用命名：使用骨架原子数和时间戳
                    cst = timezone(timedelta(hours=8))
                    timestamp = datetime.now(cst).strftime('%Y%m%d_%H%M%S')
                    scaffold_filename = f'scaffold_{n_scaffold}atoms_{timestamp}.sdf'
                
                scaffold_sdf_path = scaffold_output_dir / scaffold_filename
                
                # 保存骨架 SDF
                writer = Chem.SDWriter(str(scaffold_sdf_path))
                writer.write(scaffold_mol)
                writer.close()
                
                if logger:
                    logger.info(f'[ScaffoldGrow] 骨架结构已保存: {scaffold_sdf_path}')
                    logger.info(f'[ScaffoldGrow] 骨架原子数: {scaffold_mol.GetNumAtoms()}, 键数: {scaffold_mol.GetNumBonds()}')
    except Exception as e:
        if logger:
            logger.warning(f'[ScaffoldGrow] 保存骨架 SDF 失败: {e}')

    # ---- 生成参数 ------------------------------------------------------------
    num_samples = int(grow_cfg.get('num_samples', 10))
    start_t = int(np.clip(grow_cfg.get('start_t', 357), 1, model.num_timesteps - 1))
    stride = int(grow_cfg.get('stride', 15))
    step_size = float(grow_cfg.get('step_size', 0.3262))
    schedule = grow_cfg.get('schedule', 'lambda')
    use_with_noise = bool(grow_cfg.get('use_with_noise', True))
    use_adaptive_step = bool(grow_cfg.get('use_adaptive_step', True))
    use_time_scale = bool(grow_cfg.get('use_time_scale', False))

    # ---- 构建反扩散时间索引 --------------------------------------------------
    if schedule == 'lambda':
        lambda_a = float(grow_cfg.get('lambda_coeff_a', 47.0))
        lambda_b = float(grow_cfg.get('lambda_coeff_b', 11.0))
        time_indices = model._build_lambda_schedule(
            start_t=start_t, end_t=0, coeff_a=lambda_a, coeff_b=lambda_b
        )
    elif schedule == 'linear':
        total_steps = max(start_t // max(stride, 1), 20)
        time_indices = np.linspace(start_t, 0, total_steps + 1, dtype=int).tolist()
        time_indices = sorted(set(time_indices), reverse=True)
    else:
        time_indices = list(range(start_t, -1, -max(stride, 1)))
    if time_indices and time_indices[-1] != 0:
        time_indices.append(0)

    pos_list, v_list = [], []
    pos_traj_list, v_traj_list, log_v_traj_list = [], [], []
    time_list = []
    meta_records = []
    total_time = 0.0

    selector_cfg = {
        'min_qed': grow_cfg.get('min_qed', 0.2),
        'min_sa': grow_cfg.get('min_sa', 0.2),
        'filter_incomplete': grow_cfg.get('filter_incomplete', False),
        'qed_weight': 1.0,
        'sa_weight': 1.0,
    }

    # ---- 对每个样本独立采样 N_extra（每次可不同）-----------------------------
    for sample_idx in range(num_samples):
        n_extra = _sample_n_extra_atoms(
            grow_cfg, data.protein_pos, n_scaffold, model.num_timesteps, sample_idx=sample_idx,
        )
        _place_cfg = _pin_place_cfg_to_n_extra(
            _murcko_placement_cfg(grow_cfg, _murcko_sites_cfg), n_extra, grow_cfg,
        )
        if not uses_site_budget_placement(_place_cfg):
            n_extra = cap_n_extra_for_sites(n_extra, _attachment_sites, _murcko_sites_cfg)
        n_total = n_scaffold + n_extra

        if logger and (sample_idx == 0 or sample_idx % 10 == 0):
            logger.info(
                f'[ScaffoldGrow] 样本 {sample_idx + 1}/{num_samples} | '
                f'参考骨架原子: {n_scaffold} + 新原子: {n_extra} = '
                f'参考总计: {n_total}'
            )

        # ---- 构建单样本批次 -------------------------------------------------
        batch = Batch.from_data_list([data.clone()], follow_batch=FOLLOW_BATCH).to(device)
        batch_protein = batch.protein_element_batch

        # 口袋中心（新原子初始化位置）
        pocket_center = scatter_mean(batch.protein_pos, batch_protein, dim=0)[0]  # [3]

        # 骨架原子坐标（原始参考位置，居中前）
        scaffold_pos_orig = ref_ligand_pos[scaffold_indices] if n_scaffold > 0 else torch.zeros(0, 3, device=device)
        scaffold_log_v_orig = log_ref_v[scaffold_indices] if n_scaffold > 0 else torch.zeros(0, model.num_classes, device=device)

        # 新原子初始坐标：Murcko 侧链质心位点（grow/dynamic_locked/prudent 共用）；无位点时回退口袋中心
        extra_pos_init, _site_place_meta = build_extra_atom_positions(
            n_extra, _attachment_sites, _place_cfg,
            pocket_center, device, fallback_noise_scale=2.0, logger=logger,
            rng=_murcko_sample_rng(grow_cfg, sample_idx),
            protein_positions=batch.protein_pos,
        )
        n_extra = int(extra_pos_init.size(0))
        (
            preserved_pos, preserved_log_v, preserved_indices, zero_site_ids,
            n_preserved, n_locked, n_total,
        ) = _resolve_init_preserved_atoms(
            _attachment_sites, _site_place_meta, _place_cfg,
            ref_ligand_pos, log_ref_v, n_scaffold, n_extra,
            logger=logger, log_tag='[ScaffoldGrow]', sample_idx=sample_idx,
        )

        # 新原子初始类型：偏向 C/N/O/F，屏蔽 H、压低稀有元素权重
        # H(0) 屏蔽：只能成 1 个键，作为重原子会破坏重建
        # Cl(12) 压低：单价，容易超价
        # P(8,9) S(10,11) 压低：稀有，价态复杂
        extra_v_bias = torch.ones(model.num_classes, device=device)
        extra_v_bias[0] = -1e9   # H: 屏蔽
        extra_v_bias[12] = 0.05   # Cl: 极低保底（单价）
        extra_v_bias[8] = 0.2    # P (非芳香): 低价
        extra_v_bias[9] = 0.2    # P (芳香): 低价
        extra_v_bias[10] = 0.3   # S (非芳香): 低价
        extra_v_bias[11] = 0.3   # S (芳香): 低价
        extra_v_bias[7] = 0.7    # F: 单价但在药物中常见，中等偏低价
        extra_log_v_init = F.log_softmax(
            torch.ones(n_extra, model.num_classes, device=device) * extra_v_bias, dim=-1
        ) if n_extra > 0 else torch.zeros(0, model.num_classes, device=device)

        # 拼接：[骨架 | 零分配保留侧链 | 新原子]
        init_ligand_pos = torch.cat(
            [scaffold_pos_orig, preserved_pos, extra_pos_init], dim=0
        )  # [n_total, 3]
        init_log_ligand_v = torch.cat(
            [scaffold_log_v_orig, preserved_log_v, extra_log_v_init], dim=0
        )  # [n_total, C]
        batch_ligand = torch.zeros(n_total, dtype=torch.long, device=device)

        # 中心化：将初始位置对齐到口袋中心
        protein_pos_centered, init_pos_centered, offset = center_pos(
            batch.protein_pos, init_ligand_pos,
            batch_protein, batch_ligand, mode=center_pos_mode,
        )

        # ---- 构建 RePaint 参考态 -------------------------------------------
        # 骨架 + 保留侧链锚定；额外原子用软约束（fractional pos_mask）
        # 软约束让额外原子在扩散过程中被弹性拉回初始放置位置，防止漂移断键
        x0_pos_ref = init_pos_centered.clone()
        x0_log_v_ref = init_log_ligand_v.clone()

        extra_anchor_strength = float(grow_cfg.get('extra_anchor_strength', 0.0))
        pos_mask_local = torch.full((n_total,), extra_anchor_strength, device=device) \
            if extra_anchor_strength > 0.0 else torch.zeros(n_total, device=device)
        type_mask_local = torch.zeros(n_total, device=device)
        if n_locked > 0:
            pos_mask_local[:n_locked] = 1.0
            type_mask_local[:n_locked] = 1.0

        repaint_cfg = {
            'x0_pos': x0_pos_ref,
            'x0_log_v': x0_log_v_ref,
            'pos_mask': pos_mask_local,
            'type_mask': type_mask_local if n_locked > 0 else None,
            'use_mean_for_discrete': True,
            '_use_dual_mask': True,
        } if (n_locked > 0 or extra_anchor_strength > 0.0) else None

        if repaint_cfg is not None and logger is not None:
            _pm = repaint_cfg['pos_mask']
            logger.info(f'[ScaffoldGrow] repaint_cfg: n_locked={n_locked}, n_extra={n_extra}, '
                        f'extra_anchor_strength={extra_anchor_strength}, '
                        f'pos_mask[:n_locked]={_pm[:n_locked].tolist()[:3]}..., '
                        f'pos_mask[n_locked:]={_pm[n_locked:].tolist()[:3]}...')

        # ---- 前向扩散初始化（从 start_t 开始反扩散）------------------------
        # 新原子直接用纯噪声，骨架原子也加噪（RePaint 会在每步拉回）
        with torch.no_grad():
            noised_pos, noised_log_v = _forward_diffuse_molecule(
                model, init_pos_centered, init_log_ligand_v, batch_ligand, start_t, device
            )

        t_start_wall = time.time()
        with torch.no_grad():
            ligand_pos_out, log_v_out, pos_traj, log_v_traj = model._dynamic_diffusion(
                protein_pos=protein_pos_centered,
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

        if logger is not None and n_extra > 0:
            _drift = (ligand_pos_out[n_locked:] - init_pos_centered[n_locked:]).norm(dim=-1).mean().item()
            logger.info(f'[ScaffoldGrow] Post-diffusion: extra atom mean drift from init={_drift:.2f} A, '
                        f'expected ~0 with anchor=1.0')

        # 还原坐标偏移
        final_pos = ligand_pos_out + offset[batch_ligand]
        pos_np = final_pos.detach().cpu().numpy().astype(np.float64)
        v_np = log_v_out.argmax(dim=-1).detach().cpu().numpy().astype(np.int64)

        # 骨架路径只把 H 节点转换为 C；F/Cl 是可表达的 target-side 元素，
        # 不能为了提高重建率而静默丢失。
        # add_aromatic 模式索引: 0=H, 1=C, 2=C_arom, 3=N, 4=N_arom,
        #   5=O, 6=O_arom, 7=F, 8=P, 9=P_arom, 10=S, 11=S_arom, 12=Cl
        if n_extra > 0:
            extra_v = v_np[n_locked:]
            to_carbon = (extra_v == 0)   # H: 单价，永远不是重原子
            if to_carbon.any():
                extra_v[to_carbon] = 1  # → 非芳香 C
                v_np[n_locked:] = extra_v

        # 骨架 RMSD（仅对骨架原子，验证锚定效果）
        if n_scaffold > 0:
            scaffold_pos_ref_world = scaffold_pos_orig.detach().cpu().numpy()
            scaffold_pos_gen = pos_np[:n_scaffold]
            scaffold_rmsd = float(np.sqrt(np.mean((scaffold_pos_gen - scaffold_pos_ref_world) ** 2)))
        else:
            scaffold_rmsd = 0.0

        metric_info = evaluate_candidate(pos_np, v_np, ligand_atom_mode, selector_cfg)

        pos_list.append(pos_np)
        v_list.append(v_np)
        pos_traj_list.append([])
        v_traj_list.append([])
        log_v_traj_list.append([])
        time_list.append(t_end_wall - t_start_wall)
        meta_records.append({
            'method': 'scaffold_grow',
            'ligand_num_atoms': n_total,
            'n_scaffold': n_scaffold,
            'n_preserved': n_preserved,
            'n_extra': n_extra,
            'n_locked': n_locked,
            'zero_allocation_site_ids': zero_site_ids,
            'preserved_atom_indices': preserved_indices,
            'scaffold_rmsd': scaffold_rmsd,
            'rmsd_from_ref': float(np.sqrt(np.mean(
                (pos_np[:n_scaffold_ref] - ref_pos_np) ** 2
            ))) if n_scaffold_ref == n_total else float('nan'),
            'time': t_end_wall - t_start_wall,
            'smiles': metric_info.get('smiles'),
            'qed': metric_info.get('metrics', {}).get('qed'),
            'sa': metric_info.get('metrics', {}).get('sa'),
            'status': metric_info.get('status'),
            'is_original': False,
            'initial_extra_positions': extra_pos_init.detach().cpu().tolist(),
            'extra_placement': _site_place_meta.get('placement'),
            'site_allocation': _site_place_meta.get('site_allocation'),
        })

    # ---- 主 result_*.pt：首位插入提取后的骨架（坐标/类型子集，便于与生成样本对照分析）----
    # 骨架拓扑另存 SDF 见上文 outputs/scaffold/*.sdf；不再单独写骨架 .pt。
    if n_scaffold > 0:
        scaffold_pos_np = ref_pos_np[scaffold_indices]
        scaffold_v_np = ref_v_np[scaffold_indices]
        scaffold_metric = evaluate_candidate(scaffold_pos_np, scaffold_v_np, ligand_atom_mode, selector_cfg)
        pos_list.insert(0, scaffold_pos_np)
        v_list.insert(0, scaffold_v_np)
        pos_traj_list.insert(0, [])
        v_traj_list.insert(0, [])
        log_v_traj_list.insert(0, [])
        time_list.insert(0, 0.0)
        meta_records.insert(0, {
            'method': 'scaffold_grow',
            'ligand_num_atoms': n_scaffold,
            'n_scaffold': n_scaffold,
            'n_extra': 0,
            'scaffold_rmsd': 0.0,
            'rmsd_from_ref': 0.0,
            'time': 0.0,
            'smiles': scaffold_metric.get('smiles'),
            'qed': scaffold_metric.get('metrics', {}).get('qed'),
            'sa': scaffold_metric.get('metrics', {}).get('sa'),
            'status': scaffold_metric.get('status'),
            'is_original': True,
            'is_extracted_scaffold': True,
        })

    if logger:
        valid = sum(1 for m in meta_records if m.get('status') == 'ok' and not m.get('is_original'))
        avg_scaffold_rmsd = np.mean([
            m['scaffold_rmsd'] for m in meta_records
            if not m.get('is_original') and np.isfinite(m.get('scaffold_rmsd', float('nan')))
        ]) if valid > 0 else 0.0
        logger.info(
            f'[ScaffoldGrow] 完成: {valid}/{num_samples} 有效分子 | '
            f'平均骨架 RMSD={avg_scaffold_rmsd:.3f} Å | 总耗时 {total_time:.2f}s'
        )

    return {
        'pos_list': pos_list,
        'v_list': v_list,
        'pos_traj': pos_traj_list,
        'v_traj': v_traj_list,
        'log_v_traj': log_v_traj_list,
        'time_list': time_list,
        'meta': {
            'method': 'scaffold_grow',
            'records': meta_records,
            'scaffold_cfg': {
                'scaffold_source': grow_cfg.get('scaffold_source'),
                'n_scaffold_atoms': n_scaffold,
                'scaffold_indices': scaffold_indices,
                'n_extra_mode': grow_cfg.get('n_extra_mode'),
                'start_t': start_t,
                'attachment_sites': _attachment_sites,
                'murcko_sites_cfg': _murcko_sites_cfg,
            },
            'total_time': total_time,
        },
    }
