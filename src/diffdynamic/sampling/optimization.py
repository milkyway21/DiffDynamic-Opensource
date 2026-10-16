"""Sampling optimization responsibilities for ligand sampling."""

from __future__ import annotations

from diffdynamic.data.graph import FOLLOW_BATCH
from diffdynamic.data.graph import ProteinLigandData
from diffdynamic.data.graph import torchify_dict
from diffdynamic.models.probability import center_pos
from diffdynamic.models.probability import extract
from diffdynamic.models.probability import index_to_log_onehot
from torch_geometric.data import Batch
from diffdynamic.data.protein import rdmol_to_ligand_dict
from diffdynamic.io.monitor import GPUMonitor
from diffdynamic.io.monitor import MemoryProfiler
import numpy as np
import time
import torch
import diffdynamic.chemistry.reconstruct as reconstruct
import diffdynamic.data.transforms as trans


def _create_data_from_generated_mol(orig_data, pos, v, ligand_atom_mode, transform, logger=None):
    """从 dynamic 生成的 (pos, v) 创建 ProteinLigandData，用于后续优化。
    若重建失败则返回 None。
    """
    pos = np.asarray(pos, dtype=np.float64)
    v = np.asarray(v, dtype=np.int64)
    if pos.size == 0 or v.size == 0:
        return None
    v_tensor = torch.tensor(v, dtype=torch.long)
    atom_numbers = trans.get_atomic_number_from_index(v_tensor, mode=ligand_atom_mode)
    aromatic_flags = trans.is_aromatic_from_index(v_tensor, mode=ligand_atom_mode)
    basic_mode = (ligand_atom_mode == 'basic')
    aromatic = None if aromatic_flags is None else (aromatic_flags.tolist() if hasattr(aromatic_flags, 'tolist') else list(aromatic_flags))
    try:
        mol = reconstruct.reconstruct_from_generated(
            pos.tolist(), atom_numbers.tolist() if hasattr(atom_numbers, 'tolist') else list(atom_numbers),
            aromatic=aromatic,
            basic_mode=(aromatic is None)
        )
    except reconstruct.MolReconsError:
        return None
    if mol is None:
        return None
    ligand_dict = rdmol_to_ligand_dict(mol, pos_override=pos.astype(np.float32))
    if ligand_dict is None:
        return None
    def _to_numpy(x):
        if hasattr(x, 'cpu') and hasattr(x, 'numpy'):
            return x.cpu().numpy()
        return np.array(x) if not isinstance(x, np.ndarray) else x
    protein_dict = {
        'pos': _to_numpy(orig_data.protein_pos),
        'element': _to_numpy(orig_data.protein_element),
    }
    for key in ['molecule_name', 'is_backbone', 'atom_name', 'atom_to_aa_type', 'atom_feature']:
        attr = 'protein_' + key
        if hasattr(orig_data, attr):
            v = getattr(orig_data, attr)
            protein_dict[key] = _to_numpy(v) if isinstance(v, (np.ndarray, torch.Tensor)) else v
    data_new = ProteinLigandData.from_protein_ligand_dicts(
        protein_dict=torchify_dict(protein_dict),
        ligand_dict=torchify_dict(ligand_dict),
    )
    data_new.protein_filename = getattr(orig_data, 'protein_filename', None)
    data_new.ligand_filename = getattr(orig_data, 'ligand_filename', 'N/A')
    if transform is not None:
        data_new = transform(data_new)
    return data_new


def _run_optimization_with_cycle_filter(model, data, ref_ligand_pos, log_ref_v, ref_pos_np, ref_v_np, ref_num_atoms,
                                        opt_cfg, batch_size, num_samples, cycles, start_t, stride, step_size, noise_scale, schedule,
                                        center_pos_mode, pos_only, ligand_atom_mode, device, keep_original, logger, profiler, monitor,
                                        max_survivors_override=None):
    """每 cycle 从当前分子平行生成 num_samples 个变体，按 QED/SA 筛选，不满足的下一轮前丢弃。"""
    from diffdynamic.sampling.candidates import evaluate_candidate

    pos_list, v_list = [], []
    pos_traj_list, v_traj_list, log_v_traj_list = [], [], []
    time_list = []
    meta_records = []
    total_opt_time = 0.0

    if schedule == 'lambda':
        lambda_a = opt_cfg.get('lambda_coeff_a', 40.0)
        lambda_b = opt_cfg.get('lambda_coeff_b', 5.0)
        time_indices = model._build_lambda_schedule(start_t=start_t, end_t=0, coeff_a=lambda_a, coeff_b=lambda_b)
    elif schedule == 'linear':
        total_steps = start_t // max(stride, 1)
        total_steps = max(total_steps, 10)
        time_indices = np.linspace(start_t, 0, total_steps + 1, dtype=int).tolist()
        time_indices = sorted(set(time_indices), reverse=True)
    else:
        time_indices = list(range(start_t, -1, -max(stride, 1)))
    if time_indices and time_indices[-1] != 0:
        time_indices.append(0)

    use_with_noise = opt_cfg.get('use_with_noise', True)
    use_adaptive_step = opt_cfg.get('use_adaptive_step', True)
    use_time_scale = opt_cfg.get('use_time_scale', False)
    selector_cfg = opt_cfg.get('selector', {})

    # current_mols: list of (pos_tensor, log_v_tensor) 用于下一轮
    ref_pos_t = torch.tensor(ref_pos_np, dtype=torch.float32, device=device)
    ref_log_v_t = torch.tensor(
        np.eye(model.num_classes, dtype=np.float32)[ref_v_np],
        device=device
    )
    ref_log_v_t = torch.log(ref_log_v_t.clamp(min=1e-8))
    current_mols = [(ref_pos_t, ref_log_v_t)]

    n_cycles = max(cycles, 1)
    for cycle_idx in range(n_cycles):
        is_final_cycle = (cycle_idx == n_cycles - 1)
        # 前几轮可容忍不完整分子进入下一轮；最后一轮必须开启不完整筛选
        selector_cfg_this_cycle = {**selector_cfg, 'filter_incomplete': selector_cfg.get('filter_incomplete', True) if is_final_cycle else False}
        if logger:
            logger.info(f'[Optimization] Cycle {cycle_idx + 1}/{n_cycles} | 当前分子数: {len(current_mols)} | 不完整筛选: {"开" if selector_cfg_this_cycle.get("filter_incomplete") else "关"}')
        all_candidates = []
        for mol_idx, (pos_t, log_v_t) in enumerate(current_mols):
            n_atoms = pos_t.size(0)
            data_list = [data.clone() for _ in range(num_samples)]
            batch = Batch.from_data_list(data_list, follow_batch=FOLLOW_BATCH).to(device)
            batch_protein = batch.protein_element_batch
            batch_ligand = batch.ligand_element_batch

            pos_centered, ref_centered, offset = center_pos(
                batch.protein_pos, pos_t.unsqueeze(0).expand(num_samples, -1, -1).reshape(-1, 3),
                batch_protein, batch_ligand, mode=center_pos_mode
            )
            log_v_batch = log_v_t.unsqueeze(0).expand(num_samples, -1, -1).reshape(-1, model.num_classes)

            repaint_cfg, guidance_cfg = _build_repaint_guidance_cfg(
                opt_cfg, ref_centered, log_v_batch, batch_ligand, device
            )
            with torch.no_grad():
                noised_pos, noised_log_v = _forward_diffuse_molecule(
                    model, ref_centered, log_v_batch, batch_ligand, start_t, device
                )
                ligand_pos_cur, log_ligand_v_cur, pos_traj, log_v_traj = model._dynamic_diffusion(
                    protein_pos=batch.protein_pos,
                    protein_v=batch.protein_atom_feature.float(),
                    batch_protein=batch_protein,
                    ligand_pos=noised_pos,
                    log_ligand_v=noised_log_v,
                    batch_ligand=batch_ligand,
                    time_indices=time_indices,
                    step_size=step_size,
                    add_noise=noise_scale,
                    pos_clip=opt_cfg.get('pos_clip', None),
                    v_clip=opt_cfg.get('v_clip', None),
                    record_traj=True,
                    pos_only=pos_only,
                    use_with_noise=use_with_noise,
                    use_adaptive_step=use_adaptive_step,
                    use_time_scale=use_time_scale,
                    repaint_cfg=repaint_cfg,
                    guidance_cfg=guidance_cfg,
                )
            pos_final = ligand_pos_cur + offset[batch_ligand]
            offset_for_traj = offset[batch_ligand]
            pos_traj = [p + offset_for_traj.to(p.device) for p in pos_traj] if pos_traj else []

            for si in range(num_samples):
                mask = (batch_ligand == si)
                pos_np = pos_final[mask].detach().cpu().numpy().astype(np.float64)
                v_np = log_ligand_v_cur[mask].argmax(dim=-1).detach().cpu().numpy().astype(np.int64)
                pos_traj_np = [p[mask].detach().cpu().numpy().astype(np.float64) for p in pos_traj]
                v_traj_np = [np.argmax(lv[mask].detach().cpu().numpy(), axis=-1).astype(np.int64) for lv in log_v_traj]

                metric_info = evaluate_candidate(pos_np, v_np, ligand_atom_mode, selector_cfg_this_cycle)
                rmsd = np.sqrt(np.mean((pos_np - ref_pos_np) ** 2))
                cand = {
                    'pos': pos_np, 'v': v_np, 'pos_traj': pos_traj_np, 'v_traj': v_traj_np,
                    'log_v': log_ligand_v_cur[mask].detach().cpu().numpy().astype(np.float32),
                    'metric_info': metric_info, 'rmsd': rmsd, 'num_atoms': ref_num_atoms
                }
                all_candidates.append(cand)

        survivors = [c for c in all_candidates if c['metric_info'].get('status') == 'ok']
        if not survivors:
            # 严格按 selector 筛选：全部未过筛时不保留任何，避免输出不满足 min_qed/min_sa 的分子
            survivors = []
            if logger and all_candidates:
                logger.info(f'[Optimization] Cycle {cycle_idx + 1} 全部未过筛（QED/SA/完整性），不保留任何')
        elif logger:
            logger.info(f'[Optimization] Cycle {cycle_idx + 1} 筛选: {len(survivors)}/{len(all_candidates)} 通过')

        # 每轮最多保留 K 个：按 score 排序（越小越好），只保留前 K 个进入下一轮
        max_survivors = max_survivors_override if max_survivors_override is not None else selector_cfg.get('max_survivors_per_cycle')
        if max_survivors is not None and len(survivors) > max_survivors:
            survivors = sorted(survivors, key=lambda x: x['metric_info'].get('score', float('inf')))[:max_survivors]
            if logger:
                logger.info(f'[Optimization] Cycle {cycle_idx + 1} 按 score 截断至前 {max_survivors} 个')

        current_mols = []
        for c in survivors:
            pos_t = torch.tensor(c['pos'], dtype=torch.float32, device=device)
            log_v = np.eye(model.num_classes, dtype=np.float32)[c['v']]
            log_v = np.log(np.clip(log_v, 1e-8, 1.0))
            log_v_t = torch.tensor(log_v, dtype=torch.float32, device=device)
            current_mols.append((pos_t, log_v_t))

    # 输出最后一轮的 survivors
    for c in survivors:
        pos_list.append(c['pos'])
        v_list.append(c['v'])
        pos_traj_list.append(c['pos_traj'])
        v_traj_list.append(c['v_traj'])
        log_v_traj_list.append([np.eye(model.num_classes)[vi].astype(np.float32) for vi in c['v_traj']])
        time_list.append(0.0)
        mi = c['metric_info']
        meta_records.append({
            'method': 'optimization',
            'ligand_num_atoms': c['num_atoms'],
            'time': 0.0,
            'rmsd_from_ref': c['rmsd'],
            'smiles': mi.get('smiles'),
            'qed': mi.get('metrics', {}).get('qed'),
            'sa': mi.get('metrics', {}).get('sa'),
            'status': mi.get('status'),
            'is_original': False,
        })

    if keep_original:
        pos_list.insert(0, ref_pos_np)
        v_list.insert(0, ref_v_np)
        pos_traj_list.insert(0, [])
        v_traj_list.insert(0, [])
        log_v_traj_list.insert(0, [])
        time_list.insert(0, 0.0)
        metric_info_ref = evaluate_candidate(ref_pos_np, ref_v_np, ligand_atom_mode, selector_cfg)
        meta_records.insert(0, {
            'method': 'optimization',
            'ligand_num_atoms': ref_num_atoms,
            'time': 0.0,
            'rmsd_from_ref': 0.0,
            'smiles': metric_info_ref.get('smiles'),
            'qed': metric_info_ref.get('metrics', {}).get('qed'),
            'sa': metric_info_ref.get('metrics', {}).get('sa'),
            'status': metric_info_ref.get('status'),
            'is_original': True,
        })

    return total_opt_time, pos_list, v_list, pos_traj_list, v_traj_list, log_v_traj_list, time_list, meta_records, time_indices


def _build_repaint_guidance_cfg(opt_cfg, ref_pos_centered, log_ref_v_batch, batch_ligand, device):
    """根据 optimization.mask_repaint / optimization.property_guidance 构建 _dynamic_diffusion 的扩展参数。

    在代码中设置（YAML 无法直接传张量）：
    - opt_cfg['mask_repaint'] = dict(enable=True, atom_mask=..., fixed_eps_pos=可选)
    - opt_cfg['property_guidance'] = dict(enable=True, loss_fn=可调用对象, scale=..., t_min/t_max=...)
    """
    repaint_cfg = None
    guidance_cfg = None
    mr = opt_cfg.get('mask_repaint') or {}
    if mr.get('enable'):
        atom_mask = mr.get('atom_mask')
        if atom_mask is None:
            raise ValueError('mask_repaint.enable=true 时必须提供 atom_mask，形状与 ref_pos_centered 原子维一致 [N] 或 [N,1]')
        atom_mask = torch.as_tensor(atom_mask, dtype=torch.float32, device=device)
        if atom_mask.dim() == 1:
            atom_mask = atom_mask.view(-1, 1)
        if atom_mask.shape[0] != ref_pos_centered.shape[0]:
            raise ValueError(f'atom_mask 原子数 {atom_mask.shape[0]} 与 ref_pos_centered {ref_pos_centered.shape[0]} 不一致')
        repaint_cfg = {
            'x0_pos': ref_pos_centered,
            'x0_log_v': log_ref_v_batch,
            'atom_mask': atom_mask,
            'fixed_eps_pos': mr.get('fixed_eps_pos'),
            'use_mean_for_discrete': mr.get('use_mean_for_discrete', True),
        }
    pg = opt_cfg.get('property_guidance') or {}
    if pg.get('enable'):
        fn = pg.get('loss_fn')
        if fn is None or not callable(fn):
            raise ValueError('property_guidance.enable=true 时必须提供可微标量损失 loss_fn(pos, log_v, batch_ligand)')
        am = pg.get('atom_mask')
        if am is not None:
            am = torch.as_tensor(am, dtype=torch.float32, device=device)
        guidance_cfg = {
            'fn': fn,
            'scale': float(pg.get('scale', 0.1)),
            't_min': int(pg.get('t_min', 0)),
            't_max': int(pg.get('t_max', 10**9)),
            'atom_mask': am,
            'apply_to_types': pg.get('apply_to_types', True),
        }
    return repaint_cfg, guidance_cfg


def _forward_diffuse_molecule(model, ligand_pos, log_ligand_v, batch_ligand, t_start, device):
    """对现有分子施加前向扩散噪声到时间步 t_start。

    基于 SDEdit 原理：x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
    对于原子类型，使用 q(v_t | v_0) 的分类扩散过程。

    Args:
        model: 扩散模型实例。
        ligand_pos: 配体原始坐标 [N, 3]。
        log_ligand_v: 配体原子类型的 log one-hot [N, num_classes]。
        batch_ligand: 批次索引 [N]。
        t_start: 目标噪声时间步。
        device: 计算设备。

    Returns:
        tuple: (noised_pos, noised_log_v) 加噪后的坐标和类型。
    """
    num_graphs = batch_ligand.max().item() + 1
    t = torch.full((num_graphs,), t_start, dtype=torch.long, device=device)

    sqrt_alpha_bar = extract(model.sqrt_alphas_cumprod, t, batch_ligand)
    sqrt_one_minus_alpha_bar = extract(model.sqrt_one_minus_alphas_cumprod, t, batch_ligand)
    epsilon = torch.randn_like(ligand_pos)
    noised_pos = sqrt_alpha_bar * ligand_pos + sqrt_one_minus_alpha_bar * epsilon

    _, noised_log_v = model.q_v_sample(log_ligand_v, t, batch_ligand)

    return noised_pos, noised_log_v


def optimize_molecule(model, data, config, ligand_atom_mode, device='cuda:0', logger=None, opt_batch_size=None, n_dynamic_molecules=None):
    """通过部分扩散-去噪过程优化现有分子结构。

    核心思路（SDEdit / img2img 在分子领域的类比）：
    1. 将输入分子的坐标和原子类型前向扩散到中间时间步 t_start（添加受控噪声）
    2. 利用蛋白质口袋条件引导的反向去噪过程将分子"修复"到 t=0
    3. 噪声水平控制探索-利用权衡：
       - t_start 越高 → 噪声越大 → 结构变化越激进
       - t_start 越低 → 噪声越小 → 更保守地保留原始结构

    Args:
        model: 扩散模型实例（ScorePosNet3D 或 DiffDynamic）。
        data: ProteinLigandData，包含蛋白口袋和参考配体。
        config: 采样配置对象。
        ligand_atom_mode: 配体原子编码模式。
        device: 推理设备。
        logger: 可选日志器。
        opt_batch_size: 优化批次数。若提供则覆盖 large_step.batch_size。
           - dynamic_then_optimization 模式下应传 1：每个 dynamic 分子生成 num_samples 个优化变体。
           - optimization 单独模式下不传：使用 large_step.batch_size，从单参考生成多批变体。

    Returns:
        dict: 包含优化后的分子坐标、类型、轨迹、元信息的结果字典。
    """
    from diffdynamic.sampling.candidates import evaluate_candidate

    monitor = GPUMonitor(device=device, enable_flops=False)
    profiler = MemoryProfiler(device=device)

    opt_cfg = config.sample.get('optimization', {})
    dynamic_cfg = config.sample.get('dynamic', {})
    large_step_cfg = dynamic_cfg.get('large_step', {})
    batch_size = max(1, int(opt_batch_size)) if opt_batch_size is not None else max(1, int(large_step_cfg.get('batch_size', 1)))
    num_samples = opt_cfg.get('num_samples', 10)
    keep_original = opt_cfg.get('keep_original', False)
    start_t = opt_cfg.get('start_t', 300)
    stride = opt_cfg.get('stride', 2)
    step_size = opt_cfg.get('step_size', 0.5)
    noise_scale = opt_cfg.get('noise_scale', 0.05)
    schedule = opt_cfg.get('schedule', 'linear')
    cycles = opt_cfg.get('cycles', 1)
    center_pos_mode = config.sample.get('center_pos_mode', 'protein')
    pos_only = config.sample.get('pos_only', False)

    start_t = int(np.clip(start_t, 1, model.num_timesteps - 1))

    if not hasattr(data, 'ligand_pos') or data.ligand_pos is None or data.ligand_pos.size(0) == 0:
        raise ValueError(
            '分子优化模式需要提供参考配体（包含 3D 坐标的 SDF 文件），'
            '用于作为优化起点。请通过 --ligand_path 指定。'
        )

    ref_ligand_pos = data.ligand_pos.to(device)
    ref_ligand_v = data.ligand_atom_feature_full.to(device)
    ref_num_atoms = ref_ligand_pos.size(0)

    total_opt = batch_size * num_samples
    expected_total = total_opt + (1 if keep_original else 0)
    if logger:
        logger.info(
            f'[Optimization] 输入分子: {ref_num_atoms} 原子 | '
            f'起始时间步 t={start_t}/{model.num_timesteps} | '
            f'dynamic.large_step.batch_size={batch_size} × num_samples={num_samples} = {total_opt} 个优化变体 | 保留原分子={keep_original} | 步幅={stride} | 步长={step_size}'
        )
        logger.info(
            f'[Optimization] 每个口袋将输出 {expected_total} 个分子 '
            f'({"1原分子 + " + str(total_opt) + "优化变体" if keep_original else str(total_opt) + "个优化变体"})'
        )

    log_ref_v = index_to_log_onehot(ref_ligand_v, model.num_classes)

    pos_list, v_list = [], []
    pos_traj_list, v_traj_list, log_v_traj_list = [], [], []
    time_list = []
    meta_records = []

    profiler.checkpoint('before_optimization')

    ref_pos_np = ref_ligand_pos.detach().cpu().numpy().astype(np.float64)
    ref_v_np = np.argmax(log_ref_v.detach().cpu().numpy(), axis=-1).astype(np.int64)

    selector_cfg = opt_cfg.get('selector', {})
    enable_filter = selector_cfg.get('enable_filter', False)
    has_filter_threshold = (selector_cfg.get('min_qed') is not None or selector_cfg.get('min_sa') is not None)
    use_cycle_filter = enable_filter and has_filter_threshold and cycles >= 1

    # dynamic_then_optimization 时，max_survivors_total 限制优化分子总数，每口袋分配 max_survivors_total // n_dyn
    max_survivors_override = None
    if n_dynamic_molecules is not None and n_dynamic_molecules > 0:
        max_survivors_total = selector_cfg.get('max_survivors_total')
        if max_survivors_total is not None:
            max_survivors_override = max(1, max_survivors_total // n_dynamic_molecules)
            if logger:
                logger.info(f'[Optimization] max_survivors_total={max_survivors_total}，每口袋最多 {max_survivors_override} 个优化分子')

    opt_time_indices = None
    if use_cycle_filter:
        # 新模式：每 cycle 平行生成 num_samples 个，按 QED/SA 筛选，不满足的下一轮前丢弃
        total_opt_time, pos_list, v_list, pos_traj_list, v_traj_list, log_v_traj_list, time_list, meta_records, opt_time_indices = _run_optimization_with_cycle_filter(
            model, data, ref_ligand_pos, log_ref_v, ref_pos_np, ref_v_np, ref_num_atoms,
            opt_cfg, batch_size, num_samples, cycles, start_t, stride, step_size, noise_scale, schedule,
            center_pos_mode, pos_only, ligand_atom_mode, device, keep_original, logger, profiler, monitor,
            max_survivors_override=max_survivors_override
        )
    else:
        total_opt_time = 0.0
        # 原逻辑：外层循环 batch_size 批，每批并行生成 num_samples 个
        for batch_idx in range(batch_size):
            if logger and batch_size > 1:
                logger.info(f'[Optimization] 批次 {batch_idx + 1}/{batch_size}')

            # 构建批次：将 num_samples 个样本打包为一批，实现并行优化
            data_list = [data.clone() for _ in range(num_samples)]
            batch = Batch.from_data_list(data_list, follow_batch=FOLLOW_BATCH).to(device)
            batch_protein = batch.protein_element_batch
            batch_ligand = batch.ligand_element_batch

            protein_pos_centered, ref_pos_centered, offset = \
                center_pos(batch.protein_pos, batch.ligand_pos.clone(), batch_protein, batch_ligand, mode=center_pos_mode)

            # 将 log_ref_v 复制 num_samples 份以匹配批次
            log_ref_v_batch = log_ref_v.unsqueeze(0).expand(num_samples, -1, -1).reshape(-1, model.num_classes)

            with torch.no_grad():
                noised_pos, noised_log_v = _forward_diffuse_molecule(
                    model, ref_pos_centered, log_ref_v_batch, batch_ligand, start_t, device
                )

            if logger:
                pos_shift = (noised_pos - ref_pos_centered).norm(dim=-1).mean().item()
                logger.info(f'[Optimization] 前向扩散完成（{num_samples} 个样本并行）| 平均位移={pos_shift:.3f} Å')

            if schedule == 'lambda':
                lambda_a = opt_cfg.get('lambda_coeff_a', 40.0)
                lambda_b = opt_cfg.get('lambda_coeff_b', 5.0)
                time_indices = model._build_lambda_schedule(
                    start_t=start_t, end_t=0,
                    coeff_a=lambda_a, coeff_b=lambda_b
                )
            elif schedule == 'linear':
                total_steps = start_t // max(stride, 1)
                total_steps = max(total_steps, 10)
                time_indices = np.linspace(start_t, 0, total_steps + 1, dtype=int).tolist()
                time_indices = sorted(set(time_indices), reverse=True)
            else:
                time_indices = list(range(start_t, -1, -max(stride, 1)))

            if time_indices and time_indices[-1] != 0:
                time_indices.append(0)
            if opt_time_indices is None:
                opt_time_indices = time_indices

            profiler.checkpoint(f'opt_batch_{batch_idx}_before')
            monitor.reset_peak_stats()
            t_start_time = time.time()

            ligand_pos_current = noised_pos
            log_ligand_v_current = noised_log_v
            pos_traj_cycle, log_v_traj_cycle = [], []

            use_with_noise = opt_cfg.get('use_with_noise', True)
            use_adaptive_step = opt_cfg.get('use_adaptive_step', True)
            use_time_scale = opt_cfg.get('use_time_scale', False)

            repaint_cfg, guidance_cfg = _build_repaint_guidance_cfg(
                opt_cfg, ref_pos_centered, log_ref_v_batch, batch_ligand, device
            )
            with torch.no_grad():
                for _ in range(max(cycles, 1)):
                    ligand_pos_current, log_ligand_v_current, pos_traj, log_v_traj = model._dynamic_diffusion(
                        protein_pos=protein_pos_centered,
                        protein_v=batch.protein_atom_feature.float(),
                        batch_protein=batch_protein,
                        ligand_pos=ligand_pos_current,
                        log_ligand_v=log_ligand_v_current,
                        batch_ligand=batch_ligand,
                        time_indices=time_indices,
                        step_size=step_size,
                        add_noise=noise_scale,
                        pos_clip=opt_cfg.get('pos_clip', None),
                        v_clip=opt_cfg.get('v_clip', None),
                        record_traj=True,
                        pos_only=pos_only,
                        use_with_noise=use_with_noise,
                        use_adaptive_step=use_adaptive_step,
                        use_time_scale=use_time_scale,
                        repaint_cfg=repaint_cfg,
                        guidance_cfg=guidance_cfg,
                    )
                    pos_traj_cycle.extend(pos_traj)
                    log_v_traj_cycle.extend(log_v_traj)

            ligand_pos_final = ligand_pos_current + offset[batch_ligand]
            if pos_traj_cycle:
                offset_for_traj = offset[batch_ligand]
                pos_traj_cycle = [p + offset_for_traj.to(p.device) for p in pos_traj_cycle]

            t_end_time = time.time()
            total_opt_time += t_end_time - t_start_time

            # 按 batch_ligand 拆分批次结果
            for sample_idx in range(num_samples):
                mask = (batch_ligand == sample_idx)
                pos_np = ligand_pos_final[mask].detach().cpu().numpy().astype(np.float64)
                log_v_np = log_ligand_v_current[mask].detach().cpu().numpy().astype(np.float32)
                v_np = np.argmax(log_v_np, axis=-1).astype(np.int64)

                pos_traj_np = [p[mask].detach().cpu().numpy().astype(np.float64) for p in pos_traj_cycle]
                log_v_traj_np = [lv[mask].detach().cpu().numpy().astype(np.float32) for lv in log_v_traj_cycle]
                v_traj_np = [np.argmax(lv, axis=-1).astype(np.int64) for lv in log_v_traj_np]

                pos_list.append(pos_np)
                v_list.append(v_np)
                pos_traj_list.append(pos_traj_np)
                v_traj_list.append(v_traj_np)
                log_v_traj_list.append(log_v_traj_np)
                time_list.append((t_end_time - t_start_time) / num_samples)  # 平均耗时

                rmsd_from_ref = np.sqrt(np.mean(
                    (pos_np - ref_pos_np) ** 2
                ))

                metric_info = evaluate_candidate(pos_np, v_np, ligand_atom_mode,
                                                 opt_cfg.get('selector', {}))
                meta_records.append({
                    'method': 'optimization',
                    'ligand_num_atoms': ref_num_atoms,
                    'time': (t_end_time - t_start_time) / num_samples,
                    'rmsd_from_ref': rmsd_from_ref,
                    'smiles': metric_info.get('smiles'),
                    'qed': metric_info.get('metrics', {}).get('qed'),
                    'sa': metric_info.get('metrics', {}).get('sa'),
                    'status': metric_info.get('status'),
                    'is_original': False,
                })

            profiler.checkpoint(f'opt_batch_{batch_idx}_after')

    if keep_original:
        # 将原分子插入到输出首位
        pos_list.insert(0, ref_pos_np)
        v_list.insert(0, ref_v_np)
        pos_traj_list.insert(0, [])  # 原分子无轨迹
        v_traj_list.insert(0, [])
        log_v_traj_list.insert(0, [])
        time_list.insert(0, 0.0)
        metric_info_ref = evaluate_candidate(ref_pos_np, ref_v_np, ligand_atom_mode,
                                             opt_cfg.get('selector', {}))
        meta_records.insert(0, {
            'method': 'optimization',
            'ligand_num_atoms': ref_num_atoms,
            'time': 0.0,
            'rmsd_from_ref': 0.0,
            'smiles': metric_info_ref.get('smiles'),
            'qed': metric_info_ref.get('metrics', {}).get('qed'),
            'sa': metric_info_ref.get('metrics', {}).get('sa'),
            'status': metric_info_ref.get('status'),
            'is_original': True,
        })

    if logger:
        mem_info = monitor.get_memory_info()
        logger.info(
            f'[Optimization] 并行完成: {total_opt} 个优化变体 (large_step.batch_size×num_samples={batch_size}×{num_samples}) | '
            f'总耗时 {total_opt_time:.2f}s | mem {mem_info["allocated"]:.1f}/{mem_info["max_allocated"]:.1f} MB'
        )
        for sample_idx, m in enumerate(meta_records):
            suffix = ' (原分子)' if m.get('is_original') else ''
            logger.info(
                f'  [{sample_idx}] {ref_num_atoms} atoms | '
                f'RMSD={m["rmsd_from_ref"]:.3f} Å | '
                f'QED={m.get("qed", "N/A")} | SA={m.get("sa", "N/A")}{suffix}'
            )

    if logger:
        summary = profiler.get_summary()
        logger.info(f'[Optimization Memory Summary] Peak: {summary["peak_memory_mb"]:.1f} MB')
        opt_records = [m for m in meta_records if not m.get('is_original')]
        valid_count = sum(1 for m in opt_records if m['status'] == 'ok')
        total_count = len(pos_list)
        rmsd_list = [m["rmsd_from_ref"] for m in opt_records if "rmsd_from_ref" in m]
        avg_rmsd = np.mean(rmsd_list) if rmsd_list else 0.0
        logger.info(
            f'[Optimization] 完成: 共 {total_count} 个分子 (原分子×1 + 优化变体×{total_opt}) | '
            f'有效优化分子: {valid_count}/{total_opt} | '
            f'平均 RMSD: {avg_rmsd:.3f} Å'
        )

    opt_meta = {
        'method': 'optimization',
        'records': meta_records,
        'memory_summary': profiler.get_summary(),
        'optimization_config': {
            'start_t': start_t,
            'stride': stride,
            'step_size': step_size,
            'noise_scale': noise_scale,
            'schedule': schedule,
            'batch_size': batch_size,  # 来自 dynamic.large_step.batch_size
            'num_samples': num_samples,
            'total_opt': total_opt,
            'keep_original': keep_original,
            'cycles': cycles,
            'ref_num_atoms': ref_num_atoms,
        }
    }
    if opt_time_indices is not None:
        opt_meta['optimization_time_indices'] = opt_time_indices
    return {
        'pos_list': pos_list,
        'v_list': v_list,
        'pos_traj': pos_traj_list,
        'v_traj': v_traj_list,
        'log_v_traj': log_v_traj_list,
        'time_list': time_list,
        'meta': opt_meta
    }
