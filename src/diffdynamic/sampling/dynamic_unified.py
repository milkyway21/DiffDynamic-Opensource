"""Sampling dynamic_unified responsibilities for ligand sampling."""

from __future__ import annotations

from diffdynamic.data.graph import FOLLOW_BATCH
from torch_geometric.data import Batch
from torch_scatter import scatter_mean
from diffdynamic.evaluation import atom_num
from diffdynamic.io.gpu_monitor_recorder import log_gpu_monitor_record
from diffdynamic.io.monitor import GPUMonitor
from diffdynamic.io.monitor import MemoryProfiler
import numpy as np
import time
import torch
import torch.nn.functional as F


def _run_unified_dynamic(model, data, config, device='cuda:0', logger=None, skip_targetdiff_baseline_refine=False):  # 统一动态采样流程。
    """按照统一动态策略执行扩散采样并收集轨迹。

    Args:
        model: 已加载权重的扩散模型，需实现 `dynamic_sample_diffusion`。
        data: 单个 `ProteinLigandData` 样本，作为蛋白口袋输入。
        config: 采样配置（通常来自 YAML），读取 `sample` 相关字段。
        device: 推理使用的设备字符串。
        logger: 可选日志记录器，用于输出采样进度。

    Returns:
        dict: 包含最终坐标/类型、完整轨迹、耗时以及元信息的字典。
    """
    from diffdynamic.sampling.refinement import apply_targetdiff_baseline_refinement

    # 创建GPU监控器
    monitor = GPUMonitor(device=device, enable_flops=False)
    profiler = MemoryProfiler(device=device)
    
    dynamic_cfg = config.sample.get('dynamic', {})  # 读取动态采样配置。
    num_samples = config.sample.get('num_samples', 1)  # 生成样本数量。
    num_steps = config.sample.get('num_steps', 1000)  # 扩散步数。
    center_pos_mode = config.sample.get('center_pos_mode', 'protein')  # 坐标中心化模式。
    pos_only = config.sample.get('pos_only', False)  # 是否仅采样坐标。
    sample_num_atoms_mode = config.sample.get('sample_num_atoms', 'prior')  # 原子数量策略。

    # 注意：配置更新已在 sample_dynamic_diffusion_ligand 中完成，这里不需要重复更新

    pos_list, v_list = [], []  # 存储最终位置与类别。
    pos_traj_list, v_traj_list, log_v_traj_list = [], [], []  # 存储轨迹。
    time_list = []  # 记录耗时。
    meta_records = []  # 记录元数据。
    range_offset = 0  # 范围模式下的原子数量偏移。
    
    profiler.checkpoint('before_unified_sampling')

    for sample_idx in range(num_samples):  # 逐个生成样本。
        batch = Batch.from_data_list([data.clone()], follow_batch=FOLLOW_BATCH).to(device)  # 构造单样本批次。
        batch_protein = batch.protein_element_batch  # 获取蛋白节点批索引。

        if sample_num_atoms_mode == 'prior':  # 依据空间大小采样原子数。
            pocket_size = atom_num.get_space_size(data.protein_pos.detach().cpu().numpy())
            # 验证 pocket_size
            if np.isnan(pocket_size) or np.isinf(pocket_size) or pocket_size <= 0:
                pocket_size = 30.0
            try:
                sampled = atom_num.sample_atom_num(pocket_size)
                if isinstance(sampled, (np.ndarray, np.generic)):
                    sampled_val = float(sampled.item())
                else:
                    sampled_val = float(sampled)
                if np.isnan(sampled_val) or np.isinf(sampled_val):
                    sampled_val = 10.0
                ligand_num_atoms = max(5, int(abs(sampled_val)))  # 确保至少为5
            except Exception:
                ligand_num_atoms = 10
            batch_ligand = torch.zeros(ligand_num_atoms, dtype=torch.long, device=device)  # 全部归属单个配体。
        elif sample_num_atoms_mode == 'range':  # 按序递增原子数量。
            ligand_num_atoms = max(5, range_offset + 1)  # 计算当前样本的原子数量（从5开始递增），确保至少为5。
            range_offset += 1  # 更新偏移量，为下一个样本做准备。
            batch_ligand = torch.zeros(ligand_num_atoms, dtype=torch.long, device=device)  # 创建配体批次索引（全部归属单个配体）。
        elif sample_num_atoms_mode == 'ref':  # 使用参考配体的原子数。
            batch_ligand = batch.ligand_element_batch
            ligand_num_atoms = max(5, int((batch_ligand == 0).sum().item()))  # 确保至少为5
        else:
            raise ValueError(f'Unknown sample_num_atoms mode {sample_num_atoms_mode}')  # 未知模式报错。

        center = scatter_mean(batch.protein_pos, batch_protein, dim=0)  # 计算蛋白中心。
        init_ligand_pos = center[batch_ligand] + torch.randn((len(batch_ligand), 3), device=device)  # 初始化配体位置。
        init_pos_np = init_ligand_pos.detach().cpu().numpy().astype(np.float64)

        init_log_ligand_v = torch.zeros(len(batch_ligand), model.num_classes, device=device)  # 均匀类别对数概率。
        init_log_ligand_v = F.log_softmax(init_log_ligand_v, dim=-1)  # 归一化。

        profiler.checkpoint(f'sample_{sample_idx}_before_forward')
        monitor.reset_peak_stats()
        
        t_start = time.time()  # 记录开始时间。
        with monitor.monitor_forward(
            model,
            (batch.protein_pos, batch.protein_atom_feature.float(), batch_protein,
             init_ligand_pos, init_log_ligand_v, batch_ligand),
            log_fn=logger.info if logger else None
        ):
            result = model.dynamic_sample_diffusion(  # 调用模型进行动态采样。
                protein_pos=batch.protein_pos,
                protein_v=batch.protein_atom_feature.float(),
                batch_protein=batch_protein,
                init_ligand_pos=init_ligand_pos,
                init_log_ligand_v=init_log_ligand_v,
                batch_ligand=batch_ligand,
                num_steps=num_steps,
                center_pos_mode=center_pos_mode,
                pos_only=pos_only
            )
        t_end = time.time()  # 记录结束时间。
        profiler.checkpoint(f'sample_{sample_idx}_after_forward')

        pos = result['pred_ligand_pos'].detach().cpu().numpy().astype(np.float64)  # 获取最终坐标。
        log_v = result['pred_ligand_v'].detach().cpu().numpy().astype(np.float32)  # 获取最终类别对数概率。
        pos_traj = [p.detach().cpu().numpy().astype(np.float64) for p in result.get('pos_traj', [])]  # 坐标轨迹。
        log_v_traj = [lv.detach().cpu().numpy().astype(np.float32) for lv in result.get('log_v_traj', [])]  # 类别轨迹。
        v_traj = [np.argmax(lv, axis=-1).astype(np.int64) for lv in log_v_traj]  # 将轨迹转为类别索引。

        pos_list.append(pos)  # 累积最终坐标。
        v_list.append(np.argmax(log_v, axis=-1).astype(np.int64))  # 累积最终类别。
        pos_traj_list.append(pos_traj)  # 累积位置轨迹。
        v_traj_list.append(v_traj)  # 累积类别索引轨迹。
        log_v_traj_list.append(log_v_traj)  # 累积类别对数概率轨迹。
        time_list.append(t_end - t_start)  # 累积耗时。
        meta_records.append({  # 记录元信息。
            'method': 'unified_dynamic',
            'ligand_num_atoms': ligand_num_atoms,
            'time': t_end - t_start,
            'model_meta': result.get('meta'),
            'init_pos': init_pos_np.copy(),
        })

        if logger:
            mem_info = monitor.get_memory_info()
            logger.info(f'[Dynamic][Unified] Sample {sample_idx}: {len(pos)} atoms | time {t_end - t_start:.2f}s | mem {mem_info["allocated"]:.1f}/{mem_info["max_allocated"]:.1f} MB')
        
        # 记录GPU监控信息到Excel
        try:
            log_gpu_monitor_record(
                memory_info=monitor.get_memory_info(),
                forward_time=t_end - t_start,
                memory_summary=None,  # 将在最后统一记录
                sampling_info={
                    'mode': 'unified_dynamic',
                    'sample_idx': sample_idx,
                    'ligand_num_atoms': ligand_num_atoms,
                },
                logger=logger
            )
        except Exception as e:
            if logger:
                logger.warning(f'Failed to log GPU monitor record: {e}')

    # 记录最终显存摘要
    if logger:
        summary = profiler.get_summary()
        logger.info(f'[Memory Summary] Peak: {summary["peak_memory_mb"]:.1f} MB')
        try:
            log_gpu_monitor_record(
                memory_info=monitor.get_memory_info(),
                memory_summary=summary,
                sampling_info={
                    'mode': 'unified_dynamic',
                    'stage': 'final_summary',
                },
                logger=logger
            )
        except Exception as e:
            if logger:
                logger.warning(f'Failed to log final GPU monitor summary: {e}')

    # TargetDiff 基准扩散修复（可选）；optimization.enable 且将随后优化时延后在优化后执行
    meta_baseline_refine_ti = None
    if not skip_targetdiff_baseline_refine:
        refine_out = apply_targetdiff_baseline_refinement(
            model, data, pos_list, v_list, config, device=device, logger=logger
        )
        if len(refine_out) == 5:
            pos_list, v_list, refine_pos_traj, refine_v_traj, refine_time_indices = refine_out
            pos_traj_list = [list(pt) + list(rpt) for pt, rpt in zip(pos_traj_list, refine_pos_traj)]
            v_traj_list = [list(vt) + list(rvt) for vt, rvt in zip(v_traj_list, refine_v_traj)]
            meta_baseline_refine_ti = refine_time_indices
        else:
            pos_list, v_list = refine_out

    meta_dict = {
        'method': 'unified',
        'records': meta_records,
        'memory_summary': profiler.get_summary()
    }
    if meta_baseline_refine_ti is not None:
        meta_dict['baseline_refine_time_indices'] = meta_baseline_refine_ti
    return {
        'pos_list': pos_list,
        'v_list': v_list,
        'pos_traj': pos_traj_list,
        'v_traj': v_traj_list,
        'log_v_traj': log_v_traj_list,
        'time_list': time_list,
        'meta': meta_dict,
        'init_list': [r.get('init_pos') for r in meta_records],
    }  # 返回采样结果。
