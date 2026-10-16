"""Sampling baseline responsibilities for ligand sampling."""

from __future__ import annotations

from diffdynamic.data.graph import FOLLOW_BATCH
from diffdynamic.models.probability import log_sample_categorical
from torch_geometric.data import Batch
from torch_scatter import scatter_mean
from torch_scatter import scatter_sum
from tqdm.auto import tqdm
from diffdynamic.evaluation import atom_num
from diffdynamic.io.gpu_monitor_recorder import log_gpu_monitor_record
from diffdynamic.io.monitor import GPUMonitor
from diffdynamic.io.monitor import MemoryProfiler
import numpy as np
import time
import torch


def sample_diffusion_ligand(model, data, num_samples, batch_size=16, device='cuda:0',
                            num_steps=None, pos_only=False, center_pos_mode='protein',
                            sample_num_atoms='prior', logger=None):
    """批量运行标准扩散采样，返回位置/类型及轨迹列表。

    Args:
        model: 扩散模型实例，需实现 `sample_diffusion`。
        data: 作为模板的单个 `ProteinLigandData`。
        num_samples: 目标生成的样本数量。
        batch_size: 每批采样的数量。
        device: 运行设备。
        num_steps: 采样步数，默认为模型默认值。
        pos_only: 是否仅预测坐标而复用原始类型。
        center_pos_mode: 坐标中心化策略。
        sample_num_atoms: 原子数量选择策略（`prior/range/ref`）。
        logger: 可选的日志记录器。

    Returns:
        tuple: 包含采样坐标、类别、完整轨迹及耗时和初始坐标的八元组。
    """
    from diffdynamic.sampling.common import safe_repeat_interleave
    from diffdynamic.sampling.common import unbatch_v_traj

    # 创建GPU监控器
    monitor = GPUMonitor(device=device, enable_flops=False)
    profiler = MemoryProfiler(device=device)
    
    # 验证 batch_size 和 num_samples 参数
    batch_size = max(1, int(batch_size))  # 确保至少为1
    num_samples = max(1, int(num_samples))  # 确保至少为1
    
    all_pred_pos, all_pred_v = [], []  # 累积最终坐标与类别。
    all_pred_pos_traj, all_pred_v_traj = [], []  # 累积轨迹。
    all_pred_v0_traj, all_pred_vt_traj = [], []  # 累积初/末时间轨迹。
    all_init_pos = []  # 初始高斯坐标（与终态同坐标系）
    time_list = []  # 记录每批耗时。
    num_batch = int(np.ceil(num_samples / batch_size))  # 计算批次数。
    current_i = 0  # 范围模式偏移。
    
    profiler.checkpoint('before_baseline_sampling')
    for i in tqdm(range(num_batch)):  # 遍历每个批次，显示进度条。
        n_data = batch_size if i < num_batch - 1 else num_samples - batch_size * (num_batch - 1)  # 当前批大小（最后一批可能不满）。
        # 构建批次：克隆数据 n_data 次并组合为批次。
        batch = Batch.from_data_list([data.clone() for _ in range(n_data)], follow_batch=FOLLOW_BATCH).to(device)

        profiler.checkpoint(f'baseline_batch_{i}_before')
        monitor.reset_peak_stats()
        
        t1 = time.time()  # 记录批次起始时间。
        with torch.no_grad():
            batch_protein = batch.protein_element_batch  # 获取蛋白批索引。
            if sample_num_atoms == 'prior':
                pocket_size = atom_num.get_space_size(data.protein_pos.detach().cpu().numpy())
                # 验证 pocket_size
                if np.isnan(pocket_size) or np.isinf(pocket_size) or pocket_size <= 0:
                    pocket_size = 30.0
                ligand_num_atoms = []
                for _ in range(n_data):
                    try:
                        sampled = atom_num.sample_atom_num(pocket_size)
                        if isinstance(sampled, (np.ndarray, np.generic)):
                            sampled_val = float(sampled.item())
                        else:
                            sampled_val = float(sampled)
                        if np.isnan(sampled_val) or np.isinf(sampled_val):
                            sampled_val = 20.0
                        atom_count = max(5, int(abs(sampled_val)))  # 确保至少为5
                        ligand_num_atoms.append(atom_count)
                    except Exception:
                        ligand_num_atoms.append(20)  # 默认值（药物分子的合理原子数）
                # 最终验证：确保所有值至少为5
                ligand_num_atoms = [max(5, abs(int(n))) for n in ligand_num_atoms]
                repeats_tensor = torch.tensor(ligand_num_atoms, dtype=torch.long)
                repeats_tensor = torch.clamp(repeats_tensor, min=5)  # 确保至少为5
                # 确保值非负且至少为5
                ligand_num_atoms = [max(5, int(n)) for n in ligand_num_atoms]
                # 使用自定义的 safe_repeat_interleave 避免 CUDA 兼容性问题
                batch_ligand = safe_repeat_interleave(torch.arange(n_data, dtype=torch.long), ligand_num_atoms, device=device)
            elif sample_num_atoms == 'range':
                ligand_num_atoms = list(range(current_i + 1, current_i + n_data + 1))
                ligand_num_atoms = [max(5, abs(int(n))) for n in ligand_num_atoms]
                repeats_tensor = torch.tensor(ligand_num_atoms, dtype=torch.long)
                repeats_tensor = torch.clamp(repeats_tensor, min=5)  # 确保至少为5
                # 确保值非负且至少为5
                ligand_num_atoms = [max(5, int(n)) for n in ligand_num_atoms]
                # 使用自定义的 safe_repeat_interleave 避免 CUDA 兼容性问题
                batch_ligand = safe_repeat_interleave(torch.arange(n_data, dtype=torch.long), ligand_num_atoms, device=device)
            elif sample_num_atoms == 'ref':
                batch_ligand = batch.ligand_element_batch
                ligand_num_atoms = scatter_sum(torch.ones_like(batch_ligand), batch_ligand, dim=0).tolist()
                ligand_num_atoms = [max(5, int(n)) for n in ligand_num_atoms]  # 确保至少为5
            else:
                raise ValueError  # 未知的原子数量采样模式。

            # 初始化配体位置：以蛋白中心为基准，添加随机噪声。
            center_pos = scatter_mean(batch.protein_pos, batch_protein, dim=0)  # 计算每个样本的蛋白中心。
            batch_center_pos = center_pos[batch_ligand]  # 为每个配体原子分配对应的蛋白中心。
            init_ligand_pos = batch_center_pos + torch.randn_like(batch_center_pos)  # 在蛋白中心附近添加随机初始位置。
            init_ligand_pos_np = init_ligand_pos.detach().cpu().numpy().astype(np.float64)

            # 初始化配体类别：根据 pos_only 标志选择策略。
            if pos_only:
                init_ligand_v = batch.ligand_atom_feature_full  # 如果仅采样位置，复用原始配体类别。
            else:
                uniform_logits = torch.zeros(len(batch_ligand), model.num_classes).to(device)  # 创建均匀的类别 logits。
                init_ligand_v = log_sample_categorical(uniform_logits)  # 从均匀分布中采样初始类别。

            with monitor.monitor_forward(
                model,
                (batch.protein_pos, batch.protein_atom_feature.float(), batch_protein,
                 init_ligand_pos, init_ligand_v, batch_ligand),
                log_fn=None  # 避免日志过多
            ):
                r = model.sample_diffusion(
                    protein_pos=batch.protein_pos,
                    protein_v=batch.protein_atom_feature.float(),
                    batch_protein=batch_protein,

                    init_ligand_pos=init_ligand_pos,
                    init_ligand_v=init_ligand_v,
                    batch_ligand=batch_ligand,
                    num_steps=num_steps,
                    pos_only=pos_only,
                    center_pos_mode=center_pos_mode
                )
            # 解包采样结果：提取位置、类别和轨迹。
            ligand_pos, ligand_v, ligand_pos_traj, ligand_v_traj = r['pos'], r['v'], r['pos_traj'], r['v_traj']
            ligand_v0_traj, ligand_vt_traj = r['v0_traj'], r['vt_traj']  # 提取 v0 和 vt 预测轨迹。
            # 拆分位置轨迹：将批次结果按样本原子数拆分。
            ligand_cum_atoms = np.cumsum([0] + ligand_num_atoms)  # 计算累积原子数边界。
            ligand_pos_array = ligand_pos.cpu().numpy().astype(np.float64)
            all_pred_pos += [ligand_pos_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]] for k in
                             range(n_data)]  # num_samples * [num_atoms_i, 3]
            all_init_pos += [init_ligand_pos_np[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]].copy() for k in
                             range(n_data)]

            all_step_pos = [[] for _ in range(n_data)]
            for p in ligand_pos_traj:  # step_i
                p_array = p.cpu().numpy().astype(np.float64)
                for k in range(n_data):
                    all_step_pos[k].append(p_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]])
            all_step_pos = [np.stack(step_pos) for step_pos in
                            all_step_pos]  # num_samples * [num_steps, num_atoms_i, 3]
            all_pred_pos_traj += [p for p in all_step_pos]  # 累积位置轨迹。

            # 拆分类别轨迹：将批次结果按样本原子数拆分。
            ligand_v_array = ligand_v.cpu().numpy()
            all_pred_v += [ligand_v_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]] for k in range(n_data)]

            all_step_v = unbatch_v_traj(ligand_v_traj, n_data, ligand_cum_atoms)
            all_pred_v_traj += [v for v in all_step_v]

            if not pos_only:
                all_step_v0 = unbatch_v_traj(ligand_v0_traj, n_data, ligand_cum_atoms)
                all_pred_v0_traj += [v for v in all_step_v0]
                all_step_vt = unbatch_v_traj(ligand_vt_traj, n_data, ligand_cum_atoms)
                all_pred_vt_traj += [v for v in all_step_vt]
        t2 = time.time()  # 记录结束时间。
        time_list.append(t2 - t1)  # 累计耗时。
        profiler.checkpoint(f'baseline_batch_{i}_after')
        
        # 记录GPU监控信息（仅在关键批次记录，避免日志过多）
        if logger and (i == 0 or i == num_batch - 1):
            mem_info = monitor.get_memory_info()
            logger.info(f'[Baseline] Batch {i}: Time {t2 - t1:.2f}s | Mem {mem_info["allocated"]:.1f}/{mem_info["max_allocated"]:.1f} MB')
            try:
                log_gpu_monitor_record(
                    memory_info=mem_info,
                    forward_time=t2 - t1,
                    sampling_info={
                        'mode': 'baseline',
                        'batch_idx': i,
                        'batch_size': n_data,
                    },
                    logger=logger
                )
            except Exception as e:
                if logger:
                    logger.warning(f'Failed to log GPU monitor record for baseline batch: {e}')
        
        current_i += n_data  # 更新范围模式偏移。
    
    # 记录最终显存摘要
    if logger:
        summary = profiler.get_summary()
        logger.info(f'[Baseline Memory Summary] Peak: {summary["peak_memory_mb"]:.1f} MB')
        try:
            log_gpu_monitor_record(
                memory_info=monitor.get_memory_info(),
                memory_summary=summary,
                sampling_info={
                    'mode': 'baseline',
                    'stage': 'final_summary',
                },
                logger=logger
            )
        except Exception as e:
            if logger:
                logger.warning(f'Failed to log final GPU monitor summary: {e}')
    
    return all_pred_pos, all_pred_v, all_pred_pos_traj, all_pred_v_traj, all_pred_v0_traj, all_pred_vt_traj, time_list, all_init_pos
