"""Sampling dynamic responsibilities for ligand sampling."""

from __future__ import annotations

from diffdynamic.data.graph import FOLLOW_BATCH
from diffdynamic.models.probability import center_pos
from diffdynamic.models.probability import index_to_log_onehot
from diffdynamic.models.probability import log_sample_categorical
from rdkit import Chem
from torch_geometric.data import Batch
from torch_scatter import scatter_mean
from torch_scatter import scatter_sum
from diffdynamic.data.template import build_template_initialization_batch
from diffdynamic.data.template import load_template_geometry
from diffdynamic.evaluation import atom_num
from diffdynamic.io.gpu_monitor_recorder import log_gpu_monitor_record
from diffdynamic.io.monitor import GPUMonitor
from diffdynamic.io.monitor import MemoryProfiler
import numpy as np
import time
import torch
import torch.nn.functional as F
import diffdynamic.chemistry.reconstruct as reconstruct
import diffdynamic.data.transforms as trans


def _run_legacy_dynamic(model, data, config, ligand_atom_mode, device='cuda:0', logger=None, skip_targetdiff_baseline_refine=False):
    """按照旧版两阶段策略执行动态采样。

    Args:
        model: 扩散模型，需实现 `sample_diffusion_large_step` 与 `sample_diffusion_refinement`。
        data: 单个蛋白口袋样本。
        config: 采样配置对象，含 `sample.dynamic` 字段。
        ligand_atom_mode: 配体原子编码模式。
        device: 推理设备。
        logger: 可选日志器。

    Returns:
        dict: 精炼后的分子列表、轨迹、耗时及候选元数据。
    """
    from diffdynamic.sampling.candidates import _selector_metrics_or_skip
    from diffdynamic.sampling.candidates import evaluate_candidate
    from diffdynamic.sampling.candidates import select_top_candidates
    from diffdynamic.sampling.common import _validate_batch_indices
    from diffdynamic.sampling.common import safe_repeat_interleave
    from diffdynamic.sampling.configuration import _dynamic_subcfg_grad_cap_kwargs
    from diffdynamic.sampling.configuration import get_time_boundary
    from diffdynamic.sampling.optimization import _forward_diffuse_molecule
    from diffdynamic.sampling.prudent.refinement import _batch_refinement_forward_split_oom
    from diffdynamic.sampling.refinement import apply_targetdiff_baseline_refinement
    from diffdynamic.sampling.trajectory import _coerce_time_indices_seq

    # 创建GPU监控器
    monitor = GPUMonitor(device=device, enable_flops=False)
    profiler = MemoryProfiler(device=device)
    
    dynamic_cfg = config.sample.get('dynamic', {})  # 读取整体动态配置。
    large_cfg = dynamic_cfg.get('large_step', {})  # 大步探索阶段配置。
    refine_cfg = dynamic_cfg.get('refine', {})  # 精炼阶段配置。
    selector_cfg = dynamic_cfg.get('selector', {})  # 候选筛选配置。
    template_init_cfg = dict(dynamic_cfg.get('template_init', {}) or {})
    template_geometry = None
    if bool(template_init_cfg.get('enable', False)):
        template_path = template_init_cfg.get('ligand_path')
        scaffold_smarts = template_init_cfg.get('scaffold_smarts')
        if not template_path or not scaffold_smarts:
            raise ValueError(
                'dynamic.template_init requires ligand_path and scaffold_smarts'
            )
        template_geometry = load_template_geometry(
            template_path,
            scaffold_smarts,
            require_3d=bool(template_init_cfg.get('require_3d', True)),
        )
        if logger:
            logger.info(
                '[DynamicTemplateInit] enabled | '
                f'template={template_geometry.source} | '
                f'scaffold={template_geometry.scaffold_atom_count} | '
                f'target={template_geometry.target_atom_count}'
            )

    # 读取时间节点配置：time_boundary 保留原有功能，selection_time 用于中间筛选
    time_boundary = get_time_boundary(dynamic_cfg, 750)  # time_boundary 用于划分 large_step 和 refine
    enable_selection = selector_cfg.get('enable_selection', False)
    selection_time = selector_cfg.get('selection_time')
    
    # 确保 refine 使用 time_boundary（原有功能）
    if 'time_upper' not in refine_cfg:
        refine_cfg['time_upper'] = time_boundary
    
    if logger:
        if enable_selection and selection_time is not None:
            logger.info(f'[Selector] Selection enabled at t={selection_time}, time_boundary={time_boundary} (both coexist)')
        else:
            logger.info(f'[Selector] Using time_boundary={time_boundary} (selection disabled)')

    center_pos_mode = config.sample.get('center_pos_mode', 'protein')  # 坐标中心化策略。
    pos_only = config.sample.get('pos_only', False)  # 是否仅采样坐标。
    sample_num_atoms_mode = config.sample.get('sample_num_atoms', 'prior')  # 原子数量策略。

    large_batch_size = large_cfg.get('batch_size', config.sample.get('batch_size', 16))  # 大步批大小。
    n_repeat = large_cfg.get('n_repeat', 1)  # 重复次数。
    
    # 确保 batch_size 和 n_repeat 都是非负整数
    large_batch_size = max(1, int(large_batch_size))  # 确保至少为1
    n_repeat = max(0, int(n_repeat))  # 确保非负（0表示不执行，但不会报错）
    if template_geometry is not None and n_repeat != 1:
        raise ValueError(
            'dynamic.template_init currently requires large_step.n_repeat=1 '
            'so its per-batch scaffold RePaint state cannot be mixed across repeats'
        )
    
    if logger:
        logger.info(f'[Dynamic] Large-step batch size: {large_batch_size} | repeats: {n_repeat}')
        if n_repeat == 0:
            logger.warning('n_repeat is 0, no large-step sampling will be performed')

    total_candidates = []  # 存储所有候选。
    range_offset = 0  # 范围模式的原子偏移。
    time_records = {'large_step': [], 'refine': [], 'baseline_refine': []}  # 记录各阶段耗时。
    largestep_smiles_list = []  # 实验3：largestep 完成时强制重建的 SMILES（不中断采样）
    template_repaint_cfg = None
    
    profiler.checkpoint('before_large_step')

    for repeat_idx in range(n_repeat):  # 逐次执行大步探索。
        batch = Batch.from_data_list([data.clone() for _ in range(large_batch_size)],
                                     follow_batch=FOLLOW_BATCH).to(device)  # 构建批次。
        n_data = large_batch_size  # 当前批包含的样本数。
        batch_protein = batch.protein_element_batch  # 直接复用 PyG 创建的批次索引
        _validate_batch_indices(batch_protein, n_data, 'batch_protein',
                                logger=logger, context='legacy_large_step')

        template_batch = None
        if template_geometry is not None:
            atom_count_values = template_init_cfg.get(
                'atom_count_values', [29, 30, 31, 32, 33]
            )
            base_seed = int(config.sample.get('seed', 42))
            template_batch = build_template_initialization_batch(
                template_geometry,
                batch_size=n_data,
                atom_count_values=atom_count_values,
                seed=base_seed + repeat_idx * 1_000_003,
                coordinate_jitter_std=float(
                    template_init_cfg.get('coordinate_jitter_std', 0.20)
                ),
                extra_point_jitter_std=float(
                    template_init_cfg.get('extra_point_jitter_std', 0.35)
                ),
            )
            ligand_num_atoms = template_batch.atom_counts
            batch_ligand = safe_repeat_interleave(
                torch.arange(n_data, dtype=torch.long),
                ligand_num_atoms,
                device=device,
            )
        elif sample_num_atoms_mode == 'prior':  # 根据 pocket 尺寸采样原子数。
            pocket_size = atom_num.get_space_size(data.protein_pos.detach().cpu().numpy())
            # 验证 pocket_size 的有效性
            if np.isnan(pocket_size) or np.isinf(pocket_size) or pocket_size <= 0:
                if logger:
                    logger.warning(f'Invalid pocket_size: {pocket_size}, using default value 30.0')
                pocket_size = 30.0
            ligand_num_atoms = []
            for i in range(n_data):
                try:
                    sampled = atom_num.sample_atom_num(pocket_size)
                    # 转换为Python原生类型，处理各种可能的numpy类型
                    if isinstance(sampled, (np.ndarray, np.generic)):
                        sampled_val = float(sampled.item())
                    else:
                        sampled_val = float(sampled)
                    # 检查是否为有效数字
                    if np.isnan(sampled_val) or np.isinf(sampled_val):
                        if logger:
                            logger.warning(f'Sampled invalid value: {sampled_val}, using default 20')
                        sampled_val = 20.0
                    # 确保为正整数，至少为5（更合理的药物分子最小原子数）
                    atom_count = max(5, int(abs(sampled_val)))
                    ligand_num_atoms.append(atom_count)
                except Exception as e:
                    if logger:
                        logger.error(f'Error sampling atom num for sample {i}: {e}, using default 20')
                    ligand_num_atoms.append(20)  # 默认值（药物分子的合理原子数）
            # 最终验证：确保所有值都是正整数，至少为5
            ligand_num_atoms = [max(5, abs(int(n))) for n in ligand_num_atoms]
            # 再次检查，确保没有任何值 < 5
            for i, n in enumerate(ligand_num_atoms):
                if n < 5:
                    if logger:
                        logger.warning(f'Found atom count < 5 at index {i}: {n}, correcting to 5')
                    ligand_num_atoms[i] = 5
            # 创建tensor前最后一次验证
            try:
                # 先在CPU上创建tensor，避免CUDA兼容性问题
                repeats_tensor = torch.tensor(ligand_num_atoms, device='cpu', dtype=torch.long)
                # 验证tensor创建成功且值正确
                if len(repeats_tensor) != len(ligand_num_atoms):
                    raise ValueError(f"Tensor length mismatch: expected {len(ligand_num_atoms)}, got {len(repeats_tensor)}")
                if torch.any(repeats_tensor < 5):
                    if logger:
                        logger.warning(f"Tensor contains values < 5: {repeats_tensor.tolist()}, clamping to 5")
                    repeats_tensor = torch.clamp(repeats_tensor, min=5)  # 确保所有值 >= 5
                    # 同步更新 ligand_num_atoms
                    ligand_num_atoms = repeats_tensor.tolist()
                # 使用clamp确保所有值都 >= 5
                repeats_tensor = torch.clamp(repeats_tensor, min=1)
                # 确保所有值都是正整数
                if torch.any(repeats_tensor <= 0):
                    if logger:
                        logger.warning(f'[prior mode] Found non-positive values after clamp: {repeats_tensor.tolist()}, forcing to 20')
                    repeats_tensor = torch.full((n_data,), 20, device='cpu', dtype=torch.long)
                # 移动到目标设备
                repeats_tensor = repeats_tensor.to(device).long()
                # 验证移动后tensor仍然有效
                if torch.any(repeats_tensor <= 0) or len(repeats_tensor) != n_data:
                    if logger:
                        logger.warning(f'[prior mode] Tensor invalid after device move: {repeats_tensor.tolist()}, forcing to 20')
                    repeats_tensor = torch.full((n_data,), 20, device='cpu', dtype=torch.long).to(device).long()
                if logger:
                    logger.debug(f'[prior mode] repeats_tensor: {repeats_tensor.tolist()}, ligand_num_atoms: {ligand_num_atoms}')
            except Exception as e:
                if logger:
                    logger.error(f'Error creating repeats tensor: {e}, ligand_num_atoms={ligand_num_atoms}')
                # 如果创建tensor失败，使用默认值 20（药物分子的合理原子数）
                try:
                    repeats_tensor = torch.full((n_data,), 20, device='cpu', dtype=torch.long).to(device).long()
                except Exception as e2:
                    if logger:
                        logger.error(f'Error creating fallback tensor: {e2}, using CPU only')
                    repeats_tensor = torch.full((n_data,), 20, device='cpu', dtype=torch.long)
            # 调用前最后一次安全检查：确保设备、类型和值都正确
            repeats_tensor = repeats_tensor.to(device).long()
            if torch.any(repeats_tensor <= 0):
                if logger:
                    logger.warning(f"[prior mode] 最终修复无效的 repeats_tensor: {repeats_tensor.tolist()}，使用默认原子数 20")
                try:
                    repeats_tensor = torch.full((n_data,), 20, device='cpu', dtype=torch.long).to(device).long()
                except Exception:
                    repeats_tensor = torch.full((n_data,), 20, device='cpu', dtype=torch.long)
            # 最终验证：确保总原子数不为0
            total_atoms = sum(ligand_num_atoms)
            if total_atoms == 0:
                if logger:
                    logger.error(f"Invalid ligand atom count: total atoms is 0, using default values")
                ligand_num_atoms = [20] * n_data  # 使用默认值
            
            # 直接使用 ligand_num_atoms 的值，避免从 CUDA tensor 获取值可能出错
            # indices 在 CPU 上创建，避免在不受支持的 GPU 架构上初始化失败
            indices = torch.arange(n_data, dtype=torch.long)
            batch_ligand = safe_repeat_interleave(indices, ligand_num_atoms, device=device)
            
            # 验证批次总原子数
            if batch_ligand.numel() == 0:
                raise ValueError(f"Invalid batch_ligand: total atoms is 0, ligand_num_atoms={ligand_num_atoms}")
            
            # 验证 batch_ligand 长度与 ligand_num_atoms 总和一致
            expected_total = sum(ligand_num_atoms)
            actual_total = batch_ligand.numel()
            if actual_total != expected_total:
                raise ValueError(
                    f"[prior mode] batch_ligand length mismatch: expected {expected_total} (sum of ligand_num_atoms={ligand_num_atoms}), "
                    f"got {actual_total}. This indicates a problem with safe_repeat_interleave or ligand_num_atoms."
                )
            
            # 验证 batch_ligand 的索引范围
            if batch_ligand.max().item() >= n_data or batch_ligand.min().item() < 0:
                raise ValueError(
                    f"[prior mode] batch_ligand indices out of range: range=[{batch_ligand.min().item()}, {batch_ligand.max().item()}], "
                    f"expected [0, {n_data-1}]. ligand_num_atoms={ligand_num_atoms}"
                )
        elif sample_num_atoms_mode == 'range':  # 使用连续范围。
            ligand_num_atoms = list(range(range_offset + 1, range_offset + n_data + 1))
            range_offset += n_data
            # 验证范围模式下的原子数都是正数，至少为5（虽然应该总是正数，但为了安全起见）
            ligand_num_atoms = [max(5, abs(int(n))) for n in ligand_num_atoms]
            try:
                # 先在CPU上创建tensor，避免CUDA兼容性问题
                repeats_tensor = torch.tensor(ligand_num_atoms, device='cpu', dtype=torch.long)
                # 验证tensor创建成功
                if len(repeats_tensor) != len(ligand_num_atoms) or torch.any(repeats_tensor <= 0):
                    raise ValueError(f"Invalid tensor: length={len(repeats_tensor)}, values={repeats_tensor.tolist()}")
                repeats_tensor = torch.clamp(repeats_tensor, min=1)  # 确保所有值 >= 1
                # 确保所有值都是正整数
                if torch.any(repeats_tensor <= 0):
                    if logger:
                        logger.warning(f'[range mode] Found non-positive values: {repeats_tensor.tolist()}, forcing to 20')
                    repeats_tensor = torch.full((n_data,), 20, device='cpu', dtype=torch.long)
                # 移动到目标设备
                repeats_tensor = repeats_tensor.to(device).long()
                # 验证移动后tensor仍然有效
                if torch.any(repeats_tensor <= 0) or len(repeats_tensor) != n_data:
                    if logger:
                        logger.warning(f'[range mode] Tensor invalid after device move: {repeats_tensor.tolist()}, forcing to 20')
                    repeats_tensor = torch.full((n_data,), 20, device='cpu', dtype=torch.long).to(device).long()
                if logger:
                    logger.debug(f'[range mode] repeats_tensor: {repeats_tensor.tolist()}, ligand_num_atoms: {ligand_num_atoms}')
            except Exception as e:
                if logger:
                    logger.error(f'Error creating repeats tensor in range mode: {e}, ligand_num_atoms={ligand_num_atoms}')
                try:
                    repeats_tensor = torch.full((n_data,), 20, device='cpu', dtype=torch.long).to(device).long()
                except Exception as e2:
                    if logger:
                        logger.error(f'Error creating fallback tensor: {e2}, using CPU only')
                    repeats_tensor = torch.full((n_data,), 20, device='cpu', dtype=torch.long)
            # 调用前最后一次安全检查：确保设备、类型和值都正确
            repeats_tensor = repeats_tensor.to(device).long()
            if torch.any(repeats_tensor <= 0):
                if logger:
                    logger.warning(f"[range mode] 最终修复无效的 repeats_tensor: {repeats_tensor.tolist()}，使用默认原子数 20")
                try:
                    repeats_tensor = torch.full((n_data,), 20, device='cpu', dtype=torch.long).to(device).long()
                except Exception:
                    repeats_tensor = torch.full((n_data,), 20, device='cpu', dtype=torch.long)
            # 最终验证：确保总原子数不为0
            total_atoms = sum(ligand_num_atoms)
            if total_atoms == 0:
                if logger:
                    logger.error(f"Invalid ligand atom count: total atoms is 0, using default values")
                ligand_num_atoms = [20] * n_data  # 使用默认值
            
            # 直接使用 ligand_num_atoms 的值，避免从 CUDA tensor 获取值可能出错
            indices = torch.arange(n_data, dtype=torch.long)
            batch_ligand = safe_repeat_interleave(indices, ligand_num_atoms, device=device)
            
            # 验证批次总原子数
            if batch_ligand.numel() == 0:
                raise ValueError(f"Invalid batch_ligand: total atoms is 0, ligand_num_atoms={ligand_num_atoms}")
            
            # 验证 batch_ligand 长度与 ligand_num_atoms 总和一致
            expected_total = sum(ligand_num_atoms)
            actual_total = batch_ligand.numel()
            if actual_total != expected_total:
                raise ValueError(
                    f"[range mode] batch_ligand length mismatch: expected {expected_total} (sum of ligand_num_atoms={ligand_num_atoms}), "
                    f"got {actual_total}. This indicates a problem with safe_repeat_interleave or ligand_num_atoms."
                )
            
            # 验证 batch_ligand 的索引范围
            if batch_ligand.max().item() >= n_data or batch_ligand.min().item() < 0:
                raise ValueError(
                    f"[range mode] batch_ligand indices out of range: range=[{batch_ligand.min().item()}, {batch_ligand.max().item()}], "
                    f"expected [0, {n_data-1}]. ligand_num_atoms={ligand_num_atoms}"
                )
        elif sample_num_atoms_mode == 'ref':  # 使用参考原子数。
            batch_ligand = batch.ligand_element_batch
            ligand_num_atoms = scatter_sum(torch.ones_like(batch_ligand), batch_ligand, dim=0).tolist()
            ligand_num_atoms = [max(5, abs(int(n))) for n in ligand_num_atoms]  # 确保至少为5
            # 验证tensor创建前的值
            try:
                # 先在CPU上创建tensor，避免CUDA兼容性问题
                repeats_tensor = torch.tensor(ligand_num_atoms, device='cpu', dtype=torch.long)
                # 验证tensor创建成功
                if len(repeats_tensor) != len(ligand_num_atoms) or torch.any(repeats_tensor <= 0):
                    raise ValueError(f"Invalid tensor: length={len(repeats_tensor)}, values={repeats_tensor.tolist()}")
                repeats_tensor = torch.clamp(repeats_tensor, min=1)  # 确保所有值 >= 1
                # 确保所有值都是正整数
                if torch.any(repeats_tensor <= 0):
                    if logger:
                        logger.warning(f'[ref mode] Found non-positive values: {repeats_tensor.tolist()}, forcing to 20')
                    repeats_tensor = torch.full((n_data,), 20, device='cpu', dtype=torch.long)
                # 移动到目标设备
                repeats_tensor = repeats_tensor.to(device).long()
                # 验证移动后tensor仍然有效
                if torch.any(repeats_tensor <= 0) or len(repeats_tensor) != n_data:
                    if logger:
                        logger.warning(f'[ref mode] Tensor invalid after device move: {repeats_tensor.tolist()}, forcing to 20')
                    repeats_tensor = torch.full((n_data,), 20, device='cpu', dtype=torch.long).to(device).long()
                if logger:
                    logger.debug(f'[ref mode] repeats_tensor: {repeats_tensor.tolist()}, ligand_num_atoms: {ligand_num_atoms}')
            except Exception as e:
                if logger:
                    logger.error(f'Error creating repeats tensor in ref mode: {e}, ligand_num_atoms={ligand_num_atoms}')
                try:
                    repeats_tensor = torch.full((n_data,), 20, device='cpu', dtype=torch.long).to(device).long()
                except Exception as e2:
                    if logger:
                        logger.error(f'Error creating fallback tensor: {e2}, using CPU only')
                    repeats_tensor = torch.full((n_data,), 20, device='cpu', dtype=torch.long)
            # 确保tensor长度正确并移动到目标设备
            if len(repeats_tensor) != n_data:
                if logger:
                    logger.warning(f"[ref mode] Tensor length mismatch: {len(repeats_tensor)} != {n_data}, using default 20")
                repeats_tensor = torch.full((n_data,), 20, device='cpu', dtype=torch.long)
            # 移动到目标设备并确保类型正确，同时使用clamp确保值至少为1
            try:
                repeats_tensor = repeats_tensor.to(device).long()
                # 使用clamp确保所有值至少为1（避免负数或零）
                repeats_tensor = torch.clamp(repeats_tensor, min=1)
            except Exception as e:
                if logger:
                    logger.warning(f"[ref mode] Failed to move tensor to device {device}: {e}, using CPU with default 20")
                repeats_tensor = torch.full((n_data,), 20, device='cpu', dtype=torch.long)
            # 注意：ref模式下batch_ligand已经设置，不需要重新创建
            # 验证ref模式下的batch_ligand不为空
            if batch_ligand.numel() == 0:
                if logger:
                    logger.error(f"Invalid batch_ligand in ref mode: total atoms is 0")
                raise ValueError(f"Invalid batch_ligand: total atoms is 0 in ref mode")
        else:
            raise ValueError(f'Unknown sample_num_atoms mode {sample_num_atoms_mode}')

        _validate_batch_indices(
            batch_ligand, n_data, 'batch_ligand',
            logger=logger, context=f'legacy_large_step_{sample_num_atoms_mode}'
        )

        center = scatter_mean(batch.protein_pos, batch_protein, dim=0)  # 计算蛋白中心。
        if template_batch is None:
            init_ligand_pos = center[batch_ligand] + torch.randn_like(
                center[batch_ligand]
            )
        else:
            init_ligand_pos = torch.tensor(
                template_batch.flattened_positions(),
                dtype=torch.float32,
                device=device,
            )

        # 验证初始化后的配体位置
        if init_ligand_pos.numel() == 0:
            raise ValueError(
                f"init_ligand_pos is empty. batch_ligand.shape={batch_ligand.shape}, "
                f"center.shape={center.shape}, batch_protein.max()={batch_protein.max().item()}"
            )
        if init_ligand_pos.shape[0] != batch_ligand.shape[0]:
            raise ValueError(
                f"init_ligand_pos shape mismatch. init_ligand_pos.shape={init_ligand_pos.shape}, "
                f"batch_ligand.shape={batch_ligand.shape}"
            )

        total_atoms = len(batch_ligand)  # 当前批次总原子数。
        uniform_logits = torch.zeros(total_atoms, model.num_classes, device=device)  # 均匀类别 logits。
        if getattr(model, 'ligand_v_input', 'onehot') == 'log_prob':
            init_ligand_v_input = F.log_softmax(uniform_logits, dim=-1)
            log_mode = 'log_prob'
        else:
            init_ligand_v_input = log_sample_categorical(uniform_logits)
            log_mode = 'auto'  # 自动模式：根据模型配置自动选择输入格式。

        if template_batch is not None:
            # Convert the real-ligand coordinate cloud into a legal x_t state.
            # Atom classes start from a uniform categorical x0 and therefore
            # do not expose the template's target-side elements to the model.
            template_log_v = F.log_softmax(uniform_logits, dim=-1)
            _, template_pos_centered, template_offset = center_pos(
                batch.protein_pos,
                init_ligand_pos,
                batch_protein,
                batch_ligand,
                mode=center_pos_mode,
            )
            diffusion_start_t = min(
                int(template_init_cfg.get('diffusion_start_t', 999)),
                int(model.num_timesteps) - 1,
            )
            noised_pos, noised_log_v = _forward_diffuse_molecule(
                model,
                template_pos_centered,
                template_log_v,
                batch_ligand,
                diffusion_start_t,
                device,
            )
            init_ligand_pos = noised_pos + template_offset[batch_ligand]
            init_ligand_v_input = noised_log_v
            log_mode = 'log_prob'
            if bool(template_init_cfg.get('repaint_scaffold', True)):
                reference_v = getattr(data, 'ligand_atom_feature_full', None)
                if reference_v is None:
                    raise ValueError(
                        'dynamic.template_init repaint requires the template '
                        'ligand atom features'
                    )
                reference_v = reference_v.to(device=device, dtype=torch.long)
                if int(reference_v.shape[0]) != int(template_geometry.total_atom_count):
                    raise ValueError(
                        'template atom order/count differs from the loaded '
                        f'ligand data ({reference_v.shape[0]} vs '
                        f'{template_geometry.total_atom_count})'
                    )
                scaffold_indices = torch.tensor(
                    template_geometry.scaffold_atom_indices,
                    dtype=torch.long,
                    device=device,
                )
                scaffold_log_v = index_to_log_onehot(
                    reference_v[scaffold_indices], model.num_classes
                )
                repaint_log_v = template_log_v.clone()
                repaint_pos_mask = torch.zeros(
                    total_atoms, dtype=torch.float32, device=device
                )
                cursor = 0
                n_scaffold = int(template_geometry.scaffold_atom_count)
                for count in ligand_num_atoms:
                    repaint_log_v[cursor:cursor + n_scaffold] = scaffold_log_v
                    repaint_pos_mask[cursor:cursor + n_scaffold] = 1.0
                    cursor += int(count)
                repaint_type_mask = (
                    repaint_pos_mask.clone()
                    if bool(template_init_cfg.get('repaint_scaffold_types', True))
                    else None
                )
                template_repaint_cfg = {
                    'x0_pos': template_pos_centered,
                    'x0_log_v': repaint_log_v,
                    'pos_mask': (
                        repaint_pos_mask
                        if bool(template_init_cfg.get('repaint_scaffold_positions', True))
                        else torch.zeros_like(repaint_pos_mask)
                    ),
                    'type_mask': repaint_type_mask,
                    'use_mean_for_discrete': True,
                    '_use_dual_mask': True,
                }
            if logger:
                logger.info(
                    '[DynamicTemplateInit] q(x_t|x0) initialization | '
                    f't={diffusion_start_t} | counts={ligand_num_atoms} | '
                    f'repaint_scaffold={template_repaint_cfg is not None}'
                )

        # Same world coordinate frame as the final pred_ligand_pos.
        init_ligand_pos_np = (
            init_ligand_pos.detach().cpu().numpy().astype(np.float64)
        )

        # 验证初始化后的配体类别输入
        if init_ligand_v_input.numel() == 0:
            raise ValueError(
                f"init_ligand_v_input is empty. total_atoms={total_atoms}, "
                f"model.num_classes={model.num_classes}"
            )
        if init_ligand_v_input.shape[0] != batch_ligand.shape[0]:
            raise ValueError(
                f"init_ligand_v_input shape mismatch. init_ligand_v_input.shape={init_ligand_v_input.shape}, "
                f"batch_ligand.shape={batch_ligand.shape}"
            )

        # 验证输入到模型的维度（在调用前）
        if logger:
            logger.debug(
                f"Input validation: protein_pos.shape={batch.protein_pos.shape}, "
                f"protein_v.shape={batch.protein_atom_feature.shape}, "
                f"batch_protein.max()={batch_protein.max().item()}, "
                f"batch_protein.device={batch_protein.device}, "
                f"batch_protein.unique()={batch_protein.unique().tolist()}, "
                f"init_ligand_pos.shape={init_ligand_pos.shape}, "
                f"init_ligand_v_input.shape={init_ligand_v_input.shape}, "
                f"batch_ligand.shape={batch_ligand.shape}, "
                f"batch_ligand.max()={batch_ligand.max().item()}, "
                f"batch_ligand.device={batch_ligand.device}, "
                f"batch_ligand.unique()={batch_ligand.unique().tolist()}, "
                f"n_data={n_data}"
            )
        
        # 验证 batch_protein 索引范围
        if batch_protein.max().item() >= n_data:
            raise ValueError(
                f"batch_protein indices out of range. batch_protein.max()={batch_protein.max().item()}, "
                f"n_data={n_data}"
            )
        
        # 验证 batch_ligand 索引范围
        if batch_ligand.max().item() >= n_data:
            raise ValueError(
                f"batch_ligand indices out of range. batch_ligand.max()={batch_ligand.max().item()}, "
                f"n_data={n_data}"
            )

        profiler.checkpoint(f'large_step_repeat_{repeat_idx}_before')
        monitor.reset_peak_stats()
        
        t_start = time.time()  # 记录大步采样开始时间。
        with monitor.monitor_forward(
            model,
            (batch.protein_pos, batch.protein_atom_feature.float(), batch_protein,
             init_ligand_pos, init_ligand_v_input, batch_ligand),
            log_fn=logger.info if logger else None
        ):
            res = model.sample_diffusion_large_step(
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
                log_ligand_input_mode='log_prob' if log_mode == 'log_prob' else 'auto',
                repaint_cfg=template_repaint_cfg,
                **_dynamic_subcfg_grad_cap_kwargs(large_cfg),
            )
        t_end = time.time()  # 记录大步采样结束时间。
        time_records['large_step'].append(t_end - t_start)  # 记录大步采样耗时。
        profiler.checkpoint(f'large_step_repeat_{repeat_idx}_after')
        if logger and (
            'max_grad_fusion_iterations' in large_cfg
            or 'max_gradient_steps' in large_cfg
        ):
            _nti = res.get('time_indices')
            _cap_disp = large_cfg.get(
                'max_grad_fusion_iterations', large_cfg.get('max_gradient_steps')
            )
            logger.info(
                f'[Large step] 梯度融合迭代次数上限={_cap_disp} '
                f'→ 实际 len(time_indices)={len(_nti) if _nti is not None else 0} '
                f'（每项对应一次 for 循环 = 一次融合；非 t∈[0,999] 的「第30个时间标」）'
            )

        # 记录大步采样GPU监控信息
        if logger:
            mem_info = monitor.get_memory_info()
            logger.info(f'[Large Step] Repeat {repeat_idx} | Time: {t_end - t_start:.2f}s | Mem: {mem_info["allocated"]:.1f}/{mem_info["max_allocated"]:.1f} MB')
            try:
                log_gpu_monitor_record(
                    memory_info=mem_info,
                    forward_time=t_end - t_start,
                    sampling_info={
                        'mode': 'legacy_dynamic',
                        'stage': 'large_step',
                        'repeat_idx': repeat_idx,
                    },
                    logger=logger
                )
            except Exception as e:
                if logger:
                    logger.warning(f'Failed to log GPU monitor record for large step: {e}')

        # 提取大步采样结果并转换为 NumPy 数组。
        ligand_pos_array = res['pos'].detach().cpu().numpy().astype(np.float64)  # 配体位置数组。
        ligand_v_array = res['v'].detach().cpu().numpy()  # 配体类别索引数组。
        log_v_tensor = res['log_v'].detach().cpu()  # 配体类别对数概率张量。
        cum_atoms = np.cumsum([0] + ligand_num_atoms)  # 计算累积原子数边界，用于拆分批次结果。
        pos_traj_batch = res.get('pos_traj', []) or []
        log_v_traj_batch = res.get('log_v_traj', []) or []

        for idx in range(n_data):  # 拆分每个样本的结果。
            start, end = cum_atoms[idx], cum_atoms[idx + 1]
            pos_piece = ligand_pos_array[start:end]
            v_piece = ligand_v_array[start:end]
            log_v_piece = log_v_tensor[start:end]

            pos_traj_mol = [
                p[start:end].detach().cpu().numpy().astype(np.float64) for p in pos_traj_batch
            ]
            log_v_traj_mol = [
                lv[start:end].detach().cpu().numpy().astype(np.float32) for lv in log_v_traj_batch
            ]
            v_traj_mol = [x.argmax(axis=-1).astype(np.int64) for x in log_v_traj_mol]

            candidate = {
                'pos': pos_piece,
                'v': v_piece,
                'log_v': log_v_piece.numpy().astype(np.float32),
                'num_atoms': ligand_num_atoms[idx],
                'repeat': repeat_idx,
                'time_indices': res.get('time_indices'),
                'pos_traj': pos_traj_mol,
                'log_v_traj': log_v_traj_mol,
                'v_traj': v_traj_mol,
                'init_pos': init_ligand_pos_np[start:end].copy(),
                'template_init': (
                    dict(template_batch.records[idx])
                    if template_batch is not None else None
                ),
            }
            # large_step阶段不进行筛选，只收集候选
            total_candidates.append(candidate)  # 收集候选。

    # large_step阶段完成，不进行筛选（删除在time_boundary的筛选功能）
    if logger:
        logger.info(f'[Dynamic] Large-step completed | Total candidates: {len(total_candidates)}')

    # 实验3：largestep 完成后强制重建分子并记录 SMILES，不中断后续 refine/baseline_refine
    if dynamic_cfg.get('capture_largestep_smiles', False):
        for cand in total_candidates:
            pos_final = cand['pos']
            v_final = cand['v']
            smi = None
            try:
                v_tensor = torch.tensor(v_final, dtype=torch.long)
                atom_numbers = trans.get_atomic_number_from_index(v_tensor, mode=ligand_atom_mode)
                aromatic_flags = trans.is_aromatic_from_index(v_tensor, mode=ligand_atom_mode)
                mol = reconstruct.reconstruct_from_generated(
                    pos_final, atom_numbers, aromatic_flags,
                    basic_mode=(str(ligand_atom_mode) == 'basic'),
                )
                if mol is not None:
                    smi = Chem.MolToSmiles(mol)
            except Exception as exc:
                if logger:
                    logger.warning(f'[Ablation] largestep SMILES capture failed: {exc}')
            largestep_smiles_list.append(smi)
        if logger:
            n_ok = sum(1 for s in largestep_smiles_list if s)
            logger.info(f'[Ablation] Captured largestep SMILES: {n_ok}/{len(largestep_smiles_list)}')

    # 读取时间节点配置
    time_boundary = get_time_boundary(dynamic_cfg, 750)  # time_boundary用于划分large_step和refine
    enable_selection = selector_cfg.get('enable_selection', False)
    selection_time = selector_cfg.get('selection_time')
    
    # 确保refine使用time_boundary（原有功能）
    if 'time_upper' not in refine_cfg:
        refine_cfg['time_upper'] = time_boundary
    
    if logger:
        if enable_selection and selection_time is not None:
            logger.info(f'[Selector] Selection will be performed at t={selection_time} during refine stage')
        else:
            logger.info(f'[Selector] No selection, refine from t={time_boundary} to t=0')

    skip_refine = bool(dynamic_cfg.get('skip_refine', False))
    refined_records = []  # 存储精炼后的结果。
    if skip_refine:
        if logger:
            logger.info(
                '[Dynamic] skip_refine=True — 跳过后续 refine（含 selector 两阶段），'
                '仅使用 large_step 输出并进入下游（如 targetdiff_baseline_refine）'
            )
        for cand_idx, cand in enumerate(total_candidates):
            pos_final = cand['pos']
            v_final = cand['v']
            log_v_final = cand['log_v']
            try:
                num_atoms_rec = max(1, int(cand.get('num_atoms', 10)))
            except Exception:
                num_atoms_rec = 10
            metric_info = _selector_metrics_or_skip(
                pos_final, v_final, ligand_atom_mode, selector_cfg, enable_selection
            )
            refined_records.append({
                'pos': pos_final,
                'v': v_final,
                'log_v': log_v_final,
                'pos_traj': cand.get('pos_traj') or [],
                'v_traj': cand.get('v_traj') or [],
                'log_v_traj': cand.get('log_v_traj') or [],
                'num_atoms': num_atoms_rec,
                'source_index': cand_idx,
                'repeat_index': 0,
                'time_indices': cand.get('time_indices'),
                'init_pos': cand.get('init_pos'),
                'template_init': cand.get('template_init'),
                **metric_info
            })

    if not skip_refine:
        n_sampling = max(refine_cfg.get('n_sampling', 1), 1)  # 精炼次数。

        # 第一阶段refine：从time_boundary到selection_time（如果启用筛选）
        intermediate_candidates = total_candidates  # 初始候选集
        if enable_selection and selection_time is not None and selection_time < time_boundary:
            # 第一阶段 refine：time_boundary → selection_time；与 large_step 同类一次 batch 前向
            intermediate_refined = []
            stage1_flat_meta = [
                (cand_idx, refine_idx)
                for cand_idx, cand in enumerate(total_candidates)
                for refine_idx in range(n_sampling)
            ]
            stage1_cands = [total_candidates[cand_idx] for cand_idx, _ in stage1_flat_meta]
            if stage1_cands:
                raw_stage1 = _batch_refinement_forward_split_oom(
                    model,
                    data,
                    stage1_cands,
                    center_pos_mode=center_pos_mode,
                    pos_only=pos_only,
                    device=device,
                    time_upper=time_boundary,
                    time_lower=selection_time,
                    refine_cfg=refine_cfg,
                    refinement_grad_kwargs=_dynamic_subcfg_grad_cap_kwargs(refine_cfg),
                    error_prefix='[Dynamic] refine stage1 batch',
                    repaint_cfg=template_repaint_cfg,
                    max_chunk=refine_cfg.get('refine_batch_size', 100),
                    logger=logger,
                )
            else:
                raw_stage1 = []
            for raw, (cand_idx, refine_idx) in zip(raw_stage1, stage1_flat_meta):
                pos_intermediate = raw['pos']
                v_intermediate = raw['v']
                log_v_intermediate = raw['log_v']
                num_atoms = raw['num_atoms']
                _src_cand = total_candidates[cand_idx]
                intermediate_candidate = {
                    'pos': pos_intermediate,
                    'v': v_intermediate,
                    'log_v': log_v_intermediate,
                    'num_atoms': num_atoms,
                    'source_cand_idx': cand_idx,
                    'refine_idx': refine_idx,
                    'init_pos': _src_cand.get('init_pos'),
                    'template_init': _src_cand.get('template_init'),
                    'pos_traj': _src_cand.get('pos_traj') or [],
                    'v_traj': _src_cand.get('v_traj') or [],
                    'log_v_traj': _src_cand.get('log_v_traj') or [],
                    'time_indices': _src_cand.get('time_indices'),
                }
                metric_info = evaluate_candidate(
                    pos_intermediate, v_intermediate, ligand_atom_mode, selector_cfg
                )
                intermediate_candidate.update(metric_info)
                intermediate_refined.append(intermediate_candidate)

            # 在selection_time进行筛选
            top_percent = selector_cfg.get('top_percent')
            if top_percent is not None:
                top_n = max(1, int(len(intermediate_refined) * float(top_percent)))
            else:
                top_n = selector_cfg.get('top_n', len(intermediate_refined))
            selected_intermediate = select_top_candidates(intermediate_refined, top_n)
            if logger:
                logger.info(f'[Selector] Selection at t={selection_time} | Total: {len(intermediate_refined)} | Selected: {len(selected_intermediate)}')
            intermediate_candidates = selected_intermediate
        else:
            # 如果未启用筛选，直接使用large_step的候选
            intermediate_candidates = total_candidates

        def _large_traj_bundle_from_candidate(cand):
            """从 candidate 或 selection 的 source_cand_idx 取大步阶段轨道，与 refine 段拼接后供 step_trajectory 与 t 标签一致。"""
            if cand.get('pos_traj'):
                return (
                    cand.get('pos_traj') or [],
                    cand.get('v_traj') or [],
                    cand.get('log_v_traj') or [],
                    cand.get('time_indices'),
                )
            src = cand.get('source_cand_idx')
            if src is not None and isinstance(src, int) and 0 <= src < len(total_candidates):
                tc = total_candidates[src]
                return (
                    tc.get('pos_traj') or [],
                    tc.get('v_traj') or [],
                    tc.get('log_v_traj') or [],
                    tc.get('time_indices'),
                )
            return [], [], [], None

        # 第二阶段 refine：selection_time（或 time_boundary）→ 0；一次 batch 前向（含 n_sampling 条重复链）
        time_upper_stage2 = (
            selection_time
            if (enable_selection and selection_time is not None and selection_time < time_boundary)
            else (
                refine_cfg.get('time_upper')
                if 'time_upper' in refine_cfg
                else get_time_boundary(dynamic_cfg, 750)
            )
        )
        time_lower_stage2 = refine_cfg.get('time_lower', 0)
        stage2_flat_meta = [
            (cand_idx, refine_idx, cand)
            for cand_idx, cand in enumerate(intermediate_candidates)
            for refine_idx in range(n_sampling)
        ]
        stage2_cands = [cand for _, _, cand in stage2_flat_meta]
        if stage2_cands:
            profiler.checkpoint('refine_stage2_batch_before')
            monitor.reset_peak_stats()
            t_start = time.time()
            _mon_in = (torch.zeros(1, device=device),)
            with monitor.monitor_forward(
                model,
                _mon_in,
                log_fn=logger.info if logger else None,
            ):
                raw_stage2 = _batch_refinement_forward_split_oom(
                    model,
                    data,
                    stage2_cands,
                    center_pos_mode=center_pos_mode,
                    pos_only=pos_only,
                    device=device,
                    time_upper=time_upper_stage2,
                    time_lower=time_lower_stage2,
                    refine_cfg=refine_cfg,
                    refinement_grad_kwargs=_dynamic_subcfg_grad_cap_kwargs(refine_cfg),
                    error_prefix='[Dynamic] refine stage2 batch',
                    repaint_cfg=template_repaint_cfg,
                    max_chunk=refine_cfg.get('refine_batch_size', 100),
                    logger=logger,
                )
            t_end = time.time()
            time_records['refine'].append(t_end - t_start)
            profiler.checkpoint('refine_stage2_batch_after')
            if logger:
                mem_info = monitor.get_memory_info()
                n_flat = len(stage2_cands)
                logger.info(
                    f'[Refine] stage2 batch | chains={n_flat} | '
                    f'Time: {t_end - t_start:.2f}s | '
                    f'Mem: {mem_info["allocated"]:.1f}/{mem_info["max_allocated"]:.1f} MB'
                )
                try:
                    log_gpu_monitor_record(
                        memory_info=mem_info,
                        forward_time=t_end - t_start,
                        sampling_info={
                            'mode': 'legacy_dynamic',
                            'stage': 'refine',
                            'batch_chains': n_flat,
                            'time_upper': time_upper_stage2,
                        },
                        logger=logger,
                    )
                except Exception as e:
                    logger.warning(f'Failed to log GPU monitor record for refine batch: {e}')
        else:
            raw_stage2 = []

        for raw, (cand_idx, refine_idx, cand) in zip(raw_stage2, stage2_flat_meta):
            pos_final = raw['pos']
            v_final = raw['v']
            log_v_final = raw['log_v']
            num_atoms = raw['num_atoms']
            pos_traj_ref = raw['pos_traj_ref']
            log_v_traj_ref = raw['log_v_traj_ref']
            v_traj_ref = raw['v_traj_ref']
            rti = raw['rti']

            lpt, lvt, llv, lti = _large_traj_bundle_from_candidate(cand)
            merged_pos_traj = list(lpt) + pos_traj_ref
            merged_v_traj = list(lvt) + v_traj_ref
            merged_log_v_traj = list(llv) + log_v_traj_ref
            lt_part = [] if lti is None else (_coerce_time_indices_seq(lti) or [])
            rt_part = [] if rti is None else (_coerce_time_indices_seq(rti) or [])
            merged_time_indices = lt_part + rt_part

            metric_info = _selector_metrics_or_skip(
                pos_final, v_final, ligand_atom_mode, selector_cfg, enable_selection
            )
            _init_pos = cand.get('init_pos')
            _template_init = cand.get('template_init')
            if _init_pos is None:
                _src = cand.get('source_cand_idx', cand_idx)
                if isinstance(_src, int) and 0 <= _src < len(total_candidates):
                    _init_pos = total_candidates[_src].get('init_pos')
                    _template_init = total_candidates[_src].get(
                        'template_init'
                    )
            refined_records.append({
                'pos': pos_final,
                'v': v_final,
                'log_v': log_v_final,
                'pos_traj': merged_pos_traj,
                'v_traj': merged_v_traj,
                'log_v_traj': merged_log_v_traj,
                'num_atoms': num_atoms,
                'source_index': cand_idx,
                'repeat_index': refine_idx,
                'time_indices': merged_time_indices if merged_time_indices else rti,
                'init_pos': _init_pos,
                'template_init': _template_init,
                **metric_info
            })

    if logger:
        logger.info(f'[Dynamic] Refinement outputs: {len(refined_records)}')
        # 记录最终显存摘要
        summary = profiler.get_summary()
        logger.info(f'[Memory Summary] Peak: {summary["peak_memory_mb"]:.1f} MB')
        try:
            log_gpu_monitor_record(
                memory_info=monitor.get_memory_info(),
                memory_summary=summary,
                sampling_info={
                    'mode': 'legacy_dynamic',
                    'stage': 'final_summary',
                },
                logger=logger
            )
        except Exception as e:
            if logger:
                logger.warning(f'Failed to log final GPU monitor summary: {e}')

    refined_pos_list = [rec['pos'] for rec in refined_records]  # 汇总精炼坐标。
    refined_v_list = [rec['v'] for rec in refined_records]  # 汇总精炼类别。
    refined_pos_traj = [rec['pos_traj'] for rec in refined_records]  # 汇总轨迹。
    refined_v_traj = [rec['v_traj'] for rec in refined_records]

    # TargetDiff 基准扩散修复（可选）；optimization.enable 且将随后优化时延后在优化后执行
    meta_baseline_refine_ti = None
    if not skip_targetdiff_baseline_refine:
        _br_t0 = time.time()
        refine_out = apply_targetdiff_baseline_refinement(
            model, data, refined_pos_list, refined_v_list, config, device=device, logger=logger
        )
        time_records['baseline_refine'].append(time.time() - _br_t0)  # 记录 10 步修复 GPU 耗时
        if len(refine_out) == 5:
            refined_pos_list, refined_v_list, refine_pos_traj_add, refine_v_traj_add, refine_time_indices = refine_out
            refined_pos_traj = [list(pt) + list(rpt) for pt, rpt in zip(refined_pos_traj, refine_pos_traj_add)]
            refined_v_traj = [list(vt) + list(rvt) for vt, rvt in zip(refined_v_traj, refine_v_traj_add)]
            meta_baseline_refine_ti = refine_time_indices
        else:
            refined_pos_list, refined_v_list = refine_out

    meta_dict = {
        'large_step_candidates': total_candidates,
        'refined_candidates': refined_records,
        'time_records': time_records,
        'memory_summary': profiler.get_summary(),
        'dynamic_skip_refine': skip_refine,
        'large_step_max_grad_fusion_iterations': large_cfg.get(
            'max_grad_fusion_iterations', large_cfg.get('max_gradient_steps')
        ),
    }
    if template_geometry is not None:
        meta_dict['template_init'] = {
            'enabled': True,
            'schema_version': 1,
            'template_source': template_geometry.source,
            'scaffold_smarts': template_geometry.scaffold_smarts,
            'template_total_atom_count': template_geometry.total_atom_count,
            'scaffold_atom_count': template_geometry.scaffold_atom_count,
            'target_atom_count': template_geometry.target_atom_count,
            'diffusion_start_t': int(
                template_init_cfg.get('diffusion_start_t', 999)
            ),
            'target_side_elements_used': False,
            'target_side_bonds_used': False,
            'repaint_scaffold': template_repaint_cfg is not None,
            'repaint_scaffold_positions': bool(
                template_init_cfg.get('repaint_scaffold_positions', True)
            ),
            'repaint_scaffold_types': bool(
                template_init_cfg.get('repaint_scaffold_types', True)
            ),
        }
        meta_dict['template_init_records'] = [
            rec.get('template_init') for rec in refined_records
        ]
    if dynamic_cfg.get('capture_largestep_smiles', False) and largestep_smiles_list:
        meta_dict['largestep_smiles'] = largestep_smiles_list  # 实验3 专表用
    if meta_baseline_refine_ti is not None:
        meta_dict['baseline_refine_time_indices'] = meta_baseline_refine_ti
    # Flatten time records to match baseline expectation
    time_list = (
        time_records['large_step'] + time_records['refine'] + time_records['baseline_refine']
    )  # 合并耗时记录（含 10 步修复）。

    _ls_s = float(sum(time_records.get('large_step', [])) or 0.0)
    _rf_s = float(sum(time_records.get('refine', [])) or 0.0)
    _bs_s = float(sum(time_records.get('baseline_refine', [])) or 0.0)
    _gpu_s = _ls_s + _rf_s
    _cpu_s = 0.0
    _tot_s = _gpu_s + _bs_s
    meta_dict['timing'] = {
        'large_step_s': round(_ls_s, 2),
        'refine_s': round(_rf_s, 2),
        'gpu_s': round(_gpu_s, 2),
        'cpu_s': _cpu_s,
        'baseline_s': round(_bs_s, 2),
        'total_s': round(_tot_s, 2),
    }
    if logger:
        logger.info(
            f'[Timing] gpu={_gpu_s:.1f}s cpu={_cpu_s:.1f}s baseline={_bs_s:.1f}s '
            f'total={_tot_s:.1f}s n_mol={len(refined_pos_list)} '
            f'large_step={_ls_s:.1f}s refine={_rf_s:.1f}s'
        )

    return {
        'pos_list': refined_pos_list,
        'v_list': refined_v_list,
        'pos_traj': refined_pos_traj,
        'v_traj': refined_v_traj,
        'log_v_traj': [rec['log_v_traj'] for rec in refined_records],
        'time_list': time_list,
        'meta': meta_dict,
        'init_list': [rec.get('init_pos') for rec in refined_records],
    }  # 返回传统动态采样结果。


def sample_dynamic_diffusion_ligand(
    model, data, config, ligand_atom_mode, device='cuda:0', logger=None,
    force_method=None, skip_targetdiff_baseline_refine=False,
    segment_checkpoint_context=None,
    resume_from_pt=None,
    resume_frame=None,
    target_frame=None,
):
    """封装动态采样入口，根据配置自动选择统一或旧版策略。

    Args:
        model: 扩散模型实例。
        data: 单个蛋白口袋样本。
        config: 采样配置，需包含 `sample.dynamic.method`。
        ligand_atom_mode: 配体原子编码模式。
        device: 运行设备。
        logger: 可选日志器。
        force_method: 若提供（如 'legacy'），则强制使用该模式，忽略 config 中的 method。
            用于 dynamic_then_optimization 时强制两阶段流程，使分子数由 large_step/refine 控制。
        skip_targetdiff_baseline_refine: 为 True 时跳过 targetdiff_baseline_refine。
            当后续还有 optimization / scaffold 等阶段时须为 True，使 2b 在下游阶段全部结束后再统一执行
            （见 apply_targetdiff_baseline_refine_to_sampling_result）。

    Returns:
        dict: 动态采样输出，结构与具体实现 `_run_*` 一致。
    """
    from diffdynamic.sampling.configuration import get_time_boundary
    from diffdynamic.sampling.dynamic_unified import _run_unified_dynamic
    from diffdynamic.sampling.prudent.legacy import _run_prudent_legacy_dynamic

    dynamic_cfg = config.sample.get('dynamic', {})  # 读取动态配置。
    dynamic_method = force_method if force_method is not None else dynamic_cfg.get('method', 'auto')  # 指定方法。

    supports_unified = hasattr(model, 'dynamic_sample_diffusion')  # 检查模型是否实现统一动态接口。
    if dynamic_method == 'auto':
        dynamic_method = 'unified' if supports_unified else 'legacy'  # 自动选择。

    # 在调用具体实现之前，先更新模型的默认值，确保采样配置覆盖检查点中的训练配置
    # 处理 time_boundary：如果存在，同步到 large_step.time_lower 和 refine.time_upper（向后兼容）
    time_boundary = get_time_boundary(dynamic_cfg, None)
    if time_boundary is not None:
        # 同步到 large_step.time_lower（如果不存在）
        if 'large_step' in dynamic_cfg:
            if 'time_lower' not in dynamic_cfg['large_step']:
                dynamic_cfg['large_step']['time_lower'] = time_boundary
        # 同步到 refine.time_upper（如果不存在）
        if 'refine' in dynamic_cfg:
            if 'time_upper' not in dynamic_cfg['refine']:
                dynamic_cfg['refine']['time_upper'] = time_boundary

    if 'large_step' in dynamic_cfg:
        # 保存原始默认值（如果需要恢复）
        original_large_step_defaults = getattr(model, 'dynamic_large_step_defaults', {})
        # 更新为 sampling.yml 中的配置，采样配置优先覆盖训练配置
        model.dynamic_large_step_defaults = {**original_large_step_defaults, **dynamic_cfg['large_step']}
        if logger:
            logger.info(f'Updated model.dynamic_large_step_defaults: {model.dynamic_large_step_defaults}')
    if 'refine' in dynamic_cfg:
        # 保存原始默认值（如果需要恢复）
        original_refine_defaults = getattr(model, 'dynamic_refine_defaults', {})
        # 更新为 sampling.yml 中的配置，采样配置优先覆盖训练配置
        model.dynamic_refine_defaults = {**original_refine_defaults, **dynamic_cfg['refine']}
        if logger:
            logger.info(f'Updated model.dynamic_refine_defaults: {model.dynamic_refine_defaults}')

    prudent_on = (
        config.sample.get('mode') == 'prudent'
        or dynamic_cfg.get('prudent', {}).get('enable', False)
    )
    if prudent_on:
        # 限制 OpenMP/PyTorch 线程数：128 核过订阅会拖慢线程池里的并行 Vina（vina/meeko/openbabel 各单线程）
        try:
            _n_thr = int(dynamic_cfg.get('prudent', {}).get('torch_num_threads', 8) or 8)
            torch.set_num_threads(max(1, _n_thr))
            if logger:
                logger.info(f'[Sample] prudent 模式：torch.set_num_threads({max(1, _n_thr)})')
        except Exception:
            pass
        if logger:
            if dynamic_cfg.get('prudent', {}).get('segment_evolution_mode'):
                logger.info(
                    '[Sample] Prudent 分段演化：large_step → 段 k reverse 梯度融合上限 = '
                    'max(1, refine.max_grad_fusion_iterations − k×checkpoint_every_grad_fusions)→ 可选 '
                    'segment_pre_baseline_forward_t → targetdiff_baseline_refine → Vina/QED+SA'
                )
            else:
                logger.info('[Sample] Prudent 模式：large_step → 多轮 refine（Vina+QED+SA）→ targetdiff_baseline_refine（若启用）')
        return _run_prudent_legacy_dynamic(
            model, data, config, ligand_atom_mode,
            device=device, logger=logger,
            skip_targetdiff_baseline_refine=skip_targetdiff_baseline_refine,
            segment_checkpoint_context=segment_checkpoint_context,
        )

    if dynamic_method == 'unified':
        if not supports_unified:
            raise RuntimeError('dynamic.method is set to "unified" but model does not implement dynamic_sample_diffusion().')
        return _run_unified_dynamic(
            model, data, config, device=device, logger=logger,
            skip_targetdiff_baseline_refine=skip_targetdiff_baseline_refine,
        )  # 调用统一动态。

    return _run_legacy_dynamic(
        model, data, config, ligand_atom_mode, device=device, logger=logger,
        skip_targetdiff_baseline_refine=skip_targetdiff_baseline_refine,
    )  # fallback。
