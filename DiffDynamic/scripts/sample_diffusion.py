# -*- coding: utf-8 -*-
# 总结：
# - 根据配置执行扩散模型的统一动态或传统动态采样流程，生成配体 3D 结构及原子类型。
# - 支持温度控制、原子数量策略、候选筛选与精细化采样，最终保存生成结果和日志。
# - 输出包含轨迹、化学指标与采样耗时的详细记录，方便后续评估或对接。

import argparse  # 导入 argparse，解析命令行参数。
from concurrent.futures import ThreadPoolExecutor  # Prudent 多 seed 时跨 seed 重叠 Vina 与 GPU
import os  # 导入 os，用于路径操作。
import threading  # Prudent 多线程对接时串行化 Vina（SIGALRM 非线程安全）
import shutil  # 导入 shutil，用于复制文件/目录。
import subprocess  # 导入 subprocess，用于执行外部脚本。
import sys  # 导入 sys，用于获取Python解释器路径。
import time  # 导入 time，记录采样耗时。
from datetime import datetime, timezone, timedelta  # 导入 datetime，用于生成时间戳文件名。
from pathlib import Path  # 方便地解析仓库根目录。

# 将仓库根目录加入 sys.path，防止相对运行脚本时找不到 utils 等模块。
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_PRUDENT_VINA_THREAD_LOCK = threading.Lock()

import numpy as np  # 导入 NumPy，处理数组。
import torch  # 导入 PyTorch，执行模型推断。
import torch.nn.functional as F  # 导入函数式接口，用于 softmax 等操作。
from rdkit import Chem  # 导入 RDKit，用于分子重建与处理。
from rdkit import RDLogger  # 导入 RDKit 日志控制器
# 屏蔽 RDKit 的警告信息（如 Explicit valence）
RDLogger.DisableLog('rdApp.warning')
from torch_geometric.data import Batch  # 导入 PyG Batch，构建图批次。
from torch_geometric.transforms import Compose  # 导入转换组合工具。
from torch_scatter import scatter_sum, scatter_mean  # 导入分散聚合函数。
from tqdm.auto import tqdm  # 导入 tqdm，显示进度条。

import utils.misc as misc  # 导入通用工具（日志、配置等）。
import utils.transforms as trans  # 导入特征转换工具。
from datasets import get_dataset  # 导入数据集工厂函数。
from datasets.pl_data import FOLLOW_BATCH, ProteinLigandData, torchify_dict  # 导入 PyG follow_batch 配置。
from utils.data import PDBProtein, parse_sdf_file, rdmol_to_ligand_dict  # 导入蛋白/配体解析工具。
from models.molopt_score_model import (  # 导入模型及辅助函数。
    ScorePosNet3D,
    DiffDynamic,
    GRAD_FUSION_CAP_UNSPECIFIED,
    log_sample_categorical,
    index_to_log_onehot,
    extract,
    center_pos,
    ensure_log_ligand,
)  # ensure_log_ligand：prudent 前向加噪时与 refine 输入一致。
from utils.evaluation import atom_num, scoring_func  # 导入原子数量采样与化学评分函数。
import utils.reconstruct as reconstruct  # 导入分子重建工具。
from utils.monitor import GPUMonitor, MemoryProfiler  # 导入GPU监控工具。
from utils.gpu_monitor_recorder import log_gpu_monitor_record  # 导入GPU监控记录器。
from utils.sampling_recorder import extract_sampling_params, log_sampling_record  # 导入采样记录工具。
from show_sampling_steps import generate_sampling_steps_text  # 导入采样步骤生成函数。
from utils.molecule_id import extract_protein_id, generate_molecule_id  # 分子身份证（不拉取 evaluate/meeko）


def _dynamic_subcfg_grad_cap_kwargs(subcfg):
    """large_step / refine 子配置 → sample_diffusion_* 的梯度融合上限参数。

    仅当 YAML 显式写出键时才传入具体值；未写键用 GRAD_FUSION_CAP_UNSPECIFIED 走该段 defaults。
    显式 ``null`` 表示不截断（跑满调度），与「未写键」区分。
    """
    return {
        'max_grad_fusion_iterations': (
            subcfg['max_grad_fusion_iterations']
            if 'max_grad_fusion_iterations' in subcfg
            else GRAD_FUSION_CAP_UNSPECIFIED
        ),
        'max_gradient_steps': (
            subcfg['max_gradient_steps']
            if 'max_gradient_steps' in subcfg
            else GRAD_FUSION_CAP_UNSPECIFIED
        ),
    }


def get_time_boundary(dynamic_cfg, default=750):
    """获取阶段边界时间步，支持向后兼容
    
    优先读取 time_boundary，如果没有则从 large_step.time_lower 或 refine.time_upper 获取
    
    Args:
        dynamic_cfg: dynamic 配置字典
        default: 默认值
    
    Returns:
        int: 阶段边界时间步
    """
    # 优先读取统一的 time_boundary
    if 'time_boundary' in dynamic_cfg:
        return dynamic_cfg.get('time_boundary', default)
    
    # 向后兼容：从 large_step.time_lower 获取
    large_step_cfg = dynamic_cfg.get('large_step', {})
    if 'time_lower' in large_step_cfg:
        return large_step_cfg.get('time_lower', default)
    
    # 向后兼容：从 refine.time_upper 获取
    refine_cfg = dynamic_cfg.get('refine', {})
    if 'time_upper' in refine_cfg:
        return refine_cfg.get('time_upper', default)
    
    return default


def generate_eval_dir_name(data_id, config, timestamp=None):
    """生成包含所有配置参数的评估目录名
    
    格式: eval_{timestamp}_{data_id}_{grad_fusion_mode}_{start}_{end}_{time_lower}_{large_step_params}_{refine_params}
    
    Args:
        data_id: 数据ID
        config: 配置对象
        timestamp: 时间戳（可选，格式: YYYYMMDD_HHMMSS）
    
    Returns:
        str: 目录名
    """
    # ✅ 时间戳放在最前面（在eval_之后，data_id之前）
    if timestamp:
        parts = [f'eval_{timestamp}_{data_id}']
    else:
        # 如果没有提供时间戳，使用当前时间（本地时区CST，UTC+8）
        cst = timezone(timedelta(hours=8))
        timestamp = datetime.now(cst).strftime('%Y%m%d_%H%M%S')
        parts = [f'eval_{timestamp}_{data_id}']
    
    # Grad Fusion Lambda 参数
    grad_fusion_cfg = getattr(config.model, 'grad_fusion_lambda', None)
    if isinstance(grad_fusion_cfg, dict):
        mode = str(grad_fusion_cfg.get('mode', 'none'))
        start = str(grad_fusion_cfg.get('start', 0))
        end = str(grad_fusion_cfg.get('end', 0))
        parts.append(f'gf{mode}_{start}_{end}')
    else:
        parts.append('gfnone_0_0')
    
    # Dynamic 采样参数
    dynamic_cfg = config.sample.get('dynamic', {})
    large_step_cfg = dynamic_cfg.get('large_step', {})
    refine_cfg = dynamic_cfg.get('refine', {})
    
    # time_boundary（支持向后兼容）
    time_boundary = get_time_boundary(dynamic_cfg, 750)
    parts.append(f'tl{time_boundary}')
    
    # Large Step 参数
    large_schedule = str(large_step_cfg.get('schedule', 'none'))
    if large_schedule == 'lambda':
        large_a = str(large_step_cfg.get('lambda_coeff_a', 0))
        large_b = str(large_step_cfg.get('lambda_coeff_b', 0))
        parts.append(f'ls{large_schedule}_{large_a}_{large_b}')
    elif large_schedule == 'linear':
        large_upper = str(large_step_cfg.get('linear_step_upper', 0))
        large_lower = str(large_step_cfg.get('linear_step_lower', 0))
        parts.append(f'ls{large_schedule}_{large_upper}_{large_lower}')
    else:
        parts.append(f'ls{large_schedule}')
    
    # Refine 参数
    refine_schedule = str(refine_cfg.get('schedule', 'none'))
    if refine_schedule == 'lambda':
        refine_a = str(refine_cfg.get('lambda_coeff_a', 0))
        refine_b = str(refine_cfg.get('lambda_coeff_b', 0))
        parts.append(f'rf{refine_schedule}_{refine_a}_{refine_b}')
    elif refine_schedule == 'linear':
        refine_upper = str(refine_cfg.get('linear_step_upper', 0))
        refine_lower = str(refine_cfg.get('linear_step_lower', 0))
        parts.append(f'rf{refine_schedule}_{refine_upper}_{refine_lower}')
    else:
        parts.append(f'rf{refine_schedule}')
    
    dir_name = '_'.join(parts)
    # 替换可能不适合文件名的字符
    # 将浮点数中的点号替换为p（如1.0 -> 1p0），负号替换为m（如-1 -> m1）
    dir_name = dir_name.replace('.', 'p').replace('-', 'm')
    return dir_name


def resolve_and_set_absolute_protein_path(data, dataset_root=None, protein_root=None, logger=None):
    """将 `data.protein_filename` 解析为磁盘上存在的绝对 PDB 路径。

    与 `evaluate_pt_with_correct_reconstruct` / `VinaDockingTask.from_generated_mol` 一致：
    1) 已是可访问文件则规范为绝对路径；
    2) 相对路径则相对训练数据根目录 ``dataset_root``（即检查点里 ``data.path``）拼接；
    3) 仍失败则按 CrossDocked 惯例，用 ``protein_root`` + ``dirname(ligand_filename)`` +
       ``basename(ligand_filename)[:10] + '.pdb'`` 推断受体（同 docking_vina.from_generated_mol）；
    4) 再尝试在各 root 下直接拼接 ``protein_filename`` 相对路径。

    Args:
        data: ProteinLigandData
        dataset_root: 数据集原始目录（含 index / 口袋 PDB 的父目录）
        protein_root: 与评估时 ``--protein_root`` 相同，用于按配体路径推断受体
        logger: 可选
    """
    if data is None:
        return
    pf = getattr(data, 'protein_filename', None)
    lig_fn = getattr(data, 'ligand_filename', None)
    env_root = os.environ.get('PROTEIN_ROOT', '').strip() or None

    def _iter_roots():
        for r in (protein_root, env_root, dataset_root):
            if not r:
                continue
            try:
                yield str(Path(r).expanduser().resolve())
            except Exception:
                yield str(r)

    if pf:
        ps = str(pf)
        if os.path.isfile(ps):
            data.protein_filename = str(Path(ps).resolve())
            return
        if dataset_root:
            try:
                joined = str((Path(dataset_root).expanduser().resolve() / ps))
            except Exception:
                joined = os.path.normpath(os.path.join(str(dataset_root), ps))
            if os.path.isfile(joined):
                data.protein_filename = str(Path(joined).resolve())
                if logger:
                    logger.info(f'[Protein path] 相对路径已解析（相对 data.path）: {data.protein_filename}')
                return

    if lig_fn:
        lig_s = str(lig_fn)
        sub = os.path.join(os.path.dirname(lig_s), os.path.basename(lig_s)[:10] + '.pdb')
        for root in _iter_roots():
            alt = os.path.normpath(os.path.join(root, sub))
            if os.path.isfile(alt):
                data.protein_filename = str(Path(alt).resolve())
                if logger:
                    logger.info(f'[Protein path] 按配体+protein_root 推断受体: {data.protein_filename}')
                return

    if pf and not os.path.isabs(str(pf)):
        for root in _iter_roots():
            alt = os.path.normpath(os.path.join(root, str(pf)))
            if os.path.isfile(alt):
                data.protein_filename = str(Path(alt).resolve())
                if logger:
                    logger.info(f'[Protein path] 在数据根下找到: {data.protein_filename}')
                return

    if logger:
        _log(
            logger,
            'warning',
            f'[Protein path] 未解析到有效 PDB（protein_filename={pf!r}, ligand_filename={lig_fn!r}）。'
            f' Prudent/Vina 需传 --protein_root、环境变量 PROTEIN_ROOT，'
            f'或 sampling.yml 中 sample.dynamic.prudent.protein_root（与评估一致）。',
        )


def load_custom_pocket_data(protein_path, ligand_path=None, transform=None,
                            pocket_radius=10.0, logger=None):
    """
    从自定义蛋白/配体文件加载数据，用于生成分子。
    
    当提供配体时，自动将蛋白裁剪为配体周围 pocket_radius Å 的口袋，
    以匹配训练数据（crossdocked_v1.1_rmsd1.0_pocket10）的口袋尺度，
    避免因全蛋白输入导致生成碎片化分子。
    
    Args:
        protein_path: 蛋白 PDB 文件路径（必需，可以是全蛋白或预裁剪口袋）
        ligand_path: 参考配体 SDF/MOL2 文件路径（可选，用于自动裁剪口袋和 ref 模式原子数）
        transform: 特征转换管道（可选）
        pocket_radius: 口袋裁剪半径（Å），仅在提供配体时生效（默认: 10.0，匹配训练数据）
        logger: 日志记录器（可选）
    
    Returns:
        ProteinLigandData: 转换后的图数据对象
    """
    protein_path = Path(protein_path).resolve()
    if not protein_path.exists():
        raise FileNotFoundError(f'蛋白文件不存在: {protein_path}')
    
    protein_obj = PDBProtein(str(protein_path))
    full_atom_count = len(protein_obj.element)
    
    # 解析配体
    if ligand_path is not None:
        ligand_path = Path(ligand_path).resolve()
        if not ligand_path.exists():
            raise FileNotFoundError(f'配体文件不存在: {ligand_path}')
        ligand_dict = parse_sdf_file(str(ligand_path))
    else:
        ligand_dict = {
            'element': np.array([], dtype=np.int64),
            'pos': np.empty((0, 3), dtype=np.float32),
            'atom_feature': np.empty((0, 8), dtype=np.int64),
            'bond_index': np.empty((2, 0), dtype=np.int64),
            'bond_type': np.array([], dtype=np.int64),
        }
    
    # 自动口袋裁剪：当提供配体且 pocket_radius > 0 时，
    # 提取配体周围 pocket_radius Å 内的残基作为口袋
    if ligand_path is not None and pocket_radius > 0:
        selected_residues = protein_obj.query_residues_ligand(
            ligand_dict, radius=pocket_radius
        )
        if selected_residues:
            atom_indices = sorted(set(
                idx for res in selected_residues for idx in res['atoms']
            ))
            full_dict = protein_obj.to_dict_atom()
            pocket_dict = {
                'element': full_dict['element'][atom_indices],
                'molecule_name': full_dict.get('molecule_name', 'POCKET'),
                'pos': full_dict['pos'][atom_indices],
                'is_backbone': full_dict['is_backbone'][atom_indices],
                'atom_name': [full_dict['atom_name'][i] for i in atom_indices],
                'atom_to_aa_type': full_dict['atom_to_aa_type'][atom_indices],
            }
            pocket_atom_count = len(atom_indices)
            _log(logger, 'info',
                 f'自动裁剪口袋: {full_atom_count} → {pocket_atom_count} 原子 '
                 f'(配体周围 {pocket_radius}Å, {len(selected_residues)} 个残基)')
        else:
            pocket_dict = protein_obj.to_dict_atom()
            _log(logger, 'warning',
                 f'配体 {pocket_radius}Å 内未找到残基，使用完整蛋白 ({full_atom_count} 原子)。'
                 f'请检查蛋白和配体的坐标系是否一致。')
    else:
        pocket_dict = protein_obj.to_dict_atom()
        if full_atom_count > 500:
            _log(logger, 'warning',
                 f'蛋白原子数较多 ({full_atom_count})，可能不是裁剪过的口袋。'
                 f'建议提供参考配体 (--ligand_path) 以自动裁剪为结合位点口袋，'
                 f'或手动提取结合位点附近 10Å 的口袋 PDB。'
                 f'全蛋白输入可能导致生成碎片化分子。')
        elif logger:
            logger.info(f'蛋白原子数: {full_atom_count} (未提供配体，跳过口袋裁剪)')
    
    data = ProteinLigandData.from_protein_ligand_dicts(
        protein_dict=torchify_dict(pocket_dict),
        ligand_dict=torchify_dict(ligand_dict),
    )
    data.protein_filename = str(protein_path)
    data.ligand_filename = str(ligand_path) if ligand_path is not None else 'N/A'
    assert data.protein_pos.size(0) > 0, '蛋白口袋为空'
    
    if transform is not None:
        data = transform(data)
    return data


def _log(logger, level, msg):
    """统一日志输出：有 logger 用 logger，否则 print。"""
    if logger:
        getattr(logger, level)(msg)
    elif level == 'warning':
        print(f'⚠️  {msg}')
    else:
        print(msg)


def safe_repeat_interleave(indices, repeats, device=None):
    """Repeat indices by ``repeats`` using vectorized ops with strict validation.

    This replaces the earlier Python-loop implementation which silently produced
    degenerate results on some CUDA setups.
    """
    if isinstance(device, str):
        device = torch.device(device)
    if device is None and isinstance(indices, torch.Tensor):
        device = indices.device
    if device is None:
        device = torch.device('cpu')

    indices_tensor = torch.as_tensor(indices, dtype=torch.long, device='cpu')
    repeats_tensor = torch.as_tensor(repeats, dtype=torch.long, device='cpu')

    if indices_tensor.numel() != repeats_tensor.numel():
        raise ValueError(
            f"safe_repeat_interleave: indices and repeats size mismatch. "
            f"indices.numel()={indices_tensor.numel()}, repeats.numel()={repeats_tensor.numel()}"
        )

    if torch.any(repeats_tensor < 0):
        raise ValueError(
            f"safe_repeat_interleave: repeats must be non-negative. repeats={repeats_tensor.tolist()}"
        )
    repeats_tensor = torch.clamp(repeats_tensor, min=0)

    total = int(repeats_tensor.sum().item())
    if total == 0:
        return torch.empty(0, dtype=torch.long, device=device)

    result_tensor = torch.repeat_interleave(indices_tensor, repeats_tensor)
    return result_tensor.to(device)


def _validate_batch_indices(batch_tensor, n_data, name, logger=None, context=''):
    """Ensure batch indices cover [0, n_data-1] exactly once per sample."""
    if batch_tensor.numel() == 0:
        raise ValueError(f"{name} is empty in context={context}. n_data={n_data}")

    batch_cpu = batch_tensor.detach().cpu()
    sorted_unique = torch.sort(batch_cpu.unique()).values
    if sorted_unique.numel() != n_data:
        msg = (
            f"{name} unique size mismatch in context={context}. "
            f"expected {n_data} unique entries (0..{n_data-1}), "
            f"got {sorted_unique.numel()}: {sorted_unique.tolist()}"
        )
        if logger:
            logger.error(msg)
        raise ValueError(msg)

    expected = torch.arange(n_data, device=sorted_unique.device)
    if not torch.equal(sorted_unique, expected):
        msg = (
            f"{name} indices incorrect in context={context}. "
            f"expected {expected.tolist()}, got {sorted_unique.tolist()}"
        )
        if logger:
            logger.error(msg)
        raise ValueError(msg)


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
            'model_meta': result.get('meta')
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
        'meta': meta_dict
    }  # 返回采样结果。


def unbatch_v_traj(ligand_v_traj, n_data, ligand_cum_atoms):  # 根据累积原子数拆分类别轨迹。
    """按样本原子数量切分联合轨迹，用于还原逐分子的预测序列。

    Args:
        ligand_v_traj: 形状为 `[num_steps, total_atoms, num_types]` 的时间序列列表。
        n_data: 当前批次的样本个数。
        ligand_cum_atoms: 按样本累计的原子数边界（`[0, n1, n1+n2, ...]`）。

    Returns:
        list[list[np.ndarray]]: 长度为 `n_data` 的列表，每项为该样本的逐步类别轨迹。
    """
    all_step_v = [[] for _ in range(n_data)]  # 初始化每个样本的轨迹列表。
    for v in ligand_v_traj:  # 遍历每个时间步的类别分布。
        v_array = v.cpu().numpy()  # 转为 NumPy 数组。
        for k in range(n_data):  # 逐样本切片。
            all_step_v[k].append(v_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]])
    all_step_v = [np.stack(step_v) for step_v in all_step_v]  # num_samples * [num_steps, num_atoms_i]
    return all_step_v  # 返回拆分后的轨迹列表。


def split_tensor_by_counts(tensor, counts):  # 按原子数量拆分张量。
    """依据原子数统计将拼接张量拆分为若干子张量。

    Args:
        tensor: 已按所有样本拼接的张量（`[total_atoms, ...]`）。
        counts: 每个样本对应的原子数量列表。

    Returns:
        list[torch.Tensor]: 逐样本拆分后的张量列表。
    """
    splits = []  # 保存片段。
    start = 0  # 起始索引，用于标记当前样本在拼接张量中的起始位置。
    for count in counts:  # 遍历每个样本原子数。
        end = start + count
        splits.append(tensor[start:end])  # 提取对应片段。
        start = end
    return splits  # 返回拆分结果。


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
        当 step_trajectory_save.enable 时返回五元组，否则返回二元组。
    """
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

    record_traj = config.sample.get('step_trajectory_save', {}).get('enable', False)
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
    refine_out = apply_targetdiff_baseline_refinement(
        model, data, pos_list, v_list, config, device=device, logger=logger
    )
    if len(refine_out) >= 2:
        refined_pos_list, refined_v_list = refine_out[0], refine_out[1]
    else:
        refined_pos_list, refined_v_list = refine_out
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
    refine_out = apply_targetdiff_baseline_refinement(
        model, data, pos_list, v_list, config, device=device, logger=logger
    )
    meta = result.get('meta')
    if not isinstance(meta, dict):
        meta = {}
        result['meta'] = meta
    if len(refine_out) == 5:
        pos_list, v_list, refine_pos_traj, refine_v_traj, refine_time_indices = refine_out
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
        result['pred_ligand_pos'] = pos_list
        result['pred_ligand_v'] = v_list


def evaluate_candidate(pos_array, v_array, ligand_atom_mode, selector_cfg):  # 评估候选分子的化学指标。
    """根据采样结果评估分子质量，计算化学指标并返回评分。

    Args:
        pos_array: 形状为 `[num_atoms, 3]` 的坐标数组。
        v_array: 原子类型索引数组。
        ligand_atom_mode: 配体原子编码模式（与训练配置保持一致）。
        selector_cfg: 筛选配置，包含权重及阈值。

    Returns:
        dict: 包含化学指标、SMILES、筛选状态及综合评分。
    """
    metrics = {}  # 初始化化学指标字典。
    smiles = None  # 初始 SMILES。
    status = 'ok'  # 状态标记。
    score_value = float('inf')  # 评分值（越小越好）。

    try:
        v_tensor = torch.tensor(v_array, dtype=torch.long)  # 将类别数组转为张量。
        atom_numbers = trans.get_atomic_number_from_index(v_tensor, mode=ligand_atom_mode)  # 获取原子序号。
        aromatic_flags = trans.is_aromatic_from_index(v_tensor, mode=ligand_atom_mode)  # 获取芳香标记。
        mol = reconstruct.reconstruct_from_generated(pos_array, atom_numbers, aromatic_flags)  # 重建分子。
        if mol is not None:
            smiles = Chem.MolToSmiles(mol)  # 转换为 SMILES。
            # 分子不完整：SMILES 含 '.' 表示多个不连通片段，直接过滤
            if selector_cfg.get('filter_incomplete', True) and smiles and '.' in smiles:
                status = 'incomplete'
                metrics = scoring_func.get_chem(mol)  # 仍计算指标供日志
                score_value = float('inf')
            else:
                metrics = scoring_func.get_chem(mol)  # 计算化学指标（QED/SA）。
                qed = metrics.get('qed', float('nan'))  # 提取 QED。
                sa = metrics.get('sa', float('inf'))  # 提取 SA。
                qed_weight = selector_cfg.get('qed_weight', 1.0)  # QED 权重。
                sa_weight = selector_cfg.get('sa_weight', 1.0)  # SA 权重。
                if np.isnan(qed) or np.isnan(sa):  # 若指标非法，则标记无效。
                    score_value = float('inf')
                    status = 'metric_nan'
                else:
                    # SA和QED都越大越好，所以score = -(sa_weight * sa + qed_weight * qed)，score越小越好
                    score_value = -(sa_weight * sa + qed_weight * qed)  # 根据权重计算综合分数。
        else:
            status = 'mol_none'  # 重建返回空。
            metrics = {'qed': float('nan'), 'sa': float('inf')}
    except reconstruct.MolReconsError:
        status = 'reconstruct_failed'  # 重建抛出特定错误。
        metrics = {'qed': float('nan'), 'sa': float('inf')}
        score_value = float('inf')
    except Exception as exc:
        status = f'error:{exc.__class__.__name__}'  # 捕获其他异常。
        metrics = {'qed': float('nan'), 'sa': float('inf')}
        score_value = float('inf')

    min_qed = selector_cfg.get('min_qed')  # QED 下限。
    # SA越大越好，所以使用min_sa作为下限；为了向后兼容，也支持max_sa（但会转换为min_sa的语义）
    min_sa = selector_cfg.get('min_sa')
    max_sa = selector_cfg.get('max_sa')  # 向后兼容：如果配置了max_sa但没有min_sa，则使用max_sa作为min_sa
    if min_sa is None and max_sa is not None:
        min_sa = max_sa  # 向后兼容：将max_sa当作min_sa使用
    if metrics:
        qed = metrics.get('qed', float('nan'))
        sa = metrics.get('sa', float('inf'))
        if min_qed is not None and not np.isnan(qed) and qed < min_qed:  # 不满足 QED 下限则过滤。
            status = 'filtered_qed'
            score_value = float('inf')
        if min_sa is not None and not np.isnan(sa) and sa < min_sa:  # 不满足 SA 下限则过滤（SA越大越好）。
            status = 'filtered_sa'
            score_value = float('inf')

    return {
        'metrics': metrics,
        'smiles': smiles,
        'status': status,
        'score': score_value
    }  # 返回评估结果。


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


def _prudent_reconstruct_mol(pos_array, v_array, ligand_atom_mode, logger=None, context=''):
    """重建 RDKit Mol；失败返回 None。"""
    try:
        # 记录输入参数
        if logger:
            logger.debug(
                f'[_prudent_reconstruct_mol] {context} | '
                f'pos_shape={pos_array.shape}, '
                f'v_shape={v_array.shape}, '
                f'v_dtype={v_array.dtype}, '
                f'v_range=[{v_array.min()}, {v_array.max()}], '
                f'unique_v={len(np.unique(v_array))}'
            )
        
        # 检查 v_array 是否为整数类型
        if not np.issubdtype(v_array.dtype, np.integer):
            if logger:
                logger.warning(
                    f'[_prudent_reconstruct_mol] {context} | '
                    f'v_array is not integer type (dtype={v_array.dtype}), attempting conversion'
                )
            v_array = v_array.astype(np.int64)
        
        v_tensor = torch.tensor(v_array, dtype=torch.long)
        atom_numbers = trans.get_atomic_number_from_index(v_tensor, mode=ligand_atom_mode)
        aromatic_flags = trans.is_aromatic_from_index(v_tensor, mode=ligand_atom_mode)
        
        if logger:
            # get_atomic_number_from_index / is_aromatic_from_index 返回 list（或 aromatic 为 None），无 .tolist()
            logger.debug(
                f'[_prudent_reconstruct_mol] {context} | '
                f'atom_numbers={_step_traj_as_int_list(atom_numbers)}, '
                f'aromatic_flags={_step_traj_as_bool_list(aromatic_flags)}'
            )

        mol = reconstruct.reconstruct_from_generated(pos_array, atom_numbers, aromatic_flags)
        
        if logger:
            if mol is not None:
                logger.debug(f'[_prudent_reconstruct_mol] {context} | SUCCESS: mol created with {mol.GetNumAtoms()} atoms')
            else:
                logger.debug(f'[_prudent_reconstruct_mol] {context} | FAILED: reconstruct returned None')
        
        return mol
    except Exception as e:
        if logger:
            logger.warning(
                f'[_prudent_reconstruct_mol] {context} | EXCEPTION: {type(e).__name__}: {str(e)[:100]}'
            )
        return None


def _prudent_score_one_round(
    pool,
    *,
    seed_idx,
    rnd,
    n_rounds,
    advance_top_k,
    prudent_cfg,
    protein_path,
    lig_fn,
    vina_protein_root,
    ligand_atom_mode,
    vina_advance_thr,
    logger,
):
    """一轮 GPU refine 之后：QED/SA →（条件）Vina → 晋级筛选。仅 CPU，可在工作线程中调用。"""
    min_q_dock = float(prudent_cfg.get('min_qed_for_docking', 0.3))
    min_s_dock = float(prudent_cfg.get('min_sa_for_docking', 0.3))

    chem_round = []
    for ci, c in enumerate(pool):
        mol = _prudent_reconstruct_mol(
            c['pos'], c['v'], ligand_atom_mode,
            logger=logger,
            context=f's{seed_idx}r{rnd}c{ci}'
        )
        qed, sa, chem_err = _prudent_chem_scores(mol, logger=logger)
        chem_round.append((ci, c, mol, qed, sa, chem_err))

    scored = []
    log_chains = prudent_cfg.get('log_per_chain_scores', True)
    for ci, c, mol, qed, sa, chem_err in chem_round:
        # 一票否决：QED/SA 先筛选
        gate_ok = (
            mol is not None
            and np.isfinite(qed)
            and np.isfinite(sa)
            and qed >= min_q_dock
            and sa >= min_s_dock
        )

        if not gate_ok:
            # QED/SA 不达标：composite 设为 -inf，不跑 Vina，不参与 winner 评选
            comp = float('-inf')
            reject_reason = 'qed_sa_rejected'
            if mol is None:
                reject_reason = 'mol_reconstruct_failed_or_none'
            elif not np.isfinite(qed) or not np.isfinite(sa):
                reject_reason = 'qed_sa_not_finite'
            detail = {
                'qed': qed,
                'sa': sa,
                'vina_affinity': float('nan'),
                'vina_norm': 0.0,
                'vina_skipped': True,
                'reconstruct_error': chem_err,
                'reason': reject_reason,
            }
            if logger and log_chains:
                qed_s = f'{qed:.4f}' if np.isfinite(qed) else 'nan'
                sa_s = f'{sa:.4f}' if np.isfinite(sa) else 'nan'
                logger.info(
                    f'[Prudent][score] seed={seed_idx} r={rnd + 1}/{n_rounds} chain={ci} | '
                    f'QED={qed_s} SA={sa_s} | REJECTED ({reject_reason}, thresholds {min_q_dock}/{min_s_dock}), '
                    f'composite=-inf'
                )
        else:
            # QED/SA 达标：跑 Vina，计算综合分
            comp, detail = _prudent_composite_score(
                mol,
                prudent_cfg,
                protein_path,
                lig_fn,
                logger=logger,
                protein_root=vina_protein_root,
                prefetched_chem=(qed, sa, chem_err),
                skip_vina=False,  # 达标必跑 Vina
            )
            if logger and log_chains:
                aff = detail.get('vina_affinity')
                aff_s = (
                    f'{float(aff):.3f}'
                    if aff is not None and np.isfinite(float(aff))
                    else 'nan'
                )
                _lip_n = detail.get('lipinski', 0)
                try:
                    _lip_n = int(_lip_n) if _lip_n is not None and _lip_n != 'N/A' else 0
                except (TypeError, ValueError):
                    _lip_n = 0
                _l_dem = detail.get('lilly_demerit', 0)
                try:
                    _l_dem = int(_l_dem or 0)
                except (TypeError, ValueError):
                    _l_dem = 0
                _l_pass = bool(detail.get('lilly_passed', False))
                logger.info(
                    f'[Prudent][score] seed={seed_idx} r={rnd + 1}/{n_rounds} chain={ci} | '
                    f'QED={detail.get("qed", 0.0):.4f} SA={detail.get("sa", 0.0):.4f} '
                    f'Vina_kcal={aff_s} vina_norm={detail.get("vina_norm", 0.0):.4f} '
                    f'lipinski_n={_lip_n} lipinski_frac={detail.get("lipinski_frac", 0.0):.4f} '
                    f'lilly_demerit={_l_dem} lilly_passed={1 if _l_pass else 0} '
                    f'lilly_norm={detail.get("lilly_norm", 0.0):.4f} '
                    f'composite={comp:.4f}'
                )
        scored.append((comp, detail, c, ci))
    # 按 composite 排序（-inf 的会被排在最后，自然淘汰）
    scored.sort(key=lambda x: x[0], reverse=True)

    if logger:
        top_msg = scored[0][0] if scored else float('nan')
        logger.info(
            f'[Prudent] seed={seed_idx} round={rnd + 1}/{n_rounds} | '
            f'candidates={len(scored)} | best_composite={top_msg:.4f}'
        )

    chem_by_ci = {row[0]: row for row in chem_round}
    filtered = []

    def _annotate_pool(survivor_id_set):
        for comp, detail, c, ci in scored:
            _crow = chem_by_ci.get(ci)
            _mol = _crow[2] if _crow is not None else None
            _qed = _crow[3] if _crow is not None else 0.0
            _sa = _crow[4] if _crow is not None else 0.0
            el = _prudent_eligible_for_advance(
                _mol, _qed, _sa, detail,
                prudent_cfg, vina_advance_thr,
            )
            aff = detail.get('vina_affinity')
            try:
                aff_f = float(aff) if aff is not None and np.isfinite(float(aff)) else None
            except (TypeError, ValueError):
                aff_f = None
            c['prudent_round_analysis'] = {
                'seed_idx': int(seed_idx),
                'round': int(rnd),
                'chain_idx': int(ci),
                'composite': float(comp) if np.isfinite(float(comp)) else None,
                'qed': detail.get('qed'),
                'sa': detail.get('sa'),
                'vina_affinity': aff_f,
                'vina_norm': detail.get('vina_norm'),
                'eligible_for_advance': bool(el),
                'survivor_topk': bool(id(c) in survivor_id_set),
            }

    for t in scored:
        comp, detail, c, ci = t
        row = chem_by_ci.get(ci)
        if row is None:
            continue
        _ci, _c, mol, qed, sa, _chem_err = row
        if _prudent_eligible_for_advance(
            mol, qed, sa, detail, prudent_cfg, vina_advance_thr,
        ):
            filtered.append(t)
    if logger:
        n_elim = len(scored) - len(filtered)
        if n_elim:
            logger.info(
                f'[Prudent] seed={seed_idx} r={rnd + 1}/{n_rounds} | '
                f'晋级筛选（QED/SA 门槛 + Vina 阈值 + vina_weight 时须已对接）'
                f'→ {len(filtered)}/{len(scored)} 条可进 advance_top_k，{n_elim} 条淘汰'
            )
    _atk_mode = str(prudent_cfg.get('advance_top_k_mode', 'cap')).strip().lower()
    if _atk_mode == 'cap' and prudent_cfg.get('advance_top_k_strict') is True:
        _atk_mode = 'strict'
    continue_when_no_advance = bool(prudent_cfg.get('continue_when_no_advance', False))
    fallback_keep_top_k = max(1, int(prudent_cfg.get('fallback_keep_top_k', 1)))
    if _atk_mode in ('strict', 'require_full_k', 'full_k'):
        if len(filtered) < advance_top_k:
            if continue_when_no_advance and scored:
                fallback = scored[: min(fallback_keep_top_k, len(scored))]
                pool_out = [t[2] for t in fallback]
                survivor_ids = {id(c) for c in pool_out}
                for _, det, c, _ci in fallback:
                    c['prudent_score_detail'] = det
                if logger:
                    logger.warning(
                        f'[Prudent] seed={seed_idx} r={rnd + 1}/{n_rounds} | '
                        f'advance_top_k_mode=strict 未满足 {len(filtered)}/{advance_top_k}，'
                        f'启用 continue_when_no_advance：保留 top-{len(pool_out)} 继续 refine'
                    )
                _annotate_pool(survivor_ids)
                return pool_out, [], False
            if logger:
                logger.warning(
                    f'[Prudent] seed={seed_idx} r={rnd + 1}/{n_rounds} | '
                    f'advance_top_k_mode=strict：满足晋级条件 {len(filtered)} 条 < '
                    f'advance_top_k={advance_top_k}，本 round 不晋级任何链，该 seed 后续轮次跳过'
                )
            _annotate_pool(set())
            return [], [], True
    if not filtered:
        if continue_when_no_advance and scored:
            fallback = scored[: min(fallback_keep_top_k, len(scored))]
            pool_out = [t[2] for t in fallback]
            survivor_ids = {id(c) for c in pool_out}
            for _, det, c, _ci in fallback:
                c['prudent_score_detail'] = det
            if logger:
                logger.warning(
                    f'[Prudent] seed={seed_idx} r={rnd + 1}/{n_rounds} | '
                    f'无链满足晋级条件，启用 continue_when_no_advance：保留 top-{len(pool_out)} 继续 refine'
                )
            _annotate_pool(survivor_ids)
            return pool_out, [], False
        if logger:
            logger.warning(
                f'[Prudent] seed={seed_idx} r={rnd + 1}/{n_rounds} | '
                f'无分子满足完整晋级条件（min_qed/min_sa、'
                f'max_vina_affinity_for_advance、vina_weight 对应须对接等），'
                f'advance_top_k 保留 0 条，该 seed 后续轮次跳过'
            )
        _annotate_pool(set())
        return [], [], True

    survivors = filtered[:advance_top_k]
    pool_out = [t[2] for t in survivors]
    survivor_ids = {id(c) for c in pool_out}
    for _, det, c, _ci in survivors:
        c['prudent_score_detail'] = det

    _annotate_pool(survivor_ids)

    pending = []
    if vina_advance_thr is not None:
        for rank, (comp, det, c, ci) in enumerate(survivors):
            aff = det.get('vina_affinity')
            try:
                aff_f = float(aff) if aff is not None and np.isfinite(float(aff)) else None
            except (TypeError, ValueError):
                aff_f = None
            pending.append({
                'seed_idx': seed_idx,
                'round': rnd + 1,
                'n_rounds': n_rounds,
                'chain_idx': ci,
                'rank_in_round': rank,
                'vina_affinity': aff_f,
                'composite': float(comp) if np.isfinite(float(comp)) else None,
                'qed': det.get('qed'),
                'sa': det.get('sa'),
                'max_vina_affinity_for_advance': vina_advance_thr,
            })
    return pool_out, pending, False


def _prudent_vina_ok_for_next_round(detail, max_aff_thr):
    """对接完成后：finite Vina affinity（kcal/mol）且严格小于阈值才可晋级下一轮；未对接/nan 淘汰。

    ``max_aff_thr`` 为 ``None`` 时不做此项筛选（与旧行为兼容）。
    """
    if max_aff_thr is None:
        return True
    if not detail or detail.get('vina_skipped'):
        return False
    aff = detail.get('vina_affinity')
    if aff is None:
        return False
    try:
        a = float(aff)
    except (TypeError, ValueError):
        return False
    if not np.isfinite(a):
        return False
    return a < float(max_aff_thr)


def _prudent_eligible_for_advance(mol, qed, sa, detail, prudent_cfg, vina_advance_thr):
    """是否满足晋级下一轮的全部条件（QED/SA 已在前面一票否决，这里只检查 Vina 相关）。"""
    if mol is None:
        return False
    # QED/SA 已在 _prudent_score_one_round 中一票否决，这里只需检查 Vina 相关
    if not _prudent_vina_ok_for_next_round(detail, vina_advance_thr):
        return False
    wv = float(prudent_cfg.get('vina_weight', 0.6))
    if wv > 0.0 and (not detail or detail.get('vina_skipped')):
        return False
    return True


def _prudent_chem_scores(mol, logger=None):
    """单分子 QED/SA；mol 无效或 get_chem 失败时返回 (nan, nan, error_or_none)。"""
    if mol is None:
        return float('nan'), float('nan'), 'mol_reconstruct_failed_or_none'
    try:
        chem = scoring_func.get_chem(mol)
        qed_raw = chem.get('qed')
        sa_raw = chem.get('sa')
        qed = float(qed_raw) if qed_raw is not None else float('nan')
        sa = float(sa_raw) if sa_raw is not None else float('nan')
        return qed, sa, None
    except Exception as exc:
        if logger:
            logger.debug(f'[Prudent] get_chem 失败（无效价态等），按 QED/SA=nan 处理: {exc}')
        return float('nan'), float('nan'), str(exc)


def _prudent_composite_score(
    mol, prudent_cfg, protein_path, ligand_filename, logger=None, protein_root=None,
    prefetched_chem=None,
    skip_vina=False,
):
    """Prudent 综合分：Vina_norm、QED、SA 及可选 Lipinski（规则条数/5）、Lilly（扣分归一化）加权，权重自 YAML 归一化。

    Prudent 采样阶段在此直接 ``VinaDockingTask.run``，不通过 ``subprocess`` 调用 ``evaluate_pt`` /
    ``evaluate_pt_with_correct_reconstruct``（后两者用于离线评测流程）。

    ``protein_path`` 为可直接对接的 PDB；若无效则与 ``evaluate_pt`` 相同，用 ``protein_root`` +
    配体相对路径推断受体（``VinaDockingTask.from_generated_mol``）。

    ``prefetched_chem``：``(qed, sa, chem_error)``，避免重复 get_chem；``skip_vina=True`` 时不跑对接（每轮先统一算 QED/SA，未过阈者跳过 Vina 以省算力）。
    """
    w_v = max(0.0, float(prudent_cfg.get('vina_weight', 0.6)))
    w_q = max(0.0, float(prudent_cfg.get('qed_weight', 0.2)))
    w_s = max(0.0, float(prudent_cfg.get('sa_weight', 0.2)))
    w_li = max(0.0, float(prudent_cfg.get('lipinski_weight', 0.0)))
    w_ly = max(0.0, float(prudent_cfg.get('lilly_demerit_weight', 0.0)))
    s = w_v + w_q + w_s + w_li + w_ly
    if s <= 0:
        w_v, w_q, w_s, w_li, w_ly = 0.6, 0.2, 0.2, 0.0, 0.0
        s = 1.0
    else:
        w_v, w_q, w_s, w_li, w_ly = w_v / s, w_q / s, w_s / s, w_li / s, w_ly / s

    if mol is None:
        return float('-inf'), {'reason': 'mol_none'}

    if prefetched_chem is not None:
        qed, sa, chem_error = prefetched_chem
        qed = float(qed or 0.0)
        sa = float(sa or 0.0)
    else:
        qed, sa, chem_error = _prudent_chem_scores(mol, logger=logger)

    aff = float('nan')
    vina_norm = 0.0
    pp = str(protein_path) if protein_path else None
    use_pp = pp and os.path.isfile(pp)
    root = protein_root or os.environ.get('PROTEIN_ROOT', '').strip() or None
    if root:
        try:
            root = str(Path(root).expanduser().resolve())
        except Exception:
            pass
    if not skip_vina:
        r = None
        try:
            from utils.evaluation.docking_vina import VinaDockingTask

            with _PRUDENT_VINA_THREAD_LOCK:
                task = VinaDockingTask.from_generated_mol(
                    mol,
                    ligand_filename or 'N/A',
                    protein_path=pp if use_pp else None,
                    protein_root=root if root else './data/crossdocked',
                    tmp_dir=prudent_cfg.get('vina_tmp_dir', './tmp'),
                )
                ex = int(prudent_cfg.get('vina_exhaustiveness', 8))
                # 子线程不能用 SIGALRM 超时（signal 仅主线程）；timeout_sec=0 走无闹钟路径
                _vina_ts = None
                if threading.current_thread() is not threading.main_thread():
                    _vina_ts = 0
                r = task.run(
                    mode='score_only', exhaustiveness=ex, n_poses=1, timeout_sec=_vina_ts,
                )
            if r and r[0].get('affinity') is not None:
                aff = float(r[0]['affinity'])
        except Exception as exc:
            if logger:
                logger.warning(f'[Prudent] Vina score_only 失败: {exc}')
            aff = float('nan')
        if not use_pp and not root and logger:
            logger.debug(
                '[Prudent] 未提供有效 protein_path 与 protein_root，Vina 依赖 from_generated_mol 默认 protein_root'
            )
    elif logger:
        logger.debug(
            '[Prudent] 跳过 Vina（QED/SA 未同时达到 dock阈值，prefetched QED=%.4f SA=%.4f）',
            qed, sa,
        )

    if np.isfinite(aff):
        vina_norm = (-aff) / 10.0
        vina_norm = max(0.0, min(1.0, float(vina_norm)))
    else:
        vina_norm = 0.0

    lipinski_count = 0
    lipinski_frac = 0.0
    try:
        lip_d = scoring_func.get_lipinski(mol)
        raw_li = lip_d.get('lipinski', 0) if isinstance(lip_d, dict) else 0
        if raw_li == 'N/A' or raw_li is None:
            lipinski_count = 0
        else:
            lipinski_count = int(raw_li)
        lipinski_frac = max(0.0, min(1.0, float(lipinski_count) / 5.0))
    except Exception:
        lipinski_frac = 0.0

    lilly_norm = 0.0
    lilly_demerit = 0
    lilly_passed = False
    lilly_cutoff = 100
    try:
        from utils.evaluation.lilly_medchem_rules import evaluate_lilly_medchem_rules

        lr = evaluate_lilly_medchem_rules(mol, debug=False)
        if lr:
            lilly_demerit = int(lr.get('demerit', 0) or 0)
            lilly_cutoff = max(1, int(lr.get('demerit_cutoff', 100) or 100))
            lilly_passed = bool(lr.get('passed', False))
            if lilly_passed:
                lilly_norm = max(0.0, 1.0 - float(lilly_demerit) / float(lilly_cutoff))
            else:
                lilly_norm = 0.0
    except Exception:
        pass

    composite = (
        w_v * vina_norm + w_q * qed + w_s * sa + w_li * lipinski_frac + w_ly * lilly_norm
    )
    detail = {
        'vina_affinity': aff,
        'vina_norm': vina_norm,
        'qed': qed,
        'sa': sa,
        'lipinski': lipinski_count,
        'lipinski_frac': lipinski_frac,
        'lilly_demerit': lilly_demerit,
        'lilly_passed': lilly_passed,
        'lilly_norm': lilly_norm,
        'lilly_demerit_cutoff': lilly_cutoff,
        'weights': {
            'vina': w_v, 'qed': w_q, 'sa': w_s,
            'lipinski': w_li, 'lilly_demerit': w_ly,
        },
    }
    if chem_error is not None:
        detail['chem_error'] = chem_error
    if skip_vina:
        detail['vina_skipped'] = 'qed_sa_gate'
    return composite, detail


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
):
    """与 large_step 同类：多候选拼 ``Batch``，一次 ``sample_diffusion_refinement``，再按配体原子数拆包。

    ``refinement_grad_kwargs`` 会并入 ``sample_diffusion_refinement``（如
    ``_dynamic_subcfg_grad_cap_kwargs(refine_cfg)`` 或 Prudent 的截断参数）。

    Returns:
        list[dict]: 每条链 ``pos`` / ``v`` / ``log_v`` / ``num_atoms`` / 本段 ``pos_traj_ref`` 等（未与历史轨迹合并）。
    """
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

    if getattr(model, 'ligand_v_input', 'onehot') == 'log_prob':
        init_input = init_log_v
        log_mode = 'log_prob'
    else:
        init_input = init_log_v.argmax(dim=-1)
        log_mode = 'auto'

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
        **refinement_grad_kwargs,
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


def _prudent_refine_batch(
    model, data, cands, dynamic_cfg, refine_cfg, center_pos_mode, pos_only, device,
    time_upper, time_lower, max_grad_cap, logger=None,
    grad_fusion_anchor_t=None,
):
    """多链 refine：与 large_step 同类，``Batch.from_data_list`` 拼批后一次 ``sample_diffusion_refinement`` 前向。

    各链可具有不同配体原子数（``batch_ligand`` + ``safe_repeat_interleave``），再按原子数拆回每条链。
    """
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
    raw_list = _batch_refinement_forward_split(
        model,
        data,
        cands,
        center_pos_mode=center_pos_mode,
        pos_only=pos_only,
        device=device,
        time_upper=time_upper,
        time_lower=time_lower,
        refine_cfg=refine_cfg,
        refinement_grad_kwargs=refinement_grad_kwargs,
        error_prefix='[Prudent] refine batch',
    )
    out = []
    for i, cand in enumerate(cands):
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

        out.append({
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


def _prudent_terminal_time_upper(pool):
    """同批 refine 后各链 time_indices 末项应一致；取首链当前扩散时间下标作为下一段 ``time_upper``。"""
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
    ti = cand.get('time_indices')
    seq = _coerce_time_indices_seq(ti) if ti is not None else None
    if not seq:
        return None
    return int(seq[-1])


def _prudent_lineage_snapshot_from_refine_traj(cand, fusion_idx=0):
    """从单次 ``sample_diffusion_refinement``（含梯度融合截断）合并轨迹中取第 ``fusion_idx`` 帧作为下一段起点。

    与「650→606 记下一步」一致时取 ``fusion_idx=0``（轨迹首帧通常为第一步融合后的状态，视模型 ``record_traj`` 语义而定）。
    若轨迹不足则退回为当前 ``cand`` 深拷贝。"""
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


def _run_prudent_segment_evolution_dynamic(
    model,
    data,
    config,
    ligand_atom_mode,
    ls_out,
    total_candidates,
    *,
    device='cuda:0',
    logger=None,
    skip_targetdiff_baseline_refine=False,
    segment_checkpoint_context=None,
):
    """Prudent Cumulative：跨代累积分段演化

    每代流程：checkpoint_frame → 剩余融合至本代终点；可选 ``save_segment_generation_pool_pt`` 将
    本代 refine + baseline 之后 **pool 内全部并行链**（``n_sampling`` 条）落盘；
    ``save_segment_breakpoint_pt`` 控制是否额外保存中段断点 ``d`` 的 tensor 快照。
    """
    dynamic_cfg = config.sample.get('dynamic', {})
    refine_cfg = dynamic_cfg.get('refine', {})
    prudent_cfg = dynamic_cfg.get('prudent', {})

    time_boundary = get_time_boundary(dynamic_cfg, 750)
    time_lower_ref = int(refine_cfg.get('time_lower', 0))
    renoise_t = int(prudent_cfg.get('renoise_t', time_boundary))
    renoise_t = int(np.clip(renoise_t, 0, getattr(model, 'num_timesteps', 1000) - 1))

    # Cumulative 核心参数（只保留这些）
    n_sampling = max(1, int(prudent_cfg.get('n_sampling', 4)))
    max_segments = max(1, int(prudent_cfg.get('max_checkpoint_rounds', 5)))
    advance_top_k = max(1, int(prudent_cfg.get('advance_top_k', 1)))
    checkpoint_frame = max(1, int(prudent_cfg.get('checkpoint_frame', 2)))
    total_frames = max(1, int(prudent_cfg.get('total_frames', 24)))

    legacy_seg_disk = bool(prudent_cfg.get('save_checkpoint_every_segment'))
    explicit_bp_key = 'save_segment_breakpoint_pt' in prudent_cfg
    explicit_gen_key = 'save_segment_generation_pool_pt' in prudent_cfg
    save_bp_pt = (
        bool(prudent_cfg['save_segment_breakpoint_pt'])
        if explicit_bp_key
        else legacy_seg_disk
    )
    save_gen_pool_pt = (
        bool(prudent_cfg['save_segment_generation_pool_pt'])
        if explicit_gen_key
        else (legacy_seg_disk or True)
    )

    # 打分权重
    vina_weight = float(prudent_cfg.get('vina_weight', 0.86))
    qed_weight = float(prudent_cfg.get('qed_weight', 0.07))
    sa_weight = float(prudent_cfg.get('sa_weight', 0.07))
    vina_exhaustiveness = int(prudent_cfg.get('vina_exhaustiveness', 1))

    # 其他配置
    vina_advance_thr = prudent_cfg.get('max_vina_affinity_for_advance', None)
    if vina_advance_thr is not None:
        try:
            vina_advance_thr = float(vina_advance_thr)
        except (TypeError, ValueError):
            vina_advance_thr = None

    center_pos_mode = config.sample.get('center_pos_mode', 'protein')
    pos_only = config.sample.get('pos_only', False)

    # protein_root 用于 Vina
    protein_path = getattr(data, 'protein_filename', None)
    lig_fn = getattr(data, 'ligand_filename', None)
    vina_protein_root = prudent_cfg.get('protein_root') or os.environ.get('PROTEIN_ROOT', '').strip() or None
    if vina_protein_root:
        try:
            vina_protein_root = str(Path(vina_protein_root).expanduser().resolve())
        except Exception:
            pass

    protein_path = getattr(data, 'protein_filename', None)
    lig_fn = getattr(data, 'ligand_filename', None)
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

    center_pos_mode = config.sample.get('center_pos_mode', 'protein')
    pos_only = config.sample.get('pos_only', False)

    nti0 = total_candidates[0].get('time_indices') if total_candidates else None
    seq_ls = _coerce_time_indices_seq(nti0) if nti0 is not None else []
    n_ls_fusion = len(seq_ls)

    refined_records = []
    pending_analysis_hits = []
    segment_summaries_by_seed = []
    prudent_segment_ck_paths = []

    def _append_refined(seed_idx_local, pool_local):
        for ri, c in enumerate(pool_local):
            pr_det = c.pop('prudent_score_detail', {})
            c.pop('_seg_lineage_checkpoint', None)
            c.pop('_lineage_for_next', None)
            metric_info = evaluate_candidate(
                c['pos'], c['v'], ligand_atom_mode, dynamic_cfg.get('selector', {})
            )
            refined_records.append({
                'pos': c['pos'],
                'v': c['v'],
                'log_v': c['log_v'],
                'pos_traj': c.get('pos_traj') or [],
                'v_traj': c.get('v_traj') or [],
                'log_v_traj': c.get('log_v_traj') or [],
                'num_atoms': c['num_atoms'],
                'source_index': seed_idx_local,
                'repeat_index': ri,
                'time_indices': c.get('time_indices'),
                'prudent_composite_detail': pr_det,
                **metric_info,
            })

    # Cumulative 核心循环
    for seed_idx, seed_cand in enumerate(total_candidates):
        seg_log = []
        
        # 记录初始 seed_cand 状态
        if logger:
            logger.debug(
                f'[Prudent] SEED_INIT s{seed_idx} | '
                f'seed_cand pos_shape={seed_cand.get("pos", []).shape if hasattr(seed_cand.get("pos"), "shape") else "N/A"} | '
                f'v_dtype={seed_cand.get("v").dtype if hasattr(seed_cand.get("v"), "dtype") else "N/A"}'
            )
        
        pool = [
            _prudent_candidate_traj_cleared(
                _prudent_copy_candidate(seed_cand, logger=logger, context=f'g0_init'), 
                logger=logger, 
                context=f'g0_init_c{ci}'
            )
            for ci in range(n_sampling)
        ]
        current_t = int(time_boundary)
        final_pool_out = []

        for seg_idx in range(max_segments):
            # 全局时间轴：g0=0~26，g1=2~26，g2=4~26...
            # 每一代都固定先跑 checkpoint_frame 次融合并保存断点 d，下一代从「本代 winner 对应链」的 d 续跑。
            global_fusion_offset = seg_idx * checkpoint_frame
            remaining_steps = max(1, total_frames - global_fusion_offset)
            phase1_cap = max(1, min(checkpoint_frame, remaining_steps))

            if logger:
                logger.info(
                    f'[Prudent] g{seg_idx}s{seed_idx} t={current_t} '
                    f'global_offset={global_fusion_offset} remain={remaining_steps} phase1={phase1_cap}'
                )

            # 阶段1：先跑到断点 d（默认 2 次融合）
            pool_checkpoint = _prudent_stride_gpu_chunk(
                model, data, pool,
                time_upper=int(current_t),
                time_lower=time_lower_ref,
                stride_cap=phase1_cap,
                grad_anchor=None,
                renoise_t=renoise_t,
                refine_cfg=refine_cfg,
                dynamic_cfg=dynamic_cfg,
                center_pos_mode=center_pos_mode,
                pos_only=pos_only,
                device=device,
                logger=logger,
            )
            ck_tt = _prudent_terminal_time_upper(pool_checkpoint)

            # 保存断点 d（给下一代起点）
            if logger:
                logger.info(f'[Prudent] 存断点 s{seed_idx}g{seg_idx}f{phase1_cap} | {phase1_cap}次融合 | 下代续')
            if segment_checkpoint_context is not None and save_bp_pt:
                ck_path_frame = _prudent_save_iteration_torch_checkpoint(
                    segment_checkpoint_context,
                    data=data,
                    pool=pool_checkpoint,
                    seed_idx=seed_idx,
                    fname_middle_tail=f'seed{int(seed_idx)}_gen{int(seg_idx)}_frame{phase1_cap}',
                    checkpoint_kind='prudent_checkpoint',
                    scoring_snap=None,
                    seg_log_row={'gen': seg_idx, 'frame': phase1_cap},
                    prudent_max_iterations=max_segments,
                    ligand_atom_mode=ligand_atom_mode,
                    selector_cfg=dynamic_cfg.get('selector') or {},
                    logger=logger,
                )
                if ck_path_frame:
                    prudent_segment_ck_paths.append(os.path.abspath(ck_path_frame))

            # 阶段2：从断点 d 继续跑到本代终点
            remaining_to_end = max(0, remaining_steps - phase1_cap)
            if remaining_to_end > 0:
                pool_final = _prudent_stride_gpu_chunk(
                    model, data, pool_checkpoint,
                    time_upper=int(ck_tt) if ck_tt is not None else current_t,
                    time_lower=time_lower_ref,
                    stride_cap=remaining_to_end,
                    grad_anchor=None,
                    renoise_t=renoise_t,
                    refine_cfg=refine_cfg,
                    dynamic_cfg=dynamic_cfg,
                    center_pos_mode=center_pos_mode,
                    pos_only=pos_only,
                    device=device,
                    logger=logger,
                )
            else:
                pool_final = pool_checkpoint

            last_tt = _prudent_terminal_time_upper(pool_final)
            if last_tt is not None:
                last_tt = int(last_tt)

            _prudent_apply_targetdiff_baseline_to_pool(
                model,
                data,
                pool_final,
                config,
                device=device,
                logger=logger,
                skip=skip_targetdiff_baseline_refine,
                log_prefix=f's{seed_idx} gen{seg_idx}',
                round_idx=seg_idx,
                n_rounds=max_segments,
            )

            # 打分 & 选 winner
            pool_out, pend, broke = _prudent_score_one_round(
                pool_final,
                seed_idx=seed_idx,
                rnd=seg_idx,
                n_rounds=max_segments,
                advance_top_k=advance_top_k,
                prudent_cfg=prudent_cfg,
                protein_path=protein_path,
                lig_fn=lig_fn,
                vina_protein_root=vina_protein_root,
                ligand_atom_mode=ligand_atom_mode,
                vina_advance_thr=vina_advance_thr,
                logger=logger,
            )
            pending_analysis_hits.extend(pend)

            # 生成 scoring snapshot
            scoring_snap_full = None
            if pool_final:
                scoring_snap_full = _snapshot_prudent_generation(
                    seed_idx=seed_idx,
                    generation_idx=seg_idx,
                    restart_t_upper=int(current_t),
                    terminal_t_upper=last_tt,
                    pool=pool_final,
                    ligand_atom_mode=ligand_atom_mode,
                    selector_cfg=dynamic_cfg.get('selector') or {},
                )

            seg_log.append({
                'generation': seg_idx,
                'global_fusion_offset': global_fusion_offset,
                'checkpoint_frame': phase1_cap,
                'total_frames': total_frames,
                'remaining_steps': remaining_steps,
                'terminal_t': last_tt,
                'scoring_snapshot': scoring_snap_full,
            })

            # 每代末尾：写入本代全部并行链（refine + baseline 后，打分前）；见 save_segment_generation_pool_pt
            if (
                segment_checkpoint_context is not None
                and save_gen_pool_pt
                and _prudent_segment_generation_pt_seed_allowed(prudent_cfg, seed_idx)
            ):
                ck_path_gen = _prudent_save_iteration_torch_checkpoint(
                    segment_checkpoint_context,
                    data=data,
                    pool=pool_final,
                    seed_idx=seed_idx,
                    fname_middle_tail=f'gen{int(seg_idx)}_seed{int(seed_idx)}_chains{len(pool_final)}',
                    checkpoint_kind='prudent_generation_pool',
                    scoring_snap=scoring_snap_full,
                    seg_log_row={'gen': seg_idx, 'checkpoint_kind': 'prudent_generation_pool'},
                    prudent_max_iterations=max_segments,
                    ligand_atom_mode=ligand_atom_mode,
                    selector_cfg=dynamic_cfg.get('selector') or {},
                    logger=logger,
                )
                if ck_path_gen:
                    prudent_segment_ck_paths.append(os.path.abspath(ck_path_gen))
            elif segment_checkpoint_context is not None:
                ck_path_full = _prudent_save_iteration_torch_checkpoint(
                    segment_checkpoint_context,
                    data=data,
                    pool=pool_final,
                    seed_idx=seed_idx,
                    fname_middle_tail=f'seed{int(seed_idx)}_gen{int(seg_idx)}_full',
                    checkpoint_kind='prudent_full',
                    scoring_snap=scoring_snap_full,
                    seg_log_row={'gen': seg_idx, 'frame': 'full'},
                    prudent_max_iterations=max_segments,
                    ligand_atom_mode=ligand_atom_mode,
                    selector_cfg=dynamic_cfg.get('selector') or {},
                    logger=logger,
                )
                if ck_path_full:
                    prudent_segment_ck_paths.append(os.path.abspath(ck_path_full))

            if broke or not pool_out:
                final_pool_out = pool_out if pool_out else []
                break

            final_pool_out = pool_out

            # 准备下一代：读取 winner 对应链在断点 d 的状态并续跑
            winner = pool_out[0]
            winner_ana = winner.get('prudent_round_analysis') if isinstance(winner, dict) else None
            winner_chain_idx = 0
            if isinstance(winner_ana, dict):
                try:
                    winner_chain_idx = int(winner_ana.get('chain_idx', 0))
                except (TypeError, ValueError):
                    winner_chain_idx = 0
            
            # 记录调试信息
            if logger:
                logger.debug(
                    f'[Prudent] GEN_HANDOVER s{seed_idx} g{seg_idx}→g{seg_idx+1} | '
                    f'winner_chain_idx={winner_chain_idx} (from analysis) | '
                    f'pool_checkpoint_len={len(pool_checkpoint) if pool_checkpoint else 0} | '
                    f'winner pos_shape={winner.get("pos", []).shape if hasattr(winner.get("pos"), "shape") else "N/A"} | '
                    f'winner v_dtype={winner.get("v").dtype if hasattr(winner.get("v"), "dtype") else "N/A"}'
                )
            
            if pool_checkpoint:
                # 确保索引在有效范围内
                winner_chain_idx = int(np.clip(winner_chain_idx, 0, len(pool_checkpoint) - 1))
                checkpoint_cand = pool_checkpoint[winner_chain_idx]
                
                if logger:
                    logger.debug(
                        f'[Prudent] CHECKPOINT_SELECT s{seed_idx} g{seg_idx} | '
                        f'selected_idx={winner_chain_idx} | '
                        f'ck_pos_shape={checkpoint_cand.get("pos", []).shape if hasattr(checkpoint_cand.get("pos"), "shape") else "N/A"} | '
                        f'ck_v_dtype={checkpoint_cand.get("v").dtype if hasattr(checkpoint_cand.get("v"), "dtype") else "N/A"} | '
                        f'ck_v_range=[{checkpoint_cand.get("v").min() if hasattr(checkpoint_cand.get("v"), "min") else "N/A"}, '
                        f'{checkpoint_cand.get("v").max() if hasattr(checkpoint_cand.get("v"), "max") else "N/A"}]'
                    )
                
                winner_lineage = _prudent_copy_candidate(
                    checkpoint_cand, 
                    logger=logger, 
                    context=f'g{seg_idx}_winner_ck'
                )
            else:
                winner_lineage = _prudent_copy_candidate(
                    winner, 
                    logger=logger, 
                    context=f'g{seg_idx}_winner_final'
                )

            if logger:
                rt_next = _prudent_checkpoint_resume_t_upper(winner_lineage)
                logger.info(f'[Prudent] g{seg_idx} winner_chain={winner_chain_idx} @d(frame={phase1_cap}) → t{rt_next}')

            next_parent = _prudent_copy_candidate(
                winner_lineage, 
                logger=logger, 
                context=f'g{seg_idx+1}_parent'
            )
            next_parent.pop('_seg_lineage_checkpoint', None)
            next_parent.pop('_lineage_for_next', None)

            rt_next = _prudent_checkpoint_resume_t_upper(next_parent)
            if rt_next is None:
                rt_next = int(ck_tt) if ck_tt is not None else int(time_boundary)

            # 下一代起点：从断点 d 直接启动（取消显式加噪，噪声由refine配置处理）
            # 若需显式加噪，设置 generation_restart_renoise_t 或 generation_restart_noise_fusions
            t_max = int(getattr(model, 'num_timesteps', 1000) - 1)
            restart_renoise_t = prudent_cfg.get('generation_restart_renoise_t')
            restart_noise_fusions = prudent_cfg.get('generation_restart_noise_fusions')
            restart_t_upper = None
            if restart_renoise_t is not None:
                try:
                    restart_t_upper = int(np.clip(int(restart_renoise_t), 0, t_max))
                except (TypeError, ValueError):
                    restart_t_upper = None
            elif restart_noise_fusions is not None:
                try:
                    restart_t_upper = int(np.clip(int(rt_next) + max(0, int(restart_noise_fusions)), 0, t_max))
                except (TypeError, ValueError):
                    restart_t_upper = None
            if restart_t_upper is not None and restart_t_upper > int(rt_next):
                next_parent = _prudent_forward_diffuse_to_t(
                    model,
                    data,
                    next_parent,
                    restart_t_upper,
                    center_pos_mode,
                    pos_only,
                    device,
                    logger=logger,
                )
                current_t = int(restart_t_upper)
                if logger:
                    logger.info(
                        f'[Prudent] g{seg_idx + 1} 重启前加噪: t={rt_next} -> t={restart_t_upper}'
                    )
            else:
                current_t = int(rt_next)
                if logger and seg_idx < max_segments - 1:
                    logger.info(f'[Prudent] g{seg_idx + 1} 直接从断点 t={rt_next} 启动（无显式加噪）')

            if logger:
                logger.info(f'[Prudent] 续断点 s{seed_idx}g{seg_idx + 1} | f{phase1_cap}→余{remaining_steps - phase1_cap}融合')
            
            # 生成下一代 pool，记录详细调试
            pool = []
            for pi in range(n_sampling):
                copied = _prudent_copy_candidate(
                    next_parent, 
                    logger=logger, 
                    context=f'g{seg_idx+1}_pool_p{pi}'
                )
                pool.append(copied)
            
            # 记录 pool 状态摘要
            if logger:
                logger.debug(
                    f'[Prudent] POOL_READY s{seed_idx} g{seg_idx+1} | '
                    f'pool_size={len(pool)} | '
                    f'first_pos_shape={pool[0]["pos"].shape if pool else "N/A"} | '
                    f'first_v_dtype={pool[0]["v"].dtype if pool else "N/A"} | '
                    f'first_v_range=[{pool[0]["v"].min() if pool else "N/A"}, {pool[0]["v"].max() if pool else "N/A"}]'
                )

        _append_refined(seed_idx, final_pool_out if final_pool_out else [])
        segment_summaries_by_seed.append({'seed_idx': seed_idx, 'segments': seg_log})

    refined_pos_list = [rec['pos'] for rec in refined_records]
    refined_v_list = [rec['v'] for rec in refined_records]
    refined_pos_traj = [rec['pos_traj'] for rec in refined_records]
    refined_v_traj = [rec['v_traj'] for rec in refined_records]

    # meta_dict 精简版
    meta_dict = dict(ls_out.get('meta', {}))
    meta_dict['refined_candidates'] = refined_records
    prev_pr = meta_dict.get('prudent')
    pr_meta = dict(prev_pr) if isinstance(prev_pr, dict) else {}
    pr_meta.update({
        'cumulative_mode': True,
        'n_sampling': n_sampling,
        'max_generations': max_segments,
        'checkpoint_frame': checkpoint_frame,
        'total_frames': total_frames,
        'advance_top_k': advance_top_k,
        'weights': {
            'vina': vina_weight,
            'qed': qed_weight,
            'sa': sa_weight,
            'lipinski': float(prudent_cfg.get('lipinski_weight', 0.0)),
            'lilly_demerit': float(prudent_cfg.get('lilly_demerit_weight', 0.0)),
        },
        'vina_exhaustiveness': vina_exhaustiveness,
        'segment_summaries_by_seed': segment_summaries_by_seed,
        'pending_analysis_hits': pending_analysis_hits,
        'time_boundary': time_boundary,
        'renoise_t': renoise_t,
        'generation_restart_renoise_t': prudent_cfg.get('generation_restart_renoise_t'),
        'generation_restart_noise_fusions': prudent_cfg.get('generation_restart_noise_fusions'),
    })
    if prudent_segment_ck_paths:
        pr_meta['checkpoint_paths'] = prudent_segment_ck_paths
    meta_dict['prudent'] = pr_meta

    time_list = list(ls_out.get('time_list', []))

    return {
        'pos_list': refined_pos_list,
        'v_list': refined_v_list,
        'pos_traj': refined_pos_traj,
        'v_traj': refined_v_traj,
        'log_v_traj': [rec['log_v_traj'] for rec in refined_records],
        'time_list': time_list,
        'meta': meta_dict,
    }


def _run_prudent_legacy_dynamic(
    model, data, config, ligand_atom_mode, device='cuda:0', logger=None,
    skip_targetdiff_baseline_refine=False,
    segment_checkpoint_context=None,
):
    """Prudent：large_step 后多链 refine。

    - **默认**（``prudent.full_generation_mode=false``）：自 ``time_boundary`` 向下连续 refine，每 ``checkpoint_every_grad_fusions``
      梯度融合后打分/晋级至多 ``max_checkpoint_rounds`` 次；用尽或到达 ``time_lower`` 后仍会视情况跑满末尾 refine。
    - **线程分段演化**（``segment_evolution_mode=true``）：第 0 段 reverse 上限 = ``refine.max_grad_fusion_iterations``（或 ``generation_max_grad_fusion_iterations``）；
      第 ``k`` 段为 ``max(1, 上限 − k × checkpoint_every_grad_fusions)``；lineage 取自本段轨迹第 ``segment_lineage_traj_index`` 帧；段末可选 ``segment_pre_baseline_forward_t``，再 ``targetdiff_baseline_refine``，再 Vina/QED/SA。``max_checkpoint_rounds`` 为分段轮数。
    """
    dynamic_cfg = config.sample.get('dynamic', {})
    refine_cfg = dynamic_cfg.get('refine', {})
    prudent_cfg = dynamic_cfg.get('prudent', {})
    time_boundary = get_time_boundary(dynamic_cfg, 750)
    renoise_t = int(prudent_cfg.get('renoise_t', time_boundary))
    renoise_t = int(np.clip(renoise_t, 0, getattr(model, 'num_timesteps', 1000) - 1))

    n_sampling = max(1, int(prudent_cfg.get('n_sampling', refine_cfg.get('n_sampling', 4))))
    advance_top_k = max(1, int(prudent_cfg.get('advance_top_k', min(2, n_sampling))))
    advance_top_k = min(advance_top_k, n_sampling)
    advance_top_k_mode = str(prudent_cfg.get('advance_top_k_mode', 'cap')).strip().lower()
    if advance_top_k_mode == 'cap' and prudent_cfg.get('advance_top_k_strict') is True:
        advance_top_k_mode = 'strict'
    if advance_top_k_mode in ('require_full_k', 'full_k'):
        advance_top_k_mode = 'strict'
    if advance_top_k_mode != 'strict':
        advance_top_k_mode = 'cap'

    center_pos_mode = config.sample.get('center_pos_mode', 'protein')
    pos_only = config.sample.get('pos_only', False)

    protein_path = getattr(data, 'protein_filename', None)
    lig_fn = getattr(data, 'ligand_filename', None)
    vina_protein_root = prudent_cfg.get('protein_root') or os.environ.get('PROTEIN_ROOT', '').strip() or None
    if vina_protein_root:
        try:
            vina_protein_root = str(Path(vina_protein_root).expanduser().resolve())
        except Exception:
            pass

    # 1) 仅 large_step：临时打开 skip_refine
    prev_skip = dynamic_cfg.get('skip_refine', False)
    dynamic_cfg['skip_refine'] = True
    try:
        ls_out = _run_legacy_dynamic(
            model, data, config, ligand_atom_mode,
            device=device, logger=logger, skip_targetdiff_baseline_refine=True,
        )
    finally:
        dynamic_cfg['skip_refine'] = prev_skip

    total_candidates = ls_out['meta'].get('large_step_candidates', [])
    print(f"[DEBUG] large_step_candidates count: {len(total_candidates)}")
    if logger:
        logger.info(f'[Prudent] large_step输出 {len(total_candidates)} 个候选，准备refine到t=0')
    if not total_candidates:
        _log(logger, 'warning', '[Prudent] large_step 无候选，回退为空结果')
        return {
            'pos_list': [],
            'v_list': [],
            'pos_traj': [],
            'v_traj': [],
            'log_v_traj': [],
            'time_list': ls_out.get('time_list', []),
            'meta': {'prudent': {'error': 'no_large_step_candidates'}, **ls_out.get('meta', {})},
        }

    seg_flag = prudent_cfg.get('segment_evolution_mode')
    if seg_flag is None:
        # 兼容：若配置显式给出 cumulative 关键键，则默认启用分段累计模式
        seg_flag = prudent_cfg.get('cumulative_mode')
        if seg_flag is None and ('checkpoint_frame' in prudent_cfg or 'total_frames' in prudent_cfg):
            seg_flag = True
    segment_evolution_mode = bool(seg_flag)
    if segment_evolution_mode:
        if logger:
            logger.info(
                '[Prudent] segment_evolution_mode=true：每段 reverse 上限 = '
                'max(1, refine.max_grad_fusion_iterations − seg×checkpoint_every_grad_fusions)→ 可选 '
                'segment_pre_baseline_forward_t → baseline_refine → Vina/QED/SA；下一段 lineage 取自轨迹帧'
            )
        return _run_prudent_segment_evolution_dynamic(
            model, data, config, ligand_atom_mode, ls_out, total_candidates,
            device=device, logger=logger,
            skip_targetdiff_baseline_refine=skip_targetdiff_baseline_refine,
            segment_checkpoint_context=segment_checkpoint_context,
        )

    try:
        stride_fusions = max(1, int(prudent_cfg.get('checkpoint_every_grad_fusions', 3)))
    except (TypeError, ValueError):
        stride_fusions = 3
    try:
        max_checkpoint_rounds = max(
            1,
            int(prudent_cfg.get('max_checkpoint_rounds', prudent_cfg.get('n_rounds', 5))),
        )
    except (TypeError, ValueError):
        max_checkpoint_rounds = 5
    time_lower_ref = int(refine_cfg.get('time_lower', 0))
    selector_cfg_dyn = dict(dynamic_cfg.get('selector') or {})

    full_generation_mode = bool(prudent_cfg.get('full_generation_mode', False))
    generation_tables_by_seed = []
    full_gen_grad_cap = (
        _prudent_full_generation_grad_cap(refine_cfg, prudent_cfg)
        if full_generation_mode
        else None
    )

    time_records = {'large_step': ls_out['meta'].get('time_records', {}).get('large_step', []), 'refine': []}

    refined_records = []
    pending_analysis_hits = []
    prudent_iter_ck_paths = []

    vina_advance_thr = prudent_cfg.get('max_vina_affinity_for_advance', 4.0)
    if vina_advance_thr is not None:
        try:
            vina_advance_thr = float(vina_advance_thr)
        except (TypeError, ValueError):
            vina_advance_thr = 4.0

    use_async_vina = (
        len(total_candidates) > 1
        and prudent_cfg.get('async_vina_across_seeds', True)
    )
    _mw = prudent_cfg.get('async_vina_max_workers')
    async_max_workers = (
        min(32, max(4, len(total_candidates)))
        if _mw is None
        else max(1, int(_mw))
    )

    def _append_refined_for_seed(seed_idx_local, pool_local):
        for ri, c in enumerate(pool_local):
            pr_det = c.pop('prudent_score_detail', {})
            metric_info = evaluate_candidate(
                c['pos'], c['v'], ligand_atom_mode, dynamic_cfg.get('selector', {}))
            refined_records.append({
                'pos': c['pos'],
                'v': c['v'],
                'log_v': c['log_v'],
                'pos_traj': c.get('pos_traj') or [],
                'v_traj': c.get('v_traj') or [],
                'log_v_traj': c.get('log_v_traj') or [],
                'num_atoms': c['num_atoms'],
                'source_index': seed_idx_local,
                'repeat_index': ri,
                'time_indices': c.get('time_indices'),
                'prudent_composite_detail': pr_det,
                **metric_info,
            })

    if full_generation_mode and use_async_vina and logger:
        logger.warning(
            '[Prudent] full_generation_mode 与 async_vina_across_seeds 互不兼容；本运行改为按 seed 顺序执行（不走跨 seed Vina 线程池）。'
        )
    use_async_effective = bool(use_async_vina and not full_generation_mode)

    if segment_checkpoint_context is not None and (use_async_effective or full_generation_mode):
        if logger:
            logger.warning(
                '[Prudent] save_checkpoint_every_segment：当前分支为 '
                + ('async_vina_across_seeds' if use_async_effective else 'full_generation_mode')
                + '，暂不写入按迭代 .pt（仅 sequential 与 segment_evolution 支持）。'
            )

    if use_async_effective:
        n_seeds = len(total_candidates)
        pools = [
            [_prudent_copy_candidate(total_candidates[s]) for _ in range(n_sampling)]
            for s in range(n_seeds)
        ]
        current_ts = [int(time_boundary) for _ in range(n_seeds)]
        dead = [False] * n_seeds
        if logger:
            logger.info(
                f'[Prudent] async_vina_across_seeds：{n_seeds} 个 large_step 候选 | '
                f'checkpoint：每 {stride_fusions} 次梯度融合打分一次，至多 {max_checkpoint_rounds} 次 '
                f'（max_workers={async_max_workers}）'
            )
        with ThreadPoolExecutor(max_workers=async_max_workers) as executor:
            for chk in range(max_checkpoint_rounds):
                round_futs = {}
                for seed_idx in range(n_seeds):
                    if dead[seed_idx]:
                        continue
                    pool = pools[seed_idx]
                    ct = int(current_ts[seed_idx])
                    grad_anchor = None if chk == 0 else ct
                    pool = _prudent_stride_gpu_chunk(
                        model,
                        data,
                        pool,
                        time_upper=ct,
                        time_lower=time_lower_ref,
                        stride_cap=stride_fusions,
                        grad_anchor=grad_anchor,
                        renoise_t=renoise_t,
                        refine_cfg=refine_cfg,
                        dynamic_cfg=dynamic_cfg,
                        center_pos_mode=center_pos_mode,
                        pos_only=pos_only,
                        device=device,
                        logger=logger,
                    )
                    pools[seed_idx] = pool
                    tt = _prudent_terminal_time_upper(pool)
                    if tt is not None:
                        current_ts[seed_idx] = tt
                    _prudent_apply_targetdiff_baseline_to_pool(
                        model,
                        data,
                        pools[seed_idx],
                        config,
                        device=device,
                        logger=logger,
                        skip=skip_targetdiff_baseline_refine,
                        log_prefix=f'async s{seed_idx} chk{chk}',
                        round_idx=chk,
                        n_rounds=max_checkpoint_rounds,
                    )
                    pool_snap = [_prudent_copy_candidate(c) for c in pools[seed_idx]]
                    round_futs[seed_idx] = executor.submit(
                        _prudent_score_one_round,
                        pool_snap,
                        seed_idx=seed_idx,
                        rnd=chk,
                        n_rounds=max_checkpoint_rounds,
                        advance_top_k=advance_top_k,
                        prudent_cfg=prudent_cfg,
                        protein_path=protein_path,
                        lig_fn=lig_fn,
                        vina_protein_root=vina_protein_root,
                        ligand_atom_mode=ligand_atom_mode,
                        vina_advance_thr=vina_advance_thr,
                        logger=logger,
                    )
                for seed_idx in sorted(round_futs.keys()):
                    pool_out, pend, broke = round_futs[seed_idx].result()
                    pending_analysis_hits.extend(pend)
                    if broke:
                        dead[seed_idx] = True
                        pools[seed_idx] = []
                    else:
                        pools[seed_idx] = pool_out

                for seed_idx in range(n_seeds):
                    if dead[seed_idx]:
                        continue
                    ct = int(current_ts[seed_idx])
                    last_chk = chk == max_checkpoint_rounds - 1
                    at_end = ct <= time_lower_ref
                    if last_chk or at_end:
                        continue
                    pools[seed_idx] = _prudent_expand_survivors_for_stride(
                        pools[seed_idx],
                        n_sampling,
                        seed_idx=seed_idx,
                        chk=chk,
                        max_chk=max_checkpoint_rounds,
                        logger=logger,
                    )

        for seed_idx in range(n_seeds):
            if dead[seed_idx]:
                continue
            pool = pools[seed_idx]
            ct = current_ts[seed_idx]
            if pool and ct is not None and int(ct) > int(time_lower_ref):
                if logger:
                    logger.info(f'[Prudent] refine s{seed_idx} t={ct}→{time_lower_ref}')
                pools[seed_idx] = _prudent_stride_gpu_chunk(
                    model,
                    data,
                    pool,
                    time_upper=int(ct),
                    time_lower=time_lower_ref,
                    stride_cap=None,
                    grad_anchor=None,
                    renoise_t=renoise_t,
                    refine_cfg=refine_cfg,
                    dynamic_cfg=dynamic_cfg,
                    center_pos_mode=center_pos_mode,
                    pos_only=pos_only,
                    device=device,
                    logger=logger,
                )

        for seed_idx in range(n_seeds):
            if dead[seed_idx]:
                continue
            _append_refined_for_seed(seed_idx, pools[seed_idx])
    elif full_generation_mode:
        if logger:
            logger.info(
                f'[Prudent] 世代模式：每代一次 refine，梯度融合迭代上限={full_gen_grad_cap} '
                f'（prudent.generation_max_grad_fusion_iterations 优先，否则 refine.max_grad_fusion_iterations）；'
                f'末端 Vina/QED/SA→晋级→扩链；代数>0 时前向加噪再 refine。'
                f'每代全链写入 meta.prudent.generation_tables_by_seed（至多 {max_checkpoint_rounds} 代）。'
            )
        for seed_idx, seed_cand in enumerate(total_candidates):
            gen_table_seed = []
            pool = [
                _prudent_candidate_traj_cleared(_prudent_copy_candidate(seed_cand))
                for _ in range(n_sampling)
            ]
            for gen_idx in range(max_checkpoint_rounds):
                t_upper = _prudent_generation_restart_upper(
                    gen_idx, time_boundary, renoise_t, time_lower_ref, model, prudent_cfg,
                )
                if gen_idx > 0:
                    pool = _prudent_forward_reset_pool(
                        model, data, pool, t_upper,
                        center_pos_mode, pos_only, device, logger=logger,
                    )
                pool = _prudent_stride_gpu_chunk(
                    model,
                    data,
                    pool,
                    time_upper=int(t_upper),
                    time_lower=time_lower_ref,
                    stride_cap=full_gen_grad_cap,
                    grad_anchor=None,
                    renoise_t=renoise_t,
                    refine_cfg=refine_cfg,
                    dynamic_cfg=dynamic_cfg,
                    center_pos_mode=center_pos_mode,
                    pos_only=pos_only,
                    device=device,
                    logger=logger,
                )
                tt = _prudent_terminal_time_upper(pool)
                _prudent_apply_targetdiff_baseline_to_pool(
                    model,
                    data,
                    pool,
                    config,
                    device=device,
                    logger=logger,
                    skip=skip_targetdiff_baseline_refine,
                    log_prefix=f'fullgen s{seed_idx} g{gen_idx}',
                    round_idx=gen_idx,
                    n_rounds=max_checkpoint_rounds,
                )
                pool_ann = pool
                pool_out, pend, broke = _prudent_score_one_round(
                    pool_ann,
                    seed_idx=seed_idx,
                    rnd=gen_idx,
                    n_rounds=max_checkpoint_rounds,
                    advance_top_k=advance_top_k,
                    prudent_cfg=prudent_cfg,
                    protein_path=protein_path,
                    lig_fn=lig_fn,
                    vina_protein_root=vina_protein_root,
                    ligand_atom_mode=ligand_atom_mode,
                    vina_advance_thr=vina_advance_thr,
                    logger=logger,
                )
                gen_table_seed.append(
                    _snapshot_prudent_generation(
                        seed_idx=seed_idx,
                        generation_idx=gen_idx,
                        restart_t_upper=t_upper,
                        terminal_t_upper=tt,
                        pool=pool_ann,
                        ligand_atom_mode=ligand_atom_mode,
                        selector_cfg=selector_cfg_dyn,
                    )
                )
                pending_analysis_hits.extend(pend)
                if broke or not pool_out:
                    pool = list(pool_out) if pool_out else []
                    break
                if gen_idx == max_checkpoint_rounds - 1:
                    pool = pool_out
                    break
                pool = _prudent_expand_survivors_for_stride(
                    pool_out,
                    n_sampling,
                    seed_idx=seed_idx,
                    chk=gen_idx,
                    max_chk=max_checkpoint_rounds,
                    logger=logger,
                )
            _append_refined_for_seed(seed_idx, pool)
            generation_tables_by_seed.append(gen_table_seed)
    else:
        for seed_idx, seed_cand in enumerate(total_candidates):
            pool = [_prudent_copy_candidate(seed_cand) for _ in range(n_sampling)]
            current_t = int(time_boundary)
            for chk in range(max_checkpoint_rounds):
                restart_for_chk = int(current_t)
                grad_anchor = None if chk == 0 else int(current_t)
                pool = _prudent_stride_gpu_chunk(
                    model,
                    data,
                    pool,
                    time_upper=int(current_t),
                    time_lower=time_lower_ref,
                    stride_cap=stride_fusions,
                    grad_anchor=grad_anchor,
                    renoise_t=renoise_t,
                    refine_cfg=refine_cfg,
                    dynamic_cfg=dynamic_cfg,
                    center_pos_mode=center_pos_mode,
                    pos_only=pos_only,
                    device=device,
                    logger=logger,
                )
                tt = _prudent_terminal_time_upper(pool)
                if tt is not None:
                    current_t = int(tt)

                _prudent_apply_targetdiff_baseline_to_pool(
                    model,
                    data,
                    pool,
                    config,
                    device=device,
                    logger=logger,
                    skip=skip_targetdiff_baseline_refine,
                    log_prefix=f'seq s{seed_idx} chk{chk}',
                    round_idx=chk,
                    n_rounds=max_checkpoint_rounds,
                )

                pool, pend, broke = _prudent_score_one_round(
                    pool,
                    seed_idx=seed_idx,
                    rnd=chk,
                    n_rounds=max_checkpoint_rounds,
                    advance_top_k=advance_top_k,
                    prudent_cfg=prudent_cfg,
                    protein_path=protein_path,
                    lig_fn=lig_fn,
                    vina_protein_root=vina_protein_root,
                    ligand_atom_mode=ligand_atom_mode,
                    vina_advance_thr=vina_advance_thr,
                    logger=logger,
                )
                pending_analysis_hits.extend(pend)
                if segment_checkpoint_context is not None and not use_async_effective and not full_generation_mode:
                    legacy_snap = _snapshot_prudent_generation(
                        seed_idx=seed_idx,
                        generation_idx=chk,
                        restart_t_upper=restart_for_chk,
                        terminal_t_upper=tt,
                        pool=pool,
                        ligand_atom_mode=ligand_atom_mode,
                        selector_cfg=selector_cfg_dyn,
                    )
                    ck_path = _prudent_save_iteration_torch_checkpoint(
                        segment_checkpoint_context,
                        data=data,
                        pool=pool,
                        seed_idx=seed_idx,
                        fname_middle_tail=f'seed{int(seed_idx)}_rnd{int(chk)}',
                        checkpoint_kind='prudent_legacy_sequential_post_scoring',
                        scoring_snap=legacy_snap,
                        seg_log_row={'rnd': chk, 'scoring_snapshot': legacy_snap},
                        prudent_max_iterations=max_checkpoint_rounds,
                        ligand_atom_mode=ligand_atom_mode,
                        selector_cfg=selector_cfg_dyn,
                        logger=logger,
                    )
                    if ck_path:
                        prudent_iter_ck_paths.append(os.path.abspath(ck_path))
                if broke:
                    pool = []
                    break

                last_chk = chk == max_checkpoint_rounds - 1
                at_end = int(current_t) <= int(time_lower_ref)
                if last_chk or at_end:
                    break

                pool = _prudent_expand_survivors_for_stride(
                    pool,
                    n_sampling,
                    seed_idx=seed_idx,
                    chk=chk,
                    max_chk=max_checkpoint_rounds,
                    logger=logger,
                )

            if pool and int(current_t) > int(time_lower_ref):
                pool = _prudent_stride_gpu_chunk(
                    model,
                    data,
                    pool,
                    time_upper=int(current_t),
                    time_lower=time_lower_ref,
                    stride_cap=None,
                    grad_anchor=None,
                    renoise_t=renoise_t,
                    refine_cfg=refine_cfg,
                    dynamic_cfg=dynamic_cfg,
                    center_pos_mode=center_pos_mode,
                    pos_only=pos_only,
                    device=device,
                    logger=logger,
                )

            _append_refined_for_seed(seed_idx, pool)

    refined_pos_list = [rec['pos'] for rec in refined_records]
    refined_v_list = [rec['v'] for rec in refined_records]
    refined_pos_traj = [rec['pos_traj'] for rec in refined_records]
    refined_v_traj = [rec['v_traj'] for rec in refined_records]

    meta_baseline_refine_ti = None
    if not skip_targetdiff_baseline_refine:
        refine_out = apply_targetdiff_baseline_refinement(
            model, data, refined_pos_list, refined_v_list, config, device=device, logger=logger
        )
        if len(refine_out) == 5:
            refined_pos_list, refined_v_list, refine_pos_traj_add, refine_v_traj_add, refine_time_indices = refine_out
            refined_pos_traj = [list(pt) + list(rpt) for pt, rpt in zip(refined_pos_traj, refine_pos_traj_add)]
            refined_v_traj = [list(vt) + list(rvt) for vt, rvt in zip(refined_v_traj, refine_v_traj_add)]
            meta_baseline_refine_ti = refine_time_indices
        else:
            refined_pos_list, refined_v_list = refine_out

    meta_dict = dict(ls_out.get('meta', {}))
    meta_dict['refined_candidates'] = refined_records
    pr_legacy_meta = {
        'n_sampling': n_sampling,
        'checkpoint_every_grad_fusions': stride_fusions,
        'max_checkpoint_rounds': max_checkpoint_rounds,
        'advance_top_k': advance_top_k,
        'advance_top_k_mode': advance_top_k_mode,
        'time_boundary': time_boundary,
        'renoise_t': renoise_t,
        'first_round_max_grad_fusion_iterations': prudent_cfg.get('first_round_max_grad_fusion_iterations'),
        'max_grad_fusion_iterations': prudent_cfg.get('max_grad_fusion_iterations'),
        'grad_fusion_total_budget': prudent_cfg.get('grad_fusion_total_budget'),
        'weights': {
            'vina': prudent_cfg.get('vina_weight', 0.6),
            'qed': prudent_cfg.get('qed_weight', 0.2),
            'sa': prudent_cfg.get('sa_weight', 0.2),
        },
        'min_qed_for_docking': float(prudent_cfg.get('min_qed_for_docking', 0.3)),
        'min_sa_for_docking': float(prudent_cfg.get('min_sa_for_docking', 0.3)),
        'max_vina_affinity_for_advance': vina_advance_thr,
        'pending_analysis_hits': pending_analysis_hits,
        'async_vina_across_seeds': bool(use_async_effective),
        'async_vina_max_workers': async_max_workers if use_async_effective else None,
        'full_generation_mode': bool(full_generation_mode),
        'generation_tables_by_seed': generation_tables_by_seed,
        'generation_resume_base_t': prudent_cfg.get('generation_resume_base_t'),
        'generation_resume_t_stride': prudent_cfg.get(
            'generation_resume_t_stride', prudent_cfg.get('generation_t_stride', 0)
        ),
        'generation_grad_fusion_cap': full_gen_grad_cap,
    }
    if prudent_iter_ck_paths:
        pr_legacy_meta['iteration_checkpoint_paths'] = prudent_iter_ck_paths
    meta_dict['prudent'] = pr_legacy_meta
    if meta_baseline_refine_ti is not None:
        meta_dict['baseline_refine_time_indices'] = meta_baseline_refine_ti

    time_list = list(ls_out.get('time_list', [])) + time_records.get('refine', [])

    print(f"[DEBUG] prudent final refined_pos_list count: {len(refined_pos_list)}")
    if logger:
        logger.info(f'[Prudent] refine完成 {len(refined_pos_list)} 个分子，准备保存PT')
    return {
        'pos_list': refined_pos_list,
        'v_list': refined_v_list,
        'pos_traj': refined_pos_traj,
        'v_traj': refined_v_traj,
        'log_v_traj': [rec['log_v_traj'] for rec in refined_records],
        'time_list': time_list,
        'meta': meta_dict,
    }


def _coerce_time_indices_seq(seq):
    """将 meta 中的时间索引转为 list[int]（支持 list / numpy / torch）。"""
    if seq is None:
        return None
    if isinstance(seq, torch.Tensor):
        return [int(x) for x in seq.detach().cpu().flatten().tolist()]
    if isinstance(seq, np.ndarray):
        return [int(x) for x in seq.flatten().tolist()]
    return [int(x) for x in list(seq)]


def _get_time_indices_for_trajectory(result, config, n_steps_hint=None, sample_idx=0):
    """从 result 和 config 中提取与 pos_traj 对应的时间步序列。

    dynamic 下须合并 large_step 与 refine 两段索引；任一段缺失时仍合并另一段，避免 large（高 t）标签整体丢失。
    使用 records[sample_idx] 的 model_meta，避免多样本时错用第一条记录的调度。
    """
    meta = result.get('meta', {})
    mode = result.get('mode', 'baseline')
    num_timesteps = getattr(config.sample, 'num_steps', 1000) or 1000

    if mode in ('dynamic', 'prudent'):
        base_ti = None
        # 1) unified：model_meta.large_step_time_indices + refine_time_indices（可仅一段有值）
        records = meta.get('records', [])
        if records:
            rec = records[sample_idx] if sample_idx < len(records) else records[-1]
            model_meta = rec.get('model_meta') or {}
            large_ti = _coerce_time_indices_seq(model_meta.get('large_step_time_indices'))
            refine_ti = _coerce_time_indices_seq(model_meta.get('refine_time_indices'))
            merged = []
            if large_ti is not None:
                merged.extend(large_ti)
            if refine_ti is not None:
                merged.extend(refine_ti)
            if merged:
                base_ti = merged
        # 2) legacy：refined_candidates[].time_indices（通常仅 refine）
        if base_ti is None:
            refined = meta.get('refined_candidates', [])
            if refined:
                rc = refined[sample_idx] if sample_idx < len(refined) else refined[-1]
                ti = rc.get('time_indices')
                if ti is not None:
                    base_ti = _coerce_time_indices_seq(ti)
        # 3) 回退：按配置生成近似序列（可能与实际轨迹步数不匹配）
        if base_ti is None:
            dynamic_cfg = config.sample.get('dynamic', {})
            time_boundary = get_time_boundary(dynamic_cfg, 750)
            base_ti = list(range(num_timesteps - 1, time_boundary - 1, -1)) + list(range(time_boundary, -1, -1))
        # 4) baseline_refine 追加
        baseline_refine_ti = meta.get('baseline_refine_time_indices')
        if baseline_refine_ti is not None:
            br = _coerce_time_indices_seq(baseline_refine_ti)
            if br is not None:
                return list(base_ti) + br
        return base_ti

    if mode in ('optimization', 'dynamic_then_optimization'):
        opt_ti = meta.get('optimization_time_indices')
        if opt_ti is not None:
            base_ti = _coerce_time_indices_seq(opt_ti)
        else:
            opt_cfg = config.sample.get('optimization', {})
            start_t = int(opt_cfg.get('start_t', 300))
            n_steps = n_steps_hint or max(10, start_t)
            base_ti = np.linspace(start_t, 0, n_steps, dtype=int).tolist()
        br_ti = meta.get('baseline_refine_time_indices')
        if br_ti is not None:
            br = _coerce_time_indices_seq(br_ti)
            if br is not None:
                return base_ti + br
        return base_ti

    n_steps = n_steps_hint or num_timesteps
    return list(range(num_timesteps - 1, num_timesteps - n_steps - 1, -1))


def _step_traj_as_int_list(atom_numbers):
    """供 reconstruct 使用：get_atomic_number_from_index 已返回 list，不能调用 .tolist()。"""
    if isinstance(atom_numbers, (list, tuple)):
        return [int(x) for x in atom_numbers]
    if isinstance(atom_numbers, np.ndarray):
        return atom_numbers.astype(np.int64).flatten().tolist()
    if isinstance(atom_numbers, torch.Tensor):
        return atom_numbers.detach().cpu().flatten().long().tolist()
    return [int(x) for x in atom_numbers.tolist()]


def _step_traj_as_bool_list(aromatic_flags):
    """供 reconstruct 使用的芳香标记列表。"""
    if aromatic_flags is None:
        return None
    if isinstance(aromatic_flags, (list, tuple)):
        return [bool(x) for x in aromatic_flags]
    if isinstance(aromatic_flags, np.ndarray):
        return aromatic_flags.astype(bool).flatten().tolist()
    if isinstance(aromatic_flags, torch.Tensor):
        return aromatic_flags.detach().cpu().flatten().bool().tolist()
    return [bool(x) for x in aromatic_flags.tolist()]


def _align_time_indices_to_n_frames(time_indices, n_frames):
    """将 t 标签序列截断或填充到与 pos_traj 帧数一致。"""
    ti = list(time_indices) if time_indices is not None else []
    if n_frames <= 0:
        return []
    if len(ti) == n_frames:
        return ti
    if len(ti) > n_frames:
        # 标签多于帧时取尾部：兼容「仅保存 refine 段」而 meta 仍含 large+refine 的旧结果，避免 t 与帧错位
        return ti[-n_frames:]
    pad = ti[-1] if ti else 0
    return ti + [pad] * (n_frames - len(ti))


def _merge_step_sdfs_for_pymol(mol_dir, out_filename='trajectory_pymol.sdf', logger=None):
    """将单步 SDF 按文件名排序合并为单文件多记录 SDF，供 PyMOL 中 load 后以 state 切换播放。

    不修改各 step_*.sdf；合并文件与单步文件同目录。
    """
    files = sorted(mol_dir.glob('step_*_t*.sdf'))
    if not files:
        return False
    out_path = mol_dir / out_filename
    try:
        writer = Chem.SDWriter(str(out_path))
        try:
            for fp in files:
                suppl = Chem.SDMolSupplier(str(fp), sanitize=False, removeHs=False)
                mol = next(iter(suppl), None)
                if mol is None:
                    if logger:
                        logger.warning('[StepTrajectory] 合并跳过无法读取: %s', fp.name)
                    continue
                mol.SetProp('_Name', fp.stem)
                writer.write(mol)
        finally:
            writer.close()
    except Exception as e:
        if logger:
            logger.warning('[StepTrajectory] 合并 PyMOL 多帧 SDF 失败: %s', e)
        return False
    if logger:
        logger.info('[StepTrajectory] PyMOL 多帧轨迹: %s (%d 帧)', out_path, len(files))
    return True


def save_step_trajectory_sdf(result, result_path, config, ligand_atom_mode, pocket_id=None, timestamp=None, logger=None):
    """将采样轨迹按步保存为 SDF 文件。

    每个分子一个子文件夹，按分子身份证命名：蛋白质ID_生成时间_分子评分_序号（如 1A4K_20240101_123456_88p89_001）。
    格式与 utils.molecule_id.generate_molecule_id 一致。
    t > bond_threshold 时仅保存原子位置（无键），t <= bond_threshold 时成键并保存完整 SDF。
    merge_pymol_sdf 为 true 时，每个分子子目录下额外写入单文件多记录 SDF（默认 trajectory_pymol.sdf），便于 PyMOL 加载后按 state 播放。

    Args:
        result: 采样结果字典，含 pred_ligand_pos_traj, pred_ligand_v_traj, meta, data
        result_path: 主结果目录（轨迹写入路径由 output_dir 决定：相对路径相对于仓库根目录）
        config: 采样配置
        ligand_atom_mode: 配体原子编码模式
        pocket_id: 口袋/数据编号（data_id 或 'custom'），protein_id 提取失败时用作回退
        timestamp: 生成时间（YYYYMMDD_HHMMSS 字符串或 datetime），用于分子身份证
        logger: 可选日志
    """
    cfg = config.sample.get('step_trajectory_save', {})
    if not cfg.get('enable', False):
        return

    output_dir = cfg.get('output_dir', './diffway')
    bond_threshold = int(cfg.get('bond_threshold', 300))
    merge_pymol_sdf = cfg.get('merge_pymol_sdf', True)
    merge_sdf_filename = str(cfg.get('merge_sdf_filename', 'trajectory_pymol.sdf'))

    _out = Path(output_dir)
    root = _out if _out.is_absolute() else (REPO_ROOT / _out)
    root.mkdir(parents=True, exist_ok=True)

    pos_traj_list = result.get('pred_ligand_pos_traj', [])
    v_traj_list = result.get('pred_ligand_v_traj', [])

    if not pos_traj_list:
        if logger:
            logger.warning('[StepTrajectory] 无轨迹数据，跳过跳步留存')
        return

    n_samples = len(pos_traj_list)

    # 提取蛋白质 ID（口袋）和生成时间，用于分子身份证
    data = result.get('data')
    protein_filename = getattr(data, 'protein_filename', None) if data is not None else None
    ligand_filename = getattr(data, 'ligand_filename', None) if data is not None else None
    protein_id = extract_protein_id(ligand_filename=ligand_filename, protein_filename=protein_filename)
    if protein_id == 'UNKNOWN' and pocket_id is not None:
        protein_id = f'P{pocket_id}'  # 回退：用口袋编号
    generation_time = timestamp if timestamp else datetime.now(timezone(timedelta(hours=8)))
    meta = result.get('meta', {})
    meta_records = meta.get('records') or meta.get('refined_candidates', [])

    for sample_idx in range(n_samples):
        pos_traj = pos_traj_list[sample_idx]
        v_traj = v_traj_list[sample_idx] if sample_idx < len(v_traj_list) else None

        if not pos_traj:
            continue

        # 从 meta.records 获取 QED/SA 计算初步评分（0-100），无则用 0
        rec = meta_records[sample_idx] if sample_idx < len(meta_records) else {}
        metrics = rec.get('metrics', {})
        qed = rec.get('qed') if 'qed' in rec else metrics.get('qed')
        sa = rec.get('sa') if 'sa' in rec else metrics.get('sa')
        if qed is not None and sa is not None and not (np.isnan(qed) or np.isnan(sa)):
            score = (float(qed) + float(sa)) / 2.0 * 100.0
        else:
            score = 0.0

        mol_id_base = generate_molecule_id(protein_id, generation_time, score)
        mol_id = f'{mol_id_base}_{sample_idx + 1:03d}'  # 加序号保证唯一
        mol_dir = root / mol_id
        mol_dir.mkdir(parents=True, exist_ok=True)

        time_indices = _get_time_indices_for_trajectory(
            result, config, n_steps_hint=len(pos_traj), sample_idx=sample_idx
        )
        if (
            logger
            and time_indices is not None
            and len(time_indices) != len(pos_traj)
        ):
            logger.warning(
                '[StepTrajectory] 样本 %d：time_indices 长度 (%d) 与轨迹帧数 (%d) 不一致，'
                '已截断或末项 t 填充；文件名中 t 可能与采样步不完全一致。',
                sample_idx,
                len(time_indices),
                len(pos_traj),
            )

        ti_for_mol = _align_time_indices_to_n_frames(time_indices, len(pos_traj))

        for step_idx, pos in enumerate(pos_traj):
            pos_array = np.asarray(pos, dtype=np.float64)
            if pos_array.ndim == 3:
                pos_array = pos_array[0]
            if pos_array.size == 0:
                continue

            t = ti_for_mol[step_idx] if step_idx < len(ti_for_mol) else step_idx

            if v_traj is not None and step_idx < len(v_traj):
                v_array = np.asarray(v_traj[step_idx])
                if v_array.ndim == 2:
                    v_array = v_array[0] if v_array.shape[0] == 1 else v_array
                v_tensor = torch.tensor(v_array, dtype=torch.long)
                atom_numbers = trans.get_atomic_number_from_index(v_tensor, mode=ligand_atom_mode)
                aromatic_flags = trans.is_aromatic_from_index(v_tensor, mode=ligand_atom_mode)
            else:
                atom_numbers = np.ones(pos_array.shape[0], dtype=np.int64) * 6
                aromatic_flags = None

            out_name = mol_dir / f'step_{step_idx:04d}_t{t}.sdf'

            try:
                if t <= bond_threshold:
                    aromatic = _step_traj_as_bool_list(aromatic_flags)
                    mol = reconstruct.reconstruct_from_generated(
                        pos_array.tolist(),
                        _step_traj_as_int_list(atom_numbers),
                        aromatic=aromatic,
                        basic_mode=(aromatic is None)
                    )
                    if mol is not None:
                        Chem.MolToMolFile(mol, str(out_name))
                else:
                    reconstruct.save_positions_only_to_sdf(
                        pos_array, atom_numbers, str(out_name)
                    )
            except reconstruct.MolReconsError:
                reconstruct.save_positions_only_to_sdf(pos_array, atom_numbers, str(out_name))
            except Exception as e:
                if logger:
                    logger.warning(
                        f'[StepTrajectory] 保存 step {step_idx} t={t} 成键失败，回退为仅坐标 SDF: {e}'
                    )
                try:
                    reconstruct.save_positions_only_to_sdf(pos_array, atom_numbers, str(out_name))
                except Exception as e2:
                    if logger:
                        logger.warning(f'[StepTrajectory] 回退保存仍失败 step {step_idx} t={t}: {e2}')

        if merge_pymol_sdf:
            _merge_step_sdfs_for_pymol(mol_dir, out_filename=merge_sdf_filename, logger=logger)

    if logger:
        logger.info(f'[StepTrajectory] 跳步留存已保存至: {root} (共 {n_samples} 个分子)')


def select_top_candidates(candidates, top_n):  # 根据评分选出 top-N 候选。
    """按照综合得分排序候选分子，返回排名靠前的候选集合。

    Args:
        candidates: 由 `evaluate_candidate` 生成的候选列表。
        top_n: 需要保留的候选数量，若小于等于 0 则返回空列表。

    Returns:
        list[dict]: 经过排序与补充后的候选列表（长度不超过 `top_n`）。
    """
    if top_n <= 0:
        return []
    valid = [c for c in candidates if np.isfinite(c['score'])]  # 过滤有限分数的候选。
    if not valid:
        valid = candidates  # 若全部无效，则退回所有候选。
    sorted_candidates = sorted(valid, key=lambda x: x['score'])  # 按分数升序排序。
    if len(sorted_candidates) >= top_n:
        return sorted_candidates[:top_n]  # 返回前 top-N。
    # 补充剩余候选：如果有效候选不足 top_n，则从无效候选中选择补充。
    sorted_ids = {id(c) for c in sorted_candidates}  # 使用对象 id 避免 numpy 比较歧义。
    remaining = [c for c in candidates if id(c) not in sorted_ids]  # 找出未排序的候选。
    sorted_remaining = sorted(remaining, key=lambda x: x['score'])
    combined = sorted_candidates + sorted_remaining  # 合并两部分。
    return combined[:min(len(combined), top_n)]  # 返回最多 top-N 个候选。


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
    # 创建GPU监控器
    monitor = GPUMonitor(device=device, enable_flops=False)
    profiler = MemoryProfiler(device=device)
    
    dynamic_cfg = config.sample.get('dynamic', {})  # 读取整体动态配置。
    large_cfg = dynamic_cfg.get('large_step', {})  # 大步探索阶段配置。
    refine_cfg = dynamic_cfg.get('refine', {})  # 精炼阶段配置。
    selector_cfg = dynamic_cfg.get('selector', {})  # 候选筛选配置。

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
    
    if logger:
        logger.info(f'[Dynamic] Large-step batch size: {large_batch_size} | repeats: {n_repeat}')
        if n_repeat == 0:
            logger.warning('n_repeat is 0, no large-step sampling will be performed')

    total_candidates = []  # 存储所有候选。
    range_offset = 0  # 范围模式的原子偏移。
    time_records = {'large_step': [], 'refine': []}  # 记录各阶段耗时。
    
    profiler.checkpoint('before_large_step')

    for repeat_idx in range(n_repeat):  # 逐次执行大步探索。
        batch = Batch.from_data_list([data.clone() for _ in range(large_batch_size)],
                                     follow_batch=FOLLOW_BATCH).to(device)  # 构建批次。
        n_data = large_batch_size  # 当前批包含的样本数。
        batch_protein = batch.protein_element_batch  # 直接复用 PyG 创建的批次索引
        _validate_batch_indices(batch_protein, n_data, 'batch_protein',
                                logger=logger, context='legacy_large_step')

        if sample_num_atoms_mode == 'prior':  # 根据 pocket 尺寸采样原子数。
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
        init_ligand_pos = center[batch_ligand] + torch.randn_like(center[batch_ligand])  # 初始化配体位置。

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
            }
            # large_step阶段不进行筛选，只收集候选
            total_candidates.append(candidate)  # 收集候选。

    # large_step阶段完成，不进行筛选（删除在time_boundary的筛选功能）
    if logger:
        logger.info(f'[Dynamic] Large-step completed | Total candidates: {len(total_candidates)}')

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
            metric_info = evaluate_candidate(pos_final, v_final, ligand_atom_mode, selector_cfg)
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
                raw_stage1 = _batch_refinement_forward_split(
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
                )
            else:
                raw_stage1 = []
            for raw, (cand_idx, refine_idx) in zip(raw_stage1, stage1_flat_meta):
                pos_intermediate = raw['pos']
                v_intermediate = raw['v']
                log_v_intermediate = raw['log_v']
                num_atoms = raw['num_atoms']
                intermediate_candidate = {
                    'pos': pos_intermediate,
                    'v': v_intermediate,
                    'log_v': log_v_intermediate,
                    'num_atoms': num_atoms,
                    'source_cand_idx': cand_idx,
                    'refine_idx': refine_idx,
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
                raw_stage2 = _batch_refinement_forward_split(
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

            metric_info = evaluate_candidate(pos_final, v_final, ligand_atom_mode, selector_cfg)
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
        refine_out = apply_targetdiff_baseline_refinement(
            model, data, refined_pos_list, refined_v_list, config, device=device, logger=logger
        )
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
    if meta_baseline_refine_ti is not None:
        meta_dict['baseline_refine_time_indices'] = meta_baseline_refine_ti
    # Flatten time records to match baseline expectation
    time_list = time_records['large_step'] + time_records['refine']  # 合并耗时记录。

    return {
        'pos_list': refined_pos_list,
        'v_list': refined_v_list,
        'pos_traj': refined_pos_traj,
        'v_traj': refined_v_traj,
        'log_v_traj': [rec['log_v_traj'] for rec in refined_records],
        'time_list': time_list,
        'meta': meta_dict
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
        tuple: 包含采样坐标、类别、完整轨迹及耗时的七元组。
    """
    # 创建GPU监控器
    monitor = GPUMonitor(device=device, enable_flops=False)
    profiler = MemoryProfiler(device=device)
    
    # 验证 batch_size 和 num_samples 参数
    batch_size = max(1, int(batch_size))  # 确保至少为1
    num_samples = max(1, int(num_samples))  # 确保至少为1
    
    all_pred_pos, all_pred_v = [], []  # 累积最终坐标与类别。
    all_pred_pos_traj, all_pred_v_traj = [], []  # 累积轨迹。
    all_pred_v0_traj, all_pred_vt_traj = [], []  # 累积初/末时间轨迹。
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
    
    return all_pred_pos, all_pred_v, all_pred_pos_traj, all_pred_v_traj, all_pred_v0_traj, all_pred_vt_traj, time_list


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


# =============================================================================
# 骨架约束进化优化（Scaffold-Constrained Evolutionary Optimization）
# =============================================================================
# 设计思路：
#   阶段 1 — 双掩码骨架约束：
#       骨架原子 pos_mask=1 → 每步去噪后将骨架坐标拉回参考前向扩散态（位置固定）
#       type_mask=null      → 骨架原子元素不约束，可自由演化（允许 bioisostere 替换）
#       非骨架原子           → 位置和元素均完全自由
#   阶段 2 — DiffSBDD 式进化算法 + 改进：
#       · 维护候选种群（population_size）
#       · 噪声退火（前几代高噪声探索，后几代低噪声精炼）
#       · 多目标适应度：QED + SA + 多样性奖励 - 骨架RMSD惩罚
#       · 多样性筛选（Morgan 指纹 Tanimoto 阈值），防止种群塌缩
# =============================================================================

def _detect_murcko_scaffold_indices(mol) -> list:
    """
    使用 Murcko 分解自动识别分子主骨架原子索引。

    Murcko 骨架定义（Bemis-Murcko, 1996）：
    - 保留所有环系（ring systems）
    - 保留连接各环系的链接骨架（linker atoms）
    - 移除所有侧链（side chains / R-groups）

    算法：
    1. 计算 Murcko 骨架分子
    2. 用子结构匹配（GetSubstructMatches）将骨架原子映射回原始分子索引

    Args:
        mol: RDKit Mol 对象（需已添加氢原子以外的完整分子）。

    Returns:
        list[int]: 骨架原子的原始 0-based 索引列表。
                   若失败（无环等）返回空列表。
    """
    try:
        from rdkit.Chem.Scaffolds import MurckoScaffold
        
        # 确保分子有 RingInfo
        try:
            ri = mol.GetRingInfo()
            if ri is None or ri.NumRings() < 0:
                Chem.GetSSSR(mol)
        except:
            pass
            
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        if scaffold is None or scaffold.GetNumAtoms() == 0:
            return []
        matches = mol.GetSubstructMatches(scaffold)
        if not matches:
            return []
        return list(matches[0])
    except Exception as e:
        if logger and 'logger' in locals():
            logger.warning(f'[_detect_murcko_scaffold_indices] 失败: {e}')
        return []


def _detect_generic_murcko_scaffold_indices(mol) -> list:
    """
    使用「通用 Murcko 骨架」（Generic Murcko Scaffold）识别骨架原子。

    通用骨架将所有原子替换为碳，所有键替换为单键，仅保留拓扑结构，
    对杂环分子（含 N/O/S）也能提供稳定的骨架识别。
    与 _detect_murcko_scaffold_indices 配合使用：优先尝试普通 Murcko，
    若原子数 0 再尝试通用骨架。

    Args:
        mol: RDKit Mol 对象。

    Returns:
        list[int]: 骨架原子的原始 0-based 索引列表。
    """
    try:
        from rdkit.Chem.Scaffolds import MurckoScaffold
        
        # 确保分子有 RingInfo
        try:
            ri = mol.GetRingInfo()
            if ri is None or ri.NumRings() < 0:
                Chem.GetSSSR(mol)
        except:
            pass
            
        generic = MurckoScaffold.MakeScaffoldGeneric(
            MurckoScaffold.GetScaffoldForMol(mol)
        )
        if generic is None or generic.GetNumAtoms() == 0:
            return []
        matches = mol.GetSubstructMatches(generic)
        if not matches:
            return []
        return list(matches[0])
    except Exception as e:
        if logger and 'logger' in locals():
            logger.warning(f'[_detect_generic_murcko_scaffold_indices] 失败: {e}')
        return []


def _detect_ring_based_scaffold_indices(mol, logger=None) -> list:
    """
    基于环检测的简化骨架提取算法（MurckoScaffold 的替代方案）。
    不依赖 MurckoScaffold，直接检测环结构作为骨架。
    
    算法：
    1. 找到所有环原子（使用 SSSR - Smallest Set of Smallest Rings）
    2. 添加连接环的链接原子（linker atoms）
    3. 保留环原子和链接原子作为骨架，移除侧链原子
    
    Args:
        mol: RDKit Mol 对象。
        logger: 可选的日志记录器。
    
    Returns:
        list[int]: 骨架原子的原始 0-based 索引列表。失败时返回空列表。
    """
    try:
        # 确保 RingInfo 已初始化
        try:
            ri = mol.GetRingInfo()
            if ri is None or ri.NumRings() < 0:
                Chem.GetSSSR(mol)
                ri = mol.GetRingInfo()
        except Exception as ring_err:
            if logger:
                logger.warning(f'[_detect_ring_based] RingInfo 初始化失败: {ring_err}')
            return []
        
        if not ri or ri.NumRings() < 0:
            if logger:
                logger.warning('[_detect_ring_based] RingInfo 未初始化')
            return []
        
        # 获取所有环原子
        ring_atoms = set()
        atom_rings = ri.AtomRings()
        
        if not atom_rings:
            if logger:
                logger.info('[_detect_ring_based] 分子中没有环结构')
            return []
        
        for ring in atom_rings:
            for atom_idx in ring:
                ring_atoms.add(atom_idx)
        
        if not ring_atoms:
            return []
        
        if logger:
            logger.info(f'[_detect_ring_based] 检测到 {len(atom_rings)} 个环，{len(ring_atoms)} 个环原子')
        
        # 找到连接环的链接原子（linker atoms）
        # 链接原子定义：连接两个不同环的路径上的原子
        linker_atoms = set()
        
        for atom_idx in range(mol.GetNumAtoms()):
            if atom_idx in ring_atoms:
                continue
            
            atom = mol.GetAtomWithIdx(atom_idx)
            neighbors = atom.GetNeighbors()
            
            # 检查该原子的邻居属于哪些环
            neighbor_rings = set()
            for neighbor in neighbors:
                neighbor_idx = neighbor.GetIdx()
                # 检查邻居是否在环中
                for ring_idx, ring in enumerate(atom_rings):
                    if neighbor_idx in ring:
                        neighbor_rings.add(ring_idx)
            
            # 如果邻居属于 2 个或更多不同的环，则该原子是链接原子
            if len(neighbor_rings) >= 2:
                linker_atoms.add(atom_idx)
        
        if logger and linker_atoms:
            logger.info(f'[_detect_ring_based] 检测到 {len(linker_atoms)} 个链接原子')
        
        # 组合环原子和链接原子作为骨架
        scaffold_atoms = ring_atoms.union(linker_atoms)
        
        # 确保至少保留了一些原子
        if not scaffold_atoms:
            if logger:
                logger.warning('[_detect_ring_based] 未找到任何骨架原子')
            return []
        
        if logger:
            logger.info(f'[_detect_ring_based] 最终骨架: {len(scaffold_atoms)} 个原子（{len(ring_atoms)} 环 + {len(linker_atoms)} 链接）')
        
        return sorted(list(scaffold_atoms))
        
    except Exception as e:
        if logger:
            logger.warning(f'[_detect_ring_based] 骨架提取失败: {e}')
        return []


def _extract_scaffold_submol(ref_mol, scaffold_indices, logger=None):
    """
    从参考分子中提取骨架原子的子结构并创建新的 RDKit 分子。
    
    Args:
        ref_mol: 原始参考分子 (RDKit Mol 对象)
        scaffold_indices: 骨架原子索引列表 (0-based)
        logger: 可选的日志记录器
    
    Returns:
        Chem.Mol: 包含骨架原子和内部键的新分子，失败时返回 None
    """
    if ref_mol is None or not scaffold_indices:
        return None
    
    try:
        # 创建可编辑分子
        scaffold_indices_set = set(scaffold_indices)
        
        # 使用 RDKit 的 GetSubmol 方法提取子结构（保留键连接信息）
        # 注意：GetSubmol 会自动包含连接骨架原子的键
        atom_map = {}  # 原始索引 -> 新索引
        emol = Chem.EditableMol(Chem.Mol())
        
        # 添加原子
        for old_idx in scaffold_indices:
            if old_idx < ref_mol.GetNumAtoms():
                atom = ref_mol.GetAtomWithIdx(old_idx)
                new_idx = emol.AddAtom(atom)
                atom_map[old_idx] = new_idx
        
        # 添加骨架原子之间的键（只保留两端都在骨架中的键）
        for bond in ref_mol.GetBonds():
            begin_idx = bond.GetBeginAtomIdx()
            end_idx = bond.GetEndAtomIdx()
            
            # 只保留骨架原子之间的键
            if begin_idx in scaffold_indices_set and end_idx in scaffold_indices_set:
                begin_new = atom_map[begin_idx]
                end_new = atom_map[end_idx]
                bond_type = bond.GetBondType()
                emol.AddBond(begin_new, end_new, bond_type)
        
        # 创建最终分子
        scaffold_mol = emol.GetMol()
        
        # 设置 3D 坐标（如果有）
        if ref_mol.GetNumConformers() > 0:
            conf = ref_mol.GetConformer()
            new_conf = Chem.Conformer(len(scaffold_indices))
            
            for old_idx, new_idx in atom_map.items():
                pos = conf.GetAtomPosition(old_idx)
                new_conf.SetAtomPosition(new_idx, pos)
            
            scaffold_mol.AddConformer(new_conf)
        
        # 尝试计算 2D 坐标（用于可视化）
        try:
            Chem.Compute2DCoords(scaffold_mol)
        except Exception:
            pass
        
        return scaffold_mol
        
    except Exception as e:
        if logger:
            logger.warning(f'[_extract_scaffold_submol] 提取骨架失败: {e}')
        return None


def _resolve_scaffold_mask(
    scaffold_cfg: dict,
    ref_ligand_pos,
    ref_ligand_v,
    ligand_atom_mode: str,
    num_atoms: int,
    device,
) -> 'torch.Tensor':
    """
    根据配置解析骨架掩码（[num_atoms] float32 张量，1=骨架原子，0=非骨架原子）。

    支持五种定义方式：
    - atom_indices  ：直接指定 0-based 原子索引列表
    - smarts        ：通过 SMARTS 子结构匹配（支持多个匹配，全部标记）
    - auto_murcko   ：Bemis-Murcko 分解自动识别主骨架（环系 + 链接体）
    - auto_murcko_generic：通用 Murcko 骨架（全碳骨架，对杂环更鲁棒）
    - none          ：不约束任何原子（返回全零掩码）

    Args:
        scaffold_cfg: scaffold_optimization / scaffold_grow 配置字典。
        ref_ligand_pos: 参考配体坐标 [N,3] Tensor。
        ref_ligand_v: 参考配体原子类型 [N] Tensor。
        ligand_atom_mode: 原子编码模式（用于重建 RDKit Mol）。
        num_atoms: 原子总数。
        device: 计算设备。

    Returns:
        scaffold_mask: float32 Tensor [num_atoms]，1=骨架，0=非骨架。
    """
    source = scaffold_cfg.get('scaffold_source', 'none')
    mask_np = np.zeros(num_atoms, dtype=np.float32)

    def _try_get_mol():
        """从坐标和原子类型创建基本 RDKit 分子（使用距离阈值构建键，保留环结构）"""
        try:
            pos_np = ref_ligand_pos.detach().cpu().numpy() if hasattr(ref_ligand_pos, 'detach') else np.array(ref_ligand_pos)
            v_np = ref_ligand_v.detach().cpu().numpy() if hasattr(ref_ligand_v, 'detach') else np.array(ref_ligand_v)

            # 直接创建基本分子，避免使用可能有bug的重建函数
            mol = Chem.RWMol()
            atomic_nums = []

            # 典型共价键长度上界 (Å)，超过此距离不建键
            COVALENT_RADIUS = {6: 0.77, 7: 0.75, 8: 0.73, 9: 0.71, 15: 1.10, 16: 1.02, 17: 0.99, 35: 1.14, 53: 1.33}
            BOND_DIST_FACTOR = 1.3  # 容差系数

            for i in range(len(v_np)):
                atom_type = int(v_np[i])
                if atom_type == 0: atomic_num = 6
                elif atom_type == 1: atomic_num = 7
                elif atom_type == 2: atomic_num = 8
                elif atom_type == 3: atomic_num = 9
                elif atom_type == 4: atomic_num = 15
                elif atom_type == 5: atomic_num = 16
                elif atom_type == 6: atomic_num = 17
                elif atom_type == 7: atomic_num = 35
                elif atom_type == 8: atomic_num = 53
                else: atomic_num = 6

                atomic_nums.append(atomic_num)
                atom = Chem.Atom(atomic_num)
                mol.AddAtom(atom)

            # 基于 3D 距离构建键（保留环结构）
            n = len(atomic_nums)
            # 最大允许价态
            MAX_VALENCE = {6: 4, 7: 3, 8: 2, 9: 1, 15: 5, 16: 6, 17: 1, 35: 1, 53: 1}
            bond_pairs = []
            for i in range(n):
                ri = COVALENT_RADIUS.get(atomic_nums[i], 0.77)
                for j in range(i + 1, n):
                    rj = COVALENT_RADIUS.get(atomic_nums[j], 0.77)
                    threshold = (ri + rj) * BOND_DIST_FACTOR
                    dx = pos_np[i][0] - pos_np[j][0]
                    dy = pos_np[i][1] - pos_np[j][1]
                    dz = pos_np[i][2] - pos_np[j][2]
                    dist = (dx*dx + dy*dy + dz*dz) ** 0.5
                    if dist < threshold:
                        bond_pairs.append((i, j, dist))
            # 按距离排序，优先添加短键
            bond_pairs.sort(key=lambda x: x[2])
            atom_bond_count = [0] * n
            for i, j, dist in bond_pairs:
                max_i = MAX_VALENCE.get(atomic_nums[i], 4)
                max_j = MAX_VALENCE.get(atomic_nums[j], 4)
                if atom_bond_count[i] < max_i and atom_bond_count[j] < max_j:
                    try:
                        mol.AddBond(i, j, Chem.BondType.SINGLE)
                        atom_bond_count[i] += 1
                        atom_bond_count[j] += 1
                    except:
                        pass

            final_mol = mol.GetMol()

            # 设置 3D 坐标
            conf = Chem.Conformer(n)
            for i, (x, y, z) in enumerate(pos_np):
                conf.SetAtomPosition(i, (float(x), float(y), float(z)))
            final_mol.AddConformer(conf)

            if logger:
                logger.info(f'[_try_get_mol] 成功创建分子: {final_mol.GetNumAtoms()} 原子, {final_mol.GetNumBonds()} 键')
            return final_mol
            
        except Exception as e:
            if logger:
                logger.warning(f'[_try_get_mol] 创建基本分子失败: {e}')
            return None

    if source == 'none':
        pass

    elif source == 'atom_indices':
        indices = scaffold_cfg.get('scaffold_atom_indices') or []
        for idx in indices:
            if 0 <= int(idx) < num_atoms:
                mask_np[int(idx)] = 1.0

    elif source == 'smarts':
        smarts = scaffold_cfg.get('scaffold_smarts')
        if smarts is not None:
            try:
                pattern = Chem.MolFromSmarts(smarts)
                mol = _try_get_mol()
                if mol is not None and pattern is not None:
                    matches = mol.GetSubstructMatches(pattern)
                    for match in matches:
                        for atom_idx in match:
                            if 0 <= atom_idx < num_atoms:
                                mask_np[atom_idx] = 1.0
            except Exception:
                pass

    elif source in ('auto_murcko', 'auto_murcko_generic'):
        try:
            mol = _try_get_mol()
            if mol is not None:
                if logger:
                    logger.info(f'[ScaffoldMask] 重建分子成功: {mol.GetNumAtoms()} 个原子')

                # 优先使用环检测法（更严格，保留更多生成自由度）
                indices = _detect_ring_based_scaffold_indices(mol, logger)
                if logger:
                    logger.info(f'[ScaffoldMask] 环检测骨架: {len(indices)} 个骨架原子')

                # 环检测失败时回退到 Murcko
                if not indices:
                    if source == 'auto_murcko':
                        indices = _detect_murcko_scaffold_indices(mol)
                        if logger:
                            logger.info(f'[ScaffoldMask] Murcko 回退: {len(indices)} 个骨架原子')
                        if not indices:
                            indices = _detect_generic_murcko_scaffold_indices(mol)
                            if logger:
                                logger.info(f'[ScaffoldMask] 通用 Murcko 回退: {len(indices)} 个骨架原子')
                    else:
                        indices = _detect_generic_murcko_scaffold_indices(mol)
                        if logger:
                            logger.info(f'[ScaffoldMask] 通用 Murcko: {len(indices)} 个骨架原子')

                for atom_idx in indices:
                    if 0 <= atom_idx < num_atoms:
                        mask_np[atom_idx] = 1.0
            else:
                if logger:
                    logger.warning('[ScaffoldMask] 重建分子失败，无法提取骨架')
        except Exception as e:
            if logger:
                logger.warning(f'[ScaffoldMask] 骨架提取异常: {e}')
            pass

    return torch.tensor(mask_np, dtype=torch.float32, device=device)


def _compute_scaffold_fitness(
    pos_np: np.ndarray,
    v_np: np.ndarray,
    ligand_atom_mode: str,
    scaffold_cfg: dict,
    ref_pos_np: np.ndarray,
    scaffold_mask_np: np.ndarray,
    existing_mols_smiles: list,
) -> dict:
    """
    多目标适应度评分（越高越好）。

    fitness = qed_w * QED + sa_w * SA
              + div_w * diversity_bonus
              - rmsd_w * scaffold_RMSD

    多样性奖励：基于 Morgan 指纹，计算当前分子与已选种群的平均 Tanimoto 距离。
    种群多样性越高（越不像已有成员），奖励越大；防止种群塌缩。

    Args:
        pos_np: 当前分子坐标 [N,3]。
        v_np: 当前分子原子类型 [N]。
        ligand_atom_mode: 原子编码模式。
        scaffold_cfg: scaffold_optimization 配置。
        ref_pos_np: 参考分子坐标（用于 RMSD）。
        scaffold_mask_np: 骨架掩码 [N]，1=骨架原子。
        existing_mols_smiles: 当前种群中已有分子的 SMILES 列表（多样性对比用）。

    Returns:
        dict with keys: fitness, qed, sa, scaffold_rmsd, diversity_bonus, smiles, status
    """
    from rdkit.Chem import Descriptors, QED
    from rdkit.Chem import AllChem
    from rdkit import DataStructs

    result = {
        'fitness': -float('inf'),
        'qed': 0.0,
        'sa': 0.0,
        'scaffold_rmsd': float('inf'),
        'diversity_bonus': 0.0,
        'smiles': None,
        'status': 'failed',
    }

    try:
        mol = reconstruct.reconstruct_from_generated(pos_np, v_np, ligand_atom_mode)
    except Exception:
        mol = None

    if mol is None:
        result['status'] = 'reconstruct_failed'
        return result

    smiles = Chem.MolToSmiles(mol)
    result['smiles'] = smiles

    # 完整性检查（无断键）
    if scaffold_cfg.get('filter_incomplete', True):
        frags = smiles.split('.')
        if len(frags) > 1:
            result['status'] = 'incomplete'
            return result

    # QED/SA
    try:
        qed_val = QED.qed(mol)
    except Exception:
        qed_val = 0.0
    try:
        sa_val = scoring_func.compute_sa_score(mol)
    except Exception:
        sa_val = 0.0

    result['qed'] = float(qed_val)
    result['sa'] = float(sa_val)

    # QED/SA 阈值过滤
    min_qed = scaffold_cfg.get('min_qed', 0.0)
    min_sa = scaffold_cfg.get('min_sa', 0.0)
    if qed_val < min_qed:
        result['status'] = 'filtered_qed'
        return result
    if sa_val < min_sa:
        result['status'] = 'filtered_sa'
        return result

    # 骨架 RMSD（仅统计骨架原子）
    scaffold_mask_flat = scaffold_mask_np.astype(bool)
    if scaffold_mask_flat.any() and pos_np.shape[0] == ref_pos_np.shape[0]:
        rmsd = float(np.sqrt(np.mean((pos_np[scaffold_mask_flat] - ref_pos_np[scaffold_mask_flat]) ** 2)))
    else:
        rmsd = float(np.sqrt(np.mean((pos_np - ref_pos_np) ** 2)))
    result['scaffold_rmsd'] = rmsd

    # 多样性奖励：与种群现有成员的 Tanimoto 距离均值
    div_bonus = 0.0
    div_filter_cfg = scaffold_cfg.get('diversity_filter', {})
    fp_radius = int(div_filter_cfg.get('fingerprint_radius', 2))
    fp_nbits = int(div_filter_cfg.get('fingerprint_nbits', 2048))
    if existing_mols_smiles:
        try:
            fp_cur = AllChem.GetMorganFingerprintAsBitVect(mol, fp_radius, nBits=fp_nbits)
            distances = []
            for smi in existing_mols_smiles:
                ref_m = Chem.MolFromSmiles(smi)
                if ref_m is not None:
                    fp_ref = AllChem.GetMorganFingerprintAsBitVect(ref_m, fp_radius, nBits=fp_nbits)
                    sim = DataStructs.TanimotoSimilarity(fp_cur, fp_ref)
                    distances.append(1.0 - sim)
            if distances:
                div_bonus = float(np.mean(distances))
        except Exception:
            div_bonus = 0.0
    result['diversity_bonus'] = div_bonus

    # 综合适应度
    qed_w = float(scaffold_cfg.get('qed_weight', 1.0))
    sa_w = float(scaffold_cfg.get('sa_weight', 1.0))
    div_w = float(scaffold_cfg.get('diversity_weight', 0.3))
    rmsd_w = float(scaffold_cfg.get('rmsd_penalty_weight', 0.0))
    fitness = qed_w * qed_val + sa_w * sa_val + div_w * div_bonus - rmsd_w * rmsd
    result['fitness'] = fitness
    result['status'] = 'ok'
    return result


def _diversity_filter_population(
    candidates: list,
    population_size: int,
    scaffold_cfg: dict,
) -> list:
    """
    多样性感知种群选择：贪心选取前 population_size 个，保证 Tanimoto 相似度低于阈值。

    算法：
    1. 按 fitness 降序排列所有候选
    2. 贪心地加入新成员：若与已选任一成员 Tanimoto 相似度 > max_tanimoto，跳过
    3. 若贪心未选满 population_size（常见于 evolve 子代 SMILES 极相似），再按 fitness 序补位，
       保证在「有足够 status=ok 候选」时尽量达到 population_size，避免种群塌缩为 1 条。

    Args:
        candidates: 候选字典列表，每个含 'fitness'、'smiles' 等键。
        population_size: 目标种群大小。
        scaffold_cfg: scaffold_optimization 配置。

    Returns:
        selected: 经多样性筛选后的种群列表，按 fitness 降序。
    """
    from rdkit.Chem import AllChem
    from rdkit import DataStructs

    div_filter_cfg = scaffold_cfg.get('diversity_filter', {})
    enable_div = div_filter_cfg.get('enable', True)
    max_tanimoto = float(div_filter_cfg.get('max_tanimoto', 0.85))
    fp_radius = int(div_filter_cfg.get('fingerprint_radius', 2))
    fp_nbits = int(div_filter_cfg.get('fingerprint_nbits', 2048))

    ok_candidates = [c for c in candidates if c.get('status') == 'ok']
    ok_candidates.sort(key=lambda x: x.get('fitness', -float('inf')), reverse=True)

    if not enable_div:
        return ok_candidates[:population_size]

    selected = []
    selected_fps = []
    selected_smiles = set()
    for cand in ok_candidates:
        if len(selected) >= population_size:
            break
        smi = cand.get('smiles')
        if smi is None:
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        try:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, fp_radius, nBits=fp_nbits)
        except Exception:
            continue
        too_similar = any(
            DataStructs.TanimotoSimilarity(fp, fp_sel) > max_tanimoto
            for fp_sel in selected_fps
        )
        if not too_similar:
            selected.append(cand)
            selected_fps.append(fp)
            selected_smiles.add(smi)

    # 补位阶段 1：按不同 SMILES 补入（化学多样性）
    if len(selected) < population_size:
        for cand in ok_candidates:
            if len(selected) >= population_size:
                break
            smi = cand.get('smiles')
            if smi is None or smi in selected_smiles:
                continue
            selected.append(cand)
            selected_smiles.add(smi)

    # 补位阶段 2：SMILES 相同但 3D 坐标不同的也保留（3D 构象多样性）
    # 当 start_t 较低时子代 SMILES 常完全一致，但坐标仍有差异，保留这些 3D 变体
    if len(selected) < population_size:
        for cand in ok_candidates:
            if len(selected) >= population_size:
                break
            if cand in selected:
                continue
            pos_cand = cand.get('pos')
            if pos_cand is None:
                continue
            too_close_3d = False
            for sel in selected:
                pos_sel = sel.get('pos')
                if pos_sel is not None and pos_cand.shape == pos_sel.shape:
                    rmsd = float(np.sqrt(np.mean((pos_cand - pos_sel) ** 2)))
                    if rmsd < 0.3:
                        too_close_3d = True
                        break
            if not too_close_3d:
                selected.append(cand)

    return selected


# =============================================================================
# 骨架生长模式（Scaffold Grow）辅助函数
# =============================================================================

def _get_scaffold_mode_cfg(config) -> 'tuple[dict, str]':
    """
    从统一 scaffold 配置节中读取并展平当前 mode 的参数。

    统一结构：
        scaffold:
          enable: ...
          mode: evolve | grow
          <shared params>        # 顶层共用参数
          evolve: <evolve-only>  # 进化专属，会覆盖顶层同名键
          grow:   <grow-only>    # 生长专属，会覆盖顶层同名键

    返回：(merged_cfg, mode_str)
    merged_cfg 是将 scaffold.<mode> 子节覆盖进顶层后的合并字典，
    使各功能函数可以直接用 cfg.get('key') 获取值而无需关心 mode。
    """
    base = dict(config.sample.get('scaffold', {}))
    mode = base.get('mode', 'evolve')
    sub = base.get(mode)
    if sub is not None:
        merged = {**base, **dict(sub)}
    else:
        merged = base
    return merged, mode


def _sample_n_extra_atoms(
    grow_cfg: dict,
    protein_pos,
    n_scaffold_atoms: int,
    model_num_timesteps: int,
) -> int:
    """
    根据策略决定在骨架之外新生成的原子数（N_extra）。

    策略（grow_cfg.n_extra_mode）：
    - prior_minus_scaffold：从口袋尺寸先验采样总原子数，减去 n_scaffold_atoms
      → 新分子大小与口袋匹配，骨架大小已占用一部分配额
    - pocket_prior：完全由口袋先验决定（不减去骨架原子数）
      → 适合"在骨架基础上大幅扩展"场景
    - fixed：固定数量（n_extra_fixed）
    - range：在 [n_extra_min, n_extra_max] 内均匀随机采样

    结果均被 clamp 到 [n_extra_min_clamp, n_extra_max_clamp]。

    Args:
        grow_cfg: scaffold_grow 配置字典。
        protein_pos: 蛋白口袋原子坐标 Tensor [N_protein, 3]（用于先验采样）。
        n_scaffold_atoms: 骨架原子数。
        model_num_timesteps: 模型扩散步数（未使用，预留）。

    Returns:
        n_extra: 要新生成的原子数（>= 1）。
    """
    mode = grow_cfg.get('n_extra_mode', 'pocket_prior')
    min_clamp = int(grow_cfg.get('n_extra_min_clamp', 1))
    max_clamp = int(grow_cfg.get('n_extra_max_clamp', 40))

    if mode in ('prior_minus_scaffold', 'pocket_prior'):
        pos_np = protein_pos.detach().cpu().numpy() if hasattr(protein_pos, 'detach') else np.array(protein_pos)
        pocket_size = atom_num.get_space_size(pos_np)
        if np.isnan(pocket_size) or np.isinf(pocket_size) or pocket_size <= 0:
            pocket_size = 30.0
        total = int(atom_num.sample_atom_num(pocket_size))
        if mode == 'prior_minus_scaffold':
            n_extra = total - n_scaffold_atoms
        else:
            n_extra = total

    elif mode == 'fixed':
        n_extra = int(grow_cfg.get('n_extra_fixed', 8))

    elif mode == 'range':
        lo = int(grow_cfg.get('n_extra_min', 3))
        hi = int(grow_cfg.get('n_extra_max', 20))
        n_extra = int(np.random.randint(max(lo, 1), max(hi, lo) + 1))

    else:
        n_extra = int(grow_cfg.get('n_extra_fixed', 8))

    return int(np.clip(n_extra, min_clamp, max_clamp))


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
    from utils.masked_guidance_sampling import DualMaskedDiffusionSampler

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

    # ---- 解析骨架掩码 ---------------------------------------------------------
    # 首先尝试从原始配体文件加载分子并提取骨架（更可靠）
    scaffold_indices = []
    ref_mol = None
    ref_ligand_name = None
    
    # 尝试从数据中获取原始参考分子的 RDKit Mol 对象
    if hasattr(data, 'ligand_filename') and data.ligand_filename:
        ref_ligand_path = data.ligand_filename
        ref_ligand_name = Path(ref_ligand_path).stem
        
        if logger:
            logger.info(f'[ScaffoldGrow] 尝试加载配体文件: {ref_ligand_path}')
            logger.info(f'[ScaffoldGrow] 文件存在性检查: {os.path.exists(ref_ligand_path)}')
        
        # 如果文件不存在，尝试从数据集根目录解析
        if not os.path.exists(ref_ligand_path):
            # 尝试使用 protein_root 或 dataset_root 作为基准
            dataset_root = None
            try:
                dataset_root = config.data.path
            except Exception:
                pass
            if not dataset_root:
                try:
                    dataset_root = getattr(config.sample, 'data', {}).get('path', None) if hasattr(config, 'sample') else None
                except Exception:
                    pass

            # 如果路径是相对路径，尝试在不同位置查找
            possible_paths = [
                Path(ref_ligand_path),  # 原始路径
            ]
            # 优先使用 dataset_root（配置文件中的数据路径）
            if dataset_root:
                possible_paths.append(Path(dataset_root) / ref_ligand_path)
            # 回退到常见硬编码路径
            possible_paths += [
                Path('./data/crossdocked_pocket10_test_only') / ref_ligand_path,
                Path('/workspace/data/crossdocked_v1.1_rmsd1.0') / ref_ligand_path,
                Path('/workspace/data/crossdocked_v1.1_rmsd1.0_pocket10') / ref_ligand_path,
            ]
            
            for try_path in possible_paths:
                if try_path.exists():
                    ref_ligand_path = str(try_path)
                    if logger:
                        logger.info(f'[ScaffoldGrow] 找到配体文件: {ref_ligand_path}')
                    break
        
        if os.path.exists(ref_ligand_path):
            try:
                ref_mol = Chem.SDMolSupplier(ref_ligand_path)[0]
                if ref_mol is not None:
                    # 直接从原始分子提取 Murcko 骨架
                    scaffold_source = grow_cfg.get('scaffold_source', 'auto_murcko')
                    
                    # 确保分子有完整的环信息和化学有效性
                    try:
                        Chem.SanitizeMol(ref_mol)
                        # 显式初始化 RingInfo
                        ri = ref_mol.GetRingInfo()
                        if ri is None or ri.NumRings() < 0:
                            Chem.GetSSSR(ref_mol)
                        if logger:
                            logger.info(f'[ScaffoldGrow] 分子环信息初始化成功: {ref_mol.GetNumAtoms()} 原子, {ref_mol.GetRingInfo().NumRings()} 个环')
                    except Exception as e:
                        if logger:
                            logger.warning(f'[ScaffoldGrow] 初始化 RingInfo 失败: {e}')
                        # 继续执行，不中断流程
                    
                    if scaffold_source in ('auto_murcko', 'auto_murcko_generic'):
                        # 策略：优先使用基于环检测的算法（更稳定，不依赖 MurckoScaffold）
                        if logger:
                            logger.info('[ScaffoldGrow] 尝试基于环检测的骨架提取...')
                        indices = _detect_ring_based_scaffold_indices(ref_mol, logger)
                        
                        # 如果环检测失败，回退到 Murcko 方法
                        if not indices:
                            if logger:
                                logger.info('[ScaffoldGrow] 环检测失败，尝试 Murcko 骨架提取...')
                            indices = _detect_murcko_scaffold_indices(ref_mol)
                            if logger:
                                logger.info(f'[ScaffoldGrow] Murcko 检测到 {len(indices)} 个骨架原子')
                            
                            # 若普通 Murcko 返回空，降级到通用骨架
                            if not indices and scaffold_source == 'auto_murcko':
                                indices = _detect_generic_murcko_scaffold_indices(ref_mol)
                                if logger:
                                    logger.info(f'[ScaffoldGrow] 通用 Murcko 检测到 {len(indices)} 个骨架原子')
                        
                        scaffold_indices = indices
                        if logger and indices:
                            logger.info(f'[ScaffoldGrow] 从原始配体文件提取骨架: {len(indices)} 个原子')
                        elif logger:
                            logger.warning('[ScaffoldGrow] 所有骨架提取方法均返回空列表')
            except Exception as e:
                if logger:
                    logger.warning(f'[ScaffoldGrow] 从文件提取骨架失败: {e}')
    
    # 如果无法从文件获取骨架，尝试从坐标重建并使用基于环检测的方法
    if not scaffold_indices:
        if logger:
            logger.info('[ScaffoldGrow] 从文件提取骨架失败，尝试从坐标重建...')
        
        # 从坐标重建分子
        from utils import reconstruct
        try:
            pos_np = ref_ligand_pos.detach().cpu().numpy() if hasattr(ref_ligand_pos, 'detach') else np.array(ref_ligand_pos)
            v_np = ref_ligand_v.detach().cpu().numpy() if hasattr(ref_ligand_v, 'detach') else np.array(ref_ligand_v)
            
            reconstructed_mol = reconstruct.reconstruct_from_generated(pos_np, v_np, ligand_atom_mode)
            
            if reconstructed_mol is not None and logger:
                logger.info(f'[ScaffoldGrow] 坐标重建分子成功: {reconstructed_mol.GetNumAtoms()} 个原子')
            
            if reconstructed_mol is not None:
                # 尝试基于环检测的方法（不依赖 MurckoScaffold）
                if logger:
                    logger.info('[ScaffoldGrow] 尝试对重建分子使用基于环检测的骨架提取...')
                indices = _detect_ring_based_scaffold_indices(reconstructed_mol, logger)
                
                if indices:
                    scaffold_indices = indices
                    if logger:
                        logger.info(f'[ScaffoldGrow] 从重建分子提取骨架: {len(indices)} 个原子')
                else:
                    # 环检测失败，使用 _resolve_scaffold_mask
                    if logger:
                        logger.info('[ScaffoldGrow] 环检测失败，回退到 _resolve_scaffold_mask...')
                    scaffold_mask = _resolve_scaffold_mask(
                        grow_cfg, ref_ligand_pos, ref_ligand_v, ligand_atom_mode, n_scaffold_ref, device
                    )
                    scaffold_mask_np = scaffold_mask.cpu().numpy()
                    scaffold_indices = np.where(scaffold_mask_np > 0.5)[0].tolist()
            else:
                # 重建失败，使用 _resolve_scaffold_mask
                if logger:
                    logger.warning('[ScaffoldGrow] 坐标重建分子失败，使用 _resolve_scaffold_mask...')
                scaffold_mask = _resolve_scaffold_mask(
                    grow_cfg, ref_ligand_pos, ref_ligand_v, ligand_atom_mode, n_scaffold_ref, device
                )
                scaffold_mask_np = scaffold_mask.cpu().numpy()
                scaffold_indices = np.where(scaffold_mask_np > 0.5)[0].tolist()
        except Exception as e:
            if logger:
                logger.warning(f'[ScaffoldGrow] 坐标重建过程失败: {e}')
            # 最终回退
            scaffold_mask = _resolve_scaffold_mask(
                grow_cfg, ref_ligand_pos, ref_ligand_v, ligand_atom_mode, n_scaffold_ref, device
            )
            scaffold_mask_np = scaffold_mask.cpu().numpy()
            scaffold_indices = np.where(scaffold_mask_np > 0.5)[0].tolist()
    
    n_scaffold = len(scaffold_indices)

    if n_scaffold == 0:
        if logger:
            logger.warning('[ScaffoldGrow] 未找到骨架原子，将使用所有原子作为骨架（强制固定整个分子）')
        # 最终 fallback：使用所有原子作为骨架
        scaffold_indices = list(range(n_scaffold_ref))
        n_scaffold = n_scaffold_ref
        if logger:
            logger.info(f'[ScaffoldGrow] 强制使用所有 {n_scaffold} 个原子作为骨架')

    if logger:
        logger.info(
            f'[ScaffoldGrow] 骨架原子: {n_scaffold}/{n_scaffold_ref} '
            f'(来源: {grow_cfg.get("scaffold_source", "auto_murcko")}) | '
            f'额外原子策略: {grow_cfg.get("n_extra_mode", "prior_minus_scaffold")}'
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
                if output_dir is not None:
                    base_output_dir = Path(output_dir)
                else:
                    base_output_dir = Path(config.sample.get('output_dir', './outputs'))
                
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
        n_extra = _sample_n_extra_atoms(grow_cfg, data.protein_pos, n_scaffold, model.num_timesteps)
        n_total = n_scaffold + n_extra

        if logger and (sample_idx == 0 or n_extra != _sample_n_extra_atoms.__defaults__):
            logger.info(
                f'[ScaffoldGrow] 样本 {sample_idx + 1}/{num_samples} | '
                f'骨架原子: {n_scaffold} + 新原子: {n_extra} = 总计: {n_total}'
            )

        # ---- 构建单样本批次 -------------------------------------------------
        batch = Batch.from_data_list([data.clone()], follow_batch=FOLLOW_BATCH).to(device)
        batch_protein = batch.protein_element_batch

        # 口袋中心（新原子初始化位置）
        pocket_center = scatter_mean(batch.protein_pos, batch_protein, dim=0)[0]  # [3]

        # 骨架原子坐标（原始参考位置，居中前）
        scaffold_pos_orig = ref_ligand_pos[scaffold_indices] if n_scaffold > 0 else torch.zeros(0, 3, device=device)
        scaffold_log_v_orig = log_ref_v[scaffold_indices] if n_scaffold > 0 else torch.zeros(0, model.num_classes, device=device)

        # 新原子初始坐标：口袋中心 + 高斯噪声（尺度 2Å，与蛋白口袋尺度匹配）
        extra_pos_init = pocket_center.unsqueeze(0).expand(n_extra, -1) + \
                         torch.randn(n_extra, 3, device=device) * 2.0

        # 新原子初始类型：均匀分布（全零 → log_softmax）
        extra_log_v_init = F.log_softmax(
            torch.zeros(n_extra, model.num_classes, device=device), dim=-1
        )

        # 拼接：[骨架原子 | 新原子]
        init_ligand_pos = torch.cat([scaffold_pos_orig, extra_pos_init], dim=0)  # [n_total, 3]
        init_log_ligand_v = torch.cat([scaffold_log_v_orig, extra_log_v_init], dim=0)  # [n_total, C]
        batch_ligand = torch.zeros(n_total, dtype=torch.long, device=device)

        # 中心化：将初始位置对齐到口袋中心
        protein_pos_centered, init_pos_centered, offset = center_pos(
            batch.protein_pos, init_ligand_pos,
            batch_protein, batch_ligand, mode=center_pos_mode,
        )

        # ---- 构建 RePaint 参考态 -------------------------------------------
        # 骨架参考：已中心化的骨架原子位置 + 骨架原子类型
        # 新原子参考：口袋中心（掩码为 0，不会被拉回，设置任意值）
        x0_pos_ref = init_pos_centered.clone()  # 骨架部分将被锚定，新原子随便
        x0_log_v_ref = init_log_ligand_v.clone()

        # 掩码：骨架原子在前 n_scaffold 位（pos_mask=type_mask=1），其余为 0
        pos_mask_local = torch.zeros(n_total, device=device)
        type_mask_local = torch.zeros(n_total, device=device)
        if n_scaffold > 0:
            pos_mask_local[:n_scaffold] = 1.0
            type_mask_local[:n_scaffold] = 1.0  # 骨架生长时：元素也固定

        repaint_cfg = {
            'x0_pos': x0_pos_ref,
            'x0_log_v': x0_log_v_ref,
            'pos_mask': pos_mask_local,
            'type_mask': type_mask_local if n_scaffold > 0 else None,
            'use_mean_for_discrete': True,
            '_use_dual_mask': True,
        } if n_scaffold > 0 else None

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

        # 还原坐标偏移
        final_pos = ligand_pos_out + offset[batch_ligand]
        pos_np = final_pos.detach().cpu().numpy().astype(np.float64)
        v_np = log_v_out.argmax(dim=-1).detach().cpu().numpy().astype(np.int64)

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
            'n_extra': n_extra,
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
            },
            'total_time': total_time,
        },
    }


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
    from utils.masked_guidance_sampling import DualMaskedDiffusionSampler

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
    scaffold_mask = _resolve_scaffold_mask(
        scaffold_cfg, ref_ligand_pos, ref_ligand_v, ligand_atom_mode, ref_num_atoms, device
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='扩散模型分子采样脚本。支持从测试集或直接指定蛋白/配体文件进行采样。',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
使用示例:
  # 1. 使用测试集的指定样本（推荐用于复现测试集结果）
  python scripts/sample_diffusion.py configs/sampling.yml -i 0
  #    指定 GPU：-gpu 5  或  --device cuda:5
  
  # 2. 使用自定义蛋白和配体文件
  python scripts/sample_diffusion.py configs/sampling.yml --protein_path pocket.pdb --ligand_path ligand.sdf
  
  # 3. 使用测试集配体进行骨架生长（scaffold grow）
  # 配置: sample.scaffold.enable=true, mode=grow, after_dynamic=false
  python scripts/sample_diffusion.py configs/sampling.yml -i 0 --use_test_set
  
  # 4. 使用测试集配体进行骨架进化优化（scaffold evolve）
  # 配置: sample.scaffold.enable=true, mode=evolve, after_dynamic=false
  python scripts/sample_diffusion.py configs/sampling.yml -i 0 --use_test_set
  
  # 5. 对测试集配体进行分子优化（optimization模式）
  # 配置: sample.optimization.enable=true, after_dynamic=false
  python scripts/sample_diffusion.py configs/sampling.yml -i 0 --use_test_set

注意:
  - scaffold/optimization模式且after_dynamic=false时，会直接用输入配体作为参考/起点
  - 使用--use_test_set会明确启用测试集模式，并从数据集加载蛋白和配体
  - --molecule_path优先级最高，指定后会覆盖测试集配体
        '''
    )
    parser.add_argument('config', type=str, help='采样配置文件路径（如 configs/sampling.yml）')
    parser.add_argument('-i', '--data_id', type=int, default=None,
                        help='测试集数据索引（从0开始）。不指定且使用测试集模式时默认使用0号样本')
    parser.add_argument('--use_test_set', action='store_true',
                        help='明确使用测试集模式：从数据集加载蛋白和配体。'
                             '适用于 scaffold grow/evolve 或 optimization 模式直接优化测试集配体')
    parser.add_argument('--protein_path', type=str, default=None,
                        help='自定义蛋白口袋 PDB 文件路径（与 data_id/use_test_set 二选一）')
    parser.add_argument(
        '--protein_root',
        type=str,
        default=None,
        help='蛋白数据根目录，与 evaluate_pt --protein_root 一致；用于 Prudent 内 Vina 解析相对路径或按配体推断受体 PDB',
    )
    parser.add_argument('--ligand_path', type=str, default=None,
                        help='自定义参考配体 SDF/MOL2 文件路径（用于裁剪口袋和作为参考分子）')
    parser.add_argument('--use_dataset_for_pocket', action='store_true',
                        help='当蛋白在数据集内时，使用 index.pkl 中的配体（与测试集一致，可避免生成分子不完整）')
    parser.add_argument('--pocket_radius', type=float, default=10.0,
                        help='口袋裁剪半径（Å），提供配体时自动裁剪蛋白为口袋（默认: 10.0，匹配训练数据 pocket10）')
    parser.add_argument('--device', type=str, default='cuda:0', help='运行设备（如 cuda:0, cuda:1, cpu）')
    parser.add_argument(
        '-gpu', '--gpu',
        type=int,
        default=None,
        metavar='N',
        dest='gpu_id',
        help='指定 GPU 编号，等价于 --device cuda:N；若设置则覆盖 --device（cpu 请仍用 --device cpu）',
    )
    parser.add_argument('--batch_size', type=int, default=100, help='采样批量大小')
    parser.add_argument('--result_path', type=str, default='./outputs/pt', help='结果输出目录')
    parser.add_argument(
        '--mode', type=str,
        choices=['baseline', 'dynamic', 'optimization', 'prudent'],
        default=None,
        help='采样模式（optimization.enable 或 scaffold.enable 为 True 时被覆盖）'
    )
    parser.add_argument('--molecule_path', type=str, default=None,
                        help='待优化/参考分子的 SDF 文件路径（覆盖默认配体）。'
                             'scaffold grow/evolve/optimization 模式下用作骨架/优化起点。'
                             '未指定时默认使用测试集配体或 ligand_path 指定的配体')
    
    # Prudent 分段独立运行模式参数
    parser.add_argument('--resume-from-pt', type=str, default=None,
                        help='从指定 .pt 文件恢复运行（用于分段独立运行模式）')
    parser.add_argument('--resume-frame', type=int, default=None,
                        help='从指定 frame 开始（与 --resume-from-pt 配合使用）')
    parser.add_argument('--target-frame', type=int, default=None,
                        help='跑到指定 target frame 结束（默认24）')
    parser.add_argument('--force-generation', type=int, default=None,
                        help='强制指定 generation 编号（用于输出文件名）')
    
    args = parser.parse_args()  # 解析命令行参数。
    if args.gpu_id is not None:
        args.device = f'cuda:{int(args.gpu_id)}'

    # 验证 batch_size 参数
    args.batch_size = max(1, int(args.batch_size))  # 确保至少为1

    logger = misc.get_logger('sampling')  # 创建日志器。

    # 初始化CUDA上下文（修复CUBLAS_STATUS_NOT_INITIALIZED错误）
    if args.device.startswith('cuda'):
        try:
            # 确保CUDA可用
            if not torch.cuda.is_available():
                raise RuntimeError(f'CUDA不可用，但指定了设备: {args.device}')
            
            # 获取设备ID
            device_id = int(args.device.split(':')[1]) if ':' in args.device else 0
            
            # 检查设备ID是否有效
            if device_id >= torch.cuda.device_count():
                raise RuntimeError(f'无效的设备ID: {device_id}，可用设备数: {torch.cuda.device_count()}')
            
            # 设置当前设备并初始化CUDA上下文
            torch.cuda.set_device(device_id)
            
            # 创建一个小的tensor来强制初始化CUDA上下文和CUBLAS
            _dummy = torch.zeros(1, device=args.device)
            _dummy = _dummy + 1  # 执行一个简单的操作来初始化CUBLAS
            del _dummy
            torch.cuda.synchronize(device=args.device)  # 同步以确保初始化完成
            
            logger.info(f'CUDA上下文已初始化: {args.device} (设备ID: {device_id})')
        except Exception as e:
            logger.error(f'CUDA初始化失败: {e}')
            raise

    # 创建全局GPU监控器
    global_monitor = GPUMonitor(device=args.device, enable_flops=False)
    global_profiler = MemoryProfiler(device=args.device)
    global_profiler.checkpoint('script_start')

    # 加载配置文件：从 YAML 文件读取采样配置。
    config_path = os.path.abspath(args.config)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f'Sampling config not found: {config_path}')
    logger.info(f'Loading sampling config from: {config_path}')
    logger.info(f'Sampling config mtime: {datetime.fromtimestamp(os.path.getmtime(config_path))}')
    config = misc.load_config(config_path)  # 加载采样配置。
    dynamic_cfg = config.sample.get('dynamic', {})
    logger.info(f'Sampling config dynamic.large_step: {dynamic_cfg.get("large_step")}')
    logger.info(f'Sampling config dynamic.refine: {dynamic_cfg.get("refine")}')
    logger.info(config)  # 记录配置信息到日志。
    misc.seed_all(config.sample.seed)  # 设置随机种子，确保可复现性。

    # 加载模型检查点：从检查点文件恢复模型权重和训练配置。
    ckpt = torch.load(config.model.checkpoint, map_location=args.device)  # 加载检查点到指定设备。
    logger.info(f"Training Config: {ckpt['config']}")  # 记录训练配置信息。

    # 初始化特征转换器：创建蛋白和配体的特征提取管道。
    protein_featurizer = trans.FeaturizeProteinAtom()  # 创建蛋白原子特征化器。
    ligand_atom_mode = ckpt['config'].data.transform.ligand_atom_mode  # 从检查点读取配体原子编码模式。
    ligand_featurizer = trans.FeaturizeLigandAtom(ligand_atom_mode)  # 创建配体原子特征化器。
    transform = Compose([  # 组合多个转换器为单一管道。
        protein_featurizer,  # 蛋白特征转换。
        ligand_featurizer,  # 配体特征转换。
        trans.FeaturizeLigandBond(),  # 配体键特征转换。
    ])

    # 加载数据：支持 data_id（数据集索引）或 protein_path（自定义文件）
    use_custom_files = args.protein_path is not None
    use_test_set = getattr(args, 'use_test_set', False)

    if use_custom_files:
        # 若指定 --data_id 且蛋白在数据集内，优先使用数据集的 (pocket, ligand) 以保证与测试集一致
        use_dataset_for_pocket = getattr(args, 'use_dataset_for_pocket', False)
        dataset_root = ckpt['config'].data.path
        protein_resolved = Path(args.protein_path).resolve()
        dataset_ligand_path = None
        if use_dataset_for_pocket and dataset_root:
            dataset_root = Path(dataset_root).resolve()
            try:
                index_path = Path(dataset_root) / 'index.pkl'
                if index_path.exists():
                    import pickle
                    with open(index_path, 'rb') as f:
                        index = pickle.load(f)
                    for pocket_fn, ligand_fn, *rest in index:
                        if pocket_fn is None:
                            continue
                        full_pocket = Path(dataset_root) / pocket_fn
                        if full_pocket.resolve() == protein_resolved:
                            dataset_ligand_path = Path(dataset_root) / ligand_fn
                            if dataset_ligand_path.exists():
                                logger.info(f'使用数据集配体（与测试集一致）: {ligand_fn}')
                                break
            except Exception as e:
                if logger:
                    logger.warning(f'查找数据集配体失败: {e}')
        ligand_to_use = str(dataset_ligand_path) if dataset_ligand_path else args.ligand_path
        if args.data_id is not None:
            logger.warning('同时指定了 --protein_path 和 --data_id，优先使用 --protein_path')
        logger.info(f'从自定义文件加载: 蛋白={args.protein_path}, 配体={ligand_to_use or "无（仅蛋白模式）"}')
        data = load_custom_pocket_data(
            args.protein_path, ligand_to_use, transform=transform,
            pocket_radius=args.pocket_radius, logger=logger
        )
        # 自定义模式使用 sample_num_atoms=prior（不依赖配体）
        if config.sample.get('sample_num_atoms') == 'ref':
            logger.warning('自定义文件模式下 sample_num_atoms=ref 需要配体，已自动切换为 prior')
            config.sample.sample_num_atoms = 'prior'
        pocket_id = 'custom'
        test_set = None  # 不需要 test_set
    else:
        # 从数据集加载（测试集模式）
        dataset, subsets = get_dataset(
            config=ckpt['config'].data,  # 使用检查点中的数据集配置。
            transform=transform  # 应用特征转换管道。
        )
        train_set, test_set = subsets['train'], subsets['test']  # 提取训练集和测试集。
        logger.info(f'Successfully load the dataset (size: {len(test_set)})!')  # 记录数据集加载成功信息。

        if args.data_id is None:
            if use_test_set:
                logger.info('[测试集模式] data_id 未指定，默认使用 0 号样本。')
            else:
                logger.warning('data_id 未指定，默认使用 0 号样本。')
            args.data_id = 0

        if not (0 <= args.data_id < len(test_set)):
            raise ValueError(f'data_id 必须在 0~{len(test_set) - 1} 范围内，当前为 {args.data_id}')

        data = test_set[args.data_id]  # 选择待采样的测试样本。
        pocket_id = str(args.data_id)

        # 明确提示测试集模式信息
        if use_test_set:
            logger.info(f'[测试集模式] 已加载测试集样本: data_id={args.data_id}')
            logger.info(f'[测试集模式] 蛋白: {getattr(data, "protein_filename", "unknown")}')
            logger.info(f'[测试集模式] 配体: {getattr(data, "ligand_filename", "unknown")}')
            logger.info(f'[测试集模式] 配体原子数: {data.ligand_pos.size(0) if hasattr(data, "ligand_pos") and data.ligand_pos is not None else 0}')

            # 检查是否适合 scaffold 模式
            scaffold_cfg = config.sample.get('scaffold', {})
            if scaffold_cfg.get('enable', False):
                sc_mode = scaffold_cfg.get('mode', 'evolve')
                after_dynamic = scaffold_cfg.get('after_dynamic', True)
                logger.info(f'[测试集模式] 检测到 scaffold 模式: mode={sc_mode}, after_dynamic={after_dynamic}')
                if not after_dynamic:
                    logger.info(f'[测试集模式] scaffold.after_dynamic=false，将直接使用测试集配体作为参考进行 {sc_mode} 优化')

    # Prudent / Vina：数据集里 protein_filename 多为相对 data.path 的路径，规范为存在的绝对 PDB
    _dataset_root = None
    try:
        dp = getattr(ckpt['config'].data, 'path', None)
        if dp:
            _dataset_root = str(Path(dp).expanduser().resolve())
    except Exception:
        pass
    try:
        _prudent_yaml_root = (
            (config.sample.get('dynamic') or {}).get('prudent') or {}
        ).get('protein_root')
    except Exception:
        _prudent_yaml_root = None
    _cli_protein_root = getattr(args, 'protein_root', None)
    _vina_protein_root = _cli_protein_root or _prudent_yaml_root
    resolve_and_set_absolute_protein_path(
        data,
        dataset_root=_dataset_root,
        protein_root=_vina_protein_root,
        logger=logger,
    )

    # 加载模型：根据检查点配置实例化并加载模型权重。
    model_cfg = ckpt['config'].model  # 从检查点读取模型配置。
    
    # 允许从 sampling.yml 覆盖模型配置（如果提供）
    if hasattr(config.model, 'use_grad_fusion'):
        model_cfg.use_grad_fusion = config.model.use_grad_fusion
        logger.info(f'Override use_grad_fusion from sampling config: {config.model.use_grad_fusion}')
    if hasattr(config.model, 'grad_fusion_lambda'):
        model_cfg.grad_fusion_lambda = config.model.grad_fusion_lambda
        logger.info(f'Override grad_fusion_lambda from sampling config: {config.model.grad_fusion_lambda}')
    
    model_name = getattr(model_cfg, 'name', 'score').lower()  # 获取模型名称，默认为 'score'。
    # 支持 glintdm 和 diffdynamic 两种配置值（向后兼容）
    model_cls = DiffDynamic if model_name in ('glintdm', 'diffdynamic') else ScorePosNet3D  # 根据名称选择模型类。
    model = model_cls(  # 实例化模型。
        model_cfg,  # 模型配置。
        protein_atom_feature_dim=protein_featurizer.feature_dim,  # 蛋白原子特征维度。
        ligand_atom_feature_dim=ligand_featurizer.feature_dim  # 配体原子特征维度。
    ).to(args.device)  # 将模型移动到指定设备。
    model.load_state_dict(ckpt['model'])  # 加载模型权重。
    logger.info(f'Successfully load the model! {config.model.checkpoint}')  # 记录模型加载成功信息。
    
    # 将采样配置映射到模型配置（用于动态采样）
    if hasattr(config, 'sample') and hasattr(config.sample, 'dynamic'):
        dynamic_cfg = config.sample.dynamic
        if hasattr(dynamic_cfg, 'large_step'):
            model_cfg.dynamic_large_step = dynamic_cfg.large_step
            logger.info(f'Mapped config.sample.dynamic.large_step to model_cfg.dynamic_large_step')
        if hasattr(dynamic_cfg, 'refine'):
            model_cfg.dynamic_refine = dynamic_cfg.refine
            logger.info(f'Mapped config.sample.dynamic.refine to model_cfg.dynamic_refine')
        # 更新模型实例的配置引用
        model.dynamic_large_step_defaults = getattr(model_cfg, 'dynamic_large_step', {})
        model.dynamic_refine_defaults = getattr(model_cfg, 'dynamic_refine', {})
    
    global_profiler.checkpoint('after_model_load')
    mem_info = global_monitor.get_memory_info()
    logger.info(f'[Init] Model loaded. Memory: {mem_info["allocated"]:.1f}/{mem_info["max_allocated"]:.1f} MB')
    
    # 记录模型加载后的GPU监控信息
    try:
        log_gpu_monitor_record(
            memory_info=mem_info,
            sampling_info={
                'mode': 'initialization',
                'stage': 'model_load',
                'data_id': args.data_id,
            },
            logger=logger
        )
    except Exception as e:
        if logger:
            logger.warning(f'Failed to log GPU monitor record after model load: {e}')

    global_profiler.checkpoint('after_data_load')

    # 决定采样模式（优先级：scaffold > optimization 链 > sample.mode）
    # scaffold.mode 决定走 scaffold_optimization（evolve）还是 scaffold_grow（grow）
    # scaffold.after_dynamic=true：先 dynamic 再对每个分子做骨架进化优化
    scaffold_unified_cfg = config.sample.get('scaffold', {})
    opt_cfg = config.sample.get('optimization', {})
    if scaffold_unified_cfg.get('enable', False):
        _sc_mode = scaffold_unified_cfg.get('mode', 'evolve')
        if _sc_mode == 'grow':
            sampling_mode = 'scaffold_grow'
        elif scaffold_unified_cfg.get('after_dynamic', False):
            sampling_mode = 'dynamic_then_scaffold'
        else:
            sampling_mode = 'scaffold_optimization'
    elif opt_cfg.get('enable', False) and opt_cfg.get('after_dynamic', False):
        sampling_mode = 'dynamic_then_optimization'
    elif opt_cfg.get('enable', False):
        sampling_mode = 'optimization'
    else:
        sampling_mode = args.mode or config.sample.get('mode', 'baseline')
    if sampling_mode not in [
        'baseline', 'dynamic', 'optimization', 'prudent', 'dynamic_then_optimization',
        'scaffold_grow', 'scaffold_optimization', 'dynamic_then_scaffold',
    ]:
        raise ValueError(f'Unsupported sampling mode: {sampling_mode}')

    result_path_abs = os.path.abspath(args.result_path)
    os.makedirs(result_path_abs, exist_ok=True)
    _cst_tz_run = timezone(timedelta(hours=8))
    run_timestamp_cst = datetime.now(_cst_tz_run).strftime('%Y%m%d_%H%M%S')

    if sampling_mode == 'dynamic_then_optimization':
        # 先执行 dynamic 两阶段扩散，再对每个生成分子进行优化
        config.sample.setdefault('dynamic', {})
        config.sample.dynamic.setdefault('large_step', {})
        config.sample.dynamic['large_step'].setdefault('batch_size', args.batch_size)
        logger.info('[Dynamic+Optimization] 第一阶段：执行 dynamic 两阶段扩散（强制 legacy 模式，分子数由 large_step/refine 控制）')
        dynamic_output = sample_dynamic_diffusion_ligand(
            model=model,
            data=data,
            config=config,
            ligand_atom_mode=ligand_atom_mode,
            device=args.device,
            logger=logger,
            force_method='legacy',  # 避免使用 unified 的 num_samples 循环，确保 n_dyn = batch_size * n_repeat * n_sampling
            skip_targetdiff_baseline_refine=True,  # baseline refine 在优化结束后对所有输出分子统一执行
        )
        pos_list_dyn = dynamic_output['pos_list']
        v_list_dyn = dynamic_output['v_list']
        n_dyn_raw = len(pos_list_dyn)
        # sample.num_samples 在 baseline 下表示「每口袋分子数」；此处对齐为「最多参与优化的 dynamic 条数」
        num_samples_target = max(1, int(config.sample.get('num_samples', 100)))
        if n_dyn_raw > num_samples_target:
            logger.info(
                f'[Dynamic+Optimization] dynamic 阶段产出 {n_dyn_raw} 条，按 sample.num_samples={num_samples_target} '
                f'截断（legacy 原始条数由 large_step.batch_size×n_repeat×refine.n_sampling 决定，与 sample.num_samples 无关）'
            )
            pos_list_dyn = pos_list_dyn[:num_samples_target]
            v_list_dyn = v_list_dyn[:num_samples_target]
        n_dyn = len(pos_list_dyn)
        logger.info(f'[Dynamic+Optimization] 第一阶段完成: {n_dyn} 个分子（截断后上限 sample.num_samples={num_samples_target}）')
        logger.info('[Dynamic+Optimization] 第二阶段：对每个「RDKit 重建成功」的 dynamic 分子做 optimization')
        all_pos, all_v, all_pos_traj, all_v_traj, all_log_v_traj, all_time, all_meta = [], [], [], [], [], [], []
        optimization_time_indices = None
        mol_transform = Compose([ligand_featurizer, trans.FeaturizeLigandBond()])
        n_recon_ok = 0
        n_recon_fail = 0
        for i in range(n_dyn):
            data_i = _create_data_from_generated_mol(
                data, pos_list_dyn[i], v_list_dyn[i],
                ligand_atom_mode, mol_transform, logger
            )
            if data_i is None:
                n_recon_fail += 1
                if logger:
                    logger.warning(f'[Dynamic+Optimization] 分子 {i+1}/{n_dyn} RDKit 重建失败，跳过优化')
                continue
            n_recon_ok += 1
            opt_out = optimize_molecule(model, data_i, config, ligand_atom_mode, device=args.device, logger=logger, opt_batch_size=1, n_dynamic_molecules=n_dyn)
            if optimization_time_indices is None:
                oti = opt_out['meta'].get('optimization_time_indices')
                if oti is not None:
                    optimization_time_indices = oti
            all_pos.extend(opt_out['pos_list'])
            all_v.extend(opt_out['v_list'])
            all_pos_traj.extend(opt_out['pos_traj'])
            all_v_traj.extend(opt_out['v_traj'])
            all_log_v_traj.extend(opt_out['log_v_traj'])
            all_time.extend(opt_out['time_list'])
            for r in opt_out['meta']['records']:
                r['source_dynamic_idx'] = i
            all_meta.extend(opt_out['meta']['records'])
        n_final = len(all_pos)
        if logger:
            logger.info(
                f'[Dynamic+Optimization] 汇总: 最终写入 {n_final} 个分子 | '
                f'dynamic 条数={n_dyn} | RDKit 重建成功={n_recon_ok} | 重建失败跳过={n_recon_fail}'
            )
        if n_final < num_samples_target:
            logger.warning(
                f'[Dynamic+Optimization] 最终分子数 {n_final} < sample.num_samples={num_samples_target}。'
                f' 常见原因：① dynamic 后 RDKit 成键/重建失败过多；② optimization.selector 筛选过严；③ 仅蛋白时 prior 原子数不稳定。'
                f' 可尝试：临时设 optimization.enable=false 仅用 dynamic；或提供配体并设 sample_num_atoms=ref；'
                f'或放宽 optimization.selector（min_qed/min_sa、enable_filter）。'
            )
        meta_do = {
            'method': 'dynamic_then_optimization',
            'records': all_meta,
            'dynamic_then_optimization_stats': {
                'sample_num_samples_cap': num_samples_target,
                'dynamic_molecules_after_truncate': n_dyn,
                'reconstruct_success': n_recon_ok,
                'reconstruct_fail': n_recon_fail,
                'final_molecule_count': n_final,
            },
        }
        if optimization_time_indices is not None:
            meta_do['optimization_time_indices'] = optimization_time_indices
        result = {
            'data': data,
            'pred_ligand_pos': all_pos,
            'pred_ligand_v': all_v,
            'pred_ligand_pos_traj': all_pos_traj,
            'pred_ligand_v_traj': all_v_traj,
            'pred_ligand_log_v_traj': all_log_v_traj,
            'time': all_time,
            'meta': meta_do,
            'mode': 'dynamic_then_optimization'
        }
        apply_targetdiff_baseline_refine_to_sampling_result(
            model, data, result, config, device=args.device, logger=logger
        )
    elif sampling_mode == 'optimization':
        # 分子优化模式：对现有分子施加部分前向扩散，再用蛋白引导去噪
        config.sample.setdefault('optimization', {})

        # 如果指定了 --molecule_path，使用它作为优化起点（覆盖 ligand_path 加载的配体）
        mol_path = getattr(args, 'molecule_path', None) or opt_cfg.get('molecule_path', None)
        if mol_path is not None:
            mol_path = Path(mol_path).resolve()
            if not mol_path.exists():
                raise FileNotFoundError(f'待优化分子文件不存在: {mol_path}')
            logger.info(f'[Optimization] 从独立文件加载待优化分子: {mol_path}')
            mol_dict = parse_sdf_file(str(mol_path))
            data.ligand_element = torch.tensor(mol_dict['element'], dtype=torch.long)
            data.ligand_pos = torch.tensor(mol_dict['pos'], dtype=torch.float32)
            data.ligand_bond_index = torch.tensor(mol_dict['bond_index'], dtype=torch.long)
            data.ligand_bond_type = torch.tensor(mol_dict['bond_type'], dtype=torch.long)
            mol_transform = Compose([ligand_featurizer, trans.FeaturizeLigandBond()])
            data = mol_transform(data)
        else:
            # 未指定 molecule_path，使用当前 data 中的配体（来自测试集或 ligand_path）
            use_test_set = getattr(args, 'use_test_set', False)
            data_id = getattr(args, 'data_id', None)
            if use_test_set and data_id is not None:
                logger.info(f'[Optimization] 使用测试集配体作为优化起点: data_id={data_id}')
            elif hasattr(data, 'ligand_filename') and data.ligand_filename:
                logger.info(f'[Optimization] 使用已加载配体作为优化起点: {data.ligand_filename}')
            else:
                logger.info('[Optimization] 使用已加载配体作为优化起点')

        opt_output = optimize_molecule(
            model=model,
            data=data,
            config=config,
            ligand_atom_mode=ligand_atom_mode,
            device=args.device,
            logger=logger
        )
        result = {
            'data': data,
            'pred_ligand_pos': opt_output['pos_list'],
            'pred_ligand_v': opt_output['v_list'],
            'pred_ligand_pos_traj': opt_output['pos_traj'],
            'pred_ligand_v_traj': opt_output['v_traj'],
            'pred_ligand_log_v_traj': opt_output['log_v_traj'],
            'time': opt_output['time_list'],
            'meta': opt_output['meta'],
            'mode': 'optimization'
        }
        apply_targetdiff_baseline_refine_to_sampling_result(
            model, data, result, config, device=args.device, logger=logger
        )
    elif sampling_mode == 'dynamic_then_scaffold':
        # 两阶段流水线：先 dynamic 生成分子，再对每个分子用骨架进化优化（auto_murcko 等）
        # 第一阶段：dynamic 两阶段扩散
        config.sample.setdefault('dynamic', {})
        config.sample.dynamic.setdefault('large_step', {})
        config.sample.dynamic['large_step'].setdefault('batch_size', args.batch_size)
        logger.info('[Dynamic+ScaffoldEvolve] 第一阶段：执行 dynamic 两阶段扩散...')
        dynamic_output = sample_dynamic_diffusion_ligand(
            model=model,
            data=data,
            config=config,
            ligand_atom_mode=ligand_atom_mode,
            device=args.device,
            logger=logger,
            force_method='legacy',
            skip_targetdiff_baseline_refine=True,
        )
        pos_list_dyn = dynamic_output['pos_list']
        v_list_dyn = dynamic_output['v_list']
        n_dyn = len(pos_list_dyn)
        logger.info(
            f'[Dynamic+ScaffoldEvolve] 第一阶段完成: {n_dyn} 个分子 '
            f'（由 large_step.batch_size × n_repeat × refine.n_sampling 决定）'
        )
        # 第二阶段：对每个 RDKit 重建成功的分子做骨架进化优化
        logger.info('[Dynamic+ScaffoldEvolve] 第二阶段：对每个分子进行骨架约束进化优化...')
        config.sample.setdefault('scaffold', {})
        sc_merged, _ = _get_scaffold_mode_cfg(config)
        save_dyn_pre = bool(sc_merged.get('save_dynamic_before_scaffold', False))
        opt_style_name = bool(sc_merged.get('optimization_style_naming', True))
        dyn_sel = {
            'min_qed': sc_merged.get('min_qed'),
            'min_sa': sc_merged.get('min_sa'),
            'qed_weight': sc_merged.get('qed_weight', 1.0),
            'sa_weight': sc_merged.get('sa_weight', 1.0),
            'filter_incomplete': False,
        }
        mol_transform = Compose([ligand_featurizer, trans.FeaturizeLigandBond()])
        all_pos, all_v, all_pos_traj, all_v_traj, all_log_v_traj, all_time, all_meta = [], [], [], [], [], [], []
        prepend_meta, prepend_pos, prepend_v = [], [], []
        if save_dyn_pre:
            for di in range(n_dyn):
                pos_np = np.asarray(pos_list_dyn[di], dtype=np.float64)
                v_np = np.asarray(v_list_dyn[di], dtype=np.int64)
                metric_info = evaluate_candidate(pos_np, v_np, ligand_atom_mode, dyn_sel)
                prepend_pos.append(pos_np)
                prepend_v.append(v_np)
                prepend_meta.append({
                    'method': 'dynamic_then_scaffold',
                    'stage': 'dynamic_pre_scaffold',
                    'ligand_num_atoms': int(len(pos_np)),
                    'time': 0.0,
                    'rmsd_from_ref': 0.0,
                    'smiles': metric_info.get('smiles'),
                    'qed': metric_info.get('metrics', {}).get('qed'),
                    'sa': metric_info.get('metrics', {}).get('sa'),
                    'status': metric_info.get('status'),
                    'is_original': True,
                    'is_dynamic_raw': True,
                    'source_dynamic_idx': di,
                })
            if logger:
                logger.info(
                    f'[Dynamic+ScaffoldEvolve] 已按 save_dynamic_before_scaffold 追加 {len(prepend_meta)} 条 '
                    f'dynamic 原始分子（拼在 pred 最前）；骨架阶段临时关闭 keep_original 以免与「原分子」身份证重复'
                )
        orig_keep = None
        if save_dyn_pre:
            orig_keep = config.sample['scaffold'].get('keep_original', True)
            config.sample['scaffold']['keep_original'] = False
        n_recon_ok = 0
        n_recon_fail = 0
        try:
            for i in range(n_dyn):
                data_i = _create_data_from_generated_mol(
                    data, pos_list_dyn[i], v_list_dyn[i],
                    ligand_atom_mode, mol_transform, logger
                )
                if data_i is None:
                    n_recon_fail += 1
                    if logger:
                        logger.warning(
                            f'[Dynamic+ScaffoldEvolve] 分子 {i+1}/{n_dyn} RDKit 重建失败，跳过'
                        )
                    continue
                n_recon_ok += 1
                logger.info(
                    f'[Dynamic+ScaffoldEvolve] 正在优化第 {n_recon_ok} 个分子 '
                    f'（dynamic 第 {i+1}/{n_dyn} 条）...'
                )
                scaff_out = scaffold_optimize_molecule(
                    model=model,
                    data=data_i,
                    config=config,
                    ligand_atom_mode=ligand_atom_mode,
                    device=args.device,
                    logger=logger,
                )
                all_pos.extend(scaff_out['pos_list'])
                all_v.extend(scaff_out['v_list'])
                all_pos_traj.extend(scaff_out['pos_traj'])
                all_v_traj.extend(scaff_out['v_traj'])
                all_log_v_traj.extend(scaff_out['log_v_traj'])
                all_time.extend(scaff_out['time_list'])
                for r in scaff_out['meta']['records']:
                    r['source_dynamic_idx'] = i
                    if opt_style_name:
                        r['optimization_style_naming'] = True
                all_meta.extend(scaff_out['meta']['records'])
        finally:
            if save_dyn_pre and orig_keep is not None:
                config.sample['scaffold']['keep_original'] = orig_keep

        if save_dyn_pre and prepend_pos:
            z = len(prepend_pos)
            all_pos = prepend_pos + all_pos
            all_v = prepend_v + all_v
            all_pos_traj = [[] for _ in range(z)] + all_pos_traj
            all_v_traj = [[] for _ in range(z)] + all_v_traj
            all_log_v_traj = [[] for _ in range(z)] + all_log_v_traj
            all_time = [0.0] * z + all_time
            all_meta = prepend_meta + all_meta

        n_final = len(all_pos)
        if logger:
            logger.info(
                f'[Dynamic+ScaffoldEvolve] 汇总: 最终 {n_final} 个分子（含 dynamic 前置条数）| '
                f'dynamic 条数={n_dyn} | 重建成功={n_recon_ok} | 重建失败={n_recon_fail}'
            )
        meta_ds = {
            'method': 'dynamic_then_scaffold',
            'records': all_meta,
            'optimization_style_naming': opt_style_name,
            'excel_only_dynamic_raw': save_dyn_pre,
            'dynamic_then_scaffold_stats': {
                'dynamic_molecules': n_dyn,
                'reconstruct_success': n_recon_ok,
                'reconstruct_fail': n_recon_fail,
                'final_molecule_count': n_final,
                'prepended_dynamic_raw': len(prepend_meta),
            },
        }
        result = {
            'data': data,
            'pred_ligand_pos': all_pos,
            'pred_ligand_v': all_v,
            'pred_ligand_pos_traj': all_pos_traj,
            'pred_ligand_v_traj': all_v_traj,
            'pred_ligand_log_v_traj': all_log_v_traj,
            'time': all_time,
            'meta': meta_ds,
            'mode': 'dynamic_then_scaffold',
        }
        apply_targetdiff_baseline_refine_to_sampling_result(
            model, data, result, config, device=args.device, logger=logger
        )
    elif sampling_mode == 'scaffold_optimization':
        # 骨架约束进化优化（含 auto_murcko / smarts / atom_indices，见 _resolve_scaffold_mask）
        config.sample.setdefault('scaffold', {})
        mol_path = getattr(args, 'molecule_path', None) or opt_cfg.get('molecule_path', None)
        if mol_path is not None:
            mol_path = Path(mol_path).resolve()
            if not mol_path.exists():
                raise FileNotFoundError(f'待优化分子文件不存在: {mol_path}')
            logger.info(f'[ScaffoldOptimization] 从独立文件加载待优化分子: {mol_path}')
            mol_dict = parse_sdf_file(str(mol_path))
            data.ligand_element = torch.tensor(mol_dict['element'], dtype=torch.long)
            data.ligand_pos = torch.tensor(mol_dict['pos'], dtype=torch.float32)
            data.ligand_bond_index = torch.tensor(mol_dict['bond_index'], dtype=torch.long)
            data.ligand_bond_type = torch.tensor(mol_dict['bond_type'], dtype=torch.long)
            mol_transform = Compose([ligand_featurizer, trans.FeaturizeLigandBond()])
            data = mol_transform(data)
        else:
            # 未指定 molecule_path，使用当前 data 中的配体（来自测试集或 ligand_path）
            use_test_set = getattr(args, 'use_test_set', False)
            data_id = getattr(args, 'data_id', None)
            if use_test_set and data_id is not None:
                logger.info(f'[ScaffoldOptimization] 使用测试集配体作为优化起点: data_id={data_id}')
            elif hasattr(data, 'ligand_filename') and data.ligand_filename:
                logger.info(f'[ScaffoldOptimization] 使用已加载配体作为优化起点: {data.ligand_filename}')
            else:
                logger.info('[ScaffoldOptimization] 使用已加载配体作为优化起点')

        scaff_out = scaffold_optimize_molecule(
            model=model,
            data=data,
            config=config,
            ligand_atom_mode=ligand_atom_mode,
            device=args.device,
            logger=logger,
        )
        result = {
            'data': data,
            'pred_ligand_pos': scaff_out['pos_list'],
            'pred_ligand_v': scaff_out['v_list'],
            'pred_ligand_pos_traj': scaff_out['pos_traj'],
            'pred_ligand_v_traj': scaff_out['v_traj'],
            'pred_ligand_log_v_traj': scaff_out['log_v_traj'],
            'time': scaff_out['time_list'],
            'meta': scaff_out['meta'],
            'mode': 'scaffold_optimization',
        }
        apply_targetdiff_baseline_refine_to_sampling_result(
            model, data, result, config, device=args.device, logger=logger
        )
    elif sampling_mode == 'scaffold_grow':
        # 骨架生长：固定骨架（auto_murcko 等），其余原子从零生成
        config.sample.setdefault('scaffold', {})
        mol_path = getattr(args, 'molecule_path', None) or opt_cfg.get('molecule_path', None)
        if mol_path is not None:
            mol_path = Path(mol_path).resolve()
            if not mol_path.exists():
                raise FileNotFoundError(f'参考配体/骨架文件不存在: {mol_path}')
            logger.info(f'[ScaffoldGrow] 从独立文件加载参考分子: {mol_path}')
            mol_dict = parse_sdf_file(str(mol_path))
            data.ligand_element = torch.tensor(mol_dict['element'], dtype=torch.long)
            data.ligand_pos = torch.tensor(mol_dict['pos'], dtype=torch.float32)
            data.ligand_bond_index = torch.tensor(mol_dict['bond_index'], dtype=torch.long)
            data.ligand_bond_type = torch.tensor(mol_dict['bond_type'], dtype=torch.long)
            mol_transform = Compose([ligand_featurizer, trans.FeaturizeLigandBond()])
            data = mol_transform(data)
        else:
            # 未指定 molecule_path，使用当前 data 中的配体（来自测试集或 ligand_path）
            use_test_set = getattr(args, 'use_test_set', False)
            data_id = getattr(args, 'data_id', None)
            if use_test_set and data_id is not None:
                logger.info(f'[ScaffoldGrow] 使用测试集配体作为骨架参考: data_id={data_id}')
                logger.info(f'[ScaffoldGrow] 将提取 Murcko 骨架并固定，其余原子从零生成')
            elif hasattr(data, 'ligand_filename') and data.ligand_filename:
                logger.info(f'[ScaffoldGrow] 使用已加载配体作为骨架参考: {data.ligand_filename}')
            else:
                logger.info('[ScaffoldGrow] 使用已加载配体作为骨架参考')

        grow_out = scaffold_grow_molecule(
            model=model,
            data=data,
            config=config,
            ligand_atom_mode=ligand_atom_mode,
            device=args.device,
            logger=logger,
            output_dir=args.result_path,  # 传入结果输出目录，用于保存骨架 SDF
        )
        result = {
            'data': data,
            'pred_ligand_pos': grow_out['pos_list'],
            'pred_ligand_v': grow_out['v_list'],
            'pred_ligand_pos_traj': grow_out['pos_traj'],
            'pred_ligand_v_traj': grow_out['v_traj'],
            'pred_ligand_log_v_traj': grow_out['log_v_traj'],
            'time': grow_out['time_list'],
            'meta': grow_out['meta'],
            'mode': 'scaffold_grow',
        }
        apply_targetdiff_baseline_refine_to_sampling_result(
            model, data, result, config, device=args.device, logger=logger
        )
    elif sampling_mode in ('dynamic', 'prudent'):  # 动态采样 / Prudent 多轮筛选。
        # 确保动态采样配置存在，并设置默认值。
        config.sample.setdefault('dynamic', {})  # 如果不存在则创建空字典。
        config.sample.dynamic.setdefault('large_step', {})  # 确保大步配置存在。
        config.sample.dynamic['large_step'].setdefault('batch_size', args.batch_size)  # 设置大步批量大小。
        if sampling_mode == 'prudent':
            config.sample['mode'] = 'prudent'
            config.sample.dynamic.setdefault('prudent', {})
            config.sample.dynamic['prudent'].setdefault('enable', True)
        pr_dyn = config.sample.get('dynamic', {}).get('prudent', {})
        prudent_like = (
            sampling_mode == 'prudent'
            or config.sample.get('mode') == 'prudent'
            or bool(pr_dyn.get('enable'))
        )
        segment_ck_ctx = None
        _pr_seg_ev = prudent_like and bool(pr_dyn.get('segment_evolution_mode', False))
        _legacy_seg_ck = bool(pr_dyn.get('save_checkpoint_every_segment'))
        # OmegaConf.DictConfig 不是 dict，不可用 isinstance(..., dict) 判断「键是否存在」
        _ex_bp = pr_dyn is not None and ('save_segment_breakpoint_pt' in pr_dyn)
        _ex_gen = pr_dyn is not None and ('save_segment_generation_pool_pt' in pr_dyn)
        _want_bp = (
            bool(pr_dyn.get('save_segment_breakpoint_pt')) if _ex_bp else _legacy_seg_ck
        )
        # 分段演化：未显式配置时默认写每代全链 pool（与 save_checkpoint_every_segment 解耦）
        _want_gen = (
            bool(pr_dyn.get('save_segment_generation_pool_pt'))
            if _ex_gen
            else (_legacy_seg_ck or _pr_seg_ev)
        )
        if _pr_seg_ev and (_want_bp or _want_gen):
            ck_sub_raw = pr_dyn.get('segment_checkpoint_subdirectory', 'prudent_segment_ckpt')
            ck_sub = None if ck_sub_raw in (None, '', '~') else str(ck_sub_raw)
            ck_root = (
                os.path.join(result_path_abs, ck_sub)
                if ck_sub
                else result_path_abs
            )
            os.makedirs(ck_root, exist_ok=True)
            segment_ck_ctx = {
                'checkpoint_dir': ck_root,
                'pocket_id': str(pocket_id),
                'timestamp': run_timestamp_cst,
                'it_rounds_tag': _prudent_max_rounds_filename_tag(config),
            }
            if logger:
                logger.info(
                    f'[Prudent] segment 落盘启用 → {ck_root} '
                    f'(breakpoint_pt={_want_bp}, generation_pool_pt={_want_gen})'
                )
        # 执行动态采样（prudent 在 sample_dynamic_diffusion_ligand 内分支）。
        dynamic_output = sample_dynamic_diffusion_ligand(
            model=model,
            data=data,
            config=config,
            ligand_atom_mode=ligand_atom_mode,
            device=args.device,
            logger=logger,
            segment_checkpoint_context=segment_ck_ctx,
        )
        result = {
            'data': data,
            'pred_ligand_pos': dynamic_output['pos_list'],
            'pred_ligand_v': dynamic_output['v_list'],
            'pred_ligand_pos_traj': dynamic_output['pos_traj'],
            'pred_ligand_v_traj': dynamic_output['v_traj'],
            'pred_ligand_log_v_traj': dynamic_output['log_v_traj'],
            'time': dynamic_output['time_list'],
            'meta': dynamic_output['meta'],
            'mode': (
                'prudent' if (
                    sampling_mode == 'prudent'
                    or config.sample.get('mode') == 'prudent'
                    or config.sample.get('dynamic', {}).get('prudent', {}).get('enable')
                ) else 'dynamic'
            ),
        }
    else:  # 基线采样模式。
        # 执行标准扩散采样。
        pred_pos, pred_v, pred_pos_traj, pred_v_traj, pred_v0_traj, pred_vt_traj, time_list = sample_diffusion_ligand(
            model, data, config.sample.num_samples,
            batch_size=args.batch_size, device=args.device,
            num_steps=config.sample.num_steps,
            pos_only=config.sample.pos_only,
            center_pos_mode=config.sample.center_pos_mode,
            sample_num_atoms=config.sample.sample_num_atoms,
            logger=logger
        )
        result = {
            'data': data,
            'pred_ligand_pos': pred_pos,
            'pred_ligand_v': pred_v,
            'pred_ligand_pos_traj': pred_pos_traj,
            'pred_ligand_v_traj': pred_v_traj,
            'pred_ligand_v0_traj': pred_v0_traj,
            'pred_ligand_vt_traj': pred_vt_traj,
            'time': time_list,  # 采样耗时列表。
            'mode': 'baseline'  # 标记采样模式。
        }
    logger.info('Sample done!')  # 记录采样完成信息。
    
    global_profiler.checkpoint('after_sampling')
    
    # 输出最终显存摘要并记录
    summary = global_profiler.get_summary()
    final_mem_info = global_monitor.get_memory_info()
    logger.info(f'[Final] Peak Memory: {summary["peak_memory_mb"]:.1f} MB | Current: {final_mem_info["allocated"]:.1f}/{final_mem_info["max_allocated"]:.1f} MB')
    
    try:
        log_gpu_monitor_record(
            memory_info=final_mem_info,
            memory_summary=summary,
            sampling_info={
                'mode': sampling_mode,
                'stage': 'final',
                'data_id': args.data_id,
            },
            extra_info={
                'result_path': args.result_path,
            },
            logger=logger
        )
    except Exception as e:
        if logger:
            logger.warning(f'Failed to log final GPU monitor record: {e}')

    # 保存结果：将采样结果和配置保存到输出目录。
    result_path = result_path_abs

    config_backup = os.path.join(result_path, 'sample.yml')
    shutil.copyfile(args.config, config_backup)  # 备份采样配置文件。

    # Prudent：在文件名中附上闭环/分段迭代上限 prudent.max_checkpoint_rounds（可与按段 checkpoint 同一时间戳对齐）
    timestamp = run_timestamp_cst
    _it_rounds = _prudent_max_rounds_filename_tag(config)
    if _it_rounds is not None:
        result_file = os.path.join(result_path, f'result_{pocket_id}_it{_it_rounds}_{timestamp}.pt')
    else:
        result_file = os.path.join(result_path, f'result_{pocket_id}_{timestamp}.pt')
    
    # 准备extra_info（在保存文件之前）
    extra_info = {
        'data_id': args.data_id if not use_custom_files else 'custom',
        'protein_path': str(args.protein_path) if use_custom_files else None,
        'ligand_path': str(args.ligand_path) if use_custom_files and args.ligand_path else None,
        'config_backup': config_backup,
        'result_file': os.path.abspath(result_file),  # 保存result_file路径，用于评估时匹配
    }
    
    # 将extra_info添加到result字典中
    result['extra_info'] = extra_info

    _pr_meta = (result.get('meta') or {}).get('prudent') or {}
    _pending = _pr_meta.get('pending_analysis_hits')
    if _pending:
        _rf_abs = os.path.abspath(result_file)
        for _item in _pending:
            if isinstance(_item, dict):
                _item['result_pt'] = _rf_abs

    torch.save(result, result_file)  # 保存采样结果为 PyTorch 文件。
    logger.info(f'Results saved to: {result_file}')

    # 跳步留存：每步保存 SDF（若启用），按分子身份证（口袋+时间+评分）命名
    save_step_trajectory_sdf(
        result=result,
        result_path=result_path,
        config=config,
        ligand_atom_mode=ligand_atom_mode,
        pocket_id=pocket_id,
        timestamp=timestamp,
        logger=logger
    )

    # ⚠️ 已禁用：自动执行转换器脚本生成SDF文件
    # 原因：使用错误的converter会导致分子结构错误（缺少氢原子、键级错误等）
    # 现在改为使用正确的evaluate_pt_with_correct_reconstruct.py进行重建和评估
    # converter_script = REPO_ROOT / 'targetdiff_pt_to_sdf_converter.py'
    # sdf_output_dir = None
    # if converter_script.exists():
    #     logger.info(f'Executing converter script: {converter_script}')
    #     # 生成SDF文件的输出目录（基于.pt文件名）
    #     sdf_output_dir = os.path.join(result_path, f'sdf_{timestamp}')
    #     try:
    #         # 执行转换器脚本
    #         subprocess.run([
    #             sys.executable, str(converter_script), result_file,
    #             '--output_dir', sdf_output_dir
    #         ], check=True)
    #         logger.info(f'SDF files generated in: {sdf_output_dir}')
    #     except subprocess.CalledProcessError as e:
    #         logger.error(f'Failed to execute converter script: {e}')
    #     except Exception as e:
    #         logger.error(f'Error executing converter script: {e}')
    # else:
    #     logger.warning(f'Converter script not found: {converter_script}')
    sdf_output_dir = None  # 不再使用错误的converter生成SDF

    # 评估说明：
    # 采样完成后，请使用 evaluate_pt_with_correct_reconstruct.py 单独进行评测
    # 评测完成后，evaluate_pt_with_correct_reconstruct.py 会自动更新 sampling_history.xlsx

    # 记录采样元信息到 Excel
    sampling_params = extract_sampling_params(config)
    
    # 生成采样步骤信息
    try:
        # 递归转换 EasyDict 为普通字典
        def convert_to_dict(obj):
            # 处理 EasyDict 对象（可以像字典一样迭代）
            if hasattr(obj, 'items') and callable(obj.items):
                try:
                    return {str(k): convert_to_dict(v) for k, v in obj.items()}
                except (AttributeError, TypeError):
                    pass
            # 处理普通字典
            if isinstance(obj, dict):
                return {str(k): convert_to_dict(v) for k, v in obj.items()}
            # 处理列表和元组
            elif isinstance(obj, (list, tuple)):
                return [convert_to_dict(item) for item in obj]
            # 处理其他类型（包括基本类型、None等）
            else:
                # 尝试转换为 Python 原生类型
                if hasattr(obj, 'item') and callable(obj.item):
                    try:
                        return obj.item()
                    except Exception:
                        pass
                return obj
        
        config_dict = convert_to_dict(config)
        sampling_steps_text = generate_sampling_steps_text(config_dict)
        
        # 将采样步骤信息添加到 extra_info
        if extra_info is None:
            extra_info = {}
        extra_info['sampling_steps'] = sampling_steps_text
        
        if logger:
            logger.info('采样步骤信息已生成并添加到记录中')
    except Exception as e:
        if logger:
            logger.warning(f'生成采样步骤信息失败: {e}')
        # 即使失败也继续记录，只是不包含采样步骤信息
    
    log_sampling_record(
        params=sampling_params,
        result_dir=result_path,
        sampling_mode=result.get('mode', sampling_mode) if isinstance(result, dict) else sampling_mode,
        result_file=result_file,
        logger=logger,
        extra_info=extra_info
    )


