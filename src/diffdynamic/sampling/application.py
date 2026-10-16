"""Sampling application responsibilities for ligand sampling."""

from __future__ import annotations

from datetime import datetime
from datetime import timedelta
from datetime import timezone
from diffdynamic.data.datasets import get_dataset
from diffdynamic.models.score_model import DiffDynamic
from diffdynamic.models.score_model import ScorePosNet3D
from pathlib import Path
from diffdynamic.analysis.sampling_steps import generate_sampling_steps_text
from torch_geometric.transforms import Compose
from diffdynamic.io.gpu_monitor_recorder import log_gpu_monitor_record
from diffdynamic.io.monitor import GPUMonitor
from diffdynamic.io.monitor import MemoryProfiler
from diffdynamic.io.sampling_recorder import extract_sampling_params
from diffdynamic.io.sampling_recorder import log_sampling_record
import os
import shutil
import torch
import diffdynamic.io.config as misc
import diffdynamic.data.transforms as trans


def main(argv=None):
    """Run sampling using the established CLI and output contract."""
    from diffdynamic.data.pocket import load_custom_pocket_data
    from diffdynamic.data.pocket import resolve_and_set_absolute_protein_path
    from diffdynamic.sampling.prudent.state import _prudent_max_rounds_filename_tag
    from diffdynamic.sampling.trajectory import save_step_trajectory_sdf
    from diffdynamic.sampling.arguments import build_parser
    from diffdynamic.sampling.pipelines.dynamic_then_optimization import run_dynamic_then_optimization
    from diffdynamic.sampling.pipelines.optimization import run_optimization
    from diffdynamic.sampling.pipelines.dynamic_then_scaffold import run_dynamic_then_scaffold
    from diffdynamic.sampling.pipelines.scaffold_optimization import run_scaffold_optimization
    from diffdynamic.sampling.pipelines.scaffold_grow import run_scaffold_grow
    from diffdynamic.sampling.pipelines.scaffold_dynamic_locked import run_scaffold_dynamic_locked
    from diffdynamic.sampling.pipelines.scaffold_prudent import run_scaffold_prudent
    from diffdynamic.sampling.pipelines.dynamic import run_dynamic
    from diffdynamic.sampling.pipelines.baseline import run_baseline

    args = build_parser().parse_args(argv)  # 解析命令行参数。
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
        elif _sc_mode == 'dynamic_locked':
            sampling_mode = 'scaffold_dynamic_locked'
        elif _sc_mode == 'prudent':
            sampling_mode = 'scaffold_prudent'
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
        'scaffold_grow', 'scaffold_optimization', 'dynamic_then_scaffold', 'scaffold_dynamic_locked',
        'scaffold_prudent',
    ]:
        raise ValueError(f'Unsupported sampling mode: {sampling_mode}')

    result_path_abs = os.path.abspath(args.result_path)
    os.makedirs(result_path_abs, exist_ok=True)
    _cst_tz_run = timezone(timedelta(hours=8))
    run_timestamp_cst = datetime.now(_cst_tz_run).strftime('%Y%m%d_%H%M%S')

    if sampling_mode == 'dynamic_then_optimization':
        result = run_dynamic_then_optimization(args, model, data, config, ligand_atom_mode, ligand_featurizer, logger)
    elif sampling_mode == 'optimization':
        result = run_optimization(args, model, data, config, ligand_atom_mode, ligand_featurizer, logger, opt_cfg)
    elif sampling_mode == 'dynamic_then_scaffold':
        result = run_dynamic_then_scaffold(args, model, data, config, ligand_atom_mode, ligand_featurizer, logger)
    elif sampling_mode == 'scaffold_optimization':
        result = run_scaffold_optimization(args, model, data, config, ligand_atom_mode, ligand_featurizer, logger, opt_cfg)
    elif sampling_mode == 'scaffold_grow':
        result = run_scaffold_grow(args, model, data, config, ligand_atom_mode, ligand_featurizer, logger, opt_cfg)
    elif sampling_mode == 'scaffold_dynamic_locked':
        result = run_scaffold_dynamic_locked(args, model, data, config, ligand_atom_mode, ligand_featurizer, logger, opt_cfg)
    elif sampling_mode == 'scaffold_prudent':
        result = run_scaffold_prudent(args, model, data, config, ligand_atom_mode, ligand_featurizer, logger, opt_cfg)
    elif sampling_mode in ('dynamic', 'prudent'):
        result = run_dynamic(args, model, data, config, ligand_atom_mode, logger, pocket_id, result_path_abs, run_timestamp_cst, sampling_mode)
    else:
        result = run_baseline(args, model, data, config, logger)
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
    _pfm = result.pop('_prudent_final_pool_meta', None) or {}
    if _pfm.get('n_final_pool') is not None:
        extra_info['n_final_pool'] = int(_pfm['n_final_pool'])
    if _pfm.get('final_pool_start') is not None:
        extra_info['final_pool_start'] = int(_pfm['final_pool_start'])
    if _pfm.get('target_final_count') is not None:
        extra_info['target_final_count'] = int(_pfm['target_final_count'])
    
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
