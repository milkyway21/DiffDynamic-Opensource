"""Sampling naming responsibilities for ligand sampling."""

from __future__ import annotations

from datetime import datetime
from datetime import timedelta
from datetime import timezone


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
    from diffdynamic.sampling.configuration import get_time_boundary

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
