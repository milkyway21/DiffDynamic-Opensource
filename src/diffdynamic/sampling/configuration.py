"""Sampling configuration responsibilities for ligand sampling."""

from __future__ import annotations

from diffdynamic.models.constants import GRAD_FUSION_CAP_UNSPECIFIED


def _dynamic_subcfg_grad_cap_kwargs(subcfg):
    """large_step / refine 子配置 → sample_diffusion_* 的梯度融合上限参数。

    仅当 YAML 显式写出键时才传入具体值；未写键用 GRAD_FUSION_CAP_UNSPECIFIED 走该段 defaults。
    显式 ``null`` 表示不截断（跑满调度），与「未写键」区分。
    """
    kwargs = {
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
    if 'preserve_schedule_endpoint' in subcfg:
        kwargs['preserve_schedule_endpoint'] = bool(
            subcfg['preserve_schedule_endpoint']
        )
    return kwargs


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
