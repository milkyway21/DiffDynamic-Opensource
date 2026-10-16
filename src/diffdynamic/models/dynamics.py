"""Dynamics components of the DiffDynamic model."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from diffdynamic.models.probability import cal_kl_gradient
from diffdynamic.models.probability import clamp_by_norm
from diffdynamic.models.schedules import adaptive_step_size


class DynamicDiffusionMixin:
    """Dynamics responsibilities; registered parameter names stay unchanged."""

    def _dynamic_diffusion(self, protein_pos, protein_v, batch_protein,
                           ligand_pos, log_ligand_v, batch_ligand,
                           time_indices, step_size,
                           add_noise=0.0, pos_clip=None, v_clip=None,
                           record_traj=True, pos_only=False, 
                           enable_normalization=True, enable_monitoring=False,
                           clip_strategy='norm', use_adaptive_step=False,
                           use_with_noise=False, use_time_scale=True,
                           repaint_cfg=None, guidance_cfg=None):  # repaint_cfg: RePaint 掩码参考态；guidance_cfg: 分类器引导（对属性损失梯度）。
        time_indices = list(time_indices)
        num_graphs = batch_protein.max().item() + 1  # 获取批次数。
        pos_traj, log_v_traj = [], []  # 准备轨迹存储列表。
        
        # 初始化性能监控指标
        metrics = {'grad_norms': [], 'lambda_values': [], 'local_pos_grad_norms': [], 
                   'global_pos_grad_norms': [], 'combined_pos_grad_norms': [],
                   'local_v_grad_norms': [], 'global_v_grad_norms': [], 
                   'combined_v_grad_norms': []} if enable_monitoring else {}  # 监控指标字典。

        repaint_sampler = None  # RePaint：每步将掩码原子对齐到 q(x_{t_next}|x_0)
        if repaint_cfg is not None:
            if repaint_cfg.get('_use_dual_mask'):
                # 双掩码模式：位置掩码与类型掩码独立控制（骨架约束优化专用）
                from diffdynamic.sampling.guidance import DualMaskedDiffusionSampler
                repaint_sampler = DualMaskedDiffusionSampler(
                    model=self,
                    x0_pos=repaint_cfg['x0_pos'],
                    x0_log_v=repaint_cfg['x0_log_v'],
                    pos_mask=repaint_cfg['pos_mask'],
                    type_mask=repaint_cfg.get('type_mask'),
                    fixed_eps_pos=repaint_cfg.get('fixed_eps_pos'),
                    use_mean_for_discrete=repaint_cfg.get('use_mean_for_discrete', True),
                    type_context_mode=repaint_cfg.get(
                        '_type_context_mode', 'q'
                    ),
                )
            else:
                from diffdynamic.sampling.guidance import MaskedDiffusionSampler  # 延迟导入
                repaint_sampler = MaskedDiffusionSampler(
                    model=self,
                    x0_pos=repaint_cfg['x0_pos'],
                    x0_log_v=repaint_cfg['x0_log_v'],
                    atom_mask=repaint_cfg['atom_mask'],
                    fixed_eps_pos=repaint_cfg.get('fixed_eps_pos'),
                    use_mean_for_discrete=repaint_cfg.get('use_mean_for_discrete', True),
                )

        for i, t_scalar in enumerate(time_indices):  # 遍历时间调度列表（使用 enumerate 以便计算时间跨度）。
            guidance_g_v_cache = None  # 每步重置：分类器引导对类别分布的附加项
            t = torch.full(size=(num_graphs,), fill_value=int(t_scalar), dtype=torch.long, device=protein_pos.device)  # 构造当前批次时间张量。
            
            # 动态计算时间跨度 n（用于跳步采样时的步长缩放）
            if i < len(time_indices) - 1:
                n = abs(t_scalar - time_indices[i + 1])
            else:
                n = abs(time_indices[i] - time_indices[i - 1]) if i > 0 else 1
            n = max(1, int(n))  # 确保 n >= 1
            
            # 计算时间尺度缩放因子（基于布朗运动位移性质：位移与 sqrt(时间) 成正比）
            # 根据配置决定是否启用时间缩放
            if use_time_scale:
                time_scale_factor = np.sqrt(n)
            else:
                time_scale_factor = 1.0  # 不启用时，缩放因子为 1（相当于不缩放）
            
            # 验证输入维度（在模型前向传播前）
            if ligand_pos.numel() == 0 or log_ligand_v.numel() == 0 or batch_ligand.numel() == 0:
                raise ValueError(
                    f"_dynamic_diffusion: Empty input detected at time step {t_scalar}. "
                    f"ligand_pos.numel()={ligand_pos.numel()}, log_ligand_v.numel()={log_ligand_v.numel()}, "
                    f"batch_ligand.numel()={batch_ligand.numel()}, num_graphs={num_graphs}"
                )
            
            if ligand_pos.shape[0] != batch_ligand.shape[0]:
                raise ValueError(
                    f"_dynamic_diffusion: Dimension mismatch at time step {t_scalar}. "
                    f"ligand_pos.shape={ligand_pos.shape}, batch_ligand.shape={batch_ligand.shape}, "
                    f"batch_ligand.range=[{batch_ligand.min().item()}, {batch_ligand.max().item()}], "
                    f"num_graphs={num_graphs}"
                )
            
            # 验证批次索引范围
            if batch_ligand.max().item() >= num_graphs or batch_ligand.min().item() < 0:
                raise ValueError(
                    f"_dynamic_diffusion: Invalid batch_ligand indices at time step {t_scalar}. "
                    f"batch_ligand.range=[{batch_ligand.min().item()}, {batch_ligand.max().item()}], "
                    f"num_graphs={num_graphs}"
                )
            
            ligand_input = self._format_forward_ligand_input(log_ligand_v)  # 根据模式准备配体输入。
            
            # 验证配体输入格式
            if isinstance(ligand_input, torch.Tensor):
                if ligand_input.numel() == 0:
                    raise ValueError(
                        f"_dynamic_diffusion: Empty ligand_input after formatting at time step {t_scalar}. "
                        f"log_ligand_v.shape={log_ligand_v.shape}, ligand_input.shape={ligand_input.shape}"
                    )
                if ligand_input.shape[0] != batch_ligand.shape[0]:
                    raise ValueError(
                        f"_dynamic_diffusion: ligand_input shape mismatch at time step {t_scalar}. "
                        f"ligand_input.shape={ligand_input.shape}, batch_ligand.shape={batch_ligand.shape}"
                    )
            
            # 模型前向传播
            preds = self(
                protein_pos=protein_pos,  # 蛋白坐标。
                protein_v=protein_v,  # 蛋白特征。
                batch_protein=batch_protein,  # 蛋白批次索引。
                init_ligand_pos=ligand_pos,  # 当前配体坐标。
                init_ligand_v=ligand_input,  # 当前配体类别表示。
                batch_ligand=batch_ligand,  # 配体批次索引。
                time_step=t  # 当前时间步。
            )

            # 验证模型输出
            if 'pred_ligand_pos' not in preds:
                raise ValueError(
                    f"_dynamic_diffusion: Model output missing 'pred_ligand_pos' key at time step {t_scalar}. "
                    f"Available keys: {list(preds.keys())}"
                )
            
            pred_ligand_pos = preds['pred_ligand_pos']
            if pred_ligand_pos.numel() == 0:
                raise ValueError(
                    f"_dynamic_diffusion: Model output 'pred_ligand_pos' is empty at time step {t_scalar}. "
                    f"ligand_pos.shape={ligand_pos.shape}, batch_ligand.shape={batch_ligand.shape}, "
                    f"protein_pos.shape={protein_pos.shape}, protein_v.shape={protein_v.shape}, "
                    f"num_graphs={num_graphs}. This indicates a problem with model forward pass or model weights."
                )
            
            if pred_ligand_pos.shape[0] != ligand_pos.shape[0]:
                raise ValueError(
                    f"_dynamic_diffusion: Model output shape mismatch at time step {t_scalar}. "
                    f"pred_ligand_pos.shape={pred_ligand_pos.shape}, ligand_pos.shape={ligand_pos.shape}, "
                    f"batch_ligand.shape={batch_ligand.shape}"
                )

            if self.model_mean_type == 'noise':  # 当模型预测噪声时。
                pred_pos_noise = pred_ligand_pos - ligand_pos  # 计算预测噪声。
                
                # 验证预测噪声的形状
                if pred_pos_noise.shape[0] != ligand_pos.shape[0]:
                    raise ValueError(
                        f"_dynamic_diffusion: Predicted noise shape mismatch at time step {t_scalar}. "
                        f"pred_pos_noise.shape={pred_pos_noise.shape}, ligand_pos.shape={ligand_pos.shape}"
                    )
                
                pos0_from_e = self._predict_x0_from_eps(xt=ligand_pos, eps=pred_pos_noise, t=t, batch=batch_ligand)  # 反推 x0。
            elif self.model_mean_type == 'C0':  # 当模型直接预测 x0 时。
                pos0_from_e = pred_ligand_pos  # 直接使用预测。
            else:  # 未知模式。
                raise ValueError(f"Unknown model_mean_type: {self.model_mean_type}")  # 抛出异常。
            
            # 验证 pos0_from_e 不为空
            if pos0_from_e.numel() == 0:
                raise ValueError(
                    f"_dynamic_diffusion: pos0_from_e is empty at time step {t_scalar}. "
                    f"model_mean_type={self.model_mean_type}, pred_ligand_pos.shape={pred_ligand_pos.shape}, "
                    f"ligand_pos.shape={ligand_pos.shape}, batch_ligand.shape={batch_ligand.shape}"
                )
            
            if pos0_from_e.shape[0] != ligand_pos.shape[0]:
                raise ValueError(
                    f"_dynamic_diffusion: pos0_from_e shape mismatch at time step {t_scalar}. "
                    f"pos0_from_e.shape={pos0_from_e.shape}, ligand_pos.shape={ligand_pos.shape}"
                )

            # 在调用 q_pos_posterior 前验证批次索引的有效性（TargetDiff 原始代码的保护机制）
            if batch_ligand.max().item() >= num_graphs:
                raise ValueError(
                    f"batch_ligand contains invalid indices: max={batch_ligand.max().item()}, "
                    f"num_graphs={num_graphs}. This should not happen in normal operation. "
                    f"batch_ligand should contain values in [0, {num_graphs-1}]."
                )
            if batch_ligand.min().item() < 0:
                raise ValueError(
                    f"batch_ligand contains negative indices: min={batch_ligand.min().item()}. "
                    f"This should not happen in normal operation."
                )
            
            # 在调用 q_pos_posterior 前再次验证输入
            if pos0_from_e.shape[0] != ligand_pos.shape[0] or ligand_pos.shape[0] != batch_ligand.shape[0]:
                raise ValueError(
                    f"_dynamic_diffusion: Input dimension mismatch before q_pos_posterior at time step {t_scalar}. "
                    f"pos0_from_e.shape={pos0_from_e.shape}, ligand_pos.shape={ligand_pos.shape}, "
                    f"batch_ligand.shape={batch_ligand.shape}, num_graphs={num_graphs}"
                )
            
            pos_model_mean = self.q_pos_posterior(x0=pos0_from_e, xt=ligand_pos, t=t, batch=batch_ligand)  # 计算局部均值。
            
            # 检查维度一致性（TargetDiff 原始代码中，这应该总是匹配的）
            if pos_model_mean.shape[0] != ligand_pos.shape[0]:
                if enable_monitoring:  # 如果启用监控，记录详细信息。
                    metrics['dimension_mismatches'] = metrics.get('dimension_mismatches', [])
                    metrics['dimension_mismatches'].append({
                        'pos_model_mean_shape': pos_model_mean.shape,
                        'ligand_pos_shape': ligand_pos.shape,
                        'batch_ligand_shape': batch_ligand.shape,
                        'time_step': t_scalar,
                        'num_graphs': num_graphs,
                        'batch_ligand_range': [batch_ligand.min().item(), batch_ligand.max().item()]
                    })
                # 直接抛出错误，不要尝试修复（这会破坏批次索引的一致性）
                raise ValueError(
                    f"Dimension mismatch after q_pos_posterior: pos_model_mean.shape={pos_model_mean.shape}, "
                    f"ligand_pos.shape={ligand_pos.shape}, batch_ligand.shape={batch_ligand.shape}, "
                    f"num_graphs={num_graphs}, batch_ligand_range=[{batch_ligand.min().item()}, {batch_ligand.max().item()}]. "
                    f"This should not happen in normal operation. Please check the extract function and batch_ligand values."
                )
            
            local_pos_grad = pos_model_mean - ligand_pos  # 局部梯度（后验均值与当前点差）。
            global_pos_grad = pos0_from_e - ligand_pos  # 全局梯度（预测 x0 与当前点差）。
            
            # 计算原始梯度范数（用于监控和自适应计算，在归一化之前）
            local_pos_grad_norm_val = torch.norm(local_pos_grad, dim=-1).mean().item()  # 局部梯度平均范数。
            global_pos_grad_norm_val = torch.norm(global_pos_grad, dim=-1).mean().item()  # 全局梯度平均范数。
            grad_norm_ratio = global_pos_grad_norm_val / (local_pos_grad_norm_val + 1e-8) if local_pos_grad_norm_val > 0 else 1.0  # 计算梯度范数比率。
            
            # 增强数值稳定性：在梯度融合前添加归一化（可选）
            if enable_normalization:  # 若启用归一化。
                local_pos_grad_norm = torch.norm(local_pos_grad, dim=-1, keepdim=True)  # 计算局部梯度范数。
                local_pos_grad = local_pos_grad / (local_pos_grad_norm + 1e-8)  # 归一化局部梯度。
                global_pos_grad_norm = torch.norm(global_pos_grad, dim=-1, keepdim=True)  # 计算全局梯度范数。
                global_pos_grad = global_pos_grad / (global_pos_grad_norm + 1e-8)  # 归一化全局梯度。
            
            # 根据梯度信息计算 lambda（支持自适应模式）
            lambda_val = self._compute_grad_lambda(int(t_scalar), grad_norm_ratio=grad_norm_ratio)  # 根据时间计算融合系数。
            
            if self.use_grad_fusion:  # 若启用梯度融合。
                combined_pos_grad = lambda_val * global_pos_grad + (1.0 - lambda_val) * local_pos_grad  # 融合全局与局部梯度。
            else:  # 未启用融合。
                combined_pos_grad = local_pos_grad  # 使用局部梯度。
            
            # 使用多策略梯度裁剪
            combined_pos_grad = clamp_by_norm(combined_pos_grad, pos_clip, clip_strategy=clip_strategy)  # 对梯度进行裁剪。

            # 分类器引导：对属性预测器损失求梯度，叠加到位置更新方向（DiffGUI / classifier-free guidance 思路）
            if guidance_cfg is not None and guidance_cfg.get('fn') is not None:
                from diffdynamic.sampling.guidance import apply_classifier_guidance, guidance_in_t_range  # 延迟导入
                gc = guidance_cfg
                t_lo = int(gc.get('t_min', 0))
                t_hi = int(gc.get('t_max', self.num_timesteps - 1))
                if guidance_in_t_range(int(t_scalar), t_lo, t_hi):
                    g_pos, g_logv = apply_classifier_guidance(
                        ligand_pos,
                        log_ligand_v,
                        batch_ligand,
                        gc['fn'],
                        scale=float(gc.get('scale', 0.1)),
                        atom_mask=gc.get('atom_mask'),
                        apply_to_types=(not pos_only) and gc.get('apply_to_types', True),
                    )
                    combined_pos_grad = combined_pos_grad + g_pos
                    guidance_g_v_cache = g_logv
            
            # 自适应步长调整（可选）
            current_step_size = step_size  # 默认使用原始步长。
            if use_adaptive_step:  # 若启用自适应步长。
                combined_pos_grad_norm = torch.norm(combined_pos_grad, dim=-1).mean().item()  # 计算融合后梯度范数。
                current_step_size = adaptive_step_size(combined_pos_grad_norm, step_size)  # 自适应调整步长。
            
            # 应用时间尺度缩放因子到实际步长
            actual_step_size = current_step_size * time_scale_factor
            
            ligand_pos = ligand_pos + actual_step_size * combined_pos_grad  # 按缩放后的步长更新配体坐标。
            if add_noise and add_noise > 0:  # 如果需要添加额外噪声。
                ligand_pos = ligand_pos + add_noise * torch.randn_like(ligand_pos)  # 加入高斯扰动。

            if not pos_only:  # 若同时更新类别。
                log_ligand_v_recon = F.log_softmax(preds['pred_ligand_v'], dim=-1)  # 计算当前预测分布。
                # 根据配置选择使用标准方法或带噪声方法计算后验
                if use_with_noise:  # 使用带噪声的后验计算，提供更平滑的分布。
                    log_model_prob = self.q_v_posterior_with_noise(log_ligand_v_recon, log_ligand_v, t, batch_ligand)  # 计算带噪声的局部后验。
                else:  # 使用标准方法。
                    log_model_prob = self.q_v_posterior(log_ligand_v_recon, log_ligand_v, t, batch_ligand)  # 计算局部后验。
                local_v_grad = cal_kl_gradient(log_model_prob, log_ligand_v)  # 计算局部梯度。
                global_v_grad = cal_kl_gradient(log_ligand_v_recon, log_ligand_v)  # 计算全局梯度。
                
                # 计算原始类别梯度范数（用于监控，在归一化之前）
                local_v_grad_norm_val = torch.norm(local_v_grad, dim=-1).mean().item()  # 局部类别梯度平均范数。
                global_v_grad_norm_val = torch.norm(global_v_grad, dim=-1).mean().item()  # 全局类别梯度平均范数。
                
                # 增强数值稳定性：类别梯度归一化（可选）
                if enable_normalization:  # 若启用归一化。
                    local_v_grad_norm = torch.norm(local_v_grad, dim=-1, keepdim=True)  # 计算局部类别梯度范数。
                    local_v_grad = local_v_grad / (local_v_grad_norm + 1e-8)  # 归一化局部类别梯度。
                    global_v_grad_norm = torch.norm(global_v_grad, dim=-1, keepdim=True)  # 计算全局类别梯度范数。
                    global_v_grad = global_v_grad / (global_v_grad_norm + 1e-8)  # 归一化全局类别梯度。
                
                if self.use_grad_fusion:  # 梯度融合。
                    combined_v_grad = lambda_val * global_v_grad + (1.0 - lambda_val) * local_v_grad  # 融合类别梯度。
                else:  # 不融合。
                    combined_v_grad = local_v_grad  # 使用局部梯度。
                
                # 使用多策略梯度裁剪
                combined_v_grad = clamp_by_norm(combined_v_grad, v_clip, clip_strategy=clip_strategy)  # 裁剪梯度。
                if guidance_g_v_cache is not None:
                    combined_v_grad = combined_v_grad + guidance_g_v_cache  # 与位置引导联合反传的类别引导
                
                # 自适应步长调整（类别梯度）
                if use_adaptive_step:  # 若启用自适应步长。
                    combined_v_grad_norm = torch.norm(combined_v_grad, dim=-1).mean().item()  # 计算融合后类别梯度范数。
                    current_step_size_v = adaptive_step_size(combined_v_grad_norm, step_size)  # 自适应调整步长。
                    actual_step_size_v = current_step_size_v * time_scale_factor  # 应用时间尺度缩放因子
                else:
                    actual_step_size_v = step_size * time_scale_factor  # 应用时间尺度缩放因子
                
                log_ligand_v = F.log_softmax(log_ligand_v + actual_step_size_v * combined_v_grad, dim=-1)  # 使用缩放后的步长更新 log 概率。
            
            # 性能监控：记录关键指标
            if enable_monitoring:  # 若启用监控。
                metrics['lambda_values'].append(lambda_val)  # 记录 lambda 值。
                metrics['local_pos_grad_norms'].append(local_pos_grad_norm_val)  # 记录局部位置梯度范数。
                metrics['global_pos_grad_norms'].append(global_pos_grad_norm_val)  # 记录全局位置梯度范数。
                combined_pos_grad_norm_val = torch.norm(combined_pos_grad, dim=-1).mean().item()  # 计算融合后梯度范数。
                metrics['combined_pos_grad_norms'].append(combined_pos_grad_norm_val)  # 记录融合后位置梯度范数。
                metrics['grad_norms'].append(combined_pos_grad_norm_val)  # 记录主要梯度范数。
                if not pos_only:  # 若更新了类别。
                    combined_v_grad_norm_val = torch.norm(combined_v_grad, dim=-1).mean().item()  # 融合后类别梯度范数。
                    metrics['local_v_grad_norms'].append(local_v_grad_norm_val)  # 记录局部类别梯度范数。
                    metrics['global_v_grad_norms'].append(global_v_grad_norm_val)  # 记录全局类别梯度范数。
                    metrics['combined_v_grad_norms'].append(combined_v_grad_norm_val)  # 记录融合后类别梯度范数。

            # RePaint / 掩码融合：逆向第 i 步结束后，将掩码原子对齐到 t_next 上的前向参考态 q(x_{t_next}|x_0)
            if repaint_sampler is not None:
                ref_pos_n, ref_logv_n = repaint_sampler.reference_at_t_next(batch_ligand, time_indices, i)
                ligand_pos, log_ligand_v = repaint_sampler.blend_after_step(
                    ligand_pos, log_ligand_v, ref_pos_n, ref_logv_n
                )

            if record_traj:  # 如需记录轨迹：先留在 GPU，循环结束再一次性 D2H，避免逐步同步。
                pos_traj.append(ligand_pos.detach())
                log_v_traj.append(log_ligand_v.detach())

        if record_traj and pos_traj:
            pos_traj = [p.cpu() for p in pos_traj]
            log_v_traj = [v.cpu() for v in log_v_traj]

        # 返回最新状态、轨迹和监控指标
        if enable_monitoring:  # 若启用了监控。
            return ligand_pos, log_ligand_v, pos_traj, log_v_traj, metrics  # 返回最新状态与轨迹及监控指标。
        return ligand_pos, log_ligand_v, pos_traj, log_v_traj  # 返回最新状态与轨迹。
