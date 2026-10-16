"""Samplers components of the DiffDynamic model."""

from __future__ import annotations

from tqdm.auto import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from diffdynamic.models.constants import GRAD_FUSION_CAP_UNSPECIFIED
from diffdynamic.models.probability import center_pos
from diffdynamic.models.probability import ensure_log_ligand
from diffdynamic.models.probability import extract
from diffdynamic.models.probability import index_to_log_onehot
from diffdynamic.models.probability import log_sample_categorical


class DiffusionSamplingMixin:
    """Samplers responsibilities; registered parameter names stay unchanged."""

    @torch.no_grad()
    def sample_diffusion_large_step(self, protein_pos, protein_v, batch_protein,
                                    init_ligand_pos, init_ligand_v, batch_ligand,
                                    num_steps=None, center_pos_mode=None, pos_only=False,
                                    step_stride=None, step_size=None, add_noise=None,
                                    pos_clip=None, v_clip=None, log_ligand_input_mode='auto',
                                    max_gradient_steps=GRAD_FUSION_CAP_UNSPECIFIED,
                                    max_grad_fusion_iterations=GRAD_FUSION_CAP_UNSPECIFIED,
                                    grad_fusion_anchor_t=None,
                                    preserve_schedule_endpoint=None,
                                    repaint_cfg=None):

        if num_steps is None:  # 如果未指定步数。
            num_steps = self.num_timesteps  # 默认使用全部时间步。
        if center_pos_mode is None:  # 若未指定中心化模式。
            center_pos_mode = self.center_pos_mode  # 使用默认配置。

        defaults = self.dynamic_large_step_defaults or {}  # 读取大步采样的默认参数。
        schedule_mode = defaults.get('schedule', 'lambda')  # 获取调度模式。
        use_lambda_schedule = schedule_mode == 'lambda'  # 判断是否使用 lambda 调度。
        use_linear_schedule = schedule_mode == 'linear'  # 判断是否使用线性调度。
        if not use_lambda_schedule and not use_linear_schedule:  # 若使用固定步长调度。
            step_stride = int(step_stride if step_stride is not None else defaults.get('stride', 50))  # 设置步长。
            step_stride = max(step_stride, 1)  # 保证步长至少为 1。
        step_size = float(step_size if step_size is not None else defaults.get('step_size', 1.0))  # 设置更新步幅。
        add_noise = float(add_noise if add_noise is not None else defaults.get('noise_scale', 0.0))  # 设置附加噪声。
        pos_clip = pos_clip if pos_clip is not None else defaults.get('pos_clip', self.dynamic_pos_step_clip)  # 位置梯度裁剪阈值。
        v_clip = v_clip if v_clip is not None else defaults.get('v_clip', self.dynamic_v_step_clip)  # 类别梯度裁剪阈值。

        protein_pos, ligand_pos, offset = center_pos(
            protein_pos, init_ligand_pos, batch_protein, batch_ligand, mode=center_pos_mode)  # 按需中心化坐标。

        log_mode = log_ligand_input_mode  # 决定 log 输入模式。
        if log_mode == 'auto':  # 自动模式根据配置选择。
            log_mode = self.ligand_v_input if self.ligand_v_input in ('log_prob', 'logits', 'prob') else 'auto'  # 若模型原本使用概率形式则保持一致。
        log_ligand_v = ensure_log_ligand(init_ligand_v, self.num_classes, mode=log_mode)  # 将输入转换为 log 概率。

        min_t = max(self.num_timesteps - num_steps, 0)  # 计算最小时间步。
        if use_lambda_schedule:  # 使用 lambda 调度。
            lambda_coeff_a = defaults.get('lambda_coeff_a', 80.0)  # 调度参数 a。
            lambda_coeff_b = defaults.get('lambda_coeff_b', 20.0)  # 调度参数 b。
            lambda_floor = defaults.get('time_lower', self.num_timesteps // 2)  # 调度下限。
            lambda_floor = int(np.clip(lambda_floor, min_t, self.num_timesteps - 1))  # 限制在合法范围。
            time_indices = self._build_lambda_schedule(
                start_t=self.num_timesteps - 1,  # 起始于最后一个时间步。
                end_t=lambda_floor,  # 结束于调度下限。
                coeff_a=lambda_coeff_a,  # 传入参数 a。
                coeff_b=lambda_coeff_b  # 传入参数 b。
            )
        elif use_linear_schedule:  # 使用线性调度。
            lambda_floor = defaults.get('time_lower', self.num_timesteps // 2)  # 调度下限。
            lambda_floor = int(np.clip(lambda_floor, min_t, self.num_timesteps - 1))  # 限制在合法范围。
            # 支持两种配置方式：直接指定上下限（推荐），或指定系数（向后兼容）
            if 'linear_step_upper' in defaults or 'linear_step_lower' in defaults:
                # 新方式：直接指定上下限，自动计算系数
                linear_step_upper = defaults.get('linear_step_upper', 30.0)  # 步长上限。
                linear_step_lower = defaults.get('linear_step_lower', 20.0)  # 步长下限。
                linear_coeff_a = linear_step_upper - linear_step_lower  # 自动计算：a = 上限 - 下限。
                linear_coeff_b = linear_step_lower  # b = 下限。
            else:
                # 旧方式：直接指定系数（向后兼容）
                linear_coeff_a = defaults.get('linear_coeff_a', 10.0)  # 线性函数系数 a。
                linear_coeff_b = defaults.get('linear_coeff_b', 20.0)  # 线性函数系数 b（步长下限）。
            time_indices = self._build_linear_schedule(
                start_t=self.num_timesteps - 1,  # 起始于最后一个时间步。
                end_t=lambda_floor,  # 结束于调度下限。
                coeff_a=linear_coeff_a,  # 线性函数系数 a。
                coeff_b=linear_coeff_b  # 线性函数系数 b（步长下限）。
            )
        else:  # 使用固定步长。
            # 与 lambda/linear 一致：若配置了 time_lower（通常即 time_boundary），大步只走到该 t，否则沿用 min_t
            floor_t = defaults.get('time_lower')
            if floor_t is not None:
                end_t = int(np.clip(int(floor_t), min_t, self.num_timesteps - 1))
            else:
                end_t = int(min_t)
            time_indices = list(range(self.num_timesteps - 1, end_t - 1, -step_stride))  # 按步长向前遍历时间。

        # 截断语义：见模块级 GRAD_FUSION_CAP_UNSPECIFIED 说明
        if max_grad_fusion_iterations is not GRAD_FUSION_CAP_UNSPECIFIED:
            _cap = max_grad_fusion_iterations
        elif max_gradient_steps is not GRAD_FUSION_CAP_UNSPECIFIED:
            _cap = max_gradient_steps
        else:
            _cap = defaults.get('max_grad_fusion_iterations', defaults.get('max_gradient_steps'))

        if preserve_schedule_endpoint is None:
            preserve_schedule_endpoint = bool(
                defaults.get('preserve_schedule_endpoint', False)
            )

        time_indices = self._truncate_schedule_to_grad_fusion_iterations(
            time_indices,
            _cap,
            anchor_t=grad_fusion_anchor_t,
            preserve_endpoint=bool(preserve_schedule_endpoint),
        )

        # 大步阶段通常不使用带噪声方法，保持标准方法以获得更快的探索
        use_with_noise = defaults.get('use_with_noise', False)  # 大步阶段默认不使用带噪声方法。
        use_adaptive_step = defaults.get('use_adaptive_step', False)  # 从配置读取自适应步长开关，默认关闭。
        use_time_scale = defaults.get('use_time_scale', True)  # 从配置读取时间缩放开关，默认启用。
        ligand_pos, log_ligand_v, pos_traj, log_v_traj = self._dynamic_diffusion(
            protein_pos=protein_pos,  # 蛋白坐标。
            protein_v=protein_v,  # 蛋白特征。
            batch_protein=batch_protein,  # 蛋白批次。
            ligand_pos=ligand_pos,  # 当前配体坐标。
            log_ligand_v=log_ligand_v,  # 当前配体 log 分布。
            batch_ligand=batch_ligand,  # 配体批次。
            time_indices=time_indices,  # 已按「梯度融合迭代次数」截断后的调度
            step_size=step_size,  # 每次迭代步幅。
            add_noise=add_noise,  # 附加噪声强度。
            pos_clip=pos_clip,  # 位置梯度裁剪阈值。
            v_clip=v_clip,  # 类别梯度裁剪阈值。
            record_traj=True,  # 记录轨迹以便分析。
            pos_only=pos_only,  # 是否仅更新位置。
            use_with_noise=use_with_noise,  # 是否使用带噪声方法（大步阶段通常为 False）。
            use_adaptive_step=use_adaptive_step,  # 是否启用自适应步长机制。
            use_time_scale=use_time_scale,  # 是否启用时间步对步长的乘算。
            repaint_cfg=repaint_cfg  # 骨架位置锁定等双遮罩配置（透传给 _dynamic_diffusion）。
        )

        ligand_pos = ligand_pos + offset[batch_ligand]  # 将坐标移回原参考系。
        if pos_traj:  # 若记录了轨迹。
            offset_cpu = offset[batch_ligand].detach().cpu()  # 将偏移转到 CPU。
            pos_traj = [p + offset_cpu for p in pos_traj]  # 为每个轨迹补回偏移。

        return {  # 返回采样结果。
            'pos': ligand_pos,  # 最终配体位置。
            'log_v': log_ligand_v,  # 最终 log 概率。
            'v': log_ligand_v.argmax(dim=-1),  # 最终类别索引。
            'pos_traj': pos_traj,  # 位置轨迹。
            'log_v_traj': log_v_traj,  # 类别轨迹。
            'time_indices': time_indices,  # 使用的时间调度。
            'batch_ligand': batch_ligand,  # 配体批次信息。
            'offset': offset  # 记录用于还原的偏移。
        }

    @torch.no_grad()
    def sample_diffusion_refinement(self, protein_pos, protein_v, batch_protein,
                                    init_ligand_pos, init_ligand_v, batch_ligand,
                                    center_pos_mode=None, pos_only=False,
                                    step_stride=None, step_size=None, add_noise=None,
                                    pos_clip=None, v_clip=None,
                                    time_upper=None, time_lower=0, num_cycles=1,
                                    log_ligand_input_mode='auto',
                                    max_grad_fusion_iterations=GRAD_FUSION_CAP_UNSPECIFIED,
                                    max_gradient_steps=GRAD_FUSION_CAP_UNSPECIFIED,
                                    grad_fusion_anchor_t=None,
                                    preserve_schedule_endpoint=None,
                                    repaint_cfg=None,
                                    explicit_time_indices=None,
                                    continuation=False):

        if center_pos_mode is None:  # 未指定中心化模式时使用默认值。
            center_pos_mode = self.center_pos_mode

        defaults = self.dynamic_refine_defaults or {}  # 读取精炼阶段默认参数。
        schedule_mode = defaults.get('schedule', 'lambda')  # 获取调度模式。
        use_lambda_schedule = schedule_mode == 'lambda'  # 是否使用 lambda 调度。
        use_linear_schedule = schedule_mode == 'linear'  # 是否使用线性调度。
        if not use_lambda_schedule and not use_linear_schedule:  # 固定步长调度。
            step_stride = int(step_stride if step_stride is not None else defaults.get('stride', 10))  # 设置步长。
            step_stride = max(step_stride, 1)  # 确保步长至少为 1。
        step_size = float(step_size if step_size is not None else defaults.get('step_size', 0.2))  # 设置步幅。
        add_noise = float(add_noise if add_noise is not None else defaults.get('noise_scale', 0.0))  # 设置噪声强度。
        pos_clip = pos_clip if pos_clip is not None else defaults.get('pos_clip', self.dynamic_pos_step_clip)  # 位置梯度裁剪。
        v_clip = v_clip if v_clip is not None else defaults.get('v_clip', self.dynamic_v_step_clip)  # 类别梯度裁剪。

        protein_pos, ligand_pos, offset = center_pos(
            protein_pos, init_ligand_pos, batch_protein, batch_ligand, mode=center_pos_mode)  # 按需中心化。

        log_mode = log_ligand_input_mode  # 决定 log 输入模式。
        if log_mode == 'auto':  # 自动选择模式。
            log_mode = self.ligand_v_input if self.ligand_v_input in ('log_prob', 'logits', 'prob') else 'auto'
        log_ligand_v = ensure_log_ligand(init_ligand_v, self.num_classes, mode=log_mode)  # 转换为 log 概率。

        if time_upper is None:  # 若未指定上界。
            time_upper = int(defaults.get('time_upper', min(self.num_timesteps - 1, 500)))  # 采用默认值或 500。
        time_upper = int(np.clip(time_upper, 0, self.num_timesteps - 1))  # 限制在合法范围。
        time_lower = int(np.clip(time_lower, 0, time_upper))  # 下界不超过上界。
        
        # 根据官方 GlintDM/DiffDynamic 实现，在精炼采样初始化时使用带噪声的预测
        use_with_noise = defaults.get('use_with_noise', True)  # 默认在精炼阶段使用带噪声方法。
        clean_type_context = bool(
            repaint_cfg is not None
            and repaint_cfg.get('_clean_type_context', False)
        )
        if use_with_noise and not pos_only and not clean_type_context and not continuation:
            # 使用带噪声的方法初始化 log_ligand_v，提供更平滑的起点
            init_t = torch.full(size=(batch_ligand.max().item() + 1,), fill_value=time_upper, 
                               dtype=torch.long, device=protein_pos.device)  # 构造初始时间步。
            log_ligand_v = self.q_v_pred_with_noise(log_ligand_v, init_t, batch_ligand)  # 使用带噪声的预测初始化。

        if explicit_time_indices is not None:
            time_indices = [int(t) for t in explicit_time_indices]
        elif use_lambda_schedule:  # 使用 lambda 调度。
            lambda_coeff_a = defaults.get('lambda_coeff_a', 40.0)  # 调度参数 a。
            lambda_coeff_b = defaults.get('lambda_coeff_b', 5.0)  # 调度参数 b。
            time_indices = self._build_lambda_schedule(
                start_t=time_upper,  # 从上界开始。
                end_t=time_lower,  # 到下界结束。
                coeff_a=lambda_coeff_a,  # 传入参数 a。
                coeff_b=lambda_coeff_b  # 传入参数 b。
            )
        elif use_linear_schedule:  # 使用线性调度。
            # 支持两种配置方式：直接指定上下限（推荐），或指定系数（向后兼容）
            if 'linear_step_upper' in defaults or 'linear_step_lower' in defaults:
                # 新方式：直接指定上下限，自动计算系数
                linear_step_upper = defaults.get('linear_step_upper', 15.0)  # 步长上限。
                linear_step_lower = defaults.get('linear_step_lower', 5.0)  # 步长下限。
                linear_coeff_a = linear_step_upper - linear_step_lower  # 自动计算：a = 上限 - 下限。
                linear_coeff_b = linear_step_lower  # b = 下限。
            else:
                # 旧方式：直接指定系数（向后兼容）
                linear_coeff_a = defaults.get('linear_coeff_a', 10.0)  # 线性函数系数 a。
                linear_coeff_b = defaults.get('linear_coeff_b', 5.0)  # 线性函数系数 b（步长下限）。
            time_indices = self._build_linear_schedule(
                start_t=time_upper,  # 从上界开始。
                end_t=time_lower,  # 到下界结束。
                coeff_a=linear_coeff_a,  # 线性函数系数 a。
                coeff_b=linear_coeff_b  # 线性函数系数 b（步长下限）。
            )
        else:  # 固定步长调度。
            time_indices = list(range(time_upper, time_lower - 1, -step_stride))  # 生成等间隔时间序列。

        if max_grad_fusion_iterations is not GRAD_FUSION_CAP_UNSPECIFIED:
            _rcap = max_grad_fusion_iterations
        elif max_gradient_steps is not GRAD_FUSION_CAP_UNSPECIFIED:
            _rcap = max_gradient_steps
        else:
            _rcap = defaults.get('max_grad_fusion_iterations', defaults.get('max_gradient_steps'))
        if preserve_schedule_endpoint is None:
            preserve_schedule_endpoint = bool(
                defaults.get('preserve_schedule_endpoint', False)
            )
        time_indices = self._truncate_schedule_to_grad_fusion_iterations(
            time_indices,
            _rcap,
            anchor_t=grad_fusion_anchor_t,
            preserve_endpoint=bool(preserve_schedule_endpoint),
        )
        if explicit_time_indices is not None:
            # Prudent rounds consume disjoint slices of one original schedule.
            # Never rebuild a new lambda schedule from the preceding boundary.
            time_indices = [int(t) for t in explicit_time_indices]
            if not time_indices or any(
                t < 0 or t >= self.num_timesteps for t in time_indices
            ):
                raise ValueError('Invalid explicit refinement time indices')
            if any(a <= b for a, b in zip(time_indices, time_indices[1:])):
                raise ValueError('Refinement time indices must strictly decrease')
            if int(num_cycles) != 1:
                raise ValueError('Explicit refinement schedules require cycles=1')
            if _rcap is not None and len(time_indices) > int(_rcap):
                raise ValueError('Explicit schedule exceeds refinement step cap')

        pos_traj_total, log_v_traj_total = [], []  # 初始化轨迹列表。
        ligand_pos_current, log_ligand_v_current = ligand_pos, log_ligand_v  # 初始化当前状态。
        use_adaptive_step = defaults.get('use_adaptive_step', False)  # 从配置读取自适应步长开关，默认关闭。
        use_time_scale = defaults.get('use_time_scale', True)  # 从配置读取时间缩放开关，默认启用。
        for _ in range(max(int(num_cycles), 1)):  # 重复执行若干周期。
            ligand_pos_current, log_ligand_v_current, pos_traj, log_v_traj = self._dynamic_diffusion(
                protein_pos=protein_pos,  # 蛋白坐标。
                protein_v=protein_v,  # 蛋白特征。
                batch_protein=batch_protein,  # 蛋白批次。
                ligand_pos=ligand_pos_current,  # 当前配体坐标。
                log_ligand_v=log_ligand_v_current,  # 当前配体 log 概率。
                batch_ligand=batch_ligand,  # 配体批次。
                time_indices=time_indices,  # 调度时间序列。
                step_size=step_size,  # 步幅。
                add_noise=add_noise,  # 噪声强度。
                pos_clip=pos_clip,  # 位置梯度裁剪阈值。
                v_clip=v_clip,  # 类别梯度裁剪阈值。
                record_traj=True,  # 记录轨迹。
                pos_only=pos_only,  # 是否仅更新位置。
                use_with_noise=use_with_noise,  # 是否使用带噪声方法。
                use_adaptive_step=use_adaptive_step,  # 是否启用自适应步长机制（从配置读取）。
                use_time_scale=use_time_scale,  # 是否启用时间步对步长的乘算（从配置读取）。
                repaint_cfg=repaint_cfg  # 骨架位置锁定等双遮罩配置（透传给 _dynamic_diffusion）。
            )
            pos_traj_total.extend(pos_traj)  # 累积位置轨迹。
            log_v_traj_total.extend(log_v_traj)  # 累积类别轨迹。

        ligand_pos_final = ligand_pos_current + offset[batch_ligand]  # 将最终坐标还原到原参考系。
        if pos_traj_total:  # 若存在轨迹。
            offset_cpu = offset[batch_ligand].detach().cpu()  # 偏移转到 CPU。
            pos_traj_total = [p + offset_cpu for p in pos_traj_total]  # 为轨迹还原偏移。

        return {  # 返回精炼结果。
            'pos': ligand_pos_final,  # 最终配体位置。
            'log_v': log_ligand_v_current,  # 最终 log 概率。
            'v': log_ligand_v_current.argmax(dim=-1),  # 最终类别索引。
            'pos_traj': pos_traj_total,  # 位置轨迹。
            'log_v_traj': log_v_traj_total,  # 类别轨迹。
            'time_indices': time_indices,  # 使用的时间调度。
            'batch_ligand': batch_ligand,  # 配体批次。
            'offset': offset  # 记录中心化偏移。
        }

    @torch.no_grad()
    def sample_diffusion(self, protein_pos, protein_v, batch_protein,
                         init_ligand_pos, init_ligand_v, batch_ligand,
                         num_steps=None, center_pos_mode=None, pos_only=False, start_t=None):

        if num_steps is None:  # 若未指定步数。
            num_steps = self.num_timesteps  # 默认使用全部时间步。
        num_graphs = batch_protein.max().item() + 1  # 统计图数量。

        protein_pos, init_ligand_pos, offset = center_pos(
            protein_pos, init_ligand_pos, batch_protein, batch_ligand, mode=center_pos_mode)  # 中心化坐标。

        pos_traj, v_traj = [], []  # 初始化位置与类别轨迹。
        v0_pred_traj, vt_pred_traj = [], []  # 初始化预测概率轨迹。
        ligand_pos, ligand_v = init_ligand_pos, init_ligand_v  # 初始化配体状态。
        # time sequence  # 保留注释：构造时间序列。
        if start_t is not None:
            # 从指定 start_t 反扩散到 t=0（用于 refinement：训练时最后几步学习修复键角/位置）
            start_t = int(np.clip(start_t, 0, self.num_timesteps - 1))
            time_seq = list(range(start_t, -1, -1))  # [start_t, start_t-1, ..., 0]
        else:
            time_seq = list(reversed(range(self.num_timesteps - num_steps, self.num_timesteps)))  # 默认：高 t 端
        for i in tqdm(time_seq, desc='sampling', total=len(time_seq)):  # 遍历时间步，显示进度条。
            t = torch.full(size=(num_graphs,), fill_value=i, dtype=torch.long, device=protein_pos.device)  # 构造当前时间张量。
            preds = self(
                protein_pos=protein_pos,  # 蛋白坐标。
                protein_v=protein_v,  # 蛋白特征。
                batch_protein=batch_protein,  # 蛋白批次索引。

                init_ligand_pos=ligand_pos,  # 当前配体坐标。
                init_ligand_v=ligand_v,  # 当前配体类别表示。
                batch_ligand=batch_ligand,  # 配体批次索引。
                time_step=t  # 当前时间步。
            )
            # Compute posterior mean and variance  # 保留注释：计算后验均值与方差。
            if self.model_mean_type == 'noise':  # 若预测噪声。
                pred_pos_noise = preds['pred_ligand_pos'] - ligand_pos  # 获取噪声预测。
                pos0_from_e = self._predict_x0_from_eps(xt=ligand_pos, eps=pred_pos_noise, t=t, batch=batch_ligand)  # 反推 x0。
                v0_from_e = preds['pred_ligand_v']  # 保存类型预测 logits。
            elif self.model_mean_type == 'C0':  # 若预测 x0。
                pos0_from_e = preds['pred_ligand_pos']  # 直接使用位置预测。
                v0_from_e = preds['pred_ligand_v']  # 保存类型预测 logits。
            else:  # 未实现类型。
                raise ValueError  # 抛出异常。

            pos_model_mean = self.q_pos_posterior(x0=pos0_from_e, xt=ligand_pos, t=t, batch=batch_ligand)  # 后验均值。
            pos_log_variance = extract(self.posterior_logvar, t, batch_ligand)  # 对数方差。
            # no noise when t == 0  # 保留注释：t=0 不加噪声。
            nonzero_mask = (1 - (t == 0).float())[batch_ligand].unsqueeze(-1)  # 构造掩码。
            ligand_pos_next = pos_model_mean + nonzero_mask * (0.5 * pos_log_variance).exp() * torch.randn_like(
                ligand_pos)  # 按后验均值与方差采样下一个位置。
            ligand_pos = ligand_pos_next  # 更新位置。

            if not pos_only:  # 若需要更新类别。
                log_ligand_v_recon = F.log_softmax(v0_from_e, dim=-1)  # 计算 log 概率。
                # 支持 init_ligand_v 为 log 概率（如 TargetDiff refine 传入 log_vt）或整数索引
                if torch.is_floating_point(ligand_v):
                    log_ligand_v = ligand_v  # 已是 log 概率，直接使用
                else:
                    log_ligand_v = index_to_log_onehot(ligand_v, self.num_classes)  # 当前类别转换为 log one-hot。
                log_model_prob = self.q_v_posterior(log_ligand_v_recon, log_ligand_v, t, batch_ligand)  # 计算后验。
                ligand_v_next = log_sample_categorical(log_model_prob)  # 采样下一步类别。

                v0_pred_traj.append(log_ligand_v_recon.clone().cpu())  # 记录预测 logits。
                vt_pred_traj.append(log_model_prob.clone().cpu())  # 记录后验概率。
                ligand_v = ligand_v_next  # 更新类别。

            ori_ligand_pos = ligand_pos + offset[batch_ligand]  # 将位置还原到原参考系。
            pos_traj.append(ori_ligand_pos.clone().cpu())  # 保存位置轨迹。
            v_save = ligand_v.argmax(dim=-1) if torch.is_floating_point(ligand_v) else ligand_v
            v_traj.append(v_save.clone().cpu())  # 保存类别轨迹（统一为索引）

        ligand_pos = ligand_pos + offset[batch_ligand]  # 最终位置还原偏移。
        return {  # 返回采样结果。
            'pos': ligand_pos,  # 最终配体位置。
            'v': ligand_v,  # 最终类别标签。
            'pos_traj': pos_traj,  # 位置轨迹。
            'v_traj': v_traj,  # 类别轨迹。
            'v0_traj': v0_pred_traj,  # 每步预测 logits 轨迹。
            'vt_traj': vt_pred_traj  # 每步后验轨迹。
        }
