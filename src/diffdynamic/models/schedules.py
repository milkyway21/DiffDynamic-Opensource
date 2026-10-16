"""Schedules components of the DiffDynamic model."""

from __future__ import annotations

import numpy as np


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):  # 根据策略生成 beta 调度序列。
    def sigmoid(x):  # 定义内部 sigmoid 函数。
        return 1 / (np.exp(-x) + 1)  # 返回标准 sigmoid 输出。

    if beta_schedule == "quad":  # 二次调度：beta 取平方递增。
        betas = (  # 计算 beta 序列。
                np.linspace(
                    beta_start ** 0.5,  # 起始值开方。
                    beta_end ** 0.5,  # 结束值开方。
                    num_diffusion_timesteps,  # 采样步数。
                    dtype=np.float64,  # 使用双精度浮点。
                )
                ** 2  # 再平方以得到二次分布。
        )
    elif beta_schedule == "linear":  # 线性调度。
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64  # 等间距生成序列。
        )
    elif beta_schedule == "const":  # 常数调度。
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)  # 使用固定值填充。
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1  # Jensen-Shannon 分布调度。
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64  # 生成倒数序列。
        )
    elif beta_schedule == "sigmoid":  # Sigmoid 调度。
        betas = np.linspace(-6, 6, num_diffusion_timesteps)  # 先在区间内均匀采样。
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start  # 通过 sigmoid 缩放到目标范围。
    else:  # 不支持的调度类型。
        raise NotImplementedError(beta_schedule)  # 抛出未实现错误。
    assert betas.shape == (num_diffusion_timesteps,)  # 断言序列长度正确。
    return betas  # 返回 beta 序列。


def cosine_beta_schedule(timesteps, s=0.008):  # 生成余弦调度的 alpha 序列。
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1  # 计算步数边界。
    x = np.linspace(0, steps, steps)  # 在步数区间均匀采样。
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2  # 根据论文公式计算累计 alpha。
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]  # 归一化使得起始项为 1。
    alphas = (alphas_cumprod[1:] / alphas_cumprod[:-1])  # 通过相邻比值得到每步 alpha。

    alphas = np.clip(alphas, a_min=0.001, a_max=1.)  # 限制范围避免过小或超过 1。

    # Use sqrt of this, so the alpha in our paper is the alpha_sqrt from the
    # Gaussian diffusion in Ho et al.  # 保留原注释：说明取平方根的原因。
    alphas = np.sqrt(alphas)  # 对 alpha 取平方根与参考实现保持一致。
    return alphas  # 返回余弦调度结果。


def adaptive_step_size(current_error, base_step_size, min_step=0.1, max_step=2.0):  # 根据当前误差自适应调整步长。
    """
    根据当前误差自适应调整步长。
    
    Args:
        current_error: 当前误差值（可以是梯度范数、位置误差等）。
        base_step_size: 基础步长。
        min_step: 最小步长限制。
        max_step: 最大步长限制。
    
    Returns:
        float: 调整后的步长。
    """
    if current_error is None or current_error <= 0:  # 若误差无效或为 0。
        return base_step_size  # 返回基础步长。
    # 误差小则增大步长，误差大则减小步长。
    factor = min(max(0.5 / (current_error + 1e-8), 0.5), 2.0)  # 限制因子在 [0.5, 2.0] 范围内。
    adjusted_step = base_step_size * factor  # 计算调整后的步长。
    return min(max(adjusted_step, min_step), max_step)  # 限制在 [min_step, max_step] 范围内。


class ScheduleMixin:
    """Schedules responsibilities; registered parameter names stay unchanged."""

    def _compute_grad_lambda(self, timestep_value, mode=None, grad_norm_ratio=None):  # 根据时间步计算梯度融合系数，支持多种融合策略。
        if not self.use_grad_fusion:  # 若未开启梯度融合。
            return 0.0  # 返回零表示不融合。
        cfg = self.grad_fusion_lambda  # 读取融合配置。
        
        # 常数模式
        if isinstance(cfg, (float, int)):  # 常数模式。
            return float(cfg)  # 直接返回常数。
        
        # 字符串模式表示预设调度
        if isinstance(cfg, str):  # 字符串模式表示预设调度。
            alpha = float(timestep_value) / max(float(self.num_timesteps - 1), 1.0)  # 归一化时间步。
            if cfg == 'linear':  # 线性递减。
                return max(min(1.0 - alpha, 1.0), 0.0)  # 限制在 [0,1]。
            if cfg == 'time':  # 随时间递增。
                return max(min(alpha, 1.0), 0.0)  # 限制在 [0,1]。
            if cfg == 'auto':  # 自动模式。
                return 0.5 * (1.0 - alpha)  # 线性插值。
            return 0.5  # 默认返回 0.5。
        
        # 字典模式支持自定义调度
        if isinstance(cfg, dict):  # 字典模式支持自定义调度。
            strategy_mode = mode if mode is not None else cfg.get('mode', 'linear')  # 获取融合策略模式。
            start = float(cfg.get('start', 0.8))  # 起始值，默认为 0.8。
            end = float(cfg.get('end', 0.2))  # 结束值，默认为 0.2。
            ratio = float(timestep_value) / max(float(self.num_timesteps - 1), 1.0)  # 归一化时间步。
            
            if strategy_mode == 'linear':  # 线性衰减策略。
                lambda_val = start * ratio + end * (1.0 - ratio)  # 线性插值。
                return max(min(lambda_val, 1.0), 0.0)  # 限制在 [0,1]。
            
            elif strategy_mode == 'exponential':  # 指数衰减策略。
                if start > 0 and end > 0:  # 确保起始和结束值都大于 0。
                    decay_rate = (end / start) ** (1.0 - ratio)  # 计算衰减率。
                    lambda_val = start * decay_rate  # 指数衰减。
                else:  # 若起始或结束值为 0，回退到线性衰减。
                    lambda_val = start * ratio + end * (1.0 - ratio)  # 线性插值。
                return max(min(lambda_val, 1.0), 0.0)  # 限制在 [0,1]。
            
            elif strategy_mode == 'adaptive':  # 基于梯度大小的自适应策略。
                base_lambda = start * ratio + end * (1.0 - ratio)  # 基础线性插值。
                if grad_norm_ratio is not None and grad_norm_ratio > 0:  # 若提供了梯度范数比率。
                    # 当全局梯度范数较大时，增加其权重；否则减少其权重。
                    adaptive_factor = min(max(1.0 / (grad_norm_ratio + 1e-8), 0.5), 2.0)  # 自适应因子限制在 [0.5, 2.0]。
                    lambda_val = base_lambda * adaptive_factor  # 应用自适应因子。
                else:  # 若未提供梯度信息，回退到线性策略。
                    lambda_val = base_lambda  # 使用基础线性插值。
                return max(min(lambda_val, 1.0), 0.0)  # 限制在 [0,1]。
            
            elif strategy_mode == 'time':  # 线性升权。
                lambda_val = start + (end - start) * ratio  # 插值计算。
                return max(min(lambda_val, 1.0), 0.0)  # 限制在 [0,1]。
            
            elif strategy_mode == 'quadratic':  # 二次衰减策略（-x^power 模式）。
                # lambda = start - (start - end) * ratio^power
                # 前期（ratio接近0）变化慢，主要依赖全局梯度
                # 后期（ratio接近1）变化快，快速过渡到自由发展
                power = float(cfg.get('power', 2.0))  # 衰减指数，默认 2.0（二次）
                lambda_val = start - (start - end) * (ratio ** power)  # 幂次衰减。
                return max(min(lambda_val, 1.0), 0.0)  # 限制在 [0,1]。
            
            else:  # 未知策略模式，使用默认值。
                return 0.5  # 默认返回 0.5。
        
        return 0.5  # 默认返回 0.5。

    @staticmethod
    def _truncate_schedule_to_grad_fusion_iterations(
        time_indices,
        cap,
        anchor_t=None,
        preserve_endpoint=False,
    ):
        """截断调度列表长度。

        `time_indices` 的每一项对应 `_dynamic_diffusion` 里 for 循环的一次迭代，即一次网络前向 +
        （若 `use_grad_fusion`）一次 λ 梯度融合更新。此处的 cap 表示最多保留多少次这样的迭代，
        与扩散步数 num_timesteps（如 1000）或某个离散 t 的数值无关。

        - ``preserve_endpoint=True``：在选定起点到原调度终点之间均匀抽取最多 ``cap`` 项，确保精炼抵达低噪声终点。
        - ``anchor_t is None``：保留完整调度的前 ``cap`` 项（``time_indices[:cap]``），即从高 t 端起算。
        - ``anchor_t`` 为整数：先在**完整**调度中找与 ``anchor_t`` 最接近的离散 t 所在下标 ``i0``（并列取更小下标），
          再取 ``time_indices[i0 : i0 + cap]``，即沿用原规划的跳步间隔，从锚点附近起连续走 ``cap`` 步。
        """
        out = list(time_indices)
        if cap is None:
            return out
        try:
            n = int(cap)
        except (TypeError, ValueError):
            return out
        if n < 1:
            return out
        if preserve_endpoint:
            start_i = 0
            if anchor_t is not None:
                try:
                    at = int(anchor_t)
                    start_i = min(
                        range(len(out)), key=lambda i: (abs(int(out[i]) - at), i)
                    )
                except (TypeError, ValueError):
                    start_i = 0
            window = out[start_i:]
            if len(window) <= n:
                return window
            if n == 1:
                return [window[-1]]
            indices = np.linspace(0, len(window) - 1, n, dtype=int)
            return [window[int(i)] for i in indices]
        if anchor_t is None:
            return out[:n] if len(out) > n else out
        try:
            at = int(anchor_t)
        except (TypeError, ValueError):
            return out[:n] if len(out) > n else out
        best_i = 0
        best_d = None
        for i, t in enumerate(out):
            try:
                tv = int(t)
            except (TypeError, ValueError):
                continue
            d = abs(tv - at)
            if best_d is None or d < best_d or (d == best_d and i < best_i):
                best_d = d
                best_i = i
        if best_d is None:
            return out[:n] if len(out) > n else out
        return out[best_i : best_i + n]

    def _build_lambda_schedule(self, start_t, end_t, coeff_a, coeff_b):  # 构建自适应时间调度列表。
        """构建基于 lambda 函数的时间调度序列。
        
        Args:
            start_t: 起始时间步
            end_t: 结束时间步
            coeff_a: lambda 调度参数 a
            coeff_b: lambda 调度参数 b
        
        Returns:
            list: 时间步索引列表
        """
        start_t = int(np.clip(start_t, 0, self.num_timesteps - 1))  # 将起点裁剪到合法范围。
        end_t = int(np.clip(end_t, 0, self.num_timesteps - 1))  # 将终点裁剪到合法范围。
        if start_t == end_t:  # 若起终点一致。
            return [start_t]  # 直接返回单个时间步。

        decreasing = start_t > end_t  # 判断是否递减。
        step_sign = -1 if decreasing else 1  # 根据方向决定步符号。
        t = start_t  # 初始化当前时间。
        indices = []  # 初始化时间列表。

        while True:  # 迭代构建调度。
            indices.append(t)  # 记录当前时间。
            if t == end_t:  # 达到终点时结束。
                break
            lambda_t = float(t) / float(self.num_timesteps)  # 归一化时间。
            lambda_t = max(min(lambda_t, 1.0), 0.0)  # 限制在 [0,1]。
            n = coeff_a * lambda_t + coeff_b  # 根据参数计算步长。
            step = max(1, int(round(n)))  # 至少前进 1 步。
            t_next = t + step_sign * step  # 计算下一时间。
            if decreasing and t_next < end_t:  # 避免越界。
                t_next = end_t
            if not decreasing and t_next > end_t:  # 避免越界。
                t_next = end_t
            if t_next == t:  # 若步长为零，强制前进一格。
                t_next = t + step_sign
            t = int(np.clip(t_next, 0, self.num_timesteps - 1))  # 更新当前时间并裁剪。

        return indices  # 返回构建好的时间序列。

    def _build_linear_schedule(self, start_t, end_t, coeff_a, coeff_b):  # 构建基于线性函数的时间调度列表。
        """构建基于线性函数 f = ax + b 的时间调度序列。
        
        使用线性函数计算跳步步长：step_size = a * progress + b
        其中 progress 是进度（从1到0），b 是步长下限，a+b 是步长上限。
        第一步的步长 = a + b（上限），最后一步的步长 = b（下限），中间线性过渡。
        
        Args:
            start_t: 起始时间步
            end_t: 结束时间步
            coeff_a: 线性函数系数 a（步长上限 - 步长下限）
            coeff_b: 线性函数系数 b（步长下限）
        
        Returns:
            list: 时间步索引列表
        """
        start_t = int(np.clip(start_t, 0, self.num_timesteps - 1))  # 将起点裁剪到合法范围。
        end_t = int(np.clip(end_t, 0, self.num_timesteps - 1))  # 将终点裁剪到合法范围。
        if start_t == end_t:  # 若起终点一致。
            return [start_t]  # 直接返回单个时间步。

        decreasing = start_t > end_t  # 判断是否递减。
        step_sign = -1 if decreasing else 1  # 根据方向决定步符号。
        t = start_t  # 初始化当前时间。
        indices = []  # 初始化时间列表。
        initial_range = abs(start_t - end_t)  # 初始时间范围

        while True:  # 迭代构建调度。
            indices.append(t)  # 记录当前时间。
            if t == end_t:  # 达到终点时结束。
                break
            
            # 计算进度：剩余时间范围 / 初始时间范围
            # 第一步：progress = 1.0（步长最大）
            # 最后一步：progress = 0.0（步长最小）
            remaining_range = abs(t - end_t)
            if initial_range > 0:
                progress = float(remaining_range) / float(initial_range)
            else:
                progress = 0.0
            progress = max(0.0, min(1.0, progress))  # 限制在 [0,1]
            
            # 使用线性函数 f = a * x + b 计算步长
            # 第一步（progress=1.0）：step_size = a * 1 + b = a + b（上限）
            # 最后一步（progress=0.0）：step_size = a * 0 + b = b（下限）
            n = coeff_a * progress + coeff_b  # 根据线性函数计算步长。
            step = max(1, int(round(n)))  # 至少前进 1 步。
            t_next = t + step_sign * step  # 计算下一时间。
            if decreasing and t_next < end_t:  # 避免越界。
                t_next = end_t
            if not decreasing and t_next > end_t:  # 避免越界。
                t_next = end_t
            if t_next == t:  # 若步长为零，强制前进一格。
                t_next = t + step_sign
            t = int(np.clip(t_next, 0, self.num_timesteps - 1))  # 更新当前时间并裁剪。

        return indices  # 返回构建好的时间序列。
