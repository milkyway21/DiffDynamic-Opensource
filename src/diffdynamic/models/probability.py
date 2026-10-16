"""Probability components of the DiffDynamic model."""

from __future__ import annotations

from torch_scatter import scatter_mean
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_distance(pos, edge_index):  # 根据边索引计算节点间距离。
    return (pos[edge_index[0]] - pos[edge_index[1]]).norm(dim=-1)  # 取差向量并计算 L2 范数。


def to_torch_const(x):  # 将 NumPy 数组转换为常量形式的可注册参数。
    x = torch.from_numpy(x).float()  # 转换为浮点张量。
    x = nn.Parameter(x, requires_grad=False)  # 包装为不可训练参数便于保存到模块。
    return x  # 返回常量参数。


def center_pos(protein_pos, ligand_pos, batch_protein, batch_ligand, mode='protein'):  # 根据模式对坐标进行中心化。
    if mode == 'none':  # 不进行中心化。
        offset = 0.  # 偏移设置为 0。
        pass  # 保留位置，确保结构一致。
    elif mode == 'protein':  # 以蛋白几何中心进行平移。
        offset = scatter_mean(protein_pos, batch_protein, dim=0)  # 按批次计算蛋白坐标的均值。
        protein_pos = protein_pos - offset[batch_protein]  # 蛋白坐标减去偏移。
        ligand_pos = ligand_pos - offset[batch_ligand]  # 配体坐标同步减去偏移。
    else:  # 不支持的模式。
        raise NotImplementedError  # 抛出异常提示。
    return protein_pos, ligand_pos, offset  # 返回中心化后的坐标与偏移量。


def index_to_log_onehot(x, num_classes):  # 将分类索引转换为 log one-hot 表示。
    assert x.max().item() < num_classes, f'Error: {x.max().item()} >= {num_classes}'  # 确保索引未越界。
    x_onehot = F.one_hot(x, num_classes)  # 生成 one-hot 向量。
    # permute_order = (0, -1) + tuple(range(1, len(x.size())))  # 保留原注释：可选维度重排。
    # x_onehot = x_onehot.permute(permute_order)  # 保留原注释：示例重排操作。
    log_x = torch.log(x_onehot.float().clamp(min=1e-30))  # 对 one-hot 取 log，避免对数零。
    return log_x  # 返回 log one-hot 表示。


def log_onehot_to_index(log_x):  # 将 log one-hot 表示转换回类别索引。
    return log_x.argmax(1)  # 沿类别维度取最大值所在索引。


def categorical_kl(log_prob1, log_prob2):  # 计算两个分类分布的 KL 散度。
    kl = (log_prob1.exp() * (log_prob1 - log_prob2)).sum(dim=1)  # 使用 log 概率公式计算 KL。
    return kl  # 返回 KL 值。


def log_categorical(log_x_start, log_prob):  # 计算分类对数似然。
    return (log_x_start.exp() * log_prob).sum(dim=1)  # 对应 one-hot 权重乘 log 概率累加。


def normal_kl(mean1, logvar1, mean2, logvar2):  # 计算两个正态分布之间的 KL 散度。
    """
    KL divergence between normal distributions parameterized by mean and log-variance.
    """
    kl = 0.5 * (-1.0 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2) + (mean1 - mean2) ** 2 * torch.exp(-logvar2))  # 按闭式公式计算 KL。
    return kl.sum(-1)  # 对最后一维求和得到标量。


def log_normal(values, means, log_scales):  # 计算正态分布的对数概率密度。
    var = torch.exp(log_scales * 2)  # 根据 log σ 还原方差。
    log_prob = -((values - means) ** 2) / (2 * var) - log_scales - np.log(np.sqrt(2 * np.pi))  # 根据正态分布公式计算 logpdf。
    return log_prob.sum(-1)  # 对最后一维求和得到标量。


def log_sample_categorical(logits):  # 在 logit 空间对分类分布进行采样。
    uniform = torch.rand_like(logits)  # 采样均匀噪声以构造 Gumbel 噪声。
    gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)  # 根据公式生成 Gumbel 噪声。
    sample_index = (gumbel_noise + logits).argmax(dim=-1)  # 将噪声加到 logit 后取最大值。
    # sample_onehot = F.one_hot(sample, self.num_classes)  # 保留注释：生成 one-hot 示例。
    # log_sample = index_to_log_onehot(sample, self.num_classes)  # 保留注释：转换为 log one-hot。
    return sample_index  # 返回采样结果索引。


def log_1_min_a(a):  # 计算 log(1 - exp(a))，避免数值不稳定。
    return np.log(1 - np.exp(a) + 1e-40)  # 添加微小常数避免对数负数。


def log_add_exp(a, b):  # 计算 log(exp(a) + exp(b))，使用稳定公式。
    maximum = torch.max(a, b)  # 取较大值作为参考，避免指数溢出。
    return maximum + torch.log(torch.exp(a - maximum) + torch.exp(b - maximum))  # 返回稳定求和值。


def ensure_log_ligand(value, num_classes, mode='auto'):  # 根据指定模式将输入转换为 log 概率。
    if torch.is_floating_point(value):  # 若输入为浮点张量。
        if mode == 'log_prob':  # 已经是 log 概率。
            return value  # 直接返回。
        if mode == 'onehot':  # 期望 one-hot 索引，但收到浮点：视为 log one-hot（即 log 概率），直接返回。
            return value  # log of one-hot 即为 log 概率，与 log_prob 等价。
        if mode == 'logits':  # 输入为未归一化 logits。
            return F.log_softmax(value, dim=-1)  # 转换为 log 概率。
        if mode == 'prob':  # 输入为概率分布。
            return torch.log(value.clamp(min=1e-30))  # 对概率取对数。
        if mode == 'auto':  # 自动模式尝试推断类型。
            if value.dim() >= 2 and torch.allclose(value.sum(dim=-1), torch.ones_like(value.sum(dim=-1)), atol=1e-4):  # 若各行近似归一化。
                return torch.log(value.clamp(min=1e-30))  # 视为概率。
            return F.log_softmax(value, dim=-1)  # 否则视为 logits。
        raise ValueError(f'Unsupported ligand_v_input mode: {mode}')  # 未知模式时抛出异常。
    one_hot = F.one_hot(value.long(), num_classes=num_classes).float()  # 将离散索引转换为 one-hot。
    return torch.log(one_hot.clamp(min=1e-30))  # 返回 log one-hot。


def clamp_by_norm(tensor, max_norm, clip_strategy='norm'):  # 按指定策略裁剪张量，支持多种裁剪方法。
    if max_norm is None or max_norm <= 0:  # 无需裁剪时直接返回。
        return tensor
    
    if clip_strategy == 'norm':  # 基于范数的裁剪（原有方法）。
        norm = tensor.norm(dim=-1, keepdim=True)  # 计算最后一维的范数。
        scale = torch.ones_like(norm)  # 初始化缩放因子。
        mask = norm > max_norm  # 找出需要裁剪的位置。
        scale[mask] = max_norm / (norm[mask] + 1e-12)  # 计算缩放系数。
        return tensor * scale  # 返回裁剪后的张量。
    
    elif clip_strategy == 'value':  # 基于值的裁剪（逐元素裁剪）。
        return torch.clamp(tensor, min=-max_norm, max=max_norm)  # 将值限制在 [-max_norm, max_norm] 范围内。
    
    elif clip_strategy == 'adaptive':  # 自适应裁剪（结合范数和值）。
        # 先进行范数裁剪
        norm = tensor.norm(dim=-1, keepdim=True)  # 计算最后一维的范数。
        scale = torch.ones_like(norm)  # 初始化缩放因子。
        mask = norm > max_norm  # 找出需要裁剪的位置。
        scale[mask] = max_norm / (norm[mask] + 1e-12)  # 计算缩放系数。
        tensor_clipped = tensor * scale  # 范数裁剪后的张量。
        # 再进行值裁剪
        value_clip = max_norm * 0.5  # 值裁剪阈值设为范数阈值的一半。
        tensor_clipped = torch.clamp(tensor_clipped, min=-value_clip, max=value_clip)  # 值裁剪。
        return tensor_clipped  # 返回双重裁剪后的张量。
    
    else:  # 未知策略，回退到范数裁剪。
        norm = tensor.norm(dim=-1, keepdim=True)  # 计算最后一维的范数。
        scale = torch.ones_like(norm)  # 初始化缩放因子。
        mask = norm > max_norm  # 找出需要裁剪的位置。
        scale[mask] = max_norm / (norm[mask] + 1e-12)  # 计算缩放系数。
        return tensor * scale  # 返回裁剪后的张量。


def cal_kl_gradient(log_target, log_current):  # 计算将当前分布朝目标分布移动的梯度方向。
    target_prob = log_target.exp()  # 将目标 log 概率转换为概率。
    current_prob = log_current.exp()  # 将当前 log 概率转换为概率。
    grad = target_prob - current_prob  # 使用概率差作为梯度方向。
    return grad  # 返回梯度张量。


def extract(coef, t, batch):  # 根据时间步从常量表中取出并匹配批次。
    """
    DiffDynamic/TargetDiff 兼容实现：从常量表中提取对应时间步和批次的系数。
    
    Args:
        coef: 形状为 (num_timesteps,) 的一维张量，包含所有时间步的系数
        t: 可以是：
           - 形状为 (num_graphs,) 的张量，每个元素是对应图的时间步索引
           - 长度为 num_graphs 的列表，每个元素是对应图的时间步索引
        batch: 形状为 (num_atoms,) 的张量，每个元素是原子所属的图索引（在 [0, num_graphs-1] 范围内）
    
    Returns:
        形状为 (num_atoms, 1) 的张量，每个原子对应其所属图的时间步系数
    """
    # 确保 t 是张量格式
    if not isinstance(t, torch.Tensor):
        if isinstance(t, (list, tuple)):
            t = torch.tensor(t, dtype=torch.long, device=coef.device)
        else:
            # 标量情况：转换为单元素张量
            t = torch.tensor([t], dtype=torch.long, device=coef.device)
    
    # 确保 t 在正确的设备上
    if t.device != coef.device:
        t = t.to(coef.device)
    
    # 确保 batch 在正确的设备上
    if batch.device != coef.device:
        batch = batch.to(coef.device)
    
    # 验证输入有效性
    if batch.numel() == 0:
        raise ValueError(f"extract: batch is empty, cannot extract coefficients")
    
    if t.numel() == 0:
        raise ValueError(f"extract: t is empty, cannot extract coefficients")
    
    # 验证时间步索引范围
    if t.max().item() >= coef.shape[0] or t.min().item() < 0:
        raise IndexError(
            f"extract: time step indices out of range. t range=[{t.min().item()}, {t.max().item()}], "
            f"coef shape={coef.shape}, num_timesteps={coef.shape[0]}. "
            f"This may indicate that the sampling time step exceeds the model's training range."
        )
    
    # 计算批次数
    num_graphs = t.shape[0]
    
    # 验证 batch 索引范围
    if batch.max().item() >= num_graphs or batch.min().item() < 0:
        raise IndexError(
            f"extract: batch indices out of range. batch range=[{batch.min().item()}, {batch.max().item()}], "
            f"num_graphs={num_graphs}, batch.shape={batch.shape}, t.shape={t.shape}, "
            f"t range=[{t.min().item()}, {t.max().item()}]. "
            f"This indicates a mismatch between batch_ligand and the number of graphs."
        )
    
    # 执行索引：coef[t] 返回形状为 (num_graphs,) 的张量
    try:
        coef_t = coef[t]  # 形状: (num_graphs,)
    except IndexError as e:
        raise IndexError(
            f"extract: Failed to index coef with t. coef.shape={coef.shape}, "
            f"t={t}, t.shape={t.shape}, t range=[{t.min().item()}, {t.max().item()}]. "
            f"Original error: {e}"
        ) from e
    
    # 验证 coef_t 的形状
    if coef_t.shape[0] != num_graphs:
        raise RuntimeError(
            f"extract: coef[t] shape mismatch. Expected first dimension {num_graphs}, "
            f"got {coef_t.shape[0]}. coef.shape={coef.shape}, t={t}, t.shape={t.shape}"
        )
    
    # 使用 batch 索引：coef_t[batch] 返回形状为 (num_atoms,) 的张量
    try:
        result = coef_t[batch]  # 形状: (num_atoms,)
    except IndexError as e:
        raise IndexError(
            f"extract: Failed to index coef_t with batch. coef_t.shape={coef_t.shape}, "
            f"batch.shape={batch.shape}, batch range=[{batch.min().item()}, {batch.max().item()}], "
            f"num_graphs={num_graphs}. Original error: {e}"
        ) from e
    
    # 验证结果形状
    if result.shape[0] != batch.shape[0]:
        raise RuntimeError(
            f"extract: Result shape mismatch. Expected first dimension {batch.shape[0]}, "
            f"got {result.shape[0]}. coef_t.shape={coef_t.shape}, batch.shape={batch.shape}"
        )
    
    # 扩展最后一维以便广播：形状变为 (num_atoms, 1)
    return result.unsqueeze(-1)
