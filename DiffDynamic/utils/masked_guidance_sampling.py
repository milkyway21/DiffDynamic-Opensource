# -*- coding: utf-8 -*-
"""
掩码式逆向扩散（RePaint / DiffSBDD 思路）与分类器引导（DiffGUI 思路）工具。

与 DiffDynamic 中基于梯度更新的 _dynamic_diffusion 配合使用：
- RePaint：在每一步更新后，将「受保护原子」拉回前向扩散链上对应时刻 q(x_t | x_0) 的状态。
- 引导：对属性预测器可微损失求梯度，修正 combined_pos_grad / 类别更新方向。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Union

import torch
import torch.nn.functional as F


Tensor = torch.Tensor


def _ensure_mask_column(atom_mask: Tensor, n_atoms: int, device, dtype) -> Tensor:
    """将 [N] 或 [N,1] 掩码规范为 [N,1]，取值 [0,1]，1 表示固定/保护。"""
    if atom_mask.dim() == 1:
        m = atom_mask.view(-1, 1)
    else:
        m = atom_mask
    if m.shape[0] != n_atoms:
        raise ValueError(f'atom_mask 长度 {m.shape[0]} 与原子数 {n_atoms} 不一致')
    return m.to(device=device, dtype=dtype).clamp(0.0, 1.0)


def forward_noisy_positions(
    model,
    x0_pos: Tensor,
    batch_ligand: Tensor,
    t_step: int,
    fixed_eps: Optional[Tensor] = None,
) -> Tensor:
    """
    连续坐标前向扩散：x_t = sqrt(ᾱ_t) x_0 + sqrt(1-ᾱ_t) ε。

    若提供 fixed_eps（与 x0_pos 同形状），则 RePaint 各步使用同一 ε，轨迹可复现。
    """
    from models.molopt_score_model import extract  # 延迟导入，避免循环依赖

    num_graphs = int(batch_ligand.max().item()) + 1
    device = x0_pos.device
    t = torch.full((num_graphs,), int(t_step), dtype=torch.long, device=device)
    sqrt_alpha_bar = extract(model.sqrt_alphas_cumprod, t, batch_ligand)
    sqrt_one_minus = extract(model.sqrt_one_minus_alphas_cumprod, t, batch_ligand)
    if fixed_eps is None:
        eps = torch.randn_like(x0_pos)
    else:
        eps = fixed_eps
    return sqrt_alpha_bar * x0_pos + sqrt_one_minus * eps


def forward_noisy_atom_types(
    model,
    log_v0: Tensor,
    batch_ligand: Tensor,
    t_step: int,
    use_mean_prob: bool = True,
) -> Tensor:
    """
    离散原子类型：使用 q(v_t | v_0) 的闭式 log 概率（q_v_pred）。

    - use_mean_prob=True：返回 log_softmax(q_v_pred)，避免逐步随机采样带来的抖动（推荐用于掩码融合）。
    - use_mean_prob=False：与训练时一致做一次 categorical 采样（随机性更强）。
    """
    num_graphs = int(batch_ligand.max().item()) + 1
    device = log_v0.device
    t = torch.full((num_graphs,), int(t_step), dtype=torch.long, device=device)
    if use_mean_prob:
        log_q = model.q_v_pred(log_v0, t, batch_ligand)
        return F.log_softmax(log_q, dim=-1)
    _, log_sample = model.q_v_sample(log_v0, t, batch_ligand)
    return log_sample


def repaint_blend_states(
    ligand_pos: Tensor,
    log_ligand_v: Tensor,
    ref_pos_noisy: Tensor,
    ref_log_v_noisy: Tensor,
    atom_mask: Tensor,
) -> Tuple[Tensor, Tensor]:
    """
    RePaint 融合：对每个原子 i，
        x_i <- m_i * x_ref(i) + (1 - m_i) * x_gen(i)
    坐标按分量混合；类型在概率空间线性混合后再取 log（保持合法分布）。

    数学上：已知区域强制与「从 x_0 前向扩散到 t_next」一致，未知区域保留模型输出。
    """
    m = _ensure_mask_column(atom_mask, ligand_pos.shape[0], ligand_pos.device, ligand_pos.dtype)
    pos_out = (1.0 - m) * ligand_pos + m * ref_pos_noisy

    p_ref = torch.exp(ref_log_v_noisy)
    p_cur = torch.exp(log_ligand_v)
    m_v = m.expand_as(p_ref)
    p_mix = m_v * p_ref + (1.0 - m_v) * p_cur
    p_mix = p_mix / (p_mix.sum(dim=-1, keepdim=True) + 1e-12)
    log_v_out = torch.log(p_mix + 1e-30)
    return pos_out, log_v_out


@dataclass
class RepaintMaskState:
    """供 MaskedDiffusionSampler 保存参考分子与可选固定噪声。"""

    x0_pos: Tensor
    x0_log_v: Tensor
    atom_mask: Tensor
    fixed_eps_pos: Optional[Tensor] = None


class MaskedDiffusionSampler:
    """
    包装「参考分子 + 掩码」状态，在逆向调度 time_indices 上生成各步参考噪声态。

    典型用法：
        sampler = MaskedDiffusionSampler(model, x0_pos, x0_log_v, atom_mask)
        ref_pos, ref_logv = sampler.reference_at_t_next(batch_ligand, time_indices, step_idx)
        ligand_pos, log_ligand_v = sampler.blend_after_step(ligand_pos, log_ligand_v, ref_pos, ref_logv)
    """

    def __init__(
        self,
        model,
        x0_pos: Tensor,
        x0_log_v: Tensor,
        atom_mask: Tensor,
        fixed_eps_pos: Optional[Tensor] = None,
        use_mean_for_discrete: bool = True,
    ):
        self.model = model
        self.state = RepaintMaskState(
            x0_pos=x0_pos,
            x0_log_v=x0_log_v,
            atom_mask=atom_mask,
            fixed_eps_pos=fixed_eps_pos,
        )
        self.use_mean_for_discrete = use_mean_for_discrete

    def reference_at_t_next(
        self,
        batch_ligand: Tensor,
        time_indices: list,
        step_idx: int,
    ) -> Tuple[Tensor, Tensor]:
        """
        当前为逆向第 step_idx 步（刚完成从 time_indices[step_idx] 的去噪更新），
        应将已知子结构对齐到「下一更小时间」t_next 上的前向分布：

            t_next = time_indices[step_idx + 1] 若存在，否则 0（干净分子）。
        """
        if step_idx + 1 < len(time_indices):
            t_next = int(time_indices[step_idx + 1])
        else:
            t_next = 0
        st = self.state
        ref_pos = forward_noisy_positions(
            self.model, st.x0_pos, batch_ligand, t_next, fixed_eps=st.fixed_eps_pos
        )
        ref_logv = forward_noisy_atom_types(
            self.model,
            st.x0_log_v,
            batch_ligand,
            t_next,
            use_mean_prob=self.use_mean_for_discrete,
        )
        return ref_pos, ref_logv

    def blend_after_step(
        self,
        ligand_pos: Tensor,
        log_ligand_v: Tensor,
        ref_pos_noisy: Tensor,
        ref_log_v_noisy: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        return repaint_blend_states(
            ligand_pos,
            log_ligand_v,
            ref_pos_noisy,
            ref_log_v_noisy,
            self.state.atom_mask,
        )


def apply_classifier_guidance(
    ligand_pos: Tensor,
    log_ligand_v: Tensor,
    batch_ligand: Tensor,
    property_loss_fn: Callable[[Tensor, Tensor, Tensor], Tensor],
    scale: float = 1.0,
    atom_mask: Optional[Tensor] = None,
    apply_to_types: bool = True,
) -> Tuple[Tensor, Optional[Tensor]]:
    """
    分类器式引导：对可微损失 L 求 ∂L/∂x，修正更新方向。

    - 若希望**最大化** QED，可令 property_loss_fn 返回 ``-qed``（最小化负 QED）。
    - scale：引导强度；过大易破坏扩散动力学，建议从小（如 0.01~0.5）试起。
    - atom_mask：保护原子处梯度置 0（与掩码一致时不强行改固定子结构）。

    返回:
        grad_pos: 与 ligand_pos 同形的附加「梯度项」，应加到 combined_pos_grad 上（本实现为 -scale * ∂L/∂x）。
        grad_log_v: 若 apply_to_types 且 log_v 参与损失，则返回对 log 概率的修正项；否则 None。
    """
    # 外层常在 torch.no_grad() 采样中；此处单独开启梯度以计算 ∂L/∂x
    with torch.enable_grad():
        pos = ligand_pos.detach().clone().requires_grad_(True)
        logv = log_ligand_v.detach().clone().requires_grad_(True)
        loss = property_loss_fn(pos, logv, batch_ligand)
        if not torch.is_tensor(loss) or not loss.requires_grad:
            return torch.zeros_like(ligand_pos), None
        loss = loss.view(-1).sum()
        loss.backward()

        g_pos = -float(scale) * pos.grad
        g_logv = None
        if apply_to_types and logv.grad is not None:
            g_logv = -float(scale) * logv.grad

    if atom_mask is not None:
        m = _ensure_mask_column(atom_mask, g_pos.shape[0], g_pos.device, g_pos.dtype)
        g_pos = g_pos * (1.0 - m)
        if g_logv is not None:
            m = _ensure_mask_column(atom_mask, g_logv.shape[0], g_logv.device, g_logv.dtype)
            g_logv = g_logv * (1.0 - m.expand_as(g_logv))

    return g_pos, g_logv


def guidance_in_t_range(
    t_scalar: int,
    t_min: int,
    t_max: int,
) -> bool:
    """仅在 [t_min, t_max] 内启用引导（闭区间）。"""
    return int(t_min) <= int(t_scalar) <= int(t_max)


# =============================================================================
# 双掩码 RePaint（骨架约束优化专用）
# =============================================================================

def dual_mask_repaint_blend_states(
    ligand_pos: Tensor,
    log_ligand_v: Tensor,
    ref_pos_noisy: Tensor,
    ref_log_v_noisy: Tensor,
    pos_mask: Tensor,
    type_mask: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """
    双掩码 RePaint 融合：位置掩码与类型掩码独立控制。

    骨架约束优化场景：
    - pos_mask=1  → 该原子位置固定（拉回参考前向扩散态）
    - type_mask=1 → 该原子元素固定（拉回参考前向扩散态）
    - 骨架之外（mask=0）：位置和元素均可自由变化

    相比单一 atom_mask，此函数允许"骨架原子固定位置但可换元素"的场景：
        pos_mask = scaffold_atoms_indicator   # 骨架原子=1
        type_mask = zeros                      # 任何原子元素均自由

    数学上：
        x_pos_i <- pm_i * x_ref_pos(i) + (1-pm_i) * x_gen_pos(i)
        x_type_i <- tm_i * x_ref_type(i) + (1-tm_i) * x_gen_type(i)
    """
    pm = _ensure_mask_column(pos_mask, ligand_pos.shape[0], ligand_pos.device, ligand_pos.dtype)
    pos_out = (1.0 - pm) * ligand_pos + pm * ref_pos_noisy

    if type_mask is None:
        log_v_out = log_ligand_v
    else:
        tm = _ensure_mask_column(type_mask, log_ligand_v.shape[0], log_ligand_v.device, log_ligand_v.dtype)
        p_ref = torch.exp(ref_log_v_noisy)
        p_cur = torch.exp(log_ligand_v)
        tm_v = tm.expand_as(p_ref)
        p_mix = tm_v * p_ref + (1.0 - tm_v) * p_cur
        p_mix = p_mix / (p_mix.sum(dim=-1, keepdim=True) + 1e-12)
        log_v_out = torch.log(p_mix + 1e-30)

    return pos_out, log_v_out


@dataclass
class DualMaskRepaintState:
    """供 DualMaskedDiffusionSampler 保存参考分子、双掩码与可选固定噪声。"""

    x0_pos: Tensor
    x0_log_v: Tensor
    pos_mask: Tensor
    type_mask: Optional[Tensor]
    fixed_eps_pos: Optional[Tensor] = None


class DualMaskedDiffusionSampler:
    """
    双掩码扩散采样器：位置掩码与原子类型掩码分别控制。

    典型用法（骨架约束优化）：
        sampler = DualMaskedDiffusionSampler(
            model, x0_pos, x0_log_v,
            pos_mask=scaffold_mask,  # 骨架原子=1：位置被固定
            type_mask=None,          # 所有原子元素可自由变化
        )
        ref_pos, ref_logv = sampler.reference_at_t_next(batch_ligand, time_indices, step_idx)
        ligand_pos, log_v = sampler.blend_after_step(ligand_pos, log_v, ref_pos, ref_logv)
    """

    def __init__(
        self,
        model,
        x0_pos: Tensor,
        x0_log_v: Tensor,
        pos_mask: Tensor,
        type_mask: Optional[Tensor] = None,
        fixed_eps_pos: Optional[Tensor] = None,
        use_mean_for_discrete: bool = True,
    ):
        self.model = model
        self.state = DualMaskRepaintState(
            x0_pos=x0_pos,
            x0_log_v=x0_log_v,
            pos_mask=pos_mask,
            type_mask=type_mask,
            fixed_eps_pos=fixed_eps_pos,
        )
        self.use_mean_for_discrete = use_mean_for_discrete

    def reference_at_t_next(
        self,
        batch_ligand: Tensor,
        time_indices: list,
        step_idx: int,
    ) -> Tuple[Tensor, Tensor]:
        """
        逆向第 step_idx 步结束后，获取下一时刻的参考噪声态。
        t_next = time_indices[step_idx+1]（若存在），否则 0。
        """
        if step_idx + 1 < len(time_indices):
            t_next = int(time_indices[step_idx + 1])
        else:
            t_next = 0
        st = self.state
        ref_pos = forward_noisy_positions(
            self.model, st.x0_pos, batch_ligand, t_next, fixed_eps=st.fixed_eps_pos
        )
        ref_logv = forward_noisy_atom_types(
            self.model,
            st.x0_log_v,
            batch_ligand,
            t_next,
            use_mean_prob=self.use_mean_for_discrete,
        )
        return ref_pos, ref_logv

    def blend_after_step(
        self,
        ligand_pos: Tensor,
        log_ligand_v: Tensor,
        ref_pos_noisy: Tensor,
        ref_log_v_noisy: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        return dual_mask_repaint_blend_states(
            ligand_pos,
            log_ligand_v,
            ref_pos_noisy,
            ref_log_v_noisy,
            pos_mask=self.state.pos_mask,
            type_mask=self.state.type_mask,
        )


def compute_scaffold_pos_rmsd(
    ligand_pos: Tensor,
    ref_pos: Tensor,
    scaffold_mask: Tensor,
) -> float:
    """
    计算骨架原子的 RMSD（仅统计 scaffold_mask=1 的原子）。

    用于评估骨架约束优化中骨架位置漂移情况。
    """
    m = _ensure_mask_column(scaffold_mask, ligand_pos.shape[0], ligand_pos.device, ligand_pos.dtype)
    m_flat = m.view(-1).bool()
    if m_flat.sum() == 0:
        return 0.0
    pos_scaffold = ligand_pos[m_flat]
    ref_scaffold = ref_pos[m_flat].to(ligand_pos.device, ligand_pos.dtype)
    rmsd = torch.sqrt(((pos_scaffold - ref_scaffold) ** 2).sum(dim=-1).mean()).item()
    return float(rmsd)
