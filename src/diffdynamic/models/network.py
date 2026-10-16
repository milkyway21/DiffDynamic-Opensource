"""Network components of the DiffDynamic model."""

from __future__ import annotations

from diffdynamic.models.common import ShiftedSoftplus
from diffdynamic.models.common import compose_context
from diffdynamic.models.egnn import EGNN
from diffdynamic.models.uni_transformer import UniTransformerO2TwoUpdateGeneral
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffdynamic.models.probability import ensure_log_ligand
from diffdynamic.models.probability import log_1_min_a
from diffdynamic.models.probability import to_torch_const
from diffdynamic.models.schedules import cosine_beta_schedule
from diffdynamic.models.schedules import get_beta_schedule


def get_refine_net(refine_net_type, config):  # 根据类型与配置构建精炼网络。
    if refine_net_type == 'uni_o2':  # 当选择 UniTransformer 架构时。
        refine_net = UniTransformerO2TwoUpdateGeneral(  # 初始化 UniTransformer 精炼网络。
            num_blocks=config.num_blocks,  # 设置块数。
            num_layers=config.num_layers,  # 设置每块层数。
            hidden_dim=config.hidden_dim,  # 设置隐藏维度。
            n_heads=config.n_heads,  # 设置注意力头数。
            k=config.knn,  # 设置 kNN 邻居数。
            edge_feat_dim=config.edge_feat_dim,  # 设置边特征维度。
            num_r_gaussian=config.num_r_gaussian,  # 设置径向高斯数量。
            num_node_types=config.num_node_types,  # 设置节点类型数量。
            act_fn=config.act_fn,  # 设置激活函数。
            norm=config.norm,  # 控制是否开启归一化。
            cutoff_mode=config.cutoff_mode,  # 设置邻接截断模式。
            ew_net_type=config.ew_net_type,  # 设置边权网络类型。
            num_x2h=config.num_x2h,  # 设置坐标到特征的更新次数。
            num_h2x=config.num_h2x,  # 设置特征到坐标的更新次数。
            r_max=config.r_max,  # 设置最大距离阈值。
            x2h_out_fc=config.x2h_out_fc,  # 设置是否在 x2h 末尾使用全连接层。
            sync_twoup=config.sync_twoup  # 设置双更新同步开关。
        )
    elif refine_net_type == 'egnn':  # 当选择 EGNN 架构时。
        refine_net = EGNN(  # 初始化 EGNN 精炼网络。
            num_layers=config.num_layers,  # 设置层数。
            hidden_dim=config.hidden_dim,  # 设置隐藏维度。
            edge_feat_dim=config.edge_feat_dim,  # 设置边特征维度。
            num_r_gaussian=1,  # 固定使用单一径向尺度（EGNN 中默认）。
            k=config.knn,  # 设置 kNN 邻居数。
            cutoff_mode=config.cutoff_mode  # 设置边构建模式。
        )
    else:  # 当类型不被支持时。
        raise ValueError(refine_net_type)  # 抛出异常提示。
    return refine_net  # 返回构建好的精炼网络实例。


class SinusoidalPosEmb(nn.Module):  # 定义正弦位置编码模块。
    def __init__(self, dim):  # 初始化函数，指定维度。
        super().__init__()  # 调用父类构造函数。
        self.dim = dim  # 保存嵌入维度。

    def forward(self, x):  # 前向传播，根据输入生成时间嵌入。
        device = x.device  # 获取输入张量所在设备。
        half_dim = self.dim // 2  # 计算一半维度用于正弦/余弦配对。
        emb = np.log(10000) / (half_dim - 1)  # 计算指数步长。
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)  # 生成频率序列。
        emb = x[:, None] * emb[None, :]  # 计算时间与频率的乘积。
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)  # 拼接正弦与余弦成完整编码。
        return emb  # 返回位置嵌入。


class ScoreNetwork(nn.Module):
    """Network responsibilities; registered parameter names stay unchanged."""

    def __init__(self, config, protein_atom_feature_dim, ligand_atom_feature_dim):  # 初始化模型结构。
        super().__init__()  # 调用父类构造函数。
        self.config = config  # 保存配置对象。

        # variance schedule  # 保留原注释：说明下方配置与方差调度相关。
        self.model_mean_type = config.model_mean_type  # ['noise', 'C0']  # 指示模型输出类型（噪声或均值）。
        self.loss_v_weight = config.loss_v_weight  # 设置原子类型损失权重。
        self.ligand_v_input = getattr(config, 'ligand_v_input', 'onehot')  # 获取配体类型输入模式。
        self.use_grad_fusion = getattr(config, 'use_grad_fusion', False)  # 是否启用梯度融合策略。
        self.grad_fusion_lambda = getattr(config, 'grad_fusion_lambda', 0.5)  # 梯度融合系数或调度策略。
        self.dynamic_pos_step_clip = getattr(config, 'dynamic_pos_step_clip', None)  # 动态位置梯度裁剪阈值。
        self.dynamic_v_step_clip = getattr(config, 'dynamic_v_step_clip', None)  # 动态类别梯度裁剪阈值。
        self.loss_v2_weight = getattr(config, 'loss_v2_weight', 0.0)  # 额外类别损失权重。
        self.dynamic_large_step_defaults = getattr(config, 'dynamic_large_step', {})  # 大步采样的默认配置。
        self.dynamic_refine_defaults = getattr(config, 'dynamic_refine', {})  # 精炼采样的默认配置。
        # self.v_mode = config.v_mode  # 保留注释：遗留配置。
        # assert self.v_mode == 'categorical'  # 保留注释：对应断言。
        # self.v_net_type = getattr(config, 'v_net_type', 'mlp')  # 保留注释：额外网络类型。
        # self.bond_loss = getattr(config, 'bond_loss', False)  # 保留注释：键损失开关。
        # self.bond_net_type = getattr(config, 'bond_net_type', 'pre_att')  # 保留注释：键预测网络类型。
        # self.loss_bond_weight = getattr(config, 'loss_bond_weight', 0.)  # 保留注释：键损失权重。
        # self.loss_non_bond_weight = getattr(config, 'loss_non_bond_weight', 0.)  # 保留注释：非键损失权重。

        self.sample_time_method = getattr(config, "sample_time_method", "symmetric")  # 与旧 ckpt / 部分 YAML 兼容
        # self.loss_pos_type = config.loss_pos_type  # ['mse', 'kl']  # 保留注释：位置损失形式。
        # print(f'Loss pos mode {self.loss_pos_type} applied!')  # 保留调试输出。
        # print(f'Loss bond net type: {self.bond_net_type} '  # 保留调试输出。
        #       f'bond weight: {self.loss_bond_weight} non bond weight: {self.loss_non_bond_weight}')

        if config.beta_schedule == 'cosine':  # 根据配置选择余弦调度。
            alphas = cosine_beta_schedule(config.num_diffusion_timesteps, config.pos_beta_s) ** 2  # 获取余弦调度并平方。
            # print('cosine pos alpha schedule applied!')  # 保留注释：调试输出。
            betas = 1. - alphas  # 将 alpha 转换为 beta。
        else:  # 使用通用调度生成函数。
            betas = get_beta_schedule(
                beta_schedule=config.beta_schedule,  # 指定调度名称。
                beta_start=config.beta_start,  # 起始 beta。
                beta_end=config.beta_end,  # 结束 beta。
                num_diffusion_timesteps=config.num_diffusion_timesteps,  # 扩散步数。
            )
            alphas = 1. - betas  # 通过 beta 推导 alpha。
        alphas_cumprod = np.cumprod(alphas, axis=0)  # 计算 alpha 的累计乘积。
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])  # 构造前一项累计乘积，首项补 1。

        self.betas = to_torch_const(betas)  # 将 beta 转换为常量张量并注册。
        self.num_timesteps = self.betas.size(0)  # 记录扩散步数。
        self.alphas_cumprod = to_torch_const(alphas_cumprod)  # 注册 alpha 累乘作为常量。
        self.alphas_cumprod_prev = to_torch_const(alphas_cumprod_prev)  # 注册前置累乘。

        # calculations for diffusion q(x_t | x_{t-1}) and others  # 保留注释：说明以下常量用于前向扩散。
        self.sqrt_alphas_cumprod = to_torch_const(np.sqrt(alphas_cumprod))  # 注册 sqrt(ᾱ_t)。
        self.sqrt_one_minus_alphas_cumprod = to_torch_const(np.sqrt(1. - alphas_cumprod))  # 注册 sqrt(1-ᾱ_t)。
        self.sqrt_recip_alphas_cumprod = to_torch_const(np.sqrt(1. / alphas_cumprod))  # 注册 sqrt(1/ᾱ_t)。
        self.sqrt_recipm1_alphas_cumprod = to_torch_const(np.sqrt(1. / alphas_cumprod - 1))  # 注册 sqrt(1/ᾱ_t -1)。

        # calculations for posterior q(x_{t-1} | x_t, x_0)  # 保留注释：以下参数用于后验计算。
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)  # 计算后验方差。
        self.posterior_mean_c0_coef = to_torch_const(betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))  # 注册均值系数 (x0 部分)。
        self.posterior_mean_ct_coef = to_torch_const(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))  # 注册均值系数 (xt 部分)。
        # log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain  # 保留注释：解释对数截断。
        self.posterior_var = to_torch_const(posterior_variance)  # 注册后验方差。
        self.posterior_logvar = to_torch_const(np.log(np.append(self.posterior_var[1], self.posterior_var[1:])))  # 注册后验对数方差，首项重复第二项。

        # atom type diffusion schedule in log space  # 保留注释：以下针对离散类型扩散。
        if config.v_beta_schedule == 'cosine':  # 仅支持余弦调度。
            alphas_v = cosine_beta_schedule(self.num_timesteps, config.v_beta_s)  # 计算类型扩散的 alpha。
            # print('cosine v alpha schedule applied!')  # 保留注释：调试输出。
        else:  # 不支持其他调度。
            raise NotImplementedError  # 抛出未实现错误。
        log_alphas_v = np.log(alphas_v)  # 计算 log α。
        log_alphas_cumprod_v = np.cumsum(log_alphas_v)  # 计算 log α 的累积和。
        self.log_alphas_v = to_torch_const(log_alphas_v)  # 注册 log α。
        self.log_one_minus_alphas_v = to_torch_const(log_1_min_a(log_alphas_v))  # 注册 log(1-α)。
        self.log_alphas_cumprod_v = to_torch_const(log_alphas_cumprod_v)  # 注册累计 log α。
        self.log_one_minus_alphas_cumprod_v = to_torch_const(log_1_min_a(log_alphas_cumprod_v))  # 注册 log(1-ᾱ)。

        self.register_buffer('Lt_history', torch.zeros(self.num_timesteps))  # 记录不同时间步的损失历史。
        self.register_buffer('Lt_count', torch.zeros(self.num_timesteps))  # 记录损失统计次数。

        # model definition  # 保留注释：开始定义模型结构。
        self.hidden_dim = config.hidden_dim  # 设置模型隐藏维度。
        self.num_classes = ligand_atom_feature_dim  # 设置配体类别数量。
        if self.config.node_indicator:  # 若启用节点类型指示器。
            emb_dim = self.hidden_dim - 1  # 预留一维用于节点类别标记。
        else:  # 未启用指示器。
            emb_dim = self.hidden_dim  # 嵌入维度等于隐藏维度。

        # atom embedding  # 保留注释：原子特征嵌入。
        self.protein_atom_emb = nn.Linear(protein_atom_feature_dim, emb_dim)  # 映射蛋白原子特征到嵌入空间。

        # center pos  # 保留注释：中心化模式。
        self.center_pos_mode = config.center_pos_mode  # ['none', 'protein']  # 保存中心化策略。

        # time embedding  # 保留注释：时间嵌入。
        self.time_emb_dim = config.time_emb_dim  # 时间嵌入维度。
        self.time_emb_mode = config.time_emb_mode  # ['simple', 'sin']  # 时间嵌入模式。
        if self.time_emb_dim > 0:  # 如果启用时间嵌入。
            if self.time_emb_mode == 'simple':  # 简单模式：直接拼接时间标量。
                self.ligand_atom_emb = nn.Linear(ligand_atom_feature_dim + 1, emb_dim)  # 线性层接受多一维时间标量。
            elif self.time_emb_mode == 'sin':  # 正弦模式：使用位置编码。
                self.time_emb = nn.Sequential(
                    SinusoidalPosEmb(self.time_emb_dim),  # 生成正弦时间编码。
                    nn.Linear(self.time_emb_dim, self.time_emb_dim * 4),  # 提升维度。
                    nn.GELU(),  # 使用 GELU 激活。
                    nn.Linear(self.time_emb_dim * 4, self.time_emb_dim)  # 映射回时间嵌入维度。
                )
                self.ligand_atom_emb = nn.Linear(ligand_atom_feature_dim + self.time_emb_dim, emb_dim)  # 输入拼接时间嵌入。
            else:  # 未支持的时间模式。
                raise NotImplementedError  # 抛出异常提示。
        else:  # 不使用时间嵌入。
            self.ligand_atom_emb = nn.Linear(ligand_atom_feature_dim, emb_dim)  # 仅基于原子特征映射。

        self.refine_net_type = config.model_type  # 记录精炼网络类型。
        self.refine_net = get_refine_net(self.refine_net_type, config)  # 根据配置实例化精炼网络。
        self.v_inference = nn.Sequential(  # 构建原子类型预测头。
            nn.Linear(self.hidden_dim, self.hidden_dim),  # 第一层线性映射。
            ShiftedSoftplus(),  # 使用平移 Softplus 激活。
            nn.Linear(self.hidden_dim, ligand_atom_feature_dim),  # 输出每个类别的 logits。
        )

    def _prepare_ligand_inputs(self, init_ligand_v):  # 预处理配体类别输入，统一为张量形式。
        # 验证输入不为空
        if isinstance(init_ligand_v, torch.Tensor) and init_ligand_v.numel() == 0:
            raise ValueError(
                f"_prepare_ligand_inputs: Empty input detected. init_ligand_v.shape={init_ligand_v.shape}, "
                f"num_classes={self.num_classes}"
            )
        
        if torch.is_floating_point(init_ligand_v):  # 若输入已是浮点表示。
            if init_ligand_v.dim() != 2:  # 检查维度是否符合 [N, num_classes]。
                raise ValueError(
                    f'Expect ligand features of shape [N, num_classes] when providing floating tensors. '
                    f'Got shape {init_ligand_v.shape}, num_classes={self.num_classes}'
                )  # 抛出异常提示。
            # 验证特征维度
            if init_ligand_v.shape[1] != self.num_classes:
                raise ValueError(
                    f"_prepare_ligand_inputs: Feature dimension mismatch. init_ligand_v.shape={init_ligand_v.shape}, "
                    f"expected num_classes={self.num_classes}"
                )
            log_ligand_v = ensure_log_ligand(init_ligand_v, self.num_classes, mode=self.ligand_v_input)  # 根据模式转为 log 概率。
            if self.ligand_v_input == 'log_prob':  # 如果配置期望 log 概率。
                ligand_emb_input = init_ligand_v  # 直接使用原始输入。
            elif self.ligand_v_input == 'logits':  # 如果配置期望 logits。
                ligand_emb_input = init_ligand_v  # 按原值返回。
            elif self.ligand_v_input == 'prob':  # 如果配置期望概率分布。
                ligand_emb_input = init_ligand_v  # 直接作为概率。
            else:  # 默认转换为概率。
                ligand_emb_input = log_ligand_v.exp()  # 对 log 概率取指数得到概率。
            return ligand_emb_input, log_ligand_v, init_ligand_v  # 返回嵌入输入、log 概率和原始张量。

        ligand_emb_input = F.one_hot(init_ligand_v.long(), self.num_classes).float()  # 将离散索引转换为 one-hot。
        log_ligand_v = torch.log(ligand_emb_input.clamp(min=1e-30))  # 转化为 log 概率表示。
        return ligand_emb_input, log_ligand_v, init_ligand_v  # 返回嵌入输入、log 概率和索引。

    def _format_forward_ligand_input(self, log_ligand_v):  # 根据配置将 log 概率转换为前向所需格式。
        mode = self.ligand_v_input  # 获取输入模式。
        if mode == 'log_prob':  # 若模型期望 log 概率。
            return log_ligand_v  # 直接返回。
        if mode == 'prob':  # 若期望概率分布。
            return log_ligand_v.exp()  # 对 log 概率取指数。
        if mode == 'logits':  # 若期望 logits。
            return log_ligand_v  # 此处同样返回 log 概率作为 logits。
        # default to categorical indices  # 保留注释：默认转换为类别索引。
        return log_ligand_v.argmax(dim=-1)  # 取最大值索引作为类别标签。

    def forward(self, protein_pos, protein_v, batch_protein, init_ligand_pos, init_ligand_v, batch_ligand,
                time_step=None, return_all=False, fix_x=False):  # 前向传播，输出配体位置与类别预测。

        batch_size = batch_protein.max().item() + 1  # 计算批次数量。
        ligand_emb_input, _, _ = self._prepare_ligand_inputs(init_ligand_v)  # 准备配体输入嵌入与 log 概率。
        # time embedding  # 保留注释：以下处理时间嵌入。
        if self.time_emb_dim > 0:  # 若启用时间嵌入。
            if self.time_emb_mode == 'simple':  # 简单模式。
                input_ligand_feat = torch.cat([
                    ligand_emb_input,  # 拼接配体初始特征。
                    (time_step / self.num_timesteps)[batch_ligand].unsqueeze(-1)  # 添加归一化时间步。
                ], -1)  # 沿特征维度拼接。
            elif self.time_emb_mode == 'sin':  # 正弦位置编码模式。
                time_feat = self.time_emb(time_step)  # 计算时间嵌入向量。
                input_ligand_feat = torch.cat([ligand_emb_input, time_feat], -1)  # 拼接原始特征与时间嵌入。
            else:  # 未实现的时间模式。
                raise NotImplementedError  # 抛出异常提示。
        else:  # 未启用时间嵌入。
            input_ligand_feat = ligand_emb_input  # 直接使用初始特征。

        h_protein = self.protein_atom_emb(protein_v)  # 将蛋白原子特征映射到隐藏空间。
        init_ligand_h = self.ligand_atom_emb(input_ligand_feat)  # 将配体特征映射到隐藏空间。

        if self.config.node_indicator:  # 若使用节点类型指示器。
            h_protein = torch.cat([h_protein, torch.zeros(len(h_protein), 1).to(h_protein)], -1)  # 为蛋白节点拼接 0 指示位。
            init_ligand_h = torch.cat([init_ligand_h, torch.ones(len(init_ligand_h), 1).to(h_protein)], -1)  # 为配体节点拼接 1 指示位。

        h_all, pos_all, batch_all, mask_ligand = compose_context(  # 拼接蛋白与配体节点上下文。
            h_protein=h_protein,  # 传入蛋白隐藏特征。
            h_ligand=init_ligand_h,  # 传入配体隐藏特征。
            pos_protein=protein_pos,  # 传入蛋白坐标。
            pos_ligand=init_ligand_pos,  # 传入配体坐标。
            batch_protein=batch_protein,  # 蛋白批次索引。
            batch_ligand=batch_ligand,  # 配体批次索引。
        )

        outputs = self.refine_net(h_all, pos_all, mask_ligand, batch_all, return_all=return_all, fix_x=fix_x)  # 调用精炼网络更新几何与特征。
        final_pos, final_h = outputs['x'], outputs['h']  # 提取全局位置和隐藏特征结果。
        
        # 验证 refine_net 输出
        if final_pos.numel() == 0 or final_h.numel() == 0:
            raise ValueError(
                f"forward: refine_net returned empty outputs. final_pos.shape={final_pos.shape}, "
                f"final_h.shape={final_h.shape}, h_all.shape={h_all.shape}, pos_all.shape={pos_all.shape}, "
                f"mask_ligand.sum()={mask_ligand.sum().item()}, batch_all.shape={batch_all.shape}"
            )
        
        # 验证 mask_ligand 的有效性
        if mask_ligand.sum().item() == 0:
            raise ValueError(
                f"forward: mask_ligand contains no True values. mask_ligand.shape={mask_ligand.shape}, "
                f"batch_ligand.shape={batch_ligand.shape}, batch_ligand.numel()={batch_ligand.numel()}"
            )
        
        if final_pos.shape[0] != mask_ligand.shape[0] or final_h.shape[0] != mask_ligand.shape[0]:
            raise ValueError(
                f"forward: refine_net output shape mismatch. final_pos.shape={final_pos.shape}, "
                f"final_h.shape={final_h.shape}, mask_ligand.shape={mask_ligand.shape}"
            )
        
        final_ligand_pos = final_pos[mask_ligand]  # 选择配体对应的结果。
        final_ligand_h = final_h[mask_ligand]
        
        # 验证索引后的结果
        if final_ligand_pos.numel() == 0 or final_ligand_h.numel() == 0:
            raise ValueError(
                f"forward: Empty result after mask_ligand indexing. final_ligand_pos.shape={final_ligand_pos.shape}, "
                f"final_ligand_h.shape={final_ligand_h.shape}, mask_ligand.sum()={mask_ligand.sum().item()}, "
                f"batch_ligand.shape={batch_ligand.shape}, expected ligand atoms={batch_ligand.numel()}"
            )
        
        if final_ligand_pos.shape[0] != batch_ligand.shape[0]:
            raise ValueError(
                f"forward: Ligand position count mismatch. final_ligand_pos.shape={final_ligand_pos.shape}, "
                f"batch_ligand.shape={batch_ligand.shape}"
            )
        
        final_ligand_v = self.v_inference(final_ligand_h)  # 通过分类头预测配体原子类型 logits。

        preds = {  # 构造输出字典。
            'pred_ligand_pos': final_ligand_pos,  # 配体位置预测。
            'pred_ligand_v': final_ligand_v,  # 配体类别预测。
            'final_h': final_h,  # 所有节点的最终隐藏特征。
            'final_ligand_h': final_ligand_h  # 配体节点的最终隐藏特征。
        }
        if return_all:  # 若需要返回所有层的中间结果。
            final_all_pos, final_all_h = outputs['all_x'], outputs['all_h']  # 从精炼网络获取全层列表。
            final_all_ligand_pos = [pos[mask_ligand] for pos in final_all_pos]  # 提取每层的配体位置。
            final_all_ligand_v = [self.v_inference(h[mask_ligand]) for h in final_all_h]  # 对每层配体特征进行推断。
            preds.update({  # 将分层结果添加到输出。
                'layer_pred_ligand_pos': final_all_ligand_pos,  # 每层配体位置。
                'layer_pred_ligand_v': final_all_ligand_v  # 每层配体类别 logits。
            })
        return preds  # 返回预测结果。


SinusoidalPosEmb.__module__ = "models.molopt_score_model"
