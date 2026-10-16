"""Diffusion components of the DiffDynamic model."""

from __future__ import annotations

from torch_scatter import scatter_mean
import numpy as np
import torch
import torch.nn.functional as F
from diffdynamic.models.probability import categorical_kl
from diffdynamic.models.probability import center_pos
from diffdynamic.models.probability import extract
from diffdynamic.models.probability import index_to_log_onehot
from diffdynamic.models.probability import log_add_exp
from diffdynamic.models.probability import log_categorical
from diffdynamic.models.probability import log_normal
from diffdynamic.models.probability import log_sample_categorical
from diffdynamic.models.probability import normal_kl


class DiffusionTrainingMixin:
    """Diffusion responsibilities; registered parameter names stay unchanged."""

    def q_v_pred_one_timestep(self, log_vt_1, t, batch):  # 计算一步前向扩散的 log 概率。
        # q(vt | vt-1)  # 保留注释：说明公式来源。
        log_alpha_t = extract(self.log_alphas_v, t, batch)  # 提取对应时间步的 log α。
        log_1_min_alpha_t = extract(self.log_one_minus_alphas_v, t, batch)  # 提取 log(1-α)。

        # alpha_t * vt + (1 - alpha_t) 1 / K  # 保留注释：公式描述。
        log_probs = log_add_exp(  # 使用 log-sum-exp 合并两项概率。
            log_vt_1 + log_alpha_t,  # 前一时刻概率乘 α。
            log_1_min_alpha_t - np.log(self.num_classes)  # 均匀噪声项 (1-α)/K。
        )
        return log_probs  # 返回当前时刻的 log 概率。

    def q_v_pred(self, log_v0, t, batch):  # 计算 q(v_t | v_0) 的 log 概率。
        # compute q(vt | v0)  # 保留注释：说明操作。
        log_cumprod_alpha_t = extract(self.log_alphas_cumprod_v, t, batch)  # 提取 log ᾱ。
        log_1_min_cumprod_alpha = extract(self.log_one_minus_alphas_cumprod_v, t, batch)  # 提取 log(1-ᾱ)。

        log_probs = log_add_exp(  # 合并 v0 贡献与均匀噪声。
            log_v0 + log_cumprod_alpha_t,  # v0 对应的缩放项。
            log_1_min_cumprod_alpha - np.log(self.num_classes)  # 均匀噪声项。
        )
        return log_probs  # 返回 log 概率。

    def q_v_sample(self, log_v0, t, batch):  # 从 q(v_t | v_0) 中采样离散原子类型。
        log_qvt_v0 = self.q_v_pred(log_v0, t, batch)  # 计算条件分布。
        sample_index = log_sample_categorical(log_qvt_v0)  # 采样类别索引。
        log_sample = index_to_log_onehot(sample_index, self.num_classes)  # 转换为 log one-hot。
        return sample_index, log_sample  # 返回索引与 log one-hot。

    def q_v_pred_with_noise(self, log_v0, t, batch):  # 计算带噪声的 q(v_t | v_0) 的 log 概率。
        # compute q(vt | v0) with gumbel noise  # 保留注释：说明操作。
        log_cumprod_alpha_t = extract(self.log_alphas_cumprod_v, t, batch)  # 提取 log ᾱ。
        log_1_min_cumprod_alpha = extract(self.log_one_minus_alphas_cumprod_v, t, batch)  # 提取 log(1-ᾱ)。
        uniform = torch.rand_like(log_v0)  # 生成均匀随机数。
        gumbel_noise = -0.5 * torch.log(-torch.log(uniform + 1e-30) + 1e-30)  # 生成 Gumbel 噪声。
        # equation (3) with noise  # 保留注释：公式说明。
        log_probs = log_add_exp(  # 合并 v0 贡献与带噪声的均匀项。
            log_v0 + log_cumprod_alpha_t,  # v0 对应的缩放项。
            log_1_min_cumprod_alpha + gumbel_noise - np.log(self.num_classes)  # 带噪声的均匀项。
        )
        log_probs = F.log_softmax(log_probs, dim=-1)  # 归一化为 log 概率。
        return log_probs  # 返回 log 概率。

    def q_v_sample_with_noise(self, log_v0, t, batch):  # 从带噪声的 q(v_t | v_0) 中采样。
        # log_v0 + noise -> log_qvt_v0  # 保留注释：说明操作。
        # sample_index == new categories  # 保留注释：采样结果。
        # log_sample == noised one hot categories  # 保留注释：log one-hot 表示。
        log_qvt_v0 = self.q_v_pred_with_noise(log_v0, t, batch)  # 计算带噪声的条件分布。
        # log(vt|v0) = log(a_t*v_0 + (1-a_t)*noise)  # 保留注释：公式说明。
        sample_index = log_qvt_v0.argmax(dim=-1)  # 取最大值索引作为采样结果。
        return sample_index, log_qvt_v0  # 返回索引与 log 概率。

    def q_v_posterior(self, log_v0, log_vt, t, batch):  # 计算离散后验 q(v_{t-1} | v_t, v_0)。
        # q(vt-1 | vt, v0) = q(vt | vt-1, x0) * q(vt-1 | x0) / q(vt | x0)  # 保留注释：公式说明。
        t_minus_1 = t - 1  # 计算 t-1。
        # Remove negative values, will not be used anyway for final decoder  # 保留注释：说明下行原因。
        t_minus_1 = torch.where(t_minus_1 < 0, torch.zeros_like(t_minus_1), t_minus_1)  # 将负值裁剪为 0。
        log_qvt1_v0 = self.q_v_pred(log_v0, t_minus_1, batch)  # 计算 q(v_{t-1} | v_0)。
        unnormed_logprobs = log_qvt1_v0 + self.q_v_pred_one_timestep(log_vt, t, batch)  # 累加联合项。
        log_vt1_given_vt_v0 = unnormed_logprobs - torch.logsumexp(unnormed_logprobs, dim=-1, keepdim=True)  # 归一化得到后验。
        return log_vt1_given_vt_v0  # 返回 log 概率。

    def q_v_posterior_with_noise(self, log_v0, log_vt, t, batch):  # 计算带噪声的离散后验 q(v_{t-1} | v_t, v_0)。
        # log_v0 = log_ligand_v_recon; log_vt = log_ligand_v  # 保留注释：参数说明。
        # q(vt-1 | vt, v0) = q(vt | vt-1, x0) * q(vt-1 | x0) / q(vt | x0)  # 保留注释：公式说明。
        t_minus_1 = t - 1  # 计算 t-1。
        # Remove negative values, will not be used anyway for final decoder  # 保留注释：说明下行原因。
        t_minus_1 = torch.where(t_minus_1 < 0, torch.zeros_like(t_minus_1), t_minus_1)  # 将负值裁剪为 0。
        # q(v_t-1|v_0)  # 保留注释：计算前一步分布。
        log_qvt1_v0 = self.q_v_pred_with_noise(log_v0, t_minus_1, batch)  # 使用带噪声的预测。
        unnormed_logprobs = log_qvt1_v0 + self.q_v_pred_one_timestep(log_vt, t, batch)  # 累加联合项。
        # q(vt-1 | v0)* q(vt | vt-1, v0) / q(vt | v0)  # 保留注释：公式说明。
        log_vt1_given_vt_v0 = unnormed_logprobs - torch.logsumexp(unnormed_logprobs, dim=-1, keepdim=True)  # 归一化得到后验。
        return log_vt1_given_vt_v0  # 返回 log 概率。

    def kl_v_prior(self, log_x_start, batch):  # 计算离散变量对先验的 KL。
        num_graphs = batch.max().item() + 1  # 获取批次数。
        log_qxT_prob = self.q_v_pred(log_x_start, t=[self.num_timesteps - 1] * num_graphs, batch=batch)  # 计算最终时刻分布。
        log_half_prob = -torch.log(self.num_classes * torch.ones_like(log_qxT_prob))  # 构造均匀先验 log 概率。
        kl_prior = categorical_kl(log_qxT_prob, log_half_prob)  # 计算 KL 散度。
        kl_prior = scatter_mean(kl_prior, batch, dim=0)  # 按批次平均。
        return kl_prior  # 返回 KL 值。

    def _predict_x0_from_eps(self, xt, eps, t, batch):  # 根据噪声估计还原 x0。
        # 验证输入有效性
        if xt.numel() == 0 or eps.numel() == 0 or batch.numel() == 0:
            raise ValueError(
                f"_predict_x0_from_eps: Empty input detected. xt.numel()={xt.numel()}, "
                f"eps.numel()={eps.numel()}, batch.numel()={batch.numel()}. "
                f"This should not happen in normal operation."
            )
        
        # 验证维度一致性
        if xt.shape[0] != eps.shape[0] or xt.shape[0] != batch.shape[0]:
            raise ValueError(
                f"_predict_x0_from_eps: Dimension mismatch. xt.shape={xt.shape}, "
                f"eps.shape={eps.shape}, batch.shape={batch.shape}"
            )
        
        # 提取系数并计算 x0
        sqrt_recip_alphas_cumprod_t = extract(self.sqrt_recip_alphas_cumprod, t, batch)  # 提取系数
        sqrt_recipm1_alphas_cumprod_t = extract(self.sqrt_recipm1_alphas_cumprod, t, batch)  # 提取系数
        
        # 验证提取的系数形状
        if sqrt_recip_alphas_cumprod_t.shape[0] != xt.shape[0]:
            raise ValueError(
                f"_predict_x0_from_eps: Coefficient shape mismatch. "
                f"sqrt_recip_alphas_cumprod_t.shape={sqrt_recip_alphas_cumprod_t.shape}, "
                f"xt.shape={xt.shape}, t={t}, batch.shape={batch.shape}"
            )
        
        pos0_from_e = sqrt_recip_alphas_cumprod_t * xt - sqrt_recipm1_alphas_cumprod_t * eps  # 按扩散公式反推 x0。
        
        # 验证输出不为空
        if pos0_from_e.numel() == 0:
            raise ValueError(
                f"_predict_x0_from_eps: Returned empty tensor. xt.shape={xt.shape}, "
                f"eps.shape={eps.shape}, t={t}, batch.shape={batch.shape}, "
                f"sqrt_recip_alphas_cumprod_t.shape={sqrt_recip_alphas_cumprod_t.shape}"
            )
        
        # 验证输出形状
        if pos0_from_e.shape[0] != xt.shape[0]:
            raise ValueError(
                f"_predict_x0_from_eps: Output shape mismatch. pos0_from_e.shape={pos0_from_e.shape}, "
                f"xt.shape={xt.shape}"
            )
        
        return pos0_from_e  # 返回估计的原始坐标。

    def q_pos_sample(self, x0, t, batch):  # 前向扩散：从 x0 采样 xt = sqrt(ᾱ_t)*x0 + sqrt(1-ᾱ_t)*noise。
        """用于 TargetDiff 基准扩散修复：将生成分子 x0 前向扩散到 t 时刻，作为 sample_diffusion 的初始状态。"""
        sqrt_alpha = extract(self.sqrt_alphas_cumprod, t, batch)  # (num_atoms, 1)
        sqrt_one_minus = extract(self.sqrt_one_minus_alphas_cumprod, t, batch)  # (num_atoms, 1)
        noise = torch.randn_like(x0, device=x0.device)
        xt = sqrt_alpha * x0 + sqrt_one_minus * noise
        return xt

    def q_pos_posterior(self, x0, xt, t, batch):  # 计算连续变量后验均值。
        # Compute the mean and variance of the diffusion posterior q(x_{t-1} | x_t, x_0)  # 保留注释：说明用途。
        # 检查输入是否为空
        if batch.numel() == 0 or x0.numel() == 0 or xt.numel() == 0:  # 如果批次或输入为空。
            # 如果输入为空，应该抛出错误而不是返回空张量（这会掩盖真正的问题）
            raise ValueError(
                f"q_pos_posterior: Empty input detected. batch.numel()={batch.numel()}, "
                f"x0.numel()={x0.numel()}, xt.numel()={xt.numel()}. "
                f"This should not happen in normal operation. Please check batch_ligand and ligand_pos initialization."
            )
        
        # 确保 x0 和 xt 大小一致
        if x0.shape[0] != xt.shape[0]:  # 如果大小不一致。
            raise ValueError(f"x0 and xt must have the same size in dimension 0, got {x0.shape[0]} and {xt.shape[0]}")
        
        # 确保 batch 大小与 x0/xt 一致
        if batch.shape[0] != x0.shape[0]:  # 如果批次大小不一致。
            raise ValueError(
                f"batch must have the same size as x0/xt in dimension 0, got batch.shape[0]={batch.shape[0]} "
                f"and x0.shape[0]={x0.shape[0]}. batch range=[{batch.min().item()}, {batch.max().item()}]"
            )
        
        # 验证 t 的格式和范围
        if isinstance(t, torch.Tensor):
            if t.numel() == 0:
                raise ValueError(f"q_pos_posterior: t is empty tensor")
            num_graphs = t.shape[0]
            if batch.max().item() >= num_graphs or batch.min().item() < 0:
                raise IndexError(
                    f"q_pos_posterior: batch indices out of range. batch range=[{batch.min().item()}, {batch.max().item()}], "
                    f"num_graphs={num_graphs}, t shape={t.shape}"
                )
        
        c0_coef = extract(self.posterior_mean_c0_coef, t, batch)  # 提取 x0 系数。
        ct_coef = extract(self.posterior_mean_ct_coef, t, batch)  # 提取 xt 系数。
        
        # 确保提取的系数大小正确
        if c0_coef.shape[0] != x0.shape[0]:  # 如果系数大小不匹配。
            raise ValueError(
                f"c0_coef size {c0_coef.shape[0]} does not match x0 size {x0.shape[0]}. "
                f"t shape={t.shape if isinstance(t, torch.Tensor) else type(t)}, "
                f"batch shape={batch.shape}, batch range=[{batch.min().item()}, {batch.max().item()}]"
            )
        if ct_coef.shape[0] != xt.shape[0]:  # 如果系数大小不匹配。
            raise ValueError(
                f"ct_coef size {ct_coef.shape[0]} does not match xt size {xt.shape[0]}. "
                f"t shape={t.shape if isinstance(t, torch.Tensor) else type(t)}, "
                f"batch shape={batch.shape}, batch range=[{batch.min().item()}, {batch.max().item()}]"
            )
        
        pos_model_mean = c0_coef * x0 + ct_coef * xt  # 根据闭式公式计算均值。
        return pos_model_mean  # 返回后验均值。

    def kl_pos_prior(self, pos0, batch):  # 计算连续变量对先验的 KL。
        num_graphs = batch.max().item() + 1  # 获取批次数。
        a_pos = extract(self.alphas_cumprod, [self.num_timesteps - 1] * num_graphs, batch)  # (num_ligand_atoms, 1)  # 提取最终 ᾱ。
        pos_model_mean = a_pos.sqrt() * pos0  # 计算均值项。
        pos_log_variance = torch.log((1.0 - a_pos).sqrt())  # 计算方差的 log。
        kl_prior = normal_kl(torch.zeros_like(pos_model_mean), torch.zeros_like(pos_log_variance),
                             pos_model_mean, pos_log_variance)  # 与标准正态比较。
        kl_prior = scatter_mean(kl_prior, batch, dim=0)  # 按批次平均。
        return kl_prior  # 返回 KL。

    def sample_time(self, num_graphs, device, method):  # 采样扩散时间步。
        if method == 'importance':  # 重要性采样模式。
            if not (self.Lt_count > 10).all():  # 若统计不足。
                return self.sample_time(num_graphs, device, method='symmetric')  # 回退到对称采样。

            Lt_sqrt = torch.sqrt(self.Lt_history + 1e-10) + 0.0001  # 对损失历史开方平滑。
            Lt_sqrt[0] = Lt_sqrt[1]  # Overwrite decoder term with L1.  # 保留注释：避免第 0 项不稳定。
            pt_all = Lt_sqrt / Lt_sqrt.sum()  # 归一化得到时间分布。

            time_step = torch.multinomial(pt_all, num_samples=num_graphs, replacement=True)  # 按分布抽样时间步。
            pt = pt_all.gather(dim=0, index=time_step)  # 记录对应概率。
            return time_step, pt  # 返回抽样结果与概率。

        elif method == 'symmetric':  # 对称采样模式。
            time_step = torch.randint(
                0, self.num_timesteps, size=(num_graphs // 2 + 1,), device=device)  # 随机采样前半部分。
            time_step = torch.cat(
                [time_step, self.num_timesteps - time_step - 1], dim=0)[:num_graphs]  # 构造成对时间步并截断。
            pt = torch.ones_like(time_step).float() / self.num_timesteps  # 使用均匀概率。
            return time_step, pt  # 返回采样结果与概率。

        else:  # 未支持的采样模式。
            raise ValueError  # 抛出异常提示。

    def compute_pos_Lt(self, pos_model_mean, x0, xt, t, batch):  # 计算位置通道的单步变分损失。
        # fixed pos variance  # 保留注释：使用固定的后验方差。
        pos_log_variance = extract(self.posterior_logvar, t, batch)  # 提取对应时间步的对数方差。
        pos_true_mean = self.q_pos_posterior(x0=x0, xt=xt, t=t, batch=batch)  # 计算真实后验均值。
        kl_pos = normal_kl(pos_true_mean, pos_log_variance, pos_model_mean, pos_log_variance)  # 估计 KL 散度。
        kl_pos = kl_pos / np.log(2.)  # 换算为以 2 为底的对数。

        decoder_nll_pos = -log_normal(x0, means=pos_model_mean, log_scales=0.5 * pos_log_variance)  # 计算重建负对数似然。
        assert kl_pos.shape == decoder_nll_pos.shape  # 确认形状一致。
        mask = (t == 0).float()[batch]  # 构造掩码，t=0 时使用重建损失。
        loss_pos = scatter_mean(mask * decoder_nll_pos + (1. - mask) * kl_pos, batch, dim=0)  # 按批次聚合损失。
        return loss_pos  # 返回位置损失。

    def compute_v_Lt(self, log_v_model_prob, log_v0, log_v_true_prob, t, batch):  # 计算离散类型通道的单步损失。
        kl_v = categorical_kl(log_v_true_prob, log_v_model_prob)  # [num_atoms, ]  # 计算 KL 散度。
        decoder_nll_v = -log_categorical(log_v0, log_v_model_prob)  # L0  # 计算重建负对数似然。
        assert kl_v.shape == decoder_nll_v.shape  # 确认形状一致。
        mask = (t == 0).float()[batch]  # 构造掩码选择使用哪种损失。
        loss_v = scatter_mean(mask * decoder_nll_v + (1. - mask) * kl_v, batch, dim=0)  # 聚合得到批次平均。
        return loss_v  # 返回类型损失。

    def get_diffusion_loss(
            self, protein_pos, protein_v, batch_protein, ligand_pos, ligand_v, batch_ligand, time_step=None
    ):  # 计算扩散训练损失。
        num_graphs = batch_protein.max().item() + 1  # 统计图数量。
        protein_pos, ligand_pos, _ = center_pos(
            protein_pos, ligand_pos, batch_protein, batch_ligand, mode=self.center_pos_mode)  # 根据配置中心化坐标。

        # 1. sample noise levels  # 保留注释：步骤 1 采样噪声。
        if time_step is None:  # 若未指定时间步。
            time_step, pt = self.sample_time(num_graphs, protein_pos.device, self.sample_time_method)  # 按策略采样时间。
        else:  # 若外部指定时间步。
            pt = torch.ones_like(time_step).float() / self.num_timesteps  # 使用均匀概率。
        a = self.alphas_cumprod.index_select(0, time_step)  # (num_graphs, )  # 提取 ᾱ_t。

        # 2. perturb pos and v  # 保留注释：步骤 2 扰动位置与类型。
        a_pos = a[batch_ligand].unsqueeze(-1)  # (num_ligand_atoms, 1)  # 将 ᾱ_t 展开到配体节点。
        pos_noise = torch.zeros_like(ligand_pos)  # 初始化位置噪声。
        pos_noise.normal_()  # 采样标准正态噪声。
        # Xt = a.sqrt() * X0 + (1-a).sqrt() * eps  # 保留注释：扩散公式。
        ligand_pos_perturbed = a_pos.sqrt() * ligand_pos + (1.0 - a_pos).sqrt() * pos_noise  # pos_noise * std  # 生成扰动位置。
        # Vt = a * V0 + (1-a) / K  # 保留注释：类别扰动公式。
        log_ligand_v0 = index_to_log_onehot(ligand_v, self.num_classes)  # 将原始类别转为 log one-hot。
        ligand_v_perturbed, log_ligand_vt = self.q_v_sample(log_ligand_v0, time_step, batch_ligand)  # 采样离散扰动。

        # 3. forward-pass NN, feed perturbed pos and v, output noise  # 保留注释：步骤 3 前向网络。
        preds = self(
            protein_pos=protein_pos,  # 输入蛋白坐标。
            protein_v=protein_v,  # 输入蛋白原子特征。
            batch_protein=batch_protein,  # 输入蛋白批次索引。

            init_ligand_pos=ligand_pos_perturbed,  # 输入扰动后的配体坐标。
            init_ligand_v=ligand_v_perturbed,  # 输入扰动后的配体类别。
            batch_ligand=batch_ligand,  # 输入配体批次索引。
            time_step=time_step  # 输入时间步。
        )

        pred_ligand_pos, pred_ligand_v = preds['pred_ligand_pos'], preds['pred_ligand_v']  # 提取网络预测。
        pred_pos_noise = pred_ligand_pos - ligand_pos_perturbed  # 估计预测噪声。
        # atom position  # 保留注释：以下处理位置分支。
        if self.model_mean_type == 'noise':  # 模型预测噪声时。
            pos0_from_e = self._predict_x0_from_eps(
                xt=ligand_pos_perturbed, eps=pred_pos_noise, t=time_step, batch=batch_ligand)  # 反推 x0。
            pos_model_mean = self.q_pos_posterior(
                x0=pos0_from_e, xt=ligand_pos_perturbed, t=time_step, batch=batch_ligand)  # 计算均值。
        elif self.model_mean_type == 'C0':  # 模型预测 x0 时。
            pos_model_mean = self.q_pos_posterior(
                x0=pred_ligand_pos, xt=ligand_pos_perturbed, t=time_step, batch=batch_ligand)  # 直接使用预测。
        else:  # 未知类型。
            raise ValueError  # 抛出异常。

        # atom pos loss  # 保留注释：位置损失计算。
        if self.model_mean_type == 'C0':  # 若预测的是 x0。
            target, pred = ligand_pos, pred_ligand_pos  # 使用位置与预测进行比较。
        elif self.model_mean_type == 'noise':  # 若预测的是噪声。
            target, pred = pos_noise, pred_pos_noise  # 使用噪声与预测比较。
        else:  # 未知类型。
            raise ValueError  # 抛出异常。
        loss_pos = scatter_mean(((pred - target) ** 2).sum(-1), batch_ligand, dim=0)  # 计算 MSE 并按批次聚合。
        loss_pos = torch.mean(loss_pos)  # 对所有图取平均。

        # atom type loss  # 保留注释：类别损失计算。
        log_ligand_v_recon = F.log_softmax(pred_ligand_v, dim=-1)  # 将 logits 转为 log 概率。
        log_v_model_prob = self.q_v_posterior(log_ligand_v_recon, log_ligand_vt, time_step, batch_ligand)  # 计算模型后验。
        log_v_true_prob = self.q_v_posterior(log_ligand_v0, log_ligand_vt, time_step, batch_ligand)  # 计算真实后验。
        kl_v = self.compute_v_Lt(log_v_model_prob=log_v_model_prob, log_v0=log_ligand_v0,
                                 log_v_true_prob=log_v_true_prob, t=time_step, batch=batch_ligand)  # 计算 KL 损失。
        loss_v = torch.mean(kl_v)  # 对 KL 损失取平均。
        loss_v2 = None  # 预留额外损失项。
        if self.loss_v2_weight and self.loss_v2_weight > 0:  # 若启用额外交叉熵损失。
            loss_v2 = F.nll_loss(log_ligand_v_recon, ligand_v)  # 计算 NLL。
        loss = loss_pos + loss_v * self.loss_v_weight  # 组合位置与类型损失。
        if loss_v2 is not None:  # 若额外损失存在。
            loss = loss + self.loss_v2_weight * loss_v2  # 加权累加。

        return {  # 返回损失字典。
            'loss_pos': loss_pos,  # 位置损失。
            'loss_v': loss_v,  # 类型损失。
            'loss_v2': loss_v2 if loss_v2 is not None else torch.tensor(0., device=loss_pos.device),  # 额外损失或零张量。
            'loss': loss,  # 总损失。
            'x0': ligand_pos,  # 原始配体位置。
            'pred_ligand_pos': pred_ligand_pos,  # 预测位置。
            'pred_ligand_v': pred_ligand_v,  # 预测类别 logits。
            'pred_pos_noise': pred_pos_noise,  # 预测噪声。
            'ligand_v_recon': F.softmax(pred_ligand_v, dim=-1)  # 预测类别分布。
        }

    @torch.no_grad()
    def likelihood_estimation(
            self, protein_pos, protein_v, batch_protein, ligand_pos, ligand_v, batch_ligand, time_step
    ):  # 估计对数似然所需的 KL 项。
        protein_pos, ligand_pos, _ = center_pos(
            protein_pos, ligand_pos, batch_protein, batch_ligand, mode='protein')  # 以蛋白中心进行坐标平移。
        assert (time_step == self.num_timesteps).all() or (time_step < self.num_timesteps).all()  # 确保时间步合法。
        if (time_step == self.num_timesteps).all():  # 当 t=T 时直接返回先验 KL。
            kl_pos_prior = self.kl_pos_prior(ligand_pos, batch_ligand)  # 计算位置先验 KL。
            log_ligand_v0 = index_to_log_onehot(batch_ligand, self.num_classes)  # 构造 log one-hot。
            kl_v_prior = self.kl_v_prior(log_ligand_v0, batch_ligand)  # 计算离散先验 KL。
            return kl_pos_prior, kl_v_prior  # 返回先验项。

        # perturb pos and v  # 保留注释：以下对位置与类型进行扰动。
        a = self.alphas_cumprod.index_select(0, time_step)  # (num_graphs, )  # 提取 ᾱ_t。
        a_pos = a[batch_ligand].unsqueeze(-1)  # (num_ligand_atoms, 1)  # 展开到配体节点。
        pos_noise = torch.zeros_like(ligand_pos)  # 初始化噪声。
        pos_noise.normal_()  # 采样标准正态噪声。
        # Xt = a.sqrt() * X0 + (1-a).sqrt() * eps  # 保留注释：扩散公式。
        ligand_pos_perturbed = a_pos.sqrt() * ligand_pos + (1.0 - a_pos).sqrt() * pos_noise  # pos_noise * std  # 生成扰动位置。
        # Vt = a * V0 + (1-a) / K  # 保留注释：类别扰动公式。
        log_ligand_v0 = index_to_log_onehot(ligand_v, self.num_classes)  # 构造 log one-hot 类别。
        ligand_v_perturbed, log_ligand_vt = self.q_v_sample(log_ligand_v0, time_step, batch_ligand)  # 采样扰动类别。

        preds = self(
            protein_pos=protein_pos,  # 蛋白坐标。
            protein_v=protein_v,  # 蛋白特征。
            batch_protein=batch_protein,  # 蛋白批次。

            init_ligand_pos=ligand_pos_perturbed,  # 扰动后的配体坐标。
            init_ligand_v=ligand_v_perturbed,  # 扰动后的配体类别。
            batch_ligand=batch_ligand,  # 配体批次。
            time_step=time_step  # 时间步。
        )

        pred_ligand_pos, pred_ligand_v = preds['pred_ligand_pos'], preds['pred_ligand_v']  # 提取预测输出。
        if self.model_mean_type == 'C0':  # 仅支持模型直接预测 x0 的情况。
            pos_model_mean = self.q_pos_posterior(
                x0=pred_ligand_pos, xt=ligand_pos_perturbed, t=time_step, batch=batch_ligand)  # 计算后验均值。
        else:  # 其他模式未实现。
            raise ValueError  # 抛出异常。

        # atom type  # 保留注释：以下处理离散变量。
        log_ligand_v_recon = F.log_softmax(pred_ligand_v, dim=-1)  # 转换为 log 概率。
        log_v_model_prob = self.q_v_posterior(log_ligand_v_recon, log_ligand_vt, time_step, batch_ligand)  # 计算模型后验。
        log_v_true_prob = self.q_v_posterior(log_ligand_v0, log_ligand_vt, time_step, batch_ligand)  # 计算真实后验。

        # t = [T-1, ... , 0]  # 保留注释：遍历时间序列。
        kl_pos = self.compute_pos_Lt(pos_model_mean=pos_model_mean, x0=ligand_pos,
                                     xt=ligand_pos_perturbed, t=time_step, batch=batch_ligand)  # 计算位置 KL。
        kl_v = self.compute_v_Lt(log_v_model_prob=log_v_model_prob, log_v0=log_ligand_v0,
                                 log_v_true_prob=log_v_true_prob, t=time_step, batch=batch_ligand)  # 计算离散 KL。
        return kl_pos, kl_v  # 返回 KL 项。

    @torch.no_grad()
    def fetch_embedding(self, protein_pos, protein_v, batch_protein, ligand_pos, ligand_v, batch_ligand):  # 获取最终隐藏表示。
        preds = self(
            protein_pos=protein_pos,  # 输入蛋白坐标。
            protein_v=protein_v,  # 输入蛋白特征。
            batch_protein=batch_protein,  # 输入蛋白批次索引。

            init_ligand_pos=ligand_pos,  # 输入配体坐标。
            init_ligand_v=ligand_v,  # 输入配体类型。
            batch_ligand=batch_ligand,  # 输入配体批次索引。
            fix_x=True  # 固定坐标，仅提取特征。
        )
        return preds  # 返回包含隐藏表示的字典。
