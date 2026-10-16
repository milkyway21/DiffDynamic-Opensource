"""Score_model components of the DiffDynamic model."""

from __future__ import annotations

from diffdynamic.models.diffusion import DiffusionTrainingMixin
from diffdynamic.models.dynamics import DynamicDiffusionMixin
from diffdynamic.models.network import ScoreNetwork
from diffdynamic.models.samplers import DiffusionSamplingMixin
from diffdynamic.models.schedules import ScheduleMixin
import torch
from diffdynamic.models.constants import GRAD_FUSION_CAP_UNSPECIFIED


class ScorePosNet3D(DiffusionTrainingMixin, DiffusionSamplingMixin,
                    DynamicDiffusionMixin, ScheduleMixin, ScoreNetwork):
    """Joint coordinate/type diffusion with legacy checkpoint registration."""


class DiffDynamic(ScorePosNet3D):  # 定义 DiffDynamic 模型，继承 ScorePosNet3D。

    def __init__(self, config, protein_atom_feature_dim, ligand_atom_feature_dim):  # 初始化 DiffDynamic 模型。
        if not hasattr(config, 'ligand_v_input'):  # 若配置缺少 ligand_v_input。
            config.ligand_v_input = 'log_prob'  # 默认采用 log 概率输入。
        if not hasattr(config, 'use_grad_fusion'):  # 若未指定是否使用梯度融合。
            config.use_grad_fusion = True  # 默认开启梯度融合。
        if not hasattr(config, 'grad_fusion_lambda'):  # 若未提供融合调度。
            config.grad_fusion_lambda = {'mode': 'linear', 'start': 0.8, 'end': 0.2}  # 使用线性调度默认值。
        if not hasattr(config, 'loss_v2_weight'):  # 若未指定第二类别损失权重。
            config.loss_v2_weight = 1.0  # 默认开启额外类别损失。
        super().__init__(config, protein_atom_feature_dim, ligand_atom_feature_dim)  # 调用父类构造函数。
        # 注意：_build_lambda_schedule 方法已从父类 ScorePosNet3D 继承，无需重复定义。

    @torch.no_grad()
    def dynamic_sample_diffusion(self, protein_pos, protein_v, batch_protein,
                                 init_ligand_pos, init_log_ligand_v, batch_ligand,
                                 num_steps=None, center_pos_mode=None, pos_only=False):
        if num_steps is None:  # 若未指定步数。
            num_steps = self.num_timesteps  # 默认使用全部时间步。
        if center_pos_mode is None:  # 若未指定中心化模式。
            center_pos_mode = self.center_pos_mode  # 使用默认配置。

        large_cfg = self.dynamic_large_step_defaults or {}  # 读取大步采样配置。
        refine_cfg = self.dynamic_refine_defaults or {}  # 读取精炼配置。

        large_schedule = large_cfg.get('schedule', 'lambda')  # 获取大步调度类型。
        large_stride = None if large_schedule in ('lambda', 'linear') else large_cfg.get('stride')  # 确定大步步长（lambda/linear 调度时不需要 stride）。
        large_res = self.sample_diffusion_large_step(
            protein_pos=protein_pos,  # 蛋白坐标。
            protein_v=protein_v,  # 蛋白特征。
            batch_protein=batch_protein,  # 蛋白批次。
            init_ligand_pos=init_ligand_pos,  # 初始配体坐标。
            init_ligand_v=init_log_ligand_v,  # 初始配体 log 概率。
            batch_ligand=batch_ligand,  # 配体批次。
            num_steps=num_steps,  # 总步数。
            center_pos_mode=center_pos_mode,  # 中心化模式。
            pos_only=pos_only,  # 是否仅更新位置。
            step_stride=large_stride,  # 步长。
            step_size=large_cfg.get('step_size'),  # 步幅。
            add_noise=large_cfg.get('noise_scale'),  # 噪声强度。
            pos_clip=large_cfg.get('pos_clip'),  # 位置裁剪。
            v_clip=large_cfg.get('v_clip'),  # 类别裁剪。
            log_ligand_input_mode='log_prob',  # 大步阶段固定使用 log 概率模式。
            max_grad_fusion_iterations=(
                large_cfg['max_grad_fusion_iterations']
                if 'max_grad_fusion_iterations' in large_cfg
                else GRAD_FUSION_CAP_UNSPECIFIED
            ),
            max_gradient_steps=(
                large_cfg['max_gradient_steps']
                if 'max_gradient_steps' in large_cfg
                else GRAD_FUSION_CAP_UNSPECIFIED
            ),
        )

        refine_schedule = refine_cfg.get('schedule', 'lambda')  # 获取精炼调度类型。
        refine_stride = None if refine_schedule in ('lambda', 'linear') else refine_cfg.get('stride')  # 精炼步长（lambda/linear 调度时不需要 stride）。
        refine_res = self.sample_diffusion_refinement(
            protein_pos=protein_pos,  # 蛋白坐标。
            protein_v=protein_v,  # 蛋白特征。
            batch_protein=batch_protein,  # 蛋白批次。
            init_ligand_pos=large_res['pos'],  # 以上阶段输出的位置作为起点。
            init_ligand_v=large_res['log_v'],  # 使用大步阶段输出的 log 概率。
            batch_ligand=batch_ligand,  # 配体批次。
            center_pos_mode=center_pos_mode,  # 中心化模式。
            pos_only=pos_only,  # 是否仅更新位置。
            step_stride=refine_stride,  # 精炼步长。
            step_size=refine_cfg.get('step_size'),  # 精炼步幅。
            add_noise=refine_cfg.get('noise_scale'),  # 精炼噪声。
            pos_clip=refine_cfg.get('pos_clip'),  # 位置裁剪。
            v_clip=refine_cfg.get('v_clip'),  # 类别裁剪。
            time_upper=refine_cfg.get('time_upper'),  # 精炼上界。
            time_lower=refine_cfg.get('time_lower', 0),  # 精炼下界。
            num_cycles=refine_cfg.get('cycles', 1),  # 精炼循环次数。
            log_ligand_input_mode='log_prob',  # 精炼阶段同样使用 log 概率。
            max_grad_fusion_iterations=(
                refine_cfg['max_grad_fusion_iterations']
                if 'max_grad_fusion_iterations' in refine_cfg
                else GRAD_FUSION_CAP_UNSPECIFIED
            ),
            max_gradient_steps=(
                refine_cfg['max_gradient_steps']
                if 'max_gradient_steps' in refine_cfg
                else GRAD_FUSION_CAP_UNSPECIFIED
            ),
        )

        pos_traj = []  # 汇总位置轨迹。
        log_v_traj = []  # 汇总类别轨迹。
        # 用 is not None 判断，避免仅因 key 缺失而丢 large 段；空 list 的 extend 为无操作
        lt = large_res.get('pos_traj')
        if lt is not None:
            pos_traj.extend(lt)
        rt = refine_res.get('pos_traj')
        if rt is not None:
            pos_traj.extend(rt)

        ll = large_res.get('log_v_traj')
        if ll is not None:
            log_v_traj.extend(ll)
        rl = refine_res.get('log_v_traj')
        if rl is not None:
            log_v_traj.extend(rl)

        return {  # 返回采样结果及元数据。
            'pred_ligand_pos': refine_res['pos'],  # 最终位置预测。
            'pred_ligand_v': refine_res['log_v'],  # 最终 log 概率。
            'pos_traj': pos_traj,  # 综合位置轨迹。
            'log_v_traj': log_v_traj,  # 综合类别轨迹。
            'meta': {  # 元信息。
                'large_step_time_indices': large_res.get('time_indices'),  # 大步阶段时间调度。
                'refine_time_indices': refine_res.get('time_indices')  # 精炼阶段时间调度。
            }
        }

ScorePosNet3D.__module__ = "models.molopt_score_model"
DiffDynamic.__module__ = "models.molopt_score_model"
