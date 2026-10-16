"""Sampling pipelines baseline responsibilities for ligand sampling."""

from __future__ import annotations


def run_baseline(args, model, data, config, logger):
    """Run the baseline pipeline and return its result bundle."""
    from diffdynamic.sampling.baseline import sample_diffusion_ligand

    pred_pos, pred_v, pred_pos_traj, pred_v_traj, pred_v0_traj, pred_vt_traj, time_list, init_list = sample_diffusion_ligand(
        model, data, config.sample.num_samples,
        batch_size=args.batch_size, device=args.device,
        num_steps=config.sample.num_steps,
        pos_only=config.sample.pos_only,
        center_pos_mode=config.sample.center_pos_mode,
        sample_num_atoms=config.sample.sample_num_atoms,
        logger=logger
    )
    result = {
        'data': data,
        'pred_ligand_pos': pred_pos,
        'pred_ligand_v': pred_v,
        'pred_ligand_pos_traj': pred_pos_traj,
        'pred_ligand_v_traj': pred_v_traj,
        'pred_ligand_v0_traj': pred_v0_traj,
        'pred_ligand_vt_traj': pred_vt_traj,
        'time': time_list,  # 采样耗时列表。
        'init_ligand_pos': init_list,
        'mode': 'baseline'  # 标记采样模式。
    }

    return result
