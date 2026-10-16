"""Sampling common responsibilities for ligand sampling."""

from __future__ import annotations

import numpy as np
import torch


def _log(logger, level, msg):
    """统一日志输出：有 logger 用 logger，否则 print。"""
    if logger:
        getattr(logger, level)(msg)
    elif level == 'warning':
        print(f'⚠️  {msg}')
    else:
        print(msg)


def safe_repeat_interleave(indices, repeats, device=None):
    """Repeat indices by ``repeats`` using vectorized ops with strict validation.

    This replaces the earlier Python-loop implementation which silently produced
    degenerate results on some CUDA setups.
    """
    if isinstance(device, str):
        device = torch.device(device)
    if device is None and isinstance(indices, torch.Tensor):
        device = indices.device
    if device is None:
        device = torch.device('cpu')

    indices_tensor = torch.as_tensor(indices, dtype=torch.long, device='cpu')
    repeats_tensor = torch.as_tensor(repeats, dtype=torch.long, device='cpu')

    if indices_tensor.numel() != repeats_tensor.numel():
        raise ValueError(
            f"safe_repeat_interleave: indices and repeats size mismatch. "
            f"indices.numel()={indices_tensor.numel()}, repeats.numel()={repeats_tensor.numel()}"
        )

    if torch.any(repeats_tensor < 0):
        raise ValueError(
            f"safe_repeat_interleave: repeats must be non-negative. repeats={repeats_tensor.tolist()}"
        )
    repeats_tensor = torch.clamp(repeats_tensor, min=0)

    total = int(repeats_tensor.sum().item())
    if total == 0:
        return torch.empty(0, dtype=torch.long, device=device)

    result_tensor = torch.repeat_interleave(indices_tensor, repeats_tensor)
    return result_tensor.to(device)


def _validate_batch_indices(batch_tensor, n_data, name, logger=None, context=''):
    """Ensure batch indices cover [0, n_data-1] exactly once per sample."""
    if batch_tensor.numel() == 0:
        raise ValueError(f"{name} is empty in context={context}. n_data={n_data}")

    batch_cpu = batch_tensor.detach().cpu()
    sorted_unique = torch.sort(batch_cpu.unique()).values
    if sorted_unique.numel() != n_data:
        msg = (
            f"{name} unique size mismatch in context={context}. "
            f"expected {n_data} unique entries (0..{n_data-1}), "
            f"got {sorted_unique.numel()}: {sorted_unique.tolist()}"
        )
        if logger:
            logger.error(msg)
        raise ValueError(msg)

    expected = torch.arange(n_data, device=sorted_unique.device)
    if not torch.equal(sorted_unique, expected):
        msg = (
            f"{name} indices incorrect in context={context}. "
            f"expected {expected.tolist()}, got {sorted_unique.tolist()}"
        )
        if logger:
            logger.error(msg)
        raise ValueError(msg)


def unbatch_v_traj(ligand_v_traj, n_data, ligand_cum_atoms):  # 根据累积原子数拆分类别轨迹。
    """按样本原子数量切分联合轨迹，用于还原逐分子的预测序列。

    Args:
        ligand_v_traj: 形状为 `[num_steps, total_atoms, num_types]` 的时间序列列表。
        n_data: 当前批次的样本个数。
        ligand_cum_atoms: 按样本累计的原子数边界（`[0, n1, n1+n2, ...]`）。

    Returns:
        list[list[np.ndarray]]: 长度为 `n_data` 的列表，每项为该样本的逐步类别轨迹。
    """
    all_step_v = [[] for _ in range(n_data)]  # 初始化每个样本的轨迹列表。
    for v in ligand_v_traj:  # 遍历每个时间步的类别分布。
        v_array = v.cpu().numpy()  # 转为 NumPy 数组。
        for k in range(n_data):  # 逐样本切片。
            all_step_v[k].append(v_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]])
    all_step_v = [np.stack(step_v) for step_v in all_step_v]  # num_samples * [num_steps, num_atoms_i]
    return all_step_v  # 返回拆分后的轨迹列表。


def split_tensor_by_counts(tensor, counts):  # 按原子数量拆分张量。
    """依据原子数统计将拼接张量拆分为若干子张量。

    Args:
        tensor: 已按所有样本拼接的张量（`[total_atoms, ...]`）。
        counts: 每个样本对应的原子数量列表。

    Returns:
        list[torch.Tensor]: 逐样本拆分后的张量列表。
    """
    splits = []  # 保存片段。
    start = 0  # 起始索引，用于标记当前样本在拼接张量中的起始位置。
    for count in counts:  # 遍历每个样本原子数。
        end = start + int(count)
        splits.append(tensor[start:end])  # 提取对应片段。
        start = end
    return splits  # 返回拆分结果。


def is_cuda_oom(exc: BaseException) -> bool:
    """True if ``exc`` is a CUDA out-of-memory failure."""
    name = type(exc).__name__.lower().replace('_', '')
    if 'outofmemory' in name:
        return True
    if not isinstance(exc, (RuntimeError, MemoryError)):
        return False
    msg = str(exc).lower()
    return (
        'out of memory' in msg
        or 'cuda oom' in msg
        or 'cuda error: out of memory' in msg
    )


def scaffold_dl_gpu_batch_size(config) -> int:
    """``dynamic.large_step.batch_size`` used by scaffold_dynamic_locked reverse."""
    try:
        dynamic_cfg = config.sample.get('dynamic', {})
        large_cfg = dynamic_cfg.get('large_step', {})
        raw = large_cfg.get('batch_size', 1)
        return max(1, int(raw or 1))
    except Exception:
        return 1


def pack_repeated_protein(data, n_mol, device):
    """Repeat one pocket ``n_mol`` times (same layout as TargetDiff baseline refine)."""
    protein_pos = data.protein_pos.to(device).float()
    protein_v = data.protein_atom_feature.to(device).float()
    n_protein = int(protein_pos.shape[0])
    batch_protein = safe_repeat_interleave(
        torch.arange(n_mol, dtype=torch.long, device=device),
        [n_protein] * n_mol,
        device=device,
    )
    protein_pos_batch = protein_pos.unsqueeze(0).expand(n_mol, -1, -1).reshape(-1, 3)
    feat_dim = protein_v.shape[-1] if protein_v.dim() > 1 else 1
    protein_v_batch = protein_v.unsqueeze(0).expand(n_mol, -1, feat_dim).reshape(-1, feat_dim)
    return protein_pos_batch, protein_v_batch, batch_protein


class MixedRepaintConfig(ValueError):
    """GPU chunk mixes None/non-None RePaint configs or type masks."""


def concat_repaint_cfgs(cfgs):
    """Concatenate per-graph RePaint dicts. All-None → None. Mixed → MixedRepaintConfig."""
    if not cfgs:
        return None
    if all(cfg is None for cfg in cfgs):
        return None
    if any(cfg is None for cfg in cfgs):
        raise MixedRepaintConfig('mixed None/non-None repaint_cfg in GPU chunk')
    type_masks = [cfg.get('type_mask') for cfg in cfgs]
    if any(mask is None for mask in type_masks) and not all(
        mask is None for mask in type_masks
    ):
        raise MixedRepaintConfig('mixed type_mask in GPU chunk')
    first = cfgs[0]
    packed = {
        'x0_pos': torch.cat([cfg['x0_pos'] for cfg in cfgs], dim=0),
        'x0_log_v': torch.cat([cfg['x0_log_v'] for cfg in cfgs], dim=0),
        'pos_mask': torch.cat([cfg['pos_mask'] for cfg in cfgs], dim=0),
        'type_mask': (
            None if type_masks[0] is None
            else torch.cat(type_masks, dim=0)
        ),
        'use_mean_for_discrete': first.get('use_mean_for_discrete', True),
        '_use_dual_mask': first.get('_use_dual_mask', True),
        '_type_context_mode': first.get('_type_context_mode', 'q'),
        '_clean_type_context': first.get('_clean_type_context', False),
    }
    if first.get('fixed_eps_pos') is not None:
        packed['fixed_eps_pos'] = torch.cat(
            [cfg['fixed_eps_pos'] for cfg in cfgs], dim=0
        )
    return packed


def extra_pos_anchor_mask(
    n_total,
    n_locked,
    n_extra,
    extra_anchor_strength,
    extra_anchor_atoms=0,
    *,
    device='cpu',
    fix_scaffold_pos=True,
    distal_strength=0.0,
):
    """Build RePaint pos_mask: scaffold=1, optional extra soft anchors.

    ``extra_anchor_atoms <= 0`` means every extra atom. A positive count
    softer-locks only the first that many extras (scaffold-facing linkers).
    ``distal_strength > 0`` pins the remaining distal extras at a weaker
    strength instead of releasing them (graded anchor: exit tight, far end
    loose).  ``distal_strength=0`` keeps the historical two-tier behavior.
    """
    pos_mask = torch.zeros(int(n_total), device=device)
    n_locked = int(n_locked)
    n_extra = int(n_extra)
    if fix_scaffold_pos and n_locked > 0:
        pos_mask[:n_locked] = 1.0
    strength = float(extra_anchor_strength or 0.0)
    if strength > 0.0 and n_extra > 0:
        n_anchor = int(extra_anchor_atoms or 0)
        extra_start = n_locked
        extra_end = n_locked + n_extra
        if n_anchor <= 0 or n_anchor >= n_extra:
            pos_mask[extra_start:extra_end] = strength
        else:
            pos_mask[extra_start:extra_start + n_anchor] = strength
            distal = float(distal_strength or 0.0)
            if distal > 0.0:
                pos_mask[extra_start + n_anchor:extra_end] = distal
    return pos_mask


def run_gpu_chunks_with_oom_backoff(
    items,
    gpu_batch,
    run_chunk,
    logger=None,
    empty_cache=None,
):
    """Run ``run_chunk(chunk)`` in order; CUDA OOM halves ``gpu_batch`` and retries.

    Yields ``(used_batch, chunk, results)``. ``run_chunk`` must return a list
    with ``len(chunk)`` items. ``batch_size=1`` never retries via this helper
    (the caller already uses the single-sample path).
    """
    if not items:
        return
    gpu_batch = max(1, int(gpu_batch))
    i = 0
    n = len(items)
    while i < n:
        size = min(gpu_batch, n - i)
        chunk = items[i:i + size]
        try:
            results = run_chunk(chunk)
        except Exception as exc:
            if gpu_batch > 1 and is_cuda_oom(exc):
                if empty_cache is not None:
                    empty_cache()
                else:
                    try:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception:
                        pass
                gpu_batch = max(1, gpu_batch // 2)
                _log(
                    logger,
                    'warning',
                    '[ScaffoldDynamicLocked] CUDA OOM; retry reverse with '
                    f'batch_size={gpu_batch}',
                )
                continue
            raise
        if results is None or len(results) != len(chunk):
            raise ValueError(
                'run_chunk returned '
                f'{0 if results is None else len(results)} '
                f'items for chunk of {len(chunk)}'
            )
        yield gpu_batch, chunk, results
        i += len(chunk)

MixedRepaintConfig.__module__ = "scripts.sample_diffusion"
