"""Scaffold dual-mask contract: lock types without locking extra positions."""
from __future__ import annotations

import torch

from utils.masked_guidance_sampling import dual_mask_repaint_blend_states


def test_extra_type_lock_does_not_lock_extra_position():
    ligand_pos = torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
    ref_pos = torch.tensor([[10.0, 10.0, 10.0], [20.0, 20.0, 20.0]])
    ligand_log_v = torch.log(torch.tensor([[0.2, 0.8], [0.7, 0.3]]))
    ref_log_v = torch.log(torch.tensor([[0.9, 0.1], [0.1, 0.9]]))

    pos, log_v = dual_mask_repaint_blend_states(
        ligand_pos,
        ligand_log_v,
        ref_pos,
        ref_log_v,
        pos_mask=torch.tensor([1.0, 0.0]),
        type_mask=torch.tensor([1.0, 1.0]),
    )

    assert torch.equal(pos[0], ref_pos[0])
    assert torch.equal(pos[1], ligand_pos[1])
    assert torch.allclose(log_v, ref_log_v, atol=1e-6)
