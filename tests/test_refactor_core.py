"""CPU regression contracts captured before the source-package migration.

The numerical fixtures below came from the original monolithic model with a
small network and fixed seeds. They require neither pretrained weights nor a
GPU, and catch parameter registration, initialization and sampling changes.
"""

import hashlib
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from easydict import EasyDict

import _diffdynamic_bootstrap  # noqa: F401

from diffdynamic.data.graph import ProteinLigandData
from diffdynamic.models.constants import GRAD_FUSION_CAP_UNSPECIFIED
from diffdynamic.models.score_model import DiffDynamic, ScorePosNet3D
from diffdynamic.sampling.arguments import build_parser
from diffdynamic.sampling.configuration import _dynamic_subcfg_grad_cap_kwargs
from diffdynamic.sampling.prudent.registry import round_helpers


LEGACY_LAYOUT_SHA256 = (
    'd2d799327f973c03d26b3506c672c6e1dc201e6d3b2eee38e9725d3394bbe5eb'
)
LEGACY_FORWARD_POS = [
    [1.000718355178833, 1.000718355178833, -0.024773597717285156],
    [1.015993356704712, 1.015993356704712, 1.0612537860870361],
]
LEGACY_FORWARD_TYPES = [
    [0.2260294259, -0.0730460659, -0.0531936400, -0.0168801583,
     0.1179044023, 0.1367883384, -0.3173263669, 0.2197804749],
    [0.1727843732, -0.1698158681, -0.0632761866, -0.0055749360,
     0.0497290641, 0.0424513072, -0.2980354428, 0.1213339865],
]
LEGACY_SAMPLE_POS = [
    [0.9826194047927856, 1.0440638065338135, -0.045458387583494186],
    [0.9333292245864868, 0.9183319807052612, 1.0116856098175049],
]


def small_model_config():
    """Keep this fixture independent of user and research YAML presets."""
    return EasyDict(
        name='score', model_mean_type='C0', beta_schedule='sigmoid',
        beta_start=1e-7, beta_end=.002, v_beta_schedule='cosine', v_beta_s=.01,
        num_diffusion_timesteps=20, loss_v_weight=100.,
        sample_time_method='symmetric', time_emb_dim=0, time_emb_mode='simple',
        center_pos_mode='protein', ligand_v_input='auto', use_grad_fusion=False,
        grad_fusion_lambda=.5, loss_v2_weight=0., dynamic_pos_step_clip=None,
        dynamic_v_step_clip=None,
        dynamic_large_step={'stride': 50, 'step_size': 1., 'noise_scale': 0.},
        dynamic_refine={'stride': 10, 'step_size': .2, 'noise_scale': 0.,
                        'time_upper': 200},
        node_indicator=True, model_type='uni_o2', num_blocks=1, num_layers=1,
        hidden_dim=16, n_heads=4, edge_feat_dim=4, num_r_gaussian=20, knn=3,
        num_node_types=8, act_fn='relu', norm=True, cutoff_mode='knn',
        ew_net_type='global', num_x2h=1, num_h2x=1, r_max=10.,
        x2h_out_fc=False, sync_twoup=False,
    )


def small_inputs():
    return {
        'protein_pos': torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]),
        'protein_v': torch.ones((3, 3)),
        'batch_protein': torch.zeros(3, dtype=torch.long),
        'init_ligand_pos': torch.tensor([[1., 1., 0.], [1., 1., 1.]]),
        'init_ligand_v': torch.tensor([0, 1]),
        'batch_ligand': torch.zeros(2, dtype=torch.long),
    }


@pytest.mark.parametrize('model_class', [ScorePosNet3D, DiffDynamic])
def test_legacy_checkpoint_layout_and_initialization_rng(model_class):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(71)
        model = model_class(small_model_config(), 3, 8).eval()
        layout = [(key, list(value.shape))
                  for key, value in model.state_dict().items()]
        digest = hashlib.sha256(
            json.dumps(layout, separators=(',', ':')).encode()
        ).hexdigest()
        assert digest == LEGACY_LAYOUT_SHA256
        assert len(layout) == 88
        assert sum(p.numel() for p in model.parameters()) == 15868
        assert torch.randint(0, 2**31, (4,)).tolist() == [
            1810537452, 379695496, 760470353, 351223341,
        ]
        restored = model_class(small_model_config(), 3, 8)
        assert not restored.load_state_dict(model.state_dict(), strict=True).missing_keys


@pytest.mark.parametrize('model_class', [ScorePosNet3D, DiffDynamic])
def test_legacy_cpu_forward_values(model_class):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(71)
        model = model_class(small_model_config(), 3, 8).eval()
        with torch.no_grad():
            output = model(**small_inputs(), time_step=torch.tensor([5]))
        assert torch.allclose(output['pred_ligand_pos'],
                              torch.tensor(LEGACY_FORWARD_POS), atol=1e-7, rtol=0)
        assert torch.allclose(output['pred_ligand_v'],
                              torch.tensor(LEGACY_FORWARD_TYPES), atol=1e-7, rtol=0)


@pytest.mark.parametrize('model_class', [ScorePosNet3D, DiffDynamic])
def test_legacy_cpu_sampling_values_and_rng(model_class):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(71)
        model = model_class(small_model_config(), 3, 8).eval()
        torch.manual_seed(123)
        output = model.sample_diffusion(
            **small_inputs(), num_steps=3, center_pos_mode='protein',
        )
        assert torch.allclose(output['pos'], torch.tensor(LEGACY_SAMPLE_POS),
                              atol=1e-7, rtol=0)
        assert output['v'].tolist() == [3, 0]
        assert len(output['pos_traj']) == 3
        assert torch.randint(0, 2**31, (4,)).tolist() == [
            1362730766, 2135924991, 1781665012, 787108199,
        ]


def test_legacy_class_import_and_pickle_identity():
    from datasets.pl_data import ProteinLigandData as old_data
    from models.molopt_score_model import ScorePosNet3D as old_model

    assert old_data is ProteinLigandData
    assert old_model is ScorePosNet3D
    data = ProteinLigandData(protein_pos=torch.ones((2, 3)))
    serialized = pickle.dumps(data)
    assert b'datasets.pl_data' in serialized
    restored = pickle.loads(serialized)
    assert type(restored) is ProteinLigandData
    assert torch.equal(restored.protein_pos, data.protein_pos)


def test_legacy_private_monkeypatch_reaches_extracted_scoring(monkeypatch):
    import scripts.sample_diffusion as legacy
    from diffdynamic.sampling.prudent.reconstruction import (
        _prudent_canonical_smiles as original,
    )
    from diffdynamic.sampling.prudent.scoring import _prudent_dedup_survivors_smiles

    monkeypatch.setattr(legacy, '_prudent_canonical_smiles',
                        lambda *args, **kwargs: 'CC')
    candidates = [(float(3-i), {}, {'pos': [], 'v': []}, i) for i in range(3)]
    survivors, dropped = _prudent_dedup_survivors_smiles(candidates, 3)
    assert survivors == candidates[:1]
    assert dropped == 2
    assert legacy._prudent_canonical_smiles is not original


def test_round_registry_uses_live_explicit_dependencies(monkeypatch):
    import scripts.sample_diffusion as legacy

    replacement = lambda *args, **kwargs: []
    monkeypatch.setattr(legacy, '_prudent_refine_batch', replacement)
    helpers = round_helpers()
    assert set(helpers) == {
        '_prudent_copy_candidate', '_prudent_parse_size_gates',
        '_prudent_refine_batch', '_prudent_score_all_chains',
        '_prudent_select_survivors', 'apply_targetdiff_baseline_refinement',
    }
    assert helpers['_prudent_refine_batch'] is replacement


def test_gradient_cap_omission_and_null_keep_distinct_semantics():
    import models.molopt_score_model as legacy

    assert legacy.GRAD_FUSION_CAP_UNSPECIFIED is GRAD_FUSION_CAP_UNSPECIFIED
    omitted = _dynamic_subcfg_grad_cap_kwargs({})
    explicit_null = _dynamic_subcfg_grad_cap_kwargs(
        {'max_grad_fusion_iterations': None},
    )
    assert omitted['max_grad_fusion_iterations'] is GRAD_FUSION_CAP_UNSPECIFIED
    assert explicit_null['max_grad_fusion_iterations'] is None


def test_legacy_sampling_parser_defaults_and_aliases():
    args = build_parser().parse_args(['sample.yml'])
    assert (args.batch_size, args.device, args.result_path) == (
        100, 'cuda:0', './outputs/pt',
    )
    args = build_parser().parse_args(
        ['sample.yml', '-i', '7', '-gpu', '2', '--mode', 'prudent'],
    )
    assert (args.data_id, args.gpu_id, args.mode) == (7, 2, 'prudent')


def test_legacy_cli_help_from_another_working_directory(tmp_path):
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
    result = subprocess.run(
        [sys.executable, str(root / 'scripts/sample_diffusion.py'), '--help'],
        cwd=str(tmp_path), env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert '--resume-from-pt' in result.stdout
    assert '--protein_path' in result.stdout
