"""Geometry and web contracts for interactive scaffold initialization."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml
from fastapi.testclient import TestClient
from rdkit import Chem
from rdkit.Chem import AllChem

from server.api import ScaffoldRequest, app
from server.scaffold_context import preview_context
from utils.scaffold_inputs import (
    custom_attachment_sites, load_reference_molecule, resolve_scaffold_indices,
)
from utils.scaffold_sites import build_extra_atom_positions


@pytest.fixture
def molecule():
    mol = Chem.AddHs(Chem.MolFromSmiles('Cc1ccccc1'))
    AllChem.EmbedMolecule(mol, randomSeed=42)
    return Chem.RemoveHs(mol)


def region(anchor, center, count=3, method='gaussian'):
    return dict(anchor_atom_index=anchor, center_xyz=center, atom_count=count,
                method=method, sigma_axial=1.0, sigma_lateral=1.0)


def test_original_hydrogen_mapping(tmp_path, molecule):
    with_h = Chem.AddHs(molecule, addCoords=True)
    # Explicit hydrogens interleaved with heavy atoms must not shift UI indices.
    reordered = Chem.RenumberAtoms(with_h, list(reversed(range(with_h.GetNumAtoms()))))
    path = tmp_path / 'reference.mol'
    Chem.MolToMolFile(reordered, str(path))
    loaded = load_reference_molecule(path)
    assert loaded.GetNumAtoms() == molecule.GetNumAtoms()
    assert all(a.GetAtomicNum() != 1 for a in loaded.GetAtoms())
    assert loaded.GetAtomWithIdx(0).GetIntProp('_source_atom_index') > 0


def test_manual_mask_never_falls_back(molecule):
    with pytest.raises(ValueError, match='out of range'):
        resolve_scaffold_indices(molecule, {'scaffold_source': 'atom_indices', 'scaffold_atom_indices': '999'})
    with pytest.raises(ValueError, match='does not match'):
        resolve_scaffold_indices(molecule, {'scaffold_source': 'smarts', 'scaffold_smarts': '[Si]'})
    assert resolve_scaffold_indices(molecule, {'scaffold_source': 'atom_indices', 'scaffold_atom_indices': '0,2'}) == [0, 2]
    assert resolve_scaffold_indices(molecule, {'scaffold_source': 'none'}) == []


@pytest.mark.parametrize('methods', [('gaussian',), ('gaussian', 'corridor')])
def test_exact_cloud_budget_and_repeatability(molecule, methods):
    scaffold = resolve_scaffold_indices(molecule, {})
    anchor = scaffold[0]
    xyz = np.asarray(molecule.GetConformer().GetAtomPosition(anchor))
    regions = [region(anchor, (xyz + [4 + i * 5, 0, 0]).tolist(), 3 + i, method)
               for i, method in enumerate(methods)]
    sites = custom_attachment_sites(molecule, scaffold, regions)
    cfg = {'custom_regions': regions, 'max_per_site': 0, 'n_extra_max_clamp': 0}
    kwargs = dict(n_extra=0, attachment_sites=sites, sites_cfg=cfg,
                  fallback_center=torch.zeros(3), device='cpu',
                  protein_positions=torch.zeros((0, 3)))
    a, meta = build_extra_atom_positions(**kwargs, rng=np.random.default_rng(42))
    b, _ = build_extra_atom_positions(**kwargs, rng=np.random.default_rng(42))
    assert torch.equal(a, b)
    assert len(a) == sum(r['atom_count'] for r in regions)
    assert [s['count'] for s in meta['site_allocation']] == [r['atom_count'] for r in regions]
    assert meta['preserved_atom_indices'] == []
    distances = torch.cdist(a, a)
    distances.fill_diagonal_(float('inf'))
    assert distances.min() >= 0.8


def test_crowded_region_has_no_fallback(molecule):
    scaffold = resolve_scaffold_indices(molecule, {})
    xyz = np.asarray(molecule.GetConformer().GetAtomPosition(scaffold[0])) + [3, 0, 0]
    r = region(scaffold[0], xyz.tolist(), count=1)
    r['sigma_axial'] = r['sigma_lateral'] = 0.0001
    sites = custom_attachment_sites(molecule, scaffold, [r])
    with pytest.raises(ValueError, match='cannot place'):
        build_extra_atom_positions(1, sites, {'custom_regions': [r]}, torch.zeros(3),
                                   'cpu', rng=np.random.default_rng(42),
                                   protein_positions=torch.tensor(xyz.reshape(1, 3)))


def test_invalid_anchor_and_center(molecule):
    scaffold = resolve_scaffold_indices(molecule, {})
    xyz = list(molecule.GetConformer().GetAtomPosition(scaffold[0]))
    with pytest.raises(ValueError, match='belong'):
        custom_attachment_sites(molecule, scaffold, [region(0, [1, 2, 3])])
    with pytest.raises(ValueError, match='differ'):
        custom_attachment_sites(molecule, scaffold, [region(scaffold[0], xyz, method='corridor')])


def test_gaussian_may_be_centered_on_anchor(molecule):
    scaffold = resolve_scaffold_indices(molecule, {})
    xyz = list(molecule.GetConformer().GetAtomPosition(scaffold[0]))
    r = region(scaffold[0], xyz)
    sites = custom_attachment_sites(molecule, scaffold, [r])
    positions, _ = build_extra_atom_positions(
        3, sites, {'custom_regions': [r]}, torch.zeros(3), 'cpu',
        rng=np.random.default_rng(42), protein_positions=torch.zeros((0, 3)),
    )
    assert positions.shape == (3, 3) and torch.isfinite(positions).all()


def test_changed_manual_scaffold_does_not_reuse_stale_site_cache(tmp_path, molecule):
    from utils.scaffold_sites import load_or_extract_attachment_sites
    cache = tmp_path / 'scaffold' / 'reference_murcko_sites.json'
    cache.parent.mkdir()
    cache.write_text(json.dumps({'attachment_sites': [{'site_id': 999}]}))
    scaffold = resolve_scaffold_indices(molecule, {})
    sites = load_or_extract_attachment_sites(
        molecule, scaffold, 'atom_indices', molecule.GetConformer().GetPositions(),
        tmp_path, 'reference', {'save_json': True, 'reuse_cache': False},
    )
    assert sites and all(s['anchor_scaffold_idx'] in scaffold for s in sites)
    assert all(s['site_id'] != 999 for s in sites)


@pytest.mark.parametrize('kwargs', [
    {'workflow': 'repaint', 'scaffold_mode': 'grow'},
    {'placement': {'mode': 'custom', 'regions': []}},
    {'placement': {'mode': 'custom', 'regions': [region(1, [1, 2, 3], 30), region(1, [2, 3, 4], 30)]}},
    {'placement': {'mode': 'custom', 'regions': [region(1, [float('inf'), 2, 3])]}},
    {'protein_path': '/pocket.pdb'},
])
def test_invalid_request_rejected(kwargs):
    with pytest.raises(ValueError):
        ScaffoldRequest(**kwargs)


def test_custom_input_removes_dataset_id():
    req = ScaffoldRequest(protein_path='pocket.pdb', molecule_path='ligand.sdf', data_id=0)
    assert req.data_id is None
    assert not req.use_test_set


def test_repaint_repair_is_independent_and_defaults_off():
    assert ScaffoldRequest().scaffold_enable_refine
    assert not ScaffoldRequest(scaffold_mode='prudent', workflow='repaint').scaffold_enable_refine
    assert ScaffoldRequest(scaffold_mode='dynamic_locked', workflow='repaint',
                           scaffold_enable_refine=True).scaffold_enable_refine


def test_world_coordinate_tuples_are_portable_yaml(monkeypatch, tmp_path):
    from server import job_helpers
    monkeypatch.setattr(job_helpers, 'get_config', lambda: SimpleNamespace(diffdynamic_root=str(tmp_path)))
    base = tmp_path / 'sampling.yml'
    base.write_text('sample: {}\n')
    placement = ScaffoldRequest(placement={'mode': 'custom', 'regions': [region(1, [0, 2, 3])]}).dict()['placement']
    path = job_helpers.write_patched_config(str(base), {'sample': {'placement': placement}}, 'portable')
    saved = yaml.safe_load(open(path))
    assert saved['sample']['placement']['regions'][0]['center_xyz'] == [0, 2, 3]


def test_generated_results_readable_but_uploaded_objects_stay_strict(monkeypatch, tmp_path):
    import server.molecule_ingest as ingest
    from server.data_manager import load_pt_metadata
    root = tmp_path / 'outputs'
    from server.auth import owner_slug
    own = root / 'users' / owner_slug('alice')
    own.mkdir(parents=True)
    monkeypatch.setattr(ingest, 'OUTPUTS_ROOT', str(root))
    content = {'data': SimpleNamespace(source='trusted sampler'),
               'pred_ligand_pos': [np.zeros((2, 3)), np.zeros((5, 3))],
               'meta': {'records': [{'is_original': True}, {'status': 'ok', 'n_extra': 3}]}}
    result = own / 'result.pt'
    torch.save(content, str(result))
    metadata = load_pt_metadata(str(result), owner='alice')
    assert metadata['num_molecules'] == 2 and metadata['num_generated'] == 1
    assert metadata['samples'] == [{'status': 'ok', 'n_extra': 3}]
    uploaded = tmp_path / 'uploads' / 'same.pt'
    uploaded.parent.mkdir()
    torch.save(content, str(uploaded))
    # Inert data containers and arrays now normalize through patched PyTorch
    # in the isolated loader instead of failing in the legacy environment.
    assert load_pt_metadata(str(uploaded), owner='alice')['num_molecules'] == 2


def test_preview_uses_shared_geometry_without_scheduler(monkeypatch, molecule):
    import server.scaffold_context as context
    scaffold = resolve_scaffold_indices(molecule, {})
    monkeypatch.setattr(context, 'load_context', lambda p: (molecule, scaffold, torch.zeros((0, 3)), {}))
    xyz = np.asarray(molecule.GetConformer().GetAtomPosition(scaffold[0])) + [4, 0, 0]
    params = ScaffoldRequest(placement={'mode': 'custom', 'regions': [region(scaffold[0], xyz.tolist())]}).dict()
    monkeypatch.setattr('server.api.get_scheduler', lambda: pytest.fail('Preview must not allocate a scheduler'))
    with TestClient(app) as client:
        client.get('/app')
        response = client.post('/api/scaffold/preview', json=params)
    assert response.status_code == 200
    direct = preview_context(params)
    assert response.json()['positions'] == direct['positions']
    assert len(direct['positions']) == 3


@pytest.mark.parametrize('mode', ['grow', 'dynamic_locked'])
def test_no_exit_preview_matches_seeded_sampling_fallback(monkeypatch, molecule, mode):
    import server.scaffold_context as context
    scaffold = list(range(molecule.GetNumAtoms()))
    protein = torch.tensor([[20., 20., 20.], [24., 20., 20.]])
    monkeypatch.setattr(context, 'load_context', lambda p: (molecule, scaffold, protein, {}))
    params = ScaffoldRequest(scaffold_mode=mode, grow_n_extra_mode='fixed',
                             grow_n_extra_fixed=11, seed=0).dict()
    a = preview_context(params)
    torch.randn(100)
    b = preview_context(params)
    assert a['positions'] == b['positions'] and len(a['positions']) == 11
    positions, _ = build_extra_atom_positions(
        11, [], a['effective_parameters']['placement'], protein.mean(dim=0), 'cpu',
        fallback_noise_scale=2.0 if mode == 'grow' else 1.0,
        rng=np.random.default_rng(0), protein_positions=protein,
    )
    assert positions.tolist() == a['positions']


@pytest.mark.parametrize('mode', ['pocket_prior', 'range'])
def test_web_count_sampling_is_independent_of_global_rng(mode):
    from scripts.sample_diffusion import _sample_n_extra_atoms
    protein = torch.tensor([[20., 20., 20.], [24., 20., 20.]])
    cfg = {'web_input_contract': True, 'seed': 0, 'n_extra_mode': mode,
           'n_extra_min': 5, 'n_extra_max': 40, 'n_extra_min_clamp': 0, 'n_extra_max_clamp': 50}
    a = _sample_n_extra_atoms(cfg, protein, 7, 1000)
    np.random.random(100)
    assert a == _sample_n_extra_atoms(cfg, protein, 7, 1000)


def test_scheduler_effective_snapshot_and_replay(monkeypatch, tmp_path):
    from server.database import init_db, get_run
    from server.scheduler import JobScheduler
    import server.scheduler as scheduler_module
    init_db()
    scheduler = JobScheduler(max_concurrent=1)
    monkeypatch.setattr(scheduler, '_acquire_or_queue', lambda gpu, rec, starter: {'run_id': rec.run_id})
    def write(base, patches, job):
        data = yaml.safe_load(open(base))
        from server.job_helpers import deep_merge
        deep_merge(data, patches)
        path = tmp_path / (job + '.yml')
        path.write_text(yaml.safe_dump(data))
        return str(path)
    monkeypatch.setattr(scheduler, '_write_patched_config', write)
    r = scheduler.submit_scaffold(scaffold_mode='dynamic_locked', workflow='repaint',
                                  grow_num_samples=5, seed=0, scaffold_enable_refine=False,
                                  triggered_by='admin', skip_daily_quota=True)
    saved = get_run(r['run_id'])
    effective = yaml.safe_load(saved.config_snapshot)
    sc = effective['sample']['scaffold']
    assert sc['fix_scaffold_type']
    assert sc['grow']['seed'] == 0
    assert sc['grow']['start_t'] == 999 and sc['grow']['forward_noise_init']
    assert effective['sample']['dynamic']['refine']['max_grad_fusion_iterations'] == 30
    assert not effective['sample']['targetdiff_baseline_refine']['enable']
    replay = scheduler.submit_rerun(saved.id, triggered_by='admin', skip_daily_quota=True)
    again = json.loads(get_run(replay['run_id']).parameters)['scaffold_request']
    assert again['seed'] == 0 and again['workflow'] == 'repaint'
    assert again['grow_num_samples'] == 5


def test_runner_prefers_custom_protein():
    from server.runners.scaffold import _build_sample_cmd
    cfg = SimpleNamespace(python_bin='python', sample_script=lambda: 'sample.py')
    cmd = _build_sample_cmd(cfg, 'config.yml', {'gpu_id': 0, 'data_id': 0,
                                             'protein_path': 'p.pdb', 'ligand_path': 'l.sdf'})
    assert '--protein_path' in cmd and '-i' not in cmd


def test_prudent_fill_uses_custom_regions_and_forward_noise(monkeypatch, molecule):
    import scripts.sample_diffusion as sampling
    from datasets.pl_data import ProteinLigandData
    scaffold = resolve_scaffold_indices(molecule, {})
    reference = torch.tensor(molecule.GetConformer().GetPositions(), dtype=torch.float32)
    r = region(scaffold[0], (reference[scaffold[0]] + torch.tensor([4., 0., 0.])).tolist())
    sites = custom_attachment_sites(molecule, scaffold, [r])
    data = ProteinLigandData(protein_pos=torch.tensor([[50., 50., 50.]]),
                             protein_element=torch.tensor([6]),
                             protein_atom_feature=torch.ones((1, 2)),
                             ligand_element=torch.tensor([6] * len(reference)))
    seen = {}
    class ReverseReached(Exception):
        pass
    class Model:
        num_classes = 2
        num_timesteps = 1000
        def sample_diffusion_large_step(self, **kwargs):
            seen['reverse'] = kwargs
            raise ReverseReached()
    def forward(model, pos, log_v, batch, timestep, device):
        seen['clean'] = pos.clone()
        seen['timestep'] = timestep
        return pos + 100., log_v
    monkeypatch.setattr(sampling, '_forward_diffuse_molecule', forward)
    with pytest.raises(ReverseReached):
        sampling._prudent_locked_sample_once(
            model=Model(), data=data, device='cpu',
            grow_cfg={'seed': 42, 'n_extra_mode': 'fixed', 'n_extra_fixed': 3,
                      'forward_noise_init': True, 'start_t': 999},
            _attachment_sites=sites, _murcko_sites_cfg={'custom_regions': [r]},
            protein_centroid=data.protein_pos[0], ref_ligand_pos=reference,
            log_ref_v=torch.log(torch.full((len(reference), 2), 0.5)),
            scaffold_indices=scaffold, n_scaffold=len(scaffold), scaffold_bond_pairs=[],
            fix_scaffold_pos=True, fix_scaffold_type=True, log_mode_is_log_prob=True,
            large_cfg={}, refine_cfg={}, time_boundary=650, center_pos_mode='protein',
            pos_only=False, ligand_atom_mode='add_aromatic', selector_cfg={},
            sample_idx=0, slot_retry=0,
        )
    reverse = seen['reverse']
    assert seen['timestep'] == 999
    assert len(reverse['init_ligand_pos']) == len(scaffold) + 3
    assert torch.allclose(reverse['init_ligand_pos'], seen['clean'] + 150.)
    masks = reverse['repaint_cfg']
    assert masks['pos_mask'][:len(scaffold)].eq(1).all()
    assert masks['type_mask'][:len(scaffold)].eq(1).all()
    assert masks['pos_mask'][len(scaffold):].eq(0).all()
    assert masks['type_mask'][len(scaffold):].eq(0).all()
