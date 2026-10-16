"""Local structure previews, access boundaries and explicit atom budgets."""

from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
import yaml
from fastapi.testclient import TestClient
from rdkit import Chem
from rdkit.Chem import AllChem

from server import api
from server.auth import PlatformIdentity, owner_slug
from server.config import ServerConfig
from server.database import get_run, init_db
from server.generation_context import load_generation_context
from server.scaffold_context import dataset_protein_block, preview_context
from utils.data import PDBProtein
from utils.scaffold_inputs import resolve_scaffold_indices


@pytest.fixture
def structures(tmp_path):
    tmp_path = tmp_path / 'data'
    tmp_path.mkdir(exist_ok=True)
    mol = Chem.AddHs(Chem.MolFromSmiles('Cc1ccccc1'))
    AllChem.EmbedMolecule(mol, randomSeed=42)
    mol = Chem.RemoveHs(mol)
    ligand = tmp_path / 'reference.sdf'
    Chem.MolToMolFile(mol, str(ligand))
    protein = tmp_path / 'protein.pdb'
    lines = []
    for residue in range(1, 12):
        center = (residue - 6) * 3.8
        for name, dx, dy, element in [('N', -1.2, 0, 'N'), ('CA', 0, 0, 'C'),
                                      ('C', 1.3, 0, 'C'), ('O', 1.6, 1, 'O')]:
            lines.append(
                f'ATOM  {len(lines) + 1:5d} {name:^4s} ALA A{residue:4d}    '
                f'{center + dx:8.3f}{dy:8.3f}{0.:8.3f}'
                f'  1.00  0.00          {element:>2s}'
            )
    protein.write_text('\n'.join(lines) + '\nEND\n')
    return protein, ligand, mol


def test_ligand_preview_preserves_graph_pose_and_crop(structures):
    protein, ligand, mol = structures
    params = {'protein_path': str(protein), 'ligand_path': str(ligand),
              'pocket_radius': 4.0}
    response = load_generation_context(params)
    rendered = Chem.MolFromMolBlock(response['ligand_sdf'], removeHs=False)
    assert Chem.MolToSmiles(rendered) == Chem.MolToSmiles(mol)
    assert np.allclose(rendered.GetConformer().GetPositions(),
                       mol.GetConformer().GetPositions(), atol=0.0001)
    full = PDBProtein(str(protein))
    selected = full.query_residues_ligand(
        {'pos': mol.GetConformer().GetPositions().astype(np.float32)}, 4.0,
    )
    expected = sorted(i for residue in selected for i in residue['atoms'])
    shown = PDBProtein(response['protein_pdb'], mode='block')
    assert np.array_equal(shown.pos, np.asarray(full.pos)[expected])
    assert response['pocket_cropped']
    assert response['scaffold_indices'] == []
    assert 'positions' not in response  # Reference ligand, no initial atom cloud.
    wider = load_generation_context({**params, 'pocket_radius': 20.0})
    assert len(PDBProtein(wider['protein_pdb'], mode='block').pos) > len(shown.pos)


def test_protein_only_preview_does_not_invent_a_ligand(structures):
    protein, _, _ = structures
    response = load_generation_context({'protein_path': str(protein)})
    assert response['protein_pdb'] == protein.read_text()
    assert response['ligand_sdf'] is None and response['atoms'] == []
    assert not response['pocket_cropped']


def test_dataset_preview_reuses_sampling_split_without_a_scaffold(monkeypatch):
    import server.scaffold_context as context
    seen = {}

    def load(params):
        seen.update(params)
        return None, [], None, {'ligand_sdf': 'fixture', 'scaffold_indices': [1]}

    monkeypatch.setattr(context, 'load_context', load)
    response = load_generation_context({'data_id': 17})
    assert seen == {'data_id': 17, 'scaffold_source': 'none'}
    assert response['data_id'] == 17 and response['scaffold_indices'] == []


def test_missing_pocket_uses_matching_receptor_residues(structures):
    protein, _, _ = structures
    parsed = PDBProtein(str(protein))
    receptor = protein.with_name('target_rec.pdb')
    receptor.write_text(protein.read_text())
    data = SimpleNamespace(protein_pos=torch.tensor(np.asarray(parsed.pos)[12:24]))
    block = dataset_protein_block(data, protein.with_name('target_rec_lig_pocket10.pdb'))
    shown = PDBProtein(block, mode='block')
    assert np.array_equal(shown.pos, data.protein_pos.numpy())
    assert shown.atom_name.count('CA') == 3


def test_missing_pdb_keeps_dataset_backbone_and_chain_breaks(structures):
    protein, _, _ = structures
    parsed = PDBProtein(str(protein))
    keep = list(range(8)) + list(range(16, 24))
    data = SimpleNamespace(
        protein_pos=torch.tensor(np.asarray(parsed.pos)[keep]),
        protein_atom_name=[parsed.atom_name[i] for i in keep],
        protein_atom_to_aa_type=torch.tensor([parsed.atom_to_aa_type[i] for i in keep]),
        protein_element=torch.tensor([parsed.element[i] for i in keep]),
    )
    block = dataset_protein_block(data, protein.with_name('missing.pdb'))
    shown = PDBProtein(block, mode='block')
    assert shown.atom_name == data.protein_atom_name
    assert np.array_equal(shown.pos, data.protein_pos.numpy())
    assert block.count('TER') == 1
    assert len(shown.residues) == 4


@pytest.fixture
def preview_client(tmp_path, monkeypatch):
    cfg = ServerConfig(output_root=str(tmp_path / 'outputs'),
                       upload_dir=str(tmp_path / 'uploads'),
                       data_root=str(tmp_path / 'data'), gpu_ids=[0])
    for folder in (cfg.output_root, cfg.upload_dir, cfg.data_root):
        Path(folder).mkdir(exist_ok=True)
    monkeypatch.setattr('server.config._config', cfg)
    monkeypatch.setenv('DD_AUTH_MODE', 'platform')
    monkeypatch.setattr(api, 'lookup_platform_identity',
                        lambda **kw: PlatformIdentity('alice', False)
                        if 'bianlab_token=alice' in kw.get('cookie_header', '') else None)
    monkeypatch.setattr(api, 'get_scheduler',
                        lambda: pytest.fail('A preview must not schedule a GPU job'))
    with TestClient(api.app) as client:
        client.headers['Origin'] = 'http://testserver'
        client.cookies.set('bianlab_token', 'alice')
        yield client, cfg


def test_generation_context_security_and_no_job(preview_client, structures):
    client, cfg = preview_client
    protein, ligand, _ = structures
    body = {'protein_path': str(protein), 'ligand_path': str(ligand)}
    response = client.post('/api/generate/context', json=body)
    assert response.status_code == 200
    assert response.headers['Cache-Control'] == 'no-store'
    assert response.json()['ligand_sdf']
    assert client.post('/api/generate/context', json=body,
                       headers={'Origin': 'https://foreign.invalid'}).status_code == 403
    assert client.post('/api/generate/context', json={**body,
                       'config_path': '/custom.yml'}).status_code == 422
    bob = Path(cfg.upload_dir) / 'users' / owner_slug('bob')
    bob.mkdir(parents=True)
    secret = bob / 'private.pdb'
    secret.write_text(protein.read_text())
    assert client.post('/api/generate/context', json={
        'protein_path': str(secret),
    }).status_code == 403
    client.cookies.clear()
    assert client.post('/api/generate/context', json=body).status_code == 401


@pytest.mark.parametrize('body', [
    {}, {'grow_n_extra_mode': 'fixed'},
    *[{'grow_n_extra_mode': 'fixed', 'grow_n_extra_fixed': value}
      for value in (None, 0, -1, 51, 2.5, True, '8')],
    {'grow_n_extra_mode': 'pocket_prior', 'grow_n_extra_fixed': 8},
    {'scaffold_mode': 'evolve', 'grow_n_extra_mode': 'fixed', 'grow_n_extra_fixed': 8},
])
def test_scaffold_preview_requires_an_explicit_valid_count(preview_client, body):
    client, _ = preview_client
    assert client.post('/api/scaffold/preview', json=body).status_code == 422
    if not body:
        for placement in ('custom', [{}]):
            assert client.post('/api/scaffold/preview', json={
                'placement': placement,
            }).status_code == 422


@pytest.mark.parametrize('count', [1, 11, 35, 50])
def test_fixed_budget_survives_global_and_site_caps(monkeypatch, structures, count):
    import server.scaffold_context as context
    _, _, molecule = structures
    scaffold = resolve_scaffold_indices(molecule, {})
    protein = torch.tensor([[50., 50., 50.], [54., 50., 50.]])
    monkeypatch.setattr(context, 'load_context',
                        lambda p: (molecule, scaffold, protein, {}))
    params = api.ScaffoldPreviewRequest(grow_n_extra_mode='fixed',
                                       grow_n_extra_fixed=count).dict()
    response = preview_context(params)
    assert len(response['positions']) == count
    assert response['effective_parameters']['n_extra'] == count


def test_fixed_budget_is_saved_for_generation_and_rerun(monkeypatch, tmp_path):
    from server.scheduler import JobScheduler
    init_db()
    scheduler = JobScheduler(max_concurrent=1)
    monkeypatch.setattr(scheduler, '_acquire_or_queue',
                        lambda gpu, rec, starter: {'run_id': rec.run_id})

    def write(base, patches, job):
        from server.job_helpers import deep_merge
        data = yaml.safe_load(Path(base).read_text())
        deep_merge(data, patches)
        path = tmp_path / (job + '.yml')
        path.write_text(yaml.safe_dump(data))
        return str(path)

    monkeypatch.setattr(scheduler, '_write_patched_config', write)
    result = scheduler.submit_scaffold(grow_n_extra_mode='fixed',
                                       grow_n_extra_fixed=35, triggered_by='admin',
                                       skip_daily_quota=True)
    saved = get_run(result['run_id'])
    scaffold = yaml.safe_load(saved.config_snapshot)['sample']['scaffold']
    assert scaffold['grow']['n_extra_min_clamp'] == 35
    assert scaffold['grow']['n_extra_max_clamp'] == 35
    assert scaffold['murcko_sites']['max_per_site'] == 35
    replay = scheduler.submit_rerun(saved.id, triggered_by='admin',
                                    skip_daily_quota=True)
    again = yaml.safe_load(get_run(replay['run_id']).config_snapshot)
    assert again['sample']['scaffold']['grow']['n_extra_fixed'] == 35
