"""Portable CPU integration fixtures for the molecule evaluation pipeline.

Only molecular reconstruction and individual docking scores are simulated.
The real PT loader, preparation, pipeline execution, aggregate statistics and
PT/JSON/SDF/report writers run end to end. Expected contracts were checked
against the original evaluator before the preparation/execution split.
"""

import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from rdkit import Chem

import _diffdynamic_bootstrap  # noqa: F401


class FixtureDateTime(datetime):
    @classmethod
    def now(cls, tz=None):
        return cls(2026, 1, 2, 3, 4, 5, tzinfo=tz)


SCENARIOS = {
    'success': [0, 4],
    'mixed': [0, 1, 2, 3, 4, 6],
    'all_fail': [1, 3],
    'zero': [],
    'prudent_cache': [0, 4],
    'split': [0, 4, 5],
    'limited': [0, 4, 5],
}


def fixture_molecule(marker):
    smiles = {0: 'CCO', 4: 'CCN', 5: 'CCC'}.get(marker, 'CCO')
    mol = Chem.MolFromSmiles(smiles)
    conformer = Chem.Conformer(mol.GetNumAtoms())
    for index in range(mol.GetNumAtoms()):
        conformer.SetAtomPosition(index, (1.4 * index, .1 * index, 0.))
    mol.AddConformer(conformer)
    mol.SetIntProp('_fixture_marker', marker)
    return mol


def fixture_reconstruction(pos, types, **kwargs):
    marker = int(np.asarray(pos)[0, 0])
    if marker == 1:
        return None, {'error_type': 'reconstruct_returned_none',
                      'error': 'fixture reconstruction failure', 'atom_count': 3}
    if marker == 3:
        raise ValueError('fixture reconstruction exception')
    return fixture_molecule(marker), None


def fixture_score(mol, ligand_filename, protein_root, mol_idx=None, **kwargs):
    marker = mol.GetIntProp('_fixture_marker')
    if marker == 2:
        raise RuntimeError('fixture scoring exception')
    modes = kwargs.get('vina_modes', frozenset({'score_only'}))
    index = int(mol_idx)
    success = marker != 6
    active = bool(modes) and success
    return {
        'mol': mol, 'smiles': Chem.MolToSmiles(mol), 'success': success,
        'error': None if success else 'fixture failed docking',
        'chem': {'qed': .6 + .05 * index, 'sa': .8 - .05 * index},
        'vina_dock': [{'affinity': -8. - index}] if active and 'dock' in modes else None,
        'vina_score_only': -7. - index if active and 'score_only' in modes else None,
        'vina_minimize': -7.5 - index if active and 'minimize' in modes else None,
        'atom_type_jsd': .1 + .01 * index,
        'bond_length_jsd': {'6-6': .2 + .01 * index},
        'pair_length_jsd': {'all': .3 + .01 * index},
        'stability': {'molecule_stable': True, 'nr_stable_bonds': 2, 'n_atoms': 3},
        'basic_info': {'n_atoms': 3, 'n_bonds': 2, 'n_rings': 0, 'weight': 46.},
        'logp': 1. + index, 'tpsa': 20. + index, 'lipinski': 5, 'pains': False,
        'tanimoto_sim': .4, 'rdkit_rmsd': {'max': .2},
        'conformer_energy': 1.5, 'rdkit_valid': True,
        'lilly_medchem_passed': True, 'lilly_medchem_demerit': 0,
        'lilly_medchem_description': '',
    }


def run_evaluation_fixture(tmp_path, monkeypatch, scenario, evaluator=None,
                           save_excel=False):
    """Run a small real input through either evaluator implementation."""
    if evaluator is None:
        import evaluate_pt_with_correct_reconstruct as evaluator
    root = tmp_path / scenario
    root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv('DIFFDYNAMIC_SKIP_SCORE_CATEGORIES', '1')
    monkeypatch.setenv('DIFFDYNAMIC_RECORD_DIR', str(root / 'records'))
    monkeypatch.setenv('EVAL_MOLECULE_ID_SUFFIX', 'fixture')
    monkeypatch.setattr(evaluator, 'datetime', FixtureDateTime)
    monkeypatch.setattr(evaluator, 'reconstruct_molecule', fixture_reconstruction)
    monkeypatch.setattr(evaluator, 'evaluate_single_molecule_isolated', fixture_score)
    monkeypatch.setattr(evaluator, 'dock_reference_ligand',
                        lambda *args, **kwargs: -6.)
    monkeypatch.setattr(evaluator, '_MEDCHEM_OPTIMIZE_CFG', None)
    config = root / 'sample.yml'
    config.write_text('model: {}\nsample:\n  dynamic:\n    large_step: {}\n    refine: {}\n')
    markers = SCENARIOS[scenario]
    payload = {
        'data': SimpleNamespace(ligand_filename='fixture_ligand.sdf',
                                protein_filename='fixture_protein.pdb'),
        'pred_ligand_pos': [torch.tensor([[float(m), 0., 0.], [1., 0., 0.],
                                          [2., 0., 0.]]) for m in markers],
        'pred_ligand_v': [torch.tensor([0, 0, 2]) for _ in markers],
        'extra_info': {'data_id': 7, 'config_backup': str(config)},
        'mode': 'prudent' if scenario == 'prudent_cache' else 'dynamic',
        'meta': {},
    }
    if scenario == 'prudent_cache':
        payload['meta']['refined_candidates'] = [
            {'prudent_composite_detail': {'vina_affinity': affinity}}
            for affinity in (-6., -9.)
        ]
    if scenario == 'split':
        payload['mode'] = 'dynamic_then_optimization'
        payload['meta']['records'] = [
            {'is_dynamic_raw': True, 'source_dynamic_idx': 0},
            {'is_original': False, 'source_dynamic_idx': 0},
            {'is_original': False, 'source_dynamic_idx': 0},
        ]
    pt = root / 'result_7.pt'
    torch.save(payload, str(pt))
    output = evaluator.evaluate_pt_file(
        str(pt), str(root), output_dir=str(root / 'evaluated'),
        save_sdf=True, save_excel=save_excel, save_distribution_plots=False,
        save_intermediate_interval=2, remove_fragments=False,
        max_samples=1 if scenario == 'limited' else None,
        vina_modes=frozenset() if scenario == 'prudent_cache' else
        frozenset({'score_only'}),
    )
    return output, root


def normalized_contract(value):
    """Remove variable run roots while retaining every result field/value."""
    if isinstance(value, Chem.Mol):
        return {'smiles': Chem.MolToSmiles(value),
                'coordinates': np.asarray(value.GetConformer().GetPositions()).tolist(),
                'properties': dict(value.GetPropsAsDict())}
    if torch.is_tensor(value):
        return normalized_contract(value.tolist())
    if isinstance(value, np.ndarray):
        return normalized_contract(value.tolist())
    if isinstance(value, np.generic):
        return normalized_contract(value.item())
    if isinstance(value, dict):
        return {str(k): normalized_contract(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalized_contract(v) for v in value]
    if isinstance(value, float):
        if np.isnan(value):
            return '<NaN>'
        return round(value, 12)
    if isinstance(value, str):
        if value.startswith('/'):
            return '<PATH>/' + Path(value).name
        return value
    return value


def output_manifest(root):
    """Read real saved PT/JSON/SDF/report output as a comparable contract."""
    directories = list((root / 'evaluated').glob('eval_*'))
    if not directories:
        return {}
    run = directories[0]
    manifest = {}
    for path in sorted(run.rglob('*')):
        if not path.is_file():
            continue
        rel = str(path.relative_to(run))
        if path.suffix == '.pt':
            manifest[rel] = normalized_contract(
                torch.load(str(path), map_location='cpu', weights_only=False)
            )
        elif path.suffix == '.json':
            manifest[rel] = normalized_contract(json.loads(path.read_text()))
        elif path.suffix == '.sdf':
            mol = Chem.SDMolSupplier(str(path))[0]
            manifest[rel] = normalized_contract(mol)
        elif path.suffix == '.txt':
            manifest[rel] = re.sub(r'/\S+', '<PATH>', path.read_text())
        elif path.suffix == '.xlsx':
            import pandas as pd
            sheets = pd.read_excel(str(path), sheet_name=None)
            manifest[rel] = {name: normalized_contract(df.fillna('<NaN>').to_dict())
                             for name, df in sheets.items()}
    return manifest


# Frozen complete contracts from the original evaluator, with variable paths
# normalized by normalized_contract(). Changes must be reviewed explicitly.
GOLDEN_DIGESTS = {
    'success': ('1b0b60bcf2f3f0d2ae1e2d0a8732f89ee2b402e1a9950a6fa541c859e2c37483',
        '0231c70163a07cd3dcc71cde33f847634874073c730e4f6e5b048b2b9c08532e'),
    'mixed': ('f292c9932b9bfe30c6f82d230c2ba18d54b5d5dd569bb6b409fb1cf3a4a3aedd',
        '6d0279f4a7d99928242805637e090594ea959bc0305731114a38931999b5f042'),
    'all_fail': ('5db7e0c3e08a67775adf50593023acd4ae56c581d4c97d7f76bc9746a78a91cc',
        '3078a0724cba214e9c6fff9e9a85a387ea577bdc099206b6784b64e7cdef8724'),
    'zero': ('74234e98afe7498fb5daf1f36ac2d78acc339464f950703b8c019892f982b90b',
        '44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a'),
    'prudent_cache': ('e3b3b0466612f2be535720d4b8dd0cc5a7be50a2c88643c4a53c0f974a61abaf',
        'd156a3d9a34f7a71be645e07bcf6700e7868944aae859dc7efe453d06abdeef1'),
    'split': ('5b2be1516cf39805730dbe67578501f03bb4f7cabe80f74b04eaee6669e7edbf',
        '4de2b0242da1eda51c0b6d22e5defc4112c6f3a7c61a068d26388f41ee4a02eb'),
    'limited': ('6cb4183abf7c192dcc10dd11d6eb8d8bb6fcf42c81ff5d2f8a876071b841e5f6',
        'aaf9c99a8ee6882a7f9b502826eb720024aa6612bb87c57be811b791d06f56b9'),
}

EXPECTED_COUNTS = {
    'success': (2, 2, 2, 2),
    'mixed': (6, 4, 2, 2),
    'all_fail': (2, 0, 0, 0),
    'zero': None,
    'prudent_cache': (2, 2, 2, 2),
    'split': (3, 3, 3, 3),
    'limited': (1, 1, 1, 1),
}


def contract_digest(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode()).hexdigest()


@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_evaluation_pipeline_saved_contract(tmp_path, monkeypatch, scenario):
    output, root = run_evaluation_fixture(
        tmp_path, monkeypatch, scenario, save_excel=scenario == "split",
    )
    manifest = output_manifest(root)
    expected_output, expected_files = GOLDEN_DIGESTS[scenario]
    assert contract_digest(normalized_contract(output)) == expected_output
    assert contract_digest(manifest) == expected_files
    if output is None:
        assert EXPECTED_COUNTS[scenario] is None
        assert not manifest
        return
    assert tuple(output[k] for k in (
        "num_samples", "n_reconstruct_success", "n_eval_success", "n_complete",
    )) == EXPECTED_COUNTS[scenario]
    assert sum(name.endswith(".pt") for name in manifest) == 1
    assert any(name == "summary.json" for name in manifest)
    assert not any("intermediate" in name for name in manifest)


def test_prudent_existing_affinity_reaches_sdf(tmp_path, monkeypatch):
    output, root = run_evaluation_fixture(tmp_path, monkeypatch, "prudent_cache")
    assert output["statistics"]["vina_score_only_scores"] == [-6., -9.]
    sdf = [data for name, data in output_manifest(root).items()
           if name.endswith(".sdf")]
    assert sorted(float(mol["properties"]["Vina_ScoreOnly"]) for mol in sdf) == [
        -9., -6.,
    ]


def test_split_output_preserves_original_to_variant_identity(tmp_path, monkeypatch):
    output, root = run_evaluation_fixture(tmp_path, monkeypatch, "split")
    manifest = output_manifest(root)
    originals = [name for name in manifest if name.startswith("original_molecules/")]
    variants = [name for name in manifest if name.startswith("optimized_molecules/")]
    assert len(originals) == 1 and len(variants) == 2
    original_id = output["results"][0]["molecule_id"]
    assert output["results"][1]["molecule_id"].endswith("_op1_from_" + original_id)
    assert output["results"][2]["molecule_id"].endswith("_op2_from_" + original_id)


def test_reconstruction_and_scoring_failures_keep_distinct_accounting(
    tmp_path, monkeypatch,
):
    output, root = run_evaluation_fixture(tmp_path, monkeypatch, 'mixed')
    reconstruction = output['reconstruct_statistics']
    assert reconstruction['n_reconstruct_fail'] == 2
    assert reconstruction['failure_stats'] == {
        'MolReconsError': 0, 'reconstruct_returned_none': 1,
        'other_errors': {'reconstruct_exception': 1},
    }
    assert [(row['mol_idx'], row['success']) for row in output['results']] == [
        (0, True), (2, False), (4, True), (5, False),
    ]
    assert output['results'][1]['reconstructed'] is True
    assert 'fixture scoring exception' in output['results'][1]['error']
    assert output['statistics']['vina_score_only_scores'] == [-7., -11.]
    summary = output_manifest(root)['summary.json']
    assert summary['num_molecules'] == 2
    assert summary['mean_vina'] == -9.
    assert summary['best_vina'] == -11.


CLI_FLAG_DIGESTS = {
    'evaluate_pt_with_correct_reconstruct.py': '1a909a2ed48b3f412d44c5671d9a3a0f89fe3109fa93dd4a12e95dac5ebd159f',
    'evaluate_pocket_quality.py': 'a3dbef2f85a486b91f537ca44717b216a01565da05880fd28711352bdf75103b',
    'batch_sampleandeval_parallel.py': '113b64515041f9b0dfa4032e26d39c5260fa8ed15f37270527d68e05017bb406',
}


@pytest.mark.parametrize("entry", sorted(CLI_FLAG_DIGESTS))
def test_old_cli_option_contract_from_another_directory(tmp_path, entry):
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(root / entry), "--help"], cwd=str(tmp_path),
        env=dict(os.environ, CUDA_VISIBLE_DEVICES="", DD_ROOT=str(root)),
        text=True, capture_output=True, timeout=45,
    )
    assert result.returncode == 0, result.stderr
    flags = sorted(set(re.findall(
        r"(?<![\w-])--[A-Za-z][\w-]*|(?<![\w-])-[A-Za-z][\w-]*",
        result.stdout,
    )))
    assert contract_digest(flags) == CLI_FLAG_DIGESTS[entry]
