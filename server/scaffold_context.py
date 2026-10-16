"""Read-only scaffold context and preview; no jobs, quotas or GPU allocation."""

from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import yaml
from rdkit import Chem
from scipy.spatial import cKDTree

from server.config import get_config
from diffdynamic.data.protein import PDBProtein, parse_sdf_file
from diffdynamic.chemistry.scaffold_inputs import (
    load_reference_molecule, resolve_scaffold_indices, reference_molecule_from_data,
)
from diffdynamic.chemistry.scaffold_sites import (
    build_extra_atom_positions, get_murcko_sites_cfg,
    load_or_extract_attachment_sites,
)


@lru_cache(maxsize=2)
def _dataset_config(checkpoint, modified):
    # Only server-owned checkpoints are allowed here, never user-uploaded PT.
    return torch.load(checkpoint, map_location='cpu', weights_only=False)['config'].data


def dataset_protein_block(data, pocket_path):
    """Recover backbone names without changing the stored pocket coordinates."""
    pocket = Path(pocket_path)
    if pocket.is_file():
        return pocket.read_text()
    positions = data.protein_pos.cpu().numpy()
    # Some local datasets retain the receptor but omit the extracted pocket PDB.
    if '_rec' in pocket.name:
        receptor = pocket.with_name(pocket.name.split('_rec', 1)[0] + '_rec.pdb')
        if receptor.is_file():
            protein = PDBProtein(str(receptor))
            if len(protein.pos):
                distances, indices = cKDTree(protein.pos).query(positions)
                if np.all(distances < 0.02):
                    selected = set(indices.tolist())
                    residues = [r for r in protein.residues
                                if selected.intersection(r['atoms'])]
                    return protein.residues_to_pdb_block(residues)

    # LMDB keeps real atom names and amino-acid identities. Residue numbering
    # is for display only; peptide geometry determines breaks in the ribbon.
    names = getattr(data, 'protein_atom_name', [])
    if len(names) != len(positions):
        raise ValueError('Protein backbone metadata is unavailable for this pocket')
    aa_types = data.protein_atom_to_aa_type.tolist()
    aa_names = {value: key for key, value in PDBProtein.AA_NAME_NUMBER.items()}
    table = Chem.GetPeriodicTable()
    lines, residue, seen, previous_c = [], 0, set(), None
    for i, (name, xyz, element) in enumerate(zip(
            names, positions, data.protein_element.tolist())):
        if not seen or name in seen:
            if name == 'N' and previous_c is not None:
                if np.linalg.norm(xyz - previous_c) > 2.0:
                    lines.append('TER')
                    residue += 10
            residue += 1
            seen = set()
        seen.add(name)
        if name == 'C':
            previous_c = xyz
        symbol = table.GetElementSymbol(int(element))
        lines.append(
            f'ATOM  {i + 1:5d} {name:>4s} {aa_names.get(aa_types[i], "UNK")} '
            f'A{residue:4d}    {xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}'
            f'  1.00  0.00          {symbol:>2s}'
        )
    return '\n'.join(lines) + '\nEND\n'


def load_context(params):
    """Use the same dataset split, pocket parser and heavy atoms as sampling."""
    from diffdynamic.data.datasets import get_dataset
    from diffdynamic.data.pocket import load_custom_pocket_data

    cfg = get_config()
    molecule = params.get('molecule_path') or params.get('ligand_path')
    protein = params.get('protein_path')
    if protein:
        if not molecule:
            raise ValueError('Custom pocket requires a reference 3D molecule')
        # The UI uses one reference ligand for crop and scaffold initialization.
        data = load_custom_pocket_data(protein, molecule)
        protein_obj = PDBProtein(protein)
        residues = protein_obj.query_residues_ligand(parse_sdf_file(molecule), 10.0)
        selected = {id(residue) for residue in residues}
        residues = [residue for residue in protein_obj.residues if id(residue) in selected]
        protein_block = (protein_obj.residues_to_pdb_block(residues)
                         if residues else Path(protein).read_text())
    else:
        base = params.get('config_path') or cfg.sampling_config
        settings = yaml.safe_load(Path(base).read_text())
        checkpoint = Path(settings['model']['checkpoint'])
        if not checkpoint.is_absolute():
            checkpoint = Path(cfg.diffdynamic_root) / checkpoint
        checkpoint = checkpoint.resolve()
        trusted = Path(cfg.pretrained_models).resolve()
        if trusted not in checkpoint.parents:
            raise ValueError('Context requires a server-owned pretrained checkpoint')
        dataset_cfg = _dataset_config(str(checkpoint), checkpoint.stat().st_mtime_ns)
        # get_dataset must not trigger dataset preprocessing in a preview.
        root = Path(dataset_cfg.path)
        if not root.is_absolute():
            root = Path(cfg.diffdynamic_root) / root
        processed = root.parent / (root.name + '_processed_final.lmdb')
        if not processed.is_file():
            raise ValueError('Processed test dataset is unavailable')
        dataset, subsets = get_dataset(config=dataset_cfg)
        try:
            index = params.get('data_id')
            index = 0 if index is None else int(index)
            if not 0 <= index < len(subsets['test']):
                raise ValueError('Test data_id is out of range')
            data = subsets['test'][index]
        finally:
            if dataset.db is not None:
                dataset._close_db()
        protein = str((root / data.protein_filename).resolve())
        molecule = molecule or str((root / data.ligand_filename).resolve())
        protein_block = dataset_protein_block(data, protein)
    mol = (load_reference_molecule(molecule) if Path(molecule).is_file()
           else reference_molecule_from_data(data))
    if not params.get('molecule_path') and not params.get('ligand_path'):
        mol = reference_molecule_from_data(data)
    else:
        # Explicit dataset ligand overrides use the same float32 pose as the
        # custom-pocket parser and CLI, preserving preview/sample agreement.
        positions = parse_sdf_file(molecule)['pos']
        for i, xyz in enumerate(positions.tolist()):
            mol.GetConformer().SetAtomPosition(i, xyz)
    scaffold = resolve_scaffold_indices(mol, params)
    return mol, scaffold, data.protein_pos, {
        'protein_path': protein, 'molecule_path': molecule,
        'ligand_sdf': Chem.MolToMolBlock(mol), 'protein_pdb': protein_block,
        'scaffold_indices': scaffold,
        'atoms': [{
            'index': atom.GetIdx(), 'element': atom.GetSymbol(),
            'source_index': atom.GetIntProp('_source_atom_index'),
            'xyz': list(mol.GetConformer().GetAtomPosition(atom.GetIdx())),
        } for atom in mol.GetAtoms()],
        'coordinate_frame': 'world_angstrom',
    }


def placement_config(params):
    """Mirror the automatic web defaults; custom budgets bypass allocation."""
    sc_cfg = {
        'murcko_sites': {
            'p_active': 0.5, 'max_per_site': 20,
            'per_site_count_mode': 'sequential_random',
            'overflow_mode': 'cap', 'save_json': False,
            'n_extra_min_clamp': 6, 'n_extra_max_clamp': 20,
            'deterministic_fallback': True,
        },
    }
    sites_cfg = get_murcko_sites_cfg(sc_cfg)
    if params.get('grow_n_extra_mode') == 'fixed':
        sites_cfg['max_per_site'] = max(20, params.get('grow_n_extra_fixed', 8))
    placement = params.get('placement') or {'mode': 'auto'}
    if placement['mode'] == 'custom':
        sites_cfg['custom_regions'] = placement['regions']
        sites_cfg['preserve_zero_allocation_sidechains'] = False
    return sites_cfg


def preview_context(params):
    from diffdynamic.sampling.scaffold.inputs import (
        _pin_place_cfg_to_n_extra, _sample_n_extra_atoms,
    )
    from diffdynamic.sampling.scaffold.placement import _murcko_placement_cfg
    mol, scaffold, protein, response = load_context(params)
    cfg = placement_config(params)
    grow = {
        'n_extra_mode': params.get('grow_n_extra_mode', 'pocket_prior'),
        'n_extra_fixed': params.get('grow_n_extra_fixed', 8),
        'n_extra_min': params.get('grow_n_extra_min', 3),
        'n_extra_max': params.get('grow_n_extra_max', 20),
        'seed': params.get('seed', 42), 'web_input_contract': True,
        'n_extra_min_clamp': 6, 'n_extra_max_clamp': 20,
    }
    if cfg.get('custom_regions'):
        n_extra = sum(r['atom_count'] for r in cfg['custom_regions'])
        grow.update({'n_extra_mode': 'fixed', 'n_extra_fixed': n_extra,
                     'n_extra_min_clamp': n_extra, 'n_extra_max_clamp': n_extra})
    else:
        if grow['n_extra_mode'] == 'fixed':
            grow.update({'n_extra_min_clamp': grow['n_extra_fixed'],
                         'n_extra_max_clamp': grow['n_extra_fixed']})
        n_extra = _sample_n_extra_atoms(grow, protein, len(scaffold), 1000)
    cfg = _pin_place_cfg_to_n_extra(
        _murcko_placement_cfg(grow, cfg), n_extra, grow,
    )
    sites = load_or_extract_attachment_sites(
        mol, scaffold, params.get('scaffold_source', 'auto_murcko'),
        None, None, None, cfg,
    )
    positions, meta = build_extra_atom_positions(
        n_extra, sites, cfg,
        protein.mean(dim=0), 'cpu',
        fallback_noise_scale=2.0 if params.get('scaffold_mode', 'grow') == 'grow' else 1.0,
        rng=np.random.default_rng(params.get('seed', 42)),
        protein_positions=protein,
    )
    response.update({
        'positions': positions.tolist(), 'metadata': meta,
        'seed': params.get('seed', 42), 'sample_index': 0,
        'stage': 'before_forward_diffusion',
        'workflow': params.get('workflow', 'legacy'),
        'effective_parameters': {'placement': cfg, 'n_extra': len(positions)},
    })
    return response
