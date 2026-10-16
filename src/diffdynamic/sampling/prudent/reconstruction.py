"""Sampling prudent reconstruction responsibilities for ligand sampling."""

from __future__ import annotations

from rdkit import Chem
import numpy as np
import torch
import diffdynamic.chemistry.reconstruct as reconstruct
import diffdynamic.data.transforms as trans


def _prudent_reconstruct_mol(
    pos_array, v_array, ligand_atom_mode, logger=None, context='',
    scaffold_bonds=None, n_scaffold=None,
):
    """重建 RDKit Mol；失败返回 None。"""
    from diffdynamic.sampling.trajectory import _step_traj_as_bool_list
    from diffdynamic.sampling.trajectory import _step_traj_as_int_list

    try:
        # 记录输入参数
        if logger:
            logger.debug(
                f'[_prudent_reconstruct_mol] {context} | '
                f'pos_shape={pos_array.shape}, '
                f'v_shape={v_array.shape}, '
                f'v_dtype={v_array.dtype}, '
                f'v_range=[{v_array.min()}, {v_array.max()}], '
                f'unique_v={len(np.unique(v_array))}'
            )
        
        # 检查 v_array 是否为整数类型
        if not np.issubdtype(v_array.dtype, np.integer):
            if logger:
                logger.warning(
                    f'[_prudent_reconstruct_mol] {context} | '
                    f'v_array is not integer type (dtype={v_array.dtype}), attempting conversion'
                )
            v_array = v_array.astype(np.int64)
        
        v_tensor = torch.tensor(v_array, dtype=torch.long)
        atom_numbers = trans.get_atomic_number_from_index(v_tensor, mode=ligand_atom_mode)
        aromatic_flags = trans.is_aromatic_from_index(v_tensor, mode=ligand_atom_mode)
        
        if logger:
            # get_atomic_number_from_index / is_aromatic_from_index 返回 list（或 aromatic 为 None），无 .tolist()
            logger.debug(
                f'[_prudent_reconstruct_mol] {context} | '
                f'atom_numbers={_step_traj_as_int_list(atom_numbers)}, '
                f'aromatic_flags={_step_traj_as_bool_list(aromatic_flags)}'
            )

        basic_mode = str(ligand_atom_mode) == 'basic'
        mol = reconstruct.reconstruct_from_generated(
            pos_array, atom_numbers, aromatic_flags,
            basic_mode=basic_mode,
            scaffold_bonds=scaffold_bonds,
            n_scaffold=n_scaffold,
        )
        
        if logger:
            if mol is not None:
                logger.debug(f'[_prudent_reconstruct_mol] {context} | SUCCESS: mol created with {mol.GetNumAtoms()} atoms')
            else:
                logger.debug(f'[_prudent_reconstruct_mol] {context} | FAILED: reconstruct returned None')
        
        return mol
    except Exception as e:
        if logger:
            logger.warning(
                f'[_prudent_reconstruct_mol] {context} | EXCEPTION: {type(e).__name__}: {str(e)[:100]}'
            )
        return None


def _prudent_canonical_smiles(
    cand_or_mol=None,
    *,
    pos=None,
    v=None,
    ligand_atom_mode=None,
    scaffold_bonds=None,
    n_scaffold=None,
    logger=None,
    context='',
):
    """Return canonical SMILES for a candidate dict / Mol / (pos,v), or None."""
    mol = None
    if cand_or_mol is not None and hasattr(cand_or_mol, 'GetNumAtoms'):
        mol = cand_or_mol
    elif isinstance(cand_or_mol, dict):
        cached = cand_or_mol.get('smiles') or cand_or_mol.get('canonical_smiles')
        if cached and isinstance(cached, str) and cached not in ('', 'None', 'nan'):
            try:
                m2 = Chem.MolFromSmiles(cached)
                if m2 is not None:
                    return Chem.MolToSmiles(m2)
            except Exception:
                pass
        pos = cand_or_mol.get('pos', pos)
        v = cand_or_mol.get('v', v)
        scaffold_bonds = cand_or_mol.get('scaffold_bond_pairs', scaffold_bonds)
        n_scaffold = cand_or_mol.get('n_scaffold', n_scaffold)
    if mol is None:
        if pos is None or v is None or ligand_atom_mode is None:
            return None
        mol = _prudent_reconstruct_mol(
            np.asarray(pos), np.asarray(v), ligand_atom_mode,
            logger=logger, context=context or 'canonical_smiles',
            scaffold_bonds=scaffold_bonds, n_scaffold=n_scaffold,
        )
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(mol)
    except Exception:
        return None
