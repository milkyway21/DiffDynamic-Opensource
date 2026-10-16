"""Pipeline responsibilities extracted from reconstruct.py."""

from diffdynamic.chemistry.reconstruction.state import (
    ob,
)


def reconstruct_from_generated(
    xyz,
    atomic_nums=None,
    aromatic=None,
    basic_mode=True,
    scaffold_bonds=None,
    n_scaffold=None,
    rdkit_structure_repair=None,
    extra_attachment_bonds=None,
    covalent_factor=2.0,
    ligand_atom_mode=None,
    strict=True,
):
    """Rebuild an RDKit mol from coordinates plus element / aromatic flags.

    ``PerceiveBondOrders`` clears OpenBabel aromatic flags.  Locked scaffold
    bonds are re-seeded from ``scaffold_bonds``; generated extra atoms are
    re-seeded from ``aromatic`` indicators so C|1 / S|1 extras stay aromatic.
    """
    from diffdynamic.chemistry.reconstruction.aromaticity import (
        _force_generated_aromaticity,
        _normalize_aromatic_indicators,
        _reseed_scaffold_aromatic_bonds,
        postprocess_rd_mol_1,
        postprocess_rd_mol_2,
        restore_indicator_aromaticity,
    )
    from diffdynamic.chemistry.reconstruction.atoms import (
        MolReconsError,
        connect_the_dots,
        convert_ob_mol_to_rd_mol,
        fixup,
        make_obmol,
    )
    from diffdynamic.chemistry.reconstruction.repair import (
        _maybe_apply_rdkit_structure_repair,
    )
    from diffdynamic.chemistry.reconstruction.scaffold import (
        _force_scaffold_aromaticity,
        _normalize_scaffold_bonds,
        _reconnect_scaffold_extra_components,
        _repair_scaffold_extra_bonds,
        seed_scaffold_bonds,
    )

    if atomic_nums is None and hasattr(xyz, "ligand_pos"):
        return reconstruct_from_data(
            xyz,
            aromatic=aromatic,
            basic_mode=basic_mode,
            scaffold_bonds=scaffold_bonds,
            n_scaffold=n_scaffold,
            rdkit_structure_repair=rdkit_structure_repair,
            extra_attachment_bonds=extra_attachment_bonds,
            covalent_factor=covalent_factor,
            ligand_atom_mode=ligand_atom_mode,
            strict=strict,
        )
    if isinstance(aromatic, str) and aromatic in {"basic", "add_aromatic", "full"}:
        ligand_atom_mode = aromatic
        aromatic = None
    if ligand_atom_mode is not None and aromatic is None and atomic_nums is not None:
        return reconstruct_from_type_indices(
            xyz,
            atomic_nums,
            ligand_atom_mode=ligand_atom_mode,
            scaffold_bonds=scaffold_bonds,
            n_scaffold=n_scaffold,
            rdkit_structure_repair=rdkit_structure_repair,
            extra_attachment_bonds=extra_attachment_bonds,
            covalent_factor=covalent_factor,
            strict=strict,
        )

    flags = _normalize_aromatic_indicators(aromatic)
    # Honor a real indicator list even when callers left basic_mode=True.
    if basic_mode and flags is None:
        indicators = None
    else:
        indicators = flags

    mol, atoms = make_obmol(xyz, atomic_nums)
    fixup(atoms, mol, indicators)

    scaffold_seed_bonds = list(scaffold_bonds or [])
    attachment_seed_bonds = list(extra_attachment_bonds or [])
    protected = seed_scaffold_bonds(
        mol,
        atoms,
        scaffold_seed_bonds + attachment_seed_bonds,
    )
    n_sc = int(n_scaffold) if n_scaffold is not None else None
    if n_sc is None and protected:
        n_sc = (
            max(
                max(i, j) for i, j, *_ in _normalize_scaffold_bonds(scaffold_seed_bonds)
            )
            + 1
        )

    connect_the_dots(
        mol,
        atoms,
        indicators,
        covalent_factor=float(covalent_factor),
        protected_pairs=protected,
        n_scaffold=n_sc,
    )
    fixup(atoms, mol, indicators)

    mol.AddPolarHydrogens()
    mol.PerceiveBondOrders()
    _reseed_scaffold_aromatic_bonds(
        mol,
        atoms,
        scaffold_seed_bonds + attachment_seed_bonds,
    )
    restore_indicator_aromaticity(mol, atoms, indicators)
    fixup(atoms, mol, indicators)

    for i, a in enumerate(atoms):
        ob.OBAtomAssignTypicalImplicitHydrogens(a)
    restore_indicator_aromaticity(mol, atoms, indicators)
    fixup(atoms, mol, indicators)

    mol.AddHydrogens()
    _reseed_scaffold_aromatic_bonds(
        mol,
        atoms,
        scaffold_seed_bonds + attachment_seed_bonds,
    )
    restore_indicator_aromaticity(mol, atoms, indicators)
    fixup(atoms, mol, indicators)

    for ring in ob.OBMolRingIter(mol):
        if 5 <= ring.Size() <= 6:
            carbon_cnt = 0
            aromatic_ccnt = 0
            for ai in ring._path:
                a = mol.GetAtom(ai)
                if a.GetAtomicNum() == 6:
                    carbon_cnt += 1
                    if a.IsAromatic():
                        aromatic_ccnt += 1
            if aromatic_ccnt >= carbon_cnt / 2 and aromatic_ccnt != ring.Size():
                for ai in ring._path:
                    a = mol.GetAtom(ai)
                    a.SetAromatic(True)

    for bond in ob.OBMolBondIter(mol):
        a1 = bond.GetBeginAtom()
        a2 = bond.GetEndAtom()
        if a1.IsAromatic() and a2.IsAromatic():
            bond.SetAromatic(True)

    mol.PerceiveBondOrders()
    _reseed_scaffold_aromatic_bonds(
        mol,
        atoms,
        scaffold_seed_bonds + attachment_seed_bonds,
    )
    restore_indicator_aromaticity(mol, atoms, indicators)
    rd_mol = convert_ob_mol_to_rd_mol(mol)
    try:
        rd_mol = postprocess_rd_mol_1(rd_mol)
        rd_mol = postprocess_rd_mol_2(
            rd_mol,
            n_scaffold=n_sc,
            protected_pairs=protected,
        )
        rd_mol = _force_generated_aromaticity(rd_mol, indicators)
        rd_mol = _force_scaffold_aromaticity(
            rd_mol,
            scaffold_seed_bonds + attachment_seed_bonds,
            n_sc,
        )
        rd_mol = _repair_scaffold_extra_bonds(
            rd_mol,
            scaffold_seed_bonds,
            n_sc,
            protected_extra_bonds=attachment_seed_bonds,
        )
        if attachment_seed_bonds:
            rd_mol = _reconnect_scaffold_extra_components(rd_mol, n_sc)
    except Exception:
        if strict:
            raise MolReconsError()
        # Keep the OpenBabel conversion so callers can still cut / score.

    rd_mol = _maybe_apply_rdkit_structure_repair(rd_mol, rdkit_structure_repair)
    return rd_mol


def reconstruct_from_type_indices(
    xyz,
    type_indices,
    ligand_atom_mode="add_aromatic",
    **kwargs,
):
    """Rebuild from model class indices (C|0 / C|1), not raw atomic numbers."""
    import torch
    from diffdynamic.data import transforms as trans

    tensor = type_indices
    if hasattr(tensor, "detach"):
        tensor = tensor.detach().cpu()
    tensor = torch.as_tensor(tensor, dtype=torch.long)
    if tensor.dim() > 1:
        tensor = tensor.argmax(dim=-1)
    atomic_nums = trans.get_atomic_number_from_index(tensor, mode=ligand_atom_mode)
    aromatic = trans.is_aromatic_from_index(tensor, mode=ligand_atom_mode)
    kwargs.pop("ligand_atom_mode", None)
    kwargs.pop("aromatic", None)
    kwargs.pop("basic_mode", None)
    return reconstruct_from_generated(
        xyz,
        atomic_nums,
        aromatic,
        basic_mode=str(ligand_atom_mode) == "basic",
        **kwargs,
    )


def reconstruct_from_data(data, ligand_atom_mode="add_aromatic", **kwargs):
    """Rebuild from a ProteinLigandData-like object."""
    import torch
    from diffdynamic.data import transforms as trans

    xyz = (
        data.ligand_pos.detach().cpu().tolist()
        if hasattr(data.ligand_pos, "detach")
        else data.ligand_pos
    )
    mode = str(kwargs.pop("ligand_atom_mode", None) or ligand_atom_mode)
    if (
        hasattr(data, "ligand_atom_feature_full")
        and data.ligand_atom_feature_full is not None
    ):
        v = data.ligand_atom_feature_full
        if hasattr(v, "detach"):
            v = v.detach().cpu()
        v = torch.as_tensor(v)
        if v.dim() > 1:
            if v.size(-1) > 1 and v.dtype.is_floating_point:
                v = v.argmax(dim=-1)
            elif v.size(-1) > 1:
                v = v.argmax(dim=-1)
        return reconstruct_from_type_indices(xyz, v, ligand_atom_mode=mode, **kwargs)
    atomic_nums = (
        data.ligand_element.detach().cpu().tolist()
        if hasattr(data.ligand_element, "detach")
        else list(data.ligand_element)
    )
    aromatic = kwargs.pop("aromatic", None)
    kwargs.pop("basic_mode", None)
    return reconstruct_from_generated(
        xyz,
        atomic_nums,
        aromatic,
        basic_mode=str(mode) == "basic",
        **kwargs,
    )
