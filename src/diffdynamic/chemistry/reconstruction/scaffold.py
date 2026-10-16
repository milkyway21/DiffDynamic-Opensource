"""Scaffold responsibilities extracted from reconstruct.py."""

from diffdynamic.chemistry.reconstruction.state import (
    Chem,
    np,
    ob,
)


def _normalize_scaffold_bonds(scaffold_bonds):
    """归一化为 [(i, j, order, aromatic_flag), ...]；i/j 为 0-based 原子下标。"""
    out = []
    if not scaffold_bonds:
        return out
    for item in scaffold_bonds:
        if item is None:
            continue
        if len(item) == 2:
            i, j = int(item[0]), int(item[1])
            order, aromatic = 1, False
        elif len(item) == 3:
            i, j, order = int(item[0]), int(item[1]), int(item[2])
            aromatic = (order == 5) or (order == -1)
            if order == -1:
                order = 5
        else:
            i, j, order, aromatic = (
                int(item[0]),
                int(item[1]),
                int(item[2]),
                bool(item[3]),
            )
        if i == j:
            continue
        out.append((i, j, max(1, int(order)), bool(aromatic)))
    return out


def seed_scaffold_bonds(mol, atoms, scaffold_bonds):
    """在 connect_the_dots 之前写入受保护的骨架/锚定键。"""
    from diffdynamic.chemistry.reconstruction.atoms import _pair_key

    bonds = _normalize_scaffold_bonds(scaffold_bonds)
    protected = set()
    if not bonds or not atoms:
        return protected
    mol.BeginModify()
    for i, j, order, aromatic in bonds:
        if i < 0 or j < 0 or i >= len(atoms) or j >= len(atoms):
            continue
        a, b = atoms[i], atoms[j]
        if mol.GetBond(a, b) is not None:
            protected.add(_pair_key(i, j))
            continue
        # OpenBabel AddBond 不接受 order=5；芳香用单键 + AROMATIC flag
        is_arom = bool(aromatic) or int(order) == 5
        flag = ob.OB_AROMATIC_BOND if is_arom else 0
        bond_order = 1 if is_arom else max(1, min(3, int(order)))
        mol.AddBond(a.GetIdx(), b.GetIdx(), bond_order, flag)
        if is_arom:
            a.SetAromatic(True)
            b.SetAromatic(True)
        protected.add(_pair_key(i, j))
    mol.EndModify()
    return protected


def _force_scaffold_aromaticity(rd_mol, scaffold_bonds, n_scaffold):
    """Force aromaticity on scaffold bonds that were specified as aromatic.

    After OpenBabel -> RDKit conversion and post-processing, aromatic flags
    on scaffold bonds may be lost. This restores them by re-assigning bond
    orders to match the scaffold and then letting RDKit perceive aromaticity.
    """
    if not scaffold_bonds or rd_mol is None:
        return rd_mol
    norm = _normalize_scaffold_bonds(scaffold_bonds)
    if not norm:
        return rd_mol
    # Build a map of what the scaffold bonds should be
    target_bonds = {}
    for i, j, order, aromatic in norm:
        key = (min(i, j), max(i, j))
        target_bonds[key] = (order, aromatic)
    if not target_bonds:
        return rd_mol
    rwmol = Chem.RWMol(rd_mol)
    changed = False
    for (i, j), (order, aromatic) in target_bonds.items():
        if i >= rwmol.GetNumAtoms() or j >= rwmol.GetNumAtoms():
            continue
        bond = rwmol.GetBondBetweenAtoms(i, j)
        if bond is None:
            continue
        # Set bond order to match scaffold
        if aromatic:
            target_type = Chem.BondType.AROMATIC
        elif order == 2:
            target_type = Chem.BondType.DOUBLE
        elif order == 3:
            target_type = Chem.BondType.TRIPLE
        else:
            target_type = Chem.BondType.SINGLE
        if bond.GetBondType() != target_type:
            bond.SetBondType(target_type)
            changed = True
        if aromatic and not bond.GetIsAromatic():
            bond.SetIsAromatic(True)
            changed = True
    if not changed:
        return rd_mol
    result = rwmol.GetMol()
    # Update property cache and re-derive aromaticity
    try:
        result.UpdatePropertyCache(strict=False)
        Chem.SanitizeMol(
            result, sanitizeOps=Chem.SANITIZE_ALL ^ Chem.SANITIZE_PROPERTIES
        )
        Chem.SetAromaticity(result)
        return result
    except Exception:
        return rd_mol


def _repair_scaffold_extra_bonds(
    rd_mol,
    scaffold_bonds,
    n_scaffold,
    protected_extra_bonds=None,
):
    """Keep locked scaffold valence valid after noisy extra-atom bonding.

    Distance bonding can attach several generated atoms to a scaffold atom.
    When that makes a locked carbonyl or aromatic atom hypervalent, preserving
    the erroneous bridge is worse than dropping that bridge. This repair is
    gated by ``scaffold_bonds`` and therefore does not affect de novo paths.
    """
    from diffdynamic.chemistry.reconstruction.atoms import _pair_key

    if rd_mol is None or not scaffold_bonds or n_scaffold is None:
        return rd_mol
    n_sc = int(n_scaffold)
    if n_sc <= 0:
        return rd_mol
    norm = _normalize_scaffold_bonds(scaffold_bonds)
    if not norm:
        return rd_mol
    protected_extra = {
        _pair_key(i, j)
        for i, j, _, _ in _normalize_scaffold_bonds(protected_extra_bonds)
        if i < n_sc or j < n_sc
    }

    rwmol = Chem.RWMol(rd_mol)
    aromatic_atoms = set()
    for i, j, order, aromatic in norm:
        if i >= rwmol.GetNumAtoms() or j >= rwmol.GetNumAtoms():
            continue
        bond = rwmol.GetBondBetweenAtoms(i, j)
        if bond is None:
            continue
        if aromatic:
            target_type = Chem.BondType.AROMATIC
            aromatic_atoms.update((i, j))
        elif order == 2:
            target_type = Chem.BondType.DOUBLE
        elif order == 3:
            target_type = Chem.BondType.TRIPLE
        else:
            target_type = Chem.BondType.SINGLE
        bond.SetBondType(target_type)
        bond.SetIsAromatic(bool(aromatic))
    for atom_idx in aromatic_atoms:
        rwmol.GetAtomWithIdx(atom_idx).SetIsAromatic(True)

    periodic_table = Chem.GetPeriodicTable()

    def _default_valence(atom):
        value = periodic_table.GetDefaultValence(atom.GetAtomicNum())
        if value is None or int(value) < 0:
            return 4.0
        if atom.GetFormalCharge() > 0 and atom.GetAtomicNum() == 7:
            return 4.0
        return float(value)

    for core_idx in range(min(n_sc, rwmol.GetNumAtoms())):
        atom = rwmol.GetAtomWithIdx(core_idx)
        core_bonds = []
        extra_bonds = []
        for bond in atom.GetBonds():
            begin = bond.GetBeginAtomIdx()
            end = bond.GetEndAtomIdx()
            other = end if begin == core_idx else begin
            if other < n_sc:
                core_bonds.append(bond)
            else:
                extra_bonds.append(bond)
        if not extra_bonds:
            continue

        core_valence = sum(bond.GetBondTypeAsDouble() for bond in core_bonds)
        available = max(_default_valence(atom) - core_valence, 0.0)
        keep_count = int(np.floor(available + 1e-6))
        # A generated bond from an aromatic scaffold atom is an ordinary
        # substituent bond, never an aromatic or exocyclic double bond here.
        if core_idx in aromatic_atoms:
            for bond in extra_bonds:
                bond.SetBondType(Chem.BondType.SINGLE)
                bond.SetIsAromatic(False)

        forced = [
            bond
            for bond in extra_bonds
            if _pair_key(
                bond.GetBeginAtomIdx(),
                bond.GetEndAtomIdx(),
            )
            in protected_extra
        ]
        optional = [bond for bond in extra_bonds if bond not in forced]
        ranked = sorted(
            optional,
            key=lambda bond: (
                bond.GetBondTypeAsDouble(),
                bond.GetBeginAtomIdx(),
                bond.GetEndAtomIdx(),
            ),
        )
        keep_optional = max(keep_count - len(forced), 0)
        for bond in ranked[keep_optional:]:
            rwmol.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())

    repaired = rwmol.GetMol()
    repaired.UpdatePropertyCache(strict=False)
    try:
        Chem.SanitizeMol(
            repaired,
            sanitizeOps=Chem.SANITIZE_ALL ^ Chem.SANITIZE_PROPERTIES,
        )
    except Exception:
        # Keep the explicit scaffold topology available to the caller even
        # when an unrelated generated side chain remains chemically noisy.
        pass
    return repaired


def _reconnect_scaffold_extra_components(
    rd_mol,
    n_scaffold,
    max_distance=4.5,
):
    """Reconnect small generated components using their 3D coordinates.

    This is only called when scaffold-site attachment bonds are present. It
    repairs an extra atom that was first bonded to a full-valence scaffold
    atom and subsequently became isolated, preferring an extra-to-extra bond
    over adding another bond directly to the locked scaffold.
    """
    if rd_mol is None or n_scaffold is None:
        return rd_mol
    n_sc = int(n_scaffold)
    if n_sc <= 0 or rd_mol.GetNumAtoms() <= n_sc:
        return rd_mol
    try:
        conformer = rd_mol.GetConformer()
    except Exception:
        return rd_mol

    periodic_table = Chem.GetPeriodicTable()

    def _available_valence(atom):
        value = periodic_table.GetDefaultValence(atom.GetAtomicNum())
        if value is None or int(value) < 0:
            value = 4
        if atom.GetFormalCharge() > 0 and atom.GetAtomicNum() == 7:
            value = 4
        used = sum(bond.GetBondTypeAsDouble() for bond in atom.GetBonds())
        return float(value) - used

    rwmol = Chem.RWMol(rd_mol)
    while True:
        components = [
            set(component)
            for component in Chem.GetMolFrags(
                rwmol.GetMol(),
                asMols=False,
                sanitizeFrags=False,
            )
        ]
        main = next(
            (component for component in components if 0 in component),
            None,
        )
        if main is None:
            return rd_mol
        detached = [
            component
            for component in components
            if not component.issubset(main) and not (component & set(range(n_sc)))
        ]
        if not detached:
            break

        component = min(detached, key=len)
        candidates = []
        for detached_idx in component:
            detached_atom = rwmol.GetAtomWithIdx(detached_idx)
            if _available_valence(detached_atom) < 1.0 - 1e-6:
                continue
            detached_pos = np.asarray(
                conformer.GetAtomPosition(detached_idx),
                dtype=float,
            )
            for main_idx in main:
                if rwmol.GetBondBetweenAtoms(detached_idx, main_idx):
                    continue
                main_atom = rwmol.GetAtomWithIdx(main_idx)
                if _available_valence(main_atom) < 1.0 - 1e-6:
                    continue
                main_pos = np.asarray(
                    conformer.GetAtomPosition(main_idx),
                    dtype=float,
                )
                distance = float(np.linalg.norm(detached_pos - main_pos))
                if distance > float(max_distance):
                    continue
                # Prefer reconnecting into the generated side chain, which
                # avoids consuming another locked scaffold valence slot.
                priority = int(detached_idx < n_sc or main_idx < n_sc)
                candidates.append((priority, distance, detached_idx, main_idx))
        if not candidates:
            break
        _, _, detached_idx, main_idx = min(candidates)
        rwmol.AddBond(detached_idx, main_idx, Chem.BondType.SINGLE)

    repaired = rwmol.GetMol()
    repaired.UpdatePropertyCache(strict=False)
    try:
        Chem.SanitizeMol(
            repaired,
            sanitizeOps=Chem.SANITIZE_ALL ^ Chem.SANITIZE_PROPERTIES,
        )
    except Exception:
        pass
    return repaired
