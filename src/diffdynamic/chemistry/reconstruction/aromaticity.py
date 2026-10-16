"""Aromaticity responsibilities extracted from reconstruct.py."""

from diffdynamic.chemistry.reconstruction.state import (
    Chem,
    UPGRADE_BOND_ORDER,
    np,
    ob,
)


def _normalize_aromatic_indicators(aromatic):
    """Return a list of bools, or None.

    Callers sometimes pass ``ligand_atom_mode`` strings or type-index
    tensors as the ``aromatic`` argument.  Those are not indicator lists.
    """
    if aromatic is None:
        return None
    if isinstance(aromatic, (str, bytes)):
        return None
    if isinstance(aromatic, (bool, np.bool_, int, float, np.integer, np.floating)):
        return None
    if hasattr(aromatic, "detach"):
        aromatic = aromatic.detach().cpu().numpy()
    if isinstance(aromatic, np.ndarray):
        aromatic = aromatic.tolist()
    try:
        seq = list(aromatic)
    except TypeError:
        return None
    if not seq:
        return None
    out = []
    for item in seq:
        if hasattr(item, "item"):
            item = item.item()
        if isinstance(item, (bool, np.bool_)):
            out.append(bool(item))
        elif isinstance(item, (int, np.integer)) and int(item) in (0, 1):
            out.append(bool(int(item)))
        else:
            return None
    return out


def _reseed_scaffold_aromatic_bonds(mol, atoms, scaffold_bonds):
    """Put back aromatic flags on locked scaffold bonds after OpenBabel."""
    from diffdynamic.chemistry.reconstruction.scaffold import _normalize_scaffold_bonds

    if not scaffold_bonds or not atoms:
        return
    for i, j, _order, aromatic in _normalize_scaffold_bonds(scaffold_bonds):
        if not aromatic or i >= len(atoms) or j >= len(atoms):
            continue
        bond = mol.GetBond(atoms[i], atoms[j])
        if bond is None:
            continue
        atoms[i].SetAromatic(True)
        atoms[j].SetAromatic(True)
        bond.SetAromatic(True)


def restore_indicator_aromaticity(mol, atoms, indicators):
    """Re-apply model aromatic flags after OpenBabel PerceiveBondOrders.

    Scaffold bonds are restored separately from reference topology.  Generated
    extra atoms have no bond table, so PerceiveBondOrders turns their rings
    into aliphatic single/double bonds even when the model emitted C|1 / S|1
    and the coordinates are planar.  This puts those flags back.
    """
    flags = _normalize_aromatic_indicators(indicators)
    if not flags or not atoms:
        return
    n = min(len(atoms), len(flags))
    mol.SetAromaticPerceived(True)
    for i in range(n):
        if flags[i] and atoms[i].GetAtomicNum() > 1:
            atoms[i].SetAromatic(True)
            atoms[i].SetHyb(2)
    ob_to_i = {atom.GetIdx(): i for i, atom in enumerate(atoms)}
    for bond in ob.OBMolBondIter(mol):
        i = ob_to_i.get(bond.GetBeginAtom().GetIdx())
        j = ob_to_i.get(bond.GetEndAtom().GetIdx())
        if i is None or j is None or i >= n or j >= n:
            continue
        if flags[i] and flags[j]:
            bond.GetBeginAtom().SetAromatic(True)
            bond.GetEndAtom().SetAromatic(True)
            bond.SetAromatic(True)


def _force_generated_aromaticity(rd_mol, indicators):
    """Restore generated-atom aromaticity on the RDKit molecule."""
    flags = _normalize_aromatic_indicators(indicators)
    if rd_mol is None or not flags:
        return rd_mol
    n = min(rd_mol.GetNumAtoms(), len(flags))
    rwmol = Chem.RWMol(rd_mol)
    changed = False
    for i in range(n):
        atom = rwmol.GetAtomWithIdx(i)
        if atom.GetAtomicNum() <= 1:
            continue
        want = bool(flags[i])
        if want and not atom.GetIsAromatic():
            atom.SetIsAromatic(True)
            changed = True
    for bond in rwmol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if i >= n or j >= n:
            continue
        if flags[i] and flags[j]:
            a1 = rwmol.GetAtomWithIdx(i)
            a2 = rwmol.GetAtomWithIdx(j)
            if a1.GetAtomicNum() <= 1 or a2.GetAtomicNum() <= 1:
                continue
            if not bond.GetIsAromatic():
                bond.SetIsAromatic(True)
                changed = True
            if bond.GetBondType() == Chem.BondType.SINGLE:
                bond.SetBondType(Chem.BondType.AROMATIC)
                changed = True
    if not changed:
        return rd_mol
    result = rwmol.GetMol()
    try:
        result.UpdatePropertyCache(strict=False)
        Chem.SanitizeMol(result, sanitizeOps=Chem.SANITIZE_ALL ^ Chem.SANITIZE_KEKULIZE)
        Chem.SetAromaticity(result)
        return result
    except Exception:
        return rd_mol


def raw_obmol_from_generated(data):  # 将生成的配体上下文转为 OpenBabel 分子。
    from diffdynamic.chemistry.reconstruction.atoms import make_obmol

    xyz = data.ligand_context_pos.clone().cpu().tolist()  # 提取坐标。
    atomic_nums = data.ligand_context_element.clone().cpu().tolist()  # 提取原子序号。
    # indicators = data.ligand_context_feature_full[:, -len(ATOM_FAMILIES_ID):].clone().cpu().bool().tolist()

    mol, atoms = make_obmol(xyz, atomic_nums)  # 构建 OBMol 与 OBAtom 列表。
    return mol, atoms  # 返回分子及原子引用。


def postprocess_rd_mol_1(rdmol):  # 对 RDKit 分子进行第一阶段后处理。
    rdmol = Chem.RemoveHs(rdmol)  # 移除氢原子。

    # Construct bond nbh list  # 保留注释：构建键邻接表。
    nbh_list = {}
    for bond in rdmol.GetBonds():
        begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if begin not in nbh_list:
            nbh_list[begin] = [end]
        else:
            nbh_list[begin].append(end)

        if end not in nbh_list:
            nbh_list[end] = [begin]
        else:
            nbh_list[end].append(begin)

    # Fix missing bond-order  # 保留注释：修复缺失的键级。
    for atom in rdmol.GetAtoms():
        idx = atom.GetIdx()
        num_radical = atom.GetNumRadicalElectrons()
        if num_radical > 0:
            for j in nbh_list[idx]:
                if j <= idx:
                    continue
                nb_atom = rdmol.GetAtomWithIdx(j)
                nb_radical = nb_atom.GetNumRadicalElectrons()
                if nb_radical > 0:
                    bond = rdmol.GetBondBetweenAtoms(idx, j)
                    bond.SetBondType(UPGRADE_BOND_ORDER[bond.GetBondType()])
                    nb_atom.SetNumRadicalElectrons(nb_radical - 1)
                    num_radical -= 1
            atom.SetNumRadicalElectrons(num_radical)

        num_radical = atom.GetNumRadicalElectrons()
        if num_radical > 0:
            atom.SetNumRadicalElectrons(0)
            num_hs = atom.GetNumExplicitHs()
            atom.SetNumExplicitHs(num_hs + num_radical)

    return rdmol


def postprocess_rd_mol_2(
    rdmol,
    n_scaffold=None,
    protected_pairs=None,
):  # 对 RDKit 分子进行第二阶段后处理。
    from diffdynamic.chemistry.reconstruction.atoms import _pair_key

    rdmol_edit = Chem.RWMol(rdmol)  # 创建可编辑副本。
    n_sc = int(n_scaffold) if n_scaffold is not None else -1
    protected = set(protected_pairs or ())

    ring_info = rdmol.GetRingInfo()
    ring_info.AtomRings()
    rings = [set(r) for r in ring_info.AtomRings()]
    for i, ring_a in enumerate(rings):
        if len(ring_a) == 3:
            # 骨架内三元环不拆键，避免破坏参考拓扑
            if n_sc > 0 and all(int(a) < n_sc for a in ring_a):
                continue
            non_carbon = []
            atom_by_symb = {}
            for atom_idx in ring_a:
                symb = rdmol.GetAtomWithIdx(atom_idx).GetSymbol()
                if symb != "C":
                    non_carbon.append(atom_idx)
                if symb not in atom_by_symb:
                    atom_by_symb[symb] = [atom_idx]
                else:
                    atom_by_symb[symb].append(atom_idx)
            if len(non_carbon) == 2:
                pair = _pair_key(non_carbon[0], non_carbon[1])
                if pair not in protected and (
                    n_sc <= 0 or not all(int(a) < n_sc for a in non_carbon)
                ):
                    rdmol_edit.RemoveBond(*non_carbon)
            if "O" in atom_by_symb and len(atom_by_symb["O"]) == 2:
                o_pair = atom_by_symb["O"]
                pair = _pair_key(o_pair[0], o_pair[1])
                if pair not in protected and (
                    n_sc <= 0 or not all(int(a) < n_sc for a in o_pair)
                ):
                    rdmol_edit.RemoveBond(*o_pair)
                    rdmol_edit.GetAtomWithIdx(o_pair[0]).SetNumExplicitHs(
                        rdmol_edit.GetAtomWithIdx(o_pair[0]).GetNumExplicitHs() + 1
                    )
                    rdmol_edit.GetAtomWithIdx(o_pair[1]).SetNumExplicitHs(
                        rdmol_edit.GetAtomWithIdx(o_pair[1]).GetNumExplicitHs() + 1
                    )
    rdmol = rdmol_edit.GetMol()

    for atom in rdmol.GetAtoms():
        if atom.GetFormalCharge() > 0:
            atom.SetFormalCharge(0)

    return rdmol
