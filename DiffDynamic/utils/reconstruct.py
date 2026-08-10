"""
https://github.com/mattragoza/liGAN/blob/master/fitting.py

License: GNU General Public License v2.0
https://github.com/mattragoza/liGAN/blob/master/LICENSE
"""
import itertools  # 导入迭代工具，用于组合遍历。

import numpy as np  # 导入 NumPy。
from rdkit.Chem import AllChem as Chem  # 从 RDKit 导入化学模块。
from rdkit import Geometry  # 导入几何工具。
from openbabel import openbabel as ob  # 导入 OpenBabel 接口。
from scipy.spatial.distance import pdist  # 导入点对距离计算。
from scipy.spatial.distance import squareform  # 导入距离矩阵转换工具。


class MolReconsError(Exception):  # 定义分子重建错误类型。
    pass  # 保持空实现。


def reachable_r(a, b, seenbonds):  # 递归辅助函数，判断两原子在去掉特定键后是否连通。
    '''Recursive helper.'''

    for nbr in ob.OBAtomAtomIter(a):  # 遍历原子 a 的邻居。
        bond = a.GetBond(nbr).GetIdx()  # 获取与邻居的键索引。
        if bond not in seenbonds:  # 若该键未被访问过。
            seenbonds.add(bond)  # 标记已访问。
            if nbr == b:  # 找到目标原子。
                return True
            elif reachable_r(nbr, b, seenbonds):  # 否则继续递归搜索。
                return True
    return False  # 未找到则返回 False。


def reachable(a, b):  # 检查去掉 a-b 之间的键后，a 是否仍能到达 b。
    '''Return true if atom b is reachable from a without using the bond between them.'''
    if a.GetExplicitDegree() == 1 or b.GetExplicitDegree() == 1:  # 若任一原子度为 1。
        return False  # this is the _only_ bond for one atom
    # otherwise do recursive traversal
    seenbonds = set([a.GetBond(b).GetIdx()])  # 初始化已访问集合，排除当前键。
    return reachable_r(a, b, seenbonds)  # 调用递归辅助函数。


def forms_small_angle(a, b, cutoff=60):  # 判断 a-b 所在的角是否过小。
    '''Return true if bond between a and b is part of a small angle
    with a neighbor of a only.'''

    for nbr in ob.OBAtomAtomIter(a):  # 遍历 a 的邻居。
        if nbr != b:  # 排除 b 本身。
            degrees = b.GetAngle(a, nbr)  # 计算角度。
            if degrees < cutoff:  # 小于阈值则视为小角。
                return True
    return False  # 否则返回 False。


def make_obmol(xyz, atomic_numbers):  # 将坐标与原子序号构建为 OBMol 对象。
    mol = ob.OBMol()  # 创建空分子。
    mol.BeginModify()  # 进入修改状态。
    atoms = []  # 用于存储 OBAtom 引用。
    for xyz, t in zip(xyz, atomic_numbers):  # 遍历坐标与原子种类。
        x, y, z = xyz
        # ch = struct.channels[t]
        atom = mol.NewAtom()  # 创建新原子。
        atom.SetAtomicNum(int(t))  # 设置原子序数。
        atom.SetVector(float(x), float(y), float(z))  # 设置坐标（强制 Python float，避免 numpy 标量被 SWIG 拒绝）。
        atoms.append(atom)  # 保存原子引用。
    return mol, atoms  # 返回分子及原子列表。


def _pair_key(i: int, j: int):
    return (i, j) if i < j else (j, i)


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
                int(item[0]), int(item[1]), int(item[2]), bool(item[3]),
            )
        if i == j:
            continue
        out.append((i, j, max(1, int(order)), bool(aromatic)))
    return out


def seed_scaffold_bonds(mol, atoms, scaffold_bonds):
    """在 connect_the_dots 之前写入参考骨架键；返回受保护的 0-based 原子对集合。"""
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


def connect_the_dots(
    mol, atoms, indicators, covalent_factor=1.3,
    protected_pairs=None, n_scaffold=None,
):  # 根据原子位置尝试恢复键连接。
    '''Custom implementation of ConnectTheDots.  This is similar to
    OpenBabel's version, but is more willing to make long bonds 
    (up to maxbond long) to keep the molecule connected.  It also 
    attempts to respect atom type information from struct.
    atoms and struct need to correspond in their order
    Assumes no hydrogens or existing bonds (except optional seeded scaffold bonds).

    protected_pairs: set of (i,j) 0-based pairs that must not be deleted.
    n_scaffold: if set, do not invent new scaffold–scaffold bonds outside protected_pairs.
    '''

    """
    for now, indicators only include 'is_aromatic'
    """
    pt = Chem.GetPeriodicTable()  # 获取周期表用于查找价数。
    protected = set(protected_pairs or ())
    n_sc = int(n_scaffold) if n_scaffold is not None else -1

    # OB atom GetIdx -> 0-based index in atoms list
    ob_to_i = {a.GetIdx(): i for i, a in enumerate(atoms)}

    def _is_protected_bond(bond):
        i = ob_to_i.get(bond.GetBeginAtom().GetIdx())
        j = ob_to_i.get(bond.GetEndAtom().GetIdx())
        if i is None or j is None:
            return False
        return _pair_key(i, j) in protected

    if len(atoms) == 0:  # 若没有原子直接返回。
        return

    mol.BeginModify()  # 开始修改分子。

    # just going to to do n^2 comparisons, can worry about efficiency later
    coords = np.array([(a.GetX(), a.GetY(), a.GetZ()) for a in atoms])  # 提取所有原子坐标。
    dists = squareform(pdist(coords))  # 计算两两距离。
    # types = [struct.channels[t].name for t in struct.c]

    for i, j in itertools.combinations(range(len(atoms)), 2):  # 遍历所有原子对。
        # 骨架内部：只保留参考键，不按距离发明新骨架键
        if n_sc > 0 and i < n_sc and j < n_sc:
            if _pair_key(i, j) not in protected:
                continue
            # 已 seed 则跳过重复添加
            if mol.GetBond(atoms[i], atoms[j]) is not None:
                continue
        a = atoms[i]
        b = atoms[j]
        if mol.GetBond(a, b) is not None:
            continue
        a_r = ob.GetCovalentRad(a.GetAtomicNum()) * covalent_factor  # 原子 a 的共价半径（放大系数）。
        b_r = ob.GetCovalentRad(b.GetAtomicNum()) * covalent_factor  # 原子 b 的共价半径。
        if dists[i, j] < a_r + b_r:  # 若距离小于半径之和，认为可能有键连接。
            flag = 0
            if indicators and indicators[i] and indicators[j]:  # 若两端均被标记为芳香。
                flag = ob.OB_AROMATIC_BOND  # 设置芳香键标志。
            mol.AddBond(a.GetIdx(), b.GetIdx(), 1, flag)  # 新增单键。

    atom_maxb = {}  # 存储每个原子允许的最大键数。
    for (i, a) in enumerate(atoms):
        # set max valance to the smallest max allowed by openbabel or rdkit
        # since we want the molecule to be valid for both (rdkit is usually lower)
        maxb = min(ob.GetMaxBonds(a.GetAtomicNum()), pt.GetDefaultValence(a.GetAtomicNum()))  # 取 OpenBabel 与 RDKit 的最小允许值。

        if a.GetAtomicNum() == 16:  # sulfone check
            if count_nbrs_of_elem(a, 8) >= 2:
                maxb = 6

        # if indicators[i][ATOM_FAMILIES_ID['Donor']]:
        #     maxb -= 1 #leave room for hydrogen
        # if 'Donor' in types[i]:
        #     maxb -= 1 #leave room for hydrogen
        atom_maxb[a.GetIdx()] = maxb  # 记录最大价键。

    # remove any impossible bonds between halogens
    for bond in list(ob.OBMolBondIter(mol)):  # 遍历现有键。
        if _is_protected_bond(bond):
            continue
        a1 = bond.GetBeginAtom()
        a2 = bond.GetEndAtom()
        if atom_maxb[a1.GetIdx()] == 1 and atom_maxb[a2.GetIdx()] == 1:  # 若两端都只允许 1 个键。
            mol.DeleteBond(bond)  # 删除不可能的卤素键。

    def get_bond_info(biter):  # 返回按拉伸比例排序的键列表。
        '''Return bonds sorted by their distortion'''
        bonds = [b for b in biter]
        binfo = []
        for bond in bonds:
            bdist = bond.GetLength()
            # compute how far away from optimal we are
            a1 = bond.GetBeginAtom()
            a2 = bond.GetEndAtom()
            ideal = ob.GetCovalentRad(a1.GetAtomicNum()) + ob.GetCovalentRad(a2.GetAtomicNum())
            stretch = bdist / ideal  # 计算拉伸比。
            binfo.append((stretch, bond))
        binfo.sort(reverse=True, key=lambda t: t[0])  # most stretched bonds first
        return binfo

    binfo = get_bond_info(ob.OBMolBondIter(mol))  # 获取所有键的拉伸排序。
    # now eliminate geometrically poor bonds
    for stretch, bond in binfo:  # 遍历拉伸大的键。
        if _is_protected_bond(bond):
            continue

        # can we remove this bond without disconnecting the molecule?
        a1 = bond.GetBeginAtom()
        a2 = bond.GetEndAtom()

        # as long as we aren't disconnecting, let's remove things
        # that are excessively far away (0.45 from ConnectTheDots)
        # get bonds to be less than max allowed
        # also remove tight angles, because that is what ConnectTheDots does
        if stretch > 1.2 or forms_small_angle(a1, a2) or forms_small_angle(a2, a1):  # 拉伸过大或角度过小。
            # don't fragment the molecule
            if not reachable(a1, a2):  # 如果移除该键会断裂分子，则跳过。
                continue
            mol.DeleteBond(bond)  # 否则删除。

    # prioritize removing hypervalency causing bonds, do more valent
    # constrained atoms first since their bonds introduce the most problems
    # with reachability (e.g. oxygen)
    hypers = [(atom_maxb[a.GetIdx()], a.GetExplicitValence() - atom_maxb[a.GetIdx()], a) for a in atoms]  # 计算每个原子超价情况。
    hypers = sorted(hypers, key=lambda aa: (aa[0], -aa[1]))  # 优先处理允许价小且超价多的原子。
    for mb, diff, a in hypers:
        if a.GetExplicitValence() <= atom_maxb[a.GetIdx()]:  # 若未超价则跳过。
            continue
        binfo = get_bond_info(ob.OBAtomBondIter(a))  # 获取该原子相关键的拉伸信息。
        # Prioritize deleting extra-extra bonds over scaffold-extra bonds:
        # scaffold-extra bonds connect the sidechain to the scaffold and are
        # geometrically correct (placed at 2-4 A), so they should be kept.
        def _bond_priority(bond_item):
            stretch, bond = bond_item
            a1_idx = ob_to_i.get(bond.GetBeginAtom().GetIdx(), -1)
            a2_idx = ob_to_i.get(bond.GetEndAtom().GetIdx(), -1)
            is_scaffold_extra = (
                (a1_idx < n_sc and a2_idx >= n_sc) or
                (a2_idx < n_sc and a1_idx >= n_sc)
            ) if n_sc > 0 else False
            return (1 if is_scaffold_extra else 0, stretch)
        binfo.sort(reverse=True, key=_bond_priority)
        for stretch, bond in binfo:
            if _is_protected_bond(bond):
                continue

            if stretch < 0.9:  # the two atoms are too closed to remove the bond
                continue
            # can we remove this bond without disconnecting the molecule?
            a1 = bond.GetBeginAtom()
            a2 = bond.GetEndAtom()

            # get right valence
            if a1.GetExplicitValence() > atom_maxb[a1.GetIdx()] or a2.GetExplicitValence() > atom_maxb[a2.GetIdx()]:
                # don't fragment the molecule
                if not reachable(a1, a2):  # 确保移除后仍连通。
                    continue
                mol.DeleteBond(bond)  # 删除该键。
                if a.GetExplicitValence() <= atom_maxb[a.GetIdx()]:  # 若超价问题已解决。
                    break  # let nbr atoms choose what bonds to throw out

    mol.EndModify()  # 结束修改。


def convert_ob_mol_to_rd_mol(ob_mol, struct=None):
    '''Convert OBMol to RDKit mol, fixing up issues'''
    ob_mol.DeleteHydrogens()  # 移除氢原子，简化结构。
    n_atoms = ob_mol.NumAtoms()  # 取得原子数。
    rd_mol = Chem.RWMol()  # 创建 RDKit 可编辑分子。
    rd_conf = Chem.Conformer(n_atoms)  # 创建构象用于存储坐标。

    for ob_atom in ob.OBMolAtomIter(ob_mol):  # 遍历 OpenBabel 原子。
        rd_atom = Chem.Atom(ob_atom.GetAtomicNum())  # 复制原子序号。
        # TODO copy format charge
        if ob_atom.IsAromatic() and ob_atom.IsInRing() and ob_atom.MemberOfRingSize() <= 6:  # 针对芳香环。
            # don't commit to being aromatic unless rdkit will be okay with the ring status
            # (this can happen if the atoms aren't fit well enough)
            rd_atom.SetIsAromatic(True)
        i = rd_mol.AddAtom(rd_atom)  # 将原子加入 RDKit 分子，获取索引。
        ob_coords = ob_atom.GetVector()  # 读取 OpenBabel 坐标。
        x = ob_coords.GetX()
        y = ob_coords.GetY()
        z = ob_coords.GetZ()
        rd_coords = Geometry.Point3D(x, y, z)  # 构造 RDKit 坐标。
        rd_conf.SetAtomPosition(i, rd_coords)  # 设置原子位置。

    rd_mol.AddConformer(rd_conf)  # 添加构象。

    for ob_bond in ob.OBMolBondIter(ob_mol):  # 遍历 OpenBabel 键。
        i = ob_bond.GetBeginAtomIdx() - 1  # 转换为 RDKit 索引（从 0 开始）。
        j = ob_bond.GetEndAtomIdx() - 1
        bond_order = ob_bond.GetBondOrder()  # 读取键级。
        if bond_order == 1:
            rd_mol.AddBond(i, j, Chem.BondType.SINGLE)
        elif bond_order == 2:
            rd_mol.AddBond(i, j, Chem.BondType.DOUBLE)
        elif bond_order == 3:
            rd_mol.AddBond(i, j, Chem.BondType.TRIPLE)
        else:
            raise Exception('unknown bond order {}'.format(bond_order))

        if ob_bond.IsAromatic():  # 如果 OpenBabel 键为芳香。
            bond = rd_mol.GetBondBetweenAtoms(i, j)
            bond.SetIsAromatic(True)

    rd_mol = Chem.RemoveHs(rd_mol, sanitize=False)  # 去除所有显式氢原子（暂不执行 sanitize）。

    pt = Chem.GetPeriodicTable()
    # if double/triple bonds are connected to hypervalent atoms, decrement the order

    positions = rd_mol.GetConformer().GetPositions()  # 获取坐标。
    nonsingles = []  # 收集非单键。
    for bond in rd_mol.GetBonds():
        if bond.GetBondType() == Chem.BondType.DOUBLE or bond.GetBondType() == Chem.BondType.TRIPLE:  # 找出双/三键。
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            dist = np.linalg.norm(positions[i] - positions[j])  # 计算键长。
            nonsingles.append((dist, bond))
    nonsingles.sort(reverse=True, key=lambda t: t[0])  # 按键长从大到小排序。

    for (d, bond) in nonsingles:  # 遍历双/三键。
        a1 = bond.GetBeginAtom()
        a2 = bond.GetEndAtom()

        if calc_valence(a1) > pt.GetDefaultValence(a1.GetAtomicNum()) or \
                calc_valence(a2) > pt.GetDefaultValence(a2.GetAtomicNum()):  # 如果超出默认价。
            btype = Chem.BondType.SINGLE  # 降级为单键。
            if bond.GetBondType() == Chem.BondType.TRIPLE:  # 如果原为三键，则先降为双键。
                btype = Chem.BondType.DOUBLE
            bond.SetBondType(btype)

    for atom in rd_mol.GetAtoms():  # 遍历所有原子。
        # set nitrogens with 4 neighbors to have a charge
        if atom.GetAtomicNum() == 7 and atom.GetDegree() == 4:
            atom.SetFormalCharge(1)  # 对四价氮设置 +1 电荷。

    rd_mol = Chem.AddHs(rd_mol, addCoords=True)  # 重新加入氢原子并生成坐标。

    positions = rd_mol.GetConformer().GetPositions()  # 获取坐标矩阵。
    center = np.mean(positions[np.all(np.isfinite(positions), axis=1)], axis=0)  # 计算有限坐标点的中心。
    for atom in rd_mol.GetAtoms():  # 遍历原子。
        i = atom.GetIdx()
        pos = positions[i]
        if not np.all(np.isfinite(pos)):  # 若坐标包含 NaN。
            # hydrogens on C fragment get set to nan (shouldn't, but they do)
            rd_mol.GetConformer().SetAtomPosition(i, center)  # 将该原子坐标重置为中心位置。

    try:
        Chem.SanitizeMol(rd_mol, Chem.SANITIZE_ALL ^ Chem.SANITIZE_KEKULIZE)  # 执行除去 kekulize 的结构检查。
    except:
        raise MolReconsError()  # 若失败则抛出重建错误。
    # try:
    #     Chem.SanitizeMol(rd_mol,Chem.SANITIZE_ALL^Chem.SANITIZE_KEKULIZE)
    # except: # mtr22 - don't assume mols will pass this
    #     pass
    #     # dkoes - but we want to make failures as rare as possible and should debug them
    #     m = pybel.Molecule(ob_mol)
    #     i = np.random.randint(1000000)
    #     outname = 'bad%d.sdf'%i
    #     print("WRITING",outname)
    #     m.write('sdf',outname,overwrite=True)
    #     pickle.dump(struct,open('bad%d.pkl'%i,'wb'))

    # but at some point stop trying to enforce our aromaticity -
    # openbabel and rdkit have different aromaticity models so they
    # won't always agree.  Remove any aromatic bonds to non-aromatic atoms
    for bond in rd_mol.GetBonds():  # 再次同步键的芳香性标记。
        a1 = bond.GetBeginAtom()
        a2 = bond.GetEndAtom()
        if bond.GetIsAromatic():
            if not a1.GetIsAromatic() or not a2.GetIsAromatic():
                bond.SetIsAromatic(False)
        elif a1.GetIsAromatic() and a2.GetIsAromatic():
            bond.SetIsAromatic(True)

    return rd_mol  # 返回处理后的分子。


def calc_valence(rdatom):  # 计算 RDKit 原子的显式价。
    '''Can call GetExplicitValence before sanitize, but need to
    know this to fix up the molecule to prevent sanitization failures'''
    cnt = 0.0
    for bond in rdatom.GetBonds():
        cnt += bond.GetBondTypeAsDouble()
    return cnt


def count_nbrs_of_elem(atom, atomic_num):  # 统计指定原子的邻居中具有特定原子序号的数量。
    '''
    Count the number of neighbors atoms
    of atom with the given atomic_num.
    '''
    count = 0
    for nbr in ob.OBAtomAtomIter(atom):
        if nbr.GetAtomicNum() == atomic_num:
            count += 1
    return count


def fixup(atoms, mol, indicators):  # 调整 OBMol 原子的属性以符合指定指示器。
    '''Set atom properties to match channel.  Keep doing this
    to beat openbabel over the head with what we want to happen.'''

    """
    for now, indicators only include 'is_aromatic'
    """
    mol.SetAromaticPerceived(True)  # avoid perception
    for i, atom in enumerate(atoms):
        # ch = struct.channels[t]
        if indicators is not None:
            if indicators[i]:
                atom.SetAromatic(True)
                atom.SetHyb(2)
            else:
                atom.SetAromatic(False)

        # if ind[ATOM_FAMILIES_ID['Donor']]:
        #     if atom.GetExplicitDegree() == atom.GetHvyDegree():
        #         if atom.GetHvyDegree() == 1 and atom.GetAtomicNum() == 7:
        #             atom.SetImplicitHCount(2)
        #         else:
        #             atom.SetImplicitHCount(1) 

        # elif ind[ATOM_FAMILIES_ID['Acceptor']]: # NOT AcceptorDonor because of else
        #     atom.SetImplicitHCount(0)   

        if (atom.GetAtomicNum() in (7, 8)) and atom.IsInRing():  # Nitrogen, Oxygen
            # this is a little iffy, ommitting until there is more evidence it is a net positive
            # we don't have aromatic types for nitrogen, but if it
            # is in a ring with aromatic carbon mark it aromatic as well
            acnt = 0
            for nbr in ob.OBAtomAtomIter(atom):
                if nbr.IsAromatic():
                    acnt += 1
            if acnt > 1:
                atom.SetAromatic(True)


def raw_obmol_from_generated(data):  # 将生成的配体上下文转为 OpenBabel 分子。
    xyz = data.ligand_context_pos.clone().cpu().tolist()  # 提取坐标。
    atomic_nums = data.ligand_context_element.clone().cpu().tolist()  # 提取原子序号。
    # indicators = data.ligand_context_feature_full[:, -len(ATOM_FAMILIES_ID):].clone().cpu().bool().tolist()

    mol, atoms = make_obmol(xyz, atomic_nums)  # 构建 OBMol 与 OBAtom 列表。
    return mol, atoms  # 返回分子及原子引用。


UPGRADE_BOND_ORDER = {Chem.BondType.SINGLE: Chem.BondType.DOUBLE, Chem.BondType.DOUBLE: Chem.BondType.TRIPLE}


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
                if j <= idx: continue
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


def postprocess_rd_mol_2(rdmol, n_scaffold=None):  # 对 RDKit 分子进行第二阶段后处理。
    rdmol_edit = Chem.RWMol(rdmol)  # 创建可编辑副本。
    n_sc = int(n_scaffold) if n_scaffold is not None else -1

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
                if symb != 'C':
                    non_carbon.append(atom_idx)
                if symb not in atom_by_symb:
                    atom_by_symb[symb] = [atom_idx]
                else:
                    atom_by_symb[symb].append(atom_idx)
            if len(non_carbon) == 2:
                if n_sc <= 0 or not all(int(a) < n_sc for a in non_carbon):
                    rdmol_edit.RemoveBond(*non_carbon)
            if 'O' in atom_by_symb and len(atom_by_symb['O']) == 2:
                o_pair = atom_by_symb['O']
                if n_sc <= 0 or not all(int(a) < n_sc for a in o_pair):
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


def reconstruct_from_generated(
    xyz, atomic_nums, aromatic=None, basic_mode=True,
    scaffold_bonds=None, n_scaffold=None,
    rdkit_structure_repair=None,
):  # 从生成的坐标和元素重建 RDKit 分子。
    """
    will utilize data.ligand_pos, data.ligand_element, data.ligand_atom_feature_full to reconstruct mol

    scaffold_bonds: optional list of (i, j[, order[, aromatic]]) for atoms 0..n_scaffold-1
    to force-keep Murcko scaffold topology during OpenBabel distance bonding.

    rdkit_structure_repair: optional dict matching sample.rdkit_structure_repair in sampling.yml
        enable / config / on_reject. Applied AFTER postprocess (orthogonal to
        targetdiff_baseline_refine which runs before .pt write).
    """
    # xyz = data.ligand_pos.clone().cpu().tolist()
    # atomic_nums = data.ligand_element.clone().cpu().tolist()
    # indicators = data.ligand_atom_feature_full[:, -len(ATOM_FAMILIES_ID):].clone().cpu().bool().tolist()
    # indicators = None
    if basic_mode:
        indicators = None  # 基础模式下不使用芳香指示。
    else:
        indicators = aromatic  # 否则使用外部提供的芳香标记。

    mol, atoms = make_obmol(xyz, atomic_nums)  # 构建 OpenBabel 分子。
    fixup(atoms, mol, indicators)  # 根据指示器调整原子属性。

    protected = seed_scaffold_bonds(mol, atoms, scaffold_bonds)
    n_sc = int(n_scaffold) if n_scaffold is not None else None
    if n_sc is None and protected:
        # 从键表推断骨架前缀长度
        n_sc = max(max(i, j) for i, j, *_ in _normalize_scaffold_bonds(scaffold_bonds)) + 1

    connect_the_dots(
        mol, atoms, indicators, covalent_factor=2.0,
        protected_pairs=protected, n_scaffold=n_sc,
    )  # 根据距离连接键（骨架内参考键受保护）。
    fixup(atoms, mol, indicators)  # 再次调整原子属性确保一致。

    mol.AddPolarHydrogens()  # 添加极性氢。
    mol.PerceiveBondOrders()  # 让 OpenBabel 感知键级。
    # Re-seed scaffold aromaticity after PerceiveBondOrders, which may
    # have cleared the aromatic flags set by seed_scaffold_bonds.
    if protected:
        norm_bonds = _normalize_scaffold_bonds(scaffold_bonds)
        for i, j, order, aromatic in norm_bonds:
            if i >= len(atoms) or j >= len(atoms):
                continue
            a, b = atoms[i], atoms[j]
            bond = mol.GetBond(a, b)
            if bond is None:
                continue
            if aromatic:
                a.SetAromatic(True)
                b.SetAromatic(True)
                bond.SetAromatic(True)
    fixup(atoms, mol, indicators)  # 再次修正芳香标记。

    for (i, a) in enumerate(atoms):
        ob.OBAtomAssignTypicalImplicitHydrogens(a)  # 为原子分配典型隐式氢数。
    fixup(atoms, mol, indicators)  # 再次同步属性。

    mol.AddHydrogens()  # 添加全部氢原子。
    # Re-seed scaffold aromaticity again after hydrogen addition.
    if protected:
        norm_bonds = _normalize_scaffold_bonds(scaffold_bonds)
        for i, j, order, aromatic in norm_bonds:
            if i >= len(atoms) or j >= len(atoms):
                continue
            a, b = atoms[i], atoms[j]
            bond = mol.GetBond(a, b)
            if bond is None:
                continue
            if aromatic:
                a.SetAromatic(True)
                b.SetAromatic(True)
                bond.SetAromatic(True)
    fixup(atoms, mol, indicators)  # 再次修正属性。

    # make rings all aromatic if majority of carbons are aromatic
    for ring in ob.OBMolRingIter(mol):  # 遍历所有环，处理芳香性。
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
                # set all ring atoms to be aromatic
                for ai in ring._path:
                    a = mol.GetAtom(ai)
                    a.SetAromatic(True)

    # bonds must be marked aromatic for smiles to match
    for bond in ob.OBMolBondIter(mol):  # 确保键的芳香标记与原子一致。
        a1 = bond.GetBeginAtom()
        a2 = bond.GetEndAtom()
        if a1.IsAromatic() and a2.IsAromatic():
            bond.SetAromatic(True)

    mol.PerceiveBondOrders()
    rd_mol = convert_ob_mol_to_rd_mol(mol)  # 转换为 RDKit 分子。
    try:
        # Post-processing
        rd_mol = postprocess_rd_mol_1(rd_mol)
        rd_mol = postprocess_rd_mol_2(rd_mol, n_scaffold=n_sc)
        rd_mol = _force_scaffold_aromaticity(rd_mol, scaffold_bonds, n_sc)
    except:
        raise MolReconsError()

    rd_mol = _maybe_apply_rdkit_structure_repair(rd_mol, rdkit_structure_repair)
    return rd_mol


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
        Chem.SanitizeMol(result, sanitizeOps=Chem.SANITIZE_ALL ^ Chem.SANITIZE_PROPERTIES)
        Chem.SetAromaticity(result)
        return result
    except Exception:
        return rd_mol


def _maybe_apply_rdkit_structure_repair(rd_mol, repair_cfg):
    """Optional deterministic topology repair (rdkit-structure-repair package).

    Runs after OB→RDKit postprocess. Default off. Config shape::
        {enable: bool, config: path, on_reject: keep_original|raise}
    """
    if not repair_cfg:
        return rd_mol
    # Support OmegaConf / dict / plain object
    try:
        enable = bool(repair_cfg.get("enable", False))
    except Exception:
        enable = bool(getattr(repair_cfg, "enable", False))
    if not enable:
        return rd_mol

    import os
    import sys

    try:
        cfg_path = repair_cfg.get("config", None)
        on_reject = repair_cfg.get("on_reject", "keep_original")
    except Exception:
        cfg_path = getattr(repair_cfg, "config", None)
        on_reject = getattr(repair_cfg, "on_reject", "keep_original")

    # Ensure package importable when running from DiffDynamic root
    dd_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pkg_src = os.path.join(dd_root, "rdkit-structure-repair", "src")
    if os.path.isdir(pkg_src) and pkg_src not in sys.path:
        sys.path.insert(0, pkg_src)

    if cfg_path and not os.path.isabs(str(cfg_path)):
        cfg_path = os.path.join(dd_root, str(cfg_path))

    try:
        from structure_repair import repair_molecule  # type: ignore
        from structure_repair.config import load_config  # type: ignore

        config = load_config(cfg_path) if cfg_path else load_config()
        result = repair_molecule(rd_mol, config=config, molecule_id="reconstruct")
        if result.status == "REPAIRED" and result.repaired_mol is not None:
            return result.repaired_mol
        if result.status == "UNCHANGED" and result.repaired_mol is not None:
            # May still have cleared maps / standardize applied
            return result.repaired_mol
        if result.status in ("AMBIGUOUS", "REJECTED"):
            if on_reject == "raise":
                raise MolReconsError(
                    f"rdkit_structure_repair {result.status}: {result.reject_reason}"
                )
            return rd_mol
        return result.repaired_mol if result.repaired_mol is not None else rd_mol
    except MolReconsError:
        raise
    except Exception:
        # Never break reconstruction if the optional repair package fails
        return rd_mol


def save_positions_only_to_sdf(xyz, atomic_nums, output_path):
    """将仅含原子位置的分子保存为 SDF 文件（无键连接）。

    用于扩散过程早期（t>bond_threshold）的可视化，此时原子尚未形成合理键结构。

    Args:
        xyz: 坐标列表或数组，形状 [num_atoms, 3]
        atomic_nums: 原子序数列表或数组
        output_path: 输出 SDF 文件路径
    """
    from rdkit.Chem import RWMol

    if hasattr(xyz, 'tolist'):
        xyz = xyz.tolist()
    if hasattr(atomic_nums, 'tolist'):
        atomic_nums = atomic_nums.tolist()

    rwmol = RWMol()
    for an in atomic_nums:
        rwmol.AddAtom(Chem.Atom(int(an)))

    conf = Chem.Conformer(rwmol.GetNumAtoms())
    for i, (coord, an) in enumerate(zip(xyz, atomic_nums)):
        conf.SetAtomPosition(i, Geometry.Point3D(float(coord[0]), float(coord[1]), float(coord[2])))
    rwmol.AddConformer(conf, assignId=True)

    mol = rwmol.GetMol()
    sdf_block = Chem.MolToMolBlock(mol)
    with open(output_path, 'w') as f:
        f.write(sdf_block)
        f.write('$$$$\n')
