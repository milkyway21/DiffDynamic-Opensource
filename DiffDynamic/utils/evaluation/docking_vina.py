# NumPy 1.24+ 移除 np.int 等别名；vina 等旧依赖在 import/运行期仍会引用
import numpy as np
if not hasattr(np, 'int'):
    np.int = int
    np.float = float
    np.complex = complex
    np.bool = np.bool_
    np.unicode = str
    np.long = int

# Meeko 0.7+ 需要 Python 3.9+（否则 import meeko 会因类型注解失败）；Py3.8 请保持 meeko 0.5.x 或新建高版本 Python 环境再装 0.7.x。

from openbabel import pybel

try:
    from openbabel import openbabel as _ob_openbabel
except ImportError:
    _ob_openbabel = None

from meeko import MoleculePreparation

try:
    from meeko import obutils  # 旧版 Meeko（如 0.1.dev3、0.5.x）
except ImportError:
    obutils = None  # Meeko≥0.6 起已移除顶层 obutils；由下方 _write_ob_mol_to_file 回退

try:
    from meeko.writer import PDBQTWriterLegacy as _MeekoPDBQTWriter
except ImportError:
    _MeekoPDBQTWriter = None
from vina import Vina
import subprocess
import rdkit.Chem as Chem
from rdkit.Chem import AllChem


def _patch_rdkit_mol_has_query_if_missing():
    """Meeko 0.7+ 在 RDKitMoleculeSetup 中调用 mol.HasQuery()；部分 RDKit 版本仅提供 Atom/Bond.HasQuery。"""
    if hasattr(Chem.Mol, 'HasQuery'):
        return

    def _HasQuery(self):  # noqa: N802
        for atom in self.GetAtoms():
            if atom.HasQuery():
                return True
        for bond in self.GetBonds():
            if bond.HasQuery():
                return True
        return False

    Chem.Mol.HasQuery = _HasQuery


_patch_rdkit_mol_has_query_if_missing()

import tempfile 
import os
import sys

import shutil
import contextlib

from utils.reconstruct import reconstruct_from_generated
from utils.evaluation.docking_qvina import get_random_id, BaseDockingTask


def _mgltools_autodocktools_root():
    """AutoDockTools 包根目录（含 Utilities24/）。兼容 Py3.9+ 与仅含 MGLToolsPckgs 的 mgltools 安装。"""
    try:
        import AutoDockTools as _adt
        return _adt.__path__[0]
    except Exception:
        pass
    prefixes = []
    envp = os.environ.get('CONDA_PREFIX')
    if envp:
        prefixes.append(envp)
    prefixes.append(getattr(sys, 'prefix', ''))
    for pre in prefixes:
        if not pre:
            continue
        cand = os.path.join(pre, 'MGLToolsPckgs', 'AutoDockTools')
        util = os.path.join(cand, 'Utilities24')
        if os.path.isdir(util):
            return cand
    raise RuntimeError(
        '未找到 MGLTools AutoDockTools（`Utilities24/prepare_receptor4.py`）。'
        '请安装: conda install -c bioconda mgltools，并保证 CONDA_PREFIX 指向该环境。'
    )


def _write_ob_mol_to_file(ob_mol, fname, ftype=None):
    """
    写出 OpenBabel OBMol 到文件；与 meeko.utils.obutils.writeMolecule 行为一致。
    Meeko 0.7+ 不再提供 obutils，但配体制备流程仍会在加氢/构象步骤用到写出中间 SDF。
    """
    if obutils is not None:
        obutils.writeMolecule(ob_mol, fname)
        return
    if _ob_openbabel is None:
        raise RuntimeError(
            '当前 Meeko 版本未暴露 obutils，且 OpenBabel 缺少 openbabel 底层模块，无法写出中间 SDF'
        )
    if ftype is None:
        ext = os.path.splitext(str(fname))[1].lstrip('.').lower()
        ftype = ext if ext else 'sdf'
    conv = _ob_openbabel.OBConversion()
    conv.SetOutFormat(ftype)
    if not conv.WriteFile(ob_mol, str(fname)):
        raise RuntimeError('OpenBabel WriteFile 失败: %s' % fname)


def _meeko_prepare_expects_obmol_only():
    """
    meeko<0.3（如官方 TargetDiff 推荐的 0.1.dev3、以及 0.2.x）的 prepare() 仅接受 OpenBabel OBMol
    （内部调用 mol.NumAtoms()）；传入 RDKit Mol 会报 ``'Mol' object has no attribute 'NumAtoms'``。
    meeko≥0.3 起同时支持 RDKit 与 OBMol。
    """
    try:
        from importlib.metadata import version as _pkg_version
    except ImportError:
        try:
            from importlib_metadata import version as _pkg_version
        except ImportError:
            return False
    try:
        raw = _pkg_version('meeko')
    except Exception:
        return False
    try:
        from packaging.version import parse as _parse_ver
        return _parse_ver(raw) < _parse_ver('0.3')
    except Exception:
        pass
    parts = raw.split('.')
    if len(parts) >= 2:
        try:
            return int(parts[0]) == 0 and int(parts[1]) < 3
        except ValueError:
            pass
    return False


def _protein_pdb_relpath_from_ligand(ligand_filename):
    """
    受体 PDB 相对 protein_root 的路径（CrossDocked 风格：子目录/PDB前10字符_rec.pdb）。

    若 ligand_filename 为绝对路径（常见于 .pt 里烘焙了训练机路径），不能再用
    os.path.dirname(ligand_filename) 与 protein_root 拼接：join 会丢弃 protein_root，
    导致在另一台机器上仍访问 /workspace/... 而找不到 pqr/pdb。
    """
    base = os.path.basename(ligand_filename)
    stem_pdb = base[:10] + '.pdb'
    parent = os.path.dirname(ligand_filename)
    if os.path.isabs(ligand_filename):
        rel_parent = os.path.basename(parent.rstrip(os.sep))
        if rel_parent:
            return os.path.join(rel_parent, stem_pdb)
        return stem_pdb
    if parent:
        return os.path.join(parent, stem_pdb)
    return stem_pdb


def _crossdocked_protein_root_candidates(protein_root):
    """
    批量脚本示例常写 ``--protein_root /workspace/data``，而 CrossDocked 实际在
    ``.../data/crossdocked_v1.1_rmsd1.0/<口袋>/``。依次尝试根目录与常见子目录。
    """
    primary = os.path.normpath(os.path.expanduser(str(protein_root)))
    out = [primary]
    for sub in ('crossdocked_v1.1_rmsd1.0', 'crossdocked_v1.1_rmsd1.0_pocket10'):
        cand = os.path.join(primary, sub)
        if os.path.isdir(cand):
            nc = os.path.normpath(cand)
            if nc not in out:
                out.append(nc)
    return out


def receptor_pdb_path_under_protein_roots(protein_root, ligand_filename):
    """在 protein_root 及其常见 CrossDocked 子目录下查找 ``rel`` 受体 PDB；返回首选存在路径或回退路径。"""
    rel = _protein_pdb_relpath_from_ligand(ligand_filename)
    bases = _crossdocked_protein_root_candidates(protein_root)
    for base in bases:
        p = os.path.normpath(os.path.join(base, rel))
        if os.path.isfile(p):
            return p
    return os.path.normpath(os.path.join(bases[0], rel))


def resolve_receptor_pdb_for_docking(ligand_filename, protein_root, explicit_protein_path=None):
    """
    选择实际用于对接的受体 PDB 路径。

    当 .pt 中受体为训练环境绝对路径（如 /workspace/data/...）且该路径下仍有 PDB 时，原先会
    直接用该路径；若 ``--protein_root`` 下已存在由配体路径推断出的同名受体 PDB，则优先
    使用本机数据目录（避免 pqr/pdbqt 仍写到 /workspace）。

    若用户将 ``--protein_root`` 设为数据父目录（如 ``.../data``），会自动尝试
    ``crossdocked_v1.1_rmsd1.0`` 等子目录，与单独运行时写 ``.../data/crossdocked_v1.1_rmsd1.0`` 一致。

    无配体路径（自定义口袋，ligand_filename 为 N/A）时仅使用 explicit_protein_path。
    """
    lf = str(ligand_filename).strip() if ligand_filename is not None else ''
    if lf in ('', 'N/A', 'n/a'):
        if not explicit_protein_path:
            raise ValueError('无配体文件名时必须提供 explicit_protein_path')
        return os.path.realpath(os.path.expanduser(str(explicit_protein_path)))

    cand = receptor_pdb_path_under_protein_roots(protein_root, lf)
    if os.path.isfile(cand):
        return cand

    if explicit_protein_path:
        ep = os.path.realpath(os.path.expanduser(str(explicit_protein_path)))
        if os.path.isfile(ep):
            return ep

    return cand


def _python_for_prepare_receptor():
    override = os.environ.get('ADT_PYTHON') or os.environ.get('MGLTOOLS_PYTHON')
    if override:
        exe = override.strip().split()[0]
        if os.path.isfile(exe) or shutil.which(exe):
            return exe
    for name in ('python2.7', 'python2'):
        if shutil.which(name):
            return name
    return None


def supress_stdout(func):
    def wrapper(*a, **ka):
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull):
                return func(*a, **ka)
    return wrapper


class PrepLig(object):
    def __init__(self, input_mol, mol_format):
        if mol_format == 'smi':
            self.ob_mol = pybel.readstring('smi', input_mol)
        elif mol_format == 'sdf': 
            self.ob_mol = next(pybel.readfile(mol_format, input_mol))
        else:
            raise ValueError(f'mol_format {mol_format} not supported')
        
    def addH(self, polaronly=False, correctforph=True, PH=7): 
        self.ob_mol.OBMol.AddHydrogens(polaronly, correctforph, PH)
        _write_ob_mol_to_file(self.ob_mol.OBMol, 'tmp_h.sdf')

    def gen_conf(self):
        sdf_block = self.ob_mol.write('sdf')
        rdkit_mol = Chem.MolFromMolBlock(sdf_block, removeHs=False)
        AllChem.EmbedMolecule(rdkit_mol, Chem.rdDistGeom.ETKDGv3())
        self.ob_mol = pybel.readstring('sdf', Chem.MolToMolBlock(rdkit_mol))
        _write_ob_mol_to_file(self.ob_mol.OBMol, 'conf_h.sdf')

    @supress_stdout
    def get_pdbqt(self, lig_pdbqt=None):
        # meeko 0.1.dev3 / 0.2.x：与 TargetDiff 官方一致，仅用 OBMol（勿传 RDKit Mol）
        if _meeko_prepare_expects_obmol_only():
            preparator = MoleculePreparation()
            preparator.prepare(self.ob_mol.OBMol)
            if lig_pdbqt is not None:
                preparator.write_pdbqt_file(lig_pdbqt)
                return
            return preparator.write_pdbqt_string()

        # Meeko≥0.3：可用 RDKit Mol（与 0.1.dev3 制备结果可能略有差异，便于新版 OpenBabel 栈）
        sdf_block = self.ob_mol.write('sdf')
        rdkit_mol = Chem.MolFromMolBlock(sdf_block, removeHs=False)
        if rdkit_mol is None:
            raise ValueError('RDKit 无法从 OpenBabel 分子解析 SDF')

        preparator = MoleculePreparation()
        setups = preparator.prepare(rdkit_mol)

        if isinstance(setups, list) and len(setups) == 0:
            raise RuntimeError('Meeko prepare() 未生成任何 ligand setup')

        if isinstance(setups, list) and len(setups) > 0 and _MeekoPDBQTWriter is not None:
            pdbqt_string, is_ok, err_msg = _MeekoPDBQTWriter.write_string(setups[0])
            if not is_ok:
                raise RuntimeError('Meeko PDBQT: %s' % (err_msg,))
            if lig_pdbqt is not None:
                with open(lig_pdbqt, 'w') as f:
                    f.write(pdbqt_string)
                return
            return pdbqt_string

        if lig_pdbqt is not None:
            preparator.write_pdbqt_file(lig_pdbqt)
            return
        return preparator.write_pdbqt_string()
        

class PrepProt(object): 
    def __init__(self, pdb_file): 
        self.prot = pdb_file
    
    def del_water(self, dry_pdb_file): # optional
        with open(self.prot) as f: 
            lines = [l for l in f.readlines() if l.startswith('ATOM') or l.startswith('HETATM')] 
            dry_lines = [l for l in lines if not 'HOH' in l]
        
        with open(dry_pdb_file, 'w') as f:
            f.write(''.join(dry_lines))
        self.prot = dry_pdb_file
        
    def addH(self, prot_pqr):  # call pdb2pqr
        self.prot_pqr = prot_pqr
        subprocess.Popen(['pdb2pqr30','--ff=AMBER',self.prot, self.prot_pqr],
                         stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL).communicate()

    def get_pdbqt(self, prot_pdbqt):
        prepare_receptor = os.path.join(_mgltools_autodocktools_root(), 'Utilities24/prepare_receptor4.py')
        pqr_path = getattr(self, 'prot_pqr', None) or (self.prot[:-4] + '.pqr')
        if not os.path.isfile(pqr_path):
            raise RuntimeError('prepare_receptor 需要 pqr，但文件不存在: %s' % pqr_path)
        py = _python_for_prepare_receptor()
        if py is None:
            raise RuntimeError(
                '未找到 Python2（prepare_receptor4.py 仅支持 Py2）。请安装 python2.7 或设置 ADT_PYTHON'
            )
        r = subprocess.run(
            [py, prepare_receptor, '-r', pqr_path, '-o', prot_pdbqt],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        err = (r.stderr or b'') + (r.stdout or b'')
        err_s = err.decode('utf-8', errors='replace')[-1200:]
        if r.returncode != 0:
            raise RuntimeError(
                'prepare_receptor4 失败 (exit %s): %s' % (r.returncode, err_s)
            )
        if not os.path.isfile(prot_pdbqt):
            raise RuntimeError('prepare_receptor4 未写出 pdbqt: %s' % prot_pdbqt)


class VinaDock(object): 
    def __init__(self, lig_pdbqt, prot_pdbqt): 
        self.lig_pdbqt = lig_pdbqt
        self.prot_pdbqt = prot_pdbqt
    
    def _max_min_pdb(self, pdb, buffer):
        with open(pdb, 'r') as f: 
            lines = [l for l in f.readlines() if l.startswith('ATOM') or l.startswith('HEATATM')]
            xs = [float(l[31:39]) for l in lines]
            ys = [float(l[39:47]) for l in lines]
            zs = [float(l[47:55]) for l in lines]
            print(max(xs), min(xs))
            print(max(ys), min(ys))
            print(max(zs), min(zs))
            pocket_center = [(max(xs) + min(xs))/2, (max(ys) + min(ys))/2, (max(zs) + min(zs))/2]
            box_size = [(max(xs) - min(xs)) + buffer, (max(ys) - min(ys)) + buffer, (max(zs) - min(zs)) + buffer]
            return pocket_center, box_size
    
    def get_box(self, ref=None, buffer=0):
        '''
        ref: reference pdb to define pocket. 
        buffer: buffer size to add 

        if ref is not None: 
            get the max and min on x, y, z axis in ref pdb and add buffer to each dimension 
        else: 
            use the entire protein to define pocket 
        '''
        if ref is None: 
            ref = self.prot_pdbqt
        self.pocket_center, self.box_size = self._max_min_pdb(ref, buffer)
        print(self.pocket_center, self.box_size)

    def dock(self, score_func='vina', seed=0, mode='dock', exhaustiveness=8, n_poses=1, save_pose=False, **kwargs):
        # 调用方（如 Prudent 子线程路径）可能传 timeout_sec；vina.Vina 不接受该关键字
        kwargs.pop('timeout_sec', None)
        v = Vina(sf_name=score_func, seed=seed, verbosity=0, **kwargs)
        v.set_receptor(self.prot_pdbqt)
        v.set_ligand_from_file(self.lig_pdbqt)
        v.compute_vina_maps(center=self.pocket_center, box_size=self.box_size)
        if mode == 'score_only':
            score = v.score()[0]
        elif mode == 'minimize':
            score = v.optimize()[0]
        elif mode == 'dock':
            v.dock(exhaustiveness=exhaustiveness, n_poses=n_poses)
            score = v.energies(n_poses=n_poses)[0][0]
        else:
            raise ValueError

        if not save_pose:
            return score
        else:
            if mode == 'score_only':
                pose = None
            elif mode == 'minimize':
                tmp = tempfile.NamedTemporaryFile()
                with open(tmp.name, 'w') as f:
                    v.write_pose(tmp.name, overwrite=True)
                with open(tmp.name, 'r') as f:
                    pose = f.read()

            elif mode == 'dock':
                pose = v.poses(n_poses=n_poses)
            else:
                raise ValueError
            return score, pose


class VinaDockingTask(BaseDockingTask):

    @classmethod
    def from_generated_data(cls, data, protein_root='./data/crossdocked', **kwargs):
        # load original pdb
        protein_path = receptor_pdb_path_under_protein_roots(protein_root, data.ligand_filename)
        ligand_rdmol = reconstruct_from_generated(data.clone())
        return cls(protein_path, ligand_rdmol, **kwargs)

    @classmethod
    def from_original_data(cls, data, ligand_root='./data/crossdocked_pocket10', protein_root='./data/crossdocked',
                           **kwargs):
        protein_path = receptor_pdb_path_under_protein_roots(protein_root, data.ligand_filename)

        ligand_path = (
            data.ligand_filename
            if os.path.isabs(data.ligand_filename)
            else os.path.join(ligand_root, data.ligand_filename)
        )
        ligand_rdmol = next(iter(Chem.SDMolSupplier(ligand_path)))
        return cls(protein_path, ligand_rdmol, **kwargs)

    @classmethod
    def from_generated_mol(cls, ligand_rdmol, ligand_filename, protein_root='./data/crossdocked', **kwargs):
        # 调用方可能传入 protein_path=...（显式受体）；若留在 kwargs 里会与 cls 的首位位置参数重名
        explicit_prot = kwargs.pop('protein_path', None)
        protein_path = resolve_receptor_pdb_for_docking(
            ligand_filename, protein_root, explicit_protein_path=explicit_prot
        )
        return cls(protein_path, ligand_rdmol, **kwargs)

    def __init__(self, protein_path, ligand_rdmol, tmp_dir='./tmp', center=None,
                 size_factor=1., buffer=5.0):
        super().__init__(protein_path, ligand_rdmol)
        # self.conda_env = conda_env
        self.tmp_dir = os.path.realpath(tmp_dir)
        os.makedirs(tmp_dir, exist_ok=True)

        self.task_id = get_random_id()
        self.receptor_id = self.task_id + '_receptor'
        self.ligand_id = self.task_id + '_ligand'

        self.receptor_path = protein_path
        self.ligand_path = os.path.join(self.tmp_dir, self.ligand_id + '.sdf')

        self.recon_ligand_mol = ligand_rdmol
        ligand_rdmol = Chem.AddHs(ligand_rdmol, addCoords=True)

        sdf_writer = Chem.SDWriter(self.ligand_path)
        sdf_writer.write(ligand_rdmol)
        sdf_writer.close()
        self.ligand_rdmol = ligand_rdmol

        pos = ligand_rdmol.GetConformer(0).GetPositions()
        if center is None:
            self.center = (pos.max(0) + pos.min(0)) / 2
        else:
            self.center = center

        if size_factor is None:
            self.size_x, self.size_y, self.size_z = 20, 20, 20
        else:
            self.size_x, self.size_y, self.size_z = (pos.max(0) - pos.min(0)) * size_factor + buffer

        self.proc = None
        self.results = None
        self.output = None
        self.error_output = None
        self.docked_sdf_path = None

    def run(self, mode='dock', exhaustiveness=8, n_poses=1, **kwargs):
        ligand_pdbqt = self.ligand_path[:-4] + '.pdbqt'
        protein_pqr = self.receptor_path[:-4] + '.pqr'
        protein_pdbqt = self.receptor_path[:-4] + '.pdbqt'

        lig = PrepLig(self.ligand_path, 'sdf')
        lig.get_pdbqt(ligand_pdbqt)

        # 检查 PDBQT 文件是否为空（分子可能有价态错误导致无法正确转换）
        if not os.path.exists(ligand_pdbqt) or os.path.getsize(ligand_pdbqt) == 0:
            raise RuntimeError('PDBQT 配体文件为空或不存在，分子可能有价态错误或重建失败')
        with open(ligand_pdbqt, 'r') as f:
            content = f.read()
            content_stripped = content.strip()
            if not content_stripped:
                raise RuntimeError('PDBQT 配体文件内容为空')
            # 严格检查：必须有 ATOM/HETATM 行，且至少有一个有效坐标行（包含数字坐标）
            has_atom = False
            valid_coords = False
            for line in content.splitlines():
                if line.startswith('ATOM') or line.startswith('HETATM'):
                    has_atom = True
                    # 检查行长度和坐标字段（PDBQT 格式：x y z 在第 7-9 列，约字符位置 30-54）
                    parts = line.split()
                    if len(parts) >= 6:
                        try:
                            # 尝试解析坐标字段（通常是第 6,7,8 个字段，0-indexed）
                            x, y, z = float(parts[5]), float(parts[6]), float(parts[7])
                            valid_coords = True
                        except (ValueError, IndexError):
                            pass
            if not has_atom:
                raise RuntimeError('PDBQT 配体文件缺少 ATOM/HETATM 记录')
            if not valid_coords:
                raise RuntimeError('PDBQT 配体文件缺少有效的原子坐标')

        prot = PrepProt(self.receptor_path)
        if not os.path.exists(protein_pqr):
            prot.addH(protein_pqr)
        if not os.path.exists(protein_pdbqt):
            prot.get_pdbqt(protein_pdbqt)

        dock = VinaDock(ligand_pdbqt, protein_pdbqt)
        dock.pocket_center, dock.box_size = self.center, [self.size_x, self.size_y, self.size_z]
        score, pose = dock.dock(score_func='vina', mode=mode, exhaustiveness=exhaustiveness, n_poses=n_poses, save_pose=True, **kwargs)
        return [{'affinity': score, 'pose': pose}]


# if __name__ == '__main__':
#     lig_pdbqt = 'data/lig.pdbqt'
#     mol_file = 'data/1a4k_ligand.sdf'
#     a = PrepLig(mol_file, 'sdf')
#     # mol_file = 'CC(=C)C(=O)OCCN(C)C'
#     # a = PrepLig(mol_file, 'smi')
#     a.addH()
#     a.gen_conf()
#     a.get_pdbqt(lig_pdbqt)
#
#     prot_file = 'data/1a4k_protein_chainAB.pdb'
#     prot_dry = 'data/protein_dry.pdb'
#     prot_pqr = 'data/protein.pqr'
#     prot_pdbqt = 'data/protein.pdbqt'
#     b = PrepProt(prot_file)
#     b.del_water(prot_dry)
#     b.addH(prot_pqr)
#     b.get_pdbqt(prot_pdbqt)
#
#     dock = VinaDock(lig_pdbqt, prot_pdbqt)
#     dock.get_box()
#     dock.dock()
    
