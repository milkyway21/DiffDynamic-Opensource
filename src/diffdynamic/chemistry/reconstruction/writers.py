"""Writers responsibilities extracted from reconstruct.py."""

from diffdynamic.chemistry.reconstruction.state import (
    Chem,
    Geometry,
)


def save_positions_only_to_sdf(xyz, atomic_nums, output_path):
    """将仅含原子位置的分子保存为 SDF 文件（无键连接）。

    用于扩散过程早期（t>bond_threshold）的可视化，此时原子尚未形成合理键结构。

    Args:
        xyz: 坐标列表或数组，形状 [num_atoms, 3]
        atomic_nums: 原子序数列表或数组
        output_path: 输出 SDF 文件路径
    """
    from rdkit.Chem import RWMol

    if hasattr(xyz, "tolist"):
        xyz = xyz.tolist()
    if hasattr(atomic_nums, "tolist"):
        atomic_nums = atomic_nums.tolist()

    rwmol = RWMol()
    for an in atomic_nums:
        rwmol.AddAtom(Chem.Atom(int(an)))

    conf = Chem.Conformer(rwmol.GetNumAtoms())
    for i, (coord, an) in enumerate(zip(xyz, atomic_nums)):
        conf.SetAtomPosition(
            i, Geometry.Point3D(float(coord[0]), float(coord[1]), float(coord[2]))
        )
    rwmol.AddConformer(conf, assignId=True)

    mol = rwmol.GetMol()
    sdf_block = Chem.MolToMolBlock(mol)
    with open(output_path, "w") as f:
        f.write(sdf_block)
        f.write("$$$$\n")
