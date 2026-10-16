from diffdynamic.io.paths import project_root

"""Shared imports, constants and process configuration for diffdynamic.chemistry.reconstruction."""

import itertools  # 导入迭代工具，用于组合遍历。


import numpy as np  # 导入 NumPy。


from rdkit.Chem import AllChem as Chem  # 从 RDKit 导入化学模块。


from rdkit import Geometry  # 导入几何工具。


from openbabel import openbabel as ob  # 导入 OpenBabel 接口。


from scipy.spatial.distance import pdist  # 导入点对距离计算。


from scipy.spatial.distance import squareform  # 导入距离矩阵转换工具。


UPGRADE_BOND_ORDER = {
    Chem.BondType.SINGLE: Chem.BondType.DOUBLE,
    Chem.BondType.DOUBLE: Chem.BondType.TRIPLE,
}
