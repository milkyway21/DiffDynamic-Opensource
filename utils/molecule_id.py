# -*- coding: utf-8 -*-
"""
分子 ID / 蛋白 ID 字符串工具（与 evaluate 脚本逻辑一致）。

独立模块，避免 sample_diffusion 等为复用这两个函数而导入整块 evaluate + 对接依赖（meeko 等）。
"""

from pathlib import Path
from datetime import datetime


def extract_protein_id(ligand_filename=None, protein_filename=None):
    """
    从 ligand_filename 或 protein_filename 中提取蛋白质 ID（通常 4 位字母数字，如 1A4K）。
    """
    if protein_filename:
        try:
            protein_basename = Path(protein_filename).stem
            protein_id = protein_basename.split('_')[0].upper()
            if len(protein_id) >= 4 and protein_id.isalnum():
                return protein_id[:4]
        except Exception:
            pass

    if ligand_filename:
        try:
            ligand_basename = Path(ligand_filename).stem
            protein_id = ligand_basename.split('_')[0].upper()
            if len(protein_id) >= 4 and protein_id.isalnum():
                return protein_id[:4]
        except Exception:
            pass

    return 'UNKNOWN'


def generate_molecule_id(protein_id, generation_time, score):
    """
    生成分子身份证：蛋白质ID_生成时间_评分（小数点换成 p）。
    """
    if protein_id is None:
        protein_id = 'UNKNOWN'
    protein_id_str = str(protein_id).upper().replace('/', '_').replace('\\', '_').replace(':', '_')
    if len(protein_id_str) > 4:
        protein_id_str = protein_id_str[:4]

    if isinstance(generation_time, datetime):
        time_str = generation_time.strftime('%Y%m%d_%H%M%S')
    elif isinstance(generation_time, str):
        time_str = generation_time
    else:
        time_str = datetime.now().strftime('%Y%m%d_%H%M%S')

    score_formatted = f"{float(score):.2f}"
    score_str = score_formatted.replace('.', 'p') if '.' in score_formatted else f"{score_formatted}p00"

    return f"{protein_id_str}_{time_str}_{score_str}"
