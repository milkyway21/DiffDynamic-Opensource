"""Inputs responsibilities extracted from evaluate_pocket_quality.py."""

from diffdynamic.analysis.pocket.state import (
    AllChem,
    Chem,
    Path,
    REPO_ROOT,
    np,
    os,
    reconstruct,
    torch,
    trans,
)


def load_pt_file(pt_path):
    """加载 .pt 文件"""
    try:
        data = _torch_load_legacy(pt_path, map_location="cpu")
        return data
    except Exception as e:
        print(f"❌ 加载 .pt 文件失败: {pt_path}: {e}")
        return None


def _torch_load_legacy(path, map_location="cpu"):
    """加载项目内任意 .pt（含自定义类 pickle）。PyTorch 2.6+ 默认 weights_only=True 会失败。

    DD_PT_STRICT_LOAD=1（服务端评估用户上传等不可信 .pt 时设置）时禁止 pickle
    回退：只允许 weights_only 严格加载，失败直接抛错，防止反序列化执行恶意代码。
    """
    if torch is None:
        raise RuntimeError("需要 torch")
    from diffdynamic.io.normalized_pt import restore_normalized_result

    if os.environ.get("DD_PT_STRICT_LOAD") == "1":
        try:
            return restore_normalized_result(
                torch.load(path, map_location=map_location, weights_only=True)
            )
        except Exception as err:
            raise ValueError(
                f"不可信 .pt（用户上传等非本仓库生成文件）只允许严格加载，加载失败: {err}"
            ) from err
    try:
        return restore_normalized_result(
            torch.load(path, map_location=map_location, weights_only=False)
        )
    except TypeError:
        return restore_normalized_result(torch.load(path, map_location=map_location))


def _to_numpy(data):
    """将 torch tensor 或 numpy 转为 numpy"""
    if torch is not None and isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    return np.array(data)


def reconstruct_molecule_from_pt(pos, v, atom_mode="add_aromatic"):
    """从 .pt 中的坐标和原子类型重建 RDKit 分子"""
    if reconstruct is None or trans is None:
        return None
    try:
        pos_array = _to_numpy(pos)
        if pos_array.ndim == 1:
            pos_array = pos_array.reshape(-1, 3)

        v_tensor = (
            torch.tensor(v) if not isinstance(v, torch.Tensor) else v.detach().cpu()
        )
        if v_tensor.dim() > 1:
            v_tensor = v_tensor.argmax(dim=-1)

        atom_numbers = trans.get_atomic_number_from_index(v_tensor, mode=atom_mode)
        aromatic_flags = trans.is_aromatic_from_index(v_tensor, mode=atom_mode)

        mol = reconstruct.reconstruct_from_generated(
            pos_array.tolist(),
            atom_numbers,
            aromatic_flags,
            basic_mode=(atom_mode == "basic"),
        )
        return mol
    except Exception:
        return None


def load_molecules_from_pt(pt_path, atom_mode="add_aromatic", max_mols=None):
    """
    从 .pt 文件加载并重建所有分子

    Returns:
        list: [(mol, pos_array), ...]
    """
    data = load_pt_file(pt_path)
    if data is None:
        return []

    pred_pos = data.get("pred_ligand_pos", [])
    pred_v = data.get("pred_ligand_v", [])

    if not pred_pos or not pred_v:
        return []

    results = []
    for i, (pos, v) in enumerate(zip(pred_pos, pred_v)):
        if max_mols and len(results) >= max_mols:
            break
        mol = reconstruct_molecule_from_pt(pos, v, atom_mode)
        pos_arr = _to_numpy(pos)
        if pos_arr.ndim == 1:
            pos_arr = pos_arr.reshape(-1, 3)
        if mol is not None:
            results.append((mol, pos_arr))

    return results


def _mol_and_pos_from_rdkit_mol(mol):
    """从 RDKit 分子取首构象坐标；若无构象则尝试 Embed。"""
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        pass
    if mol.GetNumConformers() < 1 and AllChem is not None:
        try:
            AllChem.EmbedMolecule(mol, randomSeed=42)
        except Exception:
            return None
    if mol.GetNumConformers() < 1:
        return None
    conf = mol.GetConformer()
    pos = np.zeros((mol.GetNumAtoms(), 3), dtype=np.float32)
    for i in range(mol.GetNumAtoms()):
        p = conf.GetAtomPosition(i)
        pos[i, 0], pos[i, 1], pos[i, 2] = p.x, p.y, p.z
    return mol, pos


def load_molecules_from_ligand_paths(ligand_path, max_mols=None):
    """
    从自定义配体文件或目录加载 (mol, pos)，用于无 .pt 的口袋质量评估。

    支持：单个/多个 .sdf、.mol；目录下所有 *.sdf / *.mol。

    Returns:
        list[(mol, pos_array), ...]
    """
    if Chem is None:
        return []
    lp = Path(ligand_path)
    files = []
    if lp.is_dir():
        files = sorted(lp.glob("*.sdf")) + sorted(lp.glob("*.mol"))
    elif lp.is_file():
        files = [lp]
    else:
        return []

    out = []
    for fp in files:
        suf = fp.suffix.lower()
        if suf == ".sdf":
            suppl = Chem.SDMolSupplier(str(fp), sanitize=False, removeHs=False)
            for mol in suppl:
                got = _mol_and_pos_from_rdkit_mol(mol)
                if got:
                    out.append(got)
                    if max_mols and len(out) >= max_mols:
                        return out
        elif suf == ".mol":
            mol = Chem.MolFromMolFile(str(fp), sanitize=False, removeHs=False)
            got = _mol_and_pos_from_rdkit_mol(mol)
            if got:
                out.append(got)
                if max_mols and len(out) >= max_mols:
                    return out
    return out


def resolve_vina_outputs_dir(
    vina_outputs_dir=None, pt_path=None, ligand_path=None, custom_pocket_pdb=None
):
    """
    为想法 A 定位含 eval_* 子目录的 outputs 根目录（与 batch 输出一致）。
    显式 vina_outputs_dir 优先；否则从 .pt 同目录、配体/口袋路径旁推测。
    """
    if vina_outputs_dir:
        p = Path(vina_outputs_dir)
        if p.is_dir():
            return p.resolve()
        return None
    if pt_path:
        pp = Path(pt_path).resolve()
        if pp.is_file():
            return pp.parent
        return None
    candidates = []
    if custom_pocket_pdb:
        base = Path(custom_pocket_pdb).resolve().parent
        candidates.extend([base / "outputs", base])
    if ligand_path:
        lp = Path(ligand_path).resolve()
        base = lp.parent if lp.is_file() else lp
        candidates.extend([base / "outputs", base])
    seen = set()
    for c in candidates:
        c = c.resolve()
        if c in seen or not c.is_dir():
            continue
        seen.add(c)
        if list(c.glob("eval_*")):
            return c
    for c in candidates:
        c = c.resolve()
        if c.is_dir():
            return c
    return None


def resolve_receptor_pdb_from_pt(pt_path, protein_root=None):
    """
    从 .pt 的 protein_filename / ligand_filename 解析本地匹配受体 PDB。
    常见容器路径 ``/workspace/data/crossdocked_...`` 会映射到仓库
    ``data/crossdocked_pocket10_test_only``。
    """
    pt_path = Path(pt_path)
    if not pt_path.is_file():
        return None
    try:
        raw = load_pt_file(pt_path)
    except Exception:
        return None
    data = raw.get("data") if isinstance(raw, dict) else None
    protein_fn = getattr(data, "protein_filename", None) if data is not None else None
    ligand_fn = getattr(data, "ligand_filename", None) if data is not None else None
    if isinstance(data, dict):
        protein_fn = protein_fn or data.get("protein_filename")
        ligand_fn = ligand_fn or data.get("ligand_filename")
    if not protein_fn:
        return None

    pf = Path(str(protein_fn))
    if pf.is_file():
        return str(pf.resolve())

    search_roots = []
    if protein_root:
        search_roots.append(Path(protein_root))
    search_roots.extend(
        [
            REPO_ROOT / "data" / "crossdocked_pocket10_test_only",
            REPO_ROOT / "data" / "crossdocked_v1.1_rmsd1.0_pocket10",
            REPO_ROOT / "data",
        ]
    )
    for extra in (Path("/data/ye/protein-ligand"), Path("/data/ye")):
        if extra.is_dir():
            search_roots.append(extra)

    rel_bits = []
    if ligand_fn:
        lig_parent = Path(str(ligand_fn)).parent
        if str(lig_parent) not in (".", ""):
            rel_bits.append(lig_parent / pf.name)
    if len(pf.parts) >= 2:
        rel_bits.append(Path(*pf.parts[-2:]))
    rel_bits.append(Path(pf.name))

    seen = set()
    for root in search_roots:
        if not root or not Path(root).exists():
            continue
        root = Path(root)
        for rel in rel_bits:
            cand = (root / rel).resolve()
            key = str(cand)
            if key in seen:
                continue
            seen.add(key)
            if cand.is_file():
                return str(cand)
    return None


def resolve_ligand_sdf_from_pt(pt_path, protein_root=None):
    """Resolve CrossDocked-style native ligand SDF path from .pt ligand_filename."""
    pt_path = Path(pt_path)
    if not pt_path.is_file():
        return None
    try:
        raw = load_pt_file(pt_path)
    except Exception:
        return None
    data = raw.get("data") if isinstance(raw, dict) else None
    ligand_fn = getattr(data, "ligand_filename", None) if data is not None else None
    if isinstance(data, dict):
        ligand_fn = ligand_fn or data.get("ligand_filename")
    if not ligand_fn:
        return None
    lf = Path(str(ligand_fn))
    if lf.is_file():
        return str(lf.resolve())

    search_roots = []
    if protein_root:
        search_roots.append(Path(protein_root))
    search_roots.extend(
        [
            REPO_ROOT / "data" / "crossdocked_pocket10_test_only",
            REPO_ROOT / "data" / "crossdocked_v1.1_rmsd1.0_pocket10",
            REPO_ROOT / "data",
        ]
    )
    rel_bits = []
    if len(lf.parts) >= 2:
        rel_bits.append(Path(*lf.parts[-2:]))
    rel_bits.append(Path(lf.name))
    for root in search_roots:
        if not root or not Path(root).exists():
            continue
        for rel in rel_bits:
            cand = Path(root) / rel
            if cand.is_file():
                return str(cand.resolve())
        hits = list(Path(root).rglob(lf.name))
        if hits:
            return str(hits[0].resolve())
    return None
