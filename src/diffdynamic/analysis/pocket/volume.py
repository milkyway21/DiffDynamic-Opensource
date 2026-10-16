"""Volume responsibilities extracted from evaluate_pocket_quality.py."""

from diffdynamic.analysis.pocket.state import (
    Chem,
    IDEA_H_FPOCKET_PROTEIN_FALLBACK_OPTIMAL_MAX,
    IDEA_H_FPOCKET_PROTEIN_FALLBACK_OPTIMAL_MIN,
    IDEA_H_FPOCKET_PROTEIN_FALLBACK_ZERO_ABOVE,
    IDEA_H_FPOCKET_PROTEIN_FALLBACK_ZERO_BELOW,
    Path,
    np,
    os,
    re,
    shutil,
    subprocess,
    tempfile,
)


def _vdw_radius_from_atomic_num(z):
    z = int(z)
    return {
        1: 1.10,
        6: 1.70,
        7: 1.55,
        8: 1.52,
        9: 1.47,
        15: 1.80,
        16: 1.80,
        17: 1.75,
        35: 1.85,
        53: 1.98,
    }.get(z, 1.80)


def _idea_h_score_from_volume(
    volume_ang3, optimal_min, optimal_max, zero_below, zero_above
):
    """想法 H 体积分段线性评分（MC 与 FPocket 使用不同 optimal/zero 参数）。"""
    zb, za = float(zero_below), float(zero_above)
    v = float(volume_ang3)
    if v < zb or v >= za:
        score = 0.0
    elif v < optimal_min:
        denom = optimal_min - zb
        score = (v - zb) / denom if denom > 0 else 0.0
    elif v <= optimal_max:
        score = 1.0
    else:
        denom = za - optimal_max
        score = max(0.0, 1.0 - (v - optimal_max) / denom) if denom > 0 else 0.0
    if score >= 0.8:
        label = "high"
    elif score >= 0.5:
        label = "medium"
    else:
        label = "low"
    return min(1.0, score), label


def _parse_fpocket_info_txt(text):
    """解析 FPocket 的 *_info.txt 中各 Pocket 块（Volume、Score、Druggability 等）。"""
    pockets = []
    rx_hdr = re.compile(r"^Pocket\s+(\d+)\s*:\s*$", re.MULTILINE)

    def _grab(block, pat):
        mm = re.search(pat, block, re.MULTILINE | re.IGNORECASE)
        if not mm:
            return None
        try:
            return float(mm.group(1))
        except (TypeError, ValueError):
            return None

    for m in rx_hdr.finditer(text):
        start = m.end()
        m2 = rx_hdr.search(text, start)
        end = m2.start() if m2 else len(text)
        block = text[start:end]
        pid = int(m.group(1))
        pockets.append(
            {
                "pocket_id": pid,
                "score": _grab(block, r"^\s*Score\s*:\s*([0-9.eE+-]+)\s*$"),
                "druggability": _grab(
                    block, r"^\s*Druggability\s+Score\s*:\s*([0-9.eE+-]+)\s*$"
                ),
                "n_alpha_spheres": _grab(
                    block, r"^\s*Number\s+of\s+Alpha\s+Spheres\s*:\s*([0-9.eE+-]+)\s*$"
                ),
                "volume": _grab(block, r"^\s*Volume\s*:\s*([0-9.eE+-]+)\s*$"),
                "hydrophobicity_score": _grab(
                    block, r"^\s*Hydrophobicity\s+score\s*:\s*([0-9.eE+-]+)\s*$"
                ),
            }
        )
    return pockets


def resolve_fpocket_executable(fpocket_cmd="fpocket"):
    """
    解析 fpocket 可执行文件路径：--fpocket_cmd、FPOCKET_CMD 环境变量、PATH、CONDA_PREFIX/bin。
    找不到则返回 None。
    """

    def _try(s):
        if not s or not str(s).strip():
            return None
        s = str(s).strip()
        p = Path(s)
        if p.is_file() and os.access(p, os.X_OK):
            return str(p.resolve())
        w = shutil.which(s)
        return w

    cmd = (fpocket_cmd or "fpocket").strip() or "fpocket"
    x = _try(cmd)
    if x:
        return x
    x = _try(os.environ.get("FPOCKET_CMD", ""))
    if x:
        return x
    conda = os.environ.get("CONDA_PREFIX", "").strip()
    if conda:
        for name in ("fpocket", "FPocket"):
            cp = Path(conda) / "bin" / name
            if cp.is_file() and os.access(cp, os.X_OK):
                return str(cp.resolve())
    for name in ("fpocket", "FPocket"):
        w = shutil.which(name)
        if w:
            return w
    return None


def run_fpocket_on_pdb(pdb_path, fpocket_cmd="fpocket", timeout=600):
    """
    在临时目录中复制 PDB 并运行 fpocket，避免多进程评估时共用同一 ``*_out`` 目录互相覆盖。

    Returns:
        (ok, message, pockets_list)
    """
    pdb_path = Path(pdb_path).resolve()
    if not pdb_path.is_file():
        return False, f"PDB 不存在: {pdb_path}", []
    exe = resolve_fpocket_executable(fpocket_cmd)
    if not exe:
        return (
            False,
            "未找到 fpocket 可执行文件。可选: conda install -c conda-forge fpocket；"
            "或 export FPOCKET_CMD=/绝对路径/fpocket；或 --fpocket_cmd 指向可执行文件",
            [],
        )
    stem = pdb_path.stem
    with tempfile.TemporaryDirectory(prefix="fpocket_") as td:
        td = Path(td)
        work_pdb = td / f"{stem}.pdb"
        shutil.copy2(pdb_path, work_pdb)
        cmd = [exe, "-f", str(work_pdb)]
        try:
            r = subprocess.run(
                cmd, cwd=str(td), capture_output=True, text=True, timeout=int(timeout)
            )
        except FileNotFoundError:
            return False, f"无法执行（路径失效）: {exe}", []
        except subprocess.TimeoutExpired:
            return False, f"fpocket 超时 ({timeout}s)", []
        if r.returncode != 0:
            err = (r.stderr or r.stdout or "")[:800]
            return False, f"fpocket 退出码 {r.returncode}: {err}", []
        out_dir = td / f"{stem}_out"
        info_path = out_dir / f"{stem}_info.txt"
        if not info_path.is_file():
            infos = sorted(out_dir.glob("*_info.txt")) if out_dir.is_dir() else []
            info_path = infos[0] if infos else None
        if info_path is None or not info_path.is_file():
            return False, "未找到 FPocket 输出的 *_info.txt", []
        text = info_path.read_text(encoding="utf-8", errors="replace")
        pockets = _parse_fpocket_info_txt(text)
    return True, "ok", pockets


def _fpocket_pick_pocket(pockets, pocket_index_1based):
    """pocket_index_1based：与 info.txt 中 Pocket N 的 N 一致（默认 1）。"""
    if not pockets:
        return None
    want = int(pocket_index_1based)
    for p in pockets:
        if p.get("pocket_id") == want:
            return p
    idx = want - 1
    if 0 <= idx < len(pockets):
        return pockets[idx]
    return pockets[0]


def _write_ligands_multimodel_pdb(molecules_with_pos, out_path, max_models=50):
    """
    将多条 (mol, pos) 写成**单结构 PDB**（仅 ATOM + CRYST1 + END），供 FPocket 读取。

    FPocket 对输入常按「蛋白」解析：会忽略/剥离 HETATM，且对 MODEL/ENDMDL 与块内 END 的兼容性差，
    易出现 “contains no atoms”。此处将 HETATM 改为 ATOM、去掉 RDKit 块中的 END/CONECT，
    每条构象用不同残基序号拼成一条链，避免空文件。
    """
    if Chem is None:
        return False, "RDKit 未安装，无法写配体 PDB"
    try:
        from rdkit.Geometry import Point3D
    except ImportError:
        return False, "RDKit.Geometry 不可用"

    all_lines = []
    coords_box = []
    global_serial = 1
    n_mols_written = 0

    for mi, (mol, pos) in enumerate(molecules_with_pos):
        if n_mols_written >= max_models:
            break
        if mol is None or pos is None:
            continue
        pos_arr = np.asarray(pos, dtype=np.float64)
        if pos_arr.ndim == 1:
            pos_arr = pos_arr.reshape(-1, 3)
        n_atom = mol.GetNumAtoms()
        if pos_arr.shape[0] != n_atom:
            continue
        mol_h = Chem.Mol(mol)
        conf = Chem.Conformer(mol_h.GetNumAtoms())
        for i in range(mol_h.GetNumAtoms()):
            x, y, z = pos_arr[i]
            conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
        mol_h.RemoveAllConformers()
        mol_h.AddConformer(conf)
        block = Chem.MolToPDBBlock(mol_h)
        resseq = n_mols_written + 1
        for line in block.splitlines():
            if len(line) < 54:
                continue
            rec = line[:6]
            if rec.startswith("HETATM"):
                line = "ATOM  " + line[6:]
            elif not rec.startswith("ATOM"):
                continue
            # PDB 列 23–26：残基序号（1-based 列号）；每条构象一个残基，避免重叠
            new_line = line[:22] + f"{resseq:4d}" + line[26:]
            new_line = new_line[:6] + f"{global_serial:5d}" + new_line[11:]
            all_lines.append(new_line)
            try:
                coords_box.append(
                    (
                        float(new_line[30:38]),
                        float(new_line[38:46]),
                        float(new_line[46:54]),
                    )
                )
            except ValueError:
                pass
            global_serial += 1
        n_mols_written += 1

    if not all_lines:
        return (
            False,
            "无有效 (mol,pos) 可写入配体 PDB（或 RDKit 未输出 ATOM/HETATM 行）",
        )

    if coords_box:
        xs = [c[0] for c in coords_box]
        ys = [c[1] for c in coords_box]
        zs = [c[2] for c in coords_box]
        margin = 50.0
        a = max(xs) - min(xs) + 2.0 * margin
        b = max(ys) - min(ys) + 2.0 * margin
        c = max(zs) - min(zs) + 2.0 * margin
        a, b, c = max(a, 80.0), max(b, 80.0), max(c, 80.0)
    else:
        a = b = c = 100.0
    cryst1 = f"CRYST1{a:9.3f}{b:9.3f}{c:9.3f}  90.00  90.00  90.00 P 1           1\n"
    text = cryst1 + "\n".join(all_lines) + "\nEND\n"

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(text, encoding="utf-8")
    return True, ""


def _sample_uniform_in_ball(center, radius, n, rng):
    """n 个均匀分布于球内的 3D 点（Marsaglia 方向 + r∝u^(1/3)）。"""
    center = np.asarray(center, dtype=np.float64).reshape(1, 3)
    u = rng.uniform(0.0, 1.0, size=n)
    v = rng.normal(size=(n, 3))
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    v = v / norms
    rad = radius * (u ** (1.0 / 3.0))
    return center + rad[:, None] * v


def _load_protein_heavy_coords(protein_pdb):
    """
    Load protein heavy-atom coordinates + atomic numbers from a PDB file.

    Prefers Bio.PDB; falls back to fixed-column PDB parsing.
    Returns (coords (N,3), atom_nums (N,), err_msg_or_None).
    """
    _SYM_Z = {
        "C": 6,
        "N": 7,
        "O": 8,
        "S": 16,
        "P": 15,
        "SE": 34,
        "MG": 12,
        "ZN": 30,
        "CA": 20,
        "FE": 26,
        "CL": 17,
        "NA": 11,
        "F": 9,
        "BR": 35,
        "I": 53,
        "K": 19,
        "MN": 25,
        "CU": 29,
    }
    prot = Path(protein_pdb).expanduser().resolve()
    if not prot.is_file():
        return None, None, f"蛋白 PDB 不存在: {prot}"

    coords_list = []
    z_list = []

    try:
        from Bio.PDB import PDBParser

        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("prot", str(prot))
        for atom in structure.get_atoms():
            el = (atom.element or "").strip().upper()
            if not el or el in ("H", "D"):
                continue
            try:
                pos = np.asarray(atom.coord, dtype=np.float64)
            except Exception:
                continue
            coords_list.append(pos)
            z_list.append(int(_SYM_Z.get(el, 6)))
    except Exception:
        coords_list = []
        z_list = []

    if not coords_list:
        try:
            text = prot.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            return None, None, f"无法读取蛋白 PDB: {exc}"
        for line in text.splitlines():
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            if len(line) < 54:
                continue
            el_col = line[76:78].strip().upper() if len(line) >= 78 else ""
            name = line[12:16].strip().upper()
            if el_col in ("H", "D"):
                continue
            if not el_col and name.startswith("H"):
                continue
            el = el_col or "".join(c for c in name if c.isalpha())[:2].upper()
            if el in ("H", "D"):
                continue
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                zc = float(line[46:54])
            except ValueError:
                continue
            coords_list.append(np.array([x, y, zc], dtype=np.float64))
            z_list.append(int(_SYM_Z.get(el, 6)))

    if not coords_list:
        return None, None, f"蛋白 PDB 无重原子坐标: {prot}"

    coords = np.vstack(coords_list).astype(np.float64)
    atom_nums = np.asarray(z_list, dtype=np.int32)
    return coords, atom_nums, None


def _evaluate_idea_h_mc_only(
    molecules_with_pos,
    optimal_min=350,
    optimal_max=750,
    zero_below=80.0,
    zero_above=2000.0,
    centroid_radius=10.0,
    n_mc_samples=20000,
    as_fallback=False,
):
    """
    想法H 回退：质心均值 + 球内配体 vdW 占据 × 球体积。

    注意：这不是真实口袋空腔体积（忽略蛋白、随配体堆叠膨胀），仅在无受体 PDB 时使用。
    """
    from diffdynamic.analysis.pocket.clustering import _collect_atom_data

    if not molecules_with_pos:
        return {
            "score": 0.0,
            "volume_ang3": None,
            "volume_method": None,
            "quality_label": "unknown",
            "success": False,
            "message": "无分子数据",
        }

    atom_data = _collect_atom_data(molecules_with_pos)
    if atom_data is None or len(atom_data["centroids"]) < 1:
        return {
            "score": 0.0,
            "volume_ang3": None,
            "volume_method": None,
            "quality_label": "unknown",
            "success": False,
            "message": "无分子质心",
        }

    centroids = np.asarray(atom_data["centroids"], dtype=np.float64)
    ref_center = centroids.mean(axis=0)
    n_centroids = len(centroids)
    r = float(centroid_radius)
    v_ball = (4.0 / 3.0) * np.pi * (r**3)

    coords = np.asarray(atom_data["coords"], dtype=np.float64)
    atom_nums = np.asarray(atom_data["atom_nums"], dtype=np.int32)
    if len(coords) < 1:
        return {
            "score": 0.0,
            "volume_ang3": None,
            "volume_method": None,
            "quality_label": "unknown",
            "success": False,
            "message": "无原子坐标",
        }

    radii = np.array(
        [_vdw_radius_from_atomic_num(z) for z in atom_nums], dtype=np.float64
    )

    rng = np.random.default_rng(42)
    samples = _sample_uniform_in_ball(ref_center, r, n_mc_samples, rng)

    chunk = 2048
    n_hit = 0
    for s in range(0, n_mc_samples, chunk):
        blk = samples[s : s + chunk]
        d = (
            np.linalg.norm(blk[:, None, :] - coords[None, :, :], axis=2)
            - radii[None, :]
        )
        n_hit += int((d < 0.0).any(axis=1).sum())

    frac = n_hit / float(n_mc_samples)
    volume_ang3 = float(frac * v_ball)
    volume_method = (
        "mean_centroid_10A_ligand_mc_fallback"
        if as_fallback
        else "mean_centroid_10A_ligand_mc"
    )

    zb, za = float(zero_below), float(zero_above)
    score, label = _idea_h_score_from_volume(
        volume_ang3, optimal_min, optimal_max, zb, za
    )

    return {
        "score": score,
        "volume_ang3": volume_ang3,
        "volume_method": volume_method,
        "quality_label": label,
        "success": True,
        "optimal_vol_min": optimal_min,
        "optimal_vol_max": optimal_max,
        "vol_score_zero_below": zb,
        "vol_score_zero_above": za,
        "centroid_radius": r,
        "n_centroids": n_centroids,
        "reference_centroid": ref_center.tolist(),
        "single_sphere_ref_volume_ang3": float(v_ball),
        "ligand_occupancy_fraction": float(frac),
        "cavity_fraction": None,
        "n_protein_heavy": None,
        "protein_probe_ang": None,
        "ligand_proximity_ang": None,
        "fpocket_protein_pdb": None,
        "fpocket_protein_volume_ang3": None,
        "fpocket_ligand_volume_ang3": None,
        "fpocket_protein_pocket_meta": None,
        "fpocket_ligand_pocket_meta": None,
        "fpocket_ligand_success": None,
        "fpocket_skipped": False,
        "fpocket_skip_reason": None,
        "fpocket_resolved_executable": None,
        "volume_score_band": "mc_centroid_10A_fallback"
        if as_fallback
        else "mc_centroid_10A",
        "message": (
            f"{'[fallback] ' if as_fallback else ''}"
            f"配体占据体积≈{volume_ang3:.0f} Å³ (质心均值 {r:.0f}Å 球, n分子={n_centroids}, "
            f"占据率={frac * 100:.1f}%; 非蛋白空腔)"
        ),
    }


def _evaluate_idea_h_protein_cavity_mc(
    molecules_with_pos,
    protein_pdb,
    optimal_min=350,
    optimal_max=750,
    zero_below=80.0,
    zero_above=2000.0,
    centroid_radius=10.0,
    n_mc_samples=20000,
    protein_probe_ang=1.4,
    ligand_proximity_ang=3.0,
):
    """
    POVME-style cavity: points in ligand-centered sphere that
      (1) do not clash with protein heavy atoms (vdW + probe), and
      (2) lie within ligand_proximity_ang of any generated ligand heavy atom.
    """
    from diffdynamic.analysis.pocket.clustering import _collect_atom_data

    if not molecules_with_pos:
        return {
            "score": 0.0,
            "volume_ang3": None,
            "volume_method": None,
            "quality_label": "unknown",
            "success": False,
            "message": "无分子数据",
        }

    atom_data = _collect_atom_data(molecules_with_pos)
    if atom_data is None or len(atom_data["centroids"]) < 1:
        return {
            "score": 0.0,
            "volume_ang3": None,
            "volume_method": None,
            "quality_label": "unknown",
            "success": False,
            "message": "无分子质心",
        }

    prot_coords, prot_z, err = _load_protein_heavy_coords(protein_pdb)
    if err or prot_coords is None or len(prot_coords) < 1:
        return None  # caller falls back

    centroids = np.asarray(atom_data["centroids"], dtype=np.float64)
    ref_center = centroids.mean(axis=0)
    n_centroids = len(centroids)
    r = float(centroid_radius)
    v_ball = (4.0 / 3.0) * np.pi * (r**3)
    probe = float(protein_probe_ang)
    prox = float(ligand_proximity_ang)

    lig_coords = np.asarray(atom_data["coords"], dtype=np.float64)
    lig_z = np.asarray(atom_data["atom_nums"], dtype=np.int32)
    heavy_mask = lig_z > 1
    if heavy_mask.any():
        lig_heavy = lig_coords[heavy_mask]
    else:
        lig_heavy = lig_coords
    if len(lig_heavy) < 1:
        return {
            "score": 0.0,
            "volume_ang3": None,
            "volume_method": None,
            "quality_label": "unknown",
            "success": False,
            "message": "无配体重原子",
        }

    prot_radii = np.array(
        [_vdw_radius_from_atomic_num(int(z)) + probe for z in prot_z],
        dtype=np.float64,
    )

    # Restrict protein atoms to those near the sphere (speed)
    d_c = np.linalg.norm(prot_coords - ref_center.reshape(1, 3), axis=1)
    keep_p = d_c <= (r + float(prot_radii.max()) + 0.5)
    if keep_p.any():
        prot_coords = prot_coords[keep_p]
        prot_radii = prot_radii[keep_p]

    rng = np.random.default_rng(42)
    samples = _sample_uniform_in_ball(ref_center, r, int(n_mc_samples), rng)

    chunk = 1024
    n_cav = 0
    n_mc = int(n_mc_samples)
    for s in range(0, n_mc, chunk):
        blk = samples[s : s + chunk]
        # near ligand cloud
        d_lig = np.linalg.norm(blk[:, None, :] - lig_heavy[None, :, :], axis=2)
        near_lig = (d_lig <= prox).any(axis=1)
        if not near_lig.any():
            continue
        cand = blk[near_lig]
        # not inside protein
        d_prot = np.linalg.norm(cand[:, None, :] - prot_coords[None, :, :], axis=2)
        clash = (d_prot < prot_radii[None, :]).any(axis=1)
        n_cav += int((~clash).sum())

    frac = n_cav / float(n_mc)
    volume_ang3 = float(frac * v_ball)
    volume_method = f"ligand_protein_cavity_mc_{r:g}A"

    zb, za = float(zero_below), float(zero_above)
    score, label = _idea_h_score_from_volume(
        volume_ang3, optimal_min, optimal_max, zb, za
    )
    prot_path = str(Path(protein_pdb).expanduser().resolve())

    return {
        "score": score,
        "volume_ang3": volume_ang3,
        "volume_method": volume_method,
        "quality_label": label,
        "success": True,
        "optimal_vol_min": optimal_min,
        "optimal_vol_max": optimal_max,
        "vol_score_zero_below": zb,
        "vol_score_zero_above": za,
        "centroid_radius": r,
        "n_centroids": n_centroids,
        "reference_centroid": ref_center.tolist(),
        "single_sphere_ref_volume_ang3": float(v_ball),
        "ligand_occupancy_fraction": None,
        "cavity_fraction": float(frac),
        "n_protein_heavy": int(len(prot_coords)),
        "protein_probe_ang": probe,
        "ligand_proximity_ang": prox,
        "fpocket_protein_pdb": prot_path,
        "fpocket_protein_volume_ang3": None,
        "fpocket_ligand_volume_ang3": None,
        "fpocket_protein_pocket_meta": None,
        "fpocket_ligand_pocket_meta": None,
        "fpocket_ligand_success": None,
        "fpocket_skipped": False,
        "fpocket_skip_reason": None,
        "fpocket_resolved_executable": None,
        "volume_score_band": "ligand_protein_cavity_10A",
        "message": (
            f"口袋空腔≈{volume_ang3:.0f} Å³ (prot+lig MC, r={r:.0f}Å, probe={probe:.1f}Å, "
            f"lig≤{prox:.1f}Å, n分子={n_centroids}, 空腔率={frac * 100:.1f}%, "
            f"蛋白重原子={len(prot_coords)}, pdb={Path(prot_path).name})"
        ),
    }


def evaluate_idea_h_pocket_size(
    molecules_with_pos,
    optimal_min=350,
    optimal_max=750,
    zero_below=80.0,
    zero_above=2000.0,
    centroid_radius=10.0,
    n_mc_samples=20000,
    fpocket_protein_pdb=None,
    fpocket_cmd="fpocket",
    fpocket_pocket_index=1,
    fpocket_max_ligand_models=50,
    fpocket_timeout=600,
    fpocket_optimal_min=None,
    fpocket_optimal_max=None,
    fpocket_zero_below=None,
    fpocket_zero_above=None,
):
    """
    想法H：口袋体积 —— 优先「配体定位 + 蛋白排除」空腔 MC（POVME 风格）。

    - 有受体 PDB（``fpocket_protein_pdb``）：10 Å 球内、非蛋白重叠且靠近配体云的空腔体积。
    - 无 PDB / 解析失败：回退配体占据 MC（标注 fallback）。
    - 满分带默认 350–750 Å³。FPocket 体积不作为默认（见 legacy 函数）。
    """
    zb, za = float(zero_below), float(zero_above)
    r = float(centroid_radius) if centroid_radius is not None else 10.0
    pdb_arg = fpocket_protein_pdb
    if pdb_arg and str(pdb_arg).strip():
        cav = _evaluate_idea_h_protein_cavity_mc(
            molecules_with_pos,
            protein_pdb=pdb_arg,
            optimal_min=optimal_min,
            optimal_max=optimal_max,
            zero_below=zb,
            zero_above=za,
            centroid_radius=r,
            n_mc_samples=n_mc_samples,
        )
        if cav is not None and cav.get("success"):
            return cav
        # fall through to ligand-only MC
    return _evaluate_idea_h_mc_only(
        molecules_with_pos,
        optimal_min,
        optimal_max,
        zb,
        za,
        r,
        n_mc_samples,
        as_fallback=bool(pdb_arg),
    )


def evaluate_idea_h_pocket_size_legacy_fpocket(
    molecules_with_pos,
    optimal_min=350,
    optimal_max=750,
    zero_below=80.0,
    zero_above=2000.0,
    centroid_radius=10.0,
    n_mc_samples=20000,
    fpocket_protein_pdb=None,
    fpocket_cmd="fpocket",
    fpocket_pocket_index=1,
    fpocket_max_ligand_models=50,
    fpocket_timeout=600,
    fpocket_optimal_min=None,
    fpocket_optimal_max=None,
    fpocket_zero_below=None,
    fpocket_zero_above=None,
):
    """
    旧版 H：可选 FPocket Volume（已弃用主路径；保留供对照）。

    - 未指定 ``fpocket_protein_pdb``：MC（默认 10 Å）。
    - 指定 ``fpocket_protein_pdb`` 且蛋白 FPocket 成功：优先配体 FPocket Volume。
    """
    from diffdynamic.analysis.pocket.clustering import _collect_atom_data

    zb, za = float(zero_below), float(zero_above)
    pdb_arg = fpocket_protein_pdb
    if pdb_arg is None or str(pdb_arg).strip() == "":
        return _evaluate_idea_h_mc_only(
            molecules_with_pos,
            optimal_min,
            optimal_max,
            zb,
            za,
            centroid_radius,
            n_mc_samples,
        )

    prot_path = Path(pdb_arg).expanduser().resolve()
    if not prot_path.is_file():
        return {
            "score": 0.0,
            "volume_ang3": None,
            "volume_method": "fpocket_failed",
            "quality_label": "unknown",
            "success": False,
            "message": f"FPocket 蛋白 PDB 不存在: {prot_path}",
            "fpocket_protein_pdb": str(prot_path),
            "fpocket_protein_volume_ang3": None,
            "fpocket_ligand_volume_ang3": None,
            "fpocket_protein_pocket_meta": None,
            "fpocket_ligand_pocket_meta": None,
            "fpocket_ligand_success": False,
            "optimal_vol_min": optimal_min,
            "optimal_vol_max": optimal_max,
            "vol_score_zero_below": zb,
            "vol_score_zero_above": za,
        }

    exe0 = resolve_fpocket_executable(fpocket_cmd)
    if exe0 is None:
        mc = _evaluate_idea_h_mc_only(
            molecules_with_pos,
            optimal_min,
            optimal_max,
            zb,
            za,
            centroid_radius,
            n_mc_samples,
        )
        mc["fpocket_protein_pdb"] = str(prot_path)
        mc["fpocket_skipped"] = True
        mc["fpocket_skip_reason"] = "executable_not_found"
        mc["fpocket_resolved_executable"] = None
        hint = (
            "conda install -c conda-forge fpocket；或 export FPOCKET_CMD=/绝对路径/fpocket；"
            "或 --fpocket_cmd 指向可执行文件"
        )
        mc["message"] = (
            f"[FPocket 不可用，已改用 MC 体积评分] {mc['message']}（{hint}）"
        )
        return mc

    ok_p, msg_p, pockets_p = run_fpocket_on_pdb(
        prot_path, fpocket_cmd=fpocket_cmd, timeout=fpocket_timeout
    )
    if not ok_p:
        return {
            "score": 0.0,
            "volume_ang3": None,
            "volume_method": "fpocket_failed",
            "quality_label": "unknown",
            "success": False,
            "message": f"蛋白 FPocket 失败: {msg_p}",
            "fpocket_protein_pdb": str(prot_path),
            "fpocket_protein_volume_ang3": None,
            "fpocket_ligand_volume_ang3": None,
            "fpocket_protein_pocket_meta": None,
            "fpocket_ligand_pocket_meta": None,
            "fpocket_ligand_success": False,
            "optimal_vol_min": optimal_min,
            "optimal_vol_max": optimal_max,
            "vol_score_zero_below": zb,
            "vol_score_zero_above": za,
        }

    pick_p = _fpocket_pick_pocket(pockets_p, fpocket_pocket_index)
    vol_p = pick_p.get("volume") if pick_p else None
    if vol_p is None:
        return {
            "score": 0.0,
            "volume_ang3": None,
            "volume_method": "fpocket_failed",
            "quality_label": "unknown",
            "success": False,
            "message": f"蛋白 FPocket 未解析到 Pocket {fpocket_pocket_index} 的 Volume 字段",
            "fpocket_protein_pdb": str(prot_path),
            "fpocket_protein_volume_ang3": None,
            "fpocket_ligand_volume_ang3": None,
            "fpocket_protein_pocket_meta": pick_p,
            "fpocket_ligand_pocket_meta": None,
            "fpocket_ligand_success": False,
            "optimal_vol_min": optimal_min,
            "optimal_vol_max": optimal_max,
            "vol_score_zero_below": zb,
            "vol_score_zero_above": za,
        }

    vol_protein = float(vol_p)
    f_omin = float(
        fpocket_optimal_min
        if fpocket_optimal_min is not None
        else IDEA_H_FPOCKET_PROTEIN_FALLBACK_OPTIMAL_MIN
    )
    f_omax = float(
        fpocket_optimal_max
        if fpocket_optimal_max is not None
        else IDEA_H_FPOCKET_PROTEIN_FALLBACK_OPTIMAL_MAX
    )
    f_zb = float(
        fpocket_zero_below
        if fpocket_zero_below is not None
        else IDEA_H_FPOCKET_PROTEIN_FALLBACK_ZERO_BELOW
    )
    f_za = float(
        fpocket_zero_above
        if fpocket_zero_above is not None
        else IDEA_H_FPOCKET_PROTEIN_FALLBACK_ZERO_ABOVE
    )

    lig_vol = None
    lig_meta = None
    lig_ok = False
    lig_msg = ""
    if molecules_with_pos:
        with tempfile.TemporaryDirectory(prefix="fpocket_lig_") as ltd:
            lig_pdb = Path(ltd) / "ligands.pdb"
            w_ok, w_msg = _write_ligands_multimodel_pdb(
                molecules_with_pos,
                lig_pdb,
                max_models=int(fpocket_max_ligand_models),
            )
            if w_ok:
                ok_l, msg_l, pockets_l = run_fpocket_on_pdb(
                    lig_pdb,
                    fpocket_cmd=fpocket_cmd,
                    timeout=fpocket_timeout,
                )
                if ok_l:
                    if not pockets_l:
                        lig_msg = "配体 FPocket 未检测到任何口袋（小分子上常见，体积仅供参考）"
                    else:
                        pick_l = _fpocket_pick_pocket(pockets_l, fpocket_pocket_index)
                        if pick_l and pick_l.get("volume") is not None:
                            lig_vol = float(pick_l["volume"])
                            lig_meta = pick_l
                            lig_ok = True
                        else:
                            lig_msg = f"配体 FPocket 无有效 Volume（Pocket {fpocket_pocket_index}）"
                else:
                    lig_msg = f"配体 FPocket 失败: {msg_l}"
            else:
                lig_msg = w_msg

    if lig_ok and lig_vol is not None:
        vol_scored = float(lig_vol)
        score, label = _idea_h_score_from_volume(
            vol_scored, optimal_min, optimal_max, zb, za
        )
        omin_out, omax_out = float(optimal_min), float(optimal_max)
        zb_out, za_out = zb, za
        volume_score_band = "fpocket_ligand_mc_band"
        vol_method = f"fpocket_ligand_pocket{int(lig_meta.get('pocket_id', fpocket_pocket_index))}_mc_band"
    else:
        vol_scored = vol_protein
        score, label = _idea_h_score_from_volume(vol_scored, f_omin, f_omax, f_zb, f_za)
        omin_out, omax_out = f_omin, f_omax
        zb_out, za_out = f_zb, f_za
        volume_score_band = "fpocket_protein_fallback"
        vol_method = f"fpocket_protein_pocket{int(pick_p.get('pocket_id', fpocket_pocket_index))}_fallback"

    parts = [
        f"蛋白 FPocket 口袋#{int(pick_p.get('pocket_id', fpocket_pocket_index))} "
        f"Volume≈{vol_protein:.0f} Å³（参考）",
    ]
    if lig_ok and lig_vol is not None:
        parts.append(
            f"配体 FPocket Volume≈{lig_vol:.0f} Å³（用于 H 分，区间 {optimal_min:.0f}–{optimal_max:.0f} Å³ 满分，"
            f"<{zb:.0f} 或 ≥{za:.0f} 为 0）"
        )
    elif lig_msg:
        parts.append(
            f"配体 FPocket: {lig_msg}（H 分已按蛋白 FPocket 体积 + 宽区间回退）"
        )
    else:
        parts.append(
            "配体侧 FPocket 未计算或未得体积；H 分按蛋白 FPocket 体积 + 宽区间回退"
        )

    ad_h = _collect_atom_data(molecules_with_pos) if molecules_with_pos else None
    n_centroids = (
        len(ad_h["centroids"]) if ad_h and ad_h.get("centroids") is not None else None
    )

    return {
        "score": score,
        "volume_ang3": vol_scored,
        "volume_method": vol_method,
        "quality_label": label,
        "success": True,
        "optimal_vol_min": omin_out,
        "optimal_vol_max": omax_out,
        "vol_score_zero_below": zb_out,
        "vol_score_zero_above": za_out,
        "volume_score_band": volume_score_band,
        "n_centroids": n_centroids,
        "reference_centroid": None,
        "single_sphere_ref_volume_ang3": None,
        "ligand_occupancy_fraction": None,
        "fpocket_protein_pdb": str(prot_path),
        "fpocket_protein_volume_ang3": vol_protein,
        "fpocket_ligand_volume_ang3": lig_vol,
        "fpocket_protein_pocket_meta": pick_p,
        "fpocket_ligand_pocket_meta": lig_meta,
        "fpocket_ligand_success": lig_ok,
        "fpocket_skipped": False,
        "fpocket_skip_reason": None,
        "fpocket_resolved_executable": exe0,
        "message": "；".join(parts),
    }
