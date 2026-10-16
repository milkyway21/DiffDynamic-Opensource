"""Scoring responsibilities extracted from evaluate_pt_with_correct_reconstruct.py."""

from diffdynamic.evaluation.molecule.state import (
    AllChem,
    Chem,
    Counter,
    DOCK_TMP_DIR,
    Descriptors,
    Path,
    UFFOptimizeMolecule,
    VinaDockingTask,
    _DEFAULT_VINA_MODES,
    analyze,
    datetime,
    eval_atom_type,
    eval_bond_length,
    evaluate_lilly_medchem_rules,
    np,
    os,
    scoring_func,
    similarity,
    traceback,
)
from diffdynamic.evaluation.molecule import state as _state


def apply_pre_docking_preparation(
    mol,
    mmff_minimize=True,
    mmff_max_iters=200,
    use_uff_fallback=True,
    etkdg_reembed=False,
    debug=False,
):
    """
    对接前分子预处理：ETKDG重嵌3D + MMFF能量最小化，确保分子键长合理化、为现实中可存在的合理结构。

    RDKit 在生成/优化 3D 时会将键长调整到化学合理的标准值（基于内置力场与距离几何）。

    Args:
        mol: RDKit分子对象
        mmff_minimize: 是否执行MMFF能量最小化
        mmff_max_iters: MMFF最大迭代次数
        use_uff_fallback: MMFF失败时是否用UFF回退
        etkdg_reembed: 是否用ETKDGv3重新生成3D（基于标准键长距离边界，再力场优化）
        debug: 是否输出调试信息

    Returns:
        bool: 是否成功完成预处理
    """
    if mol is None:
        return False
    try:
        # 1. ETKDG 重嵌：用 RDKit 距离几何重新生成 3D，键长基于标准距离边界（化学合理值）
        if etkdg_reembed and mol.GetNumConformers() > 0:
            pos_backup = np.array(mol.GetConformer(0).GetPositions())
            mol.RemoveAllConformers()
            try:
                ret_embed = AllChem.EmbedMolecule(
                    mol, Chem.rdDistGeom.ETKDGv3(), randomSeed=42
                )
                if ret_embed == 0:
                    if debug:
                        print(f"  ✅ ETKDGv3 重嵌3D完成，键长已基于标准距离边界")
                else:
                    if debug:
                        print(f"  ⚠️  ETKDGv3 重嵌失败 (ret={ret_embed})，保留原构象")
                    conf = Chem.Conformer(mol.GetNumAtoms())
                    for i in range(mol.GetNumAtoms()):
                        conf.SetAtomPosition(i, tuple(pos_backup[i]))
                    mol.AddConformer(conf, assignId=True)
            except Exception as e:
                if debug:
                    print(f"  ⚠️  ETKDGv3 重嵌异常: {e}，保留原构象")
                conf = Chem.Conformer(mol.GetNumAtoms())
                for i in range(mol.GetNumAtoms()):
                    conf.SetAtomPosition(i, tuple(pos_backup[i]))
                mol.AddConformer(conf, assignId=True)

        if not mmff_minimize:
            return True
        if mol.GetNumConformers() == 0:
            if debug:
                print(f"  ⚠️  分子无构象，跳过MMFF预处理")
            return False
        conf_id = 0
        ret = AllChem.MMFFOptimizeMolecule(mol, confId=conf_id, maxIters=mmff_max_iters)
        if ret == 0 or ret == 1:  # 0=成功, 1=收敛
            if debug:
                print(f"  ✅ MMFF能量最小化完成 (ret={ret})")
            return True
        if use_uff_fallback:
            ret_uff = UFFOptimizeMolecule(mol, confId=conf_id, maxIters=mmff_max_iters)
            if ret_uff == 0 or ret_uff == 1:
                if debug:
                    print(f"  ✅ MMFF失败，UFF回退成功 (ret={ret_uff})")
                return True
        if debug:
            print(f"  ⚠️  MMFF/UFF预处理未完全收敛 (MMFF ret={ret})")
        return False
    except Exception as e:
        if debug:
            print(f"  ⚠️  对接前预处理异常: {e}")
        return False


def calculate_comprehensive_score(eval_result):
    """
    根据评估结果计算综合模型评分。

    公式: 100 * (基础分加权和) * PAINS惩罚 * 稳定性惩罚

    参数:
    eval_result (dict): 包含评估指标的字典

    返回:
    float: 计算后的模型评分，已钳制到 [0, 100]
    """
    # 1. 预处理 Vina 亲和力（综合分中的亲和力项）
    # 优先 score_only / minimize，与 prudent 采样一致；无则回退 dock 最佳构象亲和力。
    # 归一化: MAX(0, MIN(1, affinity / (-10)))（kcal/mol，越负越好；-10 对应满分上限）
    vina_affinity = None
    if eval_result.get("vina_score_only") is not None:
        vina_affinity = eval_result["vina_score_only"]
    elif eval_result.get("vina_minimize") is not None:
        vina_affinity = eval_result["vina_minimize"]
    elif eval_result.get("vina_dock") and len(eval_result["vina_dock"]) > 0:
        vina_affinity = eval_result["vina_dock"][0].get("affinity")

    if vina_affinity is None:
        # 如果没有亲和力数据，返回0分
        return 0.0

    affinity_norm = float(vina_affinity) / (-10.0)
    affinity_norm = max(0.0, min(1.0, affinity_norm))  # 限制在 0-1 之间

    # 2. 获取其他评分指标
    chem = eval_result.get("chem", {})
    qed = chem.get("qed", 0.0) if chem else 0.0
    sa = chem.get("sa", 0.0) if chem else 0.0
    lipinski = eval_result.get("lipinski", 0)
    if lipinski == "N/A" or lipinski is None:
        lipinski = 0
    else:
        lipinski = int(lipinski)

    # SA评分需要确认：SA评分通常是"越低越好"（合成难度），但这里假设已经归一化为"越高越好"
    # 如果SA是越低越好，需要转换：sa_normalized = 1 - sa / 10 (假设最大值为10)
    # 这里假设SA已经归一化到0-1范围，且越高越好
    sa_normalized = (
        sa if sa <= 1.0 else (1.0 - sa / 10.0)
    )  # 如果SA>1，假设是原始SA评分，需要转换

    # 子项统一钳到 [0,1]，避免异常数据使 base_score 超出 [0,1]，从而保证最终得分在 [0,100]
    qed = max(0.0, min(1.0, float(qed)))
    sa_normalized = max(0.0, min(1.0, float(sa_normalized)))
    lipinski_frac = max(0.0, min(1.0, float(lipinski) / 5.0))

    # 3. 计算加权基础分 (Base Score)
    # 权重: 亲和力(40%) + QED(30%) + SA(20%) + Lipinski(10%)
    base_score = (
        0.4 * affinity_norm + 0.3 * qed + 0.2 * sa_normalized + 0.1 * lipinski_frac
    )
    base_score = max(0.0, min(1.0, base_score))

    # 4. 计算惩罚系数 (Multipliers)
    # PAINS 惩罚: 如果检测到 (True)，系数为 0.5，否则 1.0
    pains = eval_result.get("pains", False)
    if pains == "N/A" or pains is None:
        pains = False
    pains_multiplier = 0.5 if pains else 1.0

    # 稳定性惩罚: 如果不稳定 (False)，系数为 0.9，否则 1.0
    # 注意：check_stability检查的是基于坐标推断的键级是否符合价键规则
    # 即使RDKit修复了价键错误，如果坐标质量差，仍应给予一定惩罚
    stability = eval_result.get("stability", {})
    if isinstance(stability, dict):
        molecule_stable = stability.get("molecule_stable", True)
    else:
        molecule_stable = True if stability else False
    # 如果分子不稳定（坐标质量差），给予0.9的惩罚系数
    stability_multiplier = 0.9 if not molecule_stable else 1.0

    # 5. 计算最终得分 (Total Score)，输出区间 [0, 100]
    final_score = 100.0 * base_score * pains_multiplier * stability_multiplier
    return max(0.0, min(100.0, final_score))


def dock_reference_ligand(
    ligand_sdf_path, protein_path, exhaustiveness=8, vina_timeout=20
):
    """Dock the reference ligand with Vina score_only to get baseline affinity.

    Returns:
        float or None: Vina score_only affinity (kcal/mol), None if failed.
    """
    try:
        from vina import Vina

        supplier = Chem.SDMolSupplier(str(ligand_sdf_path), sanitize=False)
        ref_mol = supplier[0] if supplier and len(supplier) > 0 else None
        if ref_mol is None:
            print(f"  ⚠️  参考配体加载失败: {ligand_sdf_path}")
            return None

        # Write temp SDF for Vina
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".sdf", delete=False) as tmp:
            tmp_path = tmp.name
            w = Chem.SDWriter(tmp_path)
            w.write(ref_mol)
            w.close()

        # Convert to PDBQT
        tmp_pdbqt = tmp_path.replace(".sdf", ".pdbqt")
        try:
            # 复用 PrepLig：meeko 0.1.dev3 仅接受 OBMol，PrepLig 已正确处理版本差异
            from diffdynamic.evaluation.docking_vina import PrepLig

            lig = PrepLig(tmp_path, "sdf")
            lig.get_pdbqt(tmp_pdbqt)
        except Exception as _e:
            # Fallback: use obabel
            import subprocess as sp

            sp.run(
                ["obabel", tmp_path, "-opdbqt", "-O", tmp_pdbqt, "--gen3d"],
                capture_output=True,
                timeout=30,
            )

        if not os.path.exists(tmp_pdbqt):
            print(f"  ⚠️  参考配体 PDBQT 转换失败")
            os.unlink(tmp_path)
            return None

        # Run Vina score_only
        vina = Vina(sf_type="vina")
        vina.set_receptor(str(protein_path))
        vina.set_ligand_from_file(tmp_pdbqt)

        # Use reference ligand center for search box
        conf = ref_mol.GetConformer()
        coords = [list(conf.GetAtomPosition(i)) for i in range(ref_mol.GetNumAtoms())]
        center = [
            np.mean([c[0] for c in coords]),
            np.mean([c[1] for c in coords]),
            np.mean([c[2] for c in coords]),
        ]
        box_size = [30, 30, 30]
        vina.compute_vina_maps(center=center, box_size=box_size)

        import signal

        def _timeout_handler(signum, frame):
            raise TimeoutError("Vina score_only timeout")

        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(vina_timeout)
        try:
            vina.score_only(exhaustiveness=exhaustiveness)
            energy = vina.energies()[0][0]
        except TimeoutError:
            print(f"  ⚠️  参考配体 Vina score_only 超时 ({vina_timeout}s)")
            energy = None
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

        # Cleanup
        os.unlink(tmp_path)
        if os.path.exists(tmp_pdbqt):
            os.unlink(tmp_pdbqt)

        return energy
    except Exception as e:
        print(f"  ⚠️  参考配体对接失败: {e}")
        return None


def evaluate_single_molecule(
    mol,
    ligand_filename,
    protein_root,
    exhaustiveness=8,
    n_poses=1,
    size_factor=1.0,
    buffer=5.0,
    debug=False,
    tmp_dir=None,
    protein_path=None,
    pre_docking_mmff_minimize=False,
    pre_docking_mmff_max_iters=200,
    pre_docking_use_uff_fallback=True,
    pre_docking_etkdg_reembed=False,
    vina_modes=None,
):
    """
    对单个分子进行化学指标计算和AutoDock Vina对接（包括dock、score_only、minimize模式）
    以及分子结构评估（原子类型分布、键长分布、原子对距离分布）

    Returns:
        dict: {'chem': {...}, 'vina_dock': [...], 'vina_score_only': float, 'vina_minimize': float,
               'atom_type_jsd': float, 'bond_length_jsd': dict, 'pair_length_jsd': dict, 'success': bool}
    """
    result = {
        "chem": None,
        "vina_dock": None,
        "vina_score_only": None,
        "vina_minimize": None,
        "atom_type_jsd": None,
        "bond_length_jsd": None,
        "pair_length_jsd": None,
        "success": False,
        "smiles": None,
        # 新增评估指标
        "stability": None,  # 稳定性评估
        "basic_info": None,  # 基础结构信息
        "logp": None,  # logP值
        "lipinski": None,  # Lipinski规则得分
        "pains": None,  # PAINS检测
        "tanimoto_sim": None,  # Tanimoto相似度
        "rdkit_rmsd": None,  # RDKit RMSD
        "conformer_energy": None,  # 构象能量
        "tpsa": None,  # TPSA（拓扑极性表面积）
        "rdkit_valid": None,  # RDKit验证通过（True/False）
        "lilly_medchem_passed": None,  # Lilly Medchem Rules是否通过
        "lilly_medchem_demerit": None,  # Lilly Medchem Rules扣分
        "lilly_medchem_description": None,  # Lilly Medchem Rules描述（匹配规则、拒绝原因等）
        "comprehensive_score": None,  # 综合模型评分
        "molecule_id": None,  # 分子身份证
        "error": None,  # 对接/评估错误信息（缺省时避免误判为「未知」）
    }

    try:
        # 0. RDKit验证检查
        try:
            # 尝试对分子进行标准化验证
            mol_copy = Chem.Mol(mol.ToBinary())  # 深拷贝分子
            Chem.SanitizeMol(mol_copy)
            result["rdkit_valid"] = True
            if debug:
                print(f"  ✅ RDKit验证: 通过")
        except Exception as e:
            result["rdkit_valid"] = False
            if debug:
                print(f"  ⚠️  RDKit验证: 失败 ({str(e)[:50]})")

        # 1. 计算化学指标
        chem_results = scoring_func.get_chem(mol)
        result["chem"] = chem_results

        # 1.1 提取logP和Lipinski规则得分（从chem结果中）
        result["logp"] = chem_results.get("logp", "N/A")
        result["lipinski"] = chem_results.get("lipinski", "N/A")

        # 1.1.1 计算TPSA（拓扑极性表面积）
        try:
            tpsa = Descriptors.TPSA(mol)
            result["tpsa"] = tpsa
            if debug:
                print(f"  ✅ TPSA: {tpsa:.2f} Å²")
        except Exception as e:
            if debug:
                print(f"  ⚠️  TPSA计算失败: {e}")

        # 1.2 计算基础结构信息
        try:
            basic_info = scoring_func.get_basic(mol)
            result["basic_info"] = {
                "n_atoms": basic_info[0],
                "n_bonds": basic_info[1],
                "n_rings": basic_info[2],
                "weight": basic_info[3],
            }
            if debug:
                print(
                    f"  ✅ 基础信息: {basic_info[0]}原子, {basic_info[1]}键, {basic_info[2]}环, 分子量={basic_info[3]:.2f}"
                )
        except Exception as e:
            if debug:
                print(f"  ⚠️  基础信息计算失败: {e}")

        # 1.3 PAINS检测
        try:
            is_pains = scoring_func.is_pains(mol)
            result["pains"] = is_pains
            if debug:
                print(f"  ✅ PAINS检测: {'是' if is_pains else '否'}")
        except Exception as e:
            if debug:
                print(f"  ⚠️  PAINS检测失败: {e}")

        # 1.4 稳定性评估（需要原子坐标和类型）
        try:
            pos = mol.GetConformer().GetPositions()
            atom_types = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
            # 将原子序数转换为analyze模块使用的编码（1,6,7,8,9,15,16,17对应H,C,N,O,F,P,S,Cl）
            atom_encoder_map = {1: 1, 6: 6, 7: 7, 8: 8, 9: 9, 15: 15, 16: 16, 17: 17}
            atom_type_indices = [
                atom_encoder_map.get(atype, 6) for atype in atom_types
            ]  # 默认使用C
            stability_result = analyze.check_stability(
                pos, np.array(atom_type_indices), return_nr_bonds=True
            )
            result["stability"] = {
                "molecule_stable": stability_result[0],
                "nr_stable_bonds": stability_result[1],
                "n_atoms": stability_result[2],
                "nr_bonds": stability_result[3] if len(stability_result) > 3 else None,
            }
            if debug:
                print(
                    f"  ✅ 稳定性: 分子稳定={stability_result[0]}, 稳定原子数={stability_result[1]}/{stability_result[2]}"
                )
        except Exception as e:
            if debug:
                print(f"  ⚠️  稳定性评估失败: {e}")

        # 1.5 Tanimoto相似度（尝试从配体文件加载参考分子）
        try:
            if ligand_filename and protein_root:
                ligand_path = Path(protein_root) / ligand_filename
                if ligand_path.exists():
                    # 尝试加载参考分子
                    ref_mol = None
                    suffix = ligand_path.suffix.lower()
                    if suffix == ".sdf":
                        supplier = Chem.SDMolSupplier(str(ligand_path), sanitize=False)
                        ref_mol = next(supplier, None)
                    elif suffix in [".mol2", ".mol"]:
                        ref_mol = Chem.MolFromMolFile(str(ligand_path), sanitize=False)
                    if ref_mol:
                        try:
                            Chem.SanitizeMol(ref_mol)
                            tanimoto = similarity.tanimoto_sim(mol, ref_mol)
                            result["tanimoto_sim"] = tanimoto
                            if debug:
                                print(f"  ✅ Tanimoto相似度: {tanimoto:.4f}")
                        except:
                            pass
        except Exception as e:
            if debug:
                print(f"  ⚠️  Tanimoto相似度计算失败: {e}")

        # 1.6 RDKit RMSD
        try:
            rmsd_results = scoring_func.get_rdkit_rmsd(mol, n_conf=20, random_seed=42)
            result["rdkit_rmsd"] = {
                "max": rmsd_results[0],
                "min": rmsd_results[1],
                "median": rmsd_results[2],
            }
            if debug and not np.isnan(rmsd_results[0]):
                print(
                    f"  ✅ RDKit RMSD: 最大={rmsd_results[0]:.4f}, 最小={rmsd_results[1]:.4f}, 中位数={rmsd_results[2]:.4f}"
                )
        except Exception as e:
            if debug:
                print(f"  ⚠️  RDKit RMSD计算失败: {e}")

        # 1.7 构象能量
        try:
            # 需要先添加氢原子
            mol_with_h = Chem.AddHs(mol)
            # 生成构象
            conf_ids = Chem.AllChem.EmbedMultipleConfs(
                mol_with_h, numConfs=1, randomSeed=42
            )
            if conf_ids:
                energies = scoring_func.get_conformer_energies(
                    mol_with_h, force_field="mmff"
                )
                if len(energies) > 0:
                    result["conformer_energy"] = float(energies[0])
                    if debug:
                        print(f"  ✅ 构象能量: {energies[0]:.4f} kcal/mol")
        except Exception as e:
            if debug:
                print(f"  ⚠️  构象能量计算失败: {e}")

        # 1.8 Lilly Medchem Rules评估
        if mol is not None:
            try:
                if debug:
                    print(f"  - 开始Lilly Medchem Rules评估...")
                lilly_result = evaluate_lilly_medchem_rules(mol, debug=debug)
                # 确保返回结果不为None
                if lilly_result is None:
                    lilly_result = {
                        "passed": False,
                        "demerit": 0,
                        "demerit_cutoff": 100,
                        "matched_rules": [],
                        "reject_reason": "evaluation_returned_none",
                        "n_heavy_atoms": 0,
                        "details": {},
                    }

                result["lilly_medchem_passed"] = lilly_result.get("passed", False)
                result["lilly_medchem_demerit"] = lilly_result.get("demerit", 0)

                if debug:
                    print(
                        f"    [数据设置] lilly_medchem_passed={result['lilly_medchem_passed']}, demerit={result['lilly_medchem_demerit']}"
                    )

                # 构建描述信息
                description_parts = []
                if not lilly_result.get("passed", False):
                    # 如果未通过，优先显示拒绝原因
                    reject_reason = lilly_result.get("reject_reason", "")
                    if reject_reason:
                        description_parts.append(f"拒绝原因: {reject_reason}")
                matched_rules = lilly_result.get("matched_rules", [])
                if matched_rules:
                    description_parts.append(f"匹配规则: {', '.join(matched_rules)}")
                demerit = lilly_result.get("demerit", 0)
                if demerit > 0:
                    demerit_cutoff = lilly_result.get("demerit_cutoff", 100)
                    description_parts.append(f"扣分: {demerit}/{demerit_cutoff}")
                n_heavy_atoms = lilly_result.get("n_heavy_atoms", 0)
                if n_heavy_atoms > 0:
                    description_parts.append(f"重原子数: {n_heavy_atoms}")

                result["lilly_medchem_description"] = (
                    "; ".join(description_parts) if description_parts else "通过"
                )

                if debug:
                    if lilly_result.get("passed", False):
                        print(
                            f"  ✅ Lilly Medchem Rules: 通过 (扣分={demerit}/{lilly_result.get('demerit_cutoff', 100)})"
                        )
                        if matched_rules:
                            print(f"     匹配规则: {', '.join(matched_rules)}")
                    else:
                        print(
                            f"  ❌ Lilly Medchem Rules: 未通过 ({lilly_result.get('reject_reason', 'unknown')})"
                        )
            except Exception as e:
                # 异常时设置默认值
                result["lilly_medchem_passed"] = False
                result["lilly_medchem_demerit"] = 0
                result["lilly_medchem_description"] = f"评估异常: {str(e)}"
                if debug:
                    print(f"  ⚠️  Lilly Medchem Rules评估失败: {e}")
        else:
            # mol为None时设置默认值
            result["lilly_medchem_passed"] = False
            result["lilly_medchem_demerit"] = 0
            result["lilly_medchem_description"] = "分子对象为None"
            if debug:
                print(f"  ⚠️  Lilly Medchem Rules评估跳过: mol为None")

        # 2. 生成SMILES
        try:
            smiles = Chem.MolToSmiles(mol)
            result["smiles"] = smiles

            # 检查是否是片段化分子
            if "." in smiles:
                if debug:
                    print(f"  ⚠️  跳过片段化分子: {smiles}")
                return result
        except:
            pass

        # 3. 计算分子结构指标
        if debug:
            print(f"  - 开始计算分子结构指标...")

        try:
            # 3.1 原子类型分布JSD
            atom_numbers = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
            atom_counter = Counter(atom_numbers)
            atom_type_jsd = eval_atom_type.eval_atom_type_distribution(atom_counter)
            result["atom_type_jsd"] = atom_type_jsd
            if debug and atom_type_jsd is not None:
                print(f"  ✅ 原子类型分布JSD: {atom_type_jsd:.4f}")
        except Exception as e:
            if debug:
                print(f"  ⚠️  原子类型分布计算失败: {e}")

        # 4. 对接前分子预处理（ETKDG重嵌3D + MMFF能量最小化，键长合理化，确保分子为现实中可存在的合理结构）
        if pre_docking_mmff_minimize or pre_docking_etkdg_reembed:
            mol_for_dock = Chem.Mol(mol.ToBinary())  # 深拷贝，避免影响后续化学指标计算
            apply_pre_docking_preparation(
                mol_for_dock,
                mmff_minimize=pre_docking_mmff_minimize,
                mmff_max_iters=pre_docking_mmff_max_iters,
                use_uff_fallback=pre_docking_use_uff_fallback,
                etkdg_reembed=pre_docking_etkdg_reembed,
                debug=debug,
            )
            mol = mol_for_dock  # 使用预处理后的分子进行对接

        # 4.1 键长/原子对分布JSD（在预处理后计算，反映优化后的键长质量）
        try:
            bond_distances = eval_bond_length.bond_distance_from_mol(mol)
            if bond_distances:
                bond_length_profile = eval_bond_length.get_bond_length_profile(
                    bond_distances
                )
                bond_length_jsd_dict = eval_bond_length.eval_bond_length_profile(
                    bond_length_profile
                )
                result["bond_length_jsd"] = bond_length_jsd_dict
                if debug:
                    jsd_values = [
                        v for v in bond_length_jsd_dict.values() if v is not None
                    ]
                    if jsd_values:
                        print(
                            f"  ✅ 键长分布JSD(预处理后): 平均 {np.mean(jsd_values):.4f} (共{len(jsd_values)}种键类型)"
                        )
        except Exception as e:
            if debug:
                print(f"  ⚠️  键长分布计算失败: {e}")
        try:
            pos = mol.GetConformer().GetPositions()
            elements = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
            pair_distances = eval_bond_length.pair_distance_from_pos_v(pos, elements)
            if pair_distances:
                pair_length_profile = eval_bond_length.get_pair_length_profile(
                    pair_distances
                )
                pair_length_jsd_dict = eval_bond_length.eval_pair_length_profile(
                    pair_length_profile
                )
                result["pair_length_jsd"] = pair_length_jsd_dict
                if debug:
                    jsd_values = [
                        v for v in pair_length_jsd_dict.values() if v is not None
                    ]
                    if jsd_values:
                        print(
                            f"  ✅ 原子对距离分布JSD(预处理后): 平均 {np.mean(jsd_values):.4f} (共{len(jsd_values)}种类型)"
                        )
        except Exception as e:
            if debug:
                print(f"  ⚠️  原子对距离分布计算失败: {e}")

        modes = _DEFAULT_VINA_MODES if vina_modes is None else frozenset(vina_modes)

        if not modes:
            result["success"] = True
        else:
            if debug:
                print(f"  - 开始AutoDock Vina（模式: {', '.join(sorted(modes))}）...")

            # ✅ 为每个分子创建独立的临时目录，避免并行执行时的文件冲突
            import os
            import threading

            if tmp_dir is None:
                pocket_id = (
                    "custom"
                    if (not ligand_filename or ligand_filename == "N/A")
                    else Path(ligand_filename).stem
                )
                pocket_id = (
                    pocket_id.replace("/", "_").replace("\\", "_").replace(":", "_")
                )
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                base_tmp_dir = DOCK_TMP_DIR / f"tmp_docking_{pocket_id}_{timestamp}"
            else:
                base_tmp_dir = Path(tmp_dir)

            process_id = os.getpid()
            thread_id = threading.current_thread().ident
            microsecond_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            unique_mol_tmp_dir = (
                base_tmp_dir / f"mol_{process_id}_{thread_id}_{microsecond_timestamp}"
            )
            unique_mol_tmp_dir.mkdir(parents=True, exist_ok=True)

            vina_task = VinaDockingTask.from_generated_mol(
                mol,
                ligand_filename,
                protein_root=protein_root,
                protein_path=protein_path,
                tmp_dir=str(unique_mol_tmp_dir),
                size_factor=size_factor,
                buffer=buffer,
            )

            docking_errors = []

            if "dock" in modes:
                try:
                    vina_dock_results = vina_task.run(
                        mode="dock", exhaustiveness=exhaustiveness, n_poses=n_poses
                    )
                    result["vina_dock"] = vina_dock_results
                    if debug and vina_dock_results:
                        print(
                            f"  ✅ dock模式成功: {vina_dock_results[0]['affinity']:.3f} kcal/mol"
                        )
                except Exception as e:
                    error_msg = f"dock模式失败: {str(e)}"
                    docking_errors.append(error_msg)
                    if debug:
                        print(f"  ⚠️  {error_msg}")

            if "score_only" in modes:
                try:
                    vina_score_only_results = vina_task.run(
                        mode="score_only",
                        exhaustiveness=exhaustiveness,
                        n_poses=n_poses,
                    )
                    if vina_score_only_results and len(vina_score_only_results) > 0:
                        result["vina_score_only"] = vina_score_only_results[0][
                            "affinity"
                        ]
                        if debug:
                            print(
                                f"  ✅ score_only模式成功: {result['vina_score_only']:.3f} kcal/mol"
                            )
                    else:
                        error_msg = "score_only模式返回空结果"
                        docking_errors.append(error_msg)
                        if debug:
                            print(f"  ⚠️  {error_msg}")
                except Exception as e:
                    error_msg = f"score_only模式失败: {str(e)}"
                    docking_errors.append(error_msg)
                    if debug:
                        print(f"  ⚠️  {error_msg}")

            if "minimize" in modes:
                try:
                    vina_minimize_results = vina_task.run(
                        mode="minimize", exhaustiveness=exhaustiveness, n_poses=n_poses
                    )
                    if vina_minimize_results and len(vina_minimize_results) > 0:
                        result["vina_minimize"] = vina_minimize_results[0]["affinity"]
                        if debug:
                            print(
                                f"  ✅ minimize模式成功: {result['vina_minimize']:.3f} kcal/mol"
                            )
                    else:
                        error_msg = "minimize模式返回空结果"
                        docking_errors.append(error_msg)
                        if debug:
                            print(f"  ⚠️  {error_msg}")
                except Exception as e:
                    error_msg = f"minimize模式失败: {str(e)}"
                    docking_errors.append(error_msg)
                    if debug:
                        print(f"  ⚠️  {error_msg}")

            if (
                result["vina_dock"]
                or result["vina_score_only"] is not None
                or result["vina_minimize"] is not None
            ):
                result["success"] = True
            elif docking_errors:
                result["error"] = "; ".join(docking_errors)
            else:
                result["error"] = (
                    "对接未返回有效评分（可开 --debug 或检查配体 PDBQT / Meeko）"
                )

        result["mol"] = mol

    except Exception as e:
        if debug:
            print(f"  ❌ 评估失败: {e}")
            traceback.print_exc()
        # 即使对接失败，也保存错误信息
        result["error"] = str(e)

    return result
