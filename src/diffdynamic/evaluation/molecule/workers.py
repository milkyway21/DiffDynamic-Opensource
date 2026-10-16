"""Workers responsibilities extracted from evaluate_pt_with_correct_reconstruct.py."""

from diffdynamic.evaluation.molecule.state import DEFAULT_SINGLE_MOL_EVAL_TIMEOUT_SEC

from diffdynamic.evaluation.molecule.state import (
    Path,
    _DEFAULT_VINA_MODES,
    datetime,
    multiprocessing,
    os,
    pickle,
    tempfile,
    time,
    traceback,
)
from diffdynamic.evaluation.molecule import state as _state


def _evaluate_single_molecule_worker(args_tuple):
    """
    在子进程中评估单个分子的工作函数

    这个函数在独立的子进程中运行，即使底层C++库崩溃也不会影响主进程。
    注意：需要在主模块中定义，以便multiprocessing可以正确导入。
    """
    from diffdynamic.evaluation.molecule.scoring import evaluate_single_molecule

    try:
        # 重新导入必要的模块（子进程中需要重新导入）
        import sys
        from pathlib import Path
        from datetime import datetime

        # 解包参数（兼容旧版10-11元组与新版15–16元组，含 pre_docking 与 vina_modes）
        protein_path = args_tuple[10] if len(args_tuple) > 10 else None
        pre_docking_mmff_minimize = args_tuple[11] if len(args_tuple) > 11 else False
        pre_docking_mmff_max_iters = args_tuple[12] if len(args_tuple) > 12 else 200
        pre_docking_use_uff_fallback = args_tuple[13] if len(args_tuple) > 13 else True
        pre_docking_etkdg_reembed = args_tuple[14] if len(args_tuple) > 14 else False
        vina_modes_tuple = args_tuple[15] if len(args_tuple) > 15 else None
        # ``None`` means the legacy default, while an empty tuple is the explicit
        # ``--vina-modes none`` sentinel.  Do not collapse both values by truthiness:
        # the isolated worker must preserve the caller's no-Vina request.
        vina_modes_eff = (
            _DEFAULT_VINA_MODES
            if vina_modes_tuple is None
            else frozenset(vina_modes_tuple)
        )
        (
            mol_pickle_path,
            ligand_filename,
            protein_root,
            exhaustiveness,
            n_poses,
            size_factor,
            buffer,
            tmp_dir,
            mol_idx,
            debug,
        ) = args_tuple[:10]

        # 加载分子对象
        try:
            with open(mol_pickle_path, "rb") as f:
                mol = pickle.load(f)
        except Exception as pickle_e:
            return {
                "mol_idx": mol_idx,
                "mol": None,
                "smiles": None,
                "chem": None,
                "vina_dock": None,
                "vina_score_only": None,
                "vina_minimize": None,
                "atom_type_jsd": None,
                "bond_length_jsd": None,
                "pair_length_jsd": None,
                "success": False,
                "error": f"无法加载分子对象: {str(pickle_e)}",
                # 新增评估指标
                "stability": None,
                "basic_info": None,
                "logp": None,
                "lipinski": None,
                "pains": None,
                "tanimoto_sim": None,
                "rdkit_rmsd": None,
                "conformer_energy": None,
                "tpsa": None,
                "rdkit_valid": None,
                "lilly_medchem_passed": None,
                "lilly_medchem_demerit": None,
                "lilly_medchem_description": None,
                "comprehensive_score": None,
                "molecule_id": None,
            }

        # 调用评估函数（需要在子进程中重新导入）
        # 注意：这里直接调用，因为函数已经在模块级别定义
        from diffdynamic.evaluation import scoring_func
        from diffdynamic.evaluation.docking_vina import VinaDockingTask
        from diffdynamic.evaluation import eval_atom_type
        from diffdynamic.evaluation import eval_bond_length
        from collections import Counter
        from rdkit import Chem

        result = evaluate_single_molecule(
            mol,
            ligand_filename,
            protein_root,
            exhaustiveness=exhaustiveness,
            n_poses=n_poses,
            size_factor=size_factor,
            buffer=buffer,
            debug=debug,  # 使用传递的debug参数
            tmp_dir=tmp_dir,
            protein_path=protein_path,
            pre_docking_mmff_minimize=pre_docking_mmff_minimize,
            pre_docking_mmff_max_iters=pre_docking_mmff_max_iters,
            pre_docking_use_uff_fallback=pre_docking_use_uff_fallback,
            pre_docking_etkdg_reembed=pre_docking_etkdg_reembed,
            vina_modes=vina_modes_eff,
        )

        # 添加分子索引
        result["mol_idx"] = mol_idx

        # 清理临时pickle文件
        try:
            os.remove(mol_pickle_path)
        except:
            pass

        return result
    except Exception as e:
        error_trace = traceback.format_exc()
        # 返回错误结果
        return {
            "mol_idx": mol_idx if "mol_idx" in locals() else -1,
            "mol": None,
            "smiles": None,
            "chem": None,
            "vina_dock": None,
            "vina_score_only": None,
            "vina_minimize": None,
            "atom_type_jsd": None,
            "bond_length_jsd": None,
            "pair_length_jsd": None,
            "success": False,
            "error": f"子进程评估异常: {str(e)}\n{error_trace[:500]}",  # 限制错误信息长度
            # 新增评估指标
            "stability": None,
            "basic_info": None,
            "logp": None,
            "lipinski": None,
            "pains": None,
            "tanimoto_sim": None,
            "rdkit_rmsd": None,
            "conformer_energy": None,
            "tpsa": None,
        }


def evaluate_single_molecule_isolated(
    mol,
    ligand_filename,
    protein_root,
    exhaustiveness=8,
    n_poses=1,
    size_factor=1.0,
    buffer=5.0,
    debug=False,
    tmp_dir=None,
    mol_idx=None,
    timeout=DEFAULT_SINGLE_MOL_EVAL_TIMEOUT_SEC,
    use_isolation=False,
    protein_path=None,
    pre_docking_mmff_minimize=False,
    pre_docking_mmff_max_iters=200,
    pre_docking_use_uff_fallback=True,
    pre_docking_etkdg_reembed=False,
    vina_modes=None,
):
    """
    评估单个分子：默认同进程调用 evaluate_single_molecule；可选子进程隔离以防底层 C++ 崩溃拖垮主进程。

    Args:
        timeout: 隔离模式下的子进程超时（秒），默认 10800（3h）；非隔离时忽略（单次 Vina 另见 VINA_DOCK_TIMEOUT_SEC，默认 128s）
        use_isolation: 是否启用单分子子进程隔离（默认 False，顺序流水线评估）

    Returns:
        dict: 评估结果；隔离模式下子进程异常时返回失败结果字典
    """
    from diffdynamic.evaluation.molecule.scoring import evaluate_single_molecule

    vm_eff = _DEFAULT_VINA_MODES if vina_modes is None else frozenset(vina_modes)
    # 如果禁用隔离模式，直接调用原函数
    if not use_isolation or os.environ.get("DISABLE_EVAL_ISOLATION", "").lower() in (
        "1",
        "true",
        "yes",
    ):
        return evaluate_single_molecule(
            mol,
            ligand_filename,
            protein_root,
            exhaustiveness=exhaustiveness,
            n_poses=n_poses,
            size_factor=size_factor,
            buffer=buffer,
            debug=debug,
            tmp_dir=tmp_dir,
            protein_path=protein_path,
            pre_docking_mmff_minimize=pre_docking_mmff_minimize,
            pre_docking_mmff_max_iters=pre_docking_mmff_max_iters,
            pre_docking_use_uff_fallback=pre_docking_use_uff_fallback,
            pre_docking_etkdg_reembed=pre_docking_etkdg_reembed,
            vina_modes=vm_eff,
        )

    # 创建临时文件保存分子对象（因为multiprocessing需要可序列化的对象）
    temp_dir = Path(tempfile.gettempdir())
    mol_pickle_path = (
        temp_dir
        / f"mol_eval_{os.getpid()}_{mol_idx if mol_idx is not None else 'unknown'}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.pkl"
    )

    try:
        # 保存分子到临时文件
        with open(mol_pickle_path, "wb") as f:
            pickle.dump(mol, f)

        # 准备参数（兼容旧版10-11元组与新版14元组，含 pre_docking 参数）
        args_tuple = (
            str(mol_pickle_path),
            ligand_filename,
            protein_root,
            exhaustiveness,
            n_poses,
            size_factor,
            buffer,
            tmp_dir,
            mol_idx if mol_idx is not None else -1,
            debug,  # 传递debug参数
            protein_path,  # 自定义模式：直接指定蛋白路径
            pre_docking_mmff_minimize,
            pre_docking_mmff_max_iters,
            pre_docking_use_uff_fallback,
            pre_docking_etkdg_reembed,
            tuple(sorted(vm_eff)),
        )

        # 使用进程池执行（每个分子一个进程）
        # 使用更可靠的超时和错误检测机制
        pool = None
        async_result = None
        try:
            pool = multiprocessing.Pool(processes=1)
            async_result = pool.apply_async(
                _evaluate_single_molecule_worker, (args_tuple,)
            )

            # 使用超时机制，并定期检查进程状态
            start_time = time.time()
            check_interval = 2  # 每2秒检查一次（更频繁的检查）
            last_progress_time = start_time

            while True:
                elapsed = time.time() - start_time

                # 如果超时，强制终止
                if elapsed >= timeout:
                    # 超时，强制终止进程池
                    if pool:
                        try:
                            pool.terminate()  # 立即终止所有工作进程
                            pool.join(timeout=2)  # 等待最多2秒
                        except:
                            pass
                        try:
                            if hasattr(pool, "_pool") and pool._pool:
                                for p in pool._pool:
                                    try:
                                        p.terminate()
                                        p.join(timeout=1)
                                        if p.is_alive():
                                            p.kill()
                                    except:
                                        pass
                        except:
                            pass

                    result = {
                        "mol_idx": mol_idx if mol_idx is not None else -1,
                        "mol": mol,
                        "smiles": None,
                        "chem": None,
                        "vina_dock": None,
                        "vina_score_only": None,
                        "vina_minimize": None,
                        "atom_type_jsd": None,
                        "bond_length_jsd": None,
                        "pair_length_jsd": None,
                        "success": False,
                        "error": f"评估超时（超过{timeout}秒），已强制终止",
                        # 新增评估指标
                        "stability": None,
                        "basic_info": None,
                        "logp": None,
                        "lipinski": None,
                        "pains": None,
                        "tanimoto_sim": None,
                        "rdkit_rmsd": None,
                        "conformer_energy": None,
                        "tpsa": None,
                        "rdkit_valid": None,
                        "lilly_medchem_passed": None,
                        "lilly_medchem_demerit": None,
                        "lilly_medchem_description": None,
                    }
                    break

                # 检查是否完成
                if async_result.ready():
                    try:
                        result = async_result.get(timeout=0.5)  # 快速获取结果
                        # 调试输出：检查返回结果中是否包含lilly_medchem字段
                        if debug and result:
                            print(
                                f"    [子进程返回] 包含lilly_medchem_passed: {'lilly_medchem_passed' in result}, "
                                f"值={result.get('lilly_medchem_passed', 'NOT_FOUND')}"
                            )
                        break
                    except Exception as e:
                        result = {
                            "mol_idx": mol_idx if mol_idx is not None else -1,
                            "mol": mol,
                            "smiles": None,
                            "chem": None,
                            "vina_dock": None,
                            "vina_score_only": None,
                            "vina_minimize": None,
                            "atom_type_jsd": None,
                            "bond_length_jsd": None,
                            "pair_length_jsd": None,
                            "success": False,
                            "error": f"获取结果异常: {str(e)}",
                            # 新增评估指标
                            "stability": None,
                            "basic_info": None,
                            "logp": None,
                            "lipinski": None,
                            "pains": None,
                            "tanimoto_sim": None,
                            "rdkit_rmsd": None,
                            "conformer_energy": None,
                            "tpsa": None,
                            "rdkit_valid": None,
                            "lilly_medchem_passed": None,
                            "lilly_medchem_demerit": None,
                            "lilly_medchem_description": None,
                        }
                        break

                # 每15秒输出一次进度提示（防止看起来卡住）
                if time.time() - last_progress_time >= 15:
                    elapsed_sec = int(elapsed)
                    print(
                        f"\n  ⏳ 分子 {mol_idx if mol_idx is not None else 'unknown'} 评估中... (已用时: {elapsed_sec}秒)",
                        flush=True,
                    )
                    last_progress_time = time.time()

                # 等待一段时间后再次检查
                sleep_time = min(check_interval, max(0.5, timeout - elapsed - 1))
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except multiprocessing.TimeoutError:
            result = {
                "mol_idx": mol_idx if mol_idx is not None else -1,
                "mol": mol,
                "smiles": None,
                "chem": None,
                "vina_dock": None,
                "vina_score_only": None,
                "vina_minimize": None,
                "atom_type_jsd": None,
                "bond_length_jsd": None,
                "pair_length_jsd": None,
                "success": False,
                "error": f"评估超时（超过{timeout}秒）",
                # 新增评估指标
                "stability": None,
                "basic_info": None,
                "logp": None,
                "lipinski": None,
                "pains": None,
                "tanimoto_sim": None,
                "rdkit_rmsd": None,
                "conformer_energy": None,
                "tpsa": None,
                "rdkit_valid": None,
                "lilly_medchem_passed": None,
                "lilly_medchem_demerit": None,
                "lilly_medchem_description": None,
            }
        except Exception as e:
            result = {
                "mol_idx": mol_idx if mol_idx is not None else -1,
                "mol": mol,
                "smiles": None,
                "chem": None,
                "vina_dock": None,
                "vina_score_only": None,
                "vina_minimize": None,
                "atom_type_jsd": None,
                "bond_length_jsd": None,
                "pair_length_jsd": None,
                "success": False,
                "error": f"子进程执行异常: {str(e)}",
                # 新增评估指标
                "stability": None,
                "basic_info": None,
                "logp": None,
                "lipinski": None,
                "pains": None,
                "tanimoto_sim": None,
                "rdkit_rmsd": None,
                "conformer_energy": None,
                "tpsa": None,
                "rdkit_valid": None,
                "lilly_medchem_passed": None,
                "lilly_medchem_demerit": None,
                "lilly_medchem_description": None,
            }
        finally:
            # 确保清理进程池
            if pool:
                try:
                    pool.terminate()
                    pool.join(timeout=2)
                except:
                    try:
                        pool.kill()
                    except:
                        pass

        # 确保分子对象被保留（从pickle文件重新加载）
        if result.get("mol") is None:
            try:
                with open(mol_pickle_path, "rb") as f:
                    result["mol"] = pickle.load(f)
            except:
                result["mol"] = mol  # 如果加载失败，使用原始分子

        # 调试输出：检查返回结果中是否包含lilly_medchem字段
        if debug and result:
            lilly_passed = result.get("lilly_medchem_passed", "NOT_FOUND")
            lilly_demerit = result.get("lilly_medchem_demerit", "NOT_FOUND")
            lilly_desc = result.get("lilly_medchem_description", "NOT_FOUND")
            print(
                f"    [返回结果检查] mol_idx={result.get('mol_idx')}, "
                f"lilly_medchem_passed={lilly_passed} (type={type(lilly_passed).__name__}), "
                f"lilly_medchem_demerit={lilly_demerit} (type={type(lilly_demerit).__name__}), "
                f"lilly_medchem_description存在={lilly_desc != 'NOT_FOUND'}"
            )

        return result

    except Exception as e:
        # 如果子进程机制失败，回退到直接调用（但会记录错误）
        if debug:
            print(f"  ⚠️  子进程隔离失败，回退到直接调用: {e}")

        try:
            return evaluate_single_molecule(
                mol,
                ligand_filename,
                protein_root,
                exhaustiveness=exhaustiveness,
                n_poses=n_poses,
                size_factor=size_factor,
                buffer=buffer,
                debug=debug,
                tmp_dir=tmp_dir,
                protein_path=protein_path,
                pre_docking_mmff_minimize=pre_docking_mmff_minimize,
                pre_docking_mmff_max_iters=pre_docking_mmff_max_iters,
                pre_docking_use_uff_fallback=pre_docking_use_uff_fallback,
                pre_docking_etkdg_reembed=pre_docking_etkdg_reembed,
                vina_modes=vm_eff,
            )
        except Exception as eval_e:
            return {
                "mol_idx": mol_idx if mol_idx is not None else -1,
                "mol": mol,
                "smiles": None,
                "chem": None,
                "vina_dock": None,
                "vina_score_only": None,
                # 新增评估指标
                "stability": None,
                "basic_info": None,
                "logp": None,
                "lipinski": None,
                "pains": None,
                "tanimoto_sim": None,
                "rdkit_rmsd": None,
                "conformer_energy": None,
                "tpsa": None,
                "rdkit_valid": None,
                "lilly_medchem_passed": None,
                "lilly_medchem_demerit": None,
                "lilly_medchem_description": None,
                "comprehensive_score": None,
                "molecule_id": None,
                "vina_minimize": None,
                "atom_type_jsd": None,
                "bond_length_jsd": None,
                "pair_length_jsd": None,
                "success": False,
                "error": f"评估失败: {str(eval_e)}",
            }
    finally:
        # 清理临时文件
        try:
            if mol_pickle_path.exists():
                os.remove(mol_pickle_path)
        except:
            pass
