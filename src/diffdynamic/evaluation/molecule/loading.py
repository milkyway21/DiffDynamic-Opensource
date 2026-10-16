"""Loading responsibilities extracted from evaluate_pt_with_correct_reconstruct.py."""

from diffdynamic.evaluation.molecule.state import (
    Path,
    SimpleNamespace,
    np,
    os,
    re,
    torch,
    traceback,
)
from diffdynamic.evaluation.molecule import state as _state


def _trusted_repo_globals():
    """本仓库采样输出会 pickle 的可信类（数据容器，不含可执行逻辑）。

    PyG 的 DataEdgeAttr/DataTensorAttr 是张量属性描述类，反序列化合法的
    ProteinLigandData 时必需；将其加入白名单不放开任意代码执行。
    """
    pass  # process state is held in state.py
    if _state._TRUSTED_SAFE_GLOBALS is None:
        classes = []
        try:
            from datasets.pl_data import ProteinLigandData

            classes.append(ProteinLigandData)
        except Exception:
            pass
        try:
            from types import SimpleNamespace

            classes.append(SimpleNamespace)
        except Exception:
            pass
        for attr_cls in ("DataEdgeAttr", "DataTensorAttr", "DataAttr", "GlobalAttr"):
            try:
                import importlib

                mod = importlib.import_module("torch_geometric.data.data")
                classes.append(getattr(mod, attr_cls))
            except Exception:
                continue
        try:
            from torch_geometric.data.storage import GlobalStorage

            classes.append(GlobalStorage)
        except Exception:
            pass
        # numpy 数组是采样输出的常见成员；_reconstruct/dtype 属于数组重建
        # 机制，不构成代码执行面。
        try:
            import numpy as _np

            classes.append(_np.core.multiarray._reconstruct)
            classes.append(_np.dtype)
            classes.append(_np.ndarray)
        except Exception:
            pass
        try:
            import numpy as _np

            if hasattr(_np, "dtypes"):
                classes.extend(
                    v
                    for v in vars(_np.dtypes).values()
                    if isinstance(v, type) and v.__name__.endswith("DType")
                )
        except Exception:
            pass
        # pickle 内字符串重建所需（编码参数仅限已注册 codec）
        try:
            import _codecs

            classes.append(_codecs.encode)
        except Exception:
            pass
        _state._TRUSTED_SAFE_GLOBALS = classes
    return _state._TRUSTED_SAFE_GLOBALS


def _torch_load_layered(pt_path_str):
    """分层加载 .pt：优先 weights_only 严格模式。

    1) weights_only=True（最严格，纯张量文件直接通过）；
    2) 失败则把本仓库可信数据类加入 safe_globals 白名单后重试；
    3) 仍失败（如文件内嵌 numpy dtype，torch 2.4 白名单无法覆盖）时，
       对本地生成的可信文件回退 weights_only=False。
       环境变量 DD_PT_STRICT_LOAD=1（服务端评估用户上传等不可信 .pt 时设置）
       表示文件非本仓库生成：严格加载失败直接抛错，绝不回退到 pickle 模式，
       防止反序列化执行恶意代码。
    """
    from diffdynamic.io.normalized_pt import restore_normalized_result

    try:
        return restore_normalized_result(
            torch.load(pt_path_str, map_location="cpu", weights_only=True)
        )
    except Exception:
        pass
    try:
        globals_ = _trusted_repo_globals()
        if globals_:
            torch.serialization.add_safe_globals(globals_)
        return restore_normalized_result(
            torch.load(pt_path_str, map_location="cpu", weights_only=True)
        )
    except Exception as strict_err:
        if os.environ.get("DD_PT_STRICT_LOAD") == "1":
            raise ValueError(
                f"不可信 .pt（用户上传等非本仓库生成文件）只允许 weights_only 严格加载，"
                f"加载失败: {type(strict_err).__name__}: {strict_err}"
            ) from strict_err
        print(
            f"   ⚠️ weights_only 严格加载失败（{type(strict_err).__name__}），"
            f"回退为信任模式加载本地生成文件: {pt_path_str}"
        )
        return restore_normalized_result(
            torch.load(pt_path_str, map_location="cpu", weights_only=False)
        )


def load_pt_file(pt_path):
    """加载.pt文件"""
    # 确保pt_path是字符串格式
    pt_path_str = str(pt_path)
    print(f"正在加载 .pt 文件: {pt_path_str}")

    # 检查文件是否存在
    pt_path_obj = Path(pt_path_str)

    # 尝试解析路径（处理Windows/Linux路径转换）
    if not pt_path_obj.exists():
        # 尝试Windows路径格式（如果是在WSL/Linux环境下）
        if pt_path_str.startswith("/mnt/"):
            # 尝试直接使用Windows路径格式
            win_path = (
                pt_path_str.replace("/mnt/", "")
                .replace("/", ":\\", 1)
                .replace("/", "\\")
            )
            print(f"   尝试Windows路径格式: {win_path}")
            win_path_obj = Path(win_path)
            if win_path_obj.exists():
                pt_path_obj = win_path_obj
                pt_path_str = str(pt_path_obj)
                print(f"   ✅ 找到文件（使用Windows路径格式）")
            else:
                print(f"❌ 错误: .pt文件不存在")
                print(f"   尝试的路径1: {pt_path_str}")
                print(f"   尝试的路径2: {win_path}")
                print(f"   当前工作目录: {os.getcwd()}")
                print(f"   建议: 请检查文件路径是否正确，或使用绝对路径")
                return None
        else:
            print(f"❌ 错误: .pt文件不存在: {pt_path_str}")
            print(f"   当前工作目录: {os.getcwd()}")
            print(f"   建议: 请检查文件路径是否正确，或使用绝对路径")
            return None

    # 检查文件权限
    try:
        if not os.access(pt_path_obj, os.R_OK):
            print(f"⚠️  警告: 文件存在但可能没有读取权限: {pt_path_str}")
    except Exception as e:
        print(f"⚠️  警告: 无法检查文件权限: {e}")

    # 检查文件大小
    file_size = None
    try:
        file_size = pt_path_obj.stat().st_size
        print(f"   文件大小: {file_size} 字节 ({file_size / (1024 * 1024):.2f} MB)")

        if file_size == 0:
            print(f"❌ 错误: .pt文件为空（大小为0字节）: {pt_path_str}")
            print(f"   可能的原因:")
            print(f"   1. 文件在保存过程中被中断")
            print(f"   2. 文件系统错误")
            print(f"   3. Windows/Linux路径映射问题（WSL环境）")
            print(f"   4. 文件权限问题")
            print(f"   诊断信息:")
            print(f"   - Linux路径: {pt_path_str}")
            if pt_path_str.startswith("/mnt/"):
                win_path = (
                    pt_path_str.replace("/mnt/", "")
                    .replace("/", ":\\", 1)
                    .replace("/", "\\")
                )
                print(f"   - 对应的Windows路径: {win_path}")
            print(f"   - 文件是否存在: {pt_path_obj.exists()}")
            print(f"   - 文件大小: {file_size} 字节")
            try:
                import stat

                file_stat = pt_path_obj.stat()
                print(f"   - 文件权限: {oct(file_stat.st_mode)}")
                print(
                    f"   - 文件所有者: UID={file_stat.st_uid}, GID={file_stat.st_gid}"
                )
            except Exception as e:
                print(f"   - 无法获取文件详细信息: {e}")
            print(f"   建议:")
            print(f"   1. 在Windows下检查文件大小:")
            if pt_path_str.startswith("/mnt/"):
                win_path = (
                    pt_path_str.replace("/mnt/", "")
                    .replace("/", ":\\", 1)
                    .replace("/", "\\")
                )
                print(f"      - 打开文件资源管理器")
                print(f"      - 导航到: {win_path}")
                print(f"      - 检查文件大小和最后修改时间")
            print(f"   2. 如果Windows下文件有内容但WSL下显示为0字节:")
            print(f"      - 可能是WSL挂载问题，尝试重启WSL: wsl --shutdown")
            print(f"      - 或者将文件复制到Linux文件系统:")
            print(f"        cp {pt_path_str} ~/data/")
            print(f"   3. 如果Windows下文件也是0字节:")
            print(f"      - 文件可能损坏，需要重新生成")
            return None
    except Exception as e:
        print(f"❌ 错误: 无法获取文件大小: {e}")
        print(f"   文件路径: {pt_path_str}")
        print(f"   文件是否存在: {pt_path_obj.exists()}")
        traceback.print_exc()
        return None

    # 如果文件大小为0，已经在上面的if中返回了，这里确保file_size不为None
    if file_size is None:
        return None

    # 尝试读取文件的前几个字节来验证文件是否可读
    try:
        with open(pt_path_obj, "rb") as f:
            first_bytes = f.read(16)
            if len(first_bytes) == 0:
                print(f"❌ 错误: 文件存在但无法读取内容（可能为空或权限问题）")
                print(f"   尝试使用ls命令检查文件:")
                print(f"   ls -lh {pt_path_str}")
                return None
            else:
                print(f"   ✅ 文件可读，前16字节: {first_bytes.hex()}")
    except PermissionError as e:
        print(f"❌ 错误: 文件权限不足，无法读取: {e}")
        print(f"   尝试使用chmod修改权限或使用sudo")
        return None
    except Exception as e:
        print(f"❌ 错误: 无法读取文件: {e}")
        print(f"   可能的原因: 文件权限问题或文件系统错误")
        print(f"   尝试使用ls命令检查文件:")
        print(f"   ls -lh {pt_path_str}")
        return None

    try:
        print(f"   正在加载文件内容...")
        data = _torch_load_layered(pt_path_str)
        print(f"✅ 成功加载 .pt 文件")
        return data
    except EOFError as e:
        print(f"❌ 加载失败: 文件可能损坏或不完整（EOFError）")
        print(f"   错误详情: {e}")
        print(f"   文件路径: {pt_path_str}")
        if file_size is not None:
            print(f"   文件大小: {file_size} 字节")
        print(f"   建议: 请检查文件是否完整，可能需要重新生成.pt文件")
        return None
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        print(f"   文件路径: {pt_path_str}")
        if file_size is not None:
            print(f"   文件大小: {file_size} 字节")
        traceback.print_exc()
        return None


def _parse_data_id_from_pt_path_decompdiff(pt_path):
    """
    从路径推断 CrossDocked / pocket 测试编号：pocketbench_010 -> 10；
    否则用文件名 eval_000_* / result_0_* 的首段数字。
    """
    s = str(pt_path)
    m = re.search(r"pocketbench_(\d+)", s, re.IGNORECASE)
    if m:
        return int(m.group(1))
    stem = Path(pt_path).stem
    m2 = re.match(r"eval_(\d+)", stem, re.IGNORECASE)
    if m2:
        return int(m2.group(1))
    m3 = re.match(r"result_(\d+)", stem, re.IGNORECASE)
    if m3:
        return int(m3.group(1))
    return None


def decompdiff_list_to_targetdiff_dict(raw_list, pt_path):
    """
    将 DecompDiff 保存的 list[dict]（每元素含 pred_pos, pred_v）转为本评估脚本使用的 dict。

    pred_v 与 DecompDiff 采样（ligand_atom_mode=add_aromatic）一致，为 **add_aromatic** 词表 0..12；
    存盘时常见 1..13 的 1-based。与 TargetDiff/本仓库 ``utils/transforms`` 的 add_aromatic 定义相同（非 basic 0..7）。

    Returns:
        (dict | None, meta): meta 可含 decompdiff_use_add_aromatic，评估阶段须使用 atom_mode=add_aromatic
    """
    if not raw_list or not isinstance(raw_list, list):
        return None, {}
    s0 = raw_list[0]
    if not isinstance(s0, dict) or "pred_pos" not in s0 or "pred_v" not in s0:
        return None, {}
    # 全文件 pred_v 数值范围
    gmin, gmax = 10**9, -1
    for x in raw_list:
        if not isinstance(x, dict) or "pred_v" not in x:
            continue
        t = np.asarray(x["pred_v"])
        if t.size:
            gmin = min(gmin, int(t.min()))
            gmax = max(gmax, int(t.max()))
    if gmax < 0:
        print("❌ DecompDiff 格式：所有 pred_v 为空")
        return None, {}
    if gmax > 24:
        print(f"❌ DecompDiff 格式：pred_v 范围异常 [{gmin}, {gmax}]，未做转换")
        return None, {}
    if gmin >= 1 and gmax <= 13:
        one_based = True
        print(
            "  ℹ️  检测到 DecompDiff .pt（list）：pred_v 为 **1-based add_aromatic**，"
            "将减 1 对齐 0..12（与 DecompDiff 采样原子编码一致）。"
        )
    elif gmin == 0 and gmax <= 12:
        one_based = False
        print(
            "  ℹ️  检测到 DecompDiff .pt（list）：pred_v 为 0-based add_aromatic（0..12）。"
        )
    else:
        one_based = False
        print(
            f"  ⚠️  DecompDiff pred_v 范围 [{gmin}, {gmax}] 非常规，按 0-based add_aromatic 解码。"
        )

    pred_ligand_pos = []
    pred_ligand_v = []
    for x in raw_list:
        p_t = torch.as_tensor(np.asarray(x["pred_pos"], dtype=np.float32))
        v = np.asarray(x["pred_v"], dtype=np.int64)
        if one_based:
            v = v - 1
        v_t = torch.as_tensor(v, dtype=torch.long)
        pred_ligand_pos.append(p_t)
        pred_ligand_v.append(v_t)

    ligand_fn, protein_fn = None, None
    for x in raw_list:
        if not isinstance(x, dict):
            continue
        if not ligand_fn and x.get("ligand_filename"):
            ligand_fn = x.get("ligand_filename")
        if not protein_fn and x.get("protein_filename"):
            protein_fn = x.get("protein_filename")
        d = x.get("data")
        if d is not None:
            if not ligand_fn and getattr(d, "ligand_filename", None):
                ligand_fn = getattr(d, "ligand_filename", None)
            if not protein_fn and getattr(d, "protein_filename", None):
                protein_fn = getattr(d, "protein_filename", None)

    dwrap = SimpleNamespace(
        ligand_filename=ligand_fn,
        protein_filename=protein_fn,
    )
    did = _parse_data_id_from_pt_path_decompdiff(pt_path)
    extra_info = {
        "data_id": did,
        "decompdiff_converted": True,
    }
    if did is not None:
        print(f"  ℹ️  从路径推断 data_id={did}，用于 index.pkl / 测试集配体补全。")
    out = {
        "pred_ligand_pos": pred_ligand_pos,
        "pred_ligand_v": pred_ligand_v,
        "data": dwrap,
        "extra_info": extra_info,
    }
    return out, {"decompdiff_use_add_aromatic": True}


def validate_pt_data(data):
    """验证.pt文件包含必需的字段"""
    if not isinstance(data, dict):
        print(
            f"❌ .pt 根对象须为 dict（或 list[DecompDiff] 已在入口转换）；当前: {type(data).__name__}"
        )
        if isinstance(data, list):
            print(
                "   若来自 DecompDiff，应含 pred_pos / pred_v；请更新 evaluate_pt_with_correct_reconstruct 转换逻辑或检查文件。"
            )
        return False

    required_keys = ["pred_ligand_pos", "pred_ligand_v", "data"]
    missing_keys = [key for key in required_keys if key not in data]

    if missing_keys:
        print(f"❌ .pt文件缺少必需字段: {missing_keys}")
        print(f"   可用字段: {list(data.keys())}")
        return False

    print(f"✅ .pt文件验证通过")
    print(f"   - 样本数量: {len(data['pred_ligand_pos'])}")
    print(f"   - 可用字段: {list(data.keys())}")
    return True
