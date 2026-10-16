"""Reconstruction responsibilities extracted from evaluate_pt_with_correct_reconstruct.py."""

from diffdynamic.evaluation.molecule.state import (
    Chem,
    np,
    reconstruct,
    reconstruct_targetdiff_legacy,
    torch,
    traceback,
    trans,
)
from diffdynamic.evaluation.molecule import state as _state


def remove_small_fragments(mol, debug=False, scaffold_substruct=None):
    """Remove very small fragments (< min_atoms), keeping all substantial fragments.

    When scaffold_substruct is provided, the scaffold-containing fragment is
    always kept regardless of size. Other fragments are kept if they have at
    least min_atoms heavy atoms.

    Args:
        mol: RDKit Mol object (may contain multiple fragments)
        debug: print debug info
        scaffold_substruct: optional RDKit Mol; scaffold fragment always kept.

    Returns:
        RDKit Mol with small fragments removed, or None if input is invalid
    """
    if mol is None:
        return None
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    if len(frags) <= 1:
        return mol  # No fragments to remove
    min_atoms = 3  # Remove fragments with fewer than 3 heavy atoms
    kept = []
    for f in frags:
        keep = f.GetNumAtoms() >= min_atoms
        if not keep and scaffold_substruct is not None:
            if f.HasSubstructMatch(scaffold_substruct):
                keep = True
        if keep:
            kept.append(f)
    if not kept:
        # Fallback: keep largest
        largest = max(frags, key=lambda m: m.GetNumAtoms())
        if debug:
            sizes = [m.GetNumAtoms() for m in frags]
            print(
                f"  Fragment removal: {len(frags)} fragments, sizes={sizes}, "
                f"all below min_atoms={min_atoms}, keeping largest ({largest.GetNumAtoms()} atoms)"
            )
        return largest
    if len(kept) == len(frags):
        if debug:
            sizes = [m.GetNumAtoms() for m in frags]
            print(
                f"  Fragment removal: {len(frags)} fragments, sizes={sizes}, "
                f"all >= {min_atoms} atoms, keeping all"
            )
        return mol  # Nothing removed
    # Combine kept fragments
    combined = kept[0]
    for f in kept[1:]:
        combined = Chem.CombineMols(combined, f)
    if debug:
        sizes = [m.GetNumAtoms() for m in frags]
        kept_sizes = [m.GetNumAtoms() for m in kept]
        print(
            f"  Fragment removal: {len(frags)} fragments, sizes={sizes}, "
            f"kept={kept_sizes}, removed {len(frags) - len(kept)} small fragments"
        )
    return combined


def _scaffold_attachment_bonds(record, scaffold_cfg, atom_count):
    """Build protected anchor bonds from the generated site allocation.

    The placement code emits extra atoms grouped by active site. Connecting
    the first atom of each group to that site's scaffold anchor keeps a noisy
    distance reconstruction from dropping an otherwise valid branch.
    """
    if not record or not scaffold_cfg:
        return []
    n_scaffold = int(scaffold_cfg.get("n_scaffold_atoms") or 0)
    if n_scaffold <= 0 or atom_count <= n_scaffold:
        return []
    sites = {
        int(site.get("site_id")): site
        for site in scaffold_cfg.get("attachment_sites") or []
        if site.get("site_id") is not None
    }
    bonds = []
    cursor = n_scaffold
    for allocation in record.get("site_allocation") or []:
        count = int(allocation.get("count") or 0)
        if count <= 0:
            continue
        site_id = allocation.get("site_id")
        site = sites.get(int(site_id)) if site_id is not None else None
        anchor = site.get("anchor_scaffold_idx") if site else None
        if anchor is not None and 0 <= int(anchor) < n_scaffold:
            if cursor < atom_count:
                bonds.append((int(anchor), int(cursor), 1, False))
        cursor += count
    return bonds


def reconstruct_molecule(
    pos,
    v,
    atom_mode="add_aromatic",
    debug=False,
    rdkit_structure_repair=None,
    scaffold_bonds=None,
    n_scaffold=None,
    extra_attachment_bonds=None,
    covalent_factor=None,
    reconstruction_method=None,
):
    """
    使用正确的reconstruct方法重建单个分子

    Args:
        pos: 原子坐标 (N, 3)
        v: 原子类型索引 (N,) 或 (N, num_classes)
        atom_mode: 原子编码模式 ('basic' 或 'add_aromatic')
        debug: 是否打印调试信息
        rdkit_structure_repair: optional dict for sample.rdkit_structure_repair;
            if None, uses module-level ``_RDKIT_STRUCTURE_REPAIR_CFG`` (set by CLI).
        scaffold_bonds: optional list of (i, j[, order[, aromatic]]) for scaffold atoms;
            passed to reconstruct_from_generated to protect scaffold topology.
        n_scaffold: int, number of scaffold atoms at the start of the atom list;
            passed to reconstruct_from_generated to protect scaffold bonds.
        extra_attachment_bonds: legacy protected site bonds. Strict scaffold
            evaluation leaves this empty so coordinates determine extra bonds.
        covalent_factor: optional distance-bonding multiplier. ``None`` keeps
            the historical reconstruction default.

    Returns:
        tuple: (mol, error_info)
        - mol: RDKit分子对象或None
        - error_info: dict包含错误类型和错误信息，成功时为None
    """
    if rdkit_structure_repair is None:
        rdkit_structure_repair = _state._RDKIT_STRUCTURE_REPAIR_CFG
    method = reconstruction_method or _state._RECONSTRUCTION_METHOD
    if method not in ("current", "targetdiff_legacy"):
        raise ValueError("Unknown reconstruction method: " + str(method))
    if method == "targetdiff_legacy" and (
        scaffold_bonds or n_scaffold or extra_attachment_bonds or rdkit_structure_repair
    ):
        raise ValueError(
            "TargetDiff legacy reconstruction accepts de novo "
            "coordinates only; scaffold locks and repair are unsupported"
        )
    error_info = None
    try:
        # 转换为numpy
        if torch.is_tensor(pos):
            pos_array = pos.detach().cpu().numpy()
        else:
            pos_array = np.array(pos)

        if torch.is_tensor(v):
            v_tensor = v.detach().cpu()
        else:
            v_tensor = torch.tensor(v)

        # 如果v是one-hot编码，转换为索引
        if v_tensor.dim() > 1:
            v_tensor = v_tensor.argmax(dim=-1)

        # 获取原子序数和芳香性标记
        atom_numbers = trans.get_atomic_number_from_index(v_tensor, mode=atom_mode)
        aromatic_flags = trans.is_aromatic_from_index(v_tensor, mode=atom_mode)

        if debug:
            print(f"  - 原子数量: {len(atom_numbers)}")
            print(f"  - 原子类型: {set(atom_numbers)}")
            if aromatic_flags:
                print(f"  - 芳香原子: {sum(aromatic_flags)}/{len(aromatic_flags)}")

        # 使用 OpenBabel 重建。PerceiveBondOrders 之后会按 aromatic_flags
        # 回盖 extra 原子芳香性；bool 列表即使未显式 basic_mode=False 也会生效。
        if method == "targetdiff_legacy":
            mol = reconstruct_targetdiff_legacy.reconstruct_from_generated(
                pos_array.tolist(),
                atom_numbers,
                aromatic_flags,
                basic_mode=(atom_mode == "basic"),
            )
        else:
            mol = reconstruct.reconstruct_from_generated(
                pos_array.tolist(),
                atom_numbers,
                aromatic_flags,
                basic_mode=(atom_mode == "basic"),
                scaffold_bonds=scaffold_bonds,
                n_scaffold=n_scaffold,
                extra_attachment_bonds=extra_attachment_bonds,
                covalent_factor=(
                    2.0 if covalent_factor is None else float(covalent_factor)
                ),
                rdkit_structure_repair=rdkit_structure_repair,
            )

        if mol is None:
            if debug:
                print("  ❌ reconstruct返回None")
            error_info = {
                "error_type": "reconstruct_returned_none",
                "error_msg": "reconstruct_from_generated返回None",
                "atom_count": len(atom_numbers),
            }
            return None, error_info

        # 验证分子
        try:
            smiles = Chem.MolToSmiles(mol)
            if debug:
                print(f"  ✅ SMILES: {smiles}")
            return mol, None
        except Exception as e:
            if debug:
                print(f"  ⚠️  生成SMILES失败: {e}")
            # SMILES生成失败但分子对象存在，仍视为成功
            return mol, None

    except (
        reconstruct.MolReconsError,
        reconstruct_targetdiff_legacy.MolReconsError,
    ) as e:
        if debug:
            print(f"  ❌ 重建失败 (MolReconsError): {e}")
        error_info = {
            "error_type": "MolReconsError",
            "error_msg": str(e),
            "atom_count": len(atom_numbers) if "atom_numbers" in locals() else 0,
        }
        return None, error_info
    except Exception as e:
        if debug:
            print(f"  ❌ 重建失败: {e}")
            traceback.print_exc()
        error_info = {
            "error_type": type(e).__name__,
            "error_msg": str(e),
            "atom_count": len(atom_numbers) if "atom_numbers" in locals() else 0,
        }
        return None, error_info
