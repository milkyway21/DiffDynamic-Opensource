"""Building responsibilities extracted from scaffold_prior.py."""

from __future__ import annotations

from diffdynamic.chemistry.profiles.state import Any, Iterable, Optional, Path

from diffdynamic.chemistry.profiles.state import (
    Counter,
    PROFILE_VERSION,
)


def build_class_scaffold_profile(
    reference_sdf: str | Path,
    native_ligand_sdf: str | Path,
    scaffold_smarts: str,
    *,
    class_name: str,
    reference_indices: Iterable[int],
    excluded_reference_indices: Iterable[int] = (),
    exploration_floor: float = 0.0,
    preserve_to_amine: bool = False,
    native_slot_by_query_index: Optional[dict[int, int]] = None,
) -> dict[str, Any]:
    """Build one trusted GSPT1 class profile with joint site budgets.

    The output stores only aggregate chemistry and component sizes per
    scaffold slot. It never stores an outside graph, coordinate, atom order,
    or SMILES.
    """
    from diffdynamic.chemistry.profiles.amine import (
        _native_scaffold_match,
        find_first_amine_path,
    )
    from diffdynamic.chemistry.profiles.fragments import (
        _build_class_fragment_patterns,
        _reference_component_profile,
    )
    from diffdynamic.chemistry.profiles.sdf import _first_molecule, iter_sdf_molecules

    native = _first_molecule(native_ligand_sdf)
    if native is None:
        raise ValueError(f"cannot parse native ligand: {native_ligand_sdf}")
    native_match = _native_scaffold_match(native, scaffold_smarts)
    sorted_native = sorted(native_match)
    if native_slot_by_query_index is None:
        native_slot_by_query_index = {
            query_idx: sorted_native.index(atom_idx)
            for query_idx, atom_idx in enumerate(native_match)
        }
    else:
        native_slot_by_query_index = {
            int(query_index): int(slot)
            for query_index, slot in native_slot_by_query_index.items()
        }
    native_amine_path = (
        find_first_amine_path(native, native_match) if preserve_to_amine else None
    )
    if preserve_to_amine and native_amine_path is None:
        raise ValueError(f"{class_name}: native ligand has no path to a non-aromatic N")
    preserved_atom_type_counts: Counter[str] = Counter()
    if native_amine_path is not None:
        for atom_idx in native_amine_path["path_atom_indices"]:
            atom = native.GetAtomWithIdx(int(atom_idx))
            preserved_atom_type_counts[
                f"{atom.GetSymbol()}|{int(atom.GetIsAromatic())}"
            ] += 1
    preserved_atom_count = len(
        native_amine_path["path_atom_indices"] if native_amine_path is not None else []
    )
    selected = {int(index) for index in reference_indices}
    excluded = {int(index) for index in excluded_reference_indices}
    selected -= excluded

    pattern_counts: Counter[tuple[tuple[int, int], ...]] = Counter()
    type_pattern_counts: Counter[tuple[int, tuple[tuple[str, int], ...]]] = Counter()
    element_counts: Counter[str] = Counter()
    aromatic_counts: Counter[str] = Counter()
    unsupported_counts: Counter[str] = Counter()
    matched_indices: list[int] = []
    unmatched_indices: list[int] = []
    total_records = 0
    for ref_idx, mol in enumerate(iter_sdf_molecules(reference_sdf)):
        total_records += 1
        if ref_idx not in selected:
            continue
        record = _reference_component_profile(
            mol,
            scaffold_smarts,
            native_slot_by_query_index,
            preserve_to_amine=preserve_to_amine,
            preserved_atom_type_counts=preserved_atom_type_counts,
            preserved_atom_count=preserved_atom_count,
        )
        if record is None:
            unmatched_indices.append(ref_idx)
            continue
        allocation = tuple(sorted(record["allocation"].items()))
        if not allocation or sum(count for _, count in allocation) != record["n_extra"]:
            unmatched_indices.append(ref_idx)
            continue
        matched_indices.append(ref_idx)
        pattern_counts[allocation] += 1
        element_counts.update(record["element_counts"])
        aromatic_counts.update(record["aromatic_counts"])
        unsupported_counts.update(record["unsupported_counts"])
        if (
            not record["unsupported_counts"]
            and sum(record["aromatic_counts"].values()) == record["n_extra"]
        ):
            type_key = (
                int(record["n_extra"]),
                tuple(sorted(record["aromatic_counts"].items())),
            )
            type_pattern_counts[type_key] += 1

    if not pattern_counts:
        raise ValueError(f"no trusted references matched class {class_name}")

    exit_counts: Counter[int] = Counter()
    size_counts: Counter[int] = Counter()
    allocation_patterns = []
    for allocation, weight in sorted(
        pattern_counts.items(),
        key=lambda item: (sum(count for _, count in item[0]), item[0]),
    ):
        total = sum(count for _, count in allocation)
        size_counts[total] += weight
        for slot, _ in allocation:
            exit_counts[slot] += weight
        allocation_patterns.append(
            {
                "n_extra": int(total),
                "site_counts": {str(slot): int(count) for slot, count in allocation},
                "weight": int(weight),
            }
        )

    extra_type_patterns = [
        {
            "n_extra": int(n_extra),
            "class_counts": {str(key): int(count) for key, count in class_counts},
            "weight": int(weight),
        }
        for (n_extra, class_counts), weight in sorted(
            type_pattern_counts.items(), key=lambda item: item[0]
        )
    ]
    fragment_patterns = []
    if not preserve_to_amine:
        fragment_patterns = _build_class_fragment_patterns(
            reference_sdf,
            scaffold_smarts,
            native_match,
            reference_indices,
            excluded_reference_indices,
            native_slot_by_query_index=native_slot_by_query_index,
        )

    native_scaffold_indices = list(sorted_native)
    if native_amine_path is not None:
        native_scaffold_indices.extend(
            int(index) for index in native_amine_path["path_atom_indices"]
        )

    weights = {
        str(slot): float(count) + float(exploration_floor)
        for slot, count in sorted(exit_counts.items())
    }
    return {
        "profile_version": PROFILE_VERSION,
        "profile_kind": "gspt1_class",
        "class_name": str(class_name),
        "n_reference_records": total_records,
        "included_reference_indices": sorted(selected),
        "excluded_reference_indices": sorted(excluded),
        "matched_reference_indices": matched_indices,
        "unmatched_reference_indices": unmatched_indices,
        "matched_reference_records": len(matched_indices),
        "n_scaffold": len(native_scaffold_indices),
        "native_scaffold_atom_indices_sorted": native_scaffold_indices,
        "preserve_to_amine": bool(preserve_to_amine),
        "preserved_amine_path_length": int(preserved_atom_count),
        "preserved_amine_atom_type_counts": dict(preserved_atom_type_counts),
        "exit_site_counts": {
            str(slot): int(count) for slot, count in sorted(exit_counts.items())
        },
        "exit_site_weights": weights,
        "n_extra_values": [int(value) for value in sorted(size_counts)],
        "n_extra_weights": [int(size_counts[value]) for value in sorted(size_counts)],
        "allocation_patterns": allocation_patterns,
        "extra_type_patterns": extra_type_patterns,
        "fragment_patterns": fragment_patterns,
        "reference_extra_element_counts": dict(element_counts),
        "reference_extra_aromatic_element_counts": dict(aromatic_counts),
        "unsupported_extra_element_counts": dict(unsupported_counts),
        "exploration_floor": float(exploration_floor),
    }


def build_scaffold_profile(
    reference_sdf: str | Path,
    native_ligand_sdf: str | Path,
    scaffold_smarts: str,
    exploration_floor: float = 1.0,
) -> dict[str, Any]:
    """Build an auditable, target-fragment-free scaffold prior."""
    from diffdynamic.chemistry.profiles.amine import _native_scaffold_match
    from diffdynamic.chemistry.profiles.fragments import (
        _reference_exit_counts,
        _reference_extra_element_counts,
    )
    from diffdynamic.chemistry.profiles.sdf import _first_molecule, iter_sdf_molecules

    native = _first_molecule(native_ligand_sdf)
    if native is None:
        raise ValueError(f"cannot parse native ligand: {native_ligand_sdf}")
    native_match = _native_scaffold_match(native, scaffold_smarts)
    sorted_native = sorted(native_match)
    native_slot_by_query_index = {
        query_idx: sorted_native.index(atom_idx)
        for query_idx, atom_idx in enumerate(native_match)
    }

    exit_counts: Counter[int] = Counter()
    size_counts: Counter[int] = Counter()
    element_counts: Counter[str] = Counter()
    aromatic_element_counts: Counter[str] = Counter()
    matched = 0
    unmatched: list[int] = []
    total = 0
    for ref_idx, mol in enumerate(iter_sdf_molecules(reference_sdf)):
        total += 1
        counts, n_extra = _reference_exit_counts(
            mol, scaffold_smarts, native_slot_by_query_index
        )
        if n_extra is None:
            unmatched.append(ref_idx)
            continue
        matched += 1
        exit_counts.update(counts)
        size_counts[n_extra] += 1
        elements, aromatic = _reference_extra_element_counts(
            mol,
            scaffold_smarts,
        )
        element_counts.update(elements)
        aromatic_element_counts.update(aromatic)

    all_slots = sorted(exit_counts)
    weights = {
        str(slot): float(exit_counts[slot]) + float(exploration_floor)
        for slot in all_slots
    }
    return {
        "profile_version": PROFILE_VERSION,
        "n_reference_records": total,
        "matched_reference_records": matched,
        "unmatched_reference_indices": unmatched,
        "n_scaffold": len(native_match),
        "native_scaffold_atom_indices_sorted": sorted_native,
        "exit_site_counts": {str(k): int(v) for k, v in exit_counts.items()},
        "exit_site_weights": weights,
        "n_extra_values": [int(k) for k in sorted(size_counts)],
        "n_extra_weights": [int(size_counts[k]) for k in sorted(size_counts)],
        "reference_extra_element_counts": dict(element_counts),
        "reference_extra_aromatic_element_counts": dict(aromatic_element_counts),
        "exploration_floor": float(exploration_floor),
    }
