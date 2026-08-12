"""Reference-derived priors for GSPT1 scaffold generation.

The profile intentionally contains only coarse information: scaffold-local
exit frequencies, extra-heavy-atom counts, and element-count multisets.  It
never stores a reference target-side fragment, atom order, bond, coordinate,
or SMILES.
"""

from __future__ import annotations

import json
from collections import Counter, deque
from pathlib import Path
from typing import Any, Iterable, Optional

from rdkit import Chem, RDLogger
from rdkit.Chem import BRICS

RDLogger.DisableLog("rdApp.*")

PROFILE_VERSION = 4
SUPPORTED_PROFILE_VERSIONS = {1, 2, 3, PROFILE_VERSION}
SUPPORTED_GENERATION_ELEMENTS = {
    "C", "N", "O", "F", "P", "S", "Cl",
}


def _normalise_text(path: Path) -> str:
    return path.read_bytes().decode("utf-8", errors="ignore").replace(
        "\r\n", "\n"
    ).replace("\r", "\n")


def iter_sdf_molecules(path: str | Path) -> Iterable[Chem.Mol]:
    """Yield molecules from both regular and malformed multi-molecule SDFs."""
    path = Path(path)
    raw_text = _normalise_text(path)
    is_multi_block = raw_text.count("$$$$") > 1
    supplied = []
    if not is_multi_block:
        try:
            supplier = Chem.SDMolSupplier(
                str(path), removeHs=False, sanitize=False
            )
            supplied = [mol for mol in supplier if mol is not None]
        except Exception:
            supplied = []
    if supplied:
        for mol in supplied:
            try:
                Chem.SanitizeMol(mol)
            except Exception:
                try:
                    Chem.SanitizeMol(
                        mol,
                        sanitizeOps=(
                            Chem.SanitizeFlags.SANITIZE_ALL
                            ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES
                        ),
                    )
                except Exception:
                    pass
            yield mol
        return

    blocks = [block.strip() for block in raw_text.split("$$$$")]
    for block in blocks:
        if not block:
            continue
        mol_block = block
        if "M  END" in mol_block:
            mol_block = mol_block.split("M  END", 1)[0] + "M  END\n"
        mol = None
        for sanitize in (True, False):
            try:
                mol = Chem.MolFromMolBlock(
                    mol_block, sanitize=sanitize, removeHs=False
                )
            except Exception:
                mol = None
            if mol is None:
                continue
            if not sanitize:
                try:
                    Chem.SanitizeMol(mol)
                except Exception:
                    try:
                        Chem.SanitizeMol(
                            mol,
                            sanitizeOps=(
                                Chem.SanitizeFlags.SANITIZE_ALL
                                ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES
                            ),
                        )
                    except Exception:
                        pass
            break
        if mol is not None:
            yield mol


def _first_molecule(path: str | Path) -> Optional[Chem.Mol]:
    return next(iter_sdf_molecules(path), None)


def _native_scaffold_match(
    mol: Chem.Mol, scaffold_smarts: str
) -> tuple[int, ...]:
    pattern = Chem.MolFromSmarts(scaffold_smarts)
    if pattern is None:
        raise ValueError(f"invalid scaffold SMARTS: {scaffold_smarts}")
    matches = mol.GetSubstructMatches(pattern, uniquify=True)
    if not matches:
        raise ValueError("scaffold SMARTS did not match native ligand")
    return tuple(int(i) for i in matches[0])


def _reference_exit_counts(
    mol: Chem.Mol,
    scaffold_smarts: str,
    native_slot_by_query_index: dict[int, int],
) -> tuple[Counter[int], Optional[int]]:
    pattern = Chem.MolFromSmarts(scaffold_smarts)
    if pattern is None:
        return Counter(), None
    matches = mol.GetSubstructMatches(pattern, uniquify=True)
    if not matches:
        return Counter(), None
    match = tuple(int(i) for i in matches[0])
    scaffold_set = set(match)
    exits: Counter[int] = Counter()
    for atom_idx in range(mol.GetNumAtoms()):
        if atom_idx in scaffold_set:
            continue
        atom = mol.GetAtomWithIdx(atom_idx)
        if atom.GetAtomicNum() <= 1:
            continue
        for neighbour in atom.GetNeighbors():
            try:
                query_idx = match.index(int(neighbour.GetIdx()))
            except ValueError:
                continue
            native_slot = native_slot_by_query_index.get(query_idx)
            if native_slot is not None:
                exits[native_slot] += 1
    n_extra = mol.GetNumHeavyAtoms() - len(match)
    return exits, max(int(n_extra), 0)


def _reference_extra_element_counts(
    mol: Chem.Mol,
    scaffold_smarts: str,
) -> tuple[Counter[str], Counter[str]]:
    """Return aggregate target-side element/aromatic counts only.

    The result is deliberately not keyed by atom index or connected
    component.  It is suitable for a weak class-marginal initialization
    ablation, but cannot reconstruct a target-side graph.
    """
    pattern = Chem.MolFromSmarts(scaffold_smarts)
    if pattern is None:
        return Counter(), Counter()
    matches = mol.GetSubstructMatches(pattern, uniquify=True)
    if not matches:
        return Counter(), Counter()
    scaffold_set = set(int(i) for i in matches[0])
    elements: Counter[str] = Counter()
    aromatic: Counter[str] = Counter()
    for atom_idx, atom in enumerate(mol.GetAtoms()):
        if atom_idx in scaffold_set or atom.GetAtomicNum() <= 1:
            continue
        symbol = atom.GetSymbol()
        elements[symbol] += 1
        aromatic[f'{symbol}|{int(atom.GetIsAromatic())}'] += 1
    return elements, aromatic


def _reference_component_profile(
    mol: Chem.Mol,
    scaffold_smarts: str,
    native_slot_by_query_index: dict[int, int],
) -> Optional[dict[str, Any]]:
    """Summarize non-scaffold components without retaining their graphs."""
    pattern = Chem.MolFromSmarts(scaffold_smarts)
    if pattern is None:
        return None
    matches = mol.GetSubstructMatches(pattern, uniquify=True)
    if not matches:
        return None
    match = tuple(int(i) for i in matches[0])
    scaffold_set = set(match)
    extra_indices = [
        atom.GetIdx()
        for atom in mol.GetAtoms()
        if atom.GetIdx() not in scaffold_set and atom.GetAtomicNum() > 1
    ]
    extra_set = set(extra_indices)
    components: list[list[int]] = []
    seen: set[int] = set()
    for start in extra_indices:
        if start in seen:
            continue
        component: list[int] = []
        queue = deque([start])
        seen.add(start)
        while queue:
            atom_idx = queue.popleft()
            component.append(atom_idx)
            for neighbour in mol.GetAtomWithIdx(atom_idx).GetNeighbors():
                neighbour_idx = int(neighbour.GetIdx())
                if neighbour_idx in extra_set and neighbour_idx not in seen:
                    seen.add(neighbour_idx)
                    queue.append(neighbour_idx)
        components.append(component)

    allocation: Counter[int] = Counter()
    element_counts: Counter[str] = Counter()
    aromatic_counts: Counter[str] = Counter()
    unsupported_counts: Counter[str] = Counter()
    for component in components:
        anchor_slots: set[int] = set()
        for atom_idx in component:
            atom = mol.GetAtomWithIdx(atom_idx)
            symbol = atom.GetSymbol()
            if symbol in SUPPORTED_GENERATION_ELEMENTS:
                element_counts[symbol] += 1
                aromatic_counts[
                    f"{symbol}|{int(atom.GetIsAromatic())}"
                ] += 1
            else:
                unsupported_counts[symbol] += 1
            for neighbour in atom.GetNeighbors():
                neighbour_idx = int(neighbour.GetIdx())
                if neighbour_idx not in scaffold_set:
                    continue
                query_idx = match.index(neighbour_idx)
                slot = native_slot_by_query_index.get(query_idx)
                if slot is not None:
                    anchor_slots.add(int(slot))
        if not anchor_slots:
            continue
        # Reference compounds in this task attach each outside component at
        # one scaffold atom. Reject ambiguous bridged components explicitly.
        if len(anchor_slots) != 1:
            return None
        allocation[next(iter(anchor_slots))] += len(component)

    return {
        "n_extra": len(extra_indices),
        "allocation": allocation,
        "element_counts": element_counts,
        "aromatic_counts": aromatic_counts,
        "unsupported_counts": unsupported_counts,
    }


def _fragment_signature(
    mol: Chem.Mol,
    atom_indices: Iterable[int],
    ring_systems: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Summarize a component as generic ring/functional-group templates.

    This intentionally discards atom indices, bonds, coordinates, and atom
    order.  The output is only a multiset of motif kinds and element classes;
    runtime placement expands it into randomized generic geometry.
    """
    component = set(int(i) for i in atom_indices)
    used: set[int] = set()
    fragments: list[dict[str, Any]] = []

    for ring in ring_systems:
        ring_atoms = set(int(i) for i in ring["atom_indices"])
        used.update(ring_atoms)
        fragments.append({
            "kind": str(ring["kind"]),
            "ring_sizes": [int(v) for v in ring["ring_sizes"]],
            "atom_counts": {
                str(key): int(value)
                for key, value in sorted(ring["atom_counts"].items())
            },
        })

    residual = component - used
    if residual:
        residual_components: list[list[int]] = []
        seen: set[int] = set()
        for start in sorted(residual):
            if start in seen:
                continue
            queue = deque([start])
            seen.add(start)
            current: list[int] = []
            while queue:
                atom_idx = queue.popleft()
                current.append(atom_idx)
                for neighbour in mol.GetAtomWithIdx(atom_idx).GetNeighbors():
                    neighbour_idx = int(neighbour.GetIdx())
                    if neighbour_idx in residual and neighbour_idx not in seen:
                        seen.add(neighbour_idx)
                        queue.append(neighbour_idx)
            residual_components.append(current)

        for current in residual_components:
            current_set = set(current)
            consumed: set[int] = set()
            for atom_idx in sorted(current):
                if atom_idx in consumed:
                    continue
                atom = mol.GetAtomWithIdx(atom_idx)
                if atom.GetSymbol() != "C":
                    continue
                oxygen_idx = None
                for neighbour in atom.GetNeighbors():
                    neighbour_idx = int(neighbour.GetIdx())
                    if neighbour_idx not in current_set:
                        continue
                    if neighbour.GetSymbol() != "O":
                        continue
                    bond = mol.GetBondBetweenAtoms(atom_idx, neighbour_idx)
                    if bond is not None and bond.GetBondType() == Chem.BondType.DOUBLE:
                        oxygen_idx = neighbour_idx
                        break
                if oxygen_idx is None:
                    continue
                consumed.update({atom_idx, oxygen_idx})
                fragments.append({
                    "kind": "carbonyl",
                    "ring_sizes": [],
                    "atom_counts": {"C|0": 1, "O|0": 1},
                })

            remaining = current_set - consumed
            if not remaining:
                continue
            counts = Counter()
            for atom_idx in remaining:
                atom = mol.GetAtomWithIdx(atom_idx)
                counts[f"{atom.GetSymbol()}|{int(atom.GetIsAromatic())}"] += 1
            kind = "hetero_atom" if len(remaining) == 1 else "chain"
            fragments.append({
                "kind": kind,
                "ring_sizes": [],
                "atom_counts": {
                    str(key): int(value)
                    for key, value in sorted(counts.items())
                },
            })

    # Stable ordering makes the profile auditable and avoids reference atom
    # order becoming an implicit condition.
    fragments.sort(key=lambda item: (
        str(item["kind"]),
        tuple(int(v) for v in item.get("ring_sizes", [])),
        tuple(sorted(item["atom_counts"].items())),
    ))
    return fragments


def _component_ring_systems(
    mol: Chem.Mol,
    component: Iterable[int],
) -> list[dict[str, Any]]:
    """Return ring systems as composition-only generic motif records."""
    component_set = set(int(i) for i in component)
    rings = [
        set(int(i) for i in ring)
        for ring in mol.GetRingInfo().AtomRings()
        if set(int(i) for i in ring) <= component_set
    ]
    systems: list[set[int]] = []
    system_ring_lists: list[list[set[int]]] = []
    for ring in rings:
        matches = [
            index for index, system in enumerate(systems)
            if system.intersection(ring)
        ]
        if not matches:
            systems.append(set(ring))
            system_ring_lists.append([ring])
            continue
        first = matches[0]
        systems[first].update(ring)
        system_ring_lists[first].append(ring)
        for index in reversed(matches[1:]):
            systems[first].update(systems[index])
            system_ring_lists[first].extend(system_ring_lists[index])
            del systems[index]
            del system_ring_lists[index]

    output = []
    for atom_set, ring_list in zip(systems, system_ring_lists):
        counts = Counter()
        for atom_idx in atom_set:
            atom = mol.GetAtomWithIdx(atom_idx)
            counts[f"{atom.GetSymbol()}|{int(atom.GetIsAromatic())}"] += 1
        aromatic = all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in atom_set)
        if len(ring_list) > 1 and aromatic:
            kind = "fused_aromatic"
        elif len(ring_list) > 1:
            kind = "fused_ring"
        elif aromatic:
            kind = "aromatic_ring"
        else:
            kind = "aliphatic_ring"
        output.append({
            "kind": kind,
            "atom_indices": sorted(atom_set),
            "ring_sizes": sorted(len(ring) for ring in ring_list),
            "atom_counts": counts,
        })
    output.sort(key=lambda item: (
        str(item["kind"]),
        tuple(item["ring_sizes"]),
        tuple(item["atom_indices"]),
    ))
    return output


def _brics_cut_bonds(mol: Chem.Mol, allowed: set[int]) -> set[tuple[int, int]]:
    """Return BRICS linker bonds wholly inside one target-side component."""
    cuts: set[tuple[int, int]] = set()
    try:
        brics_bonds = BRICS.FindBRICSBonds(mol)
        for pair, _labels in brics_bonds:
            first, second = (int(pair[0]), int(pair[1]))
            if first in allowed and second in allowed:
                cuts.add(tuple(sorted((first, second))))
    except Exception:
        return set()
    return cuts


def _component_fragments(
    mol: Chem.Mol,
    component: Iterable[int],
) -> list[dict[str, Any]]:
    """Split a side component at generic BRICS linkers, then summarize it."""
    component_set = set(int(i) for i in component)
    cuts = _brics_cut_bonds(mol, component_set)
    components: list[list[int]] = []
    seen: set[int] = set()
    for start in sorted(component_set):
        if start in seen:
            continue
        queue = deque([start])
        seen.add(start)
        current: list[int] = []
        while queue:
            atom_idx = queue.popleft()
            current.append(atom_idx)
            atom = mol.GetAtomWithIdx(atom_idx)
            for neighbour in atom.GetNeighbors():
                neighbour_idx = int(neighbour.GetIdx())
                if neighbour_idx not in component_set or neighbour_idx in seen:
                    continue
                bond_key = tuple(sorted((atom_idx, neighbour_idx)))
                if bond_key in cuts:
                    continue
                seen.add(neighbour_idx)
                queue.append(neighbour_idx)
        components.append(current)

    fragments: list[dict[str, Any]] = []
    for current in components:
        rings = _component_ring_systems(mol, current)
        fragments.extend(_fragment_signature(mol, current, rings))
    return fragments


def _reference_fragment_pattern(
    mol: Chem.Mol,
    scaffold_smarts: str,
    native_slot_by_query_index: dict[int, int],
) -> Optional[dict[str, Any]]:
    """Build one target-side fragment pattern without retaining its graph."""
    pattern = Chem.MolFromSmarts(scaffold_smarts)
    if pattern is None:
        return None
    matches = mol.GetSubstructMatches(pattern, uniquify=True)
    if not matches:
        return None
    match = tuple(int(i) for i in matches[0])
    scaffold_set = set(match)
    extra = {
        int(atom.GetIdx())
        for atom in mol.GetAtoms()
        if atom.GetIdx() not in scaffold_set and atom.GetAtomicNum() > 1
    }
    components: list[list[int]] = []
    seen: set[int] = set()
    for start in sorted(extra):
        if start in seen:
            continue
        queue = deque([start])
        seen.add(start)
        current: list[int] = []
        while queue:
            atom_idx = queue.popleft()
            current.append(atom_idx)
            for neighbour in mol.GetAtomWithIdx(atom_idx).GetNeighbors():
                neighbour_idx = int(neighbour.GetIdx())
                if neighbour_idx in extra and neighbour_idx not in seen:
                    seen.add(neighbour_idx)
                    queue.append(neighbour_idx)
        components.append(current)

    site_fragments: dict[str, list[dict[str, Any]]] = {}
    for component in components:
        slots: set[int] = set()
        for atom_idx in component:
            for neighbour in mol.GetAtomWithIdx(atom_idx).GetNeighbors():
                neighbour_idx = int(neighbour.GetIdx())
                if neighbour_idx not in scaffold_set:
                    continue
                query_idx = match.index(neighbour_idx)
                slot = native_slot_by_query_index.get(query_idx)
                if slot is not None:
                    slots.add(int(slot))
        if len(slots) != 1:
            return None
        fragments = _component_fragments(mol, component)
        if not fragments:
            return None
        slot_key = str(next(iter(slots)))
        site_fragments.setdefault(slot_key, []).extend(fragments)

    if not site_fragments:
        return None
    for fragments in site_fragments.values():
        fragments.sort(key=lambda item: (
            str(item["kind"]),
            tuple(int(v) for v in item.get("ring_sizes", [])),
            tuple(sorted(item["atom_counts"].items())),
        ))
    return {
        "n_extra": len(extra),
        "site_fragments": site_fragments,
    }


def _fragment_pattern_key(pattern: dict[str, Any]) -> tuple:
    site_key = []
    for slot, fragments in sorted(
        (pattern.get("site_fragments") or {}).items(),
        key=lambda item: int(item[0]),
    ):
        fragment_key = []
        for fragment in fragments:
            fragment_key.append((
                str(fragment.get("kind", "point")),
                tuple(int(v) for v in fragment.get("ring_sizes", [])),
                tuple(sorted(
                    (str(key), int(value))
                    for key, value in (fragment.get("atom_counts") or {}).items()
                )),
            ))
        site_key.append((str(slot), tuple(fragment_key)))
    return int(pattern.get("n_extra", 0)), tuple(site_key)


def _build_class_fragment_patterns(
    reference_sdf: str | Path,
    scaffold_smarts: str,
    native_match: tuple[int, ...],
    reference_indices: Iterable[int],
    excluded_reference_indices: Iterable[int],
) -> list[dict[str, Any]]:
    """Aggregate generic fragment layouts for one class profile."""
    native_sorted = sorted(native_match)
    native_slot_by_query_index = {
        query_idx: native_sorted.index(atom_idx)
        for query_idx, atom_idx in enumerate(native_match)
    }
    selected = {int(index) for index in reference_indices}
    excluded = {int(index) for index in excluded_reference_indices}
    counts: Counter[tuple] = Counter()
    raw_by_key: dict[tuple, dict[str, Any]] = {}
    for ref_idx, mol in enumerate(iter_sdf_molecules(reference_sdf)):
        if ref_idx not in selected or ref_idx in excluded:
            continue
        candidate = _reference_fragment_pattern(
            mol, scaffold_smarts, native_slot_by_query_index
        )
        if candidate is None:
            continue
        unsupported = False
        for fragments in candidate["site_fragments"].values():
            for fragment in fragments:
                for key in fragment["atom_counts"]:
                    symbol = str(key).rsplit("|", 1)[0]
                    if symbol not in SUPPORTED_GENERATION_ELEMENTS:
                        unsupported = True
        if unsupported:
            continue
        key = _fragment_pattern_key(candidate)
        counts[key] += 1
        raw_by_key[key] = candidate

    output = []
    for key, weight in sorted(counts.items(), key=lambda item: item[0]):
        candidate = json.loads(json.dumps(raw_by_key[key]))
        candidate["weight"] = int(weight)
        output.append(candidate)
    return output


def build_class_scaffold_profile(
    reference_sdf: str | Path,
    native_ligand_sdf: str | Path,
    scaffold_smarts: str,
    *,
    class_name: str,
    reference_indices: Iterable[int],
    excluded_reference_indices: Iterable[int] = (),
    exploration_floor: float = 0.0,
) -> dict[str, Any]:
    """Build one trusted GSPT1 class profile with joint site budgets.

    The output stores only aggregate chemistry and component sizes per
    scaffold slot. It never stores an outside graph, coordinate, atom order,
    or SMILES.
    """
    native = _first_molecule(native_ligand_sdf)
    if native is None:
        raise ValueError(f"cannot parse native ligand: {native_ligand_sdf}")
    native_match = _native_scaffold_match(native, scaffold_smarts)
    sorted_native = sorted(native_match)
    native_slot_by_query_index = {
        query_idx: sorted_native.index(atom_idx)
        for query_idx, atom_idx in enumerate(native_match)
    }
    selected = {int(index) for index in reference_indices}
    excluded = {int(index) for index in excluded_reference_indices}
    selected -= excluded

    pattern_counts: Counter[tuple[tuple[int, int], ...]] = Counter()
    type_pattern_counts: Counter[
        tuple[int, tuple[tuple[str, int], ...]]
    ] = Counter()
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
            mol, scaffold_smarts, native_slot_by_query_index
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
        allocation_patterns.append({
            "n_extra": int(total),
            "site_counts": {
                str(slot): int(count) for slot, count in allocation
            },
            "weight": int(weight),
        })

    extra_type_patterns = [
        {
            "n_extra": int(n_extra),
            "class_counts": {
                str(key): int(count) for key, count in class_counts
            },
            "weight": int(weight),
        }
        for (n_extra, class_counts), weight in sorted(
            type_pattern_counts.items(), key=lambda item: item[0]
        )
    ]
    fragment_patterns = _build_class_fragment_patterns(
        reference_sdf,
        scaffold_smarts,
        native_match,
        reference_indices,
        excluded_reference_indices,
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
        "n_scaffold": len(native_match),
        "native_scaffold_atom_indices_sorted": sorted_native,
        "exit_site_counts": {
            str(slot): int(count) for slot, count in sorted(exit_counts.items())
        },
        "exit_site_weights": weights,
        "n_extra_values": [int(value) for value in sorted(size_counts)],
        "n_extra_weights": [
            int(size_counts[value]) for value in sorted(size_counts)
        ],
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
            mol, scaffold_smarts,
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
        "reference_extra_aromatic_element_counts": dict(
            aromatic_element_counts
        ),
        "exploration_floor": float(exploration_floor),
    }


def load_scaffold_profile(path: str | Path) -> dict[str, Any]:
    """Load and validate only the public coarse-prior fields."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    profile_version = int(payload.get("profile_version", -1))
    if profile_version not in SUPPORTED_PROFILE_VERSIONS:
        raise ValueError(f"unsupported scaffold profile version: {path}")
    n_scaffold = int(payload.get("n_scaffold", 0))
    if n_scaffold <= 0:
        raise ValueError(f"invalid scaffold size in profile: {path}")
    weights = payload.get("exit_site_weights") or {}
    clean_weights = {
        str(int(slot)): max(float(weight), 0.0)
        for slot, weight in weights.items()
        if 0 <= int(slot) < n_scaffold
    }
    values = [int(v) for v in payload.get("n_extra_values", [])]
    size_weights = [float(v) for v in payload.get("n_extra_weights", [])]
    if len(values) != len(size_weights) or not values:
        values, size_weights = [], []
    element_counts = {
        str(symbol): max(float(count), 0.0)
        for symbol, count in (payload.get("reference_extra_element_counts") or {}).items()
    }
    aromatic_counts = {
        str(key): max(float(count), 0.0)
        for key, count in (
            payload.get("reference_extra_aromatic_element_counts") or {}
        ).items()
    }
    allocation_patterns = []
    for raw_pattern in payload.get("allocation_patterns") or []:
        site_counts = {
            str(int(slot)): int(count)
            for slot, count in (raw_pattern.get("site_counts") or {}).items()
            if 0 <= int(slot) < n_scaffold and int(count) > 0
        }
        n_extra = int(raw_pattern.get("n_extra", sum(site_counts.values())))
        weight = max(float(raw_pattern.get("weight", 1.0)), 0.0)
        if site_counts and sum(site_counts.values()) == n_extra and weight > 0:
            allocation_patterns.append({
                "n_extra": n_extra,
                "site_counts": site_counts,
                "weight": weight,
            })
    extra_type_patterns = []
    for raw_pattern in payload.get("extra_type_patterns") or []:
        class_counts = {}
        for raw_key, raw_count in (
            raw_pattern.get("class_counts") or {}
        ).items():
            try:
                symbol, aromatic_text = str(raw_key).rsplit("|", 1)
                aromatic = int(aromatic_text)
                count = int(raw_count)
            except (TypeError, ValueError):
                continue
            if (
                symbol not in SUPPORTED_GENERATION_ELEMENTS
                or aromatic not in (0, 1)
                or count <= 0
            ):
                continue
            class_counts[f"{symbol}|{aromatic}"] = count
        n_extra = int(
            raw_pattern.get("n_extra", sum(class_counts.values()))
        )
        weight = max(float(raw_pattern.get("weight", 1.0)), 0.0)
        if (
            class_counts
            and sum(class_counts.values()) == n_extra
            and weight > 0
        ):
            extra_type_patterns.append({
                "n_extra": n_extra,
                "class_counts": class_counts,
                "weight": weight,
            })
    fragment_patterns = []
    for raw_pattern in payload.get("fragment_patterns") or []:
        try:
            n_extra = int(raw_pattern.get("n_extra", 0))
            weight = max(float(raw_pattern.get("weight", 0.0)), 0.0)
        except (TypeError, ValueError):
            continue
        if n_extra <= 0 or weight <= 0.0:
            continue
        site_fragments = {}
        valid = True
        for raw_slot, raw_fragments in (
            raw_pattern.get("site_fragments") or {}
        ).items():
            try:
                slot = str(int(raw_slot))
            except (TypeError, ValueError):
                valid = False
                break
            if not (0 <= int(slot) < n_scaffold):
                valid = False
                break
            clean_fragments = []
            for raw_fragment in raw_fragments or []:
                kind = str(raw_fragment.get("kind", "point"))
                ring_sizes = []
                for raw_size in raw_fragment.get("ring_sizes") or []:
                    try:
                        ring_size = int(raw_size)
                    except (TypeError, ValueError):
                        valid = False
                        break
                    if ring_size < 3 or ring_size > 8:
                        valid = False
                        break
                    ring_sizes.append(ring_size)
                if not valid:
                    break
                atom_counts = {}
                for raw_key, raw_count in (
                    raw_fragment.get("atom_counts") or {}
                ).items():
                    try:
                        symbol, aromatic_text = str(raw_key).rsplit("|", 1)
                        aromatic = int(aromatic_text)
                        count = int(raw_count)
                    except (TypeError, ValueError):
                        valid = False
                        break
                    if (
                        symbol not in SUPPORTED_GENERATION_ELEMENTS
                        or aromatic not in (0, 1)
                        or count <= 0
                    ):
                        valid = False
                        break
                    atom_counts[f"{symbol}|{aromatic}"] = count
                if not valid or not atom_counts:
                    valid = False
                    break
                clean_fragments.append({
                    "kind": kind,
                    "ring_sizes": ring_sizes,
                    "atom_counts": atom_counts,
                })
            if not valid or not clean_fragments:
                valid = False
                break
            site_fragments[slot] = clean_fragments
        if not valid or not site_fragments:
            continue
        if sum(
            sum(int(count) for count in fragment["atom_counts"].values())
            for fragments in site_fragments.values()
            for fragment in fragments
        ) != n_extra:
            continue
        fragment_patterns.append({
            "n_extra": n_extra,
            "site_fragments": site_fragments,
            "weight": weight,
        })
    return {
        "profile_version": profile_version,
        "profile_kind": str(payload.get("profile_kind", "coarse")),
        "class_name": str(payload.get("class_name", "")),
        "n_scaffold": n_scaffold,
        "exit_site_weights": clean_weights,
        "n_extra_values": values,
        "n_extra_weights": size_weights,
        "reference_extra_element_counts": element_counts,
        "reference_extra_aromatic_element_counts": aromatic_counts,
        "unsupported_extra_element_counts": {
            str(symbol): max(float(count), 0.0)
            for symbol, count in (
                payload.get("unsupported_extra_element_counts") or {}
            ).items()
        },
        "allocation_patterns": allocation_patterns,
        "extra_type_patterns": extra_type_patterns,
        "fragment_patterns": fragment_patterns,
        "included_reference_indices": [
            int(index) for index in payload.get("included_reference_indices", [])
        ],
        "excluded_reference_indices": [
            int(index) for index in payload.get("excluded_reference_indices", [])
        ],
        "matched_reference_records": int(
            payload.get("matched_reference_records", 0)
        ),
        "unmatched_reference_indices": list(
            payload.get("unmatched_reference_indices", [])
        ),
    }


def write_scaffold_profile(profile: dict[str, Any], output: str | Path) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(profile, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
