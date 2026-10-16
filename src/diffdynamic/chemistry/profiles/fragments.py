"""Fragments responsibilities extracted from scaffold_prior.py."""

from __future__ import annotations

from diffdynamic.chemistry.profiles.state import Iterable, Optional, Path

from diffdynamic.chemistry.profiles.state import (
    Any,
    BRICS,
    Chem,
    Counter,
    SUPPORTED_GENERATION_ELEMENTS,
    deque,
    json,
)


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
        aromatic[f"{symbol}|{int(atom.GetIsAromatic())}"] += 1
    return elements, aromatic


def _reference_component_profile(
    mol: Chem.Mol,
    scaffold_smarts: str,
    native_slot_by_query_index: dict[int, int],
    *,
    preserve_to_amine: bool = False,
    preserved_atom_type_counts: Optional[Counter[str]] = None,
    preserved_atom_count: int = 0,
) -> Optional[dict[str, Any]]:
    """Summarize non-scaffold components without retaining their graphs."""
    from diffdynamic.chemistry.profiles.amine import find_first_amine_path

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
                aromatic_counts[f"{symbol}|{int(atom.GetIsAromatic())}"] += 1
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

    preserved_info = None
    if preserve_to_amine:
        preserved_info = find_first_amine_path(mol, match)
        if preserved_info is None:
            return None
        path_indices = set(preserved_info["path_atom_indices"])
        if not path_indices.issubset(extra_set):
            return None
        anchor_idx = int(preserved_info["anchor_atom_idx"])
        try:
            query_idx = match.index(anchor_idx)
        except ValueError:
            return None
        anchor_slot = native_slot_by_query_index.get(query_idx)
        if anchor_slot is None:
            return None
        preserve_count = int(preserved_atom_count)
        if preserve_count <= 0:
            preserve_count = len(path_indices)
        if allocation.get(int(anchor_slot), 0) < preserve_count:
            return None
        allocation[int(anchor_slot)] -= preserve_count
        if allocation[int(anchor_slot)] <= 0:
            del allocation[int(anchor_slot)]

        fixed_types = preserved_atom_type_counts or Counter()
        for raw_key, count in fixed_types.items():
            symbol, aromatic_text = str(raw_key).rsplit("|", 1)
            count = int(count)
            if symbol in SUPPORTED_GENERATION_ELEMENTS:
                aromatic_key = f"{symbol}|{int(aromatic_text)}"
                if (
                    element_counts.get(symbol, 0) < count
                    or aromatic_counts.get(aromatic_key, 0) < count
                ):
                    return None
                element_counts[symbol] -= count
                aromatic_counts[aromatic_key] -= count
                if element_counts[symbol] <= 0:
                    del element_counts[symbol]
                if aromatic_counts[aromatic_key] <= 0:
                    del aromatic_counts[aromatic_key]
            else:
                if unsupported_counts.get(symbol, 0) < count:
                    return None
                unsupported_counts[symbol] -= count
                if unsupported_counts[symbol] <= 0:
                    del unsupported_counts[symbol]

    n_extra = len(extra_indices) - (
        int(preserved_atom_count) if preserve_to_amine else 0
    )
    if n_extra < 0:
        return None
    return {
        "n_extra": n_extra,
        "allocation": allocation,
        "element_counts": element_counts,
        "aromatic_counts": aromatic_counts,
        "unsupported_counts": unsupported_counts,
        "preserved_amine_path_length": (
            len(preserved_info["path_atom_indices"])
            if preserved_info is not None
            else 0
        ),
        "preserved_amine_anchor_slot": (
            native_slot_by_query_index.get(
                match.index(int(preserved_info["anchor_atom_idx"]))
            )
            if preserved_info is not None
            else None
        ),
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
        fragments.append(
            {
                "kind": str(ring["kind"]),
                "ring_sizes": [int(v) for v in ring["ring_sizes"]],
                "atom_counts": {
                    str(key): int(value)
                    for key, value in sorted(ring["atom_counts"].items())
                },
            }
        )

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
                fragments.append(
                    {
                        "kind": "carbonyl",
                        "ring_sizes": [],
                        "atom_counts": {"C|0": 1, "O|0": 1},
                    }
                )

            remaining = current_set - consumed
            if not remaining:
                continue
            counts = Counter()
            for atom_idx in remaining:
                atom = mol.GetAtomWithIdx(atom_idx)
                counts[f"{atom.GetSymbol()}|{int(atom.GetIsAromatic())}"] += 1
            kind = "hetero_atom" if len(remaining) == 1 else "chain"
            fragments.append(
                {
                    "kind": kind,
                    "ring_sizes": [],
                    "atom_counts": {
                        str(key): int(value) for key, value in sorted(counts.items())
                    },
                }
            )

    # Stable ordering makes the profile auditable and avoids reference atom
    # order becoming an implicit condition.
    fragments.sort(
        key=lambda item: (
            str(item["kind"]),
            tuple(int(v) for v in item.get("ring_sizes", [])),
            tuple(sorted(item["atom_counts"].items())),
        )
    )
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
            index for index, system in enumerate(systems) if system.intersection(ring)
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
        output.append(
            {
                "kind": kind,
                "atom_indices": sorted(atom_set),
                "ring_sizes": sorted(len(ring) for ring in ring_list),
                "atom_counts": counts,
            }
        )
    output.sort(
        key=lambda item: (
            str(item["kind"]),
            tuple(item["ring_sizes"]),
            tuple(item["atom_indices"]),
        )
    )
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
        fragments.sort(
            key=lambda item: (
                str(item["kind"]),
                tuple(int(v) for v in item.get("ring_sizes", [])),
                tuple(sorted(item["atom_counts"].items())),
            )
        )
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
            fragment_key.append(
                (
                    str(fragment.get("kind", "point")),
                    tuple(int(v) for v in fragment.get("ring_sizes", [])),
                    tuple(
                        sorted(
                            (str(key), int(value))
                            for key, value in (
                                fragment.get("atom_counts") or {}
                            ).items()
                        )
                    ),
                )
            )
        site_key.append((str(slot), tuple(fragment_key)))
    return int(pattern.get("n_extra", 0)), tuple(site_key)


def _build_class_fragment_patterns(
    reference_sdf: str | Path,
    scaffold_smarts: str,
    native_match: tuple[int, ...],
    reference_indices: Iterable[int],
    excluded_reference_indices: Iterable[int],
    native_slot_by_query_index: Optional[dict[int, int]] = None,
) -> list[dict[str, Any]]:
    """Aggregate generic fragment layouts for one class profile."""
    from diffdynamic.chemistry.profiles.sdf import iter_sdf_molecules

    if native_slot_by_query_index is None:
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
