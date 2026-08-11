"""Reference-derived priors for GSPT1 scaffold generation.

The profile intentionally contains only coarse information: scaffold-local
exit frequencies, extra-heavy-atom counts, and optional aggregate element
counts.  It never stores a reference target-side fragment, atom order, bond,
coordinate, or SMILES.
"""

from __future__ import annotations

import json
from collections import Counter, deque
from pathlib import Path
from typing import Any, Iterable, Optional

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

PROFILE_VERSION = 2
SUPPORTED_PROFILE_VERSIONS = {1, PROFILE_VERSION}
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
