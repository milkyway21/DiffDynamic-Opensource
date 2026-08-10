"""Reference-derived priors for GSPT1 scaffold generation.

The profile intentionally contains only coarse information: scaffold-local
exit frequencies, extra-heavy-atom counts, and optional aggregate element
counts.  It never stores a reference target-side fragment, atom order, bond,
coordinate, or SMILES.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Optional

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

PROFILE_VERSION = 1


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
    if int(payload.get("profile_version", -1)) != PROFILE_VERSION:
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
    return {
        "profile_version": PROFILE_VERSION,
        "n_scaffold": n_scaffold,
        "exit_site_weights": clean_weights,
        "n_extra_values": values,
        "n_extra_weights": size_weights,
        "reference_extra_element_counts": element_counts,
        "reference_extra_aromatic_element_counts": aromatic_counts,
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
