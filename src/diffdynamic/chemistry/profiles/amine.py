"""Amine responsibilities extracted from scaffold_prior.py."""

from __future__ import annotations

from diffdynamic.chemistry.profiles.state import Any, Iterable, Optional

from diffdynamic.chemistry.profiles.state import (
    Chem,
    deque,
)


def _native_scaffold_match(mol: Chem.Mol, scaffold_smarts: str) -> tuple[int, ...]:
    pattern = Chem.MolFromSmarts(scaffold_smarts)
    if pattern is None:
        raise ValueError(f"invalid scaffold SMARTS: {scaffold_smarts}")
    matches = mol.GetSubstructMatches(pattern, uniquify=True)
    if not matches:
        raise ValueError("scaffold SMARTS did not match native ligand")
    return tuple(int(i) for i in matches[0])


def find_first_amine_path(
    mol: Chem.Mol,
    scaffold_indices: Iterable[int],
) -> Optional[dict[str, Any]]:
    """Find the shortest outside path from a scaffold to a non-aromatic N.

    The path contains only heavy atoms outside ``scaffold_indices`` and ends
    at the first non-aromatic nitrogen encountered from a scaffold boundary.
    It is used only to define the fixed prefix of the scaffold campaign; no
    path graph or coordinates are written into a generation profile.
    """
    scaffold_set = {int(index) for index in scaffold_indices}
    if mol is None or not scaffold_set:
        return None

    candidates: list[tuple[int, int, tuple[int, ...]]] = []
    for anchor_idx in sorted(scaffold_set):
        anchor = mol.GetAtomWithIdx(anchor_idx)
        for neighbour in sorted(
            anchor.GetNeighbors(), key=lambda atom: int(atom.GetIdx())
        ):
            start_idx = int(neighbour.GetIdx())
            if start_idx in scaffold_set or neighbour.GetAtomicNum() <= 1:
                continue
            queue = deque([(start_idx, (start_idx,))])
            seen = {start_idx}
            found = None
            while queue:
                atom_idx, path = queue.popleft()
                atom = mol.GetAtomWithIdx(atom_idx)
                if atom.GetSymbol() == "N" and not atom.GetIsAromatic():
                    found = path
                    break
                for next_atom in sorted(
                    atom.GetNeighbors(), key=lambda item: int(item.GetIdx())
                ):
                    next_idx = int(next_atom.GetIdx())
                    if (
                        next_idx in scaffold_set
                        or next_idx in seen
                        or next_atom.GetAtomicNum() <= 1
                    ):
                        continue
                    seen.add(next_idx)
                    queue.append((next_idx, path + (next_idx,)))
            if found is not None:
                candidates.append((len(found), anchor_idx, found))

    if not candidates:
        return None
    _, anchor_idx, path = min(candidates, key=lambda item: (item[0], item[1], item[2]))
    return {
        "anchor_atom_idx": int(anchor_idx),
        "path_atom_indices": [int(index) for index in path],
        "path_symbols": [mol.GetAtomWithIdx(int(index)).GetSymbol() for index in path],
    }


def extend_scaffold_to_first_amine(
    mol: Chem.Mol,
    scaffold_indices: Iterable[int],
) -> tuple[list[int], Optional[dict[str, Any]]]:
    """Append the deterministic amine-prefix atoms to scaffold indices."""
    base = list(dict.fromkeys(int(index) for index in scaffold_indices))
    info = find_first_amine_path(mol, base)
    if info is None:
        return base, None
    base_set = set(base)
    extension = [
        int(index) for index in info["path_atom_indices"] if int(index) not in base_set
    ]
    return base + extension, info
