"""Sdf responsibilities extracted from scaffold_prior.py."""

from __future__ import annotations

from diffdynamic.chemistry.profiles.state import Iterable, Optional

from diffdynamic.chemistry.profiles.state import (
    Chem,
    Path,
    re,
)


def _normalise_text(path: Path) -> str:
    return (
        path.read_bytes()
        .decode("utf-8", errors="ignore")
        .replace("\r\n", "\n")
        .replace("\r", "\n")
    )


def _attach_sdf_data_fields(molecule: Chem.Mol, data_block: str) -> None:
    """Restore SD data fields when parsing a manually split record."""
    field_name = None
    values: list[str] = []

    def flush() -> None:
        nonlocal field_name, values
        if field_name is not None:
            molecule.SetProp(field_name, "\n".join(values))
        field_name = None
        values = []

    for line in data_block.splitlines():
        match = re.match(r"^\s*>\s*<([^>]+)>", line)
        if match:
            flush()
            field_name = match.group(1)
            continue
        if field_name is None:
            continue
        if line == "":
            flush()
        else:
            values.append(line)
    flush()


def iter_sdf_molecules(path: str | Path) -> Iterable[Chem.Mol]:
    """Yield molecules from both regular and malformed multi-molecule SDFs."""
    path = Path(path)
    raw_text = _normalise_text(path)
    is_multi_block = raw_text.count("$$$$") > 1
    supplied = []
    if not is_multi_block:
        try:
            supplier = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=False)
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

    blocks = raw_text.split("$$$$")
    for block in blocks:
        if not block.strip():
            continue
        mol = None
        # Different producers place one or two line endings around the
        # delimiter.  Try preserving the MolBlock first, then remove only
        # the delimiter newline, and finally remove all leading blank lines.
        # The candidates cover both named legacy records and empty-name
        # blocks emitted by RDKit's SDWriter.
        candidates = [block]
        if block.startswith("\n"):
            candidates.append(block[1:])
        candidates.append(block.lstrip("\n"))
        seen_candidates = set()
        for candidate in candidates:
            if candidate in seen_candidates:
                continue
            seen_candidates.add(candidate)
            mol_block = candidate
            data_block = ""
            if "M  END" in mol_block:
                mol_block, data_block = mol_block.split("M  END", 1)
                mol_block += "M  END\n"
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
            if mol is not None:
                _attach_sdf_data_fields(mol, data_block)
                break
        if mol is not None:
            yield mol


def iter_sdf_molecules_largest_fragment(
    path: str | Path,
) -> Iterable[Chem.Mol]:
    """Yield molecules from an SDF keeping ONLY the largest connected fragment.

    DEFAULT policy since 2026-09-24: DiffDynamic reconstruction sometimes
    leaves detached atom clusters (broken scaffold bonds) next to the main
    molecule.  Any downstream SMILES comparison / exact-hit scan / property
    export must operate on the largest fragment, otherwise multi-fragment
    SMILES never match the single-fragment references.  New code should
    prefer this iterator over raw ``iter_sdf_molecules`` when comparing
    against reference molecules.
    """
    from diffdynamic.chemistry.side_cut import keep_largest_fragment

    for molecule in iter_sdf_molecules(path):
        fragment = keep_largest_fragment(molecule)
        if fragment is not None:
            yield fragment


def _first_molecule(path: str | Path) -> Optional[Chem.Mol]:
    return next(iter_sdf_molecules(path), None)
