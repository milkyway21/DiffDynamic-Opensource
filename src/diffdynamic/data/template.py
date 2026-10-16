"""Coordinate-template initialization and guarded CRBN grafting utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from rdkit import Chem

from utils.gspt1_scaffold_prior import iter_sdf_molecules


TEMPLATE_INIT_SCHEMA_VERSION = 1


class TemplateInitError(ValueError):
    """Raised when a coordinate template cannot define a valid experiment."""


class GraftRejected(ValueError):
    """Raised when a generated target fragment cannot be grafted safely."""


@dataclass(frozen=True)
class TemplateGeometry:
    """A 3D ligand divided by one scaffold-to-target attachment bond."""

    molecule: Chem.Mol
    source: str
    scaffold_smarts: str
    positions: np.ndarray
    scaffold_atom_indices: tuple[int, ...]
    target_atom_indices: tuple[int, ...]
    scaffold_anchor_atom_index: int
    target_attachment_atom_index: int
    attachment_bond_type: Chem.BondType

    @property
    def scaffold_atom_count(self) -> int:
        return len(self.scaffold_atom_indices)

    @property
    def target_atom_count(self) -> int:
        return len(self.target_atom_indices)

    @property
    def total_atom_count(self) -> int:
        return self.molecule.GetNumAtoms()


@dataclass(frozen=True)
class TemplateInitializationBatch:
    """Template x0 coordinates and lineage metadata for one model batch."""

    positions: tuple[np.ndarray, ...]
    records: tuple[dict[str, Any], ...]

    @property
    def atom_counts(self) -> list[int]:
        return [int(pos.shape[0]) for pos in self.positions]

    def flattened_positions(self) -> np.ndarray:
        if not self.positions:
            return np.empty((0, 3), dtype=np.float64)
        return np.concatenate(self.positions, axis=0)


def load_template_geometry(
    path: str | Path,
    scaffold_smarts: str,
    *,
    require_3d: bool = True,
) -> TemplateGeometry:
    """Load one heavy-atom 3D template with exactly one scaffold exit."""
    source = Path(path).expanduser().resolve()
    molecule = next(iter_sdf_molecules(source), None)
    if molecule is None:
        raise TemplateInitError(f"cannot parse template ligand: {source}")
    molecule = Chem.RemoveHs(Chem.Mol(molecule))
    if molecule.GetNumConformers() != 1:
        raise TemplateInitError(
            f"template ligand must contain one conformer: {source}"
        )

    positions = np.asarray(
        molecule.GetConformer().GetPositions(), dtype=np.float64
    )
    if positions.shape != (molecule.GetNumAtoms(), 3):
        raise TemplateInitError("template coordinate shape does not match atoms")
    if not np.isfinite(positions).all():
        raise TemplateInitError("template contains non-finite coordinates")
    if require_3d and float(np.ptp(positions[:, 2])) < 0.1:
        raise TemplateInitError(
            "template is planar; a pocket-aligned 3D conformer is required"
        )

    query = Chem.MolFromSmarts(scaffold_smarts)
    if query is None:
        raise TemplateInitError(f"invalid scaffold SMARTS: {scaffold_smarts}")
    matches = molecule.GetSubstructMatches(query, uniquify=True)
    if len(matches) != 1:
        raise TemplateInitError(
            f"expected one scaffold match, found {len(matches)} in {source}"
        )
    scaffold_atom_indices = tuple(int(index) for index in matches[0])
    scaffold_set = set(scaffold_atom_indices)

    cross_bonds = []
    for bond in molecule.GetBonds():
        begin = int(bond.GetBeginAtomIdx())
        end = int(bond.GetEndAtomIdx())
        if (begin in scaffold_set) != (end in scaffold_set):
            cross_bonds.append(bond)
    if len(cross_bonds) != 1:
        raise TemplateInitError(
            f"expected one scaffold exit bond, found {len(cross_bonds)}"
        )

    cross_bond = cross_bonds[0]
    begin = int(cross_bond.GetBeginAtomIdx())
    end = int(cross_bond.GetEndAtomIdx())
    if begin in scaffold_set:
        scaffold_anchor, target_attachment = begin, end
    else:
        scaffold_anchor, target_attachment = end, begin
    target_atom_indices = tuple(
        index
        for index in range(molecule.GetNumAtoms())
        if index not in scaffold_set
    )
    if target_attachment not in target_atom_indices:
        raise TemplateInitError("target attachment is not outside the scaffold")

    return TemplateGeometry(
        molecule=molecule,
        source=str(source),
        scaffold_smarts=str(scaffold_smarts),
        positions=positions.copy(),
        scaffold_atom_indices=scaffold_atom_indices,
        target_atom_indices=target_atom_indices,
        scaffold_anchor_atom_index=scaffold_anchor,
        target_attachment_atom_index=target_attachment,
        attachment_bond_type=cross_bond.GetBondType(),
    )


def build_template_initialization_batch(
    geometry: TemplateGeometry,
    *,
    batch_size: int,
    atom_count_values: Sequence[int],
    seed: int,
    coordinate_jitter_std: float = 0.20,
    extra_point_jitter_std: float = 0.35,
) -> TemplateInitializationBatch:
    """Build exchangeable atom clouds around a real ligand conformation.

    Only coordinates and atom counts are returned. Target-side elements and
    bonds are intentionally absent. The target attachment lineage is placed
    first after the scaffold prefix so postprocessing remains deterministic.
    """
    batch_size = int(batch_size)
    if batch_size <= 0:
        raise TemplateInitError("batch_size must be positive")
    values = np.asarray([int(value) for value in atom_count_values], dtype=int)
    if values.size == 0:
        raise TemplateInitError("atom_count_values must not be empty")
    minimum = geometry.scaffold_atom_count + 1
    if np.any(values < minimum):
        raise TemplateInitError(
            f"atom counts must be at least {minimum} to retain an exit atom"
        )
    if coordinate_jitter_std < 0 or extra_point_jitter_std <= 0:
        raise TemplateInitError("jitter standard deviations are invalid")

    rng = np.random.default_rng(int(seed))
    counts = rng.choice(values, size=batch_size, replace=True)
    target_attachment = geometry.target_attachment_atom_index
    other_target = [
        index
        for index in geometry.target_atom_indices
        if index != target_attachment
    ]
    scaffold_positions = geometry.positions[
        list(geometry.scaffold_atom_indices)
    ]

    positions_out: list[np.ndarray] = []
    records: list[dict[str, Any]] = []
    for sample_index, total_count_raw in enumerate(counts):
        total_count = int(total_count_raw)
        requested_target_count = total_count - geometry.scaffold_atom_count
        existing_target_count = min(
            requested_target_count, geometry.target_atom_count
        )
        n_other = max(0, existing_target_count - 1)
        if n_other >= len(other_target):
            selected_other = list(other_target)
        else:
            selected_other = [
                int(index)
                for index in rng.choice(
                    np.asarray(other_target, dtype=int),
                    size=n_other,
                    replace=False,
                )
            ]
        selected_target = [target_attachment, *selected_other]
        target_positions = geometry.positions[selected_target]

        n_extra = requested_target_count - len(selected_target)
        extra_positions = np.empty((0, 3), dtype=np.float64)
        if n_extra > 0:
            base_indices = rng.choice(
                np.asarray(geometry.target_atom_indices, dtype=int),
                size=n_extra,
                replace=True,
            )
            extra_positions = (
                geometry.positions[base_indices]
                + rng.normal(
                    0.0,
                    float(extra_point_jitter_std),
                    size=(n_extra, 3),
                )
            )

        sample_positions = np.concatenate(
            [scaffold_positions, target_positions, extra_positions], axis=0
        ).astype(np.float64, copy=False)
        if coordinate_jitter_std > 0:
            sample_positions = sample_positions + rng.normal(
                0.0,
                float(coordinate_jitter_std),
                size=sample_positions.shape,
            )
        if sample_positions.shape != (total_count, 3):
            raise TemplateInitError(
                "template initialization produced an invalid atom count"
            )

        n_scaffold = geometry.scaffold_atom_count
        target_local_indices = list(range(n_scaffold, total_count))
        target_template_indices = [
            *selected_target,
            *([-1] * n_extra),
        ]
        records.append({
            "schema_version": TEMPLATE_INIT_SCHEMA_VERSION,
            "sample_index": int(sample_index),
            "template_source": geometry.source,
            "scaffold_smarts": geometry.scaffold_smarts,
            "template_total_atom_count": geometry.total_atom_count,
            "sampled_total_atom_count": total_count,
            "scaffold_local_indices": list(range(n_scaffold)),
            "scaffold_template_atom_indices": list(
                geometry.scaffold_atom_indices
            ),
            "target_local_indices": target_local_indices,
            "target_template_atom_indices": target_template_indices,
            "target_attachment_local_index": n_scaffold,
            "target_attachment_template_atom_index": target_attachment,
            "scaffold_anchor_template_atom_index": (
                geometry.scaffold_anchor_atom_index
            ),
            "coordinate_jitter_std": float(coordinate_jitter_std),
            "extra_point_jitter_std": float(extra_point_jitter_std),
            "template_x0_pos": sample_positions.copy(),
        })
        positions_out.append(sample_positions)

    return TemplateInitializationBatch(
        positions=tuple(positions_out),
        records=tuple(records),
    )


def induced_submolecule(
    molecule: Chem.Mol,
    atom_indices: Sequence[int],
) -> tuple[Chem.Mol, dict[int, int]]:
    """Return an induced atom subgraph while preserving its 3D coordinates."""
    keep = sorted({int(index) for index in atom_indices})
    if not keep:
        raise GraftRejected("cannot extract an empty submolecule")
    if keep[0] < 0 or keep[-1] >= molecule.GetNumAtoms():
        raise GraftRejected("submolecule atom index is out of range")
    editable = Chem.RWMol(Chem.Mol(molecule))
    keep_set = set(keep)
    for index in range(molecule.GetNumAtoms() - 1, -1, -1):
        if index not in keep_set:
            editable.RemoveAtom(index)
    result = editable.GetMol()
    try:
        Chem.SanitizeMol(result)
    except Exception as exc:
        raise GraftRejected(f"submolecule sanitization failed: {exc}") from exc
    return result, {old: new for new, old in enumerate(keep)}


def select_attachment_component(
    molecule: Chem.Mol,
    attachment_atom_index: int,
) -> tuple[Chem.Mol, int]:
    """Keep the connected component containing the generated exit lineage."""
    attachment_atom_index = int(attachment_atom_index)
    if not 0 <= attachment_atom_index < molecule.GetNumAtoms():
        raise GraftRejected("generated attachment atom is missing")
    mappings: list[tuple[int, ...]] = []
    fragments = Chem.GetMolFrags(
        molecule,
        asMols=True,
        sanitizeFrags=False,
        fragsMolAtomMapping=mappings,
    )
    for fragment, mapping in zip(fragments, mappings):
        if attachment_atom_index not in mapping:
            continue
        new_index = mapping.index(attachment_atom_index)
        try:
            Chem.SanitizeMol(fragment)
        except Exception as exc:
            raise GraftRejected(
                f"target attachment component is invalid: {exc}"
            ) from exc
        return fragment, int(new_index)
    raise GraftRejected("generated attachment component was not found")


def graft_template_scaffold(
    geometry: TemplateGeometry,
    target_molecule: Chem.Mol,
    target_attachment_atom_index: int,
    *,
    minimum_bond_distance: float = 0.90,
    maximum_covalent_scale: float = 1.25,
    collision_covalent_scale: float = 0.65,
) -> tuple[Chem.Mol, dict[str, Any]]:
    """Attach the exact template scaffold when geometry and valence permit."""
    target_molecule = Chem.RemoveHs(Chem.Mol(target_molecule))
    target_attachment_atom_index = int(target_attachment_atom_index)
    if not 0 <= target_attachment_atom_index < target_molecule.GetNumAtoms():
        raise GraftRejected("target attachment atom is out of range")
    if target_molecule.GetNumConformers() != 1:
        raise GraftRejected("target fragment has no unique 3D conformer")

    scaffold, scaffold_mapping = induced_submolecule(
        geometry.molecule, geometry.scaffold_atom_indices
    )
    scaffold_anchor = scaffold_mapping[
        geometry.scaffold_anchor_atom_index
    ]
    scaffold_pos = np.asarray(
        scaffold.GetConformer().GetPositions(), dtype=np.float64
    )
    target_pos = np.asarray(
        target_molecule.GetConformer().GetPositions(), dtype=np.float64
    )
    anchor_position = scaffold_pos[scaffold_anchor]
    attachment_position = target_pos[target_attachment_atom_index]
    bond_distance = float(
        np.linalg.norm(anchor_position - attachment_position)
    )

    periodic_table = Chem.GetPeriodicTable()
    scaffold_atom = scaffold.GetAtomWithIdx(scaffold_anchor)
    target_atom = target_molecule.GetAtomWithIdx(
        target_attachment_atom_index
    )
    covalent_sum = float(
        periodic_table.GetRcovalent(scaffold_atom.GetAtomicNum())
        + periodic_table.GetRcovalent(target_atom.GetAtomicNum())
    )
    maximum_bond_distance = maximum_covalent_scale * covalent_sum
    if not minimum_bond_distance <= bond_distance <= maximum_bond_distance:
        raise GraftRejected(
            "attachment bond distance outside covalent range: "
            f"{bond_distance:.3f} A not in "
            f"[{minimum_bond_distance:.3f}, {maximum_bond_distance:.3f}]"
        )

    for scaffold_index, scaffold_coord in enumerate(scaffold_pos):
        for target_index, target_coord in enumerate(target_pos):
            if (
                scaffold_index == scaffold_anchor
                and target_index == target_attachment_atom_index
            ):
                continue
            distance = float(np.linalg.norm(scaffold_coord - target_coord))
            radii = float(
                periodic_table.GetRcovalent(
                    scaffold.GetAtomWithIdx(scaffold_index).GetAtomicNum()
                )
                + periodic_table.GetRcovalent(
                    target_molecule.GetAtomWithIdx(target_index).GetAtomicNum()
                )
            )
            if distance < collision_covalent_scale * radii:
                raise GraftRejected(
                    "scaffold-target atom collision: "
                    f"{scaffold_index}/{target_index} at {distance:.3f} A"
                )

    combined = Chem.CombineMols(scaffold, target_molecule)
    editable = Chem.RWMol(combined)
    combined_target_attachment = (
        scaffold.GetNumAtoms() + target_attachment_atom_index
    )
    editable.AddBond(
        scaffold_anchor,
        combined_target_attachment,
        geometry.attachment_bond_type,
    )
    grafted = editable.GetMol()
    try:
        Chem.SanitizeMol(grafted)
    except Exception as exc:
        raise GraftRejected(f"grafted molecule has invalid valence: {exc}") from exc

    return grafted, {
        "attachment_distance": bond_distance,
        "maximum_attachment_distance": maximum_bond_distance,
        "scaffold_atom_count": scaffold.GetNumAtoms(),
        "target_atom_count": target_molecule.GetNumAtoms(),
        "grafted_atom_count": grafted.GetNumAtoms(),
        "scaffold_anchor_atom_index": int(scaffold_anchor),
        "target_attachment_atom_index": int(
            combined_target_attachment
        ),
    }
