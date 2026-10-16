"""Read-only structure preview for de novo generation; never starts a job."""

from pathlib import Path

import numpy as np
from rdkit import Chem

from diffdynamic.data.protein import PDBProtein
from diffdynamic.chemistry.scaffold_inputs import load_reference_molecule


def load_generation_context(params):
    """Show the selected input in its original coordinates, without a scaffold."""
    if not params.get("protein_path"):
        from server.scaffold_context import load_context
        # Reuse the exact pretrained dataset/split resolver used by sampling.
        response = load_context({**params, "scaffold_source": "none"})[3]
        return {**response, "scaffold_indices": [], "data_id": params.get("data_id", 0)}

    protein_path = params["protein_path"]
    protein = PDBProtein(protein_path)
    positions = np.asarray(protein.pos)
    if not len(positions) or not np.isfinite(positions).all():
        raise ValueError("Protein requires finite atom coordinates")
    molecule_path = params.get("ligand_path")
    mol = load_reference_molecule(molecule_path) if molecule_path else None
    protein_block = Path(protein_path).read_text()
    pocket_cropped = False
    if mol is not None:
        ligand_positions = np.asarray(mol.GetConformer().GetPositions(), dtype=np.float32)
        residues = protein.query_residues_ligand(
            {"pos": ligand_positions}, params.get("pocket_radius", 10.0),
        )
        if residues:
            # Keep chain/residue order so a ribbon cannot connect unrelated
            # fragments through the order of ligand-neighbor discovery.
            selected = {id(residue) for residue in residues}
            residues = [residue for residue in protein.residues if id(residue) in selected]
            protein_block = protein.residues_to_pdb_block(residues)
            pocket_cropped = True
    return {
        "protein_pdb": protein_block,
        "ligand_sdf": Chem.MolToMolBlock(mol) if mol is not None else None,
        "scaffold_indices": [],
        "atoms": [{"index": atom.GetIdx(), "element": atom.GetSymbol(),
                   "xyz": list(mol.GetConformer().GetAtomPosition(atom.GetIdx()))}
                  for atom in mol.GetAtoms()] if mol is not None else [],
        "coordinate_frame": "world_angstrom",
        "pocket_cropped": pocket_cropped,
        "pocket_radius": params.get("pocket_radius", 10.0),
    }
