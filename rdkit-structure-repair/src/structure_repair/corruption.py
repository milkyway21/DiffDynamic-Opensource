"""Helpers to build corrupted molecules for tests and datasets."""

from __future__ import annotations

from typing import List, Optional, Tuple

from rdkit import Chem


def mol_from_smiles(smi: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        raise ValueError(f"Cannot parse SMILES: {smi}")
    Chem.SanitizeMol(mol)
    return mol


def corrupt_aromatic_to_all_single(mol: Chem.Mol) -> Chem.Mol:
    """Clear aromatic flags and set all formerly-aromatic bonds to single."""
    rw = Chem.RWMol(Chem.Mol(mol))
    for atom in rw.GetAtoms():
        atom.SetIsAromatic(False)
        atom.SetNumExplicitHs(0)
        atom.SetNoImplicit(False)
    for bond in rw.GetBonds():
        if bond.GetIsAromatic() or bond.GetBondType() == Chem.BondType.AROMATIC:
            bond.SetIsAromatic(False)
            bond.SetBondType(Chem.BondType.SINGLE)
        else:
            bond.SetIsAromatic(False)
    out = rw.GetMol()
    out.UpdatePropertyCache(strict=False)
    # Force low H count on ring carbons to mimic reconstruction artifacts
    rw2 = Chem.RWMol(out)
    Chem.GetSymmSSSR(rw2)
    for atom in rw2.GetAtoms():
        if atom.GetAtomicNum() == 6 and atom.IsInRingSize(6):
            atom.SetNumExplicitHs(1)
            atom.SetNoImplicit(True)
    out2 = rw2.GetMol()
    out2.UpdatePropertyCache(strict=False)
    return out2


def corrupt_c6_to_consecutive_doubles(mol: Chem.Mol) -> Chem.Mol:
    """Set a C6 carbon ring to C1=C=C=C=C=C1 style consecutive doubles."""
    rw = Chem.RWMol(Chem.Mol(mol))
    for atom in rw.GetAtoms():
        atom.SetIsAromatic(False)
        atom.SetNumExplicitHs(0)
        atom.SetNoImplicit(False)
    for bond in rw.GetBonds():
        bond.SetIsAromatic(False)
    Chem.GetSymmSSSR(rw)
    ring = None
    for r in rw.GetRingInfo().AtomRings():
        if len(r) == 6 and all(rw.GetAtomWithIdx(i).GetAtomicNum() == 6 for i in r):
            ring = list(r)
            break
    if ring is None:
        raise ValueError("No C6 carbon ring found")
    # Order ring
    ordered = [ring[0]]
    prev = -1
    cur = ring[0]
    ring_set = set(ring)
    for _ in range(5):
        for nbr in rw.GetAtomWithIdx(cur).GetNeighbors():
            ni = nbr.GetIdx()
            if ni in ring_set and ni != prev and ni not in ordered:
                prev, cur = cur, ni
                ordered.append(cur)
                break
    for i in range(6):
        b = rw.GetBondBetweenAtoms(ordered[i], ordered[(i + 1) % 6])
        if b is not None:
            b.SetBondType(Chem.BondType.DOUBLE)
            b.SetIsAromatic(False)
    out = rw.GetMol()
    out.UpdatePropertyCache(strict=False)
    return out


def corrupt_nitro_neutral_double(mol: Chem.Mol) -> Chem.Mol:
    """Convert [N+](=O)[O-] to N(=O)=O (both double, no charges)."""
    rw = Chem.RWMol(Chem.Mol(mol))
    for atom in rw.GetAtoms():
        if atom.GetAtomicNum() != 7:
            continue
        oxygens = []
        for bond in atom.GetBonds():
            o = bond.GetOtherAtom(atom)
            if o.GetAtomicNum() == 8:
                oxygens.append((o, bond))
        if len(oxygens) < 2:
            continue
        atom.SetFormalCharge(0)
        for o, bond in oxygens:
            o.SetFormalCharge(0)
            bond.SetBondType(Chem.BondType.DOUBLE)
            bond.SetIsAromatic(False)
    out = rw.GetMol()
    out.UpdatePropertyCache(strict=False)
    return out


def corrupt_clear_charges(mol: Chem.Mol) -> Chem.Mol:
    rw = Chem.RWMol(Chem.Mol(mol))
    for atom in rw.GetAtoms():
        atom.SetFormalCharge(0)
    out = rw.GetMol()
    out.UpdatePropertyCache(strict=False)
    return out
