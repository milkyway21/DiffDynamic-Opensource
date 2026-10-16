#!/usr/bin/env python3
"""Visual side-chain similarity via RDKit Fraggle.

Query is the synthetic reference (fragmented once).  Generated molecules are
scored against those fragments.  Empty Fraggle decompositions fall back to the
whole-molecule SMILES plus the same path-fingerprint Tanimoto Fraggle uses
internally, so compact fused extras never score as 0 by accident.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.Fraggle.FraggleSim import (
    GetFraggleSimilarity,
    compute_fraggle_similarity_for_subs,
    generate_fraggle_fragmentation,
    rdkitFpParams,
)

RDLogger.DisableLog("rdApp.*")

QueryMol = Chem.Mol


def morgan_fingerprint(molecule: Chem.Mol):
    return AllChem.GetMorganFingerprintAsBitVect(molecule, radius=2, nBits=2048)


def morgan_tanimoto(query: Chem.Mol, mol: Chem.Mol) -> float:
    return float(
        DataStructs.TanimotoSimilarity(
            morgan_fingerprint(query), morgan_fingerprint(mol)
        )
    )


def path_tanimoto(query: Chem.Mol, mol: Chem.Mol) -> float:
    q_fp = Chem.RDKFingerprint(query, **rdkitFpParams)
    m_fp = Chem.RDKFingerprint(mol, **rdkitFpParams)
    return float(DataStructs.TanimotoSimilarity(q_fp, m_fp))


def prepare_query(molecule: Chem.Mol) -> Chem.Mol:
    """Cache Fraggle fragments on a copy; inject the whole molecule if empty."""
    query = Chem.Mol(molecule)
    try:
        frags = list(generate_fraggle_fragmentation(query))
    except Exception:
        frags = []
    if not frags:
        smiles = Chem.MolToSmiles(query, isomericSmiles=True)
        if smiles:
            frags = [smiles]
    query._fraggleDecomp = frags
    return query


def fraggle_fragments(query: Chem.Mol) -> List[str]:
    if not hasattr(query, "_fraggleDecomp"):
        query = prepare_query(query)
    return list(getattr(query, "_fraggleDecomp") or [])


def fraggle_similarity(query: Chem.Mol, mol: Chem.Mol) -> Tuple[float, Optional[str]]:
    """Return (Fraggle score, best matching query fragment).

    ``query`` should already be ``prepare_query``'d when scoring a library.
    Direction is fixed: query = reference, mol = generated.
    """
    if query is None or mol is None:
        return 0.0, None
    if not hasattr(query, "_fraggleDecomp") or not query._fraggleDecomp:
        query = prepare_query(query)
    try:
        score, match = GetFraggleSimilarity(query, mol)
    except Exception:
        score, match = 0.0, None
    if score <= 0.0:
        fallback = path_tanimoto(query, mol)
        if fallback > score:
            score = fallback
            match = match or Chem.MolToSmiles(query, isomericSmiles=True)
    return float(score), match


def fraggle_and_morgan(
    query: Chem.Mol, mol: Chem.Mol
) -> Tuple[float, Optional[str], float]:
    score, match = fraggle_similarity(query, mol)
    return score, match, morgan_tanimoto(query, mol)


__all__ = [
    "prepare_query",
    "fraggle_fragments",
    "fraggle_similarity",
    "fraggle_and_morgan",
    "morgan_tanimoto",
    "path_tanimoto",
    "morgan_fingerprint",
    "compute_fraggle_similarity_for_subs",
]
