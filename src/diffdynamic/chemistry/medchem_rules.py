#!/usr/bin/env python3
"""Medicinal-chemistry rule engine for side-chain candidate generation.

Pure RDKit, Python 3.8 compatible, no project dependencies so that worker
processes can import it cheaply.

Input molecules are side-chain fragments carrying exactly one dummy atom
``[1*:1]`` at the scaffold boundary (same convention as the audit scripts).
Rules never touch the dummy atom; every candidate is re-validated before it is
returned.

Rule families
    R1 bioisosteric replacements   (heteroaryl swaps, CF3/OCF3, oxime ether)
    R2 potency fine-tuning         (aryl F/Cl/Me, +/-CH2 homologation)
    R3 metabolic stability         (N/O methylation, O-demethylation)
    R4 conformational locking      (piperidine F, ring CH2 -> NH/O)
    R5 polarity tuning             (CN <-> CONH2, aryl OMe)
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, FilterCatalog, rdChemReactions

RDLogger.DisableLog("rdApp.*")

DUMMY_ATOMIC_NUM = 0
DUMMY_ISOTOPE = 1
DUMMY_MAP = 1

MAX_HEAVY_DELTA = 3
MIN_SIDE_HEAVY = 4


@dataclass(frozen=True)
class RuleHit:
    """One generated candidate for one parent molecule."""

    parent_smiles: str
    smiles: str
    rule_id: str
    family: str
    heavy_atoms: int = 0


@dataclass(frozen=True)
class _ReactionRule:
    rule_id: str
    family: str
    rxn: Any


# --------------------------------------------------------------------------
# Reaction catalogue (acyclic, well-behaved edits only; aromatic ring swaps
# are done by manual RWMol element mutation because SMIRKS inherits hydrogen
# counts from the reactant atom and produces unsanitizable products).
# --------------------------------------------------------------------------

_REACTION_SPECS: List[Tuple[str, str, str]] = [
    # ---- R1 bioisosteric (acyclic) ---------------------------------------
    (
        "r1_cf3_to_ocf3",
        "R1",
        "[c:1][C:2]([F:3])([F:4])[F:5]>>[c:1][O:6][C:2]([F:3])([F:4])[F:5]",
    ),
    (
        "r1_ocf3_to_cf3",
        "R1",
        "[c:1][O:6][C:2]([F:3])([F:4])[F:5]>>[c:1][C:2]([F:3])([F:4])[F:5]",
    ),
    (
        "r1_ketone_to_oxime_ether",
        "R1",
        "[C;H0;!R:1]=[O:2]>>[C:1]=[N:3][O:4][CH3:5]",
    ),
    # ---- R3 metabolic ----------------------------------------------------
    (
        "r3_amide_n_methylation",
        "R3",
        "[N;H1:1][C:2]=[O:3]>>[N:1]([CH3])[C:2]=[O:3]",
    ),
    (
        "r3_amine_n_methylation_1",
        "R3",
        "[N;H2;!R:1]>>[N:1]([CH3])",
    ),
    (
        "r3_amine_n_methylation_2",
        "R3",
        "[N;H1;!$(N-C=[O]);!R:1]>>[N:1]([CH3])",
    ),
    (
        "r3_anisole_demethylation",
        "R3",
        "[CH3:2][O:1][c:3]>>[O:1][c:3]",
    ),
    (
        "r3_phenol_methylation",
        "R3",
        "[c:3][O;H1:1]>>[c:3][O:1][CH3]",
    ),
    # ---- R5 polarity -----------------------------------------------------
    (
        "r5_nitrile_to_amide",
        "R5",
        "[#6:1][C:2]#[N:3]>>[#6:1][C:2](=[O:4])[NH2]",
    ),
    (
        "r5_amide_to_nitrile",
        "R5",
        "[#6;!R:1][C:2](=[O:4])[#7;H2]>>[#6:1][C:2]#[N]",
    ),
    (
        "r5_aryl_methoxylation",
        "R5",
        "[c;H1:1]>>[c:1][O][CH3]",
    ),
]


@functools.lru_cache(maxsize=1)
def _reaction_rules() -> List[_ReactionRule]:
    rules: List[_ReactionRule] = []
    for rule_id, family, smarts in _REACTION_SPECS:
        try:
            rxn = rdChemReactions.ReactionFromSmarts(smarts)
            if rxn.GetNumReactantTemplates() != 1:
                continue
            rules.append(_ReactionRule(rule_id=rule_id, family=family, rxn=rxn))
        except Exception:
            continue
    return rules


@functools.lru_cache(maxsize=1)
def alert_catalog() -> FilterCatalog.FilterCatalog:
    params = FilterCatalog.FilterCatalogParams()
    catalogs = FilterCatalog.FilterCatalogParams.FilterCatalogs
    params.AddCatalog(catalogs.PAINS)
    params.AddCatalog(catalogs.BRENK)
    return FilterCatalog.FilterCatalog(params)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def canonical_side_smiles(molecule: Chem.Mol) -> str:
    """Canonical SMILES with the audit-script convention (isomeric off)."""
    return Chem.MolToSmiles(molecule, isomericSmiles=False)


def _dummy_indices(molecule: Chem.Mol) -> List[int]:
    return [
        atom.GetIdx()
        for atom in molecule.GetAtoms()
        if atom.GetAtomicNum() == DUMMY_ATOMIC_NUM
    ]


def heavy_atom_count(molecule: Chem.Mol) -> int:
    return sum(1 for atom in molecule.GetAtoms() if atom.GetAtomicNum() > 1)


def to_all_carbon_graph(smiles: str) -> Optional[str]:
    """Canonical SMILES of the all-carbon graph of a side chain.

    Convention matches ``scripts/gspt1_allc_tanimoto_audit._to_all_carbon``:
    every heavy atom becomes neutral carbon while bond orders, aromatic flags,
    and ring topology are kept; sanitization is lenient so aromatic fused
    graphs (e.g. benzofuran O->C) survive as canonical output strings.
    """
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None
    editable = Chem.RWMol(molecule)
    for atom in editable.GetAtoms():
        if atom.GetAtomicNum() <= 1 or atom.GetAtomicNum() == DUMMY_ATOMIC_NUM:
            continue
        atom.SetAtomicNum(6)
        atom.SetFormalCharge(0)
        atom.SetIsotope(0)
        atom.SetNumExplicitHs(0)
        atom.SetNoImplicit(False)
        atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    result = editable.GetMol()
    result.UpdatePropertyCache(strict=False)
    try:
        return Chem.MolToSmiles(result, canonical=True, isomericSmiles=False)
    except Exception:
        return None


def _validate_candidate(
    candidate: Chem.Mol,
    parent_heavy: int,
) -> Tuple[bool, str]:
    try:
        Chem.SanitizeMol(Chem.Mol(candidate))
    except Exception:
        return False, "sanitize_failed"
    dummies = _dummy_indices(candidate)
    if len(dummies) != 1:
        return False, "dummy_count_%d" % len(dummies)
    dummy = candidate.GetAtomWithIdx(dummies[0])
    if dummy.GetDegree() != 1:
        return False, "dummy_degree_changed"
    if dummy.GetFormalCharge() != 0:
        return False, "dummy_charge_changed"
    heavy = heavy_atom_count(candidate)
    if abs(heavy - parent_heavy) > MAX_HEAVY_DELTA:
        return False, "heavy_delta_%+d" % (heavy - parent_heavy)
    if heavy < MIN_SIDE_HEAVY:
        return False, "too_small"
    try:
        entry = alert_catalog().GetFirstMatch(Chem.Mol(candidate))
    except Exception:
        entry = None
    if entry is not None:
        return False, "alert_%s" % entry.GetDescription()[:40]
    return True, "ok"


def _heteroaryl_swaps(parent: Chem.Mol) -> List[Tuple[str, str, Chem.Mol]]:
    """R1: swap heteroatoms inside five/six-member aromatic rings.

    Element maps keep valence sane: any single heteroatom in an aromatic ring
    becomes S / O / NH, one at a time.  Hydrogen counts are recomputed by RDKit
    after mutation, so products sanitize cleanly.
    """
    out: List[Tuple[str, str, Chem.Mol]] = []
    info = parent.GetRingInfo()
    for ring in info.AtomRings():
        ring_len = len(ring)
        if ring_len not in (5, 6):
            continue
        atoms = [parent.GetAtomWithIdx(int(idx)) for idx in ring]
        if not all(atom.GetIsAromatic() for atom in atoms):
            continue
        hetero = [
            int(idx)
            for idx in ring
            if parent.GetAtomWithIdx(int(idx)).GetAtomicNum() in (7, 8, 15, 16)
            or parent.GetAtomWithIdx(int(idx)).GetAtomicNum() == 0
        ]
        if not hetero:
            continue
        for target_idx in hetero:
            for atomic_num, label in ((16, "S"), (8, "O"), (7, "NH")):
                if parent.GetAtomWithIdx(target_idx).GetAtomicNum() == atomic_num:
                    continue
                if atomic_num == 8 and ring_len == 6:
                    continue  # six-member aromatic O is not a sane swap target
                editable = Chem.RWMol(parent)
                atom = editable.GetAtomWithIdx(target_idx)
                atom.SetAtomicNum(atomic_num)
                if atomic_num == 7:
                    atom.SetNoImplicit(False)
                    atom.SetNumExplicitHs(1)
                else:
                    atom.SetNoImplicit(True)
                    atom.SetNumExplicitHs(0)
                    if atomic_num == 8:
                        atom.SetNumExplicitHs(0)
                candidate = editable.GetMol()
                try:
                    Chem.SanitizeMol(candidate)
                except Exception:
                    continue
                out.append(("r1_heteroaryl_%s" % label, "R1", candidate))
    # Deduplicate by canonical smiles.
    seen = set()
    unique: List[Tuple[str, str, Chem.Mol]] = []
    for rule_id, family, candidate in out:
        try:
            key = canonical_side_smiles(candidate)
        except Exception:
            continue
        if key in seen:
            continue
        seen.add(key)
        unique.append((rule_id, family, candidate))
    return unique


def _saturated_ring_edits(parent: Chem.Mol) -> List[Tuple[str, str, Chem.Mol]]:
    """R4: CH2 -> NH / O / NMe inside saturated six-member rings."""
    out: List[Tuple[str, str, Chem.Mol]] = []
    info = parent.GetRingInfo()
    for ring in info.AtomRings():
        if len(ring) != 6:
            continue
        atoms = [parent.GetAtomWithIdx(int(idx)) for idx in ring]
        if any(atom.GetIsAromatic() for atom in atoms):
            continue
        if not all(atom.GetAtomicNum() in (6, 7, 8) for atom in atoms):
            continue
        hetero_count = sum(1 for atom in atoms if atom.GetAtomicNum() in (7, 8))
        if hetero_count > 2:
            continue
        for idx in ring:
            atom = parent.GetAtomWithIdx(int(idx))
            if atom.GetAtomicNum() != 6 or atom.GetTotalNumHs() != 2:
                continue
            for atomic_num, label, explicit_hs in ((7, "NH", 1), (8, "O", 0)):
                editable = Chem.RWMol(parent)
                target = editable.GetAtomWithIdx(int(idx))
                target.SetAtomicNum(atomic_num)
                target.SetNoImplicit(True)
                target.SetNumExplicitHs(explicit_hs)
                candidate = editable.GetMol()
                try:
                    Chem.SanitizeMol(candidate)
                except Exception:
                    continue
                out.append(("r4_ring_ch2_to_%s" % label, "R4", candidate))
    seen = set()
    unique: List[Tuple[str, str, Chem.Mol]] = []
    for rule_id, family, candidate in out:
        try:
            key = canonical_side_smiles(candidate)
        except Exception:
            continue
        if key in seen:
            continue
        seen.add(key)
        unique.append((rule_id, family, candidate))
    return unique


def _graph_distance_to_dummy(molecule: Chem.Mol, dummy_idx: int) -> Dict[int, int]:
    distances = {dummy_idx: 0}
    frontier = [dummy_idx]
    depth = 0
    while frontier:
        depth += 1
        next_frontier: List[int] = []
        for index in frontier:
            for neighbour in molecule.GetAtomWithIdx(index).GetNeighbors():
                neighbour_idx = neighbour.GetIdx()
                if neighbour_idx not in distances:
                    distances[neighbour_idx] = depth
                    next_frontier.append(neighbour_idx)
        frontier = next_frontier
    return distances


def _manual_substitution_candidates(
    parent: Chem.Mol,
    max_per_rule: int,
) -> List[Tuple[str, str, Chem.Mol]]:
    """R2 aryl halogenation/methylation on aryl CH positions."""
    dummies = _dummy_indices(parent)
    if len(dummies) != 1:
        return []
    distances = _graph_distance_to_dummy(parent, dummies[0])
    out: List[Tuple[str, str, Chem.Mol]] = []

    # R2: aromatic CH -> F / Cl / CH3, closest-to-dummy first.
    aryl_ch = sorted(
        (
            atom.GetIdx()
            for atom in parent.GetAtoms()
            if atom.GetIsAromatic()
            and atom.GetAtomicNum() == 6
            and atom.GetTotalNumHs() == 1
        ),
        key=lambda idx: (distances.get(idx, 99), idx),
    )[:3]
    substituent_atoms = [(9, "F"), (17, "Cl"), (6, "Me")]
    for aryl_idx in aryl_ch:
        for atomic_num, label in substituent_atoms:
            if len(out) >= max_per_rule:
                break
            editable = Chem.RWMol(parent)
            new_atom = Chem.Atom(atomic_num)
            if atomic_num == 6:
                new_atom.SetNoImplicit(False)
            else:
                new_atom.SetNoImplicit(True)
            new_idx = editable.AddAtom(new_atom)
            editable.AddBond(aryl_idx, new_idx, Chem.BondType.SINGLE)
            candidate = editable.GetMol()
            out.append(("r2_aryl_%s" % label, "R2", candidate))

    # R4: piperidine 3-/4-position fluorination lives in
    # _piperidine_fluorinations(); nothing else needed here.
    return out


def _homologation_candidates(
    parent: Chem.Mol,
    max_per_rule: int,
) -> List[Tuple[str, str, Chem.Mol]]:
    """R2 +/- CH2 homologation on benzylic C-C bonds."""
    dummies = _dummy_indices(parent)
    if len(dummies) != 1:
        return []
    out: List[Tuple[str, str, Chem.Mol]] = []
    for bond in parent.GetBonds():
        begin = bond.GetBeginAtom()
        end = bond.GetEndAtom()
        pair = (begin, end)
        if any(atom.GetAtomicNum() != 6 or atom.GetIsAromatic() for atom in pair):
            continue
        if bond.GetBondType() != Chem.BondType.SINGLE:
            continue
        # Grow: single C-C where one side is benzylic CH2.
        for atom, other in (pair, (pair[1], pair[0])):
            if atom.GetTotalNumHs() != 2 or atom.IsInRing():
                continue
            if not any(
                neighbour.GetAtomicNum() == 6 and neighbour.GetIsAromatic()
                for neighbour in atom.GetNeighbors()
            ):
                continue
            if len(out) >= max_per_rule:
                break
            editable = Chem.RWMol(parent)
            insert_idx = editable.AddAtom(Chem.Atom(6))
            other_idx = other.GetIdx()
            editable.RemoveBond(atom.GetIdx(), other_idx)
            editable.AddBond(atom.GetIdx(), insert_idx, Chem.BondType.SINGLE)
            editable.AddBond(insert_idx, other_idx, Chem.BondType.SINGLE)
            out.append(("r2_ch2_grow", "R2", editable.GetMol()))
        # Shrink: remove a benzylic CH2 (other side keeps connectivity).
        benzylic = None
        for atom, other in (pair, (pair[1], pair[0])):
            if atom.GetTotalNumHs() == 2 and not atom.IsInRing():
                if any(
                    neighbour.GetAtomicNum() == 6 and neighbour.GetIsAromatic()
                    for neighbour in atom.GetNeighbors()
                ):
                    benzylic = (atom, other)
                    break
        if benzylic is not None and len(out) < max_per_rule + 1:
            atom, other = benzylic
            if other.GetDegree() >= 1 and atom.GetDegree() == 2:
                editable = Chem.RWMol(parent)
                editable.RemoveBond(atom.GetIdx(), other.GetIdx())
                editable.RemoveAtom(atom.GetIdx())
                out.append(("r2_ch2_shrink", "R2", editable.GetMol()))
    return out


def generate_candidates(
    parent_smiles: str,
    max_candidates: int = 8,
    max_per_rule: int = 4,
    rules: Optional[Dict[str, bool]] = None,
) -> List[RuleHit]:
    """Generate unique, validated candidate side chains for one parent.

    Args:
        parent_smiles: side-chain SMILES with one ``[1*:1]`` dummy atom.
        max_candidates: hard cap on returned candidates.
        max_per_rule: cap per individual rule before deduplication.
        rules: optional family switch map, e.g. ``{"R2": False}``.

    Returns:
        List of :class:`RuleHit` sorted by family then rule id.
    """
    parent = Chem.MolFromSmiles(parent_smiles)
    if parent is None:
        return []
    if len(_dummy_indices(parent)) != 1:
        return []
    parent_heavy = heavy_atom_count(parent)
    disabled = {key for key, enabled in (rules or {}).items() if not enabled}

    seen = {canonical_side_smiles(parent)}
    hits: List[RuleHit] = []

    def try_add(rule_id: str, family: str, candidate: Chem.Mol) -> bool:
        if len(hits) >= max_candidates:
            return False
        try:
            Chem.SanitizeMol(candidate)
        except Exception:
            return True  # rejected, but do not stop the loop
        smiles = canonical_side_smiles(candidate)
        if smiles in seen:
            return True
        ok, _reason = _validate_candidate(candidate, parent_heavy)
        if not ok:
            return True
        seen.add(smiles)
        hits.append(
            RuleHit(
                parent_smiles=parent_smiles,
                smiles=smiles,
                rule_id=rule_id,
                family=family,
                heavy_atoms=heavy_atom_count(candidate),
            )
        )
        return len(hits) < max_candidates

    # R1-R5 reactions.
    if (
        "R1" not in disabled
        or "R3" not in disabled
        or "R4" not in disabled
        or "R5" not in disabled
    ):
        for rule in _reaction_rules():
            if rule.family in disabled:
                continue
            try:
                product_sets = rule.rxn.RunReactants((parent,))
            except Exception:
                continue
            emitted = 0
            for product_set in product_sets:
                if emitted >= max_per_rule or not len(hits) < max_candidates:
                    break
                if not product_set:
                    continue
                product = product_set[0]
                try:
                    if len(Chem.GetMolFrags(product)) != 1:
                        continue
                except Exception:
                    continue
                emitted += 1
                try_add(rule.rule_id, rule.family, product)

    # Manual R1/R2/R4 candidates.
    manual: List[Tuple[str, str, Chem.Mol]] = []
    if "R1" not in disabled:
        manual.extend(_heteroaryl_swaps(parent))
    if "R2" not in disabled:
        manual.extend(_manual_substitution_candidates(parent, max_per_rule))
        manual.extend(_homologation_candidates(parent, max_per_rule))
    if "R4" not in disabled:
        manual.extend(_piperidine_fluorinations(parent))
        manual.extend(_saturated_ring_edits(parent))
    for rule_id, family, candidate in manual:
        if not try_add(rule_id, family, candidate):
            break

    hits.sort(key=lambda hit: (hit.family, hit.rule_id, hit.smiles))
    return hits


def _piperidine_fluorinations(parent: Chem.Mol) -> List[Tuple[str, str, Chem.Mol]]:
    out: List[Tuple[str, str, Chem.Mol]] = []
    dummies = _dummy_indices(parent)
    if len(dummies) != 1:
        return out
    distances = _graph_distance_to_dummy(parent, dummies[0])
    info = parent.GetRingInfo()
    for ring in info.AtomRings():
        if len(ring) != 6:
            continue
        atoms = [parent.GetAtomWithIdx(int(idx)) for idx in ring]
        if not all(
            atom.GetAtomicNum() in (6, 7) and not atom.GetIsAromatic() for atom in atoms
        ):
            continue
        nitrogens = [
            int(idx)
            for idx in ring
            if parent.GetAtomWithIdx(int(idx)).GetAtomicNum() == 7
        ]
        if len(nitrogens) != 1:
            continue
        n_idx = nitrogens[0]
        carbons = [
            int(idx)
            for idx in ring
            if int(idx) != n_idx
            and parent.GetAtomWithIdx(int(idx)).GetTotalNumHs() == 2
        ]
        carbons.sort(
            key=lambda idx: (abs(distances.get(idx, 99) - distances.get(n_idx, 0)), idx)
        )
        for carbon_idx in carbons[:2]:
            editable = Chem.RWMol(parent)
            fluorine = Chem.Atom(9)
            fluorine.SetNoImplicit(True)
            fluorine_idx = editable.AddAtom(fluorine)
            editable.AddBond(carbon_idx, fluorine_idx, Chem.BondType.SINGLE)
            out.append(("r4_piperidine_f", "R4", editable.GetMol()))
    return out


def embed_3d(smiles: str, seed: int = 0xF00D) -> Optional[Chem.Mol]:
    """ETKDGv3 embedding + MMFF94 optimisation; returns heavy-atom mol."""
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None
    try:
        with_hydrogens = Chem.AddHs(molecule)
        params = AllChem.ETKDGv3()
        params.randomSeed = seed
        params.useRandomCoords = True
        conformer_ids = AllChem.EmbedMultipleConfs(
            with_hydrogens, numConfs=1, params=params
        )
        if not list(conformer_ids):
            return None
        try:
            AllChem.MMFFOptimizeMol(with_hydrogens, confId=int(list(conformer_ids)[0]))
        except Exception:
            pass
        result = Chem.RemoveHs(with_hydrogens)
        Chem.SanitizeMol(result)
        return result
    except Exception:
        return None


def fingerprint(molecule: Chem.Mol):
    """Morgan bit fingerprint including the dummy atom (audit convention)."""
    return AllChem.GetMorganFingerprintAsBitVect(molecule, radius=2, nBits=2048)


__all__ = [
    "RuleHit",
    "generate_candidates",
    "canonical_side_smiles",
    "to_all_carbon_graph",
    "embed_3d",
    "fingerprint",
    "alert_catalog",
    "heavy_atom_count",
]
