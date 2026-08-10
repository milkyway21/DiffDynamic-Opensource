"""Greedy multi-round medchem optimization loop."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from rdkit import Chem

from ..candidate_scorer import score_molecule
from ..mol_edit import molecule_state_hash
from ..models import (
    OPT_STATUS_OPTIMIZED,
    OPT_STATUS_REJECTED,
    OPT_STATUS_UNCHANGED,
    OptimizeResult,
    TransformApplication,
)
from .aromatize import generate_aromatization_candidates
from .catalog import load_catalog, load_tier_policy
from .constraints import ConstraintConfig, check_candidate
from .geometry import PocketClashChecker, embed_new_atoms, restore_hydrogens, transfer_conformer
from .reward import compute_properties, compute_reward, weights_from_config
from .ring_break import generate_ring_break_candidates
from .ring_contract import generate_ring_contract_candidates
from .transforms import (
    PROP_PARENT_IDX,
    TransformCandidate,
    canonical_smiles,
    enumerate_candidates,
    prepare_parent,
)

DEFAULT_TIERS = ("T0", "T1", "T2", "T3", "T4", "T5")
DEFAULTS = {
    "enable": False,
    "tiers": list(DEFAULT_TIERS),
    "max_rounds": 3,
    "max_products_per_transform": 4,
    "max_geometry_attempts": 5,
    "min_reward_gain": 0.01,
    "random_seed": 61453,
    "embed_attempts": 10,
    "embed_time_budget_s": 3.0,
    "time_budget_s": 30.0,
    "clash_cutoff": 2.2,
    "check_pocket_clash": True,
    "require_conformer": True,
    "disabled_transform_ids": [],
}


def _cfg(config: Optional[Dict[str, Any]], key: str) -> Any:
    if config and key in config:
        return config[key]
    return DEFAULTS[key]


def _state_key(mol: Chem.Mol) -> str:
    copied = Chem.Mol(mol)
    try:
        ranks = list(Chem.CanonicalRankAtoms(copied))
        for atom in copied.GetAtoms():
            atom.SetAtomMapNum(int(ranks[atom.GetIdx()]) + 1)
    except Exception:  # noqa: BLE001
        for atom in copied.GetAtoms():
            atom.SetAtomMapNum(atom.GetIdx() + 1)
    return molecule_state_hash(copied)


def _bump(counter: Dict[str, int], key: str) -> None:
    counter[key] = counter.get(key, 0) + 1


def _clean_provenance_props(mol: Chem.Mol) -> Chem.Mol:
    out = Chem.Mol(mol)
    for atom in out.GetAtoms():
        if atom.HasProp(PROP_PARENT_IDX):
            atom.ClearProp(PROP_PARENT_IDX)
        if atom.HasProp("_dd_parent_idx"):
            atom.ClearProp("_dd_parent_idx")
    return out


def _attach_geometry(
    candidate: TransformCandidate,
    current: Chem.Mol,
    clash_checker: Optional[PocketClashChecker],
    seed: int,
    embed_attempts: int,
    embed_time_budget_s: float,
    rejected: Dict[str, int],
) -> Optional[Chem.Mol]:
    mol = Chem.Mol(candidate.mol)
    if not candidate.new_atom_indices:
        if not transfer_conformer(current, mol, candidate.parent_index_map):
            # Still try if current has conformer and atom counts match
            if current.GetNumAtoms() == mol.GetNumAtoms() and current.GetNumConformers():
                if not transfer_conformer(
                    current, mol, {i: i for i in range(mol.GetNumAtoms())}
                ):
                    _bump(rejected, "conformer_transfer_failed")
                    return None
            else:
                _bump(rejected, "conformer_transfer_failed")
                return None
    else:
        embedded = embed_new_atoms(
            current,
            mol,
            candidate.parent_index_map,
            seed=seed,
            attempts=embed_attempts,
            time_budget_s=embed_time_budget_s,
        )
        if embedded is None:
            _bump(rejected, "constrained_embed_failed")
            return None
        mol = embedded

    if clash_checker is not None and clash_checker.active:
        check_indices = candidate.new_atom_indices or None
        if clash_checker.has_clash(mol, check_indices):
            _bump(rejected, "pocket_clash")
            return None
    return mol


def optimize_molecule(
    mol: Optional[Chem.Mol],
    molecule_id: str = "mol",
    config: Optional[Dict[str, Any]] = None,
    protein_path: Optional[str] = None,
    repair_config: Optional[Dict[str, Any]] = None,
) -> OptimizeResult:
    """Run the medchem optimization loop on one reconstructed molecule."""
    config = config or {}
    repair_config = repair_config or {}

    if mol is None:
        return OptimizeResult(
            molecule_id=molecule_id,
            status=OPT_STATUS_REJECTED,
            original_mol=Chem.Mol(),
            optimized_mol=None,
            reject_reason="null_input",
        )

    original = Chem.Mol(mol)
    if _cfg(config, "require_conformer") and original.GetNumConformers() == 0:
        return OptimizeResult(
            molecule_id=molecule_id,
            status=OPT_STATUS_REJECTED,
            original_mol=original,
            optimized_mol=None,
            reject_reason="no_conformer",
        )

    if not config.get("enable", True) and "enable" in config and not config["enable"]:
        # Explicit disable
        props = compute_properties(original)
        return OptimizeResult(
            molecule_id=molecule_id,
            status=OPT_STATUS_UNCHANGED,
            original_mol=original,
            optimized_mol=Chem.Mol(original),
            properties_before=props,
            properties_after=props,
        )

    tiers = [str(t).upper() for t in (_cfg(config, "tiers") or [])]
    max_rounds = int(_cfg(config, "max_rounds"))
    min_reward = float(config.get("min_reward_gain", _cfg(config, "min_reward_gain")))
    disabled = list(_cfg(config, "disabled_transform_ids") or [])
    time_budget = float(_cfg(config, "time_budget_s"))
    deadline = time.monotonic() + max(0.01, time_budget)

    constraint_cfg = ConstraintConfig.from_dict(config.get("constraints"))
    if "constraints" in config and "max_transforms_per_molecule" in (config.get("constraints") or {}):
        pass
    weights = weights_from_config(config)
    legality_cfg = repair_config if repair_config else config

    clash_checker = None
    if _cfg(config, "check_pocket_clash") and protein_path:
        clash_checker = PocketClashChecker(
            protein_path, cutoff=float(_cfg(config, "clash_cutoff"))
        )

    # Working molecule: heavy-atom parent with provenance
    current = prepare_parent(original)
    if current is None:
        return OptimizeResult(
            molecule_id=molecule_id,
            status=OPT_STATUS_REJECTED,
            original_mol=original,
            optimized_mol=None,
            reject_reason="prepare_parent_failed",
        )
    # Keep 3D from original on matching heavy atoms
    if original.GetNumConformers():
        try:
            heavy = Chem.RemoveHs(Chem.Mol(original))
            if heavy.GetNumAtoms() == current.GetNumAtoms():
                transfer_conformer(
                    heavy, current, {i: i for i in range(current.GetNumAtoms())}
                )
        except Exception:  # noqa: BLE001
            pass

    props_before = compute_properties(current)
    props_current = dict(props_before)
    try:
        legality_before = score_molecule(current, legality_cfg)
    except Exception:  # noqa: BLE001
        legality_before = 0.0

    applied: List[TransformApplication] = []
    rejected: Dict[str, int] = {}
    visited = {_state_key(current)}
    applied_counts: Dict[str, int] = {}
    reward_total_breakdown: Dict[str, float] = {}

    smarts_tiers = [t for t in tiers if t != "T0"]
    # Empty tier list = load nothing (T0-only ablation strips SMARTS tiers).
    catalog = (
        []
        if not tiers
        else load_catalog(
            enabled_tiers=smarts_tiers,
            disabled_transform_ids=disabled,
        )
    )

    tier_policy = load_tier_policy("T0")
    seed = int(_cfg(config, "random_seed"))

    for _round in range(max_rounds):
        if time.monotonic() > deadline:
            break
        if len(applied) >= constraint_cfg.max_transforms_per_molecule:
            break

        candidates: List[TransformCandidate] = []
        if "T0" in tiers:
            candidates.extend(
                generate_aromatization_candidates(
                    current, tier_policy, config=config, rejected=rejected
                )
            )
            # Prefer 7/8 → benzene/heteroarene contraction; break remains fallback.
            candidates.extend(
                generate_ring_contract_candidates(
                    current, config=config, rejected=rejected
                )
            )
            candidates.extend(
                generate_ring_break_candidates(current, config=config, rejected=rejected)
            )
        if catalog:
            candidates.extend(
                enumerate_candidates(
                    current,
                    catalog,
                    max_products_per_transform=int(_cfg(config, "max_products_per_transform")),
                    applied_counts=applied_counts,
                )
            )

        # Filter disabled
        candidates = [c for c in candidates if c.transform_id not in disabled]
        if not candidates:
            _bump(rejected, "no_applicable_transforms")
            break

        scored: List[Tuple[float, Dict[str, float], TransformCandidate, Chem.Mol]] = []
        for cand in candidates:
            if time.monotonic() > deadline:
                break
            reason = check_candidate(
                current,
                cand.mol,
                cand.tier,
                parent_index_map=cand.parent_index_map,
                n_transforms_applied=len(applied),
                cfg=constraint_cfg,
            )
            if reason is not None:
                _bump(rejected, reason)
                continue

            props_after = compute_properties(cand.mol)
            breakdown = compute_reward(
                props_current,
                props_after,
                transform_cost=cand.cost,
                weights=weights,
                transform_bonus=cand.reward_bonus,
            )
            if breakdown["total"] < min_reward:
                _bump(rejected, "reward_below_threshold")
                continue

            try:
                legality_after = score_molecule(cand.mol, legality_cfg)
            except Exception:  # noqa: BLE001
                legality_after = legality_before
            if legality_after > legality_before + 1e-6:
                _bump(rejected, "legality_regressed")
                continue

            posed = _attach_geometry(
                cand,
                current,
                clash_checker,
                seed=seed + len(applied),
                embed_attempts=int(_cfg(config, "embed_attempts")),
                embed_time_budget_s=float(_cfg(config, "embed_time_budget_s")),
                rejected=rejected,
            )
            if posed is None:
                continue
            key = _state_key(posed)
            if key in visited:
                _bump(rejected, "repeated_state")
                continue
            scored.append((breakdown["total"], breakdown, cand, posed))

        if not scored:
            break

        scored.sort(key=lambda x: -x[0])
        best_reward, best_breakdown, best_cand, best_mol = scored[0]
        visited.add(_state_key(best_mol))
        smiles_before = canonical_smiles(current)
        current = best_mol
        props_current = compute_properties(current)
        try:
            legality_before = score_molecule(current, legality_cfg)
        except Exception:  # noqa: BLE001
            pass
        applied_counts[best_cand.transform_id] = (
            applied_counts.get(best_cand.transform_id, 0) + 1
        )
        applied.append(
            TransformApplication(
                transform_id=best_cand.transform_id,
                tier=best_cand.tier,
                name=best_cand.name,
                rationale=best_cand.rationale,
                smiles_before=smiles_before,
                smiles_after=canonical_smiles(current),
                delta_heavy_atoms=best_cand.delta_heavy,
                reward=float(best_reward),
                reward_breakdown=dict(best_breakdown),
            )
        )
        for k, v in best_breakdown.items():
            if k == "total":
                continue
            reward_total_breakdown[k] = reward_total_breakdown.get(k, 0.0) + float(v)

    props_after = compute_properties(current)
    total_reward = sum(a.reward for a in applied)
    reward_total_breakdown["total"] = total_reward

    if not applied:
        return OptimizeResult(
            molecule_id=molecule_id,
            status=OPT_STATUS_UNCHANGED,
            original_mol=original,
            optimized_mol=_clean_provenance_props(current),
            applied=[],
            total_reward=0.0,
            reward_breakdown={},
            properties_before=props_before,
            properties_after=props_after,
            rejected_counts=rejected,
        )

    final = _clean_provenance_props(current)
    return OptimizeResult(
        molecule_id=molecule_id,
        status=OPT_STATUS_OPTIMIZED,
        original_mol=original,
        optimized_mol=final,
        applied=applied,
        total_reward=total_reward,
        reward_breakdown=reward_total_breakdown,
        properties_before=props_before,
        properties_after=props_after,
        rejected_counts=rejected,
    )


def optimize_molecules(mols, config=None, protein_path=None, repair_config=None):
    return [
        optimize_molecule(
            mol,
            molecule_id=str(i),
            config=config,
            protein_path=protein_path,
            repair_config=repair_config,
        )
        for i, mol in enumerate(mols)
    ]
