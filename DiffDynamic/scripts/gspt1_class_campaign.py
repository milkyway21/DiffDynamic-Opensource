#!/usr/bin/env python3
"""Prepare or run the trusted three-class GSPT1 scaffold campaign."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from rdkit import Chem

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.gspt1_class_audit import audit_class  # noqa: E402
from scripts.gspt1_scaffold_exact_loop import _run_job  # noqa: E402
from utils.gspt1_class_setup import (  # noqa: E402
    CLASS_SPECS,
    EXCLUDED_REFERENCE_INDICES,
    Gspt1ClassSpec,
    prepare_class_profiles,
)
from utils.gspt1_scaffold_prior import iter_sdf_molecules  # noqa: E402
from utils.scaffold_sites import (  # noqa: E402
    build_extra_atom_positions,
    load_or_extract_attachment_sites,
)


DEFAULT_CONFIG = REPO_ROOT / "configs/gspt1_scaffold_exact.yml"
DEFAULT_REFERENCE = Path("/data/ye/sdf/GSPT1.sdf")
DEFAULT_NATIVE = Path(
    "/data/ye/e-drug-lab/Scientist_In_E-Drug-Lab/runs/"
    "molecular_glue_structures_ikzf2_gspt1_20260803_235811/"
    "02_gspt1_or_gstp1_molecular_glue/ligand/"
    "5HXB_85C_C_502_native.sdf"
)
DEFAULT_PROTEIN = Path(
    "/data/ye/e-drug-lab/Scientist_In_E-Drug-Lab/runs/"
    "molecular_glue_structures_ikzf2_gspt1_20260803_235811/"
    "02_gspt1_or_gstp1_molecular_glue/receptor/5HXB_receptor_clean.pdb"
)
DEFAULT_ROOT = Path(
    "/data/zhang/Ye/DiffDynamic_outputs/hsvpol/"
    "molglue_ikzf2_gspt1/diffdynamic/gspt1_scaffold_repaint_v3"
)

GEOMETRY_MODES = (
    "pocket_aware_template",
    "strict_anchor_gaussian",
    "strict_fragment_gaussian",
    "fragment_gaussian",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _write_yaml(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)


def _restrict_profile_to_counts(
    profile: dict[str, Any],
    allowed_counts: tuple[int, ...],
) -> dict[str, Any]:
    """Keep only complete coarse-prior records for selected atom counts."""
    allowed = {int(value) for value in allowed_counts}
    available = {int(value) for value in profile.get("n_extra_values", [])}
    if not allowed or not allowed <= available:
        raise ValueError(
            f"count focus {sorted(allowed)} is not a subset of "
            f"profile sizes {sorted(available)}"
        )

    restricted = copy.deepcopy(profile)
    values = [
        int(value)
        for value in profile.get("n_extra_values", [])
        if int(value) in allowed
    ]
    weights = [
        float(weight)
        for value, weight in zip(
            profile.get("n_extra_values", []),
            profile.get("n_extra_weights", []),
        )
        if int(value) in allowed
    ]
    restricted["n_extra_values"] = values
    restricted["n_extra_weights"] = weights

    for key in ("allocation_patterns", "extra_type_patterns", "fragment_patterns"):
        restricted[key] = [
            record
            for record in profile.get(key, [])
            if int(record.get("n_extra", -1)) in allowed
        ]

    # Keep the aggregate prior consistent with the selected size branch.  The
    # sampler uses these fields for the weak element/aromatic initialization.
    element_counts: dict[str, float] = {}
    aromatic_counts: dict[str, float] = {}
    exit_counts: dict[str, float] = {}
    for record in restricted["extra_type_patterns"]:
        weight = float(record.get("weight", 1.0))
        for raw_key, count in record.get("class_counts", {}).items():
            symbol, aromatic = str(raw_key).rsplit("|", 1)
            weighted_count = float(count) * weight
            element_counts[symbol] = (
                element_counts.get(symbol, 0.0) + weighted_count
            )
            if int(aromatic):
                aromatic_counts[raw_key] = (
                    aromatic_counts.get(raw_key, 0.0) + weighted_count
                )
    for record in restricted["allocation_patterns"]:
        weight = float(record.get("weight", 1.0))
        for slot, count in record.get("site_counts", {}).items():
            slot_key = str(int(slot))
            exit_counts[slot_key] = (
                exit_counts.get(slot_key, 0.0) + float(count) * weight
            )
    restricted["reference_extra_element_counts"] = element_counts
    restricted["reference_extra_aromatic_element_counts"] = aromatic_counts
    restricted["exit_site_weights"] = exit_counts
    restricted["count_focus"] = sorted(allowed)
    return restricted


def _effective_class_specs(
    no_f_count_values: tuple[int, ...],
) -> tuple[Gspt1ClassSpec, ...]:
    """Return class specs with an optional no-F size-focused branch."""
    if not no_f_count_values:
        return CLASS_SPECS
    return tuple(
        replace(spec, allowed_n_extra=no_f_count_values)
        if spec.name == "no_f"
        else spec
        for spec in CLASS_SPECS
    )


def _parse_count_values(value: str) -> tuple[int, ...]:
    """Parse a comma-separated positive integer count override."""
    if not value.strip():
        return ()
    try:
        parsed = tuple(sorted({int(item.strip()) for item in value.split(",")}))
    except ValueError as exc:
        raise ValueError(f"invalid count list: {value!r}") from exc
    if not parsed or any(item <= 0 for item in parsed):
        raise ValueError(f"count list must contain positive integers: {value!r}")
    return parsed


def _parse_geometry_mode(value: str) -> str:
    mode = str(value).strip().lower()
    if mode not in GEOMETRY_MODES:
        raise ValueError(
            f"invalid geometry mode {value!r}; choose from "
            f"{', '.join(GEOMETRY_MODES)}"
        )
    return mode


def _parse_geometry_modes(
    value: str,
    default_mode: str,
) -> dict[str, str]:
    """Parse optional per-class geometry overrides.

    The legacy single-mode flag remains the default for every class.  An
    override string such as ``no_f=pocket_aware_template,f_main=strict_fragment_gaussian``
    only changes the named classes, which keeps old manifests reproducible.
    """
    modes = {
        spec.name: _parse_geometry_mode(default_mode)
        for spec in CLASS_SPECS
    }
    if not str(value).strip():
        return modes
    valid_names = {spec.name for spec in CLASS_SPECS}
    seen: set[str] = set()
    for item in str(value).split(","):
        name, separator, raw_mode = item.partition("=")
        name = name.strip()
        if not separator or name not in valid_names:
            raise ValueError(
                f"invalid geometry override {item!r}; use class=mode for "
                f"classes {sorted(valid_names)}"
            )
        if name in seen:
            raise ValueError(f"duplicate geometry override for class {name!r}")
        seen.add(name)
        modes[name] = _parse_geometry_mode(raw_mode)
    return modes


def build_class_config(
    base: dict[str, Any],
    spec: Gspt1ClassSpec,
    profile_path: Path,
    profile: dict,
    *,
    seed: int,
    samples: int,
    geometry_mode: str = "pocket_aware_template",
) -> dict[str, Any]:
    """Create one fully pinned class config without changing de novo modes."""
    geometry_mode = _parse_geometry_mode(geometry_mode)
    config = copy.deepcopy(base)
    sample = config.setdefault("sample", {})
    sample["seed"] = int(seed)
    scaffold = sample.setdefault("scaffold", {})
    scaffold.update({
        "enable": True,
        "mode": "dynamic_locked",
        "num_samples": int(samples),
        "after_dynamic": False,
        "scaffold_source": "custom",
        "scaffold_smarts": spec.scaffold_smarts,
        "fix_scaffold_pos": True,
        "fix_scaffold_type": True,
    })
    sites = scaffold.setdefault("murcko_sites", {})
    profile_slots = sorted({
        int(slot)
        for pattern in profile["allocation_patterns"]
        for slot in pattern["site_counts"]
    })
    sites.update({
        "reference_exit_profile": str(profile_path),
        "reference_exit_slots": profile_slots,
        "reference_exit_only": True,
        "site_selection_mode": "reference_joint",
        "per_site_count_mode": "split",
        "site_budget_mode": "requested",
        "max_per_site": 30,
        "max_active_sites": max(len(profile_slots), 1),
        "overflow_mode": "cap",
        "preserve_zero_allocation_sidechains": False,
        "jitter_mode": geometry_mode,
        "jitter_std": 0.08,
        # These are opt-in scaffold geometry modes. They are consumed only by
        # build_extra_atom_positions and do not alter de novo sampling.
        "strict_anchor_gaussian": geometry_mode == "strict_anchor_gaussian",
        "strict_fragment_gaussian": geometry_mode == "strict_fragment_gaussian",
        # Soft fragment mode reuses the fragment-shaped initial cloud, while
        # leaving both its coordinates and element classes to diffusion.
        "fragment_gaussian": geometry_mode == "fragment_gaussian",
        "fragment_prior_profile": (
            str(profile_path)
            if geometry_mode in {
                "strict_fragment_gaussian",
                "fragment_gaussian",
            } else None
        ),
        "strict_fragment_center_offset": 1.55,
        "fragment_center_spacing": 2.15,
        "fragment_center_lateral_sigma": 0.45,
        "fragment_attachment_distance": 1.55,
        "fragment_attachment_spacing": 2.15,
        "fragment_geometry_sigma": 0.16,
        "fragment_ring_radius": 1.38,
        "original_min_dist": 1.2,
        "original_max_dist": 10.0,
        "pocket_min_protein_dist": 1.5,
        "pocket_min_point_dist": 0.9,
        "pocket_max_anchor_dist": 10.0,
        "pocket_growth_candidates": 128,
        "pocket_growth_step_min": 1.25,
        "pocket_growth_step_max": 1.55,
        "pocket_relax_iterations": 120,
        "reference_extra_type_prior_strength": 1.0,
        "reference_extra_type_prior_mode": "quota_random",
    })
    grow = scaffold.setdefault("grow", {})
    grow.update({
        "seed": int(seed),
        "n_extra_mode": "reference_size_prior",
        "reference_size_values": list(profile["n_extra_values"]),
        "reference_size_weights": list(profile["n_extra_weights"]),
        "n_extra_min": min(profile["n_extra_values"]),
        "n_extra_max": max(profile["n_extra_values"]),
        "n_extra_min_clamp": min(profile["n_extra_values"]),
        "n_extra_max_clamp": max(profile["n_extra_values"]),
        # Start from a valid q(x_999 | x_0) state. RePaint injects the core
        # back at every reverse step in the same noise coordinate system.
        "start_t": 999,
        "forward_noise_init": True,
        # Added-atom positions remain free; type identities receive only a
        # weak, randomly permuted multiset prior.
        "extra_anchor_strength": 0.0,
        # Weakly condition the element multiset without hard-locking identities.
        "extra_type_anchor_strength": 0.2,
    })
    if geometry_mode == "fragment_gaussian":
        # This is a scaffold-only ablation: retain the exact atom count but
        # let the normal categorical reverse process choose added elements.
        scaffold["extra_atom_type_mode"] = "diffuse"
        scaffold["lock_extra_atom_types"] = False
        grow["extra_type_anchor_strength"] = 0.0
    dynamic_refine = sample.setdefault("dynamic", {}).setdefault("refine", {})
    dynamic_refine.update({
        "max_grad_fusion_iterations": 30,
        "preserve_schedule_endpoint": True,
    })
    targetdiff = sample.setdefault("targetdiff_baseline_refine", {})
    targetdiff.update({
        "enable": True,
        # Inclusive t=29..0 gives exactly 30 baseline repair updates.
        "start_t": 29,
        "lock_prefix": "types_and_pos",
        "after_scaffold_init_only": False,
        "refine_batch_size": 100,
    })
    return config


def _ligand_for_spec(
    spec: Gspt1ClassSpec,
    native_ligand: Path,
    f_ligand: Path,
) -> Path:
    return native_ligand if spec.ligand_kind == "native" else f_ligand


def _protein_positions(path: Path) -> np.ndarray:
    positions = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        if line[76:78].strip() == "H":
            continue
        try:
            positions.append([
                float(line[30:38]),
                float(line[38:46]),
                float(line[46:54]),
            ])
        except ValueError:
            continue
    if not positions:
        raise ValueError(f"no protein coordinates found in {path}")
    return np.asarray(positions, dtype=np.float64)


def geometry_preflight(
    spec: Gspt1ClassSpec,
    ligand_path: Path,
    protein_path: Path,
    config: dict[str, Any],
    *,
    max_attempts_per_size: int = 128,
) -> dict:
    """Verify exact site budgets and non-clashing initial point clouds."""
    molecule = next(iter_sdf_molecules(ligand_path), None)
    if molecule is None:
        raise ValueError(f"cannot parse ligand: {ligand_path}")
    pattern = Chem.MolFromSmarts(spec.scaffold_smarts)
    match = molecule.GetSubstructMatch(pattern) if pattern is not None else ()
    if not match:
        raise ValueError(f"{spec.name}: scaffold does not match ligand")
    scaffold_indices = sorted(int(index) for index in match)
    conformer = molecule.GetConformer()
    ligand_positions = np.asarray([
        list(conformer.GetAtomPosition(index))
        for index in range(molecule.GetNumAtoms())
    ], dtype=np.float64)
    sites_cfg = config["sample"]["scaffold"]["murcko_sites"]
    sites_cfg = dict(sites_cfg)
    sites_cfg["save_json"] = False
    strict_geometry_only = bool(
        sites_cfg.get("strict_anchor_gaussian", False)
        or sites_cfg.get("strict_fragment_gaussian", False)
        or sites_cfg.get("fragment_gaussian", False)
    )
    sites = load_or_extract_attachment_sites(
        molecule,
        scaffold_indices,
        "custom",
        ligand_positions,
        None,
        ligand_path.stem,
        sites_cfg,
    )
    protein_np = _protein_positions(protein_path)
    protein_tensor = torch.tensor(protein_np, dtype=torch.float32)
    center = protein_tensor.mean(dim=0)
    minimum_clearance = float("inf")
    observed_counts: set[int] = set()
    observed_patterns: set[tuple[tuple[int, int], ...]] = set()
    required_patterns = {
        tuple(sorted(
            (int(slot), int(count))
            for slot, count in pattern_record["site_counts"].items()
        ))
        for pattern_record in config["_profile"]["allocation_patterns"]
    }
    attempts_by_size: dict[str, int] = {}
    for n_extra in spec.allowed_n_extra:
        required_for_size = {
            pattern_record for pattern_record in required_patterns
            if sum(count for _, count in pattern_record) == n_extra
        }
        for attempt in range(max_attempts_per_size):
            positions, meta = build_extra_atom_positions(
                n_extra,
                sites,
                sites_cfg,
                center,
                "cpu",
                rng=np.random.default_rng(10_000 * n_extra + attempt),
                protein_positions=protein_tensor,
            )
            observed_counts.add(int(positions.shape[0]))
            allocation = tuple(sorted(
                (
                    int(record["profile_slot"]),
                    int(record["count"]),
                )
                for record in meta["site_allocation"]
            ))
            observed_patterns.add(allocation)
            distances = np.linalg.norm(
                positions.numpy()[:, None, :] - protein_np[None, :, :],
                axis=2,
            )
            minimum_clearance = min(
                minimum_clearance, float(distances.min())
            )
            if attempt >= 3 and required_for_size <= observed_patterns:
                attempts_by_size[str(n_extra)] = attempt + 1
                break
        else:
            attempts_by_size[str(n_extra)] = max_attempts_per_size
    if observed_counts != set(spec.allowed_n_extra):
        raise ValueError(
            f"{spec.name}: observed counts {sorted(observed_counts)} do not "
            f"match {list(spec.allowed_n_extra)}"
        )
    if not required_patterns <= observed_patterns:
        raise ValueError(
            f"{spec.name}: preflight missed allocation patterns "
            f"{sorted(required_patterns - observed_patterns)}"
        )
    threshold = float(sites_cfg["pocket_min_protein_dist"])
    if minimum_clearance + 1e-5 < threshold and not strict_geometry_only:
        raise ValueError(
            f"{spec.name}: minimum clearance {minimum_clearance:.3f} < "
            f"{threshold:.3f} A"
        )
    return {
        "class_name": spec.name,
        "n_scaffold": len(scaffold_indices),
        "observed_n_extra": sorted(observed_counts),
        "observed_allocation_patterns": [
            {str(slot): count for slot, count in pattern_record}
            for pattern_record in sorted(observed_patterns)
        ],
        "minimum_protein_clearance": minimum_clearance,
        "protein_clearance_constraint_enforced": not strict_geometry_only,
        "attempts_by_size": attempts_by_size,
    }


def prepare_campaign(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root)
    reference_sdf = Path(args.reference_sdf)
    native_ligand = Path(args.native_ligand)
    protein = Path(args.protein)
    class_specs = _effective_class_specs(args.no_f_count_values)
    geometry_modes = dict(args.geometry_modes)
    base_config = _load_yaml(Path(args.config))
    f_ligand, profile_paths, profiles = prepare_class_profiles(
        reference_sdf, native_ligand, root
    )
    configs: dict[str, Path] = {}
    preflight = {}
    for offset, spec in enumerate(class_specs):
        profile = profiles[spec.name]
        if spec.name == "no_f" and args.no_f_count_values:
            profile = _restrict_profile_to_counts(
                profile, args.no_f_count_values
            )
            profiles[spec.name] = profile
            write_path = profile_paths[spec.name]
            write_path.write_text(
                json.dumps(profile, indent=2, ensure_ascii=True) + "\n",
                encoding="utf-8",
            )
        config = build_class_config(
            base_config,
            spec,
            profile_paths[spec.name],
            profile,
            seed=int(args.start_seed) + offset,
            samples=int(args.samples),
            geometry_mode=geometry_modes[spec.name],
        )
        # Private preflight-only payload; never serialized into sampling YAML.
        preflight_config = copy.deepcopy(config)
        preflight_config["_profile"] = profile
        preflight[spec.name] = geometry_preflight(
            spec,
            _ligand_for_spec(spec, native_ligand, f_ligand),
            protein,
            preflight_config,
        )
        config_path = root / "configs" / f"{spec.name}.yml"
        _write_yaml(config, config_path)
        configs[spec.name] = config_path
    (root / "preflight.json").write_text(
        json.dumps(preflight, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "campaign": "GSPT1 trusted discrete class campaign",
        "git_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip(),
        "created_at_unix": time.time(),
        "reference_sdf": str(reference_sdf),
        "reference_sdf_sha256": _sha256(reference_sdf),
        "native_ligand": str(native_ligand),
        "native_ligand_sha256": _sha256(native_ligand),
        "f_core_ligand": str(f_ligand),
        "f_core_ligand_sha256": _sha256(f_ligand),
        "protein": str(protein),
        "protein_sha256": _sha256(protein),
        "excluded_reference_indices": list(EXCLUDED_REFERENCE_INDICES),
        "gpu_mapping": {
            str(gpu): spec.name
            for gpu, spec in zip(args.gpus, class_specs)
        },
        "allowed_n_extra": {
            spec.name: list(spec.allowed_n_extra)
            for spec in class_specs
        },
        "count_focus": {
            "no_f": list(args.no_f_count_values),
        },
        "geometry_mode": args.geometry_mode,
        "geometry_modes": geometry_modes,
        "profiles": {name: str(path) for name, path in profile_paths.items()},
        "configs": {name: str(path) for name, path in configs.items()},
        "preflight": preflight,
        "policy": {
            "original_scaffold_position_locked": True,
            "original_scaffold_type_locked": True,
            "added_position_locked": False,
            "added_type_locked": all(
                mode in {
                    "strict_anchor_gaussian",
                    "strict_fragment_gaussian",
                }
                for mode in geometry_modes.values()
            ),
            "added_type_soft_anchor_strength": (
                1.0
                if all(
                    mode in {
                        "strict_anchor_gaussian",
                        "strict_fragment_gaussian",
                    }
                    for mode in geometry_modes.values()
                )
                else 0.0
                if all(
                    mode == "fragment_gaussian"
                    for mode in geometry_modes.values()
                )
                else None
            ),
            "added_type_locked_by_class": {
                name: mode in {
                    "strict_anchor_gaussian",
                    "strict_fragment_gaussian",
                }
                for name, mode in geometry_modes.items()
            },
            "added_type_soft_anchor_strength_by_class": {
                name: (
                    1.0 if mode in {
                        "strict_anchor_gaussian",
                        "strict_fragment_gaussian",
                    }
                    else 0.0 if mode == "fragment_gaussian" else 0.2
                )
                for name, mode in geometry_modes.items()
            },
            "diffusion_start_t": 999,
            "diffdynamic_refine_max_iterations": 30,
            "targetdiff_reverse_steps": 30,
            "reference_target_graph_used": False,
            "post_generation_graph_editing": False,
            "scaffold_attachment_reconstruction": "site_anchor_first_extra",
            "vina": False,
        },
    }
    (root / "campaign_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return {
        "root": root,
        "reference_sdf": reference_sdf,
        "native_ligand": native_ligand,
        "f_ligand": f_ligand,
        "protein": protein,
        "base_config": base_config,
        "profile_paths": profile_paths,
        "profiles": profiles,
        "class_specs": class_specs,
        "configs": configs,
        "manifest": manifest,
    }


def run_campaign(args: argparse.Namespace, prepared: dict[str, Any]) -> dict:
    root = prepared["root"]
    class_specs = prepared["class_specs"]
    state_path = root / "state.json"
    state = {}
    if args.resume and state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
    round_index = int(state.get("round_index", 0))
    next_job_id = int(state.get("next_job_id", 0))
    deadline = time.time() + float(args.hours) * 3600.0
    final_summary: dict[str, Any] = {}

    while time.time() < deadline:
        if args.max_rounds and round_index >= args.max_rounds:
            break
        jobs = []
        for gpu, spec in zip(args.gpus, class_specs):
            job_id = next_job_id
            next_job_id += 1
            seed = int(args.start_seed) + job_id
            variant_root = root / "rounds" / spec.name / f"run_{round_index:04d}"
            config = build_class_config(
                prepared["base_config"],
                spec,
                prepared["profile_paths"][spec.name],
                prepared["profiles"][spec.name],
                seed=seed,
                samples=int(args.samples),
                geometry_mode=args.geometry_modes[spec.name],
            )
            config_path = variant_root / "config.yml"
            _write_yaml(config, config_path)
            ligand = _ligand_for_spec(
                spec, prepared["native_ligand"], prepared["f_ligand"]
            )
            jobs.append((gpu, spec, job_id, variant_root, config_path, ligand))
        state_path.write_text(
            json.dumps({
                "round_index": round_index + 1,
                "next_job_id": next_job_id,
                "last_schedule": [
                    {"gpu": gpu, "class": spec.name, "job_id": job_id}
                    for gpu, spec, job_id, _, _, _ in jobs
                ],
            }, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        results = []
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(
                    _run_job,
                    root=root,
                    variant_root=variant_root,
                    config_path=config_path,
                    profile_path=prepared["profile_paths"][spec.name],
                    use_profile=True,
                    gpu=gpu,
                    job_id=job_id,
                    base_seed=int(args.start_seed),
                    samples=int(args.samples),
                    protein=prepared["protein"],
                    native_ligand=ligand,
                    deadline=deadline,
                )
                for gpu, spec, job_id, variant_root, config_path, ligand in jobs
            ]
            for future in as_completed(futures):
                results.append(future.result())
        with (root / "job_results.jsonl").open("a", encoding="utf-8") as handle:
            for result in results:
                handle.write(json.dumps(result, ensure_ascii=True) + "\n")

        class_summaries = {}
        for spec in class_specs:
            class_root = root / "rounds" / spec.name / f"run_{round_index:04d}"
            class_summaries[spec.name] = audit_class(
                [class_root],
                prepared["reference_sdf"],
                spec.reference_indices,
                spec.scaffold_smarts,
                spec.allowed_n_extra,
                class_root / "audit",
                class_name=spec.name,
            )
        final_summary = {
            "round": round_index,
            "results": results,
            "classes": class_summaries,
            "exact_reachable_count": sum(
                summary["exact_reachable_count"]
                for summary in class_summaries.values()
            ),
        }
        (root / "latest_summary.json").write_text(
            json.dumps(final_summary, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(final_summary, ensure_ascii=True), flush=True)
        if final_summary["exact_reachable_count"] > 0:
            (root / "EXACT_MATCH_FOUND").write_text(
                json.dumps(final_summary, indent=2, ensure_ascii=True) + "\n",
                encoding="utf-8",
            )
            break
        round_index += 1
    final_summary["finished_at_unix"] = time.time()
    final_summary["deadline_reached"] = time.time() >= deadline
    (root / "final_summary.json").write_text(
        json.dumps(final_summary, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return final_summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="start GPU generation")
    parser.add_argument("--hours", type=float, default=10.0)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--max-rounds", type=int, default=0)
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--reference-sdf", default=str(DEFAULT_REFERENCE))
    parser.add_argument("--native-ligand", default=str(DEFAULT_NATIVE))
    parser.add_argument("--protein", default=str(DEFAULT_PROTEIN))
    parser.add_argument("--start-seed", type=int, default=20280000)
    parser.add_argument("--gpus", default="3,4,5")
    parser.add_argument(
        "--no-f-count-values",
        default="",
        help="optional comma-separated no-F atom counts, e.g. 14",
    )
    parser.add_argument(
        "--geometry-mode",
        default="pocket_aware_template",
        choices=GEOMETRY_MODES,
        help=(
            "scaffold-only initial cloud mode; strict_fragment_gaussian "
            "uses coarse fragment/ring priors without target-side coordinates; "
            "fragment_gaussian uses the same cloud with free added elements"
        ),
    )
    parser.add_argument(
        "--geometry-modes",
        default="",
        help=(
            "optional comma-separated per-class overrides, for example "
            "no_f=pocket_aware_template,f_main=strict_fragment_gaussian"
        ),
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.gpus = [
        int(value.strip()) for value in args.gpus.split(",") if value.strip()
    ]
    if len(args.gpus) != len(CLASS_SPECS):
        parser.error("--gpus must contain exactly three GPU ids")
    try:
        args.no_f_count_values = _parse_count_values(
            args.no_f_count_values
        )
        args.geometry_mode = _parse_geometry_mode(args.geometry_mode)
        args.geometry_modes = _parse_geometry_modes(
            args.geometry_modes,
            args.geometry_mode,
        )
    except ValueError as exc:
        parser.error(str(exc))
    prepared = prepare_campaign(args)
    print(json.dumps(prepared["manifest"], indent=2, ensure_ascii=True))
    if not args.run:
        print("PREPARED_ONLY: pass --run to start GPU generation")
        return
    summary = run_campaign(args, prepared)
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
