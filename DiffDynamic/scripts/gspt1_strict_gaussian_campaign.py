#!/usr/bin/env python3
"""Run the strict, scaffold-only five-lane GSPT1 campaign.

The campaign has two no-F lanes, two F-main lanes, and one F-large lane.
Every lane uses a class-specific element-count profile, a scaffold-local
Gaussian anchor cloud, full t=999 DiffDynamic sampling, no TargetDiff/Vina,
and a no-docking similarity audit after reconstruction.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from rdkit import Chem

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.gspt1_class_audit import audit_class  # noqa: E402
from scripts.gspt1_allc_tanimoto_audit import audit as audit_allc  # noqa: E402
from scripts.gspt1_scaffold_exact_loop import _run_job  # noqa: E402
from scripts.gspt1_smiles_audit import audit  # noqa: E402
from utils.gspt1_class_setup import (  # noqa: E402
    CLASS_SPECS,
    Gspt1ClassSpec,
    prepare_class_profiles,
)
from utils.gspt1_scaffold_prior import (  # noqa: E402
    SUPPORTED_GENERATION_ELEMENTS,
    iter_sdf_molecules,
)
from utils.scaffold_sites import (  # noqa: E402
    build_extra_atom_positions,
    load_or_extract_attachment_sites,
)


DEFAULT_CONFIG = REPO_ROOT / "configs/gspt1_scaffold_exact.yml"
DEFAULT_REFERENCE = Path("/data/ye/sdf/GSPT1.sdf")
DEFAULT_NATIVE = Path(
    "/data/ye/e-drug-lab/Scientist_In_E-Drug-Lab/runs/"
    "molecular_glue_structures_ikzf2_gspt1_20260803_235811/"
    "02_gspt1_or_gstp1_molecular_glue/ligand/5HXB_85C_C_502_native.sdf"
)
DEFAULT_PROTEIN = Path(
    "/data/ye/e-drug-lab/Scientist_In_E-Drug-Lab/runs/"
    "molecular_glue_structures_ikzf2_gspt1_20260803_235811/"
    "02_gspt1_or_gstp1_molecular_glue/receptor/5HXB_receptor_clean.pdb"
)
DEFAULT_ROOT = Path(
    "/data/zhang/Ye/DiffDynamic_outputs/hsvpol/"
    "molglue_ikzf2_gspt1/diffdynamic/gspt1_scaffold_strict_gaussian_v1"
)


@dataclass(frozen=True)
class LaneSpec:
    name: str
    class_name: str
    gpu: int
    sigma: float
    center_offset: float
    tangent_shift: float
    binormal_shift: float
    slot_offsets: dict[str, float]
    slot_tangent_shifts: dict[str, float]
    slot_binormal_shifts: dict[str, float]


# Same class profiles are deliberately replicated with different local means.
# The profile still determines which scaffold exits and atom counts are used.
LANES = (
    LaneSpec(
        "no_f_anchor0_a", "no_f", 1, 0.72, 1.45, 0.00, 0.00,
        {"0": 1.45, "10": 1.45}, {"0": 0.00, "10": 0.25},
        {"0": 0.00, "10": 0.00},
    ),
    LaneSpec(
        "no_f_anchor0_b", "no_f", 2, 0.86, 2.05, 0.80, -0.35,
        {"0": 2.05, "10": 1.75}, {"0": 0.80, "10": 0.50},
        {"0": -0.35, "10": -0.20},
    ),
    LaneSpec(
        "f_main_anchor1_a", "f_main", 3, 0.70, 1.40, -0.25, 0.35,
        {"1": 1.40}, {"1": -0.25}, {"1": 0.35},
    ),
    LaneSpec(
        "f_main_anchor1_b", "f_main", 4, 0.88, 2.10, -0.85, -0.25,
        {"1": 2.10}, {"1": -0.85}, {"1": -0.25},
    ),
    LaneSpec(
        "f_large_anchor17", "f_large", 5, 0.82, 1.55, 0.45, 0.00,
        {"1": 1.55, "7": 1.20}, {"1": 0.45, "7": 0.20},
        {"1": 0.00, "7": 0.10},
    ),
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


def _get_class_spec(name: str) -> Gspt1ClassSpec:
    for spec in CLASS_SPECS:
        if spec.name == name:
            return spec
    raise KeyError(name)


def _ligand_for_spec(
    spec: Gspt1ClassSpec,
    native_ligand: Path,
    f_ligand: Path,
) -> Path:
    return native_ligand if spec.ligand_kind == "native" else f_ligand


def build_strict_config(
    base: dict[str, Any],
    spec: Gspt1ClassSpec,
    profile_path: Path,
    profile: dict[str, Any],
    lane: LaneSpec,
    *,
    seed: int,
    samples: int,
    fragment_gaussian: bool = False,
    unlock_extra_types: bool = False,
) -> dict[str, Any]:
    """Build a strict configuration without changing de novo settings."""
    config = copy.deepcopy(base)
    sample = config.setdefault("sample", {})
    sample["seed"] = int(seed)
    sample["num_steps"] = 1000
    sample["mode"] = "dynamic"
    sample["pos_only"] = False

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
        "lock_extra_atom_types": not unlock_extra_types,
        "extra_atom_type_mode": (
            "diffuse" if unlock_extra_types else "locked"
        ),
        "allow_forced_attachment_bonds": False,
        "scaffold_reconstruction_covalent_factor": 1.3,
        "save_dynamic_before_scaffold": False,
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
        "site_budget_mode": "requested",
        "per_site_count_mode": "split",
        "max_per_site": max(max(int(v) for v in profile["n_extra_values"]), 30),
        "max_active_sites": max(len(profile_slots), 1),
        "overflow_mode": "cap",
        "preserve_zero_allocation_sidechains": False,
        "strict_anchor_gaussian": not fragment_gaussian,
        "strict_fragment_gaussian": fragment_gaussian,
        "jitter_mode": (
            "strict_fragment_gaussian"
            if fragment_gaussian else "strict_anchor_gaussian"
        ),
        "jitter_std": lane.sigma,
        "strict_gaussian_sigma": lane.sigma,
        "strict_gaussian_center_offset": lane.center_offset,
        "strict_gaussian_tangent_shift": lane.tangent_shift,
        "strict_gaussian_binormal_shift": lane.binormal_shift,
        "strict_gaussian_slot_offsets": lane.slot_offsets,
        "strict_gaussian_slot_tangent_shifts": lane.slot_tangent_shifts,
        "strict_gaussian_slot_binormal_shifts": lane.slot_binormal_shifts,
        "reference_extra_type_prior_strength": (
            0.0 if unlock_extra_types else 1.0
        ),
        "reference_extra_type_prior_mode": (
            "uniform" if unlock_extra_types else "quota_random"
        ),
        "fragment_prior_profile": str(profile_path),
        "fragment_geometry_sigma": 0.16,
        "fragment_ring_radius": 1.38,
        "fragment_center_spacing": 0.85,
        "fragment_center_lateral_sigma": 0.25,
        "fragment_attachment_distance": 1.55,
        "fragment_attachment_spacing": 0.85,
        "strict_fragment_center_offset": lane.center_offset,
        "save_json": True,
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
        "start_t": 999,
        "forward_noise_init": True,
        "extra_anchor_strength": 0.0,
        "extra_type_anchor_strength": 0.0 if unlock_extra_types else 1.0,
    })

    dynamic = sample.setdefault("dynamic", {})
    dynamic.setdefault("large_step", {})["max_grad_fusion_iterations"] = 30
    dynamic.setdefault("large_step", {})["preserve_schedule_endpoint"] = True
    dynamic.setdefault("refine", {})["max_grad_fusion_iterations"] = 30
    dynamic.setdefault("refine", {})["preserve_schedule_endpoint"] = True

    # TargetDiff baseline can change the newly generated atoms after the
    # DiffDynamic chain, so strict mode excludes it entirely.
    sample.setdefault("targetdiff_baseline_refine", {}).update({
        "enable": False,
        "after_scaffold_init_only": False,
    })
    sample.setdefault("optimization", {})["enable"] = False
    return config


def _first_molecule(path: Path) -> Chem.Mol:
    molecule = next(iter_sdf_molecules(path), None)
    if molecule is None:
        raise ValueError(f"cannot parse ligand: {path}")
    return molecule


def _scaffold_indices(molecule: Chem.Mol, smarts: str) -> list[int]:
    pattern = Chem.MolFromSmarts(smarts)
    match = molecule.GetSubstructMatch(pattern) if pattern is not None else ()
    if not match:
        raise ValueError(f"scaffold does not match SMARTS: {smarts}")
    return sorted(int(index) for index in match)


def strict_preflight(
    spec: Gspt1ClassSpec,
    ligand_path: Path,
    profile: dict[str, Any],
    config: dict[str, Any],
    *,
    unlock_extra_types: bool = False,
) -> dict[str, Any]:
    """Verify count/allocation coverage and the no-template geometry contract."""
    molecule = _first_molecule(ligand_path)
    indices = _scaffold_indices(molecule, spec.scaffold_smarts)
    conformer = molecule.GetConformer()
    positions = np.asarray([
        list(conformer.GetAtomPosition(index))
        for index in range(molecule.GetNumAtoms())
    ], dtype=np.float64)
    sites_cfg = dict(config["sample"]["scaffold"]["murcko_sites"])
    scaffold_cfg = config["sample"]["scaffold"]
    grow_cfg = scaffold_cfg["grow"]
    type_mode = str(
        scaffold_cfg.get("extra_atom_type_mode", "auto")
    ).strip().lower()
    if unlock_extra_types:
        if type_mode not in {"diffuse", "unlocked", "free", "model"}:
            raise ValueError(
                f"{spec.name}: extra atom type mode is not diffuse"
            )
    elif not bool(scaffold_cfg.get("lock_extra_atom_types", False)):
        raise ValueError(
            f"{spec.name}: extra atom types are not explicitly locked"
        )
    if bool(scaffold_cfg.get("allow_forced_attachment_bonds", True)):
        raise ValueError(f"{spec.name}: forced attachment bonds must be disabled")
    if float(grow_cfg.get("extra_anchor_strength", 1.0)) != 0.0:
        raise ValueError(f"{spec.name}: extra coordinate mask must be zero")
    expected_type_mask = 0.0 if unlock_extra_types else 1.0
    if float(grow_cfg.get("extra_type_anchor_strength", 0.0)) != expected_type_mask:
        raise ValueError(
            f"{spec.name}: extra type mask must be {expected_type_mask:g}"
        )
    sites_cfg["save_json"] = False
    sites = load_or_extract_attachment_sites(
        molecule,
        indices,
        "custom",
        positions,
        None,
        ligand_path.stem,
        sites_cfg,
    )
    expected_slots = {
        int(slot)
        for pattern in profile["allocation_patterns"]
        for slot in pattern["site_counts"]
    }
    observed_slots = {int(site["profile_slot"]) for site in sites}
    if observed_slots != expected_slots:
        raise ValueError(
            f"{spec.name}: strict slots {sorted(observed_slots)} != "
            f"profile {sorted(expected_slots)}"
        )
    if any(site.get("removed_atom_positions") for site in sites):
        raise ValueError(f"{spec.name}: target-side template leaked into strict sites")
    if any(site.get("removed_atom_indices") for site in sites):
        raise ValueError(f"{spec.name}: target-side atom indices leaked into strict sites")

    unsupported = set()
    for record in profile.get("extra_type_patterns", []):
        for key in record.get("class_counts", {}):
            symbol = str(key).rsplit("|", 1)[0]
            if symbol not in SUPPORTED_GENERATION_ELEMENTS:
                unsupported.add(symbol)
    if unsupported:
        raise ValueError(f"{spec.name}: unsupported strict elements {sorted(unsupported)}")

    required_patterns = {
        tuple(sorted(
            (int(slot), int(count))
            for slot, count in record["site_counts"].items()
        ))
        for record in profile["allocation_patterns"]
    }
    observed_patterns: set[tuple[tuple[int, int], ...]] = set()
    observed_counts: set[int] = set()
    fragment_mode = bool(
        config["sample"]["scaffold"]["murcko_sites"].get(
            "strict_fragment_gaussian", False
        )
    )
    if fragment_mode and not profile.get("fragment_patterns"):
        raise ValueError(f"{spec.name}: fragment profile is empty")
    rng = np.random.default_rng(919191 + len(indices))
    for n_extra in spec.allowed_n_extra:
        for _ in range(128):
            extra_pos, meta = build_extra_atom_positions(
                n_extra,
                sites,
                sites_cfg,
                np.zeros(3, dtype=np.float32),
                "cpu",
                rng=rng,
            )
            observed_counts.add(int(extra_pos.shape[0]))
            observed_patterns.add(tuple(sorted(
                (int(record["profile_slot"]), int(record["count"]))
                for record in meta["site_allocation"]
            )))
            if fragment_mode:
                hints = meta.get("fragment_type_hints") or []
                layout = meta.get("fragment_layout") or []
                if len(hints) != int(n_extra):
                    raise ValueError(
                        f"{spec.name}: fragment hints {len(hints)} != "
                        f"n_extra={n_extra}"
                    )
                if sum(
                    int(item.get("atom_count", 0)) for item in layout
                ) != int(n_extra):
                    raise ValueError(
                        f"{spec.name}: fragment layout count mismatch"
                    )
            if required_patterns <= observed_patterns:
                break
    if observed_counts != set(spec.allowed_n_extra):
        raise ValueError(
            f"{spec.name}: strict counts {sorted(observed_counts)} != "
            f"{list(spec.allowed_n_extra)}"
        )
    if not required_patterns <= observed_patterns:
        raise ValueError(
            f"{spec.name}: missing allocation patterns "
            f"{sorted(required_patterns - observed_patterns)}"
        )
    return {
        "class_name": spec.name,
        "scaffold_atoms": len(indices),
        "profile_slots": sorted(expected_slots),
        "observed_n_extra": sorted(observed_counts),
        "observed_allocation_patterns": [
            {str(slot): count for slot, count in pattern}
            for pattern in sorted(observed_patterns)
        ],
        "target_side_template_used": False,
        "target_side_coordinates_used": False,
    }


def prepare_campaign(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root).resolve()
    reference_sdf = Path(args.reference_sdf).resolve()
    native_ligand = Path(args.native_ligand).resolve()
    protein = Path(args.protein).resolve()
    base_config = _load_yaml(Path(args.config).resolve())
    f_ligand, profile_paths, profiles = prepare_class_profiles(
        reference_sdf, native_ligand, root
    )

    specs = {spec.name: spec for spec in CLASS_SPECS}
    configs: dict[str, str] = {}
    preflight: dict[str, Any] = {}
    for lane_index, lane in enumerate(LANES):
        spec = specs[lane.class_name]
        config = build_strict_config(
            base_config,
            spec,
            profile_paths[spec.name],
            profiles[spec.name],
            lane,
            seed=int(args.start_seed) + lane_index,
            samples=int(args.samples),
            fragment_gaussian=bool(args.fragment_gaussian),
            unlock_extra_types=bool(args.unlock_extra_types),
        )
        config_path = root / "configs" / f"{lane.name}.yml"
        _write_yaml(config, config_path)
        configs[lane.name] = str(config_path)
        if lane.class_name not in preflight:
            preflight[lane.class_name] = strict_preflight(
                spec,
                _ligand_for_spec(spec, native_ligand, f_ligand),
                profiles[spec.name],
                config,
                unlock_extra_types=bool(args.unlock_extra_types),
            )

    gpu_mapping = {lane.name: lane.gpu for lane in LANES}
    if 0 in gpu_mapping.values():
        raise ValueError("strict campaign must leave GPU0 unused")
    manifest = {
        "campaign": (
            "GSPT1 scaffold fragment-local Gaussian five-lane campaign"
            if args.fragment_gaussian
            else "GSPT1 strict scaffold-local Gaussian five-lane campaign"
        ),
        "created_at_unix": time.time(),
        "git_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip(),
        "reference_sdf": str(reference_sdf),
        "reference_sdf_sha256": _sha256(reference_sdf),
        "native_ligand": str(native_ligand),
        "native_ligand_sha256": _sha256(native_ligand),
        "f_core_ligand": str(f_ligand),
        "f_core_ligand_sha256": _sha256(f_ligand),
        "protein": str(protein),
        "protein_sha256": _sha256(protein),
        "gpu_mapping": gpu_mapping,
        "lane_classes": {lane.name: lane.class_name for lane in LANES},
        "profiles": {name: str(path) for name, path in profile_paths.items()},
        "configs": configs,
        "preflight": preflight,
        "policy": {
            "gpu0_unused": True,
            "scaffold_position_locked": True,
            "scaffold_type_locked": True,
            "extra_atom_types_locked": not bool(args.unlock_extra_types),
            "extra_atom_type_mode": (
                "diffuse" if args.unlock_extra_types else "locked"
            ),
            "added_position_mask": 0.0,
            "added_type_mask": 0.0 if args.unlock_extra_types else 1.0,
            "forced_attachment_bonds": False,
            "scaffold_reconstruction_covalent_factor": 1.3,
            "added_type_condition": (
                "uniform_x0_with_model_categorical_reverse"
                if args.unlock_extra_types
                else "exact_profile_quota_as_x0_with_forward_q"
            ),
            "added_coordinate_condition": (
                "scaffold_anchor_local_generic_fragment_gaussian"
                if args.fragment_gaussian
                else "scaffold_anchor_local_iid_gaussian"
            ),
            "fragment_gaussian": bool(args.fragment_gaussian),
            "fragment_geometry_condition": (
                "generic_motif_cloud_with_exact_type_alignment"
                if args.fragment_gaussian else None
            ),
            "diffusion_start_t": 999,
            "diffdynamic_refine_max_iterations": 30,
            "targetdiff_baseline_refine": False,
            "reference_target_side_graph_used": False,
            "reference_target_side_coordinates_used": False,
            "post_generation_graph_editing": False,
            "vina": False,
            "reconstruction_workers_per_lane": 18,
            "reconstruction_workers_total": 90,
        },
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "preflight.json").write_text(
        json.dumps(preflight, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
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
        "configs": configs,
        "manifest": manifest,
        "fragment_gaussian": bool(args.fragment_gaussian),
        "unlock_extra_types": bool(args.unlock_extra_types),
    }


def _lane_config(
    prepared: dict[str, Any],
    lane: LaneSpec,
    *,
    seed: int,
    samples: int,
) -> Path:
    spec = _get_class_spec(lane.class_name)
    config = build_strict_config(
        prepared["base_config"],
        spec,
        Path(prepared["profile_paths"][spec.name]),
        prepared["profiles"][spec.name],
        lane,
        seed=seed,
        samples=samples,
        fragment_gaussian=bool(prepared.get("fragment_gaussian", False)),
        unlock_extra_types=bool(prepared.get("unlock_extra_types", False)),
    )
    path = prepared["root"] / "configs" / f"run_{lane.name}_{seed}.yml"
    _write_yaml(config, path)
    return path


def _audit_round(
    prepared: dict[str, Any],
    round_index: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = prepared["root"]
    summaries: dict[str, Any] = {}
    for lane in LANES:
        spec = _get_class_spec(lane.class_name)
        lane_root = root / "rounds" / lane.name / f"run_{round_index:04d}"
        summaries[lane.name] = audit_class(
            [lane_root],
            prepared["reference_sdf"],
            spec.reference_indices,
            spec.scaffold_smarts,
            spec.allowed_n_extra,
            lane_root / "audit",
            class_name=lane.name,
        )
    total = audit(
        [root / "rounds"],
        prepared["reference_sdf"],
        root / "rounds" / f"run_{round_index:04d}_all_similarity",
        exclude_scaffold_smarts=sorted({
            spec.scaffold_smarts for spec in CLASS_SPECS
        }),
        top_n=None,
    )
    allc = audit_allc(
        root / "rounds" / f"run_{round_index:04d}_all_similarity" / "records.csv",
        prepared["reference_sdf"],
        root / "rounds" / f"run_{round_index:04d}_allc_similarity",
    )
    return summaries, {"original": total, "allc": allc}


def _job_is_complete(
    root: Path,
    lane_name: str,
    round_index: int,
    job_id: int,
) -> bool:
    """Check the durable markers for one generation/reconstruction job."""
    variant_root = root / "rounds" / lane_name / f"run_{round_index:04d}"
    job_dir = variant_root / "gspt1" / "jobs" / f"job_{job_id:04d}"
    extract_dir = (
        variant_root / "gspt1" / "extract_cleaned" / f"job_{job_id:04d}"
    )
    return (
        (job_dir / ".sample_done").is_file()
        and (extract_dir / ".extract_done").is_file()
    )


def _legacy_round_in_progress(root: Path, state: dict[str, Any]) -> bool:
    """Detect an interrupted round written by the pre-status state format."""
    round_index = int(state.get("round_index", 0))
    schedule = state.get("last_schedule") or []
    if round_index <= 0 or not schedule:
        return False

    previous_round = round_index - 1
    if any(
        not _job_is_complete(
            root,
            str(item.get("lane", "")),
            previous_round,
            int(item.get("job_id", -1)),
        )
        for item in schedule
    ):
        return True

    latest_path = root / "latest_summary.json"
    try:
        latest = json.loads(latest_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return True
    return int(latest.get("round", -1)) != previous_round


def _write_state(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def run_campaign(args: argparse.Namespace, prepared: dict[str, Any]) -> dict[str, Any]:
    root = prepared["root"]
    state_path = root / "state.json"
    state = {}
    if args.resume and state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
    round_index = int(state.get("round_index", 0))
    next_job_id = int(state.get("next_job_id", 0))
    pending_schedule = None
    state_status = state.get("status")
    if args.resume and state_status == "running":
        pending_schedule = state.get("last_schedule") or []
    elif args.resume and not state_status and _legacy_round_in_progress(
        root, state
    ):
        # The old launcher advanced round_index before the jobs were done.
        # Rewind only the unfinished round and reuse its original seeds.
        pending_schedule = state.get("last_schedule") or []
        round_index -= 1
        next_job_id = max(
            int(item["job_id"]) for item in pending_schedule
        ) + 1
    deadline = time.time() + float(args.hours) * 3600.0
    final_summary: dict[str, Any] = {}

    # Five reconstruction processes x 18 independent evaluator workers = 90.
    os.environ["SCAFFOLD_RECONSTRUCT_WORKERS"] = "18"

    while time.time() < deadline:
        if args.max_rounds and round_index >= args.max_rounds:
            break
        jobs = []
        lane_by_name = {lane.name: lane for lane in LANES}
        if pending_schedule:
            schedule = pending_schedule
            pending_schedule = None
        else:
            schedule = [
                {
                    "lane": lane.name,
                    "class": lane.class_name,
                    "gpu": lane.gpu,
                    "job_id": next_job_id + offset,
                    "seed": int(args.start_seed) + next_job_id + offset,
                }
                for offset, lane in enumerate(LANES)
            ]
            next_job_id += len(schedule)

        for item in schedule:
            lane = lane_by_name[str(item["lane"])]
            job_id = int(item["job_id"])
            seed = int(args.start_seed) + job_id
            variant_root = root / "rounds" / lane.name / f"run_{round_index:04d}"
            config_path = _lane_config(
                prepared, lane, seed=seed, samples=int(args.samples)
            )
            spec = _get_class_spec(lane.class_name)
            ligand = _ligand_for_spec(
                spec, prepared["native_ligand"], prepared["f_ligand"]
            )
            jobs.append((lane, job_id, variant_root, config_path, ligand))

        _write_state(
            state_path,
            {
                "status": "running",
                "round_index": round_index,
                "scheduled_job_start": min(
                    int(item["job_id"]) for item in schedule
                ),
                "next_job_id": next_job_id,
                "last_schedule": schedule,
            },
        )

        results = []
        with ThreadPoolExecutor(max_workers=len(jobs)) as executor:
            futures = [
                executor.submit(
                    _run_job,
                    root=root,
                    variant_root=variant_root,
                    config_path=config_path,
                    profile_path=prepared["profile_paths"][lane.class_name],
                    use_profile=True,
                    gpu=lane.gpu,
                    job_id=job_id,
                    base_seed=int(args.start_seed),
                    samples=int(args.samples),
                    protein=prepared["protein"],
                    native_ligand=ligand,
                    deadline=deadline,
                )
                for lane, job_id, variant_root, config_path, ligand in jobs
            ]
            for future in as_completed(futures):
                results.append(future.result())
        with (root / "job_results.jsonl").open("a", encoding="utf-8") as handle:
            for result in results:
                handle.write(json.dumps(result, ensure_ascii=True) + "\n")

        lane_summaries, total_summary = _audit_round(prepared, round_index)
        final_summary = {
            "round": round_index,
            "results": results,
            "lanes": lane_summaries,
            "total_similarity": total_summary.get("original", {}),
            "allc_similarity": total_summary.get("allc", {}),
            "best_side_similarity": max(
                (row.get("best_side_similarity", 0.0)
                for row in lane_summaries.values()),
                default=0.0,
            ),
            "best_allc_side_similarity": float(
                total_summary.get("allc", {}).get(
                    "best_side_similarity", 0.0
                )
            ),
            "best_allc_full_similarity": float(
                total_summary.get("allc", {}).get(
                    "best_full_similarity", 0.0
                )
            ),
            "allc_side_exact_count": int(
                total_summary.get("allc", {}).get(
                    "allc_side_exact_count", 0
                )
            ),
            "exact_reachable_count": sum(
                int(row.get("exact_reachable_count", 0))
                for row in lane_summaries.values()
            ),
        }
        (root / "latest_summary.json").write_text(
            json.dumps(final_summary, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        round_completed = round_index
        _write_state(
            state_path,
            {
                "status": "completed",
                "round_index": round_completed + 1,
                "completed_round": round_completed,
                "next_job_id": next_job_id,
                "last_schedule": schedule,
            },
        )
        print(json.dumps(final_summary, ensure_ascii=True), flush=True)
        if final_summary["exact_reachable_count"] > 0 or int(
            total_summary.get("original", {}).get("exact_count", 0)
        ) > 0:
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
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--hours", type=float, default=10.0)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--max-rounds", type=int, default=0)
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--reference-sdf", default=str(DEFAULT_REFERENCE))
    parser.add_argument("--native-ligand", default=str(DEFAULT_NATIVE))
    parser.add_argument("--protein", default=str(DEFAULT_PROTEIN))
    parser.add_argument("--start-seed", type=int, default=20290000)
    parser.add_argument("--gpus", default="1,2,3,4,5")
    parser.add_argument(
        "--fragment-gaussian",
        action="store_true",
        help="use scaffold-only generic fragment Gaussian clouds",
    )
    parser.add_argument(
        "--unlock-extra-types",
        action="store_true",
        help=(
            "scaffold-only ablation: keep reference n_extra values but let "
            "the model predict added atom types"
        ),
    )
    args = parser.parse_args()
    gpu_ids = [int(value.strip()) for value in args.gpus.split(",") if value.strip()]
    if gpu_ids != [lane.gpu for lane in LANES]:
        parser.error("--gpus must be exactly 1,2,3,4,5; GPU0 is reserved")
    prepared = prepare_campaign(args)
    print(json.dumps(prepared["manifest"], indent=2, ensure_ascii=True))
    if not args.run:
        print("PREPARED_ONLY: pass --run to start generation")
        return
    print(json.dumps(run_campaign(args, prepared), indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
