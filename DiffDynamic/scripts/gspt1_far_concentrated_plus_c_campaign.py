#!/usr/bin/env python3
"""Run a scaffold-only GSPT1 fragment-Gaussian placement campaign.

This campaign derives a new coarse prior from the existing GSPT1 class
profiles by adding exactly one exchangeable aliphatic carbon to every
allowed extra-atom count and to its scaffold-exit allocation.  It does not
copy target-side bonds, atom order, or target-side coordinates.

The three class lanes are intentionally limited to GPUs 3, 4, and 5:
no-F, F-main, and F-large.  Added coordinates are initialized from generic
fragment geometries around scaffold exit frames, then passed through the
normal t=999 DiffDynamic reverse process with no coordinate mask.  The
fragment prior contains no target-side graph, atom order, or coordinates.
Vina is disabled.  A 30-step TargetDiff baseline refinement follows the
DiffDynamic chain while restoring the CRBN prefix types and positions.
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
from dataclasses import replace
from pathlib import Path
from typing import Any

from rdkit import Chem

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.gspt1_allc_tanimoto_audit import audit as audit_allc  # noqa: E402
from scripts.gspt1_class_audit import audit_class  # noqa: E402
from scripts.gspt1_scaffold_exact_loop import _run_job  # noqa: E402
from scripts.gspt1_smiles_audit import audit  # noqa: E402
from scripts.gspt1_strict_gaussian_campaign import (  # noqa: E402
    LaneSpec,
    _get_class_spec,
    _ligand_for_spec,
    _load_yaml,
    _sha256,
    _write_yaml,
    build_strict_config,
    strict_preflight,
)
from utils.gspt1_class_setup import (  # noqa: E402
    CLASS_SPECS,
    prepare_class_profiles,
)
from utils.gspt1_scaffold_prior import iter_sdf_molecules  # noqa: E402


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
    "02_gspt1_or_gstp1_molecular_glue/receptor/"
    "5HXB_receptor_clean.pdb"
)
DEFAULT_ROOT = Path(
    "/data/zhang/Ye/DiffDynamic_outputs/hsvpol/"
    "molglue_ikzf2_gspt1/diffdynamic/"
    "gspt1_scaffold_fragment_gaussian_plus_c_v1"
)


LANES = (
    LaneSpec(
        "no_f_fragment_plus_c",
        "no_f",
        3,
        0.20,
        2.65,
        0.00,
        0.00,
        {"0": 2.65, "10": 2.65},
        {"0": 0.00, "10": 0.00},
        {"0": 0.00, "10": 0.00},
    ),
    LaneSpec(
        "f_main_fragment_plus_c",
        "f_main",
        4,
        0.20,
        2.55,
        -0.10,
        0.15,
        {"1": 2.55},
        {"1": -0.10},
        {"1": 0.15},
    ),
    LaneSpec(
        "f_large_fragment_plus_c",
        "f_large",
        5,
        0.20,
        2.75,
        0.15,
        0.00,
        {"1": 2.75, "7": 2.75},
        {"1": 0.15, "7": 0.15},
        {"1": 0.00, "7": 0.00},
    ),
)


def _primary_slot(site_counts: dict[str, Any]) -> str:
    """Return the main exit slot without retaining any reference atom index."""
    return min(
        (str(slot) for slot, count in site_counts.items() if int(count) > 0),
        key=lambda slot: (-int(site_counts[slot]), int(slot)),
    )


def _add_carbon_type(class_counts: dict[str, Any]) -> dict[str, int]:
    counts = {str(key): int(value) for key, value in class_counts.items()}
    counts["C|0"] = counts.get("C|0", 0) + 1
    return counts


def _add_carbon_to_fragment_slot(
    fragments: list[dict[str, Any]],
) -> str:
    """Add C|0 to a generic chain, never to a ring or carbonyl core."""
    chain_candidates = [
        (index, fragment)
        for index, fragment in enumerate(fragments)
        if str(fragment.get("kind", "")) == "chain"
    ]
    if chain_candidates:
        index, fragment = max(
            chain_candidates,
            key=lambda item: int(
                item[1].get("atom_counts", {}).get("C|0", 0)
            ),
        )
        atom_counts = fragment.setdefault("atom_counts", {})
        atom_counts["C|0"] = int(atom_counts.get("C|0", 0)) + 1
        return f"extend_existing_chain:{index}"

    fragments.append({
        "kind": "chain",
        "ring_sizes": [],
        "atom_counts": {"C|0": 1},
    })
    return "append_generic_chain"


def _write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def derive_plus_one_carbon_profile(profile: dict[str, Any]) -> dict[str, Any]:
    """Derive a count-shifted, exchangeable prior from a class profile."""
    derived = copy.deepcopy(profile)
    derived["profile_kind"] = (
        f"{profile.get('profile_kind', 'coarse')}_plus_one_aliphatic_carbon"
    )
    derived["derived_from_profile_class"] = profile.get("class_name", "")
    derived["derived_transform"] = {
        "n_extra_delta": 1,
        "added_element_class": "C|0",
        "allocation_slot_rule": "add_to_primary_exit_slot",
        "fragment_rule": (
            "extend_existing_chain_else_append_generic_chain"
        ),
        "target_side_graph_used": False,
        "target_side_coordinates_used": False,
    }

    values = [int(value) + 1 for value in profile.get("n_extra_values", [])]
    derived["n_extra_values"] = values
    derived["n_extra_weights"] = [
        float(value) for value in profile.get("n_extra_weights", [])
    ]

    allocation_patterns = []
    for pattern in profile.get("allocation_patterns", []):
        site_counts = {
            str(slot): int(count)
            for slot, count in pattern.get("site_counts", {}).items()
        }
        slot = _primary_slot(site_counts)
        site_counts[slot] = site_counts.get(slot, 0) + 1
        allocation_patterns.append({
            "n_extra": int(pattern.get("n_extra", 0)) + 1,
            "site_counts": site_counts,
            "weight": float(pattern.get("weight", 1.0)),
        })
    derived["allocation_patterns"] = allocation_patterns

    type_patterns = []
    for pattern in profile.get("extra_type_patterns", []):
        type_patterns.append({
            "n_extra": int(pattern.get("n_extra", 0)) + 1,
            "class_counts": _add_carbon_type(
                pattern.get("class_counts", {})
            ),
            "weight": float(pattern.get("weight", 1.0)),
        })
    derived["extra_type_patterns"] = type_patterns

    fragments = []
    fragment_rules = []
    for pattern in profile.get("fragment_patterns", []):
        transformed = copy.deepcopy(pattern)
        transformed["n_extra"] = int(pattern.get("n_extra", 0)) + 1
        site_fragments = transformed.get("site_fragments", {})
        if site_fragments:
            slot = _primary_slot({
                str(key): sum(
                    int(value)
                    for fragment in fragments_for_slot
                    for value in fragment.get("atom_counts", {}).values()
                )
                for key, fragments_for_slot in site_fragments.items()
            })
            slot_fragments = site_fragments[slot]
            if slot_fragments:
                fragment_rules.append(
                    _add_carbon_to_fragment_slot(slot_fragments)
                )
        fragments.append(transformed)
    derived["fragment_patterns"] = fragments
    derived["derived_transform"]["fragment_rules"] = fragment_rules

    element_counts = dict(profile.get("reference_extra_element_counts", {}))
    element_counts["C"] = float(element_counts.get("C", 0.0)) + 1.0
    derived["reference_extra_element_counts"] = element_counts
    aromatic_counts = dict(
        profile.get("reference_extra_aromatic_element_counts", {})
    )
    aromatic_counts["C|0"] = float(aromatic_counts.get("C|0", 0.0)) + 1.0
    derived["reference_extra_aromatic_element_counts"] = aromatic_counts
    return derived


def _effective_spec(spec, profile: dict[str, Any]):
    return replace(
        spec,
        allowed_n_extra=tuple(int(value) for value in profile["n_extra_values"]),
    )


def _audit_round(
    prepared: dict[str, Any],
    round_index: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = prepared["root"]
    lane_summaries: dict[str, Any] = {}
    class_specs = prepared["effective_specs"]
    for lane in LANES:
        spec = class_specs[lane.class_name]
        lane_root = root / "rounds" / lane.name / f"run_{round_index:04d}"
        lane_summaries[lane.name] = audit_class(
            [lane_root],
            prepared["reference_sdf"],
            spec.reference_indices,
            spec.scaffold_smarts,
            spec.allowed_n_extra,
            lane_root / "audit",
            class_name=lane.name,
            top_n=None,
        )

    round_roots = [
        root / "rounds" / lane.name / f"run_{round_index:04d}"
        for lane in LANES
    ]
    original = audit(
        round_roots,
        prepared["reference_sdf"],
        root / "rounds" / f"run_{round_index:04d}_all_similarity",
        scaffold_atoms=18,
        top_n=None,
        exclude_scaffold_smarts=sorted({
            spec.scaffold_smarts for spec in class_specs.values()
        }),
    )
    allc = audit_allc(
        root / "rounds" / f"run_{round_index:04d}_all_similarity" / "records.csv",
        prepared["reference_sdf"],
        root / "rounds" / f"run_{round_index:04d}_allc_similarity",
        top_n=None,
    )
    return lane_summaries, {"original": original, "allc": allc}


def prepare_campaign(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root).resolve()
    reference_sdf = Path(args.reference_sdf).resolve()
    native_ligand = Path(args.native_ligand).resolve()
    protein = Path(args.protein).resolve()
    base_config = _load_yaml(Path(args.config).resolve())
    f_ligand, original_paths, original_profiles = prepare_class_profiles(
        reference_sdf,
        native_ligand,
        root,
    )

    shifted_dir = root / "profiles_plus_one_carbon"
    shifted_dir.mkdir(parents=True, exist_ok=True)
    profiles: dict[str, dict[str, Any]] = {}
    profile_paths: dict[str, Path] = {}
    effective_specs = {}
    configs: dict[str, Path] = {}
    preflight: dict[str, Any] = {}
    spec_by_name = {spec.name: spec for spec in CLASS_SPECS}

    for lane_index, lane in enumerate(LANES):
        original_profile = original_profiles[lane.class_name]
        if lane.class_name not in profiles:
            profile = derive_plus_one_carbon_profile(original_profile)
            profile_path = shifted_dir / f"{lane.class_name}.json"
            _write_json(profile, profile_path)
            profiles[lane.class_name] = profile
            profile_paths[lane.class_name] = profile_path
            effective_specs[lane.class_name] = _effective_spec(
                spec_by_name[lane.class_name], profile
            )

        spec = effective_specs[lane.class_name]
        config = build_strict_config(
            base_config,
            spec,
            profile_paths[lane.class_name],
            profiles[lane.class_name],
            lane,
            seed=int(args.start_seed) + lane_index,
            samples=int(args.samples),
            fragment_gaussian=True,
            unlock_extra_types=False,
        )
        config_path = root / "configs" / f"{lane.name}.yml"
        _write_yaml(config, config_path)
        configs[lane.name] = config_path
        if lane.class_name not in preflight:
            preflight[lane.class_name] = strict_preflight(
                spec,
                _ligand_for_spec(
                    spec,
                    native_ligand,
                    f_ligand,
                ),
                profiles[lane.class_name],
                config,
                unlock_extra_types=False,
            )

    manifest = {
        "campaign": "GSPT1 scaffold fragment Gaussian plus one carbon",
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
        "gpu_mapping": {lane.name: lane.gpu for lane in LANES},
        "lane_classes": {lane.name: lane.class_name for lane in LANES},
        "original_profiles": {
            name: str(path) for name, path in original_paths.items()
        },
        "derived_profiles": {
            name: str(path) for name, path in profile_paths.items()
        },
        "configs": {name: str(path) for name, path in configs.items()},
        "preflight": preflight,
        "policy": {
            "gpu0_unused": True,
            "gpus": [3, 4, 5],
            "samples_per_lane": int(args.samples),
            "n_extra_delta": 1,
            "added_element_class": "C|0",
            "extra_atom_types_locked": True,
            "scaffold_position_locked": True,
            "added_position_mask": 0.0,
            "added_type_mask": 1.0,
            "scaffold_only_exit_direction": True,
            "fragment_gaussian": True,
            "fragment_attachment_distance_angstrom": 1.55,
            "fragment_attachment_spacing_angstrom": 0.85,
            "fragment_center_lateral_sigma_angstrom": 0.25,
            "center_offsets_angstrom": {
                lane.name: lane.center_offset for lane in LANES
            },
            "gaussian_sigmas_angstrom": {
                lane.name: lane.sigma for lane in LANES
            },
            "diffusion_start_t": 999,
            "skip_refine": False,
            "diffdynamic_large_step_then_refine": True,
            "normal_coordinate_reverse_process": True,
            "targetdiff_baseline_refine": True,
            "targetdiff_baseline_start_t": 29,
            "targetdiff_baseline_lock_prefix": "types_and_pos",
            "forced_attachment_bonds": False,
            "reference_target_side_graph_used": False,
            "reference_target_side_coordinates_used": False,
            "post_generation_graph_editing": False,
            "vina": False,
            "reconstruction_workers_per_lane": 30,
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
        "effective_specs": effective_specs,
        "configs": configs,
        "manifest": manifest,
    }


def _lane_config(
    prepared: dict[str, Any],
    lane: LaneSpec,
    *,
    seed: int,
    samples: int,
) -> Path:
    spec = prepared["effective_specs"][lane.class_name]
    config = build_strict_config(
        prepared["base_config"],
        spec,
        prepared["profile_paths"][lane.class_name],
        prepared["profiles"][lane.class_name],
        lane,
        seed=seed,
        samples=samples,
        fragment_gaussian=True,
        unlock_extra_types=False,
    )
    path = prepared["root"] / "configs" / f"run_{lane.name}_{seed}.yml"
    _write_yaml(config, path)
    return path


def run_campaign(args: argparse.Namespace, prepared: dict[str, Any]) -> dict[str, Any]:
    root = prepared["root"]
    state_path = root / "state.json"
    state = {}
    if args.resume and state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
    round_index = int(state.get("round_index", 0))
    next_job_id = int(state.get("next_job_id", 0))
    pending_schedule = None
    if args.resume and state.get("status") == "running":
        pending_schedule = state.get("last_schedule") or []
    deadline = time.time() + float(args.hours) * 3600.0
    final_summary: dict[str, Any] = {}

    # Three reconstruction jobs x 30 independent evaluator workers = 90.
    os.environ["SCAFFOLD_RECONSTRUCT_WORKERS"] = "30"

    while time.time() < deadline:
        if args.max_rounds and round_index >= args.max_rounds:
            break
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

        jobs = []
        for item in schedule:
            lane = lane_by_name[str(item["lane"])]
            job_id = int(item["job_id"])
            seed = int(args.start_seed) + job_id
            variant_root = root / "rounds" / lane.name / f"run_{round_index:04d}"
            config_path = _lane_config(
                prepared,
                lane,
                seed=seed,
                samples=int(args.samples),
            )
            spec = prepared["effective_specs"][lane.class_name]
            ligand = _ligand_for_spec(
                spec,
                prepared["native_ligand"],
                prepared["f_ligand"],
            )
            jobs.append((lane, job_id, variant_root, config_path, ligand))

        state_path.write_text(
            json.dumps({
                "status": "running",
                "round_index": round_index,
                "scheduled_job_start": min(
                    int(item["job_id"]) for item in schedule
                ),
                "next_job_id": next_job_id,
                "last_schedule": schedule,
            }, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
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
            "total_similarity": total_summary["original"],
            "allc_similarity": total_summary["allc"],
            "best_side_similarity": max(
                (
                    float(row.get("best_side_similarity", 0.0))
                    for row in lane_summaries.values()
                ),
                default=0.0,
            ),
            "best_allc_side_similarity": float(
                total_summary["allc"].get("best_side_similarity", 0.0)
            ),
            "best_allc_full_similarity": float(
                total_summary["allc"].get("best_full_similarity", 0.0)
            ),
            "exact_count": int(total_summary["original"].get("exact_count", 0)),
            "allc_side_exact_count": int(
                total_summary["allc"].get("allc_side_exact_count", 0)
            ),
        }
        (root / "latest_summary.json").write_text(
            json.dumps(final_summary, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        _write_state = {
            "status": "completed",
            "round_index": round_index + 1,
            "completed_round": round_index,
            "next_job_id": next_job_id,
            "last_schedule": schedule,
        }
        state_path.write_text(
            json.dumps(_write_state, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(final_summary, ensure_ascii=True), flush=True)
        if final_summary["exact_count"] > 0:
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
    parser.add_argument("--start-seed", type=int, default=20310000)
    parser.add_argument("--gpus", default="3,4,5")
    args = parser.parse_args()
    if args.samples <= 0:
        parser.error("--samples must be positive")
    if args.hours <= 0:
        parser.error("--hours must be positive")
    gpu_ids = [int(value.strip()) for value in args.gpus.split(",") if value.strip()]
    if gpu_ids != [3, 4, 5]:
        parser.error("--gpus must be exactly 3,4,5")
    prepared = prepare_campaign(args)
    print(json.dumps(prepared["manifest"], indent=2, ensure_ascii=True))
    if not args.run:
        print("PREPARED_ONLY: pass --run to start generation")
        return
    print(json.dumps(run_campaign(args, prepared), indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
