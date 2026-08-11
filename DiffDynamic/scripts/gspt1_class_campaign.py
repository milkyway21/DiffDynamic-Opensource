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
    "molglue_ikzf2_gspt1/diffdynamic/gspt1_class_campaign"
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


def build_class_config(
    base: dict[str, Any],
    spec: Gspt1ClassSpec,
    profile_path: Path,
    profile: dict,
    *,
    seed: int,
    samples: int,
) -> dict[str, Any]:
    """Create one fully pinned class config without changing de novo modes."""
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
        "jitter_mode": spec.jitter_mode,
        "jitter_std": 0.08,
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
        # Added atoms are initialized, but never position- or type-masked.
        "extra_anchor_strength": 0.0,
    })
    targetdiff = sample.setdefault("targetdiff_baseline_refine", {})
    targetdiff.update({
        "enable": True,
        "start_t": 19,
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
    if minimum_clearance + 1e-5 < threshold:
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
        "attempts_by_size": attempts_by_size,
    }


def prepare_campaign(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root)
    reference_sdf = Path(args.reference_sdf)
    native_ligand = Path(args.native_ligand)
    protein = Path(args.protein)
    base_config = _load_yaml(Path(args.config))
    f_ligand, profile_paths, profiles = prepare_class_profiles(
        reference_sdf, native_ligand, root
    )
    configs: dict[str, Path] = {}
    preflight = {}
    for offset, spec in enumerate(CLASS_SPECS):
        profile = profiles[spec.name]
        config = build_class_config(
            base_config,
            spec,
            profile_paths[spec.name],
            profile,
            seed=int(args.start_seed) + offset,
            samples=int(args.samples),
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
            for gpu, spec in zip(args.gpus, CLASS_SPECS)
        },
        "profiles": {name: str(path) for name, path in profile_paths.items()},
        "configs": {name: str(path) for name, path in configs.items()},
        "preflight": preflight,
        "policy": {
            "original_scaffold_position_locked": True,
            "original_scaffold_type_locked": True,
            "added_position_locked": False,
            "added_type_locked": False,
            "targetdiff_reverse_steps": 20,
            "reference_target_graph_used": False,
            "post_generation_graph_editing": False,
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
        "configs": configs,
        "manifest": manifest,
    }


def run_campaign(args: argparse.Namespace, prepared: dict[str, Any]) -> dict:
    root = prepared["root"]
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
        for gpu, spec in zip(args.gpus, CLASS_SPECS):
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
        for spec in CLASS_SPECS:
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
    parser.add_argument("--start-seed", type=int, default=20273000)
    parser.add_argument("--gpus", default="3,4,5")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.gpus = [
        int(value.strip()) for value in args.gpus.split(",") if value.strip()
    ]
    if len(args.gpus) != len(CLASS_SPECS):
        parser.error("--gpus must contain exactly three GPU ids")
    prepared = prepare_campaign(args)
    print(json.dumps(prepared["manifest"], indent=2, ensure_ascii=True))
    if not args.run:
        print("PREPARED_ONLY: pass --run to start GPU generation")
        return
    summary = run_campaign(args, prepared)
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
