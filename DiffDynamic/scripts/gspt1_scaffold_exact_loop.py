#!/usr/bin/env python3
"""Run an auditable, scaffold-only GSPT1 generation/reconstruction loop."""

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
from pathlib import Path
from typing import Any, List, Optional

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.gspt1_smiles_audit import audit
from utils.gspt1_scaffold_prior import (
    build_scaffold_profile,
    load_scaffold_profile,
    write_scaffold_profile,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
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
SCAFFOLD_SMARTS = "O=C1CCC(N2Cc3ccccc3C2=O)C(=O)N1"

VARIANTS = (
    # Keep the native target-side spatial template fixed while leaving atom
    # classes and bonds to the diffusion model.
    # name, jitter_mode, anchor_strength, jitter_std, profile, baseline,
    # n_extra_mode, site_selection_mode
    ("native_fixed_100", "native_template", 1.00, 0.10, True, False,
     "reference_size_prior", "weighted_single"),
    ("native_anchor_075", "native_template", 0.75, 0.15, True, False,
     "reference_size_prior", "weighted_single"),
    ("native_anchor_050", "native_template", 0.50, 0.20, True, False,
     "reference_size_prior", "weighted_single"),
    ("native_anchor_020", "native_template", 0.20, 0.25, True, False,
     "reference_size_prior", "weighted_single"),
    # Ablation: keep the coarse exit profile but restore the TargetDiff
    # baseline refine, which is the strongest control in the pilot.
    ("profile_baseline_020", "native_template", 0.20, 0.25, True, True,
     "reference_size_prior", "weighted_single"),
    # Exit-profile ablation: use the reference-derived exit distribution while
    # retaining the legacy pocket-size prior and TargetDiff baseline control.
    ("profile_legacy_tbr_020", "native_template", 0.20, 0.25, True, True,
     "prior_minus_scaffold", "weighted_single"),
    # The dominant reference exit is scaffold slot 0.  Isolate that geometry
    # while retaining the native target-side coordinate template only.
    ("profile_slot0_tbr_020", "native_template", 0.20, 0.25, True, True,
     "prior_minus_scaffold", "weighted_single"),
    # Keep the native component order while rotating from its closest
    # attachment atom; this separates coordinate order from exit direction.
    ("profile_slot0_native_order_020", "native_template", 0.20, 0.25,
     True, True, "prior_minus_scaffold", "weighted_single"),
    # Same slot-0 geometry with the observed 13-25 heavy-atom size prior;
    # isolates fragment completeness from exit placement.
    ("profile_slot0_size_tbr_020", "native_template", 0.20, 0.25, True, True,
     "reference_size_prior", "weighted_single"),
    # Same profiled exits with the observed reference extra-atom size prior;
    # isolates size selection from the legacy pocket-size control.
    ("profile_size_tbr_020", "native_template", 0.20, 0.25, True, True,
     "reference_size_prior", "weighted_single"),
    # Weak chemistry marginal ablation: aggregate reference element/aromatic
    # counts initialize extra atom classes without exposing any graph detail.
    ("profile_type_prior_020", "native_template", 0.20, 0.25, True, True,
     "prior_minus_scaffold", "weighted_single"),
    # Ablation: keep the native single exit geometry, but use the reference
    # heavy-atom count prior and the same baseline refine.
    ("native_size_baseline_020", "native_template", 0.20, 0.25, False, True,
     "reference_size_prior", "legacy"),
    ("hybrid_anchor_035", "hybrid", 0.35, 0.25, True, False,
     "reference_size_prior", "weighted_single"),
    ("directional_anchor_020", "directional", 0.20, 0.25, True, False,
     "reference_size_prior", "weighted_single"),
    # Reproduces the useful pre-profile pilot geometry: native target-side
    # template, one real Murcko exit, old size prior, and baseline refine.
    ("native_tbr_legacy_020", "native_template", 0.20, 0.25, False, True,
     "prior_minus_scaffold", "legacy"),
    # Independent replicate of the best control; it uses a distinct job
    # namespace and seed while keeping the same scaffold-only configuration.
    ("native_tbr_legacy_020_replica", "native_template", 0.20, 0.25,
     False, True, "prior_minus_scaffold", "legacy"),
    # Restore the pilot's anchor-radial initial coordinates as a separate
    # control; atom types and bonds remain fully model-generated.
    ("anchor_radial_tbr_020", "anchor_radial", 0.20, 0.25, False, True,
     "prior_minus_scaffold", "legacy"),
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


def _variant_config(
    base: dict[str, Any],
    profile_path: Path,
    jitter_mode: str,
    anchor_strength: float,
    jitter_std: float,
    use_profile: bool,
    enable_baseline: bool,
    n_extra_mode: str,
    site_selection_mode: str,
    profile: dict[str, Any],
    extra_type_prior_strength: float = 0.0,
    reference_exit_slots: Optional[List[int]] = None,
    reference_exit_only: bool = False,
    reference_exit_template_order: Optional[str] = None,
) -> dict[str, Any]:
    config = copy.deepcopy(base)
    scaffold = config.setdefault("sample", {}).setdefault("scaffold", {})
    sites = scaffold.setdefault("murcko_sites", {})
    sites["reference_exit_profile"] = str(profile_path) if use_profile else None
    if site_selection_mode == "legacy":
        sites.pop("site_selection_mode", None)
    else:
        sites["site_selection_mode"] = site_selection_mode
        if site_selection_mode == "weighted_single":
            # weighted_single is selected after the per-site mode branch;
            # otherwise the inherited sequential_random mode would split
            # atoms across exits before concentration erased the weights.
            sites["per_site_count_mode"] = "split"
    sites["site_budget_mode"] = "requested"
    sites["reference_extra_type_prior_strength"] = float(
        extra_type_prior_strength
    )
    if reference_exit_slots is None:
        sites.pop("reference_exit_slots", None)
    else:
        sites["reference_exit_slots"] = [int(slot) for slot in reference_exit_slots]
    sites["reference_exit_only"] = bool(reference_exit_only)
    if reference_exit_template_order is None:
        sites.pop("reference_exit_template_order", None)
    else:
        sites["reference_exit_template_order"] = str(
            reference_exit_template_order
        )
    sites["jitter_mode"] = jitter_mode
    sites["jitter_std"] = jitter_std
    scaffold.setdefault("grow", {})["extra_anchor_strength"] = anchor_strength
    # The profile variants disable TargetDiff baseline because it is
    # unconstrained for extra atoms and can undo the target-side spatial
    # prior.  The legacy control explicitly keeps it enabled for comparison.
    baseline_refine = config.setdefault("sample", {}).setdefault(
        "targetdiff_baseline_refine", {}
    )
    baseline_refine["enable"] = enable_baseline
    grow = scaffold["grow"]
    grow["n_extra_mode"] = n_extra_mode
    if n_extra_mode in ("reference_size_prior", "reference_n_extra_prior"):
        grow["reference_size_values"] = list(profile["n_extra_values"])
        grow["reference_size_weights"] = list(profile["n_extra_weights"])
        grow["n_extra_min_clamp"] = min(profile["n_extra_values"])
        grow["n_extra_max_clamp"] = max(profile["n_extra_values"])
        grow["n_extra_min"] = min(profile["n_extra_values"])
        grow["n_extra_max"] = max(profile["n_extra_values"])
    else:
        grow["n_extra_fixed"] = 8
        grow["n_extra_min"] = 10
        grow["n_extra_max"] = 22
        grow["n_extra_min_clamp"] = 10
        grow["n_extra_max_clamp"] = 22
    return config


def _write_manifest(
    root: Path,
    reference_sdf: Path,
    native_ligand: Path,
    protein: Path,
    config: Path,
    profile: dict[str, Any],
    start_seed: int,
    gpus: list[int],
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    campaign_started_at = time.time()
    manifest_path = root / "campaign_manifest.json"
    if manifest_path.exists():
        try:
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
            previous_started_at = float(previous.get("created_at_unix", 0.0))
            if previous_started_at > 0.0:
                campaign_started_at = previous_started_at
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            pass
    manifest = {
        "campaign": "GSPT1 scaffold exact-match loop",
        "created_at_unix": campaign_started_at,
        "git_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip(),
        "reference_sdf": str(reference_sdf),
        "reference_sdf_sha256": _sha256(reference_sdf),
        "native_ligand": str(native_ligand),
        "native_ligand_sha256": _sha256(native_ligand),
        "protein": str(protein),
        "protein_sha256": _sha256(protein),
        "config": str(config),
        "scaffold_smarts": SCAFFOLD_SMARTS,
        "start_seed": start_seed,
        "gpus": gpus,
        "reference_profile": profile,
        "anti_cheating": {
            "reference_target_side_atoms_passed_to_model": False,
            "aggregate_reference_target_side_element_counts_used_as_prior": True,
            "aggregate_reference_target_side_element_prior_strength": 0.35,
            "reference_target_side_bonds_passed_to_model": False,
            "reference_target_side_coordinates_passed_to_model": False,
            "native_target_side_coordinates_used_as_spatial_prior": True,
            "native_target_side_atom_types_passed_to_model": False,
            "native_target_side_bonds_passed_to_model": False,
            "post_generation_graph_editing": False,
            "vina": False,
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _run_job(
    *,
    root: Path,
    variant_root: Path,
    config_path: Path,
    profile_path: Path,
    use_profile: bool,
    gpu: int,
    job_id: int,
    base_seed: int,
    samples: int,
    protein: Path,
    native_ligand: Path,
    deadline: float,
) -> dict[str, Any]:
    job_dir = variant_root / "gspt1" / "jobs" / f"job_{job_id:04d}"
    job_dir.mkdir(parents=True, exist_ok=True)
    log_path = job_dir / "orchestrator.log"
    env = os.environ.copy()
    env.update({
        "ROOT": str(variant_root),
        "CFG_SRC": str(config_path),
        "PROFILE": str(profile_path) if use_profile else "",
        "GPU": str(gpu),
        "JOB_ID": str(job_id),
        "BASE_SEED": str(base_seed),
        "NUM_SAMPLES": str(samples),
        "TARGET": "gspt1",
        "PROTEIN": str(protein),
        "LIGAND": str(native_ligand),
    })
    started = time.time()
    result: dict[str, Any] = {
        "job_id": job_id,
        "gpu": gpu,
        "variant_root": str(variant_root),
        "status": "failed",
    }
    try:
        with log_path.open("a", encoding="utf-8") as log_handle:
            remaining = max(1, int(deadline - time.time()))
            sample = subprocess.run(
                ["bash", str(REPO_ROOT / "scripts/run_gspt1_scaffold_job.sh")],
                cwd=REPO_ROOT,
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                timeout=remaining,
                check=False,
            )
            result["sample_returncode"] = sample.returncode
            if sample.returncode != 0:
                result["status"] = "sample_failed"
                return result

            remaining = max(1, int(deadline - time.time()))
            reconstruct = subprocess.run(
                ["bash", str(REPO_ROOT / "scripts/reconstruct_gspt1_scaffold_job.sh")],
                cwd=REPO_ROOT,
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                timeout=remaining,
                check=False,
            )
            result["reconstruct_returncode"] = reconstruct.returncode
            if reconstruct.returncode != 0:
                result["status"] = "reconstruct_failed"
                return result
        result["status"] = "ok"
    except subprocess.TimeoutExpired:
        result["status"] = "timeout"
    except Exception as exc:
        result["status"] = f"error:{type(exc).__name__}"
        result["error"] = str(exc)
    finally:
        result["elapsed_seconds"] = time.time() - started
    return result


def _variant_score(summary: dict[str, Any]) -> tuple[float, int, int]:
    return (
        float(summary.get("best_similarity", 0.0)),
        int(summary.get("threshold_counts", {}).get("0.5", 0)),
        int(summary.get("generated_unique", 0)),
    )


def _choose_variants(
    variant_summaries: dict[str, dict[str, Any]],
    run_counts: dict[str, int],
    round_index: int,
) -> list[tuple[str, str, float]]:
    if not variant_summaries:
        start = (round_index * 3) % len(VARIANTS)
        return [VARIANTS[(start + i) % len(VARIANTS)] for i in range(3)]
    ranked = sorted(
        VARIANTS,
        key=lambda item: _variant_score(variant_summaries.get(item[0], {})),
        reverse=True,
    )
    legacy_family = tuple(
        item for item in VARIANTS
        if item[0] in {
            "native_tbr_legacy_020",
            "native_tbr_legacy_020_replica",
        }
    )
    if ranked[0][0] in {item[0] for item in legacy_family} and len(legacy_family) > 1:
        chosen = [ranked[0]]
        legacy_by_run = sorted(
            legacy_family,
            key=lambda item: run_counts.get(item[0], 0),
        )
        for item in legacy_by_run:
            if item not in chosen:
                chosen.append(item)
                break
        unseen = [item for item in VARIANTS if item[0] not in variant_summaries]
        if unseen:
            chosen.append(unseen[0])
        for item in ranked:
            if item not in chosen:
                chosen.append(item)
            if len(chosen) == 3:
                break
        return chosen
    unseen = [item for item in VARIANTS if item[0] not in variant_summaries]
    if unseen:
        chosen = [ranked[0], unseen[0]]
    else:
        least_run = sorted(VARIANTS, key=lambda item: run_counts.get(item[0], 0))
        chosen = [ranked[0], least_run[0]]
    for item in ranked:
        if item not in chosen:
            chosen.append(item)
        if len(chosen) == 3:
            break
    return chosen


def run_campaign(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root)
    reference_sdf = Path(args.reference_sdf)
    native_ligand = Path(args.native_ligand)
    protein = Path(args.protein)
    config_path = Path(args.config)
    profile_path = root / "gspt1_exit_profile.json"
    if profile_path.exists():
        profile = load_scaffold_profile(profile_path)
    else:
        profile = build_scaffold_profile(
            reference_sdf,
            native_ligand,
            SCAFFOLD_SMARTS,
            exploration_floor=1.0,
        )
        write_scaffold_profile(profile, profile_path)

    _write_manifest(
        root, reference_sdf, native_ligand, protein, config_path, profile,
        args.start_seed, args.gpus,
    )
    state_path = root / "state.json"
    state = {}
    if state_path.exists() and args.resume:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    next_job_id = int(state.get("next_job_id", 0))
    round_index = int(state.get("round_index", 0))
    run_counts = dict(state.get("run_counts", {}))
    deadline = time.time() + float(args.hours) * 3600.0
    campaign_summary: dict[str, Any] = {}

    while time.time() < deadline:
        if args.max_rounds and round_index >= args.max_rounds:
            break
        variant_summaries = {}
        for name, _, _, _, _, _, _, _ in VARIANTS:
            variant_root = root / "rounds" / name
            summary_path = variant_root / "audit" / "summary.json"
            if summary_path.exists():
                try:
                    variant_summaries[name] = json.loads(
                        summary_path.read_text(encoding="utf-8")
                    )
                except json.JSONDecodeError:
                    pass

        selected = _choose_variants(variant_summaries, run_counts, round_index)
        jobs = []
        for gpu, variant in zip(args.gpus, selected):
            (
                name, jitter_mode, anchor_strength, jitter_std,
                use_profile, enable_baseline, n_extra_mode,
                site_selection_mode,
            ) = variant
            variant_root = root / "rounds" / name / f"run_{round_index:04d}"
            variant_config = variant_root / "config.yml"
            config = _variant_config(
                _load_yaml(config_path),
                profile_path,
                jitter_mode,
                anchor_strength,
                jitter_std,
                use_profile,
                enable_baseline,
                n_extra_mode,
                site_selection_mode,
                profile,
                extra_type_prior_strength={
                    "profile_type_prior_020": 0.35,
                }.get(name, 0.0),
                reference_exit_slots={
                    "profile_slot0_tbr_020": [0],
                    "profile_slot0_size_tbr_020": [0],
                    "profile_slot0_native_order_020": [0],
                }.get(name),
                reference_exit_only=name in {
                    "profile_slot0_tbr_020",
                    "profile_slot0_size_tbr_020",
                    "profile_slot0_native_order_020",
                },
                reference_exit_template_order=(
                    "native"
                    if name == "profile_slot0_native_order_020"
                    else None
                ),
            )
            _write_yaml(config, variant_config)
            jobs.append(
                (
                    gpu, variant, variant_root, variant_config,
                    next_job_id, use_profile,
                )
            )
            next_job_id += 1
            run_counts[name] = run_counts.get(name, 0) + 1

        state = {
            "next_job_id": next_job_id,
            "round_index": round_index + 1,
            "run_counts": run_counts,
            "last_schedule": [
                {"gpu": gpu, "variant": variant[0], "job_id": job_id}
                for gpu, variant, _, _, job_id, _ in jobs
            ],
        }
        root.mkdir(parents=True, exist_ok=True)
        state_path.write_text(
            json.dumps(state, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )

        results = []
        with ThreadPoolExecutor(max_workers=len(jobs)) as executor:
            futures = [
                executor.submit(
                    _run_job,
                    root=root,
                    variant_root=variant_root,
                    config_path=variant_config,
                    profile_path=profile_path,
                    use_profile=job_use_profile,
                    gpu=gpu,
                    job_id=job_id,
                    base_seed=args.start_seed,
                    samples=args.samples,
                    protein=protein,
                    native_ligand=native_ligand,
                    deadline=deadline,
                )
                for (
                    gpu, _, variant_root, variant_config, job_id,
                    job_use_profile,
                ) in jobs
            ]
            for future in as_completed(futures):
                results.append(future.result())

        with (root / "job_results.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(
                "".join(
                    json.dumps(item, ensure_ascii=True) + "\n"
                    for item in results
                )
            )

        audit_dir = root / "audit"
        campaign_summary = audit(
            [root], reference_sdf, audit_dir, scaffold_atoms=18, top_n=500
        )
        for name, _, _, _, _, _, _, _ in VARIANTS:
            variant_root = root / "rounds" / name
            if variant_root.exists():
                audit(
                    [variant_root],
                    reference_sdf,
                    variant_root / "audit",
                    scaffold_atoms=18,
                    top_n=200,
                )
        print(json.dumps({
            "round": round_index,
            "results": results,
            "summary": campaign_summary,
        }, ensure_ascii=True), flush=True)
        if int(campaign_summary.get("exact_count", 0)) > 0:
            (root / "EXACT_MATCH_FOUND").write_text(
                json.dumps(campaign_summary, indent=2, ensure_ascii=True) + "\n",
                encoding="utf-8",
            )
            break
        round_index += 1

    campaign_summary["deadline_reached"] = time.time() >= deadline
    campaign_summary["finished_at_unix"] = time.time()
    (root / "final_summary.json").write_text(
        json.dumps(campaign_summary, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return campaign_summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hours", type=float, default=10.0)
    parser.add_argument(
        "--root",
        default="/data/zhang/Ye/DiffDynamic_outputs/hsvpol/molglue_ikzf2_gspt1/diffdynamic/gspt1_scaffold_exact_loop",
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--reference-sdf", default=str(DEFAULT_REFERENCE))
    parser.add_argument("--native-ligand", default=str(DEFAULT_NATIVE))
    parser.add_argument("--protein", default=str(DEFAULT_PROTEIN))
    parser.add_argument("--start-seed", type=int, default=20270600)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--gpus", default="3,4,5")
    parser.add_argument("--max-rounds", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.gpus = [int(item.strip()) for item in args.gpus.split(",") if item.strip()]
    if not args.gpus:
        raise SystemExit("--gpus must contain at least one GPU")
    summary = run_campaign(args)
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
