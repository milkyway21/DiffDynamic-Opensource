"""Generation job submissions, independent of queue/resource implementation."""

import json
import os
from typing import Optional

from server.submissions.dependencies import dependencies as deps


class GenerationJobs:
    """Prepare and dispatch generation jobs through JobScheduler hooks."""

    def submit_generation(
        self,
        mode: str = "dynamic",
        data_id: Optional[int] = 0,
        use_test_set: bool = True,
        gpu_id: Optional[int] = None,
        batch_size: Optional[int] = 5,
        config_path: Optional[str] = None,
        auto_evaluate: bool = True,
        auto_extract: bool = True,
        remove_fragments: bool = True,
        max_samples: Optional[int] = 5,
        vina_timeout: int = 20,
        prudent_n_sampling: Optional[int] = None,
        prudent_advance_top_k: Optional[int] = None,
        prudent_max_checkpoint_rounds: Optional[int] = None,
        prudent_refine_batch_size: Optional[int] = None,
        prudent_scoring_workers: Optional[int] = None,
        triggered_by: str = "web_ui",
        skip_daily_quota: bool = False,
        medchem: Optional[dict] = None,
    ) -> dict:
        cfg = deps.get_config()
        if batch_size is None:
            batch_size = 5
        parameters = {
            "mode": mode,
            "data_id": data_id,
            "use_test_set": use_test_set,
            "batch_size": batch_size,
            "auto_evaluate": auto_evaluate,
            "auto_extract": auto_extract,
            "remove_fragments": remove_fragments,
            "max_samples": max_samples if max_samples is not None else 5,
            "vina_timeout": vina_timeout,
            "preferred_gpu": gpu_id,
            "prudent_n_sampling": prudent_n_sampling,
            "prudent_advance_top_k": prudent_advance_top_k,
            "prudent_max_checkpoint_rounds": prudent_max_checkpoint_rounds,
            "prudent_refine_batch_size": prudent_refine_batch_size,
            "prudent_scoring_workers": prudent_scoring_workers,
        }
        if medchem is not None:
            from server.medchem_schemas import MedchemOptions
            parameters["medchem"] = MedchemOptions(**medchem).dict()
        owner = self._attach_owner_params(
            parameters,
            triggered_by,
            daily_bucket="generate",
            skip_daily_quota=skip_daily_quota,
        )
        run_id = deps.create_run(
            run_type=f"generate_{mode}",
            parameters=parameters,
            config_snapshot=self._read_config_snapshot(
                config_path or cfg.sampling_config
            ),
            triggered_by=triggered_by,
            owner=owner,
        )
        deps.log_history(
            "start_generation",
            {"run_id": run_id, "mode": mode, "data_id": data_id},
            user=owner,
        )
        job_id = self._next_job_id(f"gen_{mode}")
        rec = deps.JobRecord(job_id, run_id, f"generate_{mode}", -1, owner=owner)

        def starter(rec, gpu):
            parameters["gpu_id"] = gpu
            try:
                # Web Dynamic uses the current fast schedule, not a hardcoded step label.
                patches = deps.build_generation_patches(parameters)
                params_patched = deps.write_patched_config(
                    config_path or cfg.sampling_config,
                    patches,
                    job_id,
                )
                parameters["patched_config"] = params_patched
            except Exception as e:
                # Never silently run the research preset after a patch failure.
                from server.runners.base import mark_failed

                mark_failed(rec, str(e), "generation_config_failed")
                self._release_and_dispatch(rec)
                return
            deps.run_generation(self, rec, parameters, config_path)

        return self._acquire_or_queue(gpu_id, rec, starter)

    def submit_batch_generation(
        self,
        start_id: int = 0,
        end_id: int = 0,
        gpus: Optional[str] = None,
        batch_size: Optional[int] = 5,
        config_path: Optional[str] = None,
        auto_evaluate: bool = True,
        sample_only: bool = False,
        triggered_by: str = "web_ui",
        skip_daily_quota: bool = False,
        medchem: Optional[dict] = None,
    ) -> dict:
        cfg = deps.get_config()
        parameters = {
            "start_id": start_id,
            "end_id": end_id,
            "gpus": gpus,
            "batch_size": batch_size,
            "auto_evaluate": auto_evaluate,
            "sample_only": sample_only,
            "preferred_gpu": None,
        }
        if medchem is not None:
            from server.medchem_schemas import MedchemOptions
            parameters["medchem"] = MedchemOptions(**medchem).dict()
        owner = self._attach_owner_params(
            parameters,
            triggered_by,
            daily_bucket="generate",
            skip_daily_quota=skip_daily_quota,
        )
        run_id = deps.create_run(
            run_type="generate_batch",
            parameters=parameters,
            config_snapshot=self._read_config_snapshot(
                config_path or cfg.sampling_config
            ),
            triggered_by=triggered_by,
            owner=owner,
        )
        deps.log_history("start_batch_generation", {"run_id": run_id}, user=owner)
        job_id = self._next_job_id("gen_batch")
        rec = deps.JobRecord(job_id, run_id, "generate_batch", -1, owner=owner)

        def starter(rec, gpu):
            parameters["gpu_id"] = gpu
            deps.run_batch_generation(self, rec, parameters, config_path)

        return self._acquire_or_queue(None, rec, starter)

    def submit_rerun(
        self,
        run_id: int,
        overrides: Optional[dict] = None,
        triggered_by: str = "web_ui",
        skip_daily_quota: bool = False,
    ) -> dict:
        r = deps.get_run(run_id)
        if r is None:
            return {"error": f"Run {run_id} not found"}
        params = json.loads(r.parameters) if r.parameters else {}
        if overrides:
            params.update(overrides)
        rt = r.run_type
        preferred_gpu = params.get("preferred_gpu")
        if preferred_gpu is None:
            preferred_gpu = params.get("gpu_id")
        if rt == "generate_custom":
            return self.submit_custom_generation(
                medchem=params.get("medchem"),
                protein_path=params["protein_path"],
                ligand_path=params.get("ligand_path"),
                gpu_id=preferred_gpu,
                pocket_radius=params.get("pocket_radius", 10.0),
                num_samples=params.get("num_samples", 5),
                auto_evaluate=params.get("auto_evaluate", True),
                auto_extract=params.get("auto_extract", True),
                remove_fragments=params.get("remove_fragments", True),
                triggered_by=triggered_by,
                skip_daily_quota=skip_daily_quota,
            )
        if rt.startswith("generate_") and rt != "generate_batch":
            mode = params.get("mode") or rt.replace("generate_", "")
            return self.submit_generation(
                medchem=params.get("medchem"),
                mode=mode,
                data_id=params.get("data_id"),
                use_test_set=params.get("use_test_set", True),
                gpu_id=preferred_gpu,
                batch_size=params.get("batch_size", 5),
                auto_evaluate=params.get("auto_evaluate", True),
                auto_extract=params.get("auto_extract", True),
                remove_fragments=params.get("remove_fragments", True),
                max_samples=params.get("max_samples", 5),
                vina_timeout=params.get("vina_timeout", 20),
                triggered_by=triggered_by,
                skip_daily_quota=skip_daily_quota,
            )
        if rt == "generate_batch":
            return self.submit_batch_generation(
                start_id=params.get("start_id", 0), end_id=params.get("end_id", 0),
                gpus=params.get("gpus"), batch_size=params.get("batch_size", 5),
                auto_evaluate=params.get("auto_evaluate", True),
                sample_only=params.get("sample_only", False),
                medchem=params.get("medchem"), triggered_by=triggered_by,
                skip_daily_quota=skip_daily_quota,
            )
        if rt in {"medchem", "medchem_dock"}:
            from server.medchem_schemas import MedchemRequest
            public = {key: params[key] for key in MedchemRequest.__fields__ if key in params}
            saved = json.loads(r.parameters or "{}")
            internal = {}
            # Only the stored orchestration may supply internal paths/limits.
            if public.get("source_run_id") == saved.get("source_run_id"):
                internal = {key: saved[key] for key in (
                    "source_pt", "source_pts", "source_label", "auto_evaluate",
                    "auto_extract", "max_records", "protein_root", "remove_fragments",
                ) if key in saved}
            return self.submit_medchem(**MedchemRequest(**public).dict(),
                                      **internal,
                                      triggered_by=triggered_by,
                                      skip_daily_quota=skip_daily_quota)
        if rt == "evaluate":
            return self.submit_evaluation(
                pt_path=params["pt_path"],
                protein_root=params.get("protein_root"),
                vina_modes=params.get("vina_modes", "score_only"),
                max_samples=params.get("max_samples"),
                vina_timeout=params.get("vina_timeout", 20),
                triggered_by=triggered_by,
                skip_daily_quota=skip_daily_quota,
            )
        if rt == "extract":
            return self.submit_extraction(
                pt_path=params["pt_path"],
                protein_root=params.get("protein_root"),
                remove_fragments=params.get("remove_fragments", True),
                triggered_by=triggered_by,
            )
        if rt.startswith("scaffold_") and rt != "scaffold_cascade":
            saved = params.get("scaffold_request")
            if not saved:
                return {
                    "error": "Original scaffold parameters are incomplete; submit a new task"
                }
            saved = dict(saved)
            if overrides:
                saved.update(overrides)
            # Same public schema and geometry checks as a fresh request.
            from server.scaffold_context import load_context, preview_context
            from server.schemas import ScaffoldRequest

            try:
                saved = ScaffoldRequest(**saved).dict()
                if saved["placement"]["mode"] == "custom":
                    preview_context(saved)
                elif saved["scaffold_source"] in ("smarts", "custom", "atom_indices"):
                    load_context(saved)
            except ValueError as exc:
                return {"error": str(exc)}
            saved["triggered_by"] = triggered_by
            saved["skip_daily_quota"] = skip_daily_quota
            return self.submit_scaffold(**saved)
        return {"error": f"Re-run not supported for run_type={rt}"}

    def submit_custom_generation(
        self,
        protein_path: str,
        ligand_path: Optional[str] = None,
        gpu_id: Optional[int] = None,
        pocket_radius: float = 10.0,
        num_samples: int = 5,
        config_path: Optional[str] = None,
        auto_evaluate: bool = True,
        auto_extract: bool = True,
        remove_fragments: bool = True,
        triggered_by: str = "web_ui",
        skip_daily_quota: bool = False,
        medchem: Optional[dict] = None,
    ) -> dict:
        pocket_name = os.path.splitext(os.path.basename(protein_path))[0]
        parameters = {
            "protein_path": protein_path,
            "ligand_path": ligand_path,
            "pocket_radius": pocket_radius,
            "num_samples": num_samples,
            "auto_evaluate": auto_evaluate,
            "auto_extract": auto_extract,
            "remove_fragments": remove_fragments,
            "pocket_name": pocket_name,
            "max_samples": 5,
            "vina_timeout": 20,
            "preferred_gpu": gpu_id,
        }
        if medchem is not None:
            from server.medchem_schemas import MedchemOptions
            parameters["medchem"] = MedchemOptions(**medchem).dict()
        owner = self._attach_owner_params(
            parameters,
            triggered_by,
            daily_bucket="generate",
            skip_daily_quota=skip_daily_quota,
        )
        run_id = deps.create_run(
            run_type="generate_custom",
            parameters=parameters,
            triggered_by=triggered_by,
            owner=owner,
        )
        deps.log_history(
            "start_custom_generation",
            {"run_id": run_id, "protein_path": protein_path},
            user=owner,
        )
        job_id = self._next_job_id("gen_custom")
        rec = deps.JobRecord(job_id, run_id, "generate_custom", -1, owner=owner)

        def starter(rec, gpu):
            parameters["gpu_id"] = gpu
            deps.run_custom_generation(self, rec, parameters, config_path)

        return self._acquire_or_queue(gpu_id, rec, starter)
