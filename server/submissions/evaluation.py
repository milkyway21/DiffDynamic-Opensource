"""Evaluation job submissions, independent of queue/resource implementation."""

import os
from typing import Optional

from server.submissions.dependencies import dependencies as deps


class EvaluationJobs:
    """Prepare and dispatch evaluation jobs through JobScheduler hooks."""

    def submit_evaluation(
        self,
        pt_path: str,
        protein_root: Optional[str] = None,
        # 默认 score_only：dock/minimize 由调用方显式要求才跑（慢）。
        vina_modes: str = "score_only",
        max_samples: Optional[int] = None,
        vina_timeout: int = 20,
        pocket_name: Optional[str] = None,
        triggered_by: str = "web_ui",
        skip_daily_quota: bool = False,
    ) -> dict:
        parameters = {
            "pt_path": pt_path,
            "protein_root": protein_root,
            "vina_modes": vina_modes,
            "max_samples": max_samples,
            "vina_timeout": vina_timeout,
            "pocket_name": pocket_name,
        }
        owner = self._attach_owner_params(
            parameters,
            triggered_by,
            daily_bucket="eval",
            skip_daily_quota=skip_daily_quota,
        )
        run_id = deps.create_run(
            run_type="evaluate",
            parameters=parameters,
            triggered_by=triggered_by,
            owner=owner,
        )
        deps.log_history(
            "start_evaluation", {"run_id": run_id, "pt_path": pt_path}, user=owner
        )
        job_id = self._next_job_id("eval")
        rec = deps.JobRecord(job_id, run_id, "evaluate", -1, owner=owner)
        return self._submit_cpu(rec, deps.run_evaluation, self, rec, parameters)

    def submit_extraction(
        self,
        pt_path: str,
        protein_root: Optional[str] = None,
        eval_dir: Optional[str] = None,
        remove_fragments: bool = True,
        max_samples: Optional[int] = None,
        vina_timeout: Optional[int] = None,
        vina_modes: Optional[str] = None,
        triggered_by: str = "web_ui",
        skip_daily_quota: bool = False,
        parent_run_id: Optional[int] = None,
    ) -> dict:
        parameters = {
            "pt_path": pt_path,
            "protein_root": protein_root,
            "eval_dir": eval_dir,
            "remove_fragments": remove_fragments,
            "max_samples": max_samples,
            "vina_timeout": vina_timeout,
            "vina_modes": vina_modes,
            "parent_run_id": parent_run_id,
        }
        owner = self._attach_owner_params(
            parameters,
            triggered_by,
            daily_bucket=None,
            skip_daily_quota=skip_daily_quota,
        )
        run_id = deps.create_run(
            run_type="extract",
            parameters=parameters,
            triggered_by=triggered_by,
            owner=owner,
        )
        deps.log_history(
            "start_extraction", {"run_id": run_id, "pt_path": pt_path}, user=owner
        )
        job_id = self._next_job_id("extract")
        rec = deps.JobRecord(job_id, run_id, "extract", -1, owner=owner)
        return self._submit_cpu(rec, deps.run_extraction, self, rec, parameters)

    def submit_pocket_eval(
        self,
        protein_path: str,
        ligand_path: str,
        pt_path: Optional[str] = None,
        generate_first: bool = False,
        gpu_id: Optional[int] = None,
        batch_size: Optional[int] = 5,
        pocket_radius: float = 10.0,
        max_samples: Optional[int] = None,
        triggered_by: str = "web_ui",
        skip_daily_quota: bool = False,
    ) -> dict:
        pocket_name = os.path.splitext(os.path.basename(protein_path))[0]
        parameters = {
            "pt_path": pt_path,
            "protein_path": protein_path,
            "ligand_path": ligand_path,
            "generate_first": generate_first,
            "batch_size": batch_size,
            "pocket_radius": pocket_radius,
            "max_samples": max_samples,
            "pocket_name": pocket_name,
            "preferred_gpu": gpu_id,
        }
        owner = self._attach_owner_params(
            parameters,
            triggered_by,
            daily_bucket="eval",
            skip_daily_quota=skip_daily_quota,
        )
        run_id = deps.create_run(
            run_type="pocket_eval",
            parameters=parameters,
            triggered_by=triggered_by,
            owner=owner,
        )
        deps.log_history(
            "start_pocket_eval",
            {"run_id": run_id, "protein_path": protein_path},
            user=owner,
        )
        job_id = self._next_job_id("pocket_eval")
        rec = deps.JobRecord(job_id, run_id, "pocket_eval", -1, owner=owner)

        def starter(rec, gpu):
            parameters["gpu_id"] = gpu
            deps.run_pocket_evaluation(self, rec, parameters)

        return self._acquire_or_queue(gpu_id, rec, starter)
