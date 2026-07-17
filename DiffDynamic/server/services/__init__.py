"""Thin service layer wrapping the job scheduler for API routes."""

from server.services.generation import (  # noqa: F401
    start_generation,
    start_batch_generation,
    start_custom_generation,
    start_evaluation,
    start_extraction,
    start_pocket_eval,
    start_optimization,
    start_scaffold,
    start_scaffold_cascade,
    start_rerun,
    job_status,
    cancel_job,
    list_jobs,
)

__all__ = [
    "start_generation",
    "start_batch_generation",
    "start_custom_generation",
    "start_evaluation",
    "start_extraction",
    "start_pocket_eval",
    "start_optimization",
    "start_scaffold",
    "start_scaffold_cascade",
    "start_rerun",
    "job_status",
    "cancel_job",
    "list_jobs",
]
