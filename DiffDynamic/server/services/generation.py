"""Generation / evaluation / history service facades."""

from typing import Optional

from server.jobs import get_scheduler


def start_generation(**kwargs):
    return get_scheduler().submit_generation(**kwargs)


def start_batch_generation(**kwargs):
    return get_scheduler().submit_batch_generation(**kwargs)


def start_custom_generation(**kwargs):
    return get_scheduler().submit_custom_generation(**kwargs)


def start_evaluation(**kwargs):
    return get_scheduler().submit_evaluation(**kwargs)


def start_extraction(**kwargs):
    return get_scheduler().submit_extraction(**kwargs)


def start_pocket_eval(**kwargs):
    return get_scheduler().submit_pocket_eval(**kwargs)


def start_optimization(**kwargs):
    return get_scheduler().submit_optimization(**kwargs)


def start_scaffold(**kwargs):
    return get_scheduler().submit_scaffold(**kwargs)


def start_scaffold_cascade(**kwargs):
    return get_scheduler().submit_scaffold_cascade(**kwargs)


def start_rerun(run_id: int, overrides: Optional[dict] = None, triggered_by: str = "web_ui"):
    return get_scheduler().submit_rerun(run_id, overrides=overrides, triggered_by=triggered_by)


def job_status(job_id: str):
    return get_scheduler().get_status(job_id)


def cancel_job(job_id: str) -> bool:
    return get_scheduler().cancel(job_id)


def list_jobs(status: Optional[str] = None):
    return get_scheduler().list_jobs(status)
