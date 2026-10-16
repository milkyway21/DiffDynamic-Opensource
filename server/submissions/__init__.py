"""Functional submission implementations composed into JobScheduler."""

from server.submissions.medchem import MedchemJobs
from server.submissions.evaluation import EvaluationJobs
from server.submissions.generation import GenerationJobs
from server.submissions.scaffold import ScaffoldJobs

__all__ = ["EvaluationJobs", "GenerationJobs", "ScaffoldJobs", "MedchemJobs"]
