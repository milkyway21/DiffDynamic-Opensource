"""Job configuration builders shared by submission and regression checks."""

from server.job_configs.generation import build_generation_patches
from server.job_configs.optimization import build_optimization_patches
from server.job_configs.scaffold import build_scaffold_patches

__all__ = [
    "build_generation_patches",
    "build_optimization_patches",
    "build_scaffold_patches",
]
