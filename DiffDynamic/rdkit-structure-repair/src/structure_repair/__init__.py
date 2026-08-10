"""rdkit-structure-repair: deterministic RDKit molecular structure repair + medchem optimize."""

from .models import (
    AtomEdit,
    BondEdit,
    OptimizeResult,
    RepairCandidate,
    RepairResult,
    StructureIssue,
    TransformApplication,
    OPT_STATUS_OPTIMIZED,
    OPT_STATUS_REJECTED,
    OPT_STATUS_UNCHANGED,
    STATUS_AMBIGUOUS,
    STATUS_REJECTED,
    STATUS_REPAIRED,
    STATUS_UNCHANGED,
)
from .repair_engine import repair_molecule
from .optimize import optimize_molecule
from .detector import detect_issues
from .config import load_config, default_config_path, load_optimize_config
from .atom_mapping import assign_atom_maps
from .staged_sanitize import staged_sanitize

__all__ = [
    "repair_molecule",
    "optimize_molecule",
    "detect_issues",
    "load_config",
    "load_optimize_config",
    "default_config_path",
    "assign_atom_maps",
    "staged_sanitize",
    "StructureIssue",
    "BondEdit",
    "AtomEdit",
    "RepairCandidate",
    "RepairResult",
    "OptimizeResult",
    "TransformApplication",
    "STATUS_UNCHANGED",
    "STATUS_REPAIRED",
    "STATUS_AMBIGUOUS",
    "STATUS_REJECTED",
    "OPT_STATUS_OPTIMIZED",
    "OPT_STATUS_UNCHANGED",
    "OPT_STATUS_REJECTED",
]

__version__ = "0.1.0"
