"""Medchem structure optimization layer.

Sits on top of the legality repair layer.  Where ``repair`` may never change
the molecular formula, this layer deliberately does: bioisosteric replacement,
ring closure, heteroatom insertion, metabolic soft-spot blocking, and
oversized-ring opening all change atom counts or connectivity.  Acceptance is
driven by a drug-likeness reward under hard constraints.
"""

from __future__ import annotations

from .catalog import TransformSpec, load_catalog
from .optimize_engine import optimize_molecule
from .reward import compute_properties, compute_reward

__all__ = [
    "TransformSpec",
    "load_catalog",
    "optimize_molecule",
    "compute_properties",
    "compute_reward",
]
