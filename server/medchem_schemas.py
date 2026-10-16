"""Validated settings shared by standalone and generated medchem jobs."""

import math
from typing import Literal, Optional, Tuple

from pydantic import BaseModel, Field, root_validator, validator


class MedchemEvaluation(BaseModel):
    mode: Literal["chemistry_3d", "vina_dock"] = "chemistry_3d"
    exhaustiveness: int = Field(8, ge=1, le=32)
    timeout_seconds: int = Field(120, ge=5, le=600)
    seed: int = Field(42, ge=0, le=2**31 - 1)
    center_xyz: Optional[Tuple[float, float, float]] = None
    size_xyz: Optional[Tuple[float, float, float]] = None

    @validator("center_xyz", "size_xyz")
    def finite_xyz(cls, value):
        if value is not None:
            if not all(math.isfinite(x) for x in value):
                raise ValueError("Coordinates must be finite")
        return value

    @validator("size_xyz")
    def bounded_size(cls, value):
        if value is not None and not all(10 <= x <= 40 for x in value):
            raise ValueError("Box dimensions must be between 10 and 40 Angstrom")
        return value

    @root_validator(skip_on_failure=True)
    def complete_box(cls, values):
        if (values.get("center_xyz") is None) != (values.get("size_xyz") is None):
            raise ValueError("Provide both box center and size")
        return values

    class Config:
        extra = "forbid"


class MedchemOptions(BaseModel):
    operation: Literal["check", "repair", "propose"] = "propose"
    profile: Literal["local", "aggressive"] = "local"
    top_k: int = Field(3, ge=1, le=10)
    min_similarity: float = Field(0.55, ge=0, le=1, allow_inf_nan=False)
    min_heavy_delta: int = Field(-3, ge=-10, le=0)
    max_heavy_delta: int = Field(4, ge=0, le=10)
    time_budget_seconds: float = Field(30, ge=1, le=120, allow_inf_nan=False)
    evaluation: MedchemEvaluation = Field(default_factory=MedchemEvaluation)

    class Config:
        extra = "forbid"


class MedchemRequest(BaseModel):
    input_path: Optional[str] = None
    smiles: Optional[str] = Field(None, max_length=200000)
    source_run_id: Optional[int] = Field(None, gt=0)
    protein_path: Optional[str] = None
    reference_ligand_path: Optional[str] = None
    input_aligned: bool = False
    options: MedchemOptions = Field(default_factory=MedchemOptions)

    @root_validator(skip_on_failure=True)
    def validate_source(cls, values):
        if sum(bool(values.get(k)) for k in ("input_path", "smiles", "source_run_id")) != 1:
            raise ValueError("Choose one file, SMILES text, or historical run")
        smiles = values.get("smiles")
        if smiles and len([s for s in smiles.splitlines() if s.strip()]) > 100:
            raise ValueError("At most 100 input records are allowed")
        opts = values.get("options")
        if opts and opts.evaluation.mode == "vina_dock" and not values.get("source_run_id"):
            if not values.get("protein_path"):
                raise ValueError("Docking requires a protein")
            if not values.get("reference_ligand_path") and opts.evaluation.center_xyz is None:
                raise ValueError("Docking requires a reference ligand or explicit box")
        return values

    class Config:
        extra = "forbid"
