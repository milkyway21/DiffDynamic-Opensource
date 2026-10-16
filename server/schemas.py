"""Validated public request contracts shared by HTTP routes and replay.

This module has no scheduler or application imports.
"""

from typing import List, Literal, Optional, Tuple

from pydantic import BaseModel, Field, root_validator, validator


from server.medchem_schemas import MedchemOptions


class DefaultSamplingRequest(BaseModel):
    config_path: Optional[str] = None

    @validator("config_path")
    def default_sampling_only(cls, value):
        if value and value.strip():
            raise ValueError("Web tasks use the server default sampling configuration")
        return None


class GenerateRequest(DefaultSamplingRequest):
    medchem: Optional[MedchemOptions] = None
    mode: Literal["dynamic"] = "dynamic"
    data_id: Optional[int] = None
    use_test_set: bool = True
    gpu_id: Optional[int] = None
    batch_size: Optional[int] = 5
    config_path: Optional[str] = None
    auto_evaluate: bool = True
    auto_extract: bool = True
    remove_fragments: bool = True
    max_samples: Optional[int] = 5
    vina_timeout: int = 20


class BatchGenerateRequest(DefaultSamplingRequest):
    medchem: Optional[MedchemOptions] = None
    start_id: int = 0
    end_id: int = 0
    gpus: Optional[str] = None
    batch_size: Optional[int] = 5
    config_path: Optional[str] = None
    auto_evaluate: bool = True
    sample_only: bool = False


class RerunRequest(BaseModel):
    overrides: Optional[dict] = None
    triggered_by: str = "web_ui"


class CustomGenerateRequest(DefaultSamplingRequest):
    medchem: Optional[MedchemOptions] = None
    protein_path: str
    ligand_path: Optional[str] = None
    gpu_id: Optional[int] = None
    pocket_radius: float = 10.0
    num_samples: int = 5
    config_path: Optional[str] = None
    auto_evaluate: bool = True
    auto_extract: bool = True
    remove_fragments: bool = True


class GenerationContextRequest(DefaultSamplingRequest):
    protein_path: Optional[str] = None
    ligand_path: Optional[str] = None
    data_id: int = Field(0, ge=0, le=10_000_000)
    pocket_radius: float = Field(10.0, ge=4.0, le=20.0)

    @root_validator(skip_on_failure=True)
    def ligand_requires_protein(cls, values):
        if values.get("ligand_path") and not values.get("protein_path"):
            raise ValueError("Reference ligand requires a custom protein")
        return values


class EvaluateRequest(BaseModel):
    pt_path: str
    protein_root: Optional[str] = None
    # 默认只跑 score_only（快）；dock/minimize 需前端显式勾选，
    # dock 全局搜索太慢会拖垮自动链（评估→提取）导致前端提取等待超时。
    vina_modes: str = "score_only"
    max_samples: Optional[int] = None
    vina_timeout: int = 20


class ExtractRequest(BaseModel):
    pt_path: str
    protein_root: Optional[str] = None
    remove_fragments: bool = True
    parent_run_id: Optional[int] = None
    max_samples: Optional[int] = 5
    vina_timeout: Optional[int] = 20


class PocketEvalRequest(BaseModel):
    pt_path: Optional[str] = None
    protein_path: str
    ligand_path: str
    generate_first: bool = False
    gpu_id: Optional[int] = None
    batch_size: Optional[int] = 5
    pocket_radius: float = 10.0
    max_samples: Optional[int] = None


class OptimizationRequest(DefaultSamplingRequest):
    data_id: Optional[int] = None
    protein_path: Optional[str] = None
    ligand_path: Optional[str] = None
    molecule_path: Optional[str] = None
    use_test_set: bool = True
    gpu_id: Optional[int] = None
    config_path: Optional[str] = None
    auto_evaluate: bool = False
    auto_extract: bool = False
    remove_fragments: bool = True
    num_samples: int = 2
    start_t: int = 16
    stride: int = 2
    step_size: float = 0.2
    cycles: int = 5
    schedule: str = "linear"
    after_dynamic: bool = False
    min_qed: float = 0.3
    min_sa: float = 0.3
    max_survivors_per_cycle: int = 10
    max_survivors_total: int = 300


class ScaffoldCascadeRequest(DefaultSamplingRequest):
    data_id: Optional[int] = None
    protein_path: Optional[str] = None
    ligand_path: Optional[str] = None
    gpu_id: Optional[int] = None
    config_path: Optional[str] = None
    auto_evaluate: bool = False
    auto_extract: bool = False
    remove_fragments: bool = True
    samples_per_round: int = 5
    rounds: int = 2


class ScaffoldRegion(BaseModel):
    anchor_atom_index: int = Field(..., ge=0)
    center_xyz: Tuple[float, float, float]
    atom_count: int = Field(..., ge=1, le=50)
    method: Literal["gaussian", "corridor"] = "gaussian"
    sigma_axial: float = Field(1.0, gt=0, le=10, allow_inf_nan=False)
    sigma_lateral: float = Field(1.0, gt=0, le=10, allow_inf_nan=False)

    @validator("center_xyz")
    def finite_center(cls, value):
        import math

        if not all(math.isfinite(v) for v in value):
            raise ValueError("Center coordinates must be finite")
        return value


class ScaffoldPlacement(BaseModel):
    mode: Literal["auto", "custom"] = "auto"
    regions: List[ScaffoldRegion] = Field(default_factory=list, max_items=50)

    @root_validator(skip_on_failure=True)
    def exact_budget(cls, values):
        regions = values.get("regions", [])
        if values.get("mode") == "custom":
            if not regions or sum(r.atom_count for r in regions) > 50:
                raise ValueError("Custom regions require a total of 1–50 atoms")
        elif regions:
            raise ValueError("Automatic placement must not include custom regions")
        return values


class ScaffoldRequest(DefaultSamplingRequest):
    workflow: Literal["legacy", "repaint"] = "legacy"
    seed: int = Field(42, ge=0, le=4294967295)
    placement: ScaffoldPlacement = Field(default_factory=ScaffoldPlacement)
    data_id: Optional[int] = None
    protein_path: Optional[str] = None
    ligand_path: Optional[str] = None
    molecule_path: Optional[str] = None
    use_test_set: bool = True
    gpu_id: Optional[int] = None
    config_path: Optional[str] = None
    auto_evaluate: bool = False
    auto_extract: bool = False
    remove_fragments: bool = True
    scaffold_mode: str = "grow"  # grow | evolve | dynamic_locked | prudent
    scaffold_source: str = "auto_murcko"
    scaffold_atom_indices: Optional[str] = None
    scaffold_smarts: Optional[str] = None
    fix_scaffold_pos: bool = True
    fix_scaffold_type: bool = True
    after_dynamic: bool = False
    # Legacy keeps repair enabled; the latest RePaint preset defaults it off.
    scaffold_enable_refine: Optional[bool] = None
    qed_weight: float = 1.0
    sa_weight: float = 1.0
    diversity_weight: float = 0.5
    rmsd_penalty_weight: float = 0.0
    min_qed: float = 0.15
    min_sa: float = 0.15
    grow_num_samples: int = 200
    grow_start_t: int = 357  # 100口袋最优(贝叶斯Round18)
    grow_stride: int = 15
    grow_step_size: float = 0.3262  # 100口袋最优(贝叶斯Round18)
    grow_lambda_a: int = 47  # 100口袋最优(贝叶斯Round18)
    grow_lambda_b: int = 11  # 100口袋最优(贝叶斯Round18)
    grow_n_extra_mode: str = "pocket_prior"  # 100口袋最优: +7.4%综合分
    grow_n_extra_fixed: int = 8
    grow_n_extra_min: int = 3
    grow_n_extra_max: int = 20
    grow_filter_incomplete: bool = False  # 100口袋最优: False(避免有效率暴跌)
    grow_min_qed: float = 0.2  # 100口袋最优grow阈值
    grow_min_sa: float = 0.2  # 100口袋最优grow阈值
    evolve_population_size: int = 50
    evolve_n_generations: int = 5
    evolve_children_per_parent: int = 10
    evolve_start_t_high: int = 200
    evolve_start_t_low: int = 30
    evolve_stride: int = 2
    evolve_step_size: float = 0.2
    evolve_lambda_a: int = 10
    evolve_lambda_b: int = 1
    diversity_filter_enable: bool = True
    max_tanimoto: float = 0.9
    # Scaffold-Prudent
    prudent_n_generations: int = 5
    prudent_n_chains_per_seed: int = 4
    prudent_advance_top_k: int = 2
    prudent_renoise_t: int = 600
    prudent_qed_weight: float = 0.2
    prudent_sa_weight: float = 0.2
    prudent_vina_weight: float = 0.6
    prudent_lipinski_weight: float = 0.0
    prudent_lilly_weight: float = 0.0
    prudent_vina_exhaustiveness: int = 8
    prudent_min_qed_for_docking: float = 0.15
    prudent_min_sa_for_docking: float = 0.15
    # 门控模式：adaptive=按参考配体自适应；manual=使用下方人工门控数值
    gate_mode: str = "adaptive"  # adaptive | manual
    adaptive_gates_enable: Optional[bool] = None  # 兼容旧字段；None 时由 gate_mode 决定
    adaptive_gates_actives: Optional[str] = None
    # 人工门控（gate_mode=manual 时写入 prudent）
    manual_max_logp: Optional[float] = 5.0
    manual_max_molwt: Optional[float] = 850.0
    manual_max_heavy_atoms: Optional[int] = 60
    manual_max_rings: Optional[int] = 7

    @root_validator(skip_on_failure=True)
    def compatible_workflow(cls, values):
        mode = values.get("scaffold_mode")
        if values.get("scaffold_enable_refine") is None:
            values["scaffold_enable_refine"] = values.get("workflow") != "repaint"
        if values.get("workflow") == "repaint" and mode not in (
            "dynamic_locked",
            "prudent",
        ):
            raise ValueError("RePaint requires Dynamic Locked or Prudent")
        placement = values.get("placement")
        if placement and placement.mode == "custom":
            if mode == "evolve" or values.get("scaffold_source") == "none":
                raise ValueError(
                    "Custom atom placement requires a scaffold and a grow-like mode"
                )
        if values.get("protein_path"):
            if not (values.get("molecule_path") or values.get("ligand_path")):
                raise ValueError("Custom pocket requires a reference molecule")
            values["data_id"] = None
            values["use_test_set"] = False
        return values


class ScaffoldPreviewRequest(ScaffoldRequest):
    @root_validator(pre=True)
    def require_atom_count(cls, values):
        if values.get("scaffold_mode") == "evolve":
            raise ValueError("Evolve has no initial extra-atom placement")
        placement = values.get("placement") or {}
        if not isinstance(placement, dict):
            raise ValueError("Placement must be an object")
        if placement.get("mode") != "custom":
            count = values.get("grow_n_extra_fixed")
            if (
                values.get("grow_n_extra_mode") != "fixed"
                or type(count) is not int
                or not 1 <= count <= 50
            ):
                raise ValueError(
                    "Set an integer extra-atom count from 1 to 50 before previewing"
                )
        return values
