"""Runtime configuration — paths, GPU list, server settings.

Separate from sampling.yml (which controls generation behavior).
This module manages infrastructure config for the web server.
"""

import json
import os
import subprocess
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional


@dataclass
class ServerConfig:
    # Paths
    diffdynamic_root: str = os.environ.get("DD_ROOT", "/data/ye/DiffDynamic")
    data_root: str = os.environ.get("DD_DATA", "/data/ye/DiffDynamic/data")
    output_root: str = os.environ.get("DD_OUTPUT", "/data/ye/DiffDynamic/outputs")
    pretrained_models: str = os.environ.get("DD_MODELS", "/data/ye/DiffDynamic/pretrained_models")
    sampling_config: str = os.environ.get("DD_SAMPLING_YML", "")

    # GPU
    gpu_ids: List[int] = field(default_factory=lambda: _detect_gpus())

    # Server
    host: str = "0.0.0.0"
    port: int = 7860
    max_concurrent_jobs: int = 2
    job_timeout: int = 10800  # 3 hours

    # Conda
    conda_env: str = "diffdynamic"
    python_bin: str = ""

    def __post_init__(self):
        if not self.sampling_config:
            self.sampling_config = os.path.join(self.diffdynamic_root, "configs", "sampling.yml")
        if not self.python_bin:
            self.python_bin = self._resolve_python()

    def _resolve_python(self):
        conda_base = os.environ.get("CONDA_PREFIX_1", os.path.expanduser("~/anaconda3"))
        candidate = os.path.join(conda_base, "envs", self.conda_env, "bin", "python3")
        if os.path.isfile(candidate):
            return candidate
        return "python3"

    def sample_script(self):
        return os.path.join(self.diffdynamic_root, "scripts", "sample_diffusion.py")

    def evaluate_script(self):
        return os.path.join(self.diffdynamic_root, "evaluate_pt_with_correct_reconstruct.py")

    def pocket_eval_script(self):
        return os.path.join(self.diffdynamic_root, "evaluate_pocket_quality.py")

    def extract_script(self):
        return os.path.join(self.diffdynamic_root, "extract_pt_to_sdf_excel.py")

    def scaffold_cascade_script(self):
        return os.path.join(self.diffdynamic_root, "scaffold_cascade_pipeline.py")

    def to_dict(self):
        return asdict(self)


def _detect_gpus():
    """Auto-detect available GPU IDs from nvidia-smi."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            timeout=5, stderr=subprocess.DEVNULL,
        ).decode().strip()
        return sorted(int(x) for x in out.splitlines() if x.strip())
    except Exception:
        return [0]


CONFIG_PATH = os.path.join(os.path.dirname(__file__), "server_config.json")


def load_config() -> ServerConfig:
    if os.path.isfile(CONFIG_PATH):
        with open(CONFIG_PATH) as f:
            data = json.load(f)
        return ServerConfig(**data)
    return ServerConfig()


def save_config(cfg: ServerConfig):
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg.to_dict(), f, indent=2)


# Singleton
_config: Optional[ServerConfig] = None


def get_config() -> ServerConfig:
    global _config
    if _config is None:
        _config = load_config()
    return _config
