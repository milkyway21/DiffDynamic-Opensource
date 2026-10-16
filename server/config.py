"""Runtime configuration — paths, GPU list, server settings.

Separate from sampling.yml (which controls generation behavior).
This module manages infrastructure config for the web server.
"""

import json
import os
import subprocess
from dataclasses import dataclass, field, asdict
from typing import List, Optional
from diffdynamic.io.paths import project_root


_PROJECT_ROOT = str(project_root())


@dataclass
class ServerConfig:
    # Paths
    diffdynamic_root: str = os.environ.get("DD_ROOT", _PROJECT_ROOT)
    data_root: str = os.environ.get("DD_DATA", os.path.join(_PROJECT_ROOT, "data"))
    output_root: str = os.environ.get("DD_OUTPUT", os.path.join(_PROJECT_ROOT, "outputs"))
    pretrained_models: str = os.environ.get("DD_MODELS", os.path.join(_PROJECT_ROOT, "pretrained_models"))
    sampling_config: str = os.environ.get("DD_SAMPLING_YML", "")
    upload_dir: str = os.environ.get("DD_UPLOAD", "")

    # GPU — web jobs prefer default_gpu_id when free, then other gpu_ids
    gpu_ids: List[int] = field(default_factory=lambda: _detect_gpus())
    default_gpu_id: int = 5

    # Server — bind loopback only; public access goes through BianLab gateway
    host: str = "127.0.0.1"
    port: int = 7860
    max_concurrent_jobs: int = 2
    job_timeout: int = 10800  # 3 hours
    # Web caps (runtime-tunable via PUT /api/config by admin; 0 disables the cap)
    max_jobs_per_user: int = 2        # running + pending jobs per owner
    max_pending_jobs: int = 16        # global waiting-queue cap
    max_submit_per_min: int = 12      # per-user sliding window on job submission
    max_upload_per_min: int = 10      # per-user sliding window on /api/upload
    max_upload_bytes: int = 52428800  # 50MB per uploaded file
    # Daily caps (Asia/Shanghai). 0 disables. Admins skip these in the scheduler.
    max_generate_per_day: int = 1     # generate / scaffold / optimization
    max_eval_per_day: int = 1         # evaluate / pocket-eval
    # Internal key is stored in server/.internal_api_key, not in this JSON.

    # Conda
    conda_env: str = "diffdynamic"
    python_bin: str = ""

    def __post_init__(self):
        if not self.sampling_config:
            self.sampling_config = os.path.join(self.diffdynamic_root, "configs", "sampling.yml")
        if not self.upload_dir:
            self.upload_dir = os.path.join(self.diffdynamic_root, "uploads")
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

PATH_ENVIRONMENT = {
    "DD_ROOT": "diffdynamic_root",
    "DD_DATA": "data_root",
    "DD_OUTPUT": "output_root",
    "DD_MODELS": "pretrained_models",
    "DD_SAMPLING_YML": "sampling_config",
    "DD_UPLOAD": "upload_dir",
}


def load_config() -> ServerConfig:
    """Load saved settings, with explicit deployment paths taking precedence."""
    data = {}
    if os.path.isfile(CONFIG_PATH):
        with open(CONFIG_PATH) as f:
            data = json.load(f)
    from dataclasses import fields as dc_fields
    known = {f.name for f in dc_fields(ServerConfig)}
    filtered = {k: v for k, v in data.items() if k in known}
    for variable, name in PATH_ENVIRONMENT.items():
        if variable in os.environ:
            filtered[name] = os.environ[variable]
    return ServerConfig(**filtered)


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
