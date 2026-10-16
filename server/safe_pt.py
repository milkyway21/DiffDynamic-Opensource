"""Normalize untrusted PT results outside the scientific Python environment."""

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from server.auth import owner_slug, user_output_root, user_upload_root
from server import config


def _within(path, root):
    return os.path.commonpath([os.path.realpath(path), os.path.realpath(root)]) == os.path.realpath(root)


def is_generated_pt(path, owner=None):
    cfg = config.get_config()
    root = user_output_root(cfg.output_root, owner) if owner else cfg.output_root
    return _within(path, root)


def _sandbox_command(source, output):
    root = Path(__file__).resolve().parent.parent
    python = Path(os.environ.get("DD_PT_LOADER_PYTHON", root / ".cache/dd_pt_sandbox/bin/python"))
    if not python.is_file() or not shutil.which("bwrap"):
        raise ValueError("Secure PT loader is unavailable; external PT loading is disabled")
    environment = python.parent.parent.resolve()
    worker = Path(__file__).with_name("pt_sandbox_worker.py")
    command = [
        "bwrap", "--unshare-all", "--die-with-parent", "--new-session",
        "--clearenv", "--tmpfs", "/", "--proc", "/proc", "--dev", "/dev",
    ]
    # Only runtime libraries are visible. The home directory, repository,
    # database, authentication key and GPU device files are never mounted.
    for path in ("/usr", "/lib", "/lib64"):
        if os.path.exists(path):
            command += ["--ro-bind", path, path]
    command += [
        "--ro-bind", str(environment), "/runtime",
        "--ro-bind", str(worker), "/worker.py",
        "--dir", "/input", "--ro-bind", str(source), "/input/result.pt",
        "--bind", str(output), "/output", "--tmpfs", "/tmp",
        "--setenv", "PATH", "/runtime/bin:/usr/bin",
        "--setenv", "PYTHONDONTWRITEBYTECODE", "1",
        "--setenv", "OMP_NUM_THREADS", "1", "--setenv", "OPENBLAS_NUM_THREADS", "1",
        "--chdir", "/tmp", "/runtime/bin/python", "/worker.py",
    ]
    return command


def _decode_manifest(directory):
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink() or manifest_path.stat().st_size > 8 * 1024**2:
        raise ValueError("Invalid PT manifest")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("version") != 1:
        raise ValueError("Invalid PT manifest version")
    total_bytes = 0
    nodes = 0

    def decode(value, depth=0):
        nonlocal total_bytes, nodes
        nodes += 1
        if depth > 40 or nodes > 200000:
            raise ValueError("PT result exceeds structural limits")
        if isinstance(value, list):
            return [decode(item, depth + 1) for item in value]
        if not isinstance(value, dict):
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("Invalid PT scalar")
            if isinstance(value, str) and len(value) > 100000:
                raise ValueError("PT string exceeds size limit")
            return value
        if "__array__" in value:
            name = value["__array__"]
            if not isinstance(name, str) or not re.fullmatch(r"array_[0-9]+\.npy", name):
                raise ValueError("Invalid PT array reference")
            path = directory / name
            if path.is_symlink() or not path.is_file() or path.stat().st_size > 256 * 1024**2:
                raise ValueError("Invalid PT array file")
            with path.open("rb") as stream:
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
                elif version == (2, 0):
                    shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
                else:
                    raise ValueError("Unsupported PT array format")
                expected = math.prod(shape) * dtype.itemsize
                if len(shape) > 6 or dtype.kind not in "biuf" or expected > 256 * 1024**2:
                    raise ValueError("PT array header exceeds limits")
                if expected != path.stat().st_size - stream.tell():
                    raise ValueError("Invalid PT array length")
            array = np.load(path, allow_pickle=False)
            total_bytes += array.nbytes
            if total_bytes > 256 * 1024**2 or array.ndim > 6 or array.dtype.kind not in "biuf" or not np.isfinite(array).all():
                raise ValueError("Invalid PT array content")
            return torch.from_numpy(array.copy())
        if "__attributes__" in value:
            return SimpleNamespace(**decode(value["__attributes__"], depth + 1))
        if "__mapping__" in value:
            result = {}
            for pair in value["__mapping__"]:
                if not isinstance(pair, list) or len(pair) != 2:
                    raise ValueError("Invalid PT mapping")
                key, item = pair
                if type(key) not in (int, str) or (isinstance(key, str) and key.startswith("__")):
                    raise ValueError("Invalid PT mapping key")
                result[key] = decode(item, depth + 1)
            return result
        if any(str(key).startswith("__") for key in value):
            raise ValueError("Invalid PT manifest key")
        return {key: decode(item, depth + 1) for key, item in value.items()}

    return decode(manifest["tree"])


def _validate_file_references(data, owner):
    cfg = config.get_config()
    roots = [cfg.data_root]
    if owner:
        roots += [user_upload_root(cfg.upload_dir, owner), user_output_root(cfg.output_root, owner)]
    else:
        roots += [cfg.upload_dir, cfg.output_root]
    if os.path.isdir(cfg.data_root):
        roots += [os.path.realpath(os.path.join(cfg.data_root, name))
                  for name in os.listdir(cfg.data_root)
                  if os.path.islink(os.path.join(cfg.data_root, name))]

    def check(value):
        if isinstance(value, SimpleNamespace):
            value = vars(value)
        if isinstance(value, list):
            for item in value:
                check(item)
        elif isinstance(value, dict):
            for key, item in value.items():
                if key in ("protein_filename", "ligand_filename") and item:
                    text = str(item).replace("\\", "/")
                    if ".." in text.split("/"):
                        raise ValueError("PT reference file escapes the allowed directory")
                    if os.path.isabs(text) and not any(_within(text, root) for root in roots):
                        raise ValueError("PT reference file is not owned by this account")
                    if Path(text).suffix.lower() not in (".pdb", ".sdf", ".mol", ".mol2"):
                        raise ValueError("Unsupported PT reference file")
                else:
                    check(item)
    check(data)


def load_untrusted_pt(path, owner=None):
    """Return plain data/tensors; the host never unpickles the input file."""
    source = Path(path).resolve(strict=True)
    if source.stat().st_size > 50 * 1024**2:
        raise ValueError("External PT exceeds the 50 MB loader limit")
    with tempfile.TemporaryDirectory(prefix="dd_pt_load_") as scratch:
        output = Path(scratch)
        result = subprocess.run(_sandbox_command(source, output),
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60,
                                env={"PATH": "/usr/bin:/bin"})
        if result.returncode:
            raise ValueError("Invalid or unsupported PT result; secure loader rejected the file")
        data = _decode_manifest(output)
    _validate_file_references(data, owner)
    return data


def prepare_job_pt(path, owner):
    """Jobs consume a server-created tensor-only PT, never the uploaded pickle."""
    if is_generated_pt(path, owner):
        return path
    data = load_untrusted_pt(path, owner)
    cfg = config.get_config()
    directory = Path(cfg.diffdynamic_root) / ".cache" / "normalized_pt" / owner_slug(owner)
    directory.mkdir(parents=True, mode=0o700, exist_ok=True)
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    # Keep the original filename for legacy evaluators that infer data_id from
    # result_0.pt / eval_000.pt, while the digest isolates different uploads.
    directory = directory / digest
    directory.mkdir(mode=0o700, exist_ok=True)
    destination = directory / Path(path).name
    # The input may contain attribute containers, but this output only contains
    # dense tensors and primitive values.
    fd, temporary = tempfile.mkstemp(prefix=digest, suffix=".tmp", dir=directory)
    os.close(fd)
    try:
        from diffdynamic.io.normalized_pt import normalized_result
        torch.save(normalized_result(data), temporary)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return str(destination)
