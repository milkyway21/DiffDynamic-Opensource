"""Standalone CPU-only loader. Run exclusively inside the PT sandbox."""

import codecs
import json
import math
import resource
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


class DataContainer:
    """Inert replacement for serialized project/PyG attribute containers."""


def _safe_globals():
    entries = [SimpleNamespace, codecs.encode, np.ndarray, np.dtype,
               np._core.multiarray._reconstruct]
    entries += [(np._core.multiarray._reconstruct, "numpy.core.multiarray._reconstruct")]
    entries += [value for value in vars(np.dtypes).values()
                if isinstance(value, type) and value.__name__.endswith("DType")]
    for name in (
        "datasets.pl_data.ProteinLigandData", "torch_geometric.data.data.Data",
        "torch_geometric.data.data.DataEdgeAttr", "torch_geometric.data.data.DataTensorAttr",
        "torch_geometric.data.data.DataAttr", "torch_geometric.data.data.GlobalAttr",
        "torch_geometric.data.storage.GlobalStorage",
    ):
        entries.append((DataContainer, name))
    return entries


def main():
    # CPU/time/file limits remain enforced even if a parser defect is reached.
    resource.setrlimit(resource.RLIMIT_CPU, (45, 45))
    resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))
    resource.setrlimit(resource.RLIMIT_FSIZE, (256 * 1024**2, 256 * 1024**2))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    torch.set_num_threads(1)
    version = tuple(int(part) for part in torch.__version__.split("+")[0].split(".")[:3])
    if version < (2, 10, 0):
        raise ValueError("Patched PyTorch >= 2.10.0 is required")
    with torch.serialization.safe_globals(_safe_globals()):
        data = torch.load("/input/result.pt", map_location="cpu", weights_only=True)

    output = Path("/output")
    arrays = []
    total_bytes = 0
    nodes = 0

    def encode(value, depth=0):
        nonlocal total_bytes, nodes
        nodes += 1
        if depth > 40 or nodes > 200000:
            raise ValueError("PT object is too deeply nested or too large")
        if value is None or type(value) in (bool, int, str):
            if isinstance(value, str) and len(value) > 100000:
                raise ValueError("PT string exceeds size limit")
            return value
        if type(value) is float:
            if not math.isfinite(value):
                raise ValueError("Non-finite PT value")
            return value
        if isinstance(value, torch.Tensor):
            if value.layout != torch.strided or value.is_quantized:
                raise ValueError("Only dense tensors are supported")
            value = value.detach().cpu().numpy()
        if isinstance(value, np.ndarray):
            if value.dtype.kind not in "biuf" or value.ndim > 6:
                raise ValueError("Unsupported array dtype or shape")
            total_bytes += value.nbytes
            if total_bytes > 256 * 1024**2 or not np.isfinite(value).all():
                raise ValueError("Array exceeds size limit or contains non-finite values")
            name = f"array_{len(arrays)}.npy"
            np.save(output / name, value, allow_pickle=False)
            arrays.append(name)
            return {"__array__": name}
        if isinstance(value, np.generic):
            return encode(value.item(), depth + 1)
        if isinstance(value, (DataContainer, SimpleNamespace)):
            values = vars(value)
            if "_store" in values:
                values = vars(values["_store"]).get("_mapping", {})
            else:
                values = values.get("_mapping", values)
            return {"__attributes__": encode(values, depth + 1)}
        if isinstance(value, (list, tuple)):
            return [encode(item, depth + 1) for item in value]
        if isinstance(value, dict):
            if any(type(key) not in (str, int) or
                   (isinstance(key, str) and key.startswith("__")) for key in value):
                raise ValueError("PT keys must be ordinary strings or atom indices")
            if any(type(key) is int for key in value):
                return {"__mapping__": [[key, encode(item, depth + 1)]
                                        for key, item in value.items()]}
            return {key: encode(item, depth + 1) for key, item in value.items()}
        raise ValueError("Unsupported serialized object")

    tree = encode(data)
    if not isinstance(tree, (dict, list)):
        raise ValueError("PT result must be a dictionary or a sample list")
    manifest = {"version": 1, "tree": tree}
    text = json.dumps(manifest, allow_nan=False)
    if len(text) > 8 * 1024**2:
        raise ValueError("PT manifest exceeds size limit")
    (output / "manifest.json").write_text(text)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        # Deliberately omit paths, payloads and parser traces from public errors.
        print(f"Invalid or unsupported PT result ({type(error).__name__})", file=sys.stderr)
        sys.exit(2)
