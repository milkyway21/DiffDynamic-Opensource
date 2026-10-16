"""Download pinned checkpoints and verify them before making them available."""

import json
import os
import tempfile
import urllib.request
from pathlib import Path

from .publishing import checksum


def ensure_weights(root, download=False, include_optional=False):
    root = Path(root).resolve()
    manifest = json.loads((root / "weights-manifest.json").read_text())
    results = []
    for item in manifest["weights"]:
        if not item["required"] and not include_optional:
            continue
        path = root / item["path"]
        if root not in Path(os.path.abspath(str(path))).parents:
            raise ValueError("Weight path escapes the project root")
        if path.exists():
            valid = checksum(path) == item["sha256"]
            results.append({"name": item["name"], "valid": valid, "downloaded": False})
            # A mismatching existing file may be a user's own checkpoint.
            if not valid:
                raise ValueError(
                    "Existing weight has a different checksum: " + str(path)
                )
            continue
        if not download:
            results.append({"name": item["name"], "valid": False, "missing": True})
            continue
        if root not in path.resolve().parents:
            raise ValueError("Weight download would write outside the project root")
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=".download-", dir=str(path.parent))
        try:
            with os.fdopen(fd, "wb") as output:
                with urllib.request.urlopen(item["url"], timeout=60) as response:
                    while True:
                        block = response.read(1024 * 1024)
                        if not block:
                            break
                        output.write(block)
            if checksum(temporary) != item["sha256"]:
                raise ValueError("Downloaded weight checksum mismatch: " + item["name"])
            # Hard-link creation is atomic and refuses to overwrite a concurrent file.
            os.link(temporary, str(path))
            results.append({"name": item["name"], "valid": True, "downloaded": True})
        finally:
            Path(temporary).unlink()
    return results
