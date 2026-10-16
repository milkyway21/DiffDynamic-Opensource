"""Build a source-only release candidate without touching Git's index."""

import fnmatch
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path


PRIVATE_DIRS = {
    ".git",
    "local",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".cache",
    "node_modules",
    "outputs",
    "uploads",
    "docktmp",
    "runs",
    "data",
    "analysis",
    "experiments",
    "diagnostics",
    "pretrained_models",
    "build",
    "dist",
    "ablationstudy",
    "eval",
    "tmp",
    "tmp_bench",
    "pocket_quality_vis",
    "batchsummary",
    "pics",
    "fig7",
    "gspt1_handoff",
    "$upload",
}
PRIVATE_NAMES = {"server_config.json", ".internal_api_key", ".env"}
PRIVATE_SUFFIXES = {".pt", ".pth", ".ckpt", ".db", ".log", ".pyc", ".lock",
                    ".key", ".pem", ".p12", ".pfx", ".sqlite", ".sqlite3"}


def checksum(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_files(root):
    """Enumerate only paths explicitly allowed by the publication manifest."""
    root = Path(root).resolve()
    manifest = json.loads((root / "publish-manifest.json").read_text())
    selected = {}
    for pattern in manifest["include"]:
        for path in root.glob(pattern):
            if not path.is_file():
                continue
            rel = path.relative_to(root)
            # Runtime roots must not hide source packages named data/analysis.
            if (
                rel.parts[0] in PRIVATE_DIRS
                or rel.parts[0].startswith(
                    ("workspace-", "molecules_score_", "jsdpt3010")
                )
                or any(
                    part
                    in {
                        ".git",
                        "__pycache__",
                        "node_modules",
                        ".pytest_cache",
                        ".ruff_cache",
                    }
                    for part in rel.parts
                )
            ):
                continue
            if (path.name in PRIVATE_NAMES or path.suffix in PRIVATE_SUFFIXES
                    or path.name.startswith(".env")
                    or any(part.startswith(".") for part in rel.parts[1:])):
                continue
            if any(
                fnmatch.fnmatch(rel.as_posix(), p) for p in manifest.get("exclude", [])
            ):
                continue
            if path.is_symlink() or any(
                parent.is_symlink() for parent in path.parents if parent != root
            ):
                raise ValueError("Source export refuses symlinks: " + str(rel))
            if root not in path.resolve().parents:
                raise ValueError("Source path escapes root: " + str(rel))
            selected[rel.as_posix()] = path
    return selected


def audit(root):
    root = Path(root).resolve()
    files = source_files(root)
    required = [
        "src/diffdynamic/__init__.py",
        "server/main.py",
        "ui/templates/index.html",
        "ui/templates/landing.html",
        "scripts/sample_diffusion.py",
        "weights-manifest.json",
    ]
    missing = [p for p in required if p not in files]
    if missing:
        raise ValueError("Publication is missing: " + ", ".join(missing))
    return {
        "root": str(root),
        "files": len(files),
        "bytes": sum(p.stat().st_size for p in files.values()),
        "paths": sorted(files),
        "git_mutations": False,
    }


def export(root, destination):
    """Create a new flat candidate and a hash inventory; never commit or push."""
    root, destination = Path(root).resolve(), Path(destination).absolute()
    if destination.exists():
        raise FileExistsError("Export destination already exists: " + str(destination))
    files = source_files(root)
    audit(root)
    destination.mkdir(parents=True)
    hashes = {}
    try:
        for relative, source in sorted(files.items()):
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(source), str(target))
            hashes[relative] = checksum(target)
        (destination / "SOURCE_HASHES.json").write_text(
            json.dumps(hashes, indent=2) + "\n"
        )
    except Exception:
        # Keep the failed candidate for inspection instead of deleting user data.
        (destination / "EXPORT_INCOMPLETE").write_text(
            "Export failed; do not publish.\n"
        )
        raise
    return {
        "destination": str(destination),
        "files": len(hashes),
        "git_mutations": False,
    }


def compare_to_remote(root, destination, ref="origin/master"):
    """Report a flat-tree comparison against a local remote-tracking snapshot."""
    root, destination = Path(root), Path(destination)
    result = subprocess.run(
        ["git", "ls-tree", "--full-tree", "-r", "-z", "--name-only", ref],
        cwd=str(root),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode:
        return {"reference": ref, "available": False}
    previous = set(result.stdout.rstrip("\0").split("\0")) if result.stdout else set()
    current = {
        p.relative_to(destination).as_posix()
        for p in destination.rglob("*")
        if p.is_file()
    }
    current.discard("SOURCE_HASHES.json")
    return {
        "reference": ref,
        "available": True,
        "added": sorted(current - previous),
        "removed": sorted(previous - current),
        "common": len(current & previous),
    }
