#!/usr/bin/env python3
"""Run source syntax and production tests against temporary runtime storage."""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.maintenance.publishing import audit, source_files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tests", nargs="*", help="Optional pytest paths or selectors")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    result = audit(root)
    for relative, path in source_files(root).items():
        if path.suffix == ".py":
            compile(path.read_bytes(), relative, "exec")
    print(
        "Source syntax: {} publication files checked".format(result["files"]),
        flush=True,
    )
    with tempfile.TemporaryDirectory(prefix="diffdynamic-checks-") as temporary:
        base = Path(temporary)
        for name in ("outputs", "uploads", "data"):
            (base / name).mkdir()
        env = os.environ.copy()
        env.update(
            DD_ROOT=str(root),
            DD_DB_PATH=str(base / "test.db"),
            DD_OUTPUT=str(base / "outputs"),
            DD_UPLOAD=str(base / "uploads"),
            DD_DATA=str(base / "data"),
            DD_AUTH_MODE="off",
            DD_CLEANUP_ON_START="0",
            PYTHONDONTWRITEBYTECODE="1",
        )
        tests = args.tests or sorted(
            p
            for p in result["paths"]
            if p.startswith("tests/test_") and p.endswith(".py")
        )
        command = [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "--disable-warnings",
            "-o",
            "cache_dir=" + str(base / "pytest-cache"),
        ] + tests
        return subprocess.call(command, cwd=str(root), env=env)


if __name__ == "__main__":
    raise SystemExit(main())
