"""Check frontend module contracts without a running API or production data."""

import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_frontend_scripts_assets_and_global_contracts():
    """Catch broken module links, inline handlers, assets, and locale paths."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is needed for the frontend contract check")
    result = subprocess.run(
        [node, "--experimental-vm-modules", "ui/tests/check-contracts.cjs"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
