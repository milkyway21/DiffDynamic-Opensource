"""Locate public application resources in a checkout or installed distribution."""

import os
from pathlib import Path


def project_root():
    """Resolve DD_ROOT, then the nearest complete source or installed resource tree.

    Source checkouts and wheels deliberately share the same public layout:
    configs/, server/, ui/, and the historical CLI entry points. Discover that
    layout instead of assuming how deeply this module sits under site-packages.
    """
    configured = os.environ.get("DD_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    for candidate in Path(__file__).resolve().parents:
        if (
            (candidate / "configs" / "sampling.yml").is_file()
            and (candidate / "server" / "main.py").is_file()
            and (candidate / "scripts" / "sample_diffusion.py").is_file()
        ):
            return candidate
    raise RuntimeError(
        "DiffDynamic application resources are missing; set DD_ROOT to a complete "
        "checkout or reinstall the distribution."
    )


def project_path(*parts):
    return project_root().joinpath(*parts)
