"""Pages HTTP routes; business behavior is compatibility-preserving."""

import os

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from server.dependencies import dependencies as deps

router = APIRouter()

__all__ = ["landing", "index"]


@router.get("/", response_class=HTMLResponse)
def landing(request: Request):
    """Landing page (Three.js-driven interactive showcase)."""
    landing_path = os.path.join(deps._ui_root, "templates", "landing.html")
    with open(landing_path, encoding="utf-8") as f:
        return deps._inject_runtime(f.read(), request)


@router.get("/app", response_class=HTMLResponse)
def index(request: Request):
    """Functional workbench (generation / evaluation / molecules / history)."""
    index_path = os.path.join(deps._ui_root, "templates", "index.html")
    with open(index_path, encoding="utf-8") as f:
        return deps._inject_runtime(f.read(), request)
