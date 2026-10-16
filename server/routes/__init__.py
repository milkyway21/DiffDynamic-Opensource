"""Assemble functional routers and preserve the public ``server.api`` facade."""

from importlib import import_module
from typing import Any, Dict

from fastapi import FastAPI

from server.dependencies import dependencies

ROUTE_MODULES = (
    "generation",
    "medchem",
    "scaffold",
    "evaluation",
    "jobs",
    "runs",
    "molecules",
    "files",
    "admin",
    "pages",
)


def register_routes(app: FastAPI, namespace: Dict[str, Any]) -> None:
    """Bind dependencies, register handlers, and retain their original exports."""
    dependencies.bind(namespace)
    for name in ROUTE_MODULES:
        module = import_module("server.routes." + name)
        for handler in module.__all__:
            namespace[handler] = getattr(module, handler)
        app.include_router(module.router)
