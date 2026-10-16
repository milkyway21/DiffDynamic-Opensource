"""Live runtime dependency bindings for application composition.

Each composition root binds its live namespace once. Functional modules never
import that root, while legacy ``server.api`` / ``server.scheduler`` patches
and integrations continue to replace the dependencies those modules use.
"""

from typing import Any, Mapping, Optional


class DependencyProvider:
    """Resolve application dependencies when a request uses them."""

    def __init__(self) -> None:
        self._namespace: Optional[Mapping[str, Any]] = None

    def bind(self, namespace: Mapping[str, Any]) -> None:
        self._namespace = namespace

    def __getattr__(self, name: str) -> Any:
        if self._namespace is None:
            raise RuntimeError("Dependencies must be bound during app assembly")
        try:
            return self._namespace[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


dependencies = DependencyProvider()
