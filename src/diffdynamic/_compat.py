"""Compatibility facades for historical modules.

Algorithms live in normal Python modules with explicit imports. This adapter
only preserves callers that read or monkeypatch the former all-in-one module.
Assignments propagate to bindings of the same object, and never to unrelated
names that happen to share a spelling.
"""

import importlib
import sys
from types import ModuleType


class ModuleFacade(ModuleType):
    def __getattribute__(self, name):
        if name not in {"_compat_owners", "_compat_modules", "__dict__",
                        "__class__", "__name__", "__spec__", "__loader__",
                        "__package__", "__path__", "__file__"}:
            owners = ModuleType.__getattribute__(self, "__dict__").get(
                "_compat_owners", {})
            if name in owners:
                return getattr(owners[name], name)
        return ModuleType.__getattribute__(self, name)

    def __setattr__(self, name, value):
        state = ModuleType.__getattribute__(self, "__dict__")
        owners = state.get("_compat_owners", {})
        if name in owners:
            owner = owners[name]
            old = getattr(owner, name)
            setattr(owner, name, value)
            for module in state.get("_compat_modules", ()):
                if module is not owner and vars(module).get(name) is old:
                    setattr(module, name, value)
        ModuleType.__setattr__(self, name, value)

    def __dir__(self):
        state = ModuleType.__getattribute__(self, "__dict__")
        return sorted(set(state) | set(state.get("_compat_owners", {})))


def install_facade(module_name, modules, owners=None):
    """Expose extracted modules through an existing module, with write forwarding.

    ``owners`` can explicitly select the defining module for mutable state or
    ambiguous names. Imported functions otherwise prefer their defining module.
    """
    modules = tuple(importlib.import_module(m) if isinstance(m, str) else m
                    for m in modules)
    facade = sys.modules[module_name]
    export_owners = {}
    for module in modules:
        for name, value in vars(module).items():
            if name.startswith("__"):
                continue
            if name not in export_owners or getattr(value, "__module__", None) == module.__name__:
                export_owners[name] = module
    if owners:
        for name, owner in owners.items():
            export_owners[name] = (importlib.import_module(owner)
                                   if isinstance(owner, str) else owner)
    for name, owner in export_owners.items():
        facade.__dict__[name] = getattr(owner, name)
    facade.__dict__["_compat_owners"] = export_owners
    facade.__dict__["_compat_modules"] = modules
    facade.__class__ = ModuleFacade
    return facade


def alias_module(module_name, target):
    """Make an old import path refer to exactly the canonical module object."""
    module = importlib.import_module(target)
    sys.modules[module_name] = module
    return module
