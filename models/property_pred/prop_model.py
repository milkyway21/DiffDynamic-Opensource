"""Compatibility entry point; implementation lives in ``diffdynamic.models.property_pred.prop_model``."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import _diffdynamic_bootstrap  # noqa: F401
from diffdynamic._compat import alias_module

alias_module(__name__, "diffdynamic.models.property_pred.prop_model")
