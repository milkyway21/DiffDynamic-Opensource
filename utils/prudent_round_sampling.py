"""Compatibility entry point; implementation lives in ``diffdynamic.sampling.prudent.round_sampling``."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _diffdynamic_bootstrap  # noqa: F401
from diffdynamic._compat import alias_module

alias_module(__name__, "diffdynamic.sampling.prudent.round_sampling")
