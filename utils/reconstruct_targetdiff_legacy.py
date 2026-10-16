"""Compatibility import for the frozen reconstruction implementation."""
import _diffdynamic_bootstrap
from diffdynamic._compat import alias_module
alias_module(__name__, "diffdynamic.chemistry.reconstruct_targetdiff_legacy")
