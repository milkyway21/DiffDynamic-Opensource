"""Compatibility facade over focused chemistry modules."""

from diffdynamic._compat import alias_module

alias_module(__name__, "diffdynamic.chemistry.reconstruction.legacy")
