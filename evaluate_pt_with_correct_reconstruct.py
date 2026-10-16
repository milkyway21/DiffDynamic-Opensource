"""Compatibility entry point; implementation lives in diffdynamic.evaluation.molecule.legacy."""
import _diffdynamic_bootstrap
from diffdynamic._compat import alias_module

if __name__ == "__main__":
    from diffdynamic.evaluation.molecule.legacy import main
    raise SystemExit(main())
else:
    alias_module(__name__, "diffdynamic.evaluation.molecule.legacy")
