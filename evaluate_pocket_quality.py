"""Compatibility entry point; implementation lives in diffdynamic.analysis.pocket.legacy."""
import _diffdynamic_bootstrap
from diffdynamic._compat import alias_module

if __name__ == "__main__":
    from diffdynamic.analysis.pocket.legacy import main
    raise SystemExit(main())
else:
    alias_module(__name__, "diffdynamic.analysis.pocket.legacy")
