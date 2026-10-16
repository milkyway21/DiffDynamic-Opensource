"""Compatibility entry point; implementation lives in diffdynamic.sampling.cascade."""
import _diffdynamic_bootstrap
from diffdynamic._compat import alias_module

if __name__ == "__main__":
    from diffdynamic.sampling.cascade import main
    raise SystemExit(main())
else:
    alias_module(__name__, "diffdynamic.sampling.cascade")
