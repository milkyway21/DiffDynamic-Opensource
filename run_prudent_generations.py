"""Compatibility entry point; implementation lives in diffdynamic.sampling.generations."""
import _diffdynamic_bootstrap
from diffdynamic._compat import alias_module

if __name__ == "__main__":
    from diffdynamic.sampling.generations import main
    raise SystemExit(main())
else:
    alias_module(__name__, "diffdynamic.sampling.generations")
