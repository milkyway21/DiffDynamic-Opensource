"""Compatibility entry point; implementation lives in diffdynamic.batch.split."""
import _diffdynamic_bootstrap
from diffdynamic._compat import alias_module

if __name__ == "__main__":
    from diffdynamic.batch.split import main
    raise SystemExit(main())
else:
    alias_module(__name__, "diffdynamic.batch.split")
