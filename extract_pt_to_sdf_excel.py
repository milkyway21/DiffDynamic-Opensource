"""Compatibility entry point; implementation lives in diffdynamic.cli.extract."""
import _diffdynamic_bootstrap
from diffdynamic._compat import alias_module

if __name__ == "__main__":
    from diffdynamic.cli.extract import main
    raise SystemExit(main())
else:
    alias_module(__name__, "diffdynamic.cli.extract")
