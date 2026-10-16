"""Compatibility entry point; implementation lives in ``diffdynamic.sampling.legacy``."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _diffdynamic_bootstrap  # noqa: F401
from diffdynamic._compat import alias_module

if __name__ == "__main__":
    from diffdynamic.sampling.legacy import main
    raise SystemExit(main())
else:
    alias_module(__name__, "diffdynamic.sampling.legacy")
