"""Compatibility entry point; implementation lives in diffdynamic.analysis.plots.plot_generic_violin_expanded."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _diffdynamic_bootstrap
from diffdynamic._compat import alias_module

if __name__ == "__main__":
    from diffdynamic.analysis.plots.plot_generic_violin_expanded import main
    raise SystemExit(main())
else:
    alias_module(__name__, "diffdynamic.analysis.plots.plot_generic_violin_expanded")
