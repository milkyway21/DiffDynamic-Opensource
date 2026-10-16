"""Compatibility entry point; implementation lives in diffdynamic.analysis.plots.plot_ridgeline_iceblue_f56e1a."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _diffdynamic_bootstrap
from diffdynamic._compat import alias_module

if __name__ == "__main__":
    from diffdynamic.analysis.plots.plot_ridgeline_iceblue_f56e1a import main
    raise SystemExit(main())
else:
    alias_module(__name__, "diffdynamic.analysis.plots.plot_ridgeline_iceblue_f56e1a")
