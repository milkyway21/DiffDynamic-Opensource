"""Historical module interface over the extracted implementation."""

from diffdynamic._compat import install_facade

install_facade(
    __name__,
    [
        "diffdynamic.analysis.pocket.state",
        "diffdynamic.analysis.pocket.inputs",
        "diffdynamic.analysis.pocket.aggregation",
        "diffdynamic.analysis.pocket.affinity",
        "diffdynamic.analysis.pocket.clustering",
        "diffdynamic.analysis.pocket.chemistry",
        "diffdynamic.analysis.pocket.pocket_metrics",
        "diffdynamic.analysis.pocket.volume",
        "diffdynamic.analysis.pocket.plotting_common",
        "diffdynamic.analysis.pocket.plots_geometry",
        "diffdynamic.analysis.pocket.plots_chemistry",
        "diffdynamic.analysis.pocket.plots_summary",
        "diffdynamic.analysis.pocket.pipeline",
        "diffdynamic.analysis.pocket.batch",
        "diffdynamic.analysis.pocket.application",
    ],
)
