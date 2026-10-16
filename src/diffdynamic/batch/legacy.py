"""Historical module interface over the extracted implementation."""

from diffdynamic._compat import install_facade

install_facade(
    __name__,
    [
        "diffdynamic.batch.state",
        "diffdynamic.batch.resources",
        "diffdynamic.batch.configuration",
        "diffdynamic.batch.results",
        "diffdynamic.batch.sampling",
        "diffdynamic.batch.evaluation",
        "diffdynamic.batch.benchmark",
        "diffdynamic.batch.reports",
        "diffdynamic.batch.summary",
        "diffdynamic.batch.arguments",
        "diffdynamic.batch.application",
    ],
)
