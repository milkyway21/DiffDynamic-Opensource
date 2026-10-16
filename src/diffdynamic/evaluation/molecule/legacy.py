"""Historical module interface over the extracted implementation."""

from diffdynamic._compat import install_facade

install_facade(
    __name__,
    [
        "diffdynamic.evaluation.molecule.state",
        "diffdynamic.evaluation.molecule.optimization",
        "diffdynamic.evaluation.molecule.options",
        "diffdynamic.evaluation.molecule.loading",
        "diffdynamic.evaluation.molecule.reconstruction",
        "diffdynamic.evaluation.molecule.scoring",
        "diffdynamic.evaluation.molecule.workers",
        "diffdynamic.evaluation.molecule.writers",
        "diffdynamic.evaluation.molecule.pipeline",
        "diffdynamic.evaluation.molecule.preparation",
        "diffdynamic.evaluation.molecule.execution",
        "diffdynamic.evaluation.molecule.finalization",
        "diffdynamic.evaluation.molecule.history",
        "diffdynamic.evaluation.molecule.application",
    ],
)
