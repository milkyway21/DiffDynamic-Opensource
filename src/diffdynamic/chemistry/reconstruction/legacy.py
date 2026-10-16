"""Historical module interface over the extracted implementation."""

from diffdynamic._compat import install_facade

install_facade(
    __name__,
    [
        "diffdynamic.chemistry.reconstruction.state",
        "diffdynamic.chemistry.reconstruction.atoms",
        "diffdynamic.chemistry.reconstruction.aromaticity",
        "diffdynamic.chemistry.reconstruction.scaffold",
        "diffdynamic.chemistry.reconstruction.pipeline",
        "diffdynamic.chemistry.reconstruction.repair",
        "diffdynamic.chemistry.reconstruction.writers",
    ],
)
