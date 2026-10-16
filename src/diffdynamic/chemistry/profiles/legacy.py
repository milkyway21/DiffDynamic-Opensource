"""Historical module interface over the extracted implementation."""

from diffdynamic._compat import install_facade

install_facade(
    __name__,
    [
        "diffdynamic.chemistry.profiles.state",
        "diffdynamic.chemistry.profiles.sdf",
        "diffdynamic.chemistry.profiles.amine",
        "diffdynamic.chemistry.profiles.fragments",
        "diffdynamic.chemistry.profiles.building",
        "diffdynamic.chemistry.profiles.loading",
    ],
)
