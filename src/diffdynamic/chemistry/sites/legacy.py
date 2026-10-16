"""Historical module interface over the extracted implementation."""

from diffdynamic._compat import install_facade

install_facade(
    __name__,
    [
        "diffdynamic.chemistry.sites.state",
        "diffdynamic.chemistry.sites.configuration",
        "diffdynamic.chemistry.sites.extraction",
        "diffdynamic.chemistry.sites.allocation",
        "diffdynamic.chemistry.sites.gaussian",
        "diffdynamic.chemistry.sites.fragments",
        "diffdynamic.chemistry.sites.fragment_geometry",
        "diffdynamic.chemistry.sites.pocket_geometry",
        "diffdynamic.chemistry.sites.preservation",
        "diffdynamic.chemistry.sites.placement",
    ],
)
