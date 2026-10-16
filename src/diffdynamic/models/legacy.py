"""Legacy public model namespace with compatibility-only patch forwarding."""

from diffdynamic._compat import install_facade

install_facade(__name__, [
    'diffdynamic.models.probability',
    'diffdynamic.models.schedules',
    'diffdynamic.models.network',
    'diffdynamic.models.diffusion',
    'diffdynamic.models.dynamics',
    'diffdynamic.models.samplers',
    'diffdynamic.models.score_model',
    'diffdynamic.models.constants',
])
