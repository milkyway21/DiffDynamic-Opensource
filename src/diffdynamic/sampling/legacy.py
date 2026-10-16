"""Historical sampler namespace; algorithms live in focused modules."""

from rdkit import RDLogger

from diffdynamic._compat import install_facade

RDLogger.DisableLog('rdApp.warning')

install_facade(__name__, ('diffdynamic.sampling.scaffold.placement', 'diffdynamic.sampling.configuration', 'diffdynamic.sampling.naming', 'diffdynamic.data.pocket', 'diffdynamic.sampling.common', 'diffdynamic.sampling.dynamic_unified', 'diffdynamic.sampling.refinement', 'diffdynamic.sampling.candidates', 'diffdynamic.sampling.prudent.state', 'diffdynamic.sampling.prudent.reconstruction', 'diffdynamic.sampling.prudent.scoring', 'diffdynamic.sampling.prudent.refinement', 'diffdynamic.sampling.prudent.evolution', 'diffdynamic.sampling.prudent.legacy', 'diffdynamic.sampling.trajectory', 'diffdynamic.sampling.dynamic', 'diffdynamic.sampling.baseline', 'diffdynamic.sampling.optimization', 'diffdynamic.sampling.scaffold.inputs', 'diffdynamic.sampling.scaffold.fitness', 'diffdynamic.sampling.scaffold.grow', 'diffdynamic.sampling.scaffold.locked', 'diffdynamic.sampling.scaffold.prudent', 'diffdynamic.sampling.scaffold.evolve', 'diffdynamic.sampling.arguments', 'diffdynamic.sampling.pipelines.dynamic_then_optimization', 'diffdynamic.sampling.pipelines.optimization', 'diffdynamic.sampling.pipelines.dynamic_then_scaffold', 'diffdynamic.sampling.pipelines.scaffold_optimization', 'diffdynamic.sampling.pipelines.scaffold_grow', 'diffdynamic.sampling.pipelines.scaffold_dynamic_locked', 'diffdynamic.sampling.pipelines.scaffold_prudent', 'diffdynamic.sampling.pipelines.dynamic', 'diffdynamic.sampling.pipelines.baseline', 'diffdynamic.sampling.application'))
