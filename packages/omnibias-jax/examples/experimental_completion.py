# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Run a small private candidate; no public performance promotion is implied.

JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu python \
    packages/omnibias-jax/examples/experimental_completion.py
"""
import jax
import jax.numpy as jnp
import numpy as np

from omnibias.core.operator_spec import OperatorSpec
from omnibias.jax._experimental_stage.bounds import Bound
from omnibias.jax._experimental_stage.bundles import Model
from omnibias.jax._experimental_stage.planner import Domain, Limits, Reuse, compile_operator_bundle
from omnibias.jax._experimental_stage.streaming import WeightedMeanObjective
from omnibias.jax._experimental_stage.feature_models import FeatureModel
from omnibias.jax._experimental_stage.variable_projection import LinearResidual


def main():
    if not jax.config.x64_enabled:
        raise ValueError('Set JAX_ENABLE_X64=1 before starting Python.')
    p = (jnp.array([[.3, -.2], [.1, .4]]), jnp.array([.1, -.07]),
         jnp.array([.4, -.3]), jnp.asarray(.01))
    x = jnp.array([[.2, -.1], [-.3, .05], [.1, .3]])
    domain = Domain(tuple(Bound(np.asarray(v)-1e-5, np.asarray(v)+1e-5, 0.) for v in p),
                    Bound(np.full((1, 2), -.5), np.full((1, 2), .5), 0.))
    request = (OperatorSpec.laplacian(0), OperatorSpec.laplacian(1))
    loss = WeightedMeanObjective(lambda params, row, outputs: sum(v*v for v in outputs),
                                 lambda params, row: jnp.asarray(1., dtype=row.dtype))
    program, report = compile_operator_bundle(Model('ridge', 2, 2), request, p, x,
        domain=domain, streaming_objective=loss, chunk_size=2, reuse=Reuse(1),
        limits=Limits(candidates=1, shortlist=1, cells=1), measured_samples=3)
    if program is None:
        print('Operator candidate inconclusive:', report)
    else:
        value, gradient = jax.block_until_ready(program.objective_value_and_grad(p, x))
        print('Conditionally qualified objective:', float(value), 'gradient shapes:',
              [v.shape for v in gradient])
    feature = FeatureModel('ridge', 2, 2)
    residual = LinearResidual(feature, (OperatorSpec.laplacian(0),), np.asarray(x),
                               np.array([.1, -.2, .05]), regularization=.01)
    hidden = p[:2]
    certificate = residual.qualify(hidden, chunk_size=2)
    print('Variable projection certificate:', certificate['admissible'])
    if certificate['admissible']:
        # First hidden gradients only; differentiating the eliminated readout is
        # deliberately unsupported by this research contract.
        print(residual.value_and_grad(hidden, chunk_size=2))


if __name__ == '__main__':
    main()
