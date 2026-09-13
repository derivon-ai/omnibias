# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""PCG numerical residual, failure semantics and implicit derivatives."""

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest
from omnibias.curvature.operators import pcg_solve


def test_pcg_solution_jit_and_implicit_gradient():
    a = jnp.array([[4.0, 1.0], [1.0, 3.0]])
    b = jnp.array([1.0, 2.0])

    def solve(b):
        return pcg_solve(lambda x: a @ x, b, rtol=1e-12)

    out = jax.jit(solve)(b)
    assert out.converged and not out.breakdown
    np.testing.assert_allclose(out.solution, np.linalg.solve(a, b), rtol=1e-12)
    jac = jax.jacrev(lambda b: solve(b).solution)(b)
    np.testing.assert_allclose(jac, np.linalg.inv(a), atol=1e-12)
    grad = jax.grad(lambda scale: jnp.sum(pcg_solve(lambda x: scale * (a @ x), b).solution))(2.0)
    assert np.allclose(grad, -np.sum(np.linalg.solve(a, b)) / 4)


def test_pcg_failure_and_zero_rhs():
    bad = pcg_solve(lambda x: -x, jnp.ones(3))
    assert bad.breakdown and not bad.converged
    zero = pcg_solve(lambda x: x, jnp.zeros(3))
    assert zero.converged and zero.iterations == 0
    exhausted = pcg_solve(
        lambda x: jnp.arange(1.0, 11.0) * x, jnp.ones(10), max_iterations=1, rtol=1e-14
    )
    assert not exhausted.converged
    for invalid in (float("inf"), float("nan")):
        with pytest.raises(ValueError, match="finite"):
            pcg_solve(lambda x: x, jnp.ones(3), rtol=invalid)
