# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Analytic oracles, jit/vmap, derivatives and numerical refusal states."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from omnibias.jax.affine_geometry import affine_project, linearized_equality_step

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_jit_analytic_projection_batches_vmap_row_transforms_and_empty_square(dtype):
    x = jnp.array([3, -1, 2], dtype=dtype)
    matrix = jnp.array([[1, 2, 0], [0, 1, 1]], dtype=dtype)
    rhs = jnp.array([2, -1], dtype=dtype)
    result = jax.jit(affine_project)(x, matrix, rhs)
    tolerance = 5e-6 if dtype == jnp.float32 else 3e-14
    # Exact 2x2 normal-equation solve gives multipliers=(-1,2).
    np.testing.assert_allclose(result.projected, [4, -1, 0], rtol=tolerance, atol=tolerance)
    np.testing.assert_allclose(result.correction, [1, 0, -2], rtol=tolerance, atol=tolerance)
    np.testing.assert_allclose(result.half_squared_distance, 2.5, rtol=tolerance, atol=tolerance)
    assert bool(result.valid) and 0 < result.rcond <= 1
    np.testing.assert_allclose(matrix @ result.projected, rhs, atol=tolerance)
    np.testing.assert_allclose(result.correction @ jnp.array([2, -1, 1], dtype=dtype), 0, atol=tolerance)
    np.testing.assert_allclose(affine_project(result.projected, matrix, rhs).projected,
                               result.projected, rtol=tolerance, atol=tolerance)
    queries = jnp.stack((x, 2*x, -x)).reshape(1, 3, 3)
    batched = jax.jit(affine_project)(queries, matrix, rhs)
    mapped = jax.jit(jax.vmap(lambda q: affine_project(q, matrix, rhs)))(queries[0])
    np.testing.assert_allclose(batched.projected[0], mapped.projected, rtol=tolerance, atol=tolerance)
    for transform in (jnp.array([[1e-8, 0], [0, -1e8]], dtype=dtype), jnp.array([[2, -1], [1, 3]], dtype=dtype)):
        changed = jax.jit(affine_project)(queries, transform @ matrix, transform @ rhs)
        assert bool(changed.valid.all())
        np.testing.assert_allclose(changed.projected, batched.projected, rtol=tolerance, atol=tolerance)
    empty = jax.jit(affine_project)(queries, jnp.empty((0, 3), dtype=dtype), jnp.empty((0,), dtype=dtype))
    np.testing.assert_array_equal(empty.projected, queries)
    np.testing.assert_array_equal(empty.correction, jnp.zeros_like(queries))
    square_matrix = jnp.array([[1, 2, 0], [0, 1, 1], [1, 0, 1]], dtype=dtype)
    square_rhs = jnp.array([2, -1, 4], dtype=dtype)
    square = jax.jit(affine_project)(queries, square_matrix, square_rhs)
    np.testing.assert_allclose(square.projected, jnp.broadcast_to(jnp.array([4., -1., 0.], dtype=dtype), queries.shape),
                               rtol=tolerance, atol=tolerance)
    np.testing.assert_array_equal(jax.jacfwd(lambda q: affine_project(q, square_matrix, square_rhs).projected)(x),
                                  jnp.zeros((3, 3), dtype=dtype))


def test_parameter_gradients_against_independent_formula_and_second_derivatives():
    x = jnp.array([.4, -.7, 1.2])
    matrix = jnp.array([[1.1, 2.3, .4], [.2, -.5, 1.3]])
    rhs = jnp.array([.1, -.2])
    weights = jnp.array([.7, -.3, .2])

    def loss(q, a, b):
        result = affine_project(q, a, b)
        return result.projected @ weights + .3 * result.half_squared_distance

    def oracle(q, a, b):
        # A direct 2x2 determinant solve is independent of the reduced QR path.
        g = a @ a.T
        r = a @ q - b
        det = g[0, 0]*g[1, 1] - g[0, 1]*g[1, 0]
        multipliers = jnp.array([g[1, 1]*r[0] - g[0, 1]*r[1],
                                 g[0, 0]*r[1] - g[1, 0]*r[0]]) / det
        correction = -a.T @ multipliers
        return (q + correction) @ weights + .15 * jnp.sum(correction**2)

    grads = jax.jit(jax.grad(loss, argnums=(0, 1, 2)))(x, matrix, rhs)
    expected = jax.grad(oracle, argnums=(0, 1, 2))(x, matrix, rhs)
    for actual, reference in zip(grads, expected, strict=True):
        np.testing.assert_allclose(actual, reference, rtol=3e-13, atol=3e-14)
    def packed(z):
        return loss(z[:3], z[3:9].reshape(2, 3), z[9:])
    def oracle_packed(z):
        return oracle(z[:3], z[3:9].reshape(2, 3), z[9:])
    values = jnp.concatenate((x, matrix.reshape(-1), rhs))
    hessian = jax.jit(jax.hessian(packed))(values)
    np.testing.assert_allclose(hessian, jax.hessian(oracle_packed)(values), rtol=2e-12, atol=2e-13)
    gradient = jax.grad(lambda q: affine_project(q, matrix, rhs).half_squared_distance)(x)
    np.testing.assert_allclose(gradient, -affine_project(x, matrix, rhs).correction, rtol=2e-14, atol=2e-14)
    projection_jacobian = jax.jacfwd(lambda q: affine_project(q, matrix, rhs).projected)(x)
    distance_hessian = jax.hessian(lambda q: affine_project(q, matrix, rhs).half_squared_distance)(x)
    np.testing.assert_allclose(projection_jacobian + distance_hessian, jnp.eye(3), atol=2e-14)
    np.testing.assert_allclose(projection_jacobian @ projection_jacobian, projection_jacobian, atol=2e-14)
    np.testing.assert_allclose(jax.grad(lambda a: affine_project(x, a, rhs).rcond)(matrix), jnp.zeros_like(matrix), atol=0)


@pytest.mark.parametrize("matrix,rhs", [
    ([[1, 2, 3], [2, 4, 6]], [1, 2]),
    ([[0, 0, 0], [1, 0, 0]], [1, 2]),
    ([[float("nan"), 1, 0], [0, 1, 0]], [1, 2]),
    ([[1, 0, 0], [0, 1, 0]], [float("inf"), 2]),
])
def test_invalid_constraints_refuse_under_jit_with_safe_gradients(matrix, rhs):
    args = tuple(jnp.array(v, dtype=jnp.float64) for v in ([1, 2, 3], matrix, rhs))
    result = jax.jit(affine_project)(*args)
    assert not bool(result.valid) and np.isnan(result.projected).all()
    assert np.isnan(result.correction).all() and np.isnan(result.half_squared_distance)
    gradients = jax.jit(jax.grad(lambda *p: jnp.nan_to_num(affine_project(*p).projected).sum(), argnums=(0, 1, 2)))(*args)
    assert all(np.isfinite(g).all() for g in gradients)


def test_rank_threshold_query_refusal_and_static_guards():
    x = jnp.array([[1., 2.], [jnp.nan, 2.]], dtype=jnp.float32)
    result = jax.jit(affine_project)(x, jnp.array([[1., 0.]]), jnp.array([0.]))
    np.testing.assert_array_equal(result.valid, [True, False])
    gradient = jax.grad(lambda q: jnp.nan_to_num(affine_project(q, jnp.array([[1., 0.]]), jnp.array([0.])).projected).sum())(x)
    np.testing.assert_array_equal(gradient, [[0, 1], [0, 0]])
    matrix = jnp.array([[1., 0.], [1., 1e-4]], dtype=jnp.float32)
    assert bool(jax.jit(affine_project)(jnp.zeros(2, dtype=jnp.float32), matrix, jnp.zeros(2)).valid)
    assert not bool(affine_project(jnp.zeros(2, dtype=jnp.float32), matrix, jnp.zeros(2), rank_rcond=1e-3).valid)
    with pytest.raises(ValueError, match="shape"):
        affine_project(jnp.zeros(2), jnp.ones((3, 2)), jnp.zeros(3))
    with pytest.raises(ValueError, match="rank_rcond"):
        affine_project(jnp.zeros(2), jnp.ones((1, 2)), jnp.zeros(1), rank_rcond=1)


def test_linearized_circle_is_not_nearest_point_and_derivative_is_iteration_derivative():
    x = jnp.array([2., 0.])
    def step(q):
        return linearized_equality_step(q, (jnp.sum(q**2) - 1)[None], (2*q)[None])
    result = jax.jit(step)(x)
    assert bool(result.valid)
    np.testing.assert_allclose(result.value, [1.25, 0], atol=2e-15)
    np.testing.assert_allclose(result.local_half_squared_distance, .28125, atol=2e-15)
    np.testing.assert_allclose(jax.jit(jax.jacfwd(lambda q: step(q).value))(x), [[.375, 0], [0, .625]], atol=2e-14)
    mapped = jax.jit(jax.vmap(step))(jnp.array([[2., 0.], [0., 2.]]))
    np.testing.assert_allclose(mapped.value, [[1.25, 0], [0, 1.25]], atol=2e-15)
    batched = linearized_equality_step(jnp.broadcast_to(x, (2, 2)), jnp.array([[3.], [1.]]), jnp.array([[4., 0.]]))
    np.testing.assert_allclose(batched.value, [[1.25, 0], [1.75, 0]], atol=2e-15)
