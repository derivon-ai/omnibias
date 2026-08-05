# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Differentiable QP/LP layer (JAX): KKT implicit-function gradients vs finite-diff.

The forward solve is an interior-point method, so finite differences are dominated
by solver noise for a tiny step. We use a balanced ``eps=1e-3`` (the cube-root rule
for ~1e-9 objective noise) and tolerances that reflect the barrier method's dual
accuracy; the *exact*-agreement check lives in the torch<->jax gradient parity test.
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402
from omnibias.convex.jax import lp_layer, qp_layer  # noqa: E402
from omnibias.convex.jax.solver import solve_lp, solve_qp  # noqa: E402

_ATOL = 3e-4
_RTOL = 3e-3
_EPS = 1e-3


def _arr(x: object) -> jnp.ndarray:
    return jnp.asarray(np.asarray(x, dtype=np.float64))


def _random_qp(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Unit-box QP whose unconstrained optimum -Q^{-1}c = x_target lies outside the
    # box, so the box constraints are firmly active (strict complementarity) and the
    # gradients w.r.t. A and b are non-trivial.
    rng = np.random.default_rng(seed)
    n = 3
    M = rng.standard_normal((n, n))
    Q = M @ M.T + 0.5 * np.eye(n)
    x_target = np.array([3.0, -3.0, 0.2])  # x0, x1 pushed outside [-1, 1]
    c = -Q @ x_target
    A = np.vstack([np.eye(n), -np.eye(n)])
    b = np.ones(2 * n)
    return Q, c, A, b


def _loss_eager(Q: np.ndarray, c: np.ndarray, A: np.ndarray, b: np.ndarray, w: np.ndarray) -> float:
    sol = solve_qp(_arr(Q), _arr(c), _arr(A), _arr(b))
    return float(np.dot(w, np.asarray(sol.x)))


def _dir_fd(base: np.ndarray, direction: np.ndarray, recompute, eps: float = _EPS) -> float:
    return (recompute(base + eps * direction) - recompute(base - eps * direction)) / (2.0 * eps)


def test_qp_layer_matches_eager_solver() -> None:
    Q, c, A, b = _random_qp(0)
    x_layer = np.asarray(qp_layer(_arr(Q), _arr(c), _arr(A), _arr(b)))
    x_eager = np.asarray(solve_qp(_arr(Q), _arr(c), _arr(A), _arr(b)).x)
    np.testing.assert_allclose(x_layer, x_eager, atol=1e-9)


def test_qp_layer_grad_c_vs_finite_diff() -> None:
    Q, c, A, b = _random_qp(1)
    rng = np.random.default_rng(10)
    w = rng.standard_normal(c.shape[0])

    def loss_c(cv: jnp.ndarray) -> jnp.ndarray:
        return jnp.dot(_arr(w), qp_layer(_arr(Q), cv, _arr(A), _arr(b)))

    grad = np.asarray(jax.grad(loss_c)(_arr(c)))
    for _ in range(3):
        d = rng.standard_normal(c.shape[0])
        ad = float(np.dot(grad, d))
        fd = _dir_fd(c, d, lambda cc: _loss_eager(Q, cc, A, b, w))
        np.testing.assert_allclose(ad, fd, atol=_ATOL, rtol=_RTOL)


def test_qp_layer_grad_b_vs_finite_diff() -> None:
    Q, c, A, b = _random_qp(2)
    rng = np.random.default_rng(11)
    w = rng.standard_normal(c.shape[0])

    def loss_b(bv: jnp.ndarray) -> jnp.ndarray:
        return jnp.dot(_arr(w), qp_layer(_arr(Q), _arr(c), _arr(A), bv))

    grad = np.asarray(jax.grad(loss_b)(_arr(b)))
    assert np.max(np.abs(grad)) > 1e-3  # active constraints => non-trivial db gradient
    for _ in range(3):
        d = rng.standard_normal(b.shape[0])
        ad = float(np.dot(grad, d))
        fd = _dir_fd(b, d, lambda bb: _loss_eager(Q, c, A, bb, w))
        np.testing.assert_allclose(ad, fd, atol=_ATOL, rtol=_RTOL)


def test_qp_layer_grad_A_vs_finite_diff() -> None:
    Q, c, A, b = _random_qp(3)
    rng = np.random.default_rng(12)
    w = rng.standard_normal(c.shape[0])

    def loss_A(Av: jnp.ndarray) -> jnp.ndarray:
        return jnp.dot(_arr(w), qp_layer(_arr(Q), _arr(c), Av, _arr(b)))

    grad = np.asarray(jax.grad(loss_A)(_arr(A)))
    assert np.max(np.abs(grad)) > 1e-3
    for _ in range(3):
        d = rng.standard_normal(A.shape)
        ad = float(np.sum(grad * d))
        fd = _dir_fd(A, d, lambda AA: _loss_eager(Q, c, AA, b, w))
        np.testing.assert_allclose(ad, fd, atol=_ATOL, rtol=_RTOL)


def test_qp_layer_grad_Q_symmetric_direction() -> None:
    Q, c, A, b = _random_qp(4)
    rng = np.random.default_rng(13)
    w = rng.standard_normal(c.shape[0])

    def loss_Q(Qv: jnp.ndarray) -> jnp.ndarray:
        return jnp.dot(_arr(w), qp_layer(Qv, _arr(c), _arr(A), _arr(b)))

    grad = np.asarray(jax.grad(loss_Q)(_arr(Q)))
    for _ in range(3):
        e = rng.standard_normal(Q.shape)
        d = e + e.T  # symmetric direction: matches the layer's symmetrized grad
        ad = float(np.sum(grad * d))
        fd = _dir_fd(Q, d, lambda QQ: _loss_eager(QQ, c, A, b, w))
        np.testing.assert_allclose(ad, fd, atol=_ATOL, rtol=_RTOL)


def test_lp_layer_grad_b_vs_finite_diff_and_c_is_flat() -> None:
    # LP optimum is a vertex: it moves with b but is locally constant in c.
    rng = np.random.default_rng(5)
    n = 2
    c = rng.standard_normal(n)
    A = np.vstack([rng.standard_normal((2, n)), np.eye(n), -np.eye(n)])
    b = np.concatenate([rng.uniform(0.5, 1.5, size=2), 3.0 * np.ones(2 * n)])
    w = rng.standard_normal(n)

    def loss_b(bv: jnp.ndarray) -> jnp.ndarray:
        return jnp.dot(_arr(w), lp_layer(_arr(c), _arr(A), bv))

    def eager(bb: np.ndarray) -> float:
        return float(np.dot(w, np.asarray(solve_lp(_arr(c), _arr(A), _arr(bb)).x)))

    grad_b = np.asarray(jax.grad(loss_b)(_arr(b)))
    for _ in range(3):
        d = rng.standard_normal(b.shape[0])
        np.testing.assert_allclose(
            float(np.dot(grad_b, d)), _dir_fd(b, d, eager), atol=_ATOL, rtol=_RTOL
        )

    def loss_c(cv: jnp.ndarray) -> jnp.ndarray:
        return jnp.dot(_arr(w), lp_layer(cv, _arr(A), _arr(b)))

    grad_c = np.asarray(jax.grad(loss_c)(_arr(c)))
    assert np.max(np.abs(grad_c)) < 1e-3  # vertex is flat in c
