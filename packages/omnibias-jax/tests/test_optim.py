# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Second-order PINN optimisation (JAX): Gauss-Newton / LM + self-adaptive weights.

Mirrors the torch optimiser tests: the LM core is exact on linear least squares, the
primal/dual push-through identity holds, LM recovers a nonlinear least-squares solution,
and -- the headline -- Gauss-Newton drives a Poisson PINN loss far below what Adam reaches
(the energy-natural-gradient behaviour enabled by the exact omnibias residual map).
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from jax import Array  # noqa: E402
from omnibias.jax.architectures import make_jet_mlp  # noqa: E402
from omnibias.jax.optim import (  # noqa: E402
    gauss_newton_direction,
    gauss_newton_fisher,
    gauss_newton_minimize,
    grad_norm_weights,
    make_residual_fn,
    natural_gradient_direction,
    natural_gradient_step,
)

# --- Gauss-Newton direction core -----------------------------------------


def test_gn_direction_solves_linear_least_squares_in_one_step() -> None:
    rng = np.random.RandomState(0)
    a = jnp.asarray(rng.randn(20, 5))
    b = jnp.asarray(rng.randn(20))
    p0 = jnp.zeros(5)
    delta = gauss_newton_direction(a, a @ p0 - b, damping=1e-10)
    p_star, _, _, _ = jnp.linalg.lstsq(a, b, rcond=None)
    assert jnp.allclose(p0 + delta, p_star, atol=1e-6)


def test_gn_direction_primal_dual_pushthrough_identity() -> None:
    rng = np.random.RandomState(1)
    jac = jnp.asarray(rng.randn(6, 12))  # P=12 > N=6 -> dual form
    res = jnp.asarray(rng.randn(6))
    mu = 1e-1
    d_impl = gauss_newton_direction(jac, res, damping=mu)
    p = jac.shape[1]
    d_primal = jnp.linalg.solve(jac.T @ jac + mu * jnp.eye(p), -(jac.T @ res))
    assert jnp.allclose(d_impl, d_primal, atol=1e-9)


def test_gn_direction_shape_validation() -> None:
    with pytest.raises(ValueError):
        gauss_newton_direction(jnp.zeros(3), jnp.zeros(3), 1e-3)
    with pytest.raises(ValueError):
        gauss_newton_direction(jnp.zeros((4, 2)), jnp.zeros(3), 1e-3)


# --- LM optimiser on nonlinear least squares ------------------------------


def test_gauss_newton_recovers_nonlinear_least_squares() -> None:
    t = jnp.linspace(0.0, 1.0, 24)
    true = jnp.array([2.0, -0.7])
    y = true[0] * jnp.exp(true[1] * t)

    def residual_fn(p: Array) -> Array:
        return p[0] * jnp.exp(p[1] * t) - y

    p0 = jnp.array([1.0, 0.0])
    state, history = gauss_newton_minimize(residual_fn, p0, steps=30, damping=1e-3)
    assert history[-1] < 1e-16
    assert jnp.allclose(state.params, true, atol=1e-6)
    assert all(history[i + 1] <= history[i] + 1e-18 for i in range(len(history) - 1))


# --- PINN: GN beats Adam by orders of magnitude ---------------------------


def _poisson_residual_fn() -> tuple[Array, Callable[[Array], Array]]:
    net = make_jet_mlp(1, 16, 1, depth=2, seed=0)
    x_int = jnp.linspace(0.0, 1.0, 40)[1:-1][:, None]
    x_bc = jnp.array([[0.0], [1.0]])
    f_int = (-math.pi**2) * jnp.sin(math.pi * x_int).reshape(-1)

    def build(n: object) -> Array:
        _v, _g, h = n.value_grad_hessian(x_int)  # type: ignore[attr-defined]
        return jnp.concatenate([h[:, 0, 0, 0] - f_int, n.value(x_bc).reshape(-1)])  # type: ignore[attr-defined]

    return make_residual_fn(build, net)


def _adam_minimize(residual_fn: Callable[[Array], Array], params: Array, steps: int, lr: float = 1e-3) -> float:
    def loss(p: Array) -> Array:
        return 0.5 * jnp.mean(residual_fn(p) ** 2)

    grad = jax.jit(jax.grad(loss))
    m = jnp.zeros_like(params)
    v = jnp.zeros_like(params)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(1, steps + 1):
        g = grad(params)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g**2
        mhat = m / (1 - b1**t)
        vhat = v / (1 - b2**t)
        params = params - lr * mhat / (jnp.sqrt(vhat) + eps)
    return float(loss(params))


def test_gauss_newton_beats_adam_on_poisson_pinn() -> None:
    flat0, residual_fn = _poisson_residual_fn()
    _, history = gauss_newton_minimize(residual_fn, flat0, steps=40, damping=1e-2)
    gn_loss = history[-1]
    adam_loss = _adam_minimize(residual_fn, flat0, steps=800, lr=1e-3)
    assert gn_loss < 5e-4
    assert adam_loss > 2e-3
    assert gn_loss < adam_loss / 5.0


def test_make_residual_fn_roundtrip() -> None:
    flat0, residual_fn = _poisson_residual_fn()
    r = residual_fn(flat0)
    assert r.shape[0] == 40  # 38 interior + 2 boundary
    assert jnp.all(jnp.isfinite(r))


# --- Self-adaptive loss weights -------------------------------------------


def test_grad_norm_weights_equalises_weighted_norms() -> None:
    net = make_jet_mlp(1, 16, 1, depth=2, seed=0)
    x_int = jnp.linspace(0.0, 1.0, 40)[1:-1][:, None]
    x_bc = jnp.array([[0.0], [1.0]])
    f_int = (-math.pi**2) * jnp.sin(math.pi * x_int).reshape(-1)

    def loss_pde(n: object) -> Array:
        _v, _g, h = n.value_grad_hessian(x_int)  # type: ignore[attr-defined]
        return jnp.mean((h[:, 0, 0, 0] - f_int) ** 2)

    def loss_bc(n: object) -> Array:
        return 1000.0 * jnp.mean(n.value(x_bc).reshape(-1) ** 2)  # type: ignore[attr-defined]

    from jax.flatten_util import ravel_pytree

    def gnorm(fn: Callable[[object], Array]) -> float:
        flat, _ = ravel_pytree(jax.grad(fn)(net))
        return float(jnp.linalg.norm(flat))

    weights = grad_norm_weights((loss_pde, loss_bc), net, jnp.ones(2), alpha=0.0, ref_index=0)
    weighted = jnp.array([weights[0] * gnorm(loss_pde), weights[1] * gnorm(loss_bc)])
    assert jnp.allclose(weighted, weighted[0], rtol=1e-6)
    assert float(weights[0]) == pytest.approx(1.0, abs=1e-9)


def test_grad_norm_weights_validation() -> None:
    net = make_jet_mlp(1, 4, 1, depth=1, seed=0)

    def lf(n: object) -> Array:
        return jnp.mean(n.value(jnp.zeros((1, 1))) ** 2)  # type: ignore[attr-defined]

    with pytest.raises(ValueError):
        grad_norm_weights((lf,), net, jnp.ones(1), alpha=2.0)
    with pytest.raises(ValueError):
        grad_norm_weights((lf,), net, jnp.ones(1), ref_index=3)


# --- Natural-gradient / Riemannian direction --------------------


def _spd(n: int, seed: int) -> Array:
    rng = np.random.RandomState(seed)
    m = rng.randn(n, n)
    return jnp.asarray(m @ m.T + n * np.eye(n))


def test_natural_gradient_direction_dense_solves_metric_system() -> None:
    """``natural_gradient_direction`` solves ``(M + mu I) delta = g`` exactly on a dense metric."""
    m = _spd(6, 21)
    g = jnp.asarray(np.random.RandomState(22).randn(6))
    for mu in (0.0, 1e-3, 1.0):
        delta = natural_gradient_direction(m, g, damping=mu)
        ref = jnp.linalg.solve(m + mu * jnp.eye(6), g)
        assert jnp.allclose(delta, ref, atol=1e-10)


def test_natural_gradient_direction_validation() -> None:
    with pytest.raises(ValueError):
        natural_gradient_direction(jnp.zeros((3, 4)), jnp.zeros(3))
    with pytest.raises(ValueError):
        natural_gradient_direction(jnp.eye(3), jnp.zeros(4))
    with pytest.raises(ValueError):
        natural_gradient_direction(jnp.eye(3), jnp.zeros(3), damping=-1.0)


def test_gauss_newton_fisher_equals_newton_on_linear_least_squares() -> None:
    """For a residual linear in ``theta`` the GN Fisher is the Hessian: one natural step recovers lstsq."""
    rng = np.random.RandomState(30)
    x = jnp.asarray(rng.randn(20, 5))
    y = jnp.asarray(rng.randn(20))

    def residual_fn(theta: Array) -> Array:
        return x @ theta - y

    theta0 = jnp.zeros(5)
    fisher, g = gauss_newton_fisher(residual_fn, theta0)
    assert jnp.allclose(fisher, (x.T @ x) / 20.0, atol=1e-10)
    assert jnp.allclose(g, (x.T @ (x @ theta0 - y)) / 20.0, atol=1e-10)
    delta = natural_gradient_direction(fisher, g, damping=0.0)
    p_star, _, _, _ = jnp.linalg.lstsq(x, y, rcond=None)
    assert jnp.allclose(theta0 - delta, p_star, atol=1e-6)


def test_natural_gradient_step_recovers_regression_in_one_step() -> None:
    """The Fisher-scoring parameter update lands on the least-squares solution in one step."""
    rng = np.random.RandomState(31)
    x = jnp.asarray(rng.randn(24, 5))
    y = jnp.asarray(rng.randn(24))

    def residual_fn(theta: Array) -> Array:
        return x @ theta - y

    theta0 = jnp.zeros(5)
    fisher, g = gauss_newton_fisher(residual_fn, theta0)
    theta1 = natural_gradient_step(theta0, g, fisher, learning_rate=1.0, damping=0.0)
    p_star, _, _, _ = jnp.linalg.lstsq(x, y, rcond=None)
    assert jnp.allclose(theta1, p_star, atol=1e-6)


def test_natural_gradient_step_newton_on_quadratic() -> None:
    """With the exact Hessian as metric the natural step is Newton: exact on a quadratic."""
    a = _spd(6, 32)
    b = jnp.asarray(np.random.RandomState(33).randn(6))
    theta_star = jnp.linalg.solve(a, b)
    theta = jnp.zeros(6)
    for _ in range(2):
        grad = a @ theta - b  # grad of 0.5 theta^T A theta - b^T theta
        theta = natural_gradient_step(theta, grad, a, learning_rate=1.0, damping=0.0)
    assert jnp.allclose(theta, theta_star, atol=1e-8)
