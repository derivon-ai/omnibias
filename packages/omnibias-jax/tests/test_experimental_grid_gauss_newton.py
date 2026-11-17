# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Explicit-Jacobian, step and refusal contracts for :mod:`omnibias.jax.experimental.grid_gauss_newton`."""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from jax.flatten_util import ravel_pytree  # noqa: E402
from omnibias.jax.experimental import RidgeTerm  # noqa: E402
from omnibias.jax.experimental.closed_ridge import closed_ridge_operator  # noqa: E402
from omnibias.jax.experimental.grid_gauss_newton import closed_ridge_grid_gauss_newton  # noqa: E402
from omnibias.jax.experimental.grid_ridge import closed_ridge_grid_loss  # noqa: E402

TERMS = {
    "laplace": [RidgeTerm("laplace", 1)],
    "mixed": [
        RidgeTerm("laplace", 2, 0.5),
        RidgeTerm("coordinate", 3, 1.0, 0),
        RidgeTerm("coordinate", 1, 0.3, 1),
    ],
    "high": [RidgeTerm("coordinate", 8, 1.0, 1), RidgeTerm("laplace", 3, -0.2)],
}


def _grid(D: int, N: int, H: int, seed: int):
    rng = np.random.default_rng(seed)
    W = jnp.asarray(rng.standard_normal((H, D)) * 0.6)
    b = jnp.asarray(rng.standard_normal(H) * 0.3)
    c = jnp.asarray(rng.standard_normal(H))
    nodes = [jnp.asarray(np.sort(rng.uniform(-1.2, 1.2, N + d))) for d in range(D)]
    weights = [jnp.asarray(rng.uniform(0.2, 1.0, N + d)) for d in range(D)]
    return W, b, c, nodes, weights


def _explicit(nodes, weights):
    mesh = jnp.meshgrid(*nodes, indexing="ij")
    x = jnp.stack([m.reshape(-1) for m in mesh], axis=1)
    q = jnp.ones(())
    for w in weights:
        q = jnp.multiply.outer(q, w)
    return x, q.reshape(-1)


def _close(a, b, rtol):
    a, b = np.asarray(a), np.asarray(b)
    assert np.max(np.abs(a - b)) <= rtol * max(np.max(np.abs(b)), 1e-300)


def _reference(W, b, c, x, q, terms, act):
    theta, unravel = ravel_pytree((W, b, c))
    J = jax.jacrev(lambda t: closed_ridge_operator(*unravel(t), x, terms, act))(theta)
    return J.T @ (q[:, None] * J), J


@pytest.mark.parametrize("act", ["sin", "cos", "exp"])
@pytest.mark.parametrize("name", sorted(TERMS))
@pytest.mark.parametrize("D,N,H", [(1, 8, 4), (2, 5, 4), (3, 4, 3), (4, 3, 3)])
def test_matches_explicit_jacobian(act, name, D, N, H):
    terms = [t for t in TERMS[name] if t.kind == "laplace" or t.axis < D]
    W, b, c, nodes, weights = _grid(D, N, H, seed=D * 13 + H)
    x, q = _explicit(nodes, weights)
    expected, _ = _reference(W, b, c, x, q, terms, act)
    got = closed_ridge_grid_gauss_newton(W, b, c, nodes, terms, weights=weights, activation=act)
    assert got.shape == ((D + 2) * H, (D + 2) * H)
    _close(got, expected, 1e-13)


def test_laplacian_matches_nested_autodiff():
    W, b, c, nodes, weights = _grid(2, 5, 4, seed=5)
    x, q = _explicit(nodes, weights)
    theta, unravel = ravel_pytree((W, b, c))

    def lap(t):
        W_, b_, c_ = unravel(t)

        def u(p):
            return jnp.sum(c_ * jnp.sin(W_ @ p + b_))

        return jax.vmap(lambda p: jnp.trace(jax.hessian(u)(p)))(x)

    J = jax.jacrev(lap)(theta)
    got = closed_ridge_grid_gauss_newton(W, b, c, nodes, TERMS["laplace"], weights=weights)
    _close(got, J.T @ (q[:, None] * J), 1e-13)


def test_symmetric_positive_semidefinite():
    W, b, c, nodes, weights = _grid(3, 5, 5, seed=2)
    G = closed_ridge_grid_gauss_newton(W, b, c, nodes, TERMS["mixed"], weights=weights)
    _close(G, G.T, 1e-14)
    eig = np.linalg.eigvalsh(np.asarray(G))
    assert eig.min() >= -1e-12 * eig.max()


def test_default_weights_are_the_grid_mean():
    W, b, c, nodes, _ = _grid(3, 4, 3, seed=3)
    x, _ = _explicit(nodes, [jnp.ones_like(n) for n in nodes])
    expected, _ = _reference(
        W, b, c, x, jnp.full(x.shape[:1], 1.0 / x.shape[0]), TERMS["mixed"], "sin"
    )
    _close(closed_ridge_grid_gauss_newton(W, b, c, nodes, TERMS["mixed"]), expected, 1e-13)


def test_jit_and_vmap_match_eager():
    W, b, c, nodes, weights = _grid(2, 6, 4, seed=11)
    terms = TERMS["high"]

    def gn(W, b, c):
        return closed_ridge_grid_gauss_newton(W, b, c, nodes, terms, weights=weights)

    eager = gn(W, b, c)
    _close(jax.jit(gn)(W, b, c), eager, 1e-13)
    Ws = jnp.stack([W, 0.5 * W])
    batched = jax.vmap(gn, in_axes=(0, None, None))(Ws, b, c)
    _close(batched[0], eager, 1e-13)
    _close(batched[1], gn(0.5 * W, b, c), 1e-13)


def test_gauss_newton_step_matches_explicit_jacobian():
    W, b, c, nodes, weights = _grid(2, 6, 3, seed=17)
    rng = np.random.default_rng(0)
    target = [jnp.asarray(rng.standard_normal((2, n.shape[0]))) for n in nodes]
    terms = TERMS["mixed"]
    x, q = _explicit(nodes, weights)
    f = sum(jnp.multiply.outer(target[0][k], target[1][k]).reshape(-1) for k in range(2))
    G_ref, J = _reference(W, b, c, x, q, terms, "sin")
    r = closed_ridge_operator(W, b, c, x, terms, "sin") - f
    grads = jax.grad(closed_ridge_grid_loss, argnums=(0, 1, 2))(
        W, b, c, nodes, terms, weights=weights, target=target
    )
    g = ravel_pytree(grads)[0]
    _close(g, J.T @ (q * r), 1e-12)
    G = closed_ridge_grid_gauss_newton(W, b, c, nodes, terms, weights=weights)
    damp = 1e-3 * jnp.trace(G_ref) / G.shape[0] * jnp.eye(G.shape[0])
    _close(jnp.linalg.solve(G + damp, g), jnp.linalg.solve(G_ref + damp, J.T @ (q * r)), 1e-9)


@pytest.mark.parametrize("act", ["tanh", "sigmoid", "no_such_activation"])
def test_non_exponential_activations_are_refused(act):
    W, b, c, nodes, _ = _grid(2, 4, 3, seed=0)
    with pytest.raises(ValueError):
        closed_ridge_grid_gauss_newton(W, b, c, nodes, TERMS["laplace"], activation=act)


def test_malformed_grids_are_refused():
    W, b, c, nodes, weights = _grid(2, 4, 3, seed=0)
    terms = TERMS["laplace"]
    with pytest.raises(ValueError):
        closed_ridge_grid_gauss_newton(W, b, c, nodes[:1], terms)
    with pytest.raises(ValueError):
        closed_ridge_grid_gauss_newton(W, b, c, nodes, terms, weights=[weights[0], weights[0]])
    with pytest.raises(ValueError):
        closed_ridge_grid_gauss_newton(W, b, c, nodes, [RidgeTerm("coordinate", 1, 1.0, 2)])
