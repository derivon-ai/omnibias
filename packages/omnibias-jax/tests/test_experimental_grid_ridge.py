# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Explicit-grid, gradient and refusal contracts for :mod:`omnibias.jax.experimental.grid_ridge`."""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from omnibias.jax.experimental import RidgeTerm  # noqa: E402
from omnibias.jax.experimental.grid_ridge import closed_ridge_grid_loss  # noqa: E402

ACTS = {"sin": jnp.sin, "cos": jnp.cos, "exp": jnp.exp}
TERMS = {
    "laplace": [RidgeTerm("laplace", 1)],
    "mixed": [
        RidgeTerm("laplace", 2, 0.5),
        RidgeTerm("coordinate", 3, 1.0, 0),
        RidgeTerm("coordinate", 1, 0.3, 1),
    ],
    "high": [RidgeTerm("coordinate", 8, 1.0, 1), RidgeTerm("laplace", 3, -0.2)],
}


def _grid(D: int, N: int, H: int, seed: int, Q: int = 2, scale: float = 0.6):
    rng = np.random.default_rng(seed)
    W = jnp.asarray(rng.standard_normal((H, D)) * scale)
    b = jnp.asarray(rng.standard_normal(H) * 0.3)
    c = jnp.asarray(rng.standard_normal(H) / np.sqrt(H))
    nodes = [jnp.asarray(np.sort(rng.uniform(-1.5, 1.5, N + d))) for d in range(D)]
    weights = [jnp.asarray(rng.uniform(0.2, 1.0, N + d)) for d in range(D)]
    target = [jnp.asarray(rng.standard_normal((Q, N + d))) for d in range(D)]
    return W, b, c, nodes, weights, target


def _explicit(nodes, weights, target):
    """Explicit points, product weights and target values in C order."""
    mesh = jnp.meshgrid(*nodes, indexing="ij")
    x = jnp.stack([m.reshape(-1) for m in mesh], axis=1)
    q = jnp.ones(())
    for w in weights:
        q = jnp.multiply.outer(q, w)
    f = 0.0
    for k in range(target[0].shape[0]):
        t = jnp.ones(())
        for factor in target:
            t = jnp.multiply.outer(t, factor[k])
        f = f + t
    return x, q.reshape(-1), jnp.reshape(f, (-1,))


def _reference_loss(W, b, c, x, q, f, terms, act):
    from omnibias.jax.experimental.closed_ridge import closed_ridge_operator

    r = closed_ridge_operator(W, b, c, x, terms, act) - f
    return 0.5 * jnp.sum(q * r * r)


def _close(a, b, rtol):
    a, b = np.asarray(a), np.asarray(b)
    assert np.max(np.abs(a - b)) <= rtol * max(np.max(np.abs(b)), 1e-300)


@pytest.mark.parametrize("act", sorted(ACTS))
@pytest.mark.parametrize("name", sorted(TERMS))
@pytest.mark.parametrize("D,N,H", [(1, 9, 5), (2, 6, 7), (3, 4, 6), (4, 3, 4)])
@pytest.mark.parametrize("with_target", [True, False])
def test_matches_explicit_grid_value_and_gradient(act, name, D, N, H, with_target):
    terms = [t for t in TERMS[name] if t.kind == "laplace" or t.axis < D]
    W, b, c, nodes, weights, target = _grid(D, N, H, seed=D * 31 + N)
    tgt = target if with_target else None
    x, q, f = _explicit(nodes, weights, target)
    f = f if with_target else jnp.zeros_like(f)

    def grid(p):
        return closed_ridge_grid_loss(
            p[0], p[1], p[2], nodes, terms, weights=weights, target=tgt, activation=act
        )

    def ref(p):
        return _reference_loss(p[0], p[1], p[2], x, q, f, terms, act)

    v, g = jax.value_and_grad(grid)((W, b, c))
    v_ref, g_ref = jax.value_and_grad(ref)((W, b, c))
    _close(v, v_ref, 1e-12)
    for a, r in zip(g, g_ref, strict=True):
        _close(a, r, 1e-11)


def test_laplacian_matches_nested_autodiff():
    W, b, c, nodes, weights, target = _grid(2, 5, 4, seed=7)
    x, q, f = _explicit(nodes, weights, target)

    def u(p):
        return jnp.sum(c * jnp.sin(W @ p + b))

    lap = jax.vmap(lambda p: jnp.trace(jax.hessian(u)(p)))(x)
    expected = 0.5 * jnp.sum(q * (lap - f) ** 2)
    got = closed_ridge_grid_loss(
        W, b, c, nodes, [RidgeTerm("laplace", 1)], weights=weights, target=target
    )
    _close(got, expected, 1e-12)


def test_default_weights_are_the_grid_mean():
    W, b, c, nodes, _, target = _grid(3, 4, 5, seed=3)
    x, _, f = _explicit(nodes, [jnp.ones_like(n) for n in nodes], target)
    terms = TERMS["mixed"]
    expected = _reference_loss(W, b, c, x, jnp.full(x.shape[:1], 1.0 / x.shape[0]), f, terms, "sin")
    _close(closed_ridge_grid_loss(W, b, c, nodes, terms, target=target), expected, 1e-12)


@pytest.mark.parametrize("act", sorted(ACTS))
def test_single_node_axis_is_a_boundary_face(act):
    W, b, c, nodes, weights, target = _grid(3, 7, 5, seed=13)
    nodes[1], weights[1], target[1] = nodes[1][:1], jnp.ones(1), target[1][:, :1]
    x, q, f = _explicit(nodes, weights, target)
    terms = TERMS["mixed"]
    expected = _reference_loss(W, b, c, x, q, f, terms, act)
    got = closed_ridge_grid_loss(
        W, b, c, nodes, terms, weights=weights, target=target, activation=act
    )
    _close(got, expected, 1e-12)


def test_jit_matches_eager():
    W, b, c, nodes, weights, target = _grid(3, 5, 6, seed=11)
    terms = TERMS["high"]

    def loss(W, b, c):
        return closed_ridge_grid_loss(W, b, c, nodes, terms, weights=weights, target=target)

    eager = jax.value_and_grad(loss, argnums=(0, 1, 2))(W, b, c)
    jitted = jax.jit(jax.value_and_grad(loss, argnums=(0, 1, 2)))(W, b, c)
    for a, r in zip(
        jax.tree_util.tree_leaves(jitted), jax.tree_util.tree_leaves(eager), strict=True
    ):
        _close(a, r, 1e-13)


@pytest.mark.parametrize("act", ["tanh", "sigmoid", "no_such_activation"])
def test_non_exponential_activations_are_refused(act):
    W, b, c, nodes, _, _ = _grid(2, 4, 3, seed=0)
    with pytest.raises(ValueError):
        closed_ridge_grid_loss(W, b, c, nodes, [RidgeTerm("laplace", 1)], activation=act)


def test_malformed_grids_are_refused():
    W, b, c, nodes, weights, target = _grid(2, 4, 3, seed=0)
    terms = [RidgeTerm("laplace", 1)]
    with pytest.raises(ValueError):
        closed_ridge_grid_loss(W, b, c, nodes[:1], terms)
    with pytest.raises(ValueError):
        closed_ridge_grid_loss(W, b, c, nodes, terms, weights=[weights[0], weights[0]])
    with pytest.raises(ValueError):
        closed_ridge_grid_loss(W, b, c, nodes, terms, target=[target[0], target[1][:1]])
    with pytest.raises(ValueError):
        closed_ridge_grid_loss(W, b, c, nodes, [RidgeTerm("coordinate", 1, 1.0, 2)])
