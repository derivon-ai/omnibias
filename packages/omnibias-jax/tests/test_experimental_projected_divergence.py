# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Divergence of deep MLP vector fields through the input/output projection vs Jacobian traces."""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from omnibias.jax.experimental.projected_divergence import (  # noqa: E402
    projected_deep_divergence,
)

_ACT = {"tanh": jnp.tanh, "sin": jnp.sin, "gelu": lambda z: jax.nn.gelu(z, approximate=False)}


def _layers(rng, dims, outputs, act="tanh", affine_at=None):
    layers = []
    for i in range(len(dims) - 1):
        W = jnp.asarray(rng.standard_normal((dims[i + 1], dims[i])) / np.sqrt(dims[i]))
        b = jnp.asarray(0.1 * rng.standard_normal(dims[i + 1]))
        layers.append((W, b, None if i == affine_at else act))
    W = jnp.asarray(rng.standard_normal((outputs, dims[-1])) / np.sqrt(dims[-1]))
    layers.append((W, jnp.asarray(0.1 * rng.standard_normal(outputs)), None))
    return layers


def _forward(layers, x):
    a = x
    for W, b, spec in layers:
        a = a @ W.T + (0.0 if b is None else b)
        if spec is not None:
            a = _ACT[spec](a)
    return a


def _reference(layers, X, index=None):
    def point(x):
        jac = jax.jacfwd(lambda y: _forward(layers, y))(x)
        idx = np.arange(jac.shape[0]) if index is None else np.asarray(index)
        return _forward(layers, x), jnp.sum(jac[np.arange(jac.shape[0]), idx])

    return jax.vmap(point)(X)


def _close(a, b, tol=1e-12):
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape
    np.testing.assert_allclose(a, b, rtol=tol, atol=tol * (1.0 + np.max(np.abs(b))))


# (dim, width, hidden layers): no hidden layer, one, two, and deeper networks with the projected
# columns (width small against dim) or the input columns (dim small against width).
_SHAPES = [
    (7, 5, 0),
    (7, 5, 1),
    (4, 9, 1),
    (7, 5, 2),
    (3, 8, 2),
    (9, 4, 3),
    (3, 8, 3),
    (12, 5, 4),
    (4, 9, 4),
]


@pytest.mark.parametrize(("dim", "width", "depth"), _SHAPES)
@pytest.mark.parametrize("act", ["tanh", "sin", "gelu"])
def test_divergence_matches_jacobian_trace(dim, width, depth, act):
    rng = np.random.default_rng(dim * 10 + width + depth)
    layers = _layers(rng, [dim] + [width] * depth, dim, act)
    X = jnp.asarray(rng.standard_normal((4, dim)))
    for a, b in zip(projected_deep_divergence(X, layers), _reference(layers, X), strict=True):
        _close(a, b)


@pytest.mark.parametrize("depth", [1, 2, 3])
def test_affine_hidden_layer(depth):
    rng = np.random.default_rng(depth)
    layers = _layers(rng, [8] + [5] * depth, 8, affine_at=0)
    X = jnp.asarray(rng.standard_normal((3, 8)))
    for a, b in zip(projected_deep_divergence(X, layers), _reference(layers, X), strict=True):
        _close(a, b)


@pytest.mark.parametrize("depth", [1, 2, 3, 4])
def test_coordinates_pair_outputs_with_a_subset_of_inputs(depth):
    rng = np.random.default_rng(40 + depth)
    layers = _layers(rng, [7] + [4] * depth, 6)
    X = jnp.asarray(rng.standard_normal((3, 7)))
    for index in (range(6), range(1, 7), [6, 0, 2, 1, 5, 3]):
        ours = projected_deep_divergence(X, layers, coordinates=index)
        for a, b in zip(ours, _reference(layers, X, index), strict=True):
            _close(a, b)


@pytest.mark.parametrize(
    ("dim", "width", "depth"), [(7, 5, 1), (7, 5, 2), (9, 4, 3), (3, 8, 3), (12, 5, 4)]
)
def test_parameter_and_input_gradients(dim, width, depth):
    rng = np.random.default_rng(70 + dim + width + depth)
    layers = _layers(rng, [dim] + [width] * depth, dim)
    X = jnp.asarray(rng.standard_normal((5, dim)))
    f = jnp.asarray(rng.standard_normal(5))
    params = [(W, b) for W, b, _ in layers]
    specs = [s for _, _, s in layers]

    def loss(ps, X, fn):
        s, div = fn(X, [(W, b, sp) for (W, b), sp in zip(ps, specs, strict=True)])
        return jnp.mean((div - f) ** 2) + 0.5 * jnp.mean(jnp.sum(s * s, axis=-1))

    ours = jax.grad(loss, argnums=(0, 1))(params, X, projected_deep_divergence)
    ref = jax.grad(loss, argnums=(0, 1))(params, X, lambda X, ls: _reference(ls, X))
    for a, b in zip(jax.tree_util.tree_leaves(ours), jax.tree_util.tree_leaves(ref), strict=True):
        _close(a, b, 1e-11)


def test_unbatched_jit_and_vmap():
    rng = np.random.default_rng(5)
    layers = _layers(rng, [9, 4, 4, 4], 9)
    X = jnp.asarray(rng.standard_normal((3, 9)))
    s, div = projected_deep_divergence(X, layers)
    s0, div0 = projected_deep_divergence(X[0], layers)
    assert s0.shape == (9,) and div0.shape == ()
    _close(s0, s[0])
    _close(div0, div[0])
    jitted = jax.jit(lambda X: projected_deep_divergence(X, layers))
    mapped = jax.vmap(lambda x: projected_deep_divergence(x, layers))
    for a, b in zip(jitted(X), (s, div), strict=True):
        _close(a, b)
    for a, b in zip(mapped(X), (s, div), strict=True):
        _close(a, b)


def test_validation():
    rng = np.random.default_rng(6)
    layers = _layers(rng, [5, 4], 5)
    X = jnp.ones((2, 5))
    with pytest.raises(ValueError, match="layers must be non-empty"):
        projected_deep_divergence(X, [])
    with pytest.raises(ValueError, match="x must have shape"):
        projected_deep_divergence(jnp.ones((2, 2, 5)), layers)
    with pytest.raises(ValueError, match="last layer must be affine"):
        projected_deep_divergence(X, [layers[0], (layers[1][0], layers[1][1], "tanh")])
    with pytest.raises(ValueError, match="pass coordinates"):
        projected_deep_divergence(jnp.ones((2, 6)), [(jnp.ones((4, 6)), None, "tanh"), layers[1]])
    with pytest.raises(ValueError, match="coordinates must be"):
        projected_deep_divergence(X, layers, coordinates=[0, 1, 2, 3, 5])
    with pytest.raises(ValueError, match="coordinates must be"):
        projected_deep_divergence(X, layers, coordinates=[0, 1, 2])
    with pytest.raises(ValueError, match=r"layer 0 weight must have shape"):
        projected_deep_divergence(jnp.ones((2, 6)), layers, coordinates=range(5))
    with pytest.raises(ValueError, match=r"layer 1 weight must have shape"):
        projected_deep_divergence(X, [layers[0], (jnp.ones((5, 3)), None, None)])
