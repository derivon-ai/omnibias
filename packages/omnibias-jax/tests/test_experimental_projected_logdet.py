# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Jacobian log-determinants of residual MLP blocks through the Sylvester identity vs dense slogdet."""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from omnibias.jax.experimental.projected_logdet import (  # noqa: E402
    projected_deep_logdet,
)

_ACT = {"tanh": jnp.tanh, "sin": jnp.sin, "gelu": lambda z: jax.nn.gelu(z, approximate=False)}


def _layers(rng, dims, outputs, act="tanh", affine_at=None, scale=0.4):
    layers = []
    for i in range(len(dims) - 1):
        W = jnp.asarray(rng.standard_normal((dims[i + 1], dims[i])) / np.sqrt(dims[i]))
        b = jnp.asarray(0.1 * rng.standard_normal(dims[i + 1]))
        layers.append((W, b, None if i == affine_at else act))
    W = jnp.asarray(scale * rng.standard_normal((outputs, dims[-1])) / np.sqrt(dims[-1]))
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
        sign, logabsdet = jnp.linalg.slogdet(jnp.eye(jac.shape[0]) + jac[:, idx])
        return _forward(layers, x), sign, logabsdet

    return jax.vmap(point)(X)


def _close(a, b, tol=1e-12):
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape
    np.testing.assert_allclose(a, b, rtol=tol, atol=tol * (1.0 + np.max(np.abs(b))))


# (dim, width, hidden layers): no hidden layer, one, two, and deeper networks with the H_n x H_n
# Sylvester block (width small against dim) or the D x D Jacobian (dim small against width).
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
    (6, 6, 2),
]


@pytest.mark.parametrize(("dim", "width", "depth"), _SHAPES)
@pytest.mark.parametrize("act", ["tanh", "sin", "gelu"])
def test_logdet_matches_dense_slogdet(dim, width, depth, act):
    rng = np.random.default_rng(dim * 10 + width + depth)
    layers = _layers(rng, [dim] + [width] * depth, dim, act)
    X = jnp.asarray(rng.standard_normal((4, dim)))
    for a, b in zip(projected_deep_logdet(X, layers), _reference(layers, X), strict=True):
        _close(a, b)


@pytest.mark.parametrize(("dim", "width", "depth"), [(7, 5, 1), (9, 4, 3), (3, 8, 2)])
def test_negative_determinants_keep_their_sign(dim, width, depth):
    rng = np.random.default_rng(90 + dim + width + depth)
    layers = _layers(rng, [dim] + [width] * depth, dim, scale=6.0)
    X = jnp.asarray(rng.standard_normal((16, dim)))
    ours = projected_deep_logdet(X, layers)
    ref = _reference(layers, X)
    assert np.any(np.asarray(ref[1]) < 0) and np.any(np.asarray(ref[1]) > 0)
    for a, b in zip(ours, ref, strict=True):
        _close(a, b, 1e-10)


@pytest.mark.parametrize("depth", [1, 2, 3])
def test_affine_hidden_layer(depth):
    rng = np.random.default_rng(depth)
    layers = _layers(rng, [8] + [5] * depth, 8, affine_at=0)
    X = jnp.asarray(rng.standard_normal((3, 8)))
    for a, b in zip(projected_deep_logdet(X, layers), _reference(layers, X), strict=True):
        _close(a, b)


@pytest.mark.parametrize(("depth", "width"), [(1, 4), (2, 4), (3, 4), (2, 9)])
def test_coordinates_select_a_block_of_inputs(depth, width):
    rng = np.random.default_rng(40 + depth + width)
    layers = _layers(rng, [7] + [width] * depth, 6)
    X = jnp.asarray(rng.standard_normal((3, 7)))
    for index in (range(6), range(1, 7), [6, 0, 2, 1, 5, 3]):
        ours = projected_deep_logdet(X, layers, coordinates=index)
        for a, b in zip(ours, _reference(layers, X, index), strict=True):
            _close(a, b)


@pytest.mark.parametrize(
    ("dim", "width", "depth"), [(7, 5, 1), (7, 5, 2), (9, 4, 3), (3, 8, 3), (12, 5, 4)]
)
def test_parameter_and_input_gradients(dim, width, depth):
    rng = np.random.default_rng(70 + dim + width + depth)
    layers = _layers(rng, [dim] + [width] * depth, dim)
    X = jnp.asarray(rng.standard_normal((5, dim)))
    params = [(W, b) for W, b, _ in layers]
    specs = [s for _, _, s in layers]

    def nll(ps, X, fn):
        s, _, logabsdet = fn(X, [(W, b, sp) for (W, b), sp in zip(ps, specs, strict=True)])
        y = X + s
        return jnp.mean(0.5 * jnp.sum(y * y, axis=-1) - logabsdet)

    ours = jax.grad(nll, argnums=(0, 1))(params, X, projected_deep_logdet)
    ref = jax.grad(nll, argnums=(0, 1))(params, X, lambda X, ls: _reference(ls, X))
    for a, b in zip(jax.tree_util.tree_leaves(ours), jax.tree_util.tree_leaves(ref), strict=True):
        _close(a, b, 1e-11)


def test_unbatched_jit_and_vmap():
    rng = np.random.default_rng(5)
    layers = _layers(rng, [9, 4, 4, 4], 9)
    X = jnp.asarray(rng.standard_normal((3, 9)))
    full = projected_deep_logdet(X, layers)
    single = projected_deep_logdet(X[0], layers)
    assert single[0].shape == (9,) and single[1].shape == () and single[2].shape == ()
    for a, b in zip(single, full, strict=True):
        _close(a, b[0])
    jitted = jax.jit(lambda X: projected_deep_logdet(X, layers))
    mapped = jax.vmap(lambda x: projected_deep_logdet(x, layers))
    for a, b in zip(jitted(X), full, strict=True):
        _close(a, b)
    for a, b in zip(mapped(X), full, strict=True):
        _close(a, b)


def test_validation():
    rng = np.random.default_rng(6)
    layers = _layers(rng, [5, 4], 5)
    X = jnp.ones((2, 5))
    with pytest.raises(ValueError, match="layers must be non-empty"):
        projected_deep_logdet(X, [])
    with pytest.raises(ValueError, match="x must have shape"):
        projected_deep_logdet(jnp.ones((2, 2, 5)), layers)
    with pytest.raises(ValueError, match="last layer must be affine"):
        projected_deep_logdet(X, [layers[0], (layers[1][0], layers[1][1], "tanh")])
    with pytest.raises(ValueError, match="pass coordinates"):
        projected_deep_logdet(jnp.ones((2, 6)), [(jnp.ones((4, 6)), None, "tanh"), layers[1]])
    with pytest.raises(ValueError, match="coordinates must be"):
        projected_deep_logdet(X, layers, coordinates=[0, 1, 2, 3, 5])
    with pytest.raises(ValueError, match="coordinates must be"):
        projected_deep_logdet(X, layers, coordinates=[0, 1, 2])
    with pytest.raises(ValueError, match=r"layer 0 weight must have shape"):
        projected_deep_logdet(jnp.ones((2, 6)), layers, coordinates=range(5))
    with pytest.raises(ValueError, match=r"layer 1 weight must have shape"):
        projected_deep_logdet(X, [layers[0], (jnp.ones((5, 3)), None, None)])
