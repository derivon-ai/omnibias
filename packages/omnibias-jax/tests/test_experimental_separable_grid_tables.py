# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Stacked MLP derivative tables of :mod:`omnibias.jax.experimental.separable_grid` against Taylor mode and autodiff."""

from __future__ import annotations

import math

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from omnibias.jax.experimental.separable_grid import (  # noqa: E402
    SeparableTerm,
    axis_derivative_tables,
    mlp_axis_tables,
    separable_grid_loss,
)

_ACT = {"tanh": jnp.tanh, "sin": jnp.sin}


def _layers(key, D, sizes, bias=0.3):
    out = []
    for i in range(len(sizes) - 1):
        key, kw, kb = jax.random.split(key, 3)
        std = (2.0 / (sizes[i] + sizes[i + 1])) ** 0.5
        out.append(
            (
                std * jax.random.normal(kw, (D, sizes[i], sizes[i + 1])),
                bias * jax.random.normal(kb, (D, sizes[i + 1])),
            )
        )
    return out


def _axis_mlp(layers, d, act):
    def fn(x):
        h = x[:, None]
        for W, b in layers[:-1]:
            h = act(h @ W[d] + b[d])
        W, b = layers[-1]
        return h @ W[d] + b[d]

    return fn


def _nodes(D, N, seed):
    rng = np.random.default_rng(seed)
    return jnp.asarray(np.sort(rng.uniform(-1.5, 1.5, (D, N)), axis=1))


def _scale(x):
    return float(jnp.max(jnp.abs(x)))


@pytest.mark.parametrize("activation", ["tanh", "sin"])
@pytest.mark.parametrize("order", [0, 1, 2, 4, 6])
@pytest.mark.parametrize("sizes", [(1, 5), (1, 8, 6), (1, 16, 12, 16, 7)])
def test_tables_match_taylor_mode(activation, order, sizes):
    D = 3
    layers = _layers(jax.random.PRNGKey(order + 10 * len(sizes)), D, sizes)
    nodes = _nodes(D, 11, order)
    got = mlp_axis_tables(layers, nodes, order, activation=activation)
    assert got.shape == (D, order + 1, 11, sizes[-1])
    for d in range(D):
        ref = axis_derivative_tables(_axis_mlp(layers, d, _ACT[activation]), nodes[d], order)
        for k in range(order + 1):
            assert np.allclose(got[d, k], ref[k], rtol=0.0, atol=1e-13 * max(1.0, _scale(ref[k])))


@pytest.mark.parametrize("activation", ["tanh", "sin"])
def test_tables_match_nested_autodiff(activation):
    D, K = 2, 4
    layers = _layers(jax.random.PRNGKey(3), D, (1, 6, 5, 3))
    nodes = _nodes(D, 7, 3)
    got = mlp_axis_tables(layers, nodes, K, activation=activation)
    for d in range(D):
        fn = _axis_mlp(layers, d, _ACT[activation])
        for r in range(3):
            g = lambda t, r=r, fn=fn: fn(t[None])[0, r]  # noqa: E731
            for k in range(K + 1):
                ref = jax.vmap(g)(nodes[d])
                assert np.allclose(got[d, k, :, r], ref, rtol=1e-12, atol=1e-12)
                g = jax.grad(g)


def test_one_layer_tanh_closed_form():
    w, b = 1.3, -0.2
    layers = [
        (jnp.full((1, 1, 1), w), jnp.full((1, 1), b)),
        (jnp.ones((1, 1, 1)), jnp.zeros((1, 1))),
    ]
    x = jnp.linspace(-1.0, 1.0, 5)
    got = mlp_axis_tables(layers, x[None], 2)[0, :, :, 0]
    t = jnp.tanh(w * x + b)
    assert np.allclose(got[0], t, atol=1e-15)
    assert np.allclose(got[1], w * (1 - t**2), atol=1e-15)
    assert np.allclose(got[2], -2 * w**2 * t * (1 - t**2), atol=1e-15)


def test_parameter_gradients_match_taylor_mode():
    D, K = 2, 5
    layers = _layers(jax.random.PRNGKey(5), D, (1, 8, 8, 4))
    nodes = _nodes(D, 9, 5)
    probe = jax.random.normal(jax.random.PRNGKey(6), (D, K + 1, 9, 4))

    def stacked(p):
        return jnp.sum(probe * mlp_axis_tables(p, nodes, K))

    def looped(p):
        return sum(
            jnp.sum(probe[d] * axis_derivative_tables(_axis_mlp(p, d, jnp.tanh), nodes[d], K))
            for d in range(D)
        )

    g1, g2 = jax.grad(stacked)(layers), jax.grad(looped)(layers)
    for a, b in zip(jax.tree_util.tree_leaves(g1), jax.tree_util.tree_leaves(g2), strict=True):
        assert np.allclose(a, b, rtol=0.0, atol=1e-12 * max(1.0, _scale(b)))


def test_loss_with_stacked_tables_matches_per_axis_tables():
    D, N, R = 3, 6, 4
    layers = _layers(jax.random.PRNGKey(7), D, (1, 8, 8, R))
    nodes = _nodes(D, N, 7)
    terms = [
        SeparableTerm(1.0, (2, 0, 0)),
        SeparableTerm(1.0, (0, 2, 0)),
        SeparableTerm(1.0, (0, 0, 2)),
    ]
    terms += [SeparableTerm(0.4, (1, 0, 3)), SeparableTerm(-2.0, (0, 0, 0))]
    target = [jnp.cos(jnp.arange(1, 3)[:, None] * nodes[d][None]) for d in range(D)]

    def stacked(p):
        return separable_grid_loss(mlp_axis_tables(p, nodes, 3), terms, target=target)

    def looped(p):
        tabs = [axis_derivative_tables(_axis_mlp(p, d, jnp.tanh), nodes[d], 3) for d in range(D)]
        return separable_grid_loss(tabs, terms, target=target)

    v1, g1 = jax.value_and_grad(stacked)(layers)
    v2, g2 = jax.value_and_grad(looped)(layers)
    assert abs(float(v1 - v2)) <= 1e-13 * abs(float(v2))
    for a, b in zip(jax.tree_util.tree_leaves(g1), jax.tree_util.tree_leaves(g2), strict=True):
        assert np.allclose(a, b, rtol=0.0, atol=1e-12 * max(1.0, _scale(b)))


def test_jit_matches_eager():
    layers = _layers(jax.random.PRNGKey(8), 2, (1, 4, 3))
    nodes = _nodes(2, 5, 8)
    eager = mlp_axis_tables(layers, nodes, 3, activation="sin")
    jitted = jax.jit(lambda p, x: mlp_axis_tables(p, x, 3, activation="sin"))(layers, nodes)
    assert np.allclose(eager, jitted, rtol=1e-15, atol=1e-15)


def test_factorial_scaling_of_high_orders():
    layers = [(jnp.ones((1, 1, 1)), jnp.zeros((1, 1))), (jnp.ones((1, 1, 1)), jnp.zeros((1, 1)))]
    x = jnp.asarray([[0.3]])
    got = mlp_axis_tables(layers, x, 8, activation="sin")[0, :, 0, 0]
    ref = [math.sin(0.3 + k * math.pi / 2) for k in range(9)]
    assert np.allclose(got, ref, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"order": -1}, "non-negative integer"),
        ({"order": 1.5}, "non-negative integer"),
        ({"activation": "relu"}, "activation must be one of"),
        ({"nodes": jnp.zeros(4)}, r"shape \(D, N\)"),
        ({"nodes": jnp.zeros((2, 0))}, r"shape \(D, N\)"),
        ({"layers": []}, "at least one layer"),
        ({"layers": [(jnp.ones((2, 2, 3)), jnp.ones((2, 3)))]}, "layer weights must have shape"),
        ({"layers": [(jnp.ones((2, 1, 3)), jnp.ones((2, 4)))]}, "layer weights must have shape"),
        ({"layers": [(jnp.ones((3, 1, 3)), jnp.ones((3, 3)))]}, "layer weights must have shape"),
    ],
)
def test_refusals(kwargs, message):
    args = {
        "layers": [(jnp.ones((2, 1, 3)), jnp.ones((2, 3)))],
        "nodes": jnp.zeros((2, 4)),
        "order": 2,
        "activation": "tanh",
    }
    args.update(kwargs)
    with pytest.raises(ValueError, match=message):
        mlp_axis_tables(args["layers"], args["nodes"], args["order"], activation=args["activation"])
