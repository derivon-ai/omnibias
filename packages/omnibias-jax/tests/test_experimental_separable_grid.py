# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Explicit-grid, derivative-table, gradient and refusal contracts for :mod:`omnibias.jax.experimental.separable_grid`."""

from __future__ import annotations

import string

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from omnibias.jax.experimental.separable_grid import (  # noqa: E402
    SeparableTerm,
    axis_derivative_tables,
    separable_grid_loss,
)


def _mlp_init(key, R, width=8, depth=3):
    sizes = [1] + [width] * (depth - 1) + [R]
    params = []
    for i in range(depth):
        key, k = jax.random.split(key)
        std = (2.0 / (sizes[i] + sizes[i + 1])) ** 0.5
        params.append(
            (std * jax.random.normal(k, (sizes[i], sizes[i + 1])), 0.1 * jnp.ones(sizes[i + 1]))
        )
    return params


def _mlp(params, x, act=jnp.tanh):
    h = x[:, None]
    for W, b in params[:-1]:
        h = act(h @ W + b)
    W, b = params[-1]
    return h @ W + b


def _operators(D):
    """Constant coefficients, a mixed partial, an order-0 term and separable coefficient factors."""
    zero = (0,) * D

    def e(*pairs):
        a = [0] * D
        for d, k in pairs:
            a[d] += k
        return tuple(a)

    ops = {
        "laplace": [SeparableTerm(1.0, e((d, 2))) for d in range(D)],
        "helmholtz_mixed": [SeparableTerm(1.0, e((d, 2))) for d in range(D)]
        + [SeparableTerm(9.0, zero)]
        + (
            [SeparableTerm(0.7, e((0, 1), (D - 1, 3)))]
            if D > 1
            else [SeparableTerm(0.7, e((0, 4)))]
        ),
        "drift": [SeparableTerm(1.0, e((d, 2))) for d in range(D)]
        + [
            SeparableTerm(-1.0, e((d, 1)), tuple(1 if a == d else 0 for a in range(D)))
            for d in range(D)
        ]
        + [SeparableTerm(0.5, zero, (2,) * D)],
    }
    return ops


def _setup(D, N, R, seed, Q=2, signed=False):
    rng = np.random.default_rng(seed)
    key = jax.random.PRNGKey(seed)
    params = []
    for _ in range(D):
        key, k = jax.random.split(key)
        params.append(_mlp_init(k, R))
    nodes = [jnp.asarray(np.sort(rng.uniform(-1.0, 1.0, N + d))) for d in range(D)]
    lo = -0.5 if signed else 0.1
    weights = [jnp.asarray(rng.uniform(lo, 1.0, N + d)) for d in range(D)]
    target = [jnp.asarray(rng.standard_normal((Q, N + d))) for d in range(D)]
    coefs = [jnp.stack([jnp.ones_like(x), x, jnp.exp(-x * x)]) for x in nodes]
    return params, nodes, weights, target, coefs


def _tables(params, nodes, terms):
    D = len(nodes)
    orders = [max(t.orders[d] for t in terms) for d in range(D)]
    return [
        axis_derivative_tables(lambda y, p=params[d]: _mlp(p, y), nodes[d], orders[d])
        for d in range(D)
    ]


def _explicit_loss(tables, terms, weights, target, coefs):
    """Materialize the residual on the grid, one outer product per term (the reference route)."""
    D = len(tables)
    L = string.ascii_lowercase[:D]
    spec = ",".join(f"{c}z" for c in L) + "->" + L
    r = 0.0
    for t in terms:
        mats = [tables[d][t.orders[d]] for d in range(D)]
        if t.factors is not None:
            mats = [coefs[d][t.factors[d]][:, None] * m for d, m in enumerate(mats)]
        r = r + t.coeff * jnp.einsum(spec, *mats)
    if target is not None:
        r = r - jnp.einsum(spec, *[f.T for f in target])
    q = jnp.einsum(",".join(L) + "->" + L, *weights)
    return 0.5 * jnp.sum(q * r * r)


def test_tables_match_analytic_derivatives():
    rng = np.random.default_rng(0)
    om, ph = jnp.asarray(rng.uniform(0.5, 3.0, 5)), jnp.asarray(rng.uniform(0, 6.0, 5))
    x = jnp.asarray(np.linspace(-1.0, 1.0, 11))
    tab = axis_derivative_tables(lambda y: jnp.sin(y[:, None] * om + ph), x, 6)
    assert tab.shape == (7, 11, 5)
    for k in range(7):
        ref = om**k * jnp.sin(x[:, None] * om + ph + k * jnp.pi / 2)
        np.testing.assert_allclose(
            tab[k], ref, rtol=1e-12, atol=1e-12 * float(jnp.max(jnp.abs(om**k)))
        )


def test_tables_match_nested_forward_mode():
    params = _mlp_init(jax.random.PRNGKey(3), 4)
    x = jnp.asarray(np.linspace(-0.9, 0.8, 7))
    tab = axis_derivative_tables(lambda y: _mlp(params, y), x, 4)

    def scalar(y):
        return _mlp(params, y[None])[0]

    g = scalar
    for k in range(5):
        np.testing.assert_allclose(tab[k], jax.vmap(g)(x), rtol=1e-12, atol=1e-13)
        g = (lambda g: lambda y: jax.jvp(g, (y,), (jnp.ones_like(y),))[1])(g)


@pytest.mark.parametrize("D", [1, 2, 3, 4])
@pytest.mark.parametrize("op", ["laplace", "helmholtz_mixed", "drift"])
@pytest.mark.parametrize("with_target", [True, False])
def test_loss_and_gradient_match_explicit_grid(D, op, with_target):
    params, nodes, weights, target, coefs = _setup(D, 4 if D < 4 else 3, 3, 10 * D + len(op))
    terms = _operators(D)[op]
    tgt = target if with_target else None

    def gram(p):
        return separable_grid_loss(
            _tables(p, nodes, terms), terms, weights=weights, target=tgt, coefficients=coefs
        )

    def explicit(p):
        return _explicit_loss(_tables(p, nodes, terms), terms, weights, tgt, coefs)

    vg, gg = jax.value_and_grad(gram)(params)
    ve, ge = jax.value_and_grad(explicit)(params)
    np.testing.assert_allclose(vg, ve, rtol=1e-12)
    for a, b in zip(jax.tree_util.tree_leaves(gg), jax.tree_util.tree_leaves(ge), strict=True):
        np.testing.assert_allclose(
            a, b, rtol=1e-10, atol=1e-12 * float(jnp.max(jnp.abs(b)) + 1e-300)
        )


def test_signed_weights_and_table_gradients():
    params, nodes, weights, target, coefs = _setup(3, 4, 3, 77, signed=True)
    terms = _operators(3)["helmholtz_mixed"]
    tabs = _tables(params, nodes, terms)
    vg, gg = jax.value_and_grad(
        lambda t: separable_grid_loss(t, terms, weights=weights, target=target)
    )(tabs)
    ve, ge = jax.value_and_grad(lambda t: _explicit_loss(t, terms, weights, target, None))(tabs)
    np.testing.assert_allclose(vg, ve, rtol=1e-12)
    for a, b in zip(gg, ge, strict=True):
        np.testing.assert_allclose(a, b, rtol=1e-11, atol=1e-13)


def test_laplacian_matches_nested_autodiff_of_the_network():
    D, N, R = 2, 4, 3
    params, nodes, weights, target, _ = _setup(D, N, R, 5)
    terms = _operators(D)["laplace"]

    def u(x):
        return jnp.sum(
            jnp.prod(jnp.stack([_mlp(params[d], x[d : d + 1])[0] for d in range(D)]), axis=0)
        )

    mesh = jnp.meshgrid(*nodes, indexing="ij")
    pts = jnp.stack([m.reshape(-1) for m in mesh], axis=1)
    lap = jax.vmap(lambda x: jnp.trace(jax.hessian(u)(x)))(pts)
    f = sum(
        jnp.multiply.outer(target[0][k], target[1][k]).reshape(-1)
        for k in range(target[0].shape[0])
    )
    q = jnp.multiply.outer(weights[0], weights[1]).reshape(-1)
    ref = 0.5 * jnp.sum(q * (lap - f) ** 2)
    got = separable_grid_loss(_tables(params, nodes, terms), terms, weights=weights, target=target)
    np.testing.assert_allclose(got, ref, rtol=1e-12)


def test_default_weights_are_the_grid_mean():
    params, nodes, _, target, _ = _setup(2, 5, 3, 9)
    terms = _operators(2)["laplace"]
    tabs = _tables(params, nodes, terms)
    mean_w = [jnp.full(x.shape, 1.0 / x.shape[0]) for x in nodes]
    np.testing.assert_allclose(
        separable_grid_loss(tabs, terms, target=target),
        separable_grid_loss(tabs, terms, weights=mean_w, target=target),
        rtol=1e-14,
    )


def test_single_node_axis_is_a_boundary_face():
    params, nodes, weights, target, coefs = _setup(3, 4, 3, 21)
    terms = [SeparableTerm(1.0, (0, 0, 0)), SeparableTerm(0.3, (1, 0, 2))]
    face = [nodes[0], nodes[1][:1], nodes[2]]
    fw = [weights[0], weights[1][:1], weights[2]]
    ft = [target[0], target[1][:, :1], target[2]]
    tabs = _tables(params, face, terms)
    np.testing.assert_allclose(
        separable_grid_loss(tabs, terms, weights=fw, target=ft),
        _explicit_loss(tabs, terms, fw, ft, None),
        rtol=1e-12,
    )


def test_jit_equals_eager_and_vmaps_over_targets():
    params, nodes, weights, target, coefs = _setup(2, 5, 3, 31)
    terms = _operators(2)["drift"]
    tabs = _tables(params, nodes, terms)

    def loss(t, tg):
        return separable_grid_loss(t, terms, weights=weights, target=tg, coefficients=coefs)

    np.testing.assert_allclose(jax.jit(loss)(tabs, target), loss(tabs, target), rtol=1e-14)
    batch = [jnp.stack([f, 2.0 * f]) for f in target]
    got = jax.vmap(loss, in_axes=(None, 0))(tabs, batch)
    np.testing.assert_allclose(got[1], loss(tabs, [2.0 * target[0], 2.0 * target[1]]), rtol=1e-13)


def test_refusals():
    params, nodes, weights, target, coefs = _setup(2, 4, 3, 41)
    terms = _operators(2)["laplace"]
    tabs = _tables(params, nodes, terms)
    with pytest.raises(ValueError, match="at least one axis"):
        separable_grid_loss([], terms)
    with pytest.raises(ValueError, match="at least one operator term"):
        separable_grid_loss(tabs, [])
    with pytest.raises(ValueError, match="same rank"):
        separable_grid_loss([tabs[0], tabs[1][:, :, :2]], terms)
    with pytest.raises(ValueError, match="holds orders up to 2"):
        separable_grid_loss(tabs, [SeparableTerm(1.0, (3, 0))])
    with pytest.raises(ValueError, match="one entry per axis"):
        separable_grid_loss(tabs, [SeparableTerm(1.0, (1,))])
    with pytest.raises(ValueError, match="non-negative"):
        separable_grid_loss(tabs, [SeparableTerm(1.0, (-1, 0))])
    with pytest.raises(ValueError, match="needs the coefficients"):
        separable_grid_loss(tabs, [SeparableTerm(1.0, (1, 0), (1, 0))])
    with pytest.raises(ValueError, match="out of range"):
        separable_grid_loss(tabs, [SeparableTerm(1.0, (1, 0), (3, 0))], coefficients=coefs)
    with pytest.raises(ValueError, match="weight arrays must have shape"):
        separable_grid_loss(tabs, terms, weights=[weights[0][:2], weights[1]])
    with pytest.raises(ValueError, match="expected 2 target"):
        separable_grid_loss(tabs, terms, target=target[:1])
    with pytest.raises(ValueError, match="same number of terms"):
        separable_grid_loss(tabs, terms, target=[target[0], target[1][:1]])
    with pytest.raises(ValueError, match="(K_i \\+ 1, N_i, R)"):
        separable_grid_loss([tabs[0][0], tabs[1]], terms)
    with pytest.raises(ValueError, match="non-negative integer"):
        axis_derivative_tables(lambda y: y[:, None], nodes[0], -1)
    with pytest.raises(ValueError, match="one-dimensional"):
        axis_derivative_tables(lambda y: y[:, None], jnp.zeros((2, 2)), 1)
    with pytest.raises(ValueError, match=r"\(N, R\) features"):
        axis_derivative_tables(lambda y: y, nodes[0], 1)
