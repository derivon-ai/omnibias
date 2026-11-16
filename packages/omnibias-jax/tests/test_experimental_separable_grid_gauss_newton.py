# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Gauss-Newton products of :mod:`omnibias.jax.experimental.separable_grid` against dense Jacobians."""

from __future__ import annotations

import string

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.flatten_util  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from omnibias.jax.experimental.separable_grid import (  # noqa: E402
    SeparableTerm,
    axis_derivative_tables,
    mlp_axis_tables,
    separable_grid_gauss_newton,
    separable_grid_loss,
)


def _e(D, *pairs):
    a = [0] * D
    for d, k in pairs:
        a[d] += k
    return tuple(a)


def _operators(D):
    zero = (0,) * D
    mixed = (
        [SeparableTerm(0.7, _e(D, (0, 1), (D - 1, 3)))]
        if D > 1
        else [SeparableTerm(0.7, _e(D, (0, 4)))]
    )
    return {
        "laplace": [SeparableTerm(1.0, _e(D, (d, 2))) for d in range(D)],
        "helmholtz_mixed": [SeparableTerm(1.0, _e(D, (d, 2))) for d in range(D)]
        + [SeparableTerm(9.0, zero)]
        + mixed,
        "drift": [SeparableTerm(1.0, _e(D, (d, 2))) for d in range(D)]
        + [
            SeparableTerm(-1.0, _e(D, (d, 1)), tuple(1 if a == d else 0 for a in range(D)))
            for d in range(D)
        ]
        + [SeparableTerm(0.5, zero, (2,) * D)],
    }


def _axis_params(key, R, width=6):
    sizes = [1, width, width, R]
    out = []
    for i in range(3):
        key, kw, kb = jax.random.split(key, 3)
        std = (2.0 / (sizes[i] + sizes[i + 1])) ** 0.5
        out.append(
            (
                std * jax.random.normal(kw, (sizes[i], sizes[i + 1])),
                0.2 * jax.random.normal(kb, (sizes[i + 1],)),
            )
        )
    return out


def _mlp(params, x):
    h = x[:, None]
    for W, b in params[:-1]:
        h = jnp.tanh(h @ W + b)
    W, b = params[-1]
    return h @ W + b


def _setup(D, N, R, seed, with_target):
    rng = np.random.default_rng(seed)
    key = jax.random.PRNGKey(seed)
    params = [_axis_params(k, R) for k in jax.random.split(key, D)]
    nodes = [jnp.asarray(np.sort(rng.uniform(-1.0, 1.0, N + d))) for d in range(D)]
    weights = [jnp.asarray(rng.uniform(0.2, 1.0, N + d)) for d in range(D)]
    coefficients = [jnp.stack([jnp.ones_like(x), x, jnp.cos(x)]) for x in nodes]
    target = (
        [jnp.asarray(rng.standard_normal((2, N + d))) for d in range(D)] if with_target else None
    )
    return params, nodes, weights, coefficients, target


def _explicit_residual(tabs, terms, coefficients, target):
    D = len(tabs)
    L = string.ascii_lowercase[:D]
    spec = ",".join(f"{c}z" for c in L) + "->" + L
    r = 0.0
    for t in terms:
        mats = [
            tabs[d][t.orders[d]]
            if t.factors is None
            else coefficients[d][t.factors[d]][:, None] * tabs[d][t.orders[d]]
            for d in range(D)
        ]
        r = r + t.coeff * jnp.einsum(spec, *mats)
    if target is not None:
        r = r - jnp.einsum(spec, *[f.T for f in target])
    return r


def _grid_weights(weights):
    D = len(weights)
    L = string.ascii_lowercase[:D]
    return jnp.einsum(",".join(L) + "->" + L, *weights)


@pytest.mark.parametrize("with_target", [False, True])
@pytest.mark.parametrize("op", ["laplace", "helmholtz_mixed", "drift"])
@pytest.mark.parametrize("D", [1, 2, 3, 4])
def test_product_matches_dense_jacobian(D, op, with_target):
    N = {1: 7, 2: 5, 3: 4, 4: 3}[D]
    params, nodes, weights, coefficients, target = _setup(D, N, 3, 10 * D + len(op), with_target)
    terms = _operators(D)[op]
    order = max(max(t.orders) for t in terms)

    def tables_fn(p):
        return [
            axis_derivative_tables(lambda x, q=p[d]: _mlp(q, x), nodes[d], order) for d in range(D)
        ]

    kw = {"weights": weights, "target": target, "coefficients": coefficients}
    loss, grad, gnvp = separable_grid_gauss_newton(tables_fn, params, terms, **kw)
    flat, unravel = jax.flatten_util.ravel_pytree(params)
    resid = jax.jit(
        lambda f: _explicit_residual(tables_fn(unravel(f)), terms, coefficients, target).ravel()
    )
    J = jax.jacfwd(resid)(flat)
    q = _grid_weights(weights).ravel()
    r = resid(flat)
    rng = np.random.default_rng(D)
    v = jnp.asarray(rng.standard_normal(flat.shape))
    ref = J.T @ (q * (J @ v))
    got = jax.flatten_util.ravel_pytree(gnvp(unravel(v)))[0]
    scale = float(jnp.max(jnp.abs(ref)))
    assert float(jnp.max(jnp.abs(got - ref))) <= 1e-12 * scale
    assert abs(float(loss) - 0.5 * float(jnp.sum(q * r * r))) <= 1e-12 * float(jnp.sum(q * r * r))
    g_ref = J.T @ (q * r)
    g_got = jax.flatten_util.ravel_pytree(grad)[0]
    assert float(jnp.max(jnp.abs(g_got - g_ref))) <= 1e-12 * float(jnp.max(jnp.abs(g_ref)))


def test_loss_and_gradient_equal_separable_grid_loss():
    D, N, R, K = 3, 6, 4, 2
    key = jax.random.PRNGKey(0)
    layers = []
    for a, b in ((1, 8), (8, 8), (8, R)):
        key, k1, k2 = jax.random.split(key, 3)
        layers.append(
            (jax.random.normal(k1, (D, a, b)) / a**0.5, 0.1 * jax.random.normal(k2, (D, b)))
        )
    nodes = jnp.stack([jnp.linspace(-1.0, 1.0, N)] * D)
    terms = _operators(D)["laplace"] + [SeparableTerm(-3.0, (0, 0, 0))]
    target = [jnp.cos(jnp.arange(1, 3)[:, None] * nodes[d][None]) for d in range(D)]

    def tables_fn(p):
        return mlp_axis_tables(p, nodes, K)

    loss, grad, _ = separable_grid_gauss_newton(tables_fn, layers, terms, target=target)
    ref_loss, ref_grad = jax.value_and_grad(
        lambda p: separable_grid_loss(tables_fn(p), terms, target=target)
    )(layers)
    assert abs(float(loss - ref_loss)) <= 1e-13 * abs(float(ref_loss))
    for a, b in zip(
        jax.tree_util.tree_leaves(grad), jax.tree_util.tree_leaves(ref_grad), strict=True
    ):
        assert np.allclose(a, b, rtol=0.0, atol=1e-12 * float(jnp.max(jnp.abs(b))))


def _product_fixture():
    params, nodes, weights, coefficients, target = _setup(3, 4, 3, 5, True)
    terms = _operators(3)["drift"]

    def tables_fn(p):
        return [axis_derivative_tables(lambda x, q=p[d]: _mlp(q, x), nodes[d], 2) for d in range(3)]

    out = separable_grid_gauss_newton(
        tables_fn, params, terms, weights=weights, target=target, coefficients=coefficients
    )
    return params, out


def _dot(u, v):
    return sum(
        float(jnp.vdot(a, b))
        for a, b in zip(jax.tree_util.tree_leaves(u), jax.tree_util.tree_leaves(v), strict=True)
    )


def _random_like(params, seed):
    leaves, treedef = jax.tree_util.tree_flatten(params)
    rng = np.random.default_rng(seed)
    return jax.tree_util.tree_unflatten(
        treedef, [jnp.asarray(rng.standard_normal(a.shape)) for a in leaves]
    )


def test_product_is_symmetric_positive_semidefinite_and_linear():
    params, (_, _, gnvp) = _product_fixture()
    u, v = _random_like(params, 1), _random_like(params, 2)
    Gu, Gv = gnvp(u), gnvp(v)
    assert abs(_dot(u, Gv) - _dot(v, Gu)) <= 1e-12 * abs(_dot(u, Gv))
    assert _dot(v, Gv) >= 0.0
    combo = gnvp(jax.tree_util.tree_map(lambda a, b: 2.5 * a - b, u, v))
    expect = jax.tree_util.tree_map(lambda a, b: 2.5 * a - b, Gu, Gv)
    for a, b in zip(
        jax.tree_util.tree_leaves(combo), jax.tree_util.tree_leaves(expect), strict=True
    ):
        assert np.allclose(a, b, rtol=0.0, atol=1e-12 * float(jnp.max(jnp.abs(b)) + 1.0))


def test_damped_step_by_conjugate_gradients_inside_jit():
    params, (_, grad, gnvp) = _product_fixture()
    lam = 1e-2

    @jax.jit
    def step(g):
        return jax.scipy.sparse.linalg.cg(
            lambda x: jax.tree_util.tree_map(lambda a, b: a + lam * b, gnvp(x), x),
            g,
            tol=1e-14,
            maxiter=2000,
        )[0]

    x = step(grad)
    flat_g, unravel = jax.flatten_util.ravel_pytree(grad)
    G = jnp.stack(
        [jax.flatten_util.ravel_pytree(gnvp(unravel(e)))[0] for e in jnp.eye(flat_g.size)], axis=1
    )
    ref = jnp.linalg.solve(G + lam * jnp.eye(flat_g.size), flat_g)
    got = jax.flatten_util.ravel_pytree(x)[0]
    assert float(jnp.max(jnp.abs(got - ref))) <= 1e-7 * float(jnp.max(jnp.abs(ref)))


def test_refuses_invalid_terms():
    params, nodes, weights, coefficients, target = _setup(2, 4, 3, 3, False)

    def tables_fn(p):
        return [axis_derivative_tables(lambda x, q=p[d]: _mlp(q, x), nodes[d], 1) for d in range(2)]

    with pytest.raises(ValueError, match="holds orders up to 1"):
        separable_grid_gauss_newton(tables_fn, params, [SeparableTerm(1.0, (2, 0))])
    with pytest.raises(ValueError, match="coefficients tables"):
        separable_grid_gauss_newton(tables_fn, params, [SeparableTerm(1.0, (1, 0), (0, 0))])
