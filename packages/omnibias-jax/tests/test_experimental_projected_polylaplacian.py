# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Tri-Laplacian through the projected bi-Laplacian program vs dense nested autodiff."""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from omnibias.jax.experimental.projected_bilaplacian import projected_deep_bilaplacian  # noqa: E402
from omnibias.jax.experimental.projected_polylaplacian import (
    projected_deep_trilaplacian,  # noqa: E402
)

_ACT = {"tanh": jnp.tanh, "sin": jnp.sin, None: lambda z: z}


def _layers(rng, dim, widths, acts, scale=1.0):
    dims = [dim] + list(widths)
    layers = []
    for i, act in enumerate(acts):
        W = jnp.asarray(scale * rng.standard_normal((dims[i + 1], dims[i])) / np.sqrt(dims[i]))
        layers.append((W, jnp.asarray(0.1 * rng.standard_normal(dims[i + 1])), act))
    layers.append((jnp.asarray(rng.standard_normal((1, dims[-1])) / np.sqrt(dims[-1])), jnp.asarray([0.2]), None))
    return layers


def _net(layers, x):
    a = x
    for W, b, spec in layers:
        a = _ACT[spec](W @ a + b)
    return a[0]


def _lap_of(f, idx):
    def lap(x):
        return jnp.trace(jax.hessian(f)(x)[np.ix_(idx, idx)])

    return lap


def _dense_terms(layers, X, idx):
    """(u, |grad|^2, Delta, Delta^2, Delta^3, u_t) by nested dense autodiff over the coordinates idx."""
    idx = np.asarray(idx)

    def point(x):
        u = lambda y: _net(layers, y)  # noqa: E731
        l1 = _lap_of(u, idx)
        l2 = _lap_of(l1, idx)
        l3 = _lap_of(l2, idx)
        g = jax.grad(u)(x)
        return u(x), g[idx] @ g[idx], l1(x), l2(x), l3(x), g[0]

    return jax.vmap(point)(X)


def _close(a, b, tol=1e-9):
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape, (a.shape, b.shape)
    np.testing.assert_allclose(a, b, rtol=tol, atol=tol * (1.0 + np.max(np.abs(b))))


# (input dimension, widths, activations): projection active (D > H), D <= H, deeper, affine middle layer, sin.
_CASES = [
    (7, [3, 4, 4], ["tanh"] * 3),
    (6, [4, 4], ["tanh", "tanh"]),
    (3, [5, 4, 4], ["tanh"] * 3),
    (8, [3, 5, 4], ["tanh", None, "tanh"]),
    (7, [3, 4], ["sin", "sin"]),
]


@pytest.mark.parametrize(("dim", "widths", "acts"), _CASES)
def test_terms_match_dense_autodiff(dim, widths, acts):
    rng = np.random.default_rng(10 + dim)
    layers = _layers(rng, dim, widths, acts)
    X = jnp.asarray(rng.standard_normal((3, dim)) / np.sqrt(dim))
    value, g2, lap, bil, tri, dv = projected_deep_trilaplacian(X, layers)
    assert dv is None
    ref = _dense_terms(layers, X, np.arange(dim))
    for got, want in zip((value, g2, lap, bil, tri), ref[:5], strict=False):
        _close(got[:, 0], want)


def test_coordinates_time_dependent_and_drift():
    rng = np.random.default_rng(3)
    D = 7
    layers = _layers(rng, D + 1, [3, 4, 4], ["tanh"] * 3)
    X = jnp.asarray(rng.standard_normal((3, D + 1)) / np.sqrt(D))
    v = np.zeros((D + 1, 1))
    v[0, 0] = 1.0
    value, g2, lap, bil, tri, dv = projected_deep_trilaplacian(X, layers, coordinates=np.arange(1, D + 1), directions=jnp.asarray(v))
    ref = _dense_terms(layers, X, np.arange(1, D + 1))
    for got, want in zip((value, g2, lap, bil, tri), ref[:5], strict=False):
        _close(got[:, 0], want)
    _close(dv[:, 0, 0], ref[5])


def test_outer_over_coordinates_agrees_with_projected_outer():
    rng = np.random.default_rng(4)
    layers = _layers(rng, 9, [3, 4, 4], ["tanh"] * 3)
    X = jnp.asarray(rng.standard_normal((2, 9)) / 3.0)
    a = projected_deep_trilaplacian(X, layers)[4]
    b = projected_deep_trilaplacian(X, layers, outer="coordinates")[4]
    _close(a, b, 1e-10)


def test_lower_terms_are_the_bilaplacian_program():
    rng = np.random.default_rng(5)
    layers = _layers(rng, 8, [3, 4, 4], ["tanh"] * 3)
    X = jnp.asarray(rng.standard_normal((3, 8)) / 3.0)
    got = projected_deep_trilaplacian(X, layers)
    want = projected_deep_bilaplacian(X, layers)
    for g, w in zip(got[:4], want[:4], strict=False):
        _close(g, w, 1e-13)


@pytest.mark.parametrize("chunk", [1, 3, 100])
def test_chunking_and_padding_do_not_change_values(chunk):
    rng = np.random.default_rng(6)
    layers = _layers(rng, 9, [4, 4, 4], ["tanh"] * 3)
    X = jnp.asarray(rng.standard_normal((3, 9)) / 3.0)
    ref = projected_deep_trilaplacian(X, layers)[4]
    _close(projected_deep_trilaplacian(X, layers, chunk=chunk)[4], ref, 1e-12)


def _loss(fn, layers, X, f):
    return 0.5 * jnp.mean((fn(X, layers) - f) ** 2)


def _tri_module(X, layers):
    return projected_deep_trilaplacian(X, layers)[4][:, 0]


def _tri_dense(X, layers):
    return _dense_terms(layers, X, np.arange(X.shape[1]))[4]


def _params(layers):
    return [(W, b) for W, b, _ in layers]


def _with_params(layers, params):
    return [(W, b, spec) for (W, b), (_, _, spec) in zip(params, layers, strict=False)]


@pytest.mark.parametrize(("dim", "widths"), [(7, [3, 4, 4]), (3, [5, 4, 4])])
def test_parameter_gradient_and_hessian_vector_product(dim, widths):
    rng = np.random.default_rng(20 + dim)
    layers = _layers(rng, dim, widths, ["tanh"] * len(widths))
    X = jnp.asarray(rng.standard_normal((3, dim)) / np.sqrt(dim))
    f = jnp.asarray(rng.standard_normal(3))
    params = _params(layers)

    def make(fn):
        return lambda p: _loss(fn, _with_params(layers, p), X, f)

    lm, ld = make(_tri_module), make(_tri_dense)
    assert abs(float(lm(params)) - float(ld(params))) < 1e-9 * (1 + abs(float(ld(params))))
    gm, gd = jax.grad(lm)(params), jax.grad(ld)(params)
    for (Wm, bm), (Wd, bd) in zip(gm, gd, strict=False):
        _close(Wm, Wd, 1e-8)
        _close(bm, bd, 1e-8)
    tangent = [(jnp.asarray(rng.standard_normal(W.shape)), jnp.asarray(rng.standard_normal(b.shape))) for W, b in params]
    hm = jax.jvp(jax.grad(lm), (params,), (tangent,))[1]
    hd = jax.jvp(jax.grad(ld), (params,), (tangent,))[1]
    for (Wm, bm), (Wd, bd) in zip(hm, hd, strict=False):
        _close(Wm, Wd, 1e-7)
        _close(bm, bd, 1e-7)


def test_rank_deficient_first_layer_uses_the_stable_fallback():
    rng = np.random.default_rng(7)
    layers = _layers(rng, 8, [4, 4, 4], ["tanh"] * 3)
    W1 = np.asarray(layers[0][0]).copy()
    W1[3] = W1[0] + W1[1]
    layers[0] = (jnp.asarray(W1), layers[0][1], layers[0][2])
    X = jnp.asarray(rng.standard_normal((2, 8)) / 3.0)
    f = jnp.asarray(rng.standard_normal(2))
    _close(projected_deep_trilaplacian(X, layers)[4][:, 0], _tri_dense(X, layers), 1e-8)
    params = _params(layers)

    def make(fn):
        return lambda p: _loss(fn, _with_params(layers, p), X, f)

    gm, gd = jax.grad(make(_tri_module))(params), jax.grad(make(_tri_dense))(params)
    for (Wm, bm), (Wd, bd) in zip(gm, gd, strict=False):
        _close(Wm, Wd, 1e-7)
        _close(bm, bd, 1e-7)


def test_phase_field_crystal_residual():
    rng = np.random.default_rng(8)
    D = 6
    layers = _layers(rng, D + 1, [3, 4, 4], ["tanh"] * 3)
    X = jnp.asarray(rng.standard_normal((3, D + 1)) / np.sqrt(D))
    v = np.zeros((D + 1, 1))
    v[0, 0] = 1.0
    u, g2, lap, bil, tri, dv = projected_deep_trilaplacian(
        X, layers, coordinates=np.arange(1, D + 1), directions=jnp.asarray(v)
    )
    r = -0.3
    res = dv[:, 0, 0] - (1 + r) * lap[:, 0] - 2 * bil[:, 0] - tri[:, 0] - 3 * u[:, 0] ** 2 * lap[:, 0] - 6 * u[:, 0] * g2[:, 0]

    def pfc(x):
        f = lambda y: _net(layers, y)  # noqa: E731
        idx = np.arange(1, D + 1)
        lap_of = lambda g: _lap_of(g, idx)  # noqa: E731

        def rhs(y):
            return (1 + r) * f(y) + 2 * lap_of(f)(y) + lap_of(lap_of(f))(y) + f(y) ** 3

        return jax.grad(f)(x)[0] - lap_of(rhs)(x)

    _close(res, jax.vmap(pfc)(X), 1e-8)


def test_single_point_jit_and_validation():
    rng = np.random.default_rng(9)
    layers = _layers(rng, 7, [3, 4], ["tanh", "tanh"])
    x = jnp.asarray(rng.standard_normal(7) / 3.0)
    one = projected_deep_trilaplacian(x, layers)
    many = jax.jit(lambda z: projected_deep_trilaplacian(z, layers))(x[None, :])
    for a, b in zip(one[:5], many[:5], strict=False):
        _close(a, b[0], 1e-12)
    with pytest.raises(ValueError):
        projected_deep_trilaplacian(x, layers, outer="bogus")
    with pytest.raises(ValueError):
        projected_deep_trilaplacian(x, [])
