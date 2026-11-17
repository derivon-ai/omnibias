# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Woodbury solves with residual MLP blocks vs dense solves with the autodiff Jacobian."""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from omnibias.jax.experimental.residual_woodbury import (  # noqa: E402
    residual_inverse,
    residual_inverse_newton,
    residual_solve,
    residual_solve_transpose,
)

_ACT = {"tanh": jnp.tanh, "sin": jnp.sin, "gelu": lambda z: jax.nn.gelu(z, approximate=False)}


def _layers(rng, dims, outputs, act="tanh", scale=0.5):
    layers = []
    for i in range(len(dims) - 1):
        W = jnp.asarray(rng.standard_normal((dims[i + 1], dims[i])) / np.sqrt(dims[i]))
        b = jnp.asarray(0.1 * rng.standard_normal(dims[i + 1]))
        layers.append((W, b, act))
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


def _dense(layers, X, G, transpose):
    def point(x, g):
        J = jax.jacfwd(lambda y: _forward(layers, y))(x)
        M = jnp.eye(J.shape[0]) + (J.T if transpose else J)
        return _forward(layers, x), jnp.linalg.solve(M, g)

    return jax.vmap(point)(X, G)


def _close(a, b, tol=1e-11):
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape
    np.testing.assert_allclose(a, b, rtol=tol, atol=tol * (1.0 + np.max(np.abs(b))))


# (dim, width, hidden layers): the H_n x H_n Woodbury system (width small against dim) and the D x D Jacobian
# (dim small against width), one hidden layer and deeper networks, and the dim == width boundary.
_SHAPES = [(7, 3, 1), (9, 4, 2), (12, 5, 3), (40, 6, 2), (3, 8, 1), (4, 9, 3), (6, 6, 2), (8, 2, 4)]


@pytest.mark.parametrize(("dim", "width", "depth"), _SHAPES)
@pytest.mark.parametrize("act", ["tanh", "sin", "gelu"])
@pytest.mark.parametrize("transpose", [False, True])
def test_solve_matches_dense(dim, width, depth, act, transpose):
    rng = np.random.default_rng(dim * 100 + width * 10 + depth)
    layers = _layers(rng, [dim] + [width] * depth, dim, act)
    X = jnp.asarray(rng.standard_normal((5, dim)))
    G = jnp.asarray(rng.standard_normal((5, dim)))
    value, y = residual_solve(X, layers, G, transpose=transpose)
    ref_value, ref_y = _dense(layers, X, G, transpose)
    _close(value, ref_value)
    _close(y, ref_y)
    # the solution satisfies the system
    def apply(x, v):
        f = lambda z: _forward(layers, z)  # noqa: E731
        if transpose:
            return v + jax.vjp(f, x)[1](v)[0]
        return v + jax.jvp(f, (x,), (v,))[1]

    _close(jax.vmap(apply)(X, y), G)


def test_transpose_alias_and_single_point():
    rng = np.random.default_rng(3)
    layers = _layers(rng, [10, 4, 4], 10)
    x = jnp.asarray(rng.standard_normal(10))
    g = jnp.asarray(rng.standard_normal(10))
    v1, y1 = residual_solve_transpose(x, layers, g)
    v2, y2 = residual_solve(x[None], layers, g[None], transpose=True)
    assert y1.shape == (10,) and v1.shape == (10,)
    _close(y1, y2[0])
    _close(v1, v2[0])


def test_adjoint_is_the_transposed_system():
    rng = np.random.default_rng(5)
    layers = _layers(rng, [11, 4, 4], 11)
    X = jnp.asarray(rng.standard_normal((6, 11)))
    G = jnp.asarray(rng.standard_normal((6, 11)))
    H = jnp.asarray(rng.standard_normal((6, 11)))
    _, y = residual_solve(X, layers, G)
    _, z = residual_solve(X, layers, H, transpose=True)
    # <H, (I+J)^{-1} G> == <(I+J)^{-T} H, G>
    _close(jnp.sum(H * y, axis=-1), jnp.sum(z * G, axis=-1))


@pytest.mark.parametrize(("dim", "width", "depth"), [(9, 3, 1), (12, 4, 3), (3, 7, 2)])
def test_parameter_gradients_match_dense(dim, width, depth):
    rng = np.random.default_rng(40 + dim + width + depth)
    layers = _layers(rng, [dim] + [width] * depth, dim)
    X = jnp.asarray(rng.standard_normal((4, dim)))
    G = jnp.asarray(rng.standard_normal((4, dim)))
    T = jnp.asarray(rng.standard_normal((4, dim)))
    flat = [(W, b) for W, b, _ in layers]
    specs = [s for *_, s in layers]

    def rebuild(p):
        return [(W, b, s) for (W, b), s in zip(p, specs, strict=True)]

    def ours(p):
        return 0.5 * jnp.mean(jnp.sum((residual_solve(X, rebuild(p), G)[1] - T) ** 2, axis=-1))

    def ref(p):
        return 0.5 * jnp.mean(jnp.sum((_dense(rebuild(p), X, G, False)[1] - T) ** 2, axis=-1))

    ga, gb = jax.grad(ours)(flat), jax.grad(ref)(flat)
    for (Wa, ba), (Wb, bb) in zip(ga, gb, strict=True):
        _close(Wa, Wb, 1e-9)
        _close(ba, bb, 1e-9)
    # input gradients too
    _close(jax.grad(lambda x: jnp.sum(residual_solve(x, layers, G)[1]))(X),
           jax.grad(lambda x: jnp.sum(_dense(layers, x, G, False)[1]))(X), 1e-9)


@pytest.mark.parametrize(("dim", "width", "depth"), [(12, 4, 2), (30, 5, 3), (4, 9, 2)])
def test_newton_inverts_the_residual_block(dim, width, depth):
    rng = np.random.default_rng(7 + dim)
    layers = _layers(rng, [dim] + [width] * depth, dim, scale=0.8)
    X = jnp.asarray(rng.standard_normal((6, dim)) / np.sqrt(dim))
    Y = X + _forward(layers, X)
    x, steps, res = residual_inverse_newton(Y, layers, tol=1e-13)
    assert int(steps) <= 12
    assert float(jnp.max(res)) <= 1e-13
    _close(x, X, 1e-10)
    # a start close to the solution takes no more steps
    x2, steps2, _ = residual_inverse_newton(Y, layers, tol=1e-13, x0=X + 1e-3)
    _close(x2, X, 1e-10)
    assert int(steps2) <= int(steps)


def test_newton_single_point_and_converged_start():
    rng = np.random.default_rng(11)
    layers = _layers(rng, [8, 3, 3], 8, scale=0.8)
    x_true = jnp.asarray(rng.standard_normal(8) / 3)
    y = x_true + _forward(layers, x_true)
    x, steps, res = residual_inverse_newton(y, layers, x0=x_true)
    assert x.shape == (8,) and res.shape == ()
    assert int(steps) == 0
    _close(x, x_true)


def test_jit_and_vmap():
    rng = np.random.default_rng(13)
    layers = _layers(rng, [10, 4, 4], 10)
    X = jnp.asarray(rng.standard_normal((4, 10)))
    G = jnp.asarray(rng.standard_normal((4, 10)))
    jitted = jax.jit(lambda x, g: residual_solve(x, layers, g)[1])
    _close(jitted(X, G), residual_solve(X, layers, G)[1])
    mapped = jax.vmap(lambda x, g: residual_solve(x, layers, g)[1])(X, G)
    _close(mapped, residual_solve(X, layers, G)[1])


def test_validation():
    rng = np.random.default_rng(17)
    layers = _layers(rng, [6, 3], 6)
    X = jnp.zeros((2, 6))
    with pytest.raises(ValueError, match="rhs must have the shape of x"):
        residual_solve(X, layers, jnp.zeros((2, 5)))
    with pytest.raises(ValueError, match="at least one hidden layer"):
        residual_solve(X, layers[-1:], X)
    with pytest.raises(ValueError, match="affine"):
        residual_solve(X, [layers[0], (layers[1][0], layers[1][1], "tanh")], X)
    with pytest.raises(ValueError, match="outputs"):
        residual_solve(jnp.zeros((2, 5)), _layers(rng, [5, 3], 4), jnp.zeros((2, 5)))
    with pytest.raises(ValueError, match="shape"):
        residual_inverse_newton(jnp.zeros((2, 3, 4)), layers)


def test_inverse_methods_agree_and_auto_switches_only_when_the_iteration_is_slow():
    rng = np.random.default_rng(23)
    dim, width = 24, 8
    layers = _layers(rng, [dim] + [width] * 2, dim, scale=0.05)  # a strong contraction: no solve is needed
    X = jnp.asarray(rng.standard_normal((5, dim)) / np.sqrt(dim))
    Y = X + _forward(layers, X)
    results = {m: residual_inverse(Y, layers, method=m, tol=1e-13) for m in ("newton", "fixed_point", "auto")}
    for method, (x, _, res) in results.items():
        assert float(jnp.max(res)) <= 1e-13, method
        _close(x, X, 1e-10)
    assert int(results["auto"][1]) == int(results["fixed_point"][1])
    # a map whose fixed-point factor is close to one: the plain iteration is still far from the tolerance after 500 steps
    layers = _layers(rng, [dim] + [width] * 2, dim, scale=0.9)
    Y = X + _forward(layers, X)
    slow = residual_inverse(Y, layers, method="fixed_point", tol=1e-13)
    newton = residual_inverse(Y, layers, method="newton", tol=1e-13)
    auto = residual_inverse(Y, layers, method="auto", tol=1e-13)
    assert float(jnp.max(slow[2])) > 1e-13 and int(slow[1]) == 500
    for x, steps, res in (newton, auto):
        assert float(jnp.max(res)) <= 1e-13 and int(steps) < 40
        _close(x, X, 1e-10)


def test_inverse_auto_converges_where_the_fixed_point_does_not():
    rng = np.random.default_rng(29)
    dim, width = 16, 16
    W1 = jnp.asarray(rng.standard_normal((width, dim)) / np.sqrt(dim))
    b1 = jnp.asarray(0.1 * rng.standard_normal(width))
    # s(x) = W1^T tanh(W1 x + b1): I + J is positive definite (a monotone map); the fixed-point factor is about |W1|^2 > 1
    layers = [(W1, b1, "tanh"), (W1.T, jnp.zeros(dim), None)]
    X = jnp.asarray(rng.standard_normal((4, dim)) / np.sqrt(dim))
    Y = X + _forward(layers, X)
    _, _, res_fixed = residual_inverse(Y, layers, method="fixed_point", tol=1e-12, max_iter=60)
    x, steps, res = residual_inverse(Y, layers, method="auto", tol=1e-12, max_iter=60)
    assert float(jnp.max(res_fixed)) > 1e-6
    assert float(jnp.max(res)) <= 1e-12 and int(steps) < 60
    _close(x, X, 1e-8)
    single, _, _ = residual_inverse(Y[0], layers, method="auto", tol=1e-12, x0=X[0])
    assert single.shape == (dim,)


def test_inverse_validation():
    rng = np.random.default_rng(31)
    layers = _layers(rng, [6, 3], 6)
    with pytest.raises(ValueError, match="method must be"):
        residual_inverse(jnp.zeros((2, 6)), layers, method="secant")
