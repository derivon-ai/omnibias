# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Oracle and gradient contracts for :mod:`omnibias.jax.experimental.tanh_ridge`."""

from __future__ import annotations

import mpmath as mp
import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from omnibias.jax.experimental import (  # noqa: E402
    RidgeTerm,
    tanh_ridge_operator,
    tanh_ridge_residual_loss,
)
from omnibias.jax.laplacian import neural_field_polylaplacian  # noqa: E402


def _data(D: int, H: int = 7, B: int = 9, seed: int = 0):
    rng = np.random.default_rng(seed)
    W = jnp.asarray(rng.standard_normal((H, D)) * 0.6)
    b = jnp.asarray(rng.standard_normal(H) * 0.3)
    c = jnp.asarray(rng.standard_normal(H) / np.sqrt(H))
    x = jnp.asarray(rng.standard_normal((B, D)))
    f = jnp.asarray(rng.standard_normal(B))
    return W, b, c, x, f


@pytest.mark.parametrize("D", [2, 3, 12])
@pytest.mark.parametrize("k", [1, 2, 3])
@pytest.mark.parametrize("accurate", [False, True])
def test_laplace_power_matches_repository_closed_form(D, k, accurate):
    W, b, c, x, _ = _data(D)
    got = tanh_ridge_operator(W, b, c, x, [RidgeTerm("laplace", k)], accurate_tails=accurate)
    ref = neural_field_polylaplacian(x, W, b, c, "tanh", k)
    np.testing.assert_allclose(np.asarray(got), np.asarray(ref), rtol=1e-9, atol=1e-11)


@pytest.mark.parametrize("n", [2, 3, 4, 5, 6])
def test_coordinate_derivative_matches_nested_autodiff(n):
    W, b, c, x, _ = _data(2, H=4, B=3)

    def u(point):
        return jnp.sum(c * jnp.tanh(W @ point + b))

    def nth(point):
        g = lambda p: u(p)  # noqa: E731
        for _ in range(n):
            g = (lambda h: lambda p: jax.grad(h)(p)[1])(g)
        return g(point)

    ref = jax.vmap(nth)(x)
    got = tanh_ridge_operator(W, b, c, x, [RidgeTerm("coordinate", n, axis=1)])
    np.testing.assert_allclose(np.asarray(got), np.asarray(ref), rtol=1e-8, atol=1e-9)


@pytest.mark.parametrize("n", [2, 4, 6])
def test_exact_table_against_arbitrary_precision(n):
    W, b, c, x, _ = _data(3, H=3, B=2)
    got = np.asarray(tanh_ridge_operator(W, b, c, x, [RidgeTerm("coordinate", n, axis=2)]))
    with mp.workdps(40):
        ref = []
        for row in np.asarray(x):
            acc = mp.mpf(0)
            for h in range(3):
                z = sum(mp.mpf(float(W[h, d])) * mp.mpf(float(row[d])) for d in range(3)) + mp.mpf(
                    float(b[h])
                )
                acc += mp.mpf(float(c[h])) * mp.mpf(float(W[h, 2])) ** n * mp.diff(mp.tanh, z, n)
            ref.append(float(acc))
    np.testing.assert_allclose(got, np.asarray(ref), rtol=1e-10, atol=1e-12)


TERMS = [
    (RidgeTerm("laplace", 1, 0.5), RidgeTerm("laplace", 2, 0.25), RidgeTerm("laplace", 3, 0.125)),
    (RidgeTerm("coordinate", 4, 1.0, 1), RidgeTerm("laplace", 1, -0.3)),
    (RidgeTerm("coordinate", 6, 1.0, 0),),
]


@pytest.mark.parametrize("terms", TERMS)
@pytest.mark.parametrize("D", [2, 12])
@pytest.mark.parametrize("accurate", [False, True])
def test_custom_gradient_matches_autodiff_of_operator(terms, D, accurate):
    W, b, c, x, f = _data(D)

    def ref_loss(W, b, c):
        r = tanh_ridge_operator(W, b, c, x, terms, accurate_tails=accurate) - f
        return 0.5 * jnp.mean(r * r)

    def loss(W, b, c):
        return tanh_ridge_residual_loss(W, b, c, x, f, terms, accurate_tails=accurate)

    v, g = jax.value_and_grad(loss, argnums=(0, 1, 2))(W, b, c)
    vr, gr = jax.value_and_grad(ref_loss, argnums=(0, 1, 2))(W, b, c)
    np.testing.assert_allclose(float(v), float(vr), rtol=1e-12)
    for a, r in zip(g, gr, strict=True):
        np.testing.assert_allclose(np.asarray(a), np.asarray(r), rtol=1e-9, atol=1e-11)


def test_jit_and_repeat_calls_share_one_rule():
    W, b, c, x, f = _data(3)
    terms = [RidgeTerm("laplace", 2)]
    fn = jax.jit(
        jax.value_and_grad(
            lambda W, b, c: tanh_ridge_residual_loss(W, b, c, x, f, terms), argnums=(0, 1, 2)
        )
    )
    first = fn(W, b, c)
    second = fn(W, b, c)
    np.testing.assert_array_equal(np.asarray(first[0]), np.asarray(second[0]))


@pytest.mark.parametrize(
    "terms",
    [
        [],
        [RidgeTerm("laplace", 0)],
        [RidgeTerm("coordinate", 9)],
        [RidgeTerm("laplace", 5)],
        [RidgeTerm("coordinate", 2, 1.0, 5)],
        [RidgeTerm("curl", 1)],
    ],  # type: ignore[arg-type]
)
def test_unsupported_terms_are_refused(terms):
    W, b, c, x, f = _data(3)
    with pytest.raises(ValueError):
        tanh_ridge_residual_loss(W, b, c, x, f, terms)
