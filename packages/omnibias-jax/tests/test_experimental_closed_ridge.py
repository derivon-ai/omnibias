# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Oracle, gradient and structure contracts for :mod:`omnibias.jax.experimental.closed_ridge`."""

from __future__ import annotations

from fractions import Fraction

import mpmath as mp
import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from omnibias.jax.experimental import RidgeTerm  # noqa: E402
from omnibias.jax.experimental.closed_ridge import (  # noqa: E402
    closed_ridge_operator,
    closed_ridge_residual_loss,
    derivative_closed_system,
)

ACTS = {"sin": jnp.sin, "cos": jnp.cos, "exp": jnp.exp}
MP_DIFF = {
    "sin": lambda z, m: mp.sin(z + m * mp.pi / 2),
    "cos": lambda z, m: mp.cos(z + m * mp.pi / 2),
    "exp": lambda z, m: mp.exp(z),
}


def _data(D: int, H: int = 7, B: int = 9, seed: int = 0, scale: float = 0.6):
    rng = np.random.default_rng(seed)
    W = jnp.asarray(rng.standard_normal((H, D)) * scale)
    b = jnp.asarray(rng.standard_normal(H) * 0.3)
    c = jnp.asarray(rng.standard_normal(H) / np.sqrt(H))
    x = jnp.asarray(rng.standard_normal((B, D)) * 0.5)
    f = jnp.asarray(rng.standard_normal(B))
    return W, b, c, x, f


def test_registered_systems():
    states, C, a = derivative_closed_system("sin")
    assert states == ("sin", "cos")
    assert C == ((0, 1), (-1, 0)) and a == (1, 0)
    assert derivative_closed_system("exp")[1:] == (((Fraction(1),),), (Fraction(1),))


@pytest.mark.parametrize("act", ["tanh", "sigmoid", "gaussian", "softplus", "no_such_activation"])
def test_non_closed_activations_are_refused(act):
    W, b, c, x, f = _data(2)
    with pytest.raises(ValueError):
        closed_ridge_residual_loss(W, b, c, x, f, [RidgeTerm("laplace", 1)], act)


@pytest.mark.parametrize("act", sorted(ACTS))
@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6])
def test_coordinate_derivative_matches_nested_autodiff(act, n):
    W, b, c, x, _ = _data(2, H=4, B=3)
    s = ACTS[act]

    def u(point):
        return jnp.sum(c * s(W @ point + b))

    g = u
    for _ in range(n):
        g = (lambda h: lambda p: jax.grad(h)(p)[1])(g)
    ref = jax.vmap(g)(x)
    got = closed_ridge_operator(W, b, c, x, [RidgeTerm("coordinate", n, axis=1)], act)
    np.testing.assert_allclose(np.asarray(got), np.asarray(ref), rtol=1e-9, atol=1e-11)


@pytest.mark.parametrize("act", sorted(ACTS))
@pytest.mark.parametrize("D", [2, 3, 7])
def test_laplacian_matches_hessian_trace(act, D):
    W, b, c, x, _ = _data(D)
    s = ACTS[act]

    def u(point):
        return jnp.sum(c * s(W @ point + b))

    ref = jax.vmap(lambda p: jnp.trace(jax.hessian(u)(p)))(x)
    got = closed_ridge_operator(W, b, c, x, [RidgeTerm("laplace", 1)], act)
    np.testing.assert_allclose(np.asarray(got), np.asarray(ref), rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("act", sorted(ACTS))
@pytest.mark.parametrize(
    "terms",
    [
        (RidgeTerm("laplace", 3),),
        (RidgeTerm("coordinate", 13, 1.0, 2),),
        (RidgeTerm("laplace", 8, 0.5), RidgeTerm("coordinate", 7, -2.0, 0)),
    ],
)
def test_high_orders_against_arbitrary_precision(act, terms):
    W, b, c, x, _ = _data(3, H=3, B=2, scale=0.4)
    got = np.asarray(closed_ridge_operator(W, b, c, x, terms, act))
    Wn, bn, cn, xn = (np.asarray(v) for v in (W, b, c, x))
    with mp.workdps(40):
        ref = []
        for row in xn:
            acc = mp.mpf(0)
            for h in range(3):
                z = sum(mp.mpf(float(Wn[h, d])) * mp.mpf(float(row[d])) for d in range(3)) + mp.mpf(
                    float(bn[h])
                )
                q = sum(mp.mpf(float(Wn[h, d])) ** 2 for d in range(3))
                for t in terms:
                    m = 2 * t.order if t.kind == "laplace" else t.order
                    g = (
                        q**t.order
                        if t.kind == "laplace"
                        else mp.mpf(float(Wn[h, t.axis])) ** t.order
                    )
                    acc += mp.mpf(float(cn[h])) * mp.mpf(t.coeff) * g * MP_DIFF[act](z, m)
            ref.append(float(acc))
    np.testing.assert_allclose(got, np.asarray(ref), rtol=1e-11, atol=1e-13)


TERMS = [
    (RidgeTerm("laplace", 1, 0.5), RidgeTerm("laplace", 2, 0.25), RidgeTerm("laplace", 3, 0.125)),
    (
        RidgeTerm("coordinate", 4, 1.0, 1),
        RidgeTerm("laplace", 1, -0.3),
        RidgeTerm("coordinate", 1, 0.7, 0),
    ),
    (RidgeTerm("coordinate", 3, 1.0, 0), RidgeTerm("coordinate", 5, 0.2, 1)),
    (RidgeTerm("coordinate", 9, 1.0, 0),),
]


@pytest.mark.parametrize("act", sorted(ACTS))
@pytest.mark.parametrize("terms", TERMS)
@pytest.mark.parametrize("D", [2, 7, 12])
def test_custom_gradient_matches_autodiff_of_operator(act, terms, D):
    W, b, c, x, f = _data(D)

    def ref_loss(W, b, c):
        r = closed_ridge_operator(W, b, c, x, terms, act) - f
        return 0.5 * jnp.mean(r * r)

    def loss(W, b, c):
        return closed_ridge_residual_loss(W, b, c, x, f, terms, act)

    v, g = jax.value_and_grad(loss, argnums=(0, 1, 2))(W, b, c)
    vr, gr = jax.value_and_grad(ref_loss, argnums=(0, 1, 2))(W, b, c)
    np.testing.assert_allclose(float(v), float(vr), rtol=1e-12)
    for a_, r_ in zip(g, gr, strict=True):
        np.testing.assert_allclose(np.asarray(a_), np.asarray(r_), rtol=1e-9, atol=1e-11)


def _transcendental_count(fn, *args) -> int:
    jaxpr = jax.make_jaxpr(fn)(*args)
    names = {"sin", "cos", "exp"}

    def subjaxprs(params):
        for value in params.values():
            for item in value if isinstance(value, tuple | list) else (value,):
                if hasattr(item, "eqns"):
                    yield item
                elif hasattr(getattr(item, "jaxpr", None), "eqns"):
                    yield item.jaxpr

    def walk(jx) -> int:
        n = 0
        for eqn in jx.eqns:
            n += eqn.primitive.name in names
            for sub in subjaxprs(eqn.params):
                n += walk(sub)
        return n

    return walk(jaxpr.jaxpr)


@pytest.mark.parametrize("act", ["sin", "exp"])
def test_value_and_gradient_work_is_order_independent(act):
    W, b, c, x, f = _data(3)
    r = len(derivative_closed_system(act)[0])
    counts = []
    for terms in (
        [RidgeTerm("laplace", 1), RidgeTerm("coordinate", 1, 1.0, 0)],
        [
            RidgeTerm("laplace", 8),
            RidgeTerm("coordinate", 15, 1.0, 1),
            RidgeTerm("coordinate", 2, 1.0, 0),
        ],
    ):
        vg = jax.value_and_grad(
            lambda W, b, c, terms=terms: closed_ridge_residual_loss(W, b, c, x, f, terms, act),
            argnums=(0, 1, 2),
        )
        counts.append(_transcendental_count(vg, W, b, c))
    assert counts[0] == counts[1] <= 2 * r


def test_jit_and_repeat_calls_share_one_rule():
    W, b, c, x, f = _data(3)
    terms = [RidgeTerm("laplace", 2)]
    fn = jax.jit(
        jax.value_and_grad(
            lambda W, b, c: closed_ridge_residual_loss(W, b, c, x, f, terms), argnums=(0, 1, 2)
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
        [RidgeTerm("coordinate", 65)],
        [RidgeTerm("laplace", 33)],
        [RidgeTerm("coordinate", 2, 1.0, 5)],
        [RidgeTerm("curl", 1)],
    ],  # type: ignore[arg-type]
)
def test_unsupported_terms_are_refused(terms):
    W, b, c, x, f = _data(3)
    with pytest.raises(ValueError):
        closed_ridge_residual_loss(W, b, c, x, f, terms)
