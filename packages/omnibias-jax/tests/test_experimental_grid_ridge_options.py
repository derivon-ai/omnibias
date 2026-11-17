# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Grid-value targets and separable variable coefficients of :mod:`omnibias.jax.experimental.grid_ridge`."""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from omnibias.jax.experimental import RidgeTerm  # noqa: E402
from omnibias.jax.experimental.closed_ridge import closed_ridge_operator  # noqa: E402
from omnibias.jax.experimental.grid_ridge import closed_ridge_grid_loss  # noqa: E402

ACTS = ("sin", "cos", "exp")
DIMS = [(1, 9, 5), (2, 6, 7), (3, 4, 6), (4, 3, 4)]


def _setup(D: int, N: int, H: int, seed: int, U: int = 3, Q: int = 2):
    rng = np.random.default_rng(seed)
    W = jnp.asarray(rng.standard_normal((H, D)) * 0.6)
    b = jnp.asarray(rng.standard_normal(H) * 0.3)
    c = jnp.asarray(rng.standard_normal(H) / np.sqrt(H))
    nodes = [jnp.asarray(np.sort(rng.uniform(-1.5, 1.5, N + d))) for d in range(D)]
    weights = [jnp.asarray(rng.uniform(0.2, 1.0, N + d)) for d in range(D)]
    coefs = [jnp.asarray(rng.uniform(0.5, 1.5, (U - (d % 2), N + d))) for d in range(D)]
    target = [jnp.asarray(rng.standard_normal((Q, N + d))) for d in range(D)]
    values = jnp.asarray(rng.standard_normal(tuple(N + d for d in range(D))))
    return W, b, c, nodes, weights, coefs, target, values


def _terms(D: int):
    """Terms with repeated and distinct coefficient patterns (two pairs share a pattern); the last is the identity."""
    terms = [
        RidgeTerm("laplace", 1, 0.7),
        RidgeTerm("coordinate", 2, -0.4, 0),
        RidgeTerm("coordinate", 1, 0.9, D - 1),
    ]
    terms += [RidgeTerm("laplace", 2, 0.05), RidgeTerm("coordinate", 0, 1.3, 0)]
    factors = [
        tuple(1 if d == 0 else 0 for d in range(D)),
        tuple(1 if d == 0 else 0 for d in range(D)),
        tuple(d % 2 for d in range(D)),
        (0,) * D,
        (0,) * D,
    ]
    return terms, factors


def _outer(vectors):
    t = jnp.ones(())
    for v in vectors:
        t = jnp.multiply.outer(t, v)
    return t.reshape(-1)


def _points(nodes):
    mesh = jnp.meshgrid(*nodes, indexing="ij")
    return jnp.stack([m.reshape(-1) for m in mesh], axis=1)


def _reference(W, b, c, nodes, weights, terms, act, *, f=None, coefs=None, factors=None):
    x = _points(nodes)
    q = _outer(weights)
    Lu = 0.0
    for k, term in enumerate(terms):
        a = 1.0 if coefs is None else _outer([coefs[d][factors[k][d]] for d in range(len(nodes))])
        if term.order == 0:
            Lu = Lu + a * term.coeff * (getattr(jnp, act)(x @ W.T + b) @ c)
        else:
            Lu = Lu + a * closed_ridge_operator(W, b, c, x, [term], act)
    r = Lu - (0.0 if f is None else f)
    return 0.5 * jnp.sum(q * r * r)


def _close(a, b, rtol):
    a, b = np.asarray(a), np.asarray(b)
    assert np.max(np.abs(a - b)) <= rtol * max(np.max(np.abs(b)), 1e-300)


def _check(grid, ref, params):
    vg, gg = jax.value_and_grad(grid)(params)
    vr, gr = jax.value_and_grad(ref)(params)
    _close(vg, vr, 1e-12)
    for a, r in zip(gg, gr, strict=True):
        _close(a, r, 1e-11)


@pytest.mark.parametrize("act", ACTS)
@pytest.mark.parametrize("D,N,H", DIMS)
def test_grid_value_target_matches_explicit_grid(act, D, N, H):
    W, b, c, nodes, weights, _, _, values = _setup(D, N, H, seed=7 * D + N)
    terms, _ = _terms(D)

    def grid(p):
        return closed_ridge_grid_loss(
            *p, nodes, terms, weights=weights, target_values=values, activation=act
        )

    def ref(p):
        return _reference(*p, nodes, weights, terms, act, f=values.reshape(-1))

    _check(grid, ref, (W, b, c))


@pytest.mark.parametrize("act", ACTS)
@pytest.mark.parametrize("D,N,H", DIMS[:3])
def test_grid_values_of_a_separable_target_equal_the_factor_form(act, D, N, H):
    W, b, c, nodes, weights, _, target, _ = _setup(D, N, H, seed=11 * D + N)
    terms, _ = _terms(D)
    values = sum(_outer([target[d][k] for d in range(D)]) for k in range(target[0].shape[0]))
    shape = tuple(x.shape[0] for x in nodes)
    a = closed_ridge_grid_loss(
        W, b, c, nodes, terms, weights=weights, target=target, activation=act
    )
    v = closed_ridge_grid_loss(
        W,
        b,
        c,
        nodes,
        terms,
        weights=weights,
        target_values=values.reshape(shape),
        activation=act,
    )
    _close(v, a, 1e-12)


@pytest.mark.parametrize("act", ACTS)
@pytest.mark.parametrize("D,N,H", DIMS)
@pytest.mark.parametrize("target_kind", ["none", "factors", "values"])
def test_variable_coefficients_match_explicit_grid(act, D, N, H, target_kind):
    W, b, c, nodes, weights, coefs, target, values = _setup(D, N, H, seed=13 * D + N)
    terms, factors = _terms(D)
    kw = (
        {"target": target}
        if target_kind == "factors"
        else {"target_values": values}
        if target_kind == "values"
        else {}
    )
    f = None
    if target_kind == "factors":
        f = sum(_outer([target[d][k] for d in range(D)]) for k in range(target[0].shape[0]))
    elif target_kind == "values":
        f = values.reshape(-1)

    def grid(p):
        return closed_ridge_grid_loss(
            *p,
            nodes,
            terms,
            weights=weights,
            coefficients=coefs,
            term_factors=factors,
            activation=act,
            **kw,
        )

    def ref(p):
        return _reference(*p, nodes, weights, terms, act, f=f, coefs=coefs, factors=factors)

    _check(grid, ref, (W, b, c))


@pytest.mark.parametrize("act", ACTS)
def test_unit_coefficients_reproduce_the_constant_coefficient_loss(act):
    D, N, H = 3, 5, 6
    W, b, c, nodes, weights, _, target, _ = _setup(D, N, H, seed=5)
    terms, _ = _terms(D)
    ones = [jnp.ones((1, x.shape[0])) for x in nodes]
    plain = closed_ridge_grid_loss(
        W, b, c, nodes, terms, weights=weights, target=target, activation=act
    )
    unit = closed_ridge_grid_loss(
        W,
        b,
        c,
        nodes,
        terms,
        weights=weights,
        target=target,
        coefficients=ones,
        term_factors=[(0,) * D] * len(terms),
        activation=act,
    )
    _close(unit, plain, 1e-13)


def test_variable_coefficient_drift_matches_nested_autodiff():
    """``Delta u - x . grad u`` (Ornstein-Uhlenbeck generator) against Hessians and gradients of the network."""
    D, N, H = 2, 7, 5
    W, b, c, nodes, weights, _, _, _ = _setup(D, N, H, seed=3)
    terms = [RidgeTerm("laplace", 1)] + [RidgeTerm("coordinate", 1, -1.0, d) for d in range(D)]
    coefs = [jnp.stack([jnp.ones_like(x), x]) for x in nodes]
    factors = [(0,) * D] + [tuple(1 if e == d else 0 for e in range(D)) for d in range(D)]

    def u(y):
        return jnp.sum(c * jnp.sin(W @ y + b))

    x = _points(nodes)
    lap = jax.vmap(lambda y: jnp.trace(jax.hessian(u)(y)))(x)
    drift = jax.vmap(lambda y: y @ jax.grad(u)(y))(x)
    r = lap - drift
    ref = 0.5 * jnp.sum(_outer(weights) * r * r)
    got = closed_ridge_grid_loss(
        W, b, c, nodes, terms, weights=weights, coefficients=coefs, term_factors=factors
    )
    _close(got, ref, 1e-12)


def test_options_compile_under_jit():
    D, N, H = 3, 4, 5
    W, b, c, nodes, weights, coefs, _, values = _setup(D, N, H, seed=17)
    terms, factors = _terms(D)

    def loss(W, b, c):
        return closed_ridge_grid_loss(
            W,
            b,
            c,
            nodes,
            terms,
            weights=weights,
            target_values=values,
            coefficients=coefs,
            term_factors=factors,
        )

    _close(jax.jit(loss)(W, b, c), loss(W, b, c), 1e-14)


def test_identity_term_matches_explicit_helmholtz():
    D, N, H = 2, 6, 5
    W, b, c, nodes, weights, _, target, _ = _setup(D, N, H, seed=23)
    terms = [RidgeTerm("laplace", 1), RidgeTerm("laplace", 0, 16.0)]
    f = sum(_outer([target[d][k] for d in range(D)]) for k in range(target[0].shape[0]))
    for act in ACTS:
        got = closed_ridge_grid_loss(
            W, b, c, nodes, terms, weights=weights, target=target, activation=act
        )
        _close(got, _reference(W, b, c, nodes, weights, terms, act, f=f), 1e-12)


def test_refusals():
    D, N, H = 2, 4, 3
    W, b, c, nodes, weights, coefs, target, values = _setup(D, N, H, seed=1)
    terms, factors = _terms(D)
    with pytest.raises(ValueError, match="positive integer"):
        closed_ridge_grid_loss(W, b, c, nodes, [RidgeTerm("laplace", -1)])
    with pytest.raises(ValueError, match="at least one"):
        closed_ridge_grid_loss(W, b, c, nodes, [])
    with pytest.raises(ValueError, match="either as separable factors"):
        closed_ridge_grid_loss(W, b, c, nodes, terms, target=target, target_values=values)
    with pytest.raises(ValueError, match="grid shape"):
        closed_ridge_grid_loss(W, b, c, nodes, terms, target_values=values.T)
    with pytest.raises(ValueError, match="given together"):
        closed_ridge_grid_loss(W, b, c, nodes, terms, coefficients=coefs)
    with pytest.raises(ValueError, match="given together"):
        closed_ridge_grid_loss(W, b, c, nodes, terms, term_factors=factors)
    with pytest.raises(ValueError, match="coefficient factor arrays"):
        closed_ridge_grid_loss(W, b, c, nodes, terms, coefficients=coefs[:1], term_factors=factors)
    with pytest.raises(ValueError, match=r"shape \(U_i, N_i\)"):
        closed_ridge_grid_loss(
            W,
            b,
            c,
            nodes,
            terms,
            coefficients=[cf.T for cf in coefs],
            term_factors=factors,
        )
    with pytest.raises(ValueError, match="one factor tuple per term"):
        closed_ridge_grid_loss(W, b, c, nodes, terms, coefficients=coefs, term_factors=factors[:-1])
    with pytest.raises(ValueError, match="factor indices"):
        closed_ridge_grid_loss(
            W, b, c, nodes, terms, coefficients=coefs, term_factors=[(0,)] * len(terms)
        )
    with pytest.raises(ValueError, match="out of range"):
        closed_ridge_grid_loss(
            W,
            b,
            c,
            nodes,
            terms,
            coefficients=coefs,
            term_factors=[(5, 0)] * len(terms),
        )
