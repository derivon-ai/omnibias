# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Projected residual Gram matrices (first-layer identity, D > width, fourth-order blocks) vs nested autodiff."""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from omnibias.jax.experimental import projected_gram as prg  # noqa: E402
from omnibias.jax.experimental.projected_gram import (  # noqa: E402
    projected_residual_gauss_newton,
    projected_residual_gram,
    projected_residual_ntk_diagonal,
)

_ACT = {"tanh": jnp.tanh, "sin": jnp.sin, "relu": jax.nn.relu, None: lambda z: z}


def _layers(rng, dims, acts, no_bias=()):
    out = []
    for i in range(len(dims) - 1):
        W = jnp.asarray(rng.standard_normal((dims[i + 1], dims[i])) / np.sqrt(dims[i]))
        b = None if i in no_bias else jnp.asarray(0.2 * rng.standard_normal(dims[i + 1]))
        out.append((W, b, acts[i]))
    return out


def _net(params, layers, z):
    a = z
    for (W, b), (_, _, act) in zip(params, layers, strict=True):
        a = _ACT[act](W @ a + (0.0 if b is None else b))
    return a


def _metric(D, coordinates, diffusion):
    w = np.zeros(D)
    idx = np.arange(D) if coordinates is None else np.asarray(coordinates)
    w[idx] = 1.0 if diffusion is None else np.asarray(diffusion) ** 2
    return jnp.asarray(w)


def _reference(blocks, layers, coordinates, diffusion, directions, damping=None, momentum=0.0, prev=None):
    """``r``, ``K = J J^T`` (and the dual step) from per-row nested autodiff of the unprojected network.

    With ``prev`` (a per-layer ``[(dW, db)]``) and ``momentum`` the step is the SPRING update
    ``J^T (K + damping I)^{-1} (r - mu J prev) + mu prev``.
    """
    params = [(W, b) for W, b, _ in layers]
    D = blocks[0][0].shape[1]
    w = _metric(D, coordinates, diffusion)
    V = jnp.zeros((D, 0)) if directions is None else jnp.asarray(directions)
    rs, Js = [], []
    for x, res, order in blocks:

        def row(p, z, res=res, order=order):
            def fn(q):
                return _net(p, layers, q)

            def lap_of(q):
                return jnp.einsum("cdd,d->c", jax.hessian(fn)(q), w)

            u = fn(z)
            g = jax.jacfwd(fn)(z)
            gs = (g * g) @ w
            dv = (g @ V).T
            args = {
                0: (u[None], None, None, None, z[None]),
                1: (u[None], gs[None], None, dv[None] if directions is not None else None, z[None]),
                2: (
                    u[None],
                    gs[None],
                    lap_of(z)[None],
                    dv[None] if directions is not None else None,
                    z[None],
                ),
                4: (
                    u[None],
                    gs[None],
                    lap_of(z)[None],
                    jnp.einsum("cdd,d->c", jax.hessian(lap_of)(z), w)[None],
                    dv[None] if directions is not None else None,
                    z[None],
                ),
            }[order]
            return jnp.reshape(res(*args), (-1,))

        rs.append(jax.vmap(row, in_axes=(None, 0))(params, x).reshape(-1))
        Jt = jax.vmap(jax.jacrev(row), in_axes=(None, 0))(params, x)
        Js.append(jax.tree_util.tree_map(lambda v: v.reshape((-1,) + v.shape[2:]), Jt))
    r = jnp.concatenate(rs)
    J = jax.tree_util.tree_map(lambda *v: jnp.concatenate(v), *Js)
    flat = jnp.concatenate([v.reshape(r.size, -1) for v in jax.tree_util.tree_leaves(J)], axis=1)
    K = flat @ flat.T
    if damping is None:
        return r, K
    rhs = r
    if prev is not None:  # ``None`` biases are empty pytrees, so the leaves of J and of prev line up
        pairs = zip(jax.tree_util.tree_leaves(J), jax.tree_util.tree_leaves(list(prev)), strict=True)
        rhs = r - momentum * sum(jnp.tensordot(j.reshape(r.size, -1), p.reshape(-1), axes=1) for j, p in pairs)
    alpha = jnp.linalg.solve(K + damping * jnp.eye(r.size), rhs)
    step = jax.tree_util.tree_map(lambda v: jnp.tensordot(alpha, v, axes=1), J)
    if prev is not None:
        step = jax.tree_util.tree_map(lambda s, p: s + momentum * p, step, list(prev))
    return r, K, step


def _close(a, b, tol=1e-11):
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape
    np.testing.assert_allclose(a, b, rtol=tol, atol=tol * np.max(np.abs(b)))


def _heat(u, gs, lap, dv, x):
    return dv[:, 0, 0] - lap[:, 0] + u[:, 0] ** 3 - jnp.sin(x[:, 1])


def _hjb(u, gs, lap, dv, x):
    return lap[:, 0] + 0.5 * gs[:, 0] + u[:, 0] ** 2 - jnp.sin(x[:, 0])


def _first_order(u, gs, lap, dv, x):
    return gs[:, 0] - u[:, 0] * x[:, 0]


def _boundary(u, gs, lap, dv, x):
    return u[:, 0] - x[:, 0] * x[:, 1]


def _ks(u, gs, lap, bilap, dv, x):
    out = bilap[:, 0] + lap[:, 0] + 0.5 * gs[:, 0] - jnp.cos(x[:, 1])
    return out if dv is None else out + dv[:, 0, 0]


def _ch(u, gs, lap, bilap, dv, x):
    return (
        -(3.0 * u[:, 0] ** 2 - 1.0) * lap[:, 0]
        - 6.0 * u[:, 0] * gs[:, 0]
        + bilap[:, 0]
        + (0.0 if dv is None else dv[:, 0, 0])
    )


def _system(u, gs, lap, dv, x):
    return jnp.stack([lap[:, 0] - u[:, 1] * gs[:, 1], gs[:, 0] + u[:, 0] ** 3 - x[:, 0]], axis=1)


def _case(name):
    rng = np.random.default_rng(
        {"wide": 0, "heat": 1, "mixed": 2, "ks": 3, "ks_proj": 4, "system": 5, "rank": 6}[name]
    )
    if name == "wide":  # D > H, all coordinates, no drift
        layers = _layers(rng, [7, 4, 4, 1], ["tanh", "tanh", None])
        x = jnp.asarray(rng.standard_normal((4, 7)))
        return [(x, _hjb, 2), (x[:2] + 0.3, _boundary, 0)], layers, None, None, None
    if name == "heat":  # D > H, time drift, space Laplacian, anisotropic diffusion
        layers = _layers(rng, [8, 5, 5, 1], ["tanh", "tanh", None])
        x = jnp.asarray(rng.standard_normal((4, 8)))
        V = np.zeros((8, 1))
        V[0, 0] = 1.0
        return (
            [(x, _heat, 2), (x[:2] + 0.3, _boundary, 0)],
            layers,
            list(range(1, 8)),
            jnp.asarray(rng.uniform(0.5, 1.5, 7)),
            V,
        )
    if name == "mixed":  # D <= H, all orders 0/1/2 with a constant drift in two directions
        layers = _layers(rng, [3, 7, 6, 1], ["tanh", "sin", None], no_bias=(1,))
        x = jnp.asarray(rng.standard_normal((4, 3)))
        V = np.eye(3)[:, :2]
        return (
            [(x, _hjb, 2), (x[:2], _first_order, 1), (x[:3] - 0.2, _boundary, 0)],
            layers,
            None,
            None,
            V,
        )
    if name == "ks":  # fourth order, D <= H, time drift
        layers = _layers(rng, [3, 8, 8, 1], ["tanh", "tanh", None])
        x = jnp.asarray(rng.standard_normal((3, 3)))
        V = np.zeros((3, 1))
        V[0, 0] = 1.0
        return [(x, _ks, 4), (x[:2] + 0.1, _boundary, 0)], layers, [1, 2], None, V
    if name == "ks_proj":  # fourth order, m > H (projected first layer), no drift, diffusion
        layers = _layers(rng, [6, 3, 5, 1], ["tanh", "tanh", None], no_bias=(0,))
        x = jnp.asarray(rng.standard_normal((3, 6)))
        return (
            [(x, _ch, 4), (x[:2] - 0.1, _hjb, 2)],
            layers,
            None,
            jnp.asarray(rng.uniform(0.6, 1.4, 6)),
            None,
        )
    if name == "system":  # two outputs, two residual components
        layers = _layers(rng, [5, 4, 6, 2], ["tanh", "tanh", None])
        x = jnp.asarray(rng.standard_normal((3, 5)))
        return (
            [(x, _system, 2), (x[:2], lambda u, gs, lap, dv, z: u - z[:, :2], 0)],
            layers,
            None,
            None,
            None,
        )
    # rank-deficient: W_1 has rank 2 below min(m, H), so the projection basis is not unique
    layers = _layers(rng, [6, 4, 4, 1], ["tanh", "tanh", None])
    W = np.asarray(rng.standard_normal((4, 2)) @ rng.standard_normal((2, 6)))
    layers[0] = (jnp.asarray(W), layers[0][1], "tanh")
    x = jnp.asarray(rng.standard_normal((4, 6)))
    return [(x, _hjb, 2)], layers, None, None, None


CASES = ["wide", "heat", "mixed", "ks", "ks_proj", "system", "rank"]


@pytest.mark.parametrize("name", CASES)
@pytest.mark.parametrize("strategy", ["auto", "streams", "explicit"])
def test_gram_matches_nested_autodiff(name, strategy):
    blocks, layers, coords, diff, V = _case(name)
    r_ref, K_ref = _reference(blocks, layers, coords, diff, V)
    r, K = projected_residual_gram(
        blocks, layers, coordinates=coords, diffusion=diff, directions=V, strategy=strategy
    )
    _close(r, r_ref)
    _close(K, K_ref)
    _close(K, K.T, 1e-14)
    assert float(jnp.min(jnp.linalg.eigvalsh(K))) > -1e-9 * float(jnp.max(jnp.abs(K)))


@pytest.mark.parametrize("name", ["heat", "ks", "ks_proj", "system"])
def test_ntk_diagonal_matches_gram_diagonal(name):
    blocks, layers, coords, diff, V = _case(name)
    kw = {"coordinates": coords, "diffusion": diff, "directions": V}
    r, K = projected_residual_gram(blocks, layers, **kw)
    r2, d = projected_residual_ntk_diagonal(blocks, layers, **kw)
    _close(r2, r, 1e-14)
    _close(d, jnp.diag(K), 1e-12)


@pytest.mark.parametrize("name", ["heat", "ks_proj", "mixed"])
def test_gauss_newton_step_matches_nested_autodiff(name):
    blocks, layers, coords, diff, V = _case(name)
    damping = 1e-3
    r_ref, K_ref, step_ref = _reference(blocks, layers, coords, diff, V, damping)
    r, K, step = projected_residual_gauss_newton(
        blocks, layers, damping, coordinates=coords, diffusion=diff, directions=V
    )
    _close(r, r_ref)
    _close(K, K_ref)
    assert len(step) == len(layers)
    for (dW, db), (rW, rb), (_, b, _) in zip(step, step_ref, layers, strict=True):
        _close(dW, rW, 1e-9)
        if b is None:
            assert db is None
        else:
            _close(db, rb, 1e-9)


def _previous_step(layers, seed=0):
    rng = np.random.default_rng(100 + seed)
    return [
        (jnp.asarray(0.3 * rng.standard_normal(W.shape)), None if b is None else jnp.asarray(0.3 * rng.standard_normal(b.shape)))
        for W, b, _ in layers
    ]


@pytest.mark.parametrize("name", ["heat", "ks_proj", "mixed", "ks", "system"])
def test_spring_step_matches_nested_autodiff(name):
    """SPRING ``J^T (K + lam I)^{-1} (r - mu J d) + mu d`` against the per-row nested-autodiff Jacobian."""
    blocks, layers, coords, diff, V = _case(name)
    damping, mu = 1e-3, 0.7
    prev = _previous_step(layers)
    r_ref, _, step_ref = _reference(blocks, layers, coords, diff, V, damping, mu, prev)
    r, _, step = projected_residual_gauss_newton(
        blocks, layers, damping, coordinates=coords, diffusion=diff, directions=V, momentum=mu, prev_delta=prev
    )
    _close(r, r_ref)
    for (dW, db), (rW, rb), (_, b, _) in zip(step, step_ref, layers, strict=True):
        _close(dW, rW, 1e-9)
        if b is None:
            assert db is None
        else:
            _close(db, rb, 1e-9)


def test_spring_with_zero_momentum_or_zero_history_is_gauss_newton():
    blocks, layers, coords, diff, V = _case("heat")
    kw = {"coordinates": coords, "diffusion": diff, "directions": V}
    _, _, plain = projected_residual_gauss_newton(blocks, layers, 1e-3, **kw)
    zero = [(jnp.zeros_like(W), None if b is None else jnp.zeros_like(b)) for W, b, _ in layers]
    for mu, prev in ((0.0, _previous_step(layers)), (0.9, zero)):
        _, _, step = projected_residual_gauss_newton(blocks, layers, 1e-3, momentum=mu, prev_delta=prev, **kw)
        for (dW, db), (pW, pb) in zip(step, plain, strict=True):
            _close(dW, pW, 1e-12)
            _close(db, pb, 1e-12)


def test_spring_vector_damping_and_jit():
    blocks, layers, coords, diff, V = _case("heat")
    kw = {"coordinates": coords, "diffusion": diff, "directions": V, "momentum": 0.5}
    prev = _previous_step(layers)
    lams = jnp.asarray([1e-2, 1e-4])
    _, _, many = projected_residual_gauss_newton(blocks, layers, lams, prev_delta=prev, **kw)
    for i, lam in enumerate([1e-2, 1e-4]):
        _, _, one = projected_residual_gauss_newton(blocks, layers, lam, prev_delta=prev, **kw)
        for (dW, db), (sW, sb) in zip(many, one, strict=True):
            _close(dW[i], sW, 1e-9)
            _close(db[i], sb, 1e-9)
    specs = [spec for _, _, spec in layers]

    def run(weights, history):
        ly = [(W, b, s) for (W, b), s in zip(weights, specs, strict=True)]
        return projected_residual_gauss_newton(blocks, ly, 1e-3, prev_delta=history, **kw)[2]

    weights = [(W, b) for W, b, _ in layers]
    eager, jitted = run(weights, prev), jax.jit(run)(weights, prev)
    for (a, ab), (c, cb) in zip(eager, jitted, strict=True):
        _close(c, a, 1e-12)
        _close(cb, ab, 1e-12)


def test_spring_argument_refusals():
    blocks, layers, _, _, _ = _case("wide")
    with pytest.raises(ValueError, match="momentum needs prev_delta"):
        projected_residual_gauss_newton(blocks, layers, 1e-3, momentum=0.5)
    with pytest.raises(ValueError, match="one .dW, db. per layer"):
        projected_residual_gauss_newton(blocks, layers, 1e-3, momentum=0.5, prev_delta=_previous_step(layers)[:1])


def test_vector_damping_matches_scalar_steps():
    blocks, layers, coords, diff, V = _case("heat")
    kw = {"coordinates": coords, "diffusion": diff, "directions": V}
    lams = jnp.asarray([1e-2, 1e-4])
    _, _, many = projected_residual_gauss_newton(blocks, layers, lams, **kw)
    for i, lam in enumerate([1e-2, 1e-4]):
        _, _, one = projected_residual_gauss_newton(blocks, layers, lam, **kw)
        for (dW, db), (sW, sb), (_, b, _) in zip(many, one, layers, strict=True):
            _close(dW[i], sW, 1e-9)
            if b is not None:
                _close(db[i], sb, 1e-9)


def test_explicit_first_layer_rows_agree(monkeypatch):
    blocks, layers, coords, diff, V = _case("heat")
    kw = {"coordinates": coords, "diffusion": diff, "directions": V}
    _, K = projected_residual_gram(blocks, layers, **kw)
    monkeypatch.setattr(prg, "_EXPLICIT_FIRST", True)
    _, K_rows = projected_residual_gram(blocks, layers, **kw)
    _close(K, K_rows, 1e-13)


def test_jit_over_weights():
    blocks, layers, coords, diff, V = _case("ks_proj")
    specs = [spec for _, _, spec in layers]

    def run(weights):
        ly = [(W, b, s) for (W, b), s in zip(weights, specs, strict=True)]
        return projected_residual_gram(blocks, ly, coordinates=coords, diffusion=diff, directions=V)

    weights = [(W, b) for W, b, _ in layers]
    r, K = run(weights)
    rj, Kj = jax.jit(run)(weights)
    _close(rj, r, 1e-13)
    _close(Kj, K, 1e-13)


def test_padded_rows_are_exactly_zero():
    from omnibias.jax.experimental.residual_gram import masked_residual, pad_block

    blocks, layers, coords, diff, V = _case("ks")
    x, res, _ = blocks[0]
    xp, mask = pad_block(x, bucket=8)
    kw = {"coordinates": coords, "diffusion": diff, "directions": V}
    _, K0 = projected_residual_gram([(x, res, 4)], layers, **kw)
    _, K1 = projected_residual_gram([(xp, masked_residual(res, mask), 4)], layers, **kw)
    n = x.shape[0]
    _close(K1[:n, :n], K0, 1e-13)
    assert float(jnp.max(jnp.abs(K1[n:]))) == 0.0


def test_piecewise_activation_guards():
    rng = np.random.default_rng(9)
    layers = _layers(rng, [5, 3, 1], ["relu", None])
    x = jnp.asarray(rng.standard_normal((3, 5)))
    with pytest.raises(ValueError, match="relu"):
        projected_residual_gram([(x, _hjb, 2)], layers)
    with pytest.raises(ValueError, match="relu"):
        projected_residual_gram([(x, _ks, 4)], layers)
    _, K = projected_residual_gram([(x, _hjb, 2)], layers, allow_almost_everywhere=True)
    assert bool(jnp.all(jnp.isfinite(K)))
    _, K0 = projected_residual_gram([(x, _boundary, 0)], layers)
    assert bool(jnp.all(jnp.isfinite(K0)))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"strategy": "fast"},
        {"coordinates": [0, 0]},
        {"coordinates": [7]},
        {"diffusion": jnp.ones(3)},
        {"directions": jnp.ones((3, 1))},
    ],
)
def test_argument_refusals(kwargs):
    blocks, layers, _, _, _ = _case("wide")
    with pytest.raises(ValueError):
        projected_residual_gram(blocks, layers, **kwargs)


def test_order_refusal_names_the_supported_orders():
    blocks, layers, _, _, _ = _case("wide")
    with pytest.raises(ValueError, match="order must be 0, 1, 2 or 4"):
        projected_residual_gram([(blocks[0][0], _hjb, 3)], layers)
