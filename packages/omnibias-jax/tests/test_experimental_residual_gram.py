# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Gram matrices and Gauss-Newton steps of deep-MLP PDE residuals through the stream identity vs nested autodiff."""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from omnibias.jax.experimental import residual_gram as rg  # noqa: E402
from omnibias.jax.experimental.residual_gram import (  # noqa: E402
    deep_residual_gauss_newton,
    deep_residual_gram,
    deep_residual_ntk_diagonal,
    masked_residual,
    pad_block,
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


def _reference(blocks, layers, w, damping=None):
    """r, K = J J^T and optionally J^T (K + damping I)^{-1} r from jax.hessian / jacrev per row."""
    params = [(W, b) for W, b, _ in layers]
    rs, Js = [], []
    for x, res, order in blocks:

        def row(p, z, res=res, order=order):
            u = _net(p, layers, z)
            g = jax.jacfwd(lambda q: _net(p, layers, q))(z).T
            lap = jnp.einsum("ckk,k->c", jax.hessian(lambda q: _net(p, layers, q))(z), w)
            out = res(
                u[None], g[None] if order >= 1 else None, lap[None] if order == 2 else None, z[None]
            )
            return jnp.reshape(out, (-1,))

        r = jax.vmap(row, in_axes=(None, 0))(params, x)
        Jt = jax.vmap(jax.jacrev(row), in_axes=(None, 0))(params, x)
        rs.append(r.reshape(-1))
        Js.append(jax.tree_util.tree_map(lambda v: v.reshape((-1,) + v.shape[2:]), Jt))
    r = jnp.concatenate(rs)
    J = jax.tree_util.tree_map(lambda *v: jnp.concatenate(v), *Js)
    flat = jnp.concatenate([v.reshape(r.size, -1) for v in jax.tree_util.tree_leaves(J)], axis=1)
    K = flat @ flat.T
    if damping is None:
        return r, K
    alpha = jnp.linalg.solve(K + damping * jnp.eye(r.size), r)
    return r, K, jax.tree_util.tree_map(lambda v: jnp.tensordot(alpha, v, axes=1), J)


def _weights(D, coordinates, diffusion):
    w = np.zeros(D)
    idx = np.arange(D) if coordinates is None else np.asarray(coordinates)
    w[idx] = 1.0 if diffusion is None else np.asarray(diffusion) ** 2
    return jnp.asarray(w)


def _close(a, b, tol=1e-12):
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape
    np.testing.assert_allclose(a, b, rtol=tol, atol=tol * np.max(np.abs(b)))


def _poisson(u, g, lap, x):
    return lap[:, 0] - jnp.cos(x[:, 1])


def _hjb(u, g, lap, x):
    return lap[:, 0] + 0.5 * jnp.sum(g[..., 0] ** 2, axis=-1) + u[:, 0] ** 2 - jnp.sin(x[:, 0])


def _boundary(u, g, lap, x):
    return u[:, 0] - x[:, 0] * x[:, 1]


def _advection(u, g, lap, x):
    return g[:, 0, 0] + 0.3 * g[:, 1, 0] - u[:, 0]


def _system(u, g, lap, x):
    return jnp.stack([lap[:, 0] - u[:, 1] * g[:, 0, 0], lap[:, 1] + u[:, 0] ** 3 - x[:, 2]], axis=1)


def _heat(u, g, lap, x):
    return g[:, 0, 0] - lap[:, 0]


def _navier_stokes(u, g, lap, x, nu=0.05):
    """Steady incompressible Navier-Stokes rows (Kovasznay form) of the vector field ``(u, v, p)`` in two inputs."""
    ux, uy, vx, vy, px, py = g[:, 0, 0], g[:, 1, 0], g[:, 0, 1], g[:, 1, 1], g[:, 0, 2], g[:, 1, 2]
    return jnp.stack(
        [
            u[:, 0] * ux + u[:, 1] * uy + px - nu * lap[:, 0],
            u[:, 0] * vx + u[:, 1] * vy + py - nu * lap[:, 1],
            ux + vy,
        ],
        axis=1,
    )


def _case(name):
    rng = np.random.default_rng({"hjb": 0, "blocks": 1, "system": 2, "heat": 3, "ns": 4}[name])
    if name == "ns":
        layers = _layers(rng, [2, 8, 8, 3], ["tanh", "tanh", None])
        x = jnp.asarray(rng.standard_normal((5, 2)))
        inflow = jnp.asarray(rng.standard_normal((3, 2)))
        return (
            [(x, _navier_stokes, 2), (inflow, lambda u, g, lap, z: u - 0.1, 0)],
            layers,
            None,
            None,
        )
    if name == "hjb":
        layers = _layers(rng, [2, 6, 6, 6, 1], ["tanh", "tanh", "tanh", None])
        return [(jnp.asarray(rng.standard_normal((5, 2))), _hjb, 2)], layers, None, None
    if name == "blocks":
        layers = _layers(rng, [2, 6, 6, 6, 1], ["tanh", "tanh", "tanh", None])
        xb = jnp.asarray(rng.standard_normal((3, 2)))
        x = jnp.asarray(rng.standard_normal((5, 2)))
        return [(x, _poisson), (xb, _boundary, 0), (xb + 1.0, _advection, 1)], layers, None, None
    if name == "system":
        layers = _layers(rng, [3, 5, 7, 2], ["sin", None, None], no_bias=(1,))
        x = jnp.asarray(rng.standard_normal((4, 3)))
        return [(x, _system, 2), (x[:2], lambda u, g, lap, z: u - z[:, :2], 0)], layers, None, None
    layers = _layers(rng, [3, 8, 8, 1], ["tanh", "tanh", None])
    x = jnp.asarray(rng.standard_normal((6, 3)))
    return [(x, _heat, 2)], layers, [1, 2], jnp.asarray([0.7, 1.3])


def _normalized(blocks):
    return [(b[0], b[1], b[2] if len(b) > 2 else 2) for b in blocks]


@pytest.mark.parametrize("name", ["hjb", "blocks", "system", "heat", "ns"])
@pytest.mark.parametrize("strategy", ["auto", "streams", "explicit"])
def test_gram_matches_nested_autodiff(name, strategy):
    blocks, layers, coords, diff = _case(name)
    D = blocks[0][0].shape[1]
    r_ref, K_ref = _reference(_normalized(blocks), layers, _weights(D, coords, diff))
    r, K = deep_residual_gram(blocks, layers, coordinates=coords, diffusion=diff, strategy=strategy)
    _close(r, r_ref)
    _close(K, K_ref)
    _close(K, K.T, 1e-14)
    assert float(jnp.min(jnp.linalg.eigvalsh(K))) > -1e-10 * float(jnp.max(jnp.abs(K)))


@pytest.mark.parametrize("name", ["blocks", "system", "heat", "ns"])
def test_gauss_newton_step_matches_nested_autodiff(name):
    blocks, layers, coords, diff = _case(name)
    D = blocks[0][0].shape[1]
    damping = 1e-3
    r_ref, K_ref, step_ref = _reference(
        _normalized(blocks), layers, _weights(D, coords, diff), damping
    )
    r, K, step = deep_residual_gauss_newton(
        blocks, layers, damping, coordinates=coords, diffusion=diff
    )
    _close(r, r_ref)
    _close(K, K_ref)
    assert len(step) == len(layers)
    for (dW, db), (rW, rb), (_, b, _) in zip(step, step_ref, layers, strict=True):
        _close(dW, rW, 1e-10)
        if b is None:
            assert db is None
        else:
            _close(db, rb, 1e-10)


def test_jit_and_strategies_agree():
    blocks, layers, _, _ = _case("blocks")
    specs = [spec for _, _, spec in layers]

    def run(weights, strategy):
        ly = [(W, b, s) for (W, b), s in zip(weights, specs, strict=True)]
        return deep_residual_gram(blocks, ly, strategy=strategy)

    weights = [(W, b) for W, b, _ in layers]
    ref = run(weights, "explicit")
    for strategy in ("auto", "streams"):
        out = jax.jit(run, static_argnums=1)(weights, strategy)
        _close(out[0], ref[0])
        _close(out[1], ref[1])


def test_auto_rule_uses_identity_only_for_wide_layers(monkeypatch):
    calls = []
    original = rg._streams_gram

    def counting(*args):
        calls.append(args[0].shape)
        return original(*args)

    monkeypatch.setattr(rg, "_streams_gram", counting)
    rng = np.random.default_rng(5)
    x = jnp.asarray(rng.standard_normal((4, 1)))
    wide = _layers(rng, [1, 40, 40, 1], ["tanh", "tanh", None])
    deep_residual_gram([(x, lambda u, g, lap, z: lap[:, 0])], wide)
    assert len(calls) == 1
    calls.clear()
    narrow = _layers(rng, [1, 8, 8, 1], ["tanh", "tanh", None])
    deep_residual_gram([(x, lambda u, g, lap, z: lap[:, 0])], narrow)
    assert not calls


def test_piecewise_activation_guard():
    rng = np.random.default_rng(7)
    layers = _layers(rng, [2, 6, 1], ["relu", None])
    x = jnp.asarray(rng.standard_normal((4, 2)))
    with pytest.raises(ValueError, match="relu"):
        deep_residual_gram([(x, _poisson)], layers)
    _, K = deep_residual_gram([(x, _poisson)], layers, allow_almost_everywhere=True)
    assert bool(jnp.all(jnp.isfinite(K)))
    _, K0 = deep_residual_gram([(x, _boundary, 0)], layers)
    r_ref, K_ref = _reference([(x, _boundary, 0)], layers, jnp.ones(2))
    _close(K0, K_ref)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"strategy": "fast"},
        {"coordinates": [0, 0]},
        {"coordinates": [2]},
        {"coordinates": []},
        {"diffusion": jnp.ones(3)},
    ],
)
def test_argument_refusals(kwargs):
    blocks, layers, _, _ = _case("hjb")
    with pytest.raises(ValueError):
        deep_residual_gram(blocks, layers, **kwargs)


def test_shape_refusals():
    blocks, layers, _, _ = _case("hjb")
    x = blocks[0][0]
    with pytest.raises(ValueError, match="order"):
        deep_residual_gram([(x, _hjb, 3)], layers)
    with pytest.raises(ValueError, match="shape"):
        deep_residual_gram([(x, lambda u, g, lap, z: lap[:2, 0])], layers)
    with pytest.raises(ValueError, match="non-empty"):
        deep_residual_gram([], layers)
    with pytest.raises(ValueError, match="non-empty"):
        deep_residual_gram(blocks, [])
    with pytest.raises(ValueError, match="dimension"):
        deep_residual_gram([(x, _hjb), (jnp.ones((2, 3)), _hjb)], layers)
    with pytest.raises(ValueError, match="first layer"):
        deep_residual_gram([(jnp.ones((2, 3)), _hjb)], layers)


@pytest.mark.parametrize("dim", [4, 5, 7])
@pytest.mark.parametrize("pairing", ["rows", "cyclic", "auto"])
def test_cyclic_stream_pairing_is_exact(monkeypatch, dim, pairing):
    """Odd and even stream counts (``T = dim + 2``) through every pairing rule, identity forced on each layer."""
    monkeypatch.setattr(rg, "_PAIRING", pairing)
    rng = np.random.default_rng(20 + dim)
    layers = _layers(rng, [dim, 9, 9, 1], ["tanh", "tanh", None])
    x = jnp.asarray(rng.standard_normal((4, dim)))
    blocks = [(x, _hjb, 2), (x[:2] + 0.5, _boundary, 0)]
    r_ref, K_ref = _reference(blocks, layers, jnp.ones(dim))
    r, K = deep_residual_gram(blocks, layers, strategy="streams")
    _close(r, r_ref)
    _close(K, K_ref)


def test_pairing_defaults_to_rows_and_auto_uses_cyclic_only_for_long_stream_sets(monkeypatch):
    calls = []
    original = rg._cyclic_gram

    def counting(*args):
        calls.append(args[1].shape[2])
        return original(*args)

    monkeypatch.setattr(rg, "_cyclic_gram", counting)
    assert rg._PAIRING == "rows"
    rng = np.random.default_rng(31)
    for pairing in ("rows", "auto"):
        monkeypatch.setattr(rg, "_PAIRING", pairing)
        for dim in (2, 4):
            layers = _layers(rng, [dim, 8, 8, 1], ["tanh", "tanh", None])
            x = jnp.asarray(rng.standard_normal((3, dim)))
            deep_residual_gram([(x, lambda u, g, lap, z: lap[:, 0])], layers, strategy="streams")
        if pairing == "rows":
            assert not calls
    assert calls and set(calls) == {6}


def test_rule_constant_hook_chooses_identity_or_explicit_rows(monkeypatch):
    """``cost(i, j)`` replaces the rule constant per block pair; both choices give the same Gram matrix."""
    original = rg._assemble
    streams, rows = [], []
    monkeypatch.setattr(
        rg, "_streams_gram", lambda *a, _o=rg._streams_gram: streams.append(1) or _o(*a)
    )
    monkeypatch.setattr(
        rg, "_explicit_jacobian", lambda *a, _o=rg._explicit_jacobian: rows.append(1) or _o(*a)
    )
    rng = np.random.default_rng(37)
    layers = _layers(rng, [3, 8, 8, 1], ["tanh", "tanh", None])
    x = jnp.asarray(rng.standard_normal((4, 3)))
    blocks = [(x, lambda u, g, lap, z: lap[:, 0])]
    out = {}
    for name, c in (("identity", 0.0), ("rows", 1e9)):
        streams.clear()
        rows.clear()
        monkeypatch.setattr(rg, "_assemble", lambda *a, _c=c: original(*a, cost=lambda i, j: _c))
        out[name] = deep_residual_gram(blocks, layers)[1]
        assert (len(streams) > 0, len(rows) > 0) == (name == "identity", name == "rows")
    _close(out["identity"], out["rows"])


@pytest.mark.parametrize("name", ["hjb", "blocks", "system", "heat"])
@pytest.mark.parametrize("strategy", ["auto", "streams", "explicit"])
def test_ntk_diagonal_matches_gram_diagonal(name, strategy):
    blocks, layers, coords, diff = _case(name)
    D = blocks[0][0].shape[1]
    r_ref, K_ref = _reference(_normalized(blocks), layers, _weights(D, coords, diff))
    r, d = deep_residual_ntk_diagonal(
        blocks, layers, coordinates=coords, diffusion=diff, strategy=strategy
    )
    _close(r, r_ref)
    _close(d, jnp.diag(K_ref))
    assert bool(jnp.all(d >= 0))


def test_ntk_diagonal_chunked_rows_and_jit(monkeypatch):
    """A small chunk forces ``lax.map`` over padded row chunks, including a ragged last chunk."""
    blocks, layers, _, _ = _case("blocks")
    _, K_ref = _reference(_normalized(blocks), layers, jnp.ones(2))
    monkeypatch.setattr(rg, "_DIAG_CHUNK", 1)
    specs = [spec for _, _, spec in layers]

    def run(weights, strategy):
        ly = [(W, b, s) for (W, b), s in zip(weights, specs, strict=True)]
        return deep_residual_ntk_diagonal(blocks, ly, strategy=strategy)

    weights = [(W, b) for W, b, _ in layers]
    for strategy in ("auto", "explicit"):
        _, d = jax.jit(run, static_argnums=1)(weights, strategy)
        _close(d, jnp.diag(K_ref))


def test_ntk_diagonal_rule_constant_does_not_change_values(monkeypatch):
    rng = np.random.default_rng(37)
    x = jnp.asarray(rng.standard_normal((4, 1)))
    layers = _layers(rng, [1, 48, 48, 1], ["tanh", "tanh", None])
    blocks = [(x, lambda u, g, lap, z: lap[:, 0])]
    _, d_auto = deep_residual_ntk_diagonal(blocks, layers)
    monkeypatch.setattr(rg, "_DIAG_COST", 1e6)
    _, d_rows = deep_residual_ntk_diagonal(blocks, layers)
    monkeypatch.setattr(rg, "_DIAG_COST", 1e-6)
    _, d_streams = deep_residual_ntk_diagonal(blocks, layers)
    _close(d_auto, d_rows)
    _close(d_auto, d_streams)


def test_pad_block_is_exact_and_reuses_one_program():
    blocks, layers, _, _ = _case("blocks")
    x = blocks[0][0]
    f = jnp.asarray(np.random.default_rng(41).standard_normal(x.shape[0]))

    def residual(fv):
        return lambda u, g, lap, z: lap[:, 0] - fv

    r0, K0 = deep_residual_gram([(x, residual(f))], layers)
    xp, mask, fp = pad_block(x, f, bucket=8)
    assert xp.shape[0] == 8 and fp.shape == (8,) and float(mask.sum()) == x.shape[0]
    r1, K1 = deep_residual_gram([(xp, masked_residual(residual(fp), mask))], layers)
    n = x.shape[0]
    _close(r1[:n], r0, 1e-14)
    _close(K1[:n, :n], K0, 1e-14)
    assert float(jnp.max(jnp.abs(r1[n:]))) == 0.0
    assert float(jnp.max(jnp.abs(K1[n:]))) == 0.0
    assert float(jnp.max(jnp.abs(K1[:, n:]))) == 0.0
    _, _, s0 = deep_residual_gauss_newton([(x, residual(f))], layers, 1e-3)
    _, _, s1 = deep_residual_gauss_newton([(xp, masked_residual(residual(fp), mask))], layers, 1e-3)
    for (a, ab), (b, bb) in zip(s0, s1, strict=True):
        _close(b, a, 1e-12)
        _close(bb, ab, 1e-12)

    traces = []

    @jax.jit
    def program(xs, ms, fs):
        traces.append(1)
        return deep_residual_gram([(xs, masked_residual(residual(fs), ms))], layers)[1]

    for m in (3, 4, 5):
        xq, mq, fq = pad_block(x[:m], f[:m], bucket=8)
        Kq = program(xq, mq, fq)
        _, Kref = deep_residual_gram([(x[:m], residual(f[:m]))], layers)
        _close(Kq[:m, :m], Kref, 1e-14)
    assert len(traces) == 1


def test_pad_block_refusals():
    x = jnp.ones((3, 2))
    with pytest.raises(ValueError, match="bucket"):
        pad_block(x, bucket=0)
    with pytest.raises(ValueError, match="leading axis"):
        pad_block(x, jnp.ones(4), bucket=4)
    with pytest.raises(ValueError, match="at least one"):
        pad_block(jnp.ones((0, 2)), bucket=4)
    xp, mask = pad_block(x, bucket=3)
    assert xp.shape == (3, 2) and float(mask.sum()) == 3.0
