# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""First-layer projected second-order operators of deep MLPs vs nested autodiff."""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from omnibias.jax.experimental.projected_laplacian import (  # noqa: E402
    input_factor,
    projected_deep_laplacian,
)

_ACT = {"tanh": jnp.tanh, "sin": jnp.sin, "sigmoid": jax.nn.sigmoid, "softplus": jax.nn.softplus}


def _layers(rng, dims, act="tanh", outputs=1, readout=True, affine_at=None):
    layers = []
    for i in range(len(dims) - 1):
        W = jnp.asarray(rng.standard_normal((dims[i + 1], dims[i])) / np.sqrt(dims[i]))
        b = jnp.asarray(0.1 * rng.standard_normal(dims[i + 1]))
        layers.append((W, b, None if i == affine_at else act))
    if readout:
        layers.append(
            (
                jnp.asarray(rng.standard_normal((outputs, dims[-1])) / np.sqrt(dims[-1])),
                jnp.asarray(0.1 * rng.standard_normal(outputs)),
                None,
            )
        )
    return layers


def _forward(layers, x):
    a = x
    for W, b, spec in layers:
        a = a @ W.T + (0.0 if b is None else b)
        if spec is not None:
            a = _ACT[spec](a)
    return a


def _reference(layers, X, S=None, V=None):
    dim = X.shape[-1]
    A = jnp.eye(dim) if S is None else S @ S.T
    S_ = jnp.eye(dim) if S is None else S

    def point(x, v):
        jac = jax.jacfwd(lambda y: _forward(layers, y))(x)  # (C, D)
        hess = jax.hessian(lambda y: _forward(layers, y))(x)  # (C, D, D)
        lap = jnp.einsum("cij,ij->c", hess, A)
        g2 = jnp.sum((jac @ S_) ** 2, axis=-1)
        d = None if v is None else (jac @ v).T
        return _forward(layers, x), g2, lap, d

    if V is None:
        u, g2, lap, _ = jax.vmap(lambda x: point(x, None))(X)
        return u, g2, lap, None
    if V.ndim == 2:
        return jax.vmap(lambda x: point(x, V))(X)
    return jax.vmap(point)(X, V)


def _split(layers):
    return [(W, b) for W, b, _ in layers], [spec for _, _, spec in layers]


def _join(params, specs):
    return [(W, b, spec) for (W, b), spec in zip(params, specs, strict=True)]


def _close(a, b, tol=1e-12):
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape
    np.testing.assert_allclose(a, b, rtol=0, atol=tol * (np.max(np.abs(b)) + 1e-300))


@pytest.mark.parametrize("act", ["tanh", "sin", "sigmoid", "softplus"])
@pytest.mark.parametrize(
    ("dim", "width", "depth"), [(3, 8, 1), (6, 8, 2), (8, 8, 3), (20, 6, 2), (40, 5, 3)]
)
def test_laplacian_and_gradient_norm_match_nested_autodiff(act, dim, width, depth):
    rng = np.random.default_rng(dim * 100 + width + depth)
    layers = _layers(rng, [dim] + [width] * depth, act)
    X = jnp.asarray(rng.standard_normal((5, dim)))
    u, g2, lap, d = projected_deep_laplacian(X, layers)
    ru, rg2, rlap, _ = _reference(layers, X)
    assert d is None
    _close(u, ru)
    _close(g2, rg2)
    _close(lap, rlap)


@pytest.mark.parametrize(("dim", "rank"), [(12, 3), (12, 12), (30, 9), (30, 30)])
def test_constant_diffusion_metric(dim, rank):
    rng = np.random.default_rng(dim + rank)
    layers = _layers(rng, [dim, 7, 7], "tanh", outputs=2)
    S = jnp.asarray(rng.standard_normal((dim, rank)) / np.sqrt(rank))
    X = jnp.asarray(rng.standard_normal((4, dim)))
    _, g2, lap, _ = projected_deep_laplacian(X, layers, diffusion=S)
    _, rg2, rlap, _ = _reference(layers, X, S=S)
    _close(g2, rg2)
    _close(lap, rlap)


@pytest.mark.parametrize("per_point", [False, True])
def test_drift_directions(per_point):
    rng = np.random.default_rng(7)
    dim = 25
    layers = _layers(rng, [dim, 6, 6, 6], "tanh", outputs=3)
    X = jnp.asarray(rng.standard_normal((4, dim)))
    V = jnp.asarray(rng.standard_normal((4, dim, 2) if per_point else (dim, 2)))
    u, g2, lap, d = projected_deep_laplacian(X, layers, directions=V)
    ru, rg2, rlap, rd = _reference(layers, X, V=V)
    assert d.shape == (4, 2, 3)
    _close(d, rd)
    _close(lap, rlap)
    _close(g2, rg2)


def test_rank_deficient_first_layer_values_and_gradients():
    rng = np.random.default_rng(11)
    dim = 30
    layers = _layers(rng, [dim, 6, 6], "tanh")
    W1, b1, s1 = layers[0]
    W1 = W1.at[-1].set(W1[0] + W1[1]).at[-2].set(2.0 * W1[2])
    layers[0] = (W1, b1, s1)
    X = jnp.asarray(rng.standard_normal((5, dim)))
    _, g2, lap, _ = projected_deep_laplacian(X, layers)
    _, rg2, rlap, _ = _reference(layers, X)
    _close(g2, rg2)
    _close(lap, rlap)

    params, specs = _split(layers)

    def loss(ps, fn):
        _, g, lp, _ = fn(_join(ps, specs))
        return jnp.mean((lp - g) ** 2)

    ga = jax.grad(lambda ps: loss(ps, lambda q: projected_deep_laplacian(X, q)))(params)
    gr = jax.grad(lambda ps: loss(ps, lambda q: _reference(q, X)))(params)
    for a, b in zip(jax.tree_util.tree_leaves(ga), jax.tree_util.tree_leaves(gr), strict=True):
        assert bool(jnp.all(jnp.isfinite(a)))
        _close(a, b, 1e-11)


def _conditioned_first_layer(rng, width, dim, cond):
    U, _ = np.linalg.qr(rng.standard_normal((width, width)))
    V, _ = np.linalg.qr(rng.standard_normal((dim, width)))
    return jnp.asarray((U * np.geomspace(1.0, 1.0 / cond, width)) @ V.T)


@pytest.mark.parametrize("cond", [1e1, 1e3, 1e6, 1e9])
def test_first_layer_conditioning_values_and_gradients(cond):
    rng = np.random.default_rng(int(np.log10(cond)))
    dim, width = 24, 6
    layers = _layers(rng, [dim, width, width], "tanh")
    layers[0] = (_conditioned_first_layer(rng, width, dim, cond), layers[0][1], "tanh")
    X = jnp.asarray(rng.standard_normal((4, dim)))
    F = input_factor(layers[0][0])
    _close(F @ F.T, layers[0][0] @ layers[0][0].T)
    params, specs = _split(layers)

    def loss(ps, fn):
        _, g, lp, _ = fn(_join(ps, specs))
        return jnp.mean((lp + g) ** 2)

    ga = jax.grad(lambda ps: loss(ps, lambda q: projected_deep_laplacian(X, q)))(params)
    gr = jax.grad(lambda ps: loss(ps, lambda q: _reference(q, X)))(params)
    for a, b in zip(jax.tree_util.tree_leaves(ga), jax.tree_util.tree_leaves(gr), strict=True):
        _close(a, b, 1e-11)


def test_vmapped_first_layers_with_failed_factorization_keep_finite_gradients():
    rng = np.random.default_rng(13)
    dim, width = 20, 5
    W_good = jnp.asarray(rng.standard_normal((width, dim)))
    W_bad = W_good.at[-1].set(W_good[0] - W_good[1])
    Ws = jnp.stack([W_good, W_bad])
    C = jnp.asarray(rng.standard_normal((width, width)))

    def metric_loss(W):
        F = input_factor(W)
        return jnp.sum(C * (F @ F.T))

    grads = jax.vmap(jax.grad(metric_loss))(Ws)
    for W, g in zip(Ws, grads, strict=True):
        assert bool(jnp.all(jnp.isfinite(g)))
        _close(g, (C + C.T) @ W)


@pytest.mark.parametrize(("dim", "width"), [(5, 8), (24, 6)])
def test_parameter_gradients_with_metric_and_drift(dim, width):
    rng = np.random.default_rng(dim * width)
    layers = _layers(rng, [dim, width, width], "tanh")
    S = jnp.asarray(rng.standard_normal((dim, dim)) / np.sqrt(dim))
    V = jnp.asarray(rng.standard_normal((dim, 1)))
    X = jnp.asarray(rng.standard_normal((6, dim)))
    f = jnp.asarray(rng.standard_normal((6, 1)))

    params, specs = _split(layers)

    def loss(ps, S, fn):
        _, g, lp, d = fn(_join(ps, specs), S)
        r = lp - 0.5 * g + d[:, 0] - f
        return 0.5 * jnp.mean(r * r)

    ours = jax.grad(loss, argnums=(0, 1))(
        params, S, lambda q, s: projected_deep_laplacian(X, q, diffusion=s, directions=V)
    )
    ref = jax.grad(loss, argnums=(0, 1))(params, S, lambda q, s: _reference(q, X, S=s, V=V))
    for a, b in zip(jax.tree_util.tree_leaves(ours), jax.tree_util.tree_leaves(ref), strict=True):
        _close(a, b, 1e-11)


@pytest.mark.parametrize(
    ("dim", "width", "rank", "cond"), [(14, 6, None, 1e0), (14, 6, None, 3e3), (20, 5, 9, 1e0)]
)
def test_hessian_vector_products_of_projected_losses(dim, width, rank, cond):
    rng = np.random.default_rng(dim + width)
    layers = _layers(rng, [dim, width, width], "tanh")
    W1, b1, s1 = layers[0]
    layers[0] = (_conditioned_first_layer(rng, width, dim, cond), b1, s1)
    S = None if rank is None else jnp.asarray(rng.standard_normal((dim, rank)) / np.sqrt(rank))
    X = jnp.asarray(rng.standard_normal((3, dim)) / np.sqrt(dim))
    V = -X[:, :, None]
    params, specs = _split(layers)
    tangent = jax.tree_util.tree_map(lambda p: jnp.asarray(rng.standard_normal(p.shape)), params)

    def loss(ps, fn):
        _, g, lp, d = fn(_join(ps, specs))
        return jnp.sum((lp - g + d[:, 0]) ** 2)

    def ours(ps):
        return loss(ps, lambda q: projected_deep_laplacian(X, q, diffusion=S, directions=V))

    def ref(ps):
        return loss(ps, lambda q: _reference(q, X, S=S, V=V))

    hvp_ours = jax.jvp(jax.grad(ours), (params,), (tangent,))[1]
    hvp_ref = jax.jvp(jax.grad(ref), (params,), (tangent,))[1]
    for a, b in zip(
        jax.tree_util.tree_leaves(hvp_ours), jax.tree_util.tree_leaves(hvp_ref), strict=True
    ):
        _close(a, b, 1e-10)


def test_multiple_outputs_affine_middle_layer_and_no_readout():
    rng = np.random.default_rng(3)
    dim = 18
    for layers in (
        _layers(rng, [dim, 5, 5, 5], "tanh", outputs=4, affine_at=1),
        _layers(rng, [dim, 5, 4], "sin", readout=False),
    ):
        X = jnp.asarray(rng.standard_normal((3, dim)))
        u, g2, lap, _ = projected_deep_laplacian(X, layers)
        ru, rg2, rlap, _ = _reference(layers, X)
        _close(u, ru)
        _close(g2, rg2)
        _close(lap, rlap)


def test_unbatched_point_jit_and_vmap():
    rng = np.random.default_rng(5)
    dim = 16
    layers = _layers(rng, [dim, 6, 6], "tanh")
    X = jnp.asarray(rng.standard_normal((4, dim)))
    V = jnp.asarray(rng.standard_normal((dim, 2)))
    batched = projected_deep_laplacian(X, layers, directions=V)
    single = projected_deep_laplacian(X[1], layers, directions=V)
    for a, b in zip(single, batched, strict=True):
        _close(a, b[1])
    params, specs = _split(layers)
    jitted = jax.jit(lambda x, ps: projected_deep_laplacian(x, _join(ps, specs), directions=V))(
        X, params
    )
    mapped = jax.vmap(lambda x: projected_deep_laplacian(x, layers, directions=V))(X)
    for a, b, c in zip(jitted, mapped, batched, strict=True):
        _close(a, c)
        _close(b, c)


@pytest.mark.parametrize(("dim", "width"), [(4, 8), (8, 8), (40, 8)])
def test_input_factor_reproduces_metric_with_min_columns(dim, width):
    rng = np.random.default_rng(dim)
    W1 = jnp.asarray(rng.standard_normal((width, dim)))
    F = input_factor(W1)
    assert F.shape == (width, min(dim, width))
    _close(F @ F.T, W1 @ W1.T)
    S = jnp.asarray(rng.standard_normal((dim, 3)))
    G = input_factor(W1, S)
    assert G.shape == (width, 3)
    _close(G @ G.T, W1 @ S @ S.T @ W1.T)


def test_refusals():
    rng = np.random.default_rng(0)
    layers = _layers(rng, [6, 4], "tanh")
    X = jnp.ones((2, 6))
    with pytest.raises(ValueError, match="layers must be non-empty"):
        projected_deep_laplacian(X, [])
    with pytest.raises(ValueError, match="x must have shape"):
        projected_deep_laplacian(jnp.ones((2, 2, 6)), layers)
    with pytest.raises(ValueError, match="first layer weight must have shape"):
        projected_deep_laplacian(jnp.ones((2, 5)), layers)
    with pytest.raises(ValueError, match="diffusion must have shape"):
        projected_deep_laplacian(X, layers, diffusion=jnp.ones((5, 2)))
    with pytest.raises(ValueError, match="directions must have shape"):
        projected_deep_laplacian(X, layers, directions=jnp.ones((5, 2)))
    for name in ("relu", "prelu"):
        with pytest.raises(ValueError, match="a.e.-zero second derivative"):
            projected_deep_laplacian(X, [(layers[0][0], layers[0][1], name)])
    with pytest.raises(ValueError, match=r"layer 1 weight must have shape"):
        projected_deep_laplacian(X, [layers[0], (jnp.ones((1, 3)), None, None)])
    with pytest.raises(ValueError, match=r"shape \(H, D\)"):
        input_factor(jnp.ones(3))


def _selection(dim, index):
    E = np.zeros((dim, len(index)))
    E[np.asarray(index), np.arange(len(index))] = 1.0
    return jnp.asarray(E)


@pytest.mark.parametrize("index", [range(1, 13), range(0, 5), [7, 0, 3, 11, 2, 9, 4, 12], [4]])
@pytest.mark.parametrize("width", [6, 16])
def test_coordinates_equal_an_explicit_selection(index, width):
    index = list(index)
    rng = np.random.default_rng(len(index) * 31 + width)
    dim = 13
    layers = _layers(rng, [dim, width, width], "tanh", outputs=2)
    X = jnp.asarray(rng.standard_normal((4, dim)))
    V = jnp.eye(dim)[:, :1]
    ours = projected_deep_laplacian(X, layers, coordinates=index, directions=V)
    ref = _reference(layers, X, S=_selection(dim, index), V=V)
    for a, b in zip(ours, ref, strict=True):
        _close(a, b)


@pytest.mark.parametrize("coordinates", [None, [2, 5, 1, 9, 0, 7, 3, 8]])
def test_diagonal_diffusion(coordinates):
    rng = np.random.default_rng(17)
    dim = 10
    layers = _layers(rng, [dim, 5, 5], "sin")
    m = dim if coordinates is None else len(coordinates)
    sigma = jnp.asarray(0.5 + rng.random(m))
    E = jnp.eye(dim) if coordinates is None else _selection(dim, coordinates)
    X = jnp.asarray(rng.standard_normal((3, dim)))
    _, g2, lap, _ = projected_deep_laplacian(X, layers, diffusion=sigma, coordinates=coordinates)
    _, rg2, rlap, _ = _reference(layers, X, S=E * sigma)
    _close(g2, rg2)
    _close(lap, rlap)


def test_coordinates_with_a_diffusion_factor():
    rng = np.random.default_rng(23)
    dim, index = 15, list(range(1, 15))
    layers = _layers(rng, [dim, 6, 6], "tanh")
    S = jnp.asarray(rng.standard_normal((len(index), 9)) / 3.0)
    X = jnp.asarray(rng.standard_normal((3, dim)))
    _, g2, lap, _ = projected_deep_laplacian(X, layers, diffusion=S, coordinates=index)
    _, rg2, rlap, _ = _reference(layers, X, S=_selection(dim, index) @ S)
    _close(g2, rg2)
    _close(lap, rlap)


@pytest.mark.parametrize(("index", "width"), [(list(range(1, 21)), 7), ([3, 0, 8, 5], 9)])
def test_parameter_and_diffusion_gradients_with_coordinates(index, width):
    rng = np.random.default_rng(width)
    dim = 21
    layers = _layers(rng, [dim, width, width], "tanh")
    sigma = jnp.asarray(0.5 + rng.random(len(index)))
    V = jnp.eye(dim)[:, :1]
    X = jnp.asarray(rng.standard_normal((5, dim)) / np.sqrt(dim))
    f = jnp.asarray(rng.standard_normal((5, 1)))
    E = _selection(dim, index)
    params, specs = _split(layers)

    def loss(ps, s, fn):
        _, g, lp, d = fn(_join(ps, specs), s)
        r = d[:, 0] - lp + 0.5 * g - f
        return 0.5 * jnp.mean(r * r)

    def ours(q, s):
        return projected_deep_laplacian(X, q, diffusion=s, coordinates=index, directions=V)

    got = jax.grad(loss, argnums=(0, 1))(params, sigma, ours)
    ref = jax.grad(loss, argnums=(0, 1))(params, sigma, lambda q, s: _reference(q, X, S=E * s, V=V))
    for a, b in zip(jax.tree_util.tree_leaves(got), jax.tree_util.tree_leaves(ref), strict=True):
        _close(a, b, 1e-11)


def test_hessian_vector_products_with_coordinates():
    rng = np.random.default_rng(29)
    dim, width, index = 16, 6, list(range(1, 16))
    layers = _layers(rng, [dim, width, width], "tanh")
    X = jnp.asarray(rng.standard_normal((3, dim)) / np.sqrt(dim))
    V = jnp.eye(dim)[:, :1]
    E = _selection(dim, index)
    params, specs = _split(layers)
    tangent = jax.tree_util.tree_map(lambda p: jnp.asarray(rng.standard_normal(p.shape)), params)

    def loss(ps, fn):
        _, g, lp, d = fn(_join(ps, specs))
        return jnp.sum((d[:, 0] - lp + g) ** 2)

    def ours(ps):
        return loss(ps, lambda q: projected_deep_laplacian(X, q, coordinates=index, directions=V))

    def ref(ps):
        return loss(ps, lambda q: _reference(q, X, S=E, V=V))

    hvp_ours = jax.jvp(jax.grad(ours), (params,), (tangent,))[1]
    hvp_ref = jax.jvp(jax.grad(ref), (params,), (tangent,))[1]
    for a, b in zip(
        jax.tree_util.tree_leaves(hvp_ours), jax.tree_util.tree_leaves(hvp_ref), strict=True
    ):
        _close(a, b, 1e-10)


def test_coordinates_under_jit_and_input_factor():
    rng = np.random.default_rng(31)
    dim, index = 12, [0, 2, 4, 6, 8, 10, 11]
    layers = _layers(rng, [dim, 5, 5], "tanh")
    X = jnp.asarray(rng.standard_normal((4, dim)))
    params, specs = _split(layers)
    jitted = jax.jit(
        lambda x, ps: projected_deep_laplacian(x, _join(ps, specs), coordinates=index)
    )(X, params)
    for a, b in zip(jitted, projected_deep_laplacian(X, layers, coordinates=index), strict=True):
        if b is not None:
            _close(a, b)
    W1 = layers[0][0]
    sigma = jnp.asarray(0.5 + rng.random(len(index)))
    F = input_factor(W1, sigma, coordinates=index)
    assert F.shape == (5, 5)
    M = W1[:, np.asarray(index)] * sigma
    _close(F @ F.T, M @ M.T)


def test_coordinate_refusals():
    rng = np.random.default_rng(0)
    layers = _layers(rng, [6, 4], "tanh")
    X = jnp.ones((2, 6))
    for bad in ([], [0, 6], [-1, 2], [1, 1], [0.0, 1.0], [[0, 1]]):
        with pytest.raises(ValueError, match="coordinates must be distinct integer indices"):
            projected_deep_laplacian(X, layers, coordinates=bad)
    with pytest.raises(ValueError, match=r"diffusion must have shape \(3,\)"):
        projected_deep_laplacian(X, layers, coordinates=[0, 1, 2], diffusion=jnp.ones(6))
    with pytest.raises(ValueError, match=r"diffusion must have shape \(6,\)"):
        projected_deep_laplacian(X, layers, diffusion=jnp.ones((6, 2, 1)))


def _point_reference(layers, X, s, coordinates=None, V=None):
    dim = X.shape[-1]
    E = _selection(dim, range(dim) if coordinates is None else coordinates)

    def point(x, sb, v):
        jac = jax.jacfwd(lambda y: _forward(layers, y))(x)
        hess = jax.hessian(lambda y: _forward(layers, y))(x)
        S = E * sb
        lap = jnp.einsum("cij,ij->c", hess, S @ S.T)
        g2 = jnp.sum((jac @ S) ** 2, axis=-1)
        d = None if v is None else (jac @ v).T
        return _forward(layers, x), g2, lap, d

    if V is None:
        u, g2, lap, _ = jax.vmap(lambda x, sb: point(x, sb, None))(X, s)
        return u, g2, lap, None
    return jax.vmap(point)(X, s, V)


_POINT_CASES = [
    (6, 8, 3, None),
    (20, 8, 3, None),
    (20, 6, 4, None),
    (21, 6, 5, list(range(1, 21))),
    (17, 8, 2, [3, 0, 16, 5, 9, 12, 1, 7, 14, 2, 10]),
]


@pytest.mark.parametrize(("dim", "width", "batch", "coordinates"), _POINT_CASES)
@pytest.mark.parametrize("act", ["tanh", "sin"])
def test_point_diffusion_matches_nested_autodiff(dim, width, batch, coordinates, act):
    rng = np.random.default_rng(dim * width + batch)
    layers = _layers(rng, [dim, width, width], act, outputs=2)
    m = dim if coordinates is None else len(coordinates)
    X = jnp.asarray(rng.standard_normal((batch, dim)))
    s = jnp.asarray(rng.standard_normal((batch, m)))
    V = jnp.asarray(rng.standard_normal((batch, dim, 2)))
    ours = projected_deep_laplacian(
        X, layers, point_diffusion=s, coordinates=coordinates, directions=V
    )
    for a, b in zip(ours, _point_reference(layers, X, s, coordinates, V), strict=True):
        _close(a, b)


def test_point_diffusion_with_equal_rows_is_the_shared_diagonal():
    rng = np.random.default_rng(41)
    dim, index = 25, list(range(1, 25))
    layers = _layers(rng, [dim, 7, 7], "tanh")
    X = jnp.asarray(rng.standard_normal((6, dim)))
    sigma = jnp.asarray(0.5 + rng.random(len(index)))
    rows = jnp.broadcast_to(sigma, (6, len(index)))
    ours = projected_deep_laplacian(X, layers, point_diffusion=rows, coordinates=index)
    shared = projected_deep_laplacian(X, layers, diffusion=sigma, coordinates=index)
    for a, b in zip(ours[:3], shared[:3], strict=True):
        _close(a, b)


@pytest.mark.parametrize(("dim", "width", "batch"), [(20, 8, 3), (20, 6, 4), (5, 8, 3)])
def test_point_diffusion_parameter_and_coefficient_gradients(dim, width, batch):
    rng = np.random.default_rng(dim + width + batch)
    layers = _layers(rng, [dim, width, width], "tanh")
    index = list(range(1, dim))
    X = jnp.asarray(rng.standard_normal((batch, dim)) / np.sqrt(dim))
    s = jnp.asarray(0.5 + rng.random((batch, dim - 1)))
    V = jnp.concatenate([jnp.ones((batch, 1)), -X[:, 1:]], axis=1)[:, :, None]
    f = jnp.asarray(rng.standard_normal((batch, 1)))
    params, specs = _split(layers)

    def loss(ps, s, fn):
        _, g, lp, d = fn(_join(ps, specs), s)
        r = d[:, 0] + lp - 0.5 * g - f
        return 0.5 * jnp.mean(r * r)

    ours = jax.grad(loss, argnums=(0, 1))(
        params,
        s,
        lambda q, s: projected_deep_laplacian(
            X, q, point_diffusion=s, coordinates=index, directions=V
        ),
    )
    ref = jax.grad(loss, argnums=(0, 1))(
        params, s, lambda q, s: _point_reference(q, X, s, index, V)
    )
    for a, b in zip(jax.tree_util.tree_leaves(ours), jax.tree_util.tree_leaves(ref), strict=True):
        _close(a, b, 1e-11)


@pytest.mark.parametrize(("dim", "width", "batch"), [(18, 8, 3), (18, 6, 4)])
def test_point_diffusion_hessian_vector_products(dim, width, batch):
    rng = np.random.default_rng(3 * dim + width)
    layers = _layers(rng, [dim, width, width], "tanh")
    X = jnp.asarray(rng.standard_normal((batch, dim)) / np.sqrt(dim))
    s = jnp.asarray(0.5 + rng.random((batch, dim)))
    params, specs = _split(layers)
    tangent = jax.tree_util.tree_map(lambda p: jnp.asarray(rng.standard_normal(p.shape)), params)

    def loss(ps, fn):
        _, g, lp, _ = fn(_join(ps, specs))
        return jnp.sum((lp - g) ** 2)

    def ours(ps):
        return loss(ps, lambda q: projected_deep_laplacian(X, q, point_diffusion=s))

    def ref(ps):
        return loss(ps, lambda q: _point_reference(q, X, s))

    hvp_ours = jax.jvp(jax.grad(ours), (params,), (tangent,))[1]
    hvp_ref = jax.jvp(jax.grad(ref), (params,), (tangent,))[1]
    for a, b in zip(
        jax.tree_util.tree_leaves(hvp_ours), jax.tree_util.tree_leaves(hvp_ref), strict=True
    ):
        _close(a, b, 1e-10)


@pytest.mark.parametrize("batch", [2, 4])
def test_point_diffusion_rank_deficient_point_values_and_gradients(batch):
    rng = np.random.default_rng(43 + batch)
    dim, width = 20, 6
    layers = _layers(rng, [dim, width, width], "tanh")
    X = jnp.asarray(rng.standard_normal((batch, dim)))
    s = jnp.asarray(0.5 + rng.random((batch, dim)))
    s = s.at[0, 3:].set(0.0)
    _, g2, lap, _ = projected_deep_laplacian(X, layers, point_diffusion=s)
    _, rg2, rlap, _ = _point_reference(layers, X, s)
    _close(g2, rg2)
    _close(lap, rlap)
    params, specs = _split(layers)

    def loss(ps, fn):
        _, g, lp, _ = fn(_join(ps, specs))
        return jnp.mean((lp + g) ** 2)

    ga = jax.grad(lambda ps: loss(ps, lambda q: projected_deep_laplacian(X, q, point_diffusion=s)))(
        params
    )
    gr = jax.grad(lambda ps: loss(ps, lambda q: _point_reference(q, X, s)))(params)
    for a, b in zip(jax.tree_util.tree_leaves(ga), jax.tree_util.tree_leaves(gr), strict=True):
        assert bool(jnp.all(jnp.isfinite(a)))
        _close(a, b, 1e-11)


def test_point_diffusion_unbatched_jit_and_vmap():
    rng = np.random.default_rng(47)
    dim, width = 22, 6
    layers = _layers(rng, [dim, width, width], "tanh")
    X = jnp.asarray(rng.standard_normal((4, dim)))
    s = jnp.asarray(0.5 + rng.random((4, dim)))
    batched = projected_deep_laplacian(X, layers, point_diffusion=s)
    single = projected_deep_laplacian(X[1], layers, point_diffusion=s[1])
    for a, b in zip(single[:3], batched[:3], strict=True):
        _close(a, b[1])
    params, specs = _split(layers)
    jitted = jax.jit(
        lambda x, s, ps: projected_deep_laplacian(x, _join(ps, specs), point_diffusion=s)
    )(X, s, params)
    mapped = jax.vmap(lambda x, s: projected_deep_laplacian(x, layers, point_diffusion=s))(X, s)
    for a, b, c in zip(jitted[:3], mapped[:3], batched[:3], strict=True):
        _close(a, c)
        _close(b, c)


def test_point_diffusion_refusals():
    rng = np.random.default_rng(0)
    layers = _layers(rng, [6, 4], "tanh")
    X = jnp.ones((2, 6))
    with pytest.raises(ValueError, match="either diffusion or point_diffusion"):
        projected_deep_laplacian(X, layers, diffusion=jnp.ones(6), point_diffusion=jnp.ones((2, 6)))
    with pytest.raises(ValueError, match=r"point_diffusion must have shape \(2, 6\)"):
        projected_deep_laplacian(X, layers, point_diffusion=jnp.ones((2, 5)))
    with pytest.raises(ValueError, match=r"point_diffusion must have shape \(3,\)"):
        projected_deep_laplacian(
            X[0], layers, point_diffusion=jnp.ones((1, 3)), coordinates=[0, 1, 2]
        )
