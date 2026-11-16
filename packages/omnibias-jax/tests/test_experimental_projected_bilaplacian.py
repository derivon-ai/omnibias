# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Projected forward bi-Laplacian of deep MLPs vs nested autodiff on the full input."""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from omnibias.jax.experimental.projected_bilaplacian import projected_deep_bilaplacian  # noqa: E402

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
    A = jnp.eye(X.shape[-1]) if S is None else S @ S.T

    def lap(f):
        return lambda y: jnp.einsum("cij,ij->c", jax.hessian(f)(y), A)

    def point(x, v):
        u = lambda y: _forward(layers, y)  # noqa: E731
        J = jax.jacobian(u)(x)
        g2 = jnp.einsum("ci,ij,cj->c", J, A, J)
        dd = None if v is None else jnp.einsum("cd,dq->qc", J, v)
        return u(x), g2, lap(u)(x), lap(lap(u))(x), dd

    if V is None or V.ndim == 2:
        return jax.vmap(lambda x: point(x, V))(X)
    return jax.vmap(point)(X, V)


def _split(layers):
    return [(W, b) for W, b, _ in layers], [spec for _, _, spec in layers]


def _join(params, specs):
    return [(W, b, spec) for (W, b), spec in zip(params, specs, strict=True)]


def _close(a, b, tol=1e-12):
    if b is None:
        assert a is None
        return
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape
    np.testing.assert_allclose(a, b, rtol=0, atol=tol * (np.max(np.abs(b)) + 1e-300))


@pytest.mark.parametrize("act", ["tanh", "sin", "sigmoid", "softplus"])
@pytest.mark.parametrize(
    ("dim", "width", "depth"),
    [(3, 6, 1), (5, 5, 2), (12, 5, 2), (16, 4, 3), (12, 10, 2), (9, 13, 2)],
)
def test_bilaplacian_matches_nested_autodiff(act, dim, width, depth):
    rng = np.random.default_rng(dim * 100 + width + depth)
    layers = _layers(rng, [dim] + [width] * depth, act)
    X = jnp.asarray(rng.standard_normal((3, dim)) / np.sqrt(dim))
    for a, b in zip(projected_deep_bilaplacian(X, layers), _reference(layers, X), strict=True):
        _close(a, b)


@pytest.mark.parametrize(("dim", "width", "rank"), [(34, 33, None), (36, 40, 33)])
def test_block_storage_values_and_gradients(dim, width, rank):
    rng = np.random.default_rng(dim + width)
    layers = _layers(rng, [dim, width, width], "tanh")
    S = None if rank is None else jnp.asarray(rng.standard_normal((dim, rank)) / np.sqrt(rank))
    X = jnp.asarray(rng.standard_normal((2, dim)) / np.sqrt(dim))
    for a, b in zip(
        projected_deep_bilaplacian(X, layers, diffusion=S), _reference(layers, X, S=S), strict=True
    ):
        _close(a, b)
    params, specs = _split(layers)

    def loss(ps, fn):
        u, g2, lap, bilap, _ = fn(_join(ps, specs))
        return jnp.sum(bilap**2) + jnp.sum(lap * u) + jnp.sum(g2 * u)

    ours = jax.grad(lambda ps: loss(ps, lambda q: projected_deep_bilaplacian(X, q, diffusion=S)))
    ref = jax.grad(lambda ps: loss(ps, lambda q: _reference(q, X, S=S)))
    for a, b in zip(
        jax.tree_util.tree_leaves(ours(params)), jax.tree_util.tree_leaves(ref(params)), strict=True
    ):
        _close(a, b, 1e-11)


@pytest.mark.parametrize(("dim", "width"), [(12, 12), (34, 34)])
def test_parameter_gradients_compile_and_agree_under_jit(dim, width):
    """Packed (k < 32) and block (k >= 32) storage give the same parameter gradients eagerly and under ``jit``.

    The planned operator's compile-and-differentiate path is covered in ``test_experimental_planner_v2``.
    """
    rng = np.random.default_rng(dim)
    layers = _layers(rng, [dim, width, width], "tanh")
    X = jnp.asarray(rng.standard_normal((3, dim)) / np.sqrt(dim))
    params, specs = _split(layers)

    def loss(ps):
        u, g2, lap, bilap, _ = projected_deep_bilaplacian(X, _join(ps, specs))
        return jnp.sum(bilap**2) + jnp.sum(lap * u) + jnp.sum(g2 * u)

    eager = jax.tree_util.tree_leaves(jax.grad(loss)(params))
    compiled = jax.tree_util.tree_leaves(jax.jit(jax.grad(loss))(params))
    again = jax.tree_util.tree_leaves(
        jax.jit(jax.grad(loss))(params)
    )  # a second trace must not see the first one's constants
    for a, b, c in zip(eager, compiled, again, strict=True):
        _close(b, a, 1e-12)
        _close(c, a, 1e-12)


@pytest.mark.parametrize(("dim", "rank"), [(10, 3), (10, 10), (14, 8)])
def test_constant_diffusion_metric(dim, rank):
    rng = np.random.default_rng(dim + rank)
    layers = _layers(rng, [dim, 5, 5], "tanh", outputs=2)
    S = jnp.asarray(rng.standard_normal((dim, rank)) / np.sqrt(rank))
    X = jnp.asarray(rng.standard_normal((3, dim)) / np.sqrt(dim))
    for a, b in zip(
        projected_deep_bilaplacian(X, layers, diffusion=S), _reference(layers, X, S=S), strict=True
    ):
        _close(a, b)


@pytest.mark.parametrize("width", [5, 10])
@pytest.mark.parametrize("deficient", [False, True])
def test_parameter_gradients(deficient, width):
    rng = np.random.default_rng(17)
    dim = 14
    layers = _layers(rng, [dim, width, width], "tanh")
    if deficient:
        W1, b1, s1 = layers[0]
        layers[0] = (W1.at[-1].set(W1[0] - 0.5 * W1[1]), b1, s1)
    X = jnp.asarray(rng.standard_normal((3, dim)) / np.sqrt(dim))
    f = jnp.asarray(rng.standard_normal((3, 1)))
    params, specs = _split(layers)

    def loss(ps, fn):
        _, _, lap, bilap, _ = fn(_join(ps, specs))
        r = bilap - 0.3 * lap - f
        return 0.5 * jnp.mean(r * r)

    ours = jax.grad(lambda ps: loss(ps, lambda q: projected_deep_bilaplacian(X, q)))(params)
    ref = jax.grad(lambda ps: loss(ps, lambda q: _reference(q, X)))(params)
    for a, b in zip(jax.tree_util.tree_leaves(ours), jax.tree_util.tree_leaves(ref), strict=True):
        assert bool(jnp.all(jnp.isfinite(a)))
        _close(a, b, 1e-11)


@pytest.mark.parametrize(("dim", "width", "rank"), [(6, 5, None), (12, 10, None), (13, 11, 9)])
def test_forward_mode_and_hessian_vector_products(dim, width, rank):
    rng = np.random.default_rng(31 + dim)
    layers = _layers(rng, [dim, width, width], "tanh")
    S = None if rank is None else jnp.asarray(rng.standard_normal((dim, rank)) / np.sqrt(rank))
    X = jnp.asarray(rng.standard_normal((2, dim)) / np.sqrt(dim))
    params, specs = _split(layers)
    tangent = jax.tree_util.tree_map(lambda p: jnp.asarray(rng.standard_normal(p.shape)), params)

    def loss(ps, fn):
        u, _, lap, bilap, _ = fn(_join(ps, specs))
        return jnp.sum(bilap**2) + jnp.sum(lap * u)

    def ours(ps):
        return loss(ps, lambda q: projected_deep_bilaplacian(X, q, diffusion=S))

    def ref(ps):
        return loss(ps, lambda q: _reference(q, X, S=S))

    _close(jax.jvp(ours, (params,), (tangent,))[1], jax.jvp(ref, (params,), (tangent,))[1], 1e-11)
    hvp_ours = jax.jvp(jax.grad(ours), (params,), (tangent,))[1]
    hvp_ref = jax.jvp(jax.grad(ref), (params,), (tangent,))[1]
    for a, b in zip(
        jax.tree_util.tree_leaves(hvp_ours), jax.tree_util.tree_leaves(hvp_ref), strict=True
    ):
        _close(a, b, 1e-10)


def test_metric_gradient():
    rng = np.random.default_rng(23)
    dim = 9
    layers = _layers(rng, [dim, 4, 4], "sin")
    S = jnp.asarray(rng.standard_normal((dim, dim)) / np.sqrt(dim))
    X = jnp.asarray(rng.standard_normal((2, dim)) / np.sqrt(dim))

    def loss(S, fn):
        return jnp.sum(fn(S)[3] ** 2) + jnp.sum(fn(S)[1] ** 2)

    ours = jax.grad(loss)(S, lambda s: projected_deep_bilaplacian(X, layers, diffusion=s))
    ref = jax.grad(loss)(S, lambda s: _reference(layers, X, S=s))
    _close(ours, ref, 1e-11)


def test_affine_middle_layer_multiple_outputs_and_no_readout():
    rng = np.random.default_rng(3)
    dim = 11
    for layers in (
        _layers(rng, [dim, 4, 4, 4], "tanh", outputs=3, affine_at=1),
        _layers(rng, [dim, 10, 9], "tanh", outputs=2, affine_at=0),
        _layers(rng, [dim, 5, 3], "sin", readout=False),
    ):
        X = jnp.asarray(rng.standard_normal((2, dim)) / np.sqrt(dim))
        out = projected_deep_bilaplacian(X, layers)
        for a, b in zip(out, _reference(layers, X), strict=True):
            _close(a, b)


@pytest.mark.parametrize("width", [5, 10])
def test_unbatched_jit_and_vmap(width):
    rng = np.random.default_rng(5)
    dim = 10
    layers = _layers(rng, [dim, width, width], "tanh")
    X = jnp.asarray(rng.standard_normal((3, dim)) / np.sqrt(dim))
    batched = projected_deep_bilaplacian(X, layers)
    single = projected_deep_bilaplacian(X[1], layers)
    params, specs = _split(layers)
    jitted = jax.jit(lambda x, ps: projected_deep_bilaplacian(x, _join(ps, specs)))(X, params)
    mapped = jax.vmap(lambda x: projected_deep_bilaplacian(x, layers))(X)
    for s, b, j, m in zip(single, batched, jitted, mapped, strict=True):
        if b is None:
            assert s is None and j is None and m is None
            continue
        _close(s, b[1])
        _close(j, b)
        _close(m, b)


def test_refusals():
    rng = np.random.default_rng(0)
    layers = _layers(rng, [6, 4], "tanh")
    X = jnp.ones((2, 6))
    with pytest.raises(ValueError, match="non-empty"):
        projected_deep_bilaplacian(X, [])
    with pytest.raises(ValueError, match="shape"):
        projected_deep_bilaplacian(jnp.ones((2, 2, 6)), layers)
    with pytest.raises(ValueError, match="first layer"):
        projected_deep_bilaplacian(jnp.ones((2, 5)), layers)
    with pytest.raises(ValueError, match="diffusion"):
        projected_deep_bilaplacian(X, layers, diffusion=jnp.ones((5, 2)))
    with pytest.raises(ValueError, match="directions"):
        projected_deep_bilaplacian(X, layers, directions=jnp.ones((5, 1)))
    with pytest.raises(ValueError, match="directions"):
        projected_deep_bilaplacian(X[0], layers, directions=jnp.ones((1, 6, 1)))
    W, b, _ = layers[0]
    with pytest.raises(KeyError, match="Unknown activation"):
        projected_deep_bilaplacian(X, [(W, b, "definitely_not_an_activation")])
    with pytest.raises(ValueError, match="order 4"):
        projected_deep_bilaplacian(X, [(W, b, "arctan")])


@pytest.mark.parametrize("act", ["relu", "elu", "selu", "huber", "softsign", "hardswish", "prelu"])
def test_activations_with_jumps_below_order_four_are_refused(act):
    rng = np.random.default_rng(1)
    W = jnp.asarray(rng.standard_normal((4, 6)))
    X = jnp.asarray(rng.standard_normal((2, 6)))
    with pytest.raises(ValueError, match="three times"):
        projected_deep_bilaplacian(X, [(W, None, act)])
    out = projected_deep_bilaplacian(X, [(W, None, act)], allow_almost_everywhere=True)
    assert all(bool(jnp.all(jnp.isfinite(o))) for o in out if o is not None)


def _selection(dim, index):
    E = np.zeros((dim, len(index)))
    E[np.asarray(index), np.arange(len(index))] = 1.0
    return jnp.asarray(E)


@pytest.mark.parametrize(
    ("index", "width", "diagonal"),
    [(list(range(1, 10)), 5, False), ([6, 0, 3, 8], 7, True), (list(range(1, 10)), 12, True)],
)
def test_coordinates_and_diagonal_diffusion(index, width, diagonal):
    rng = np.random.default_rng(len(index) + width)
    dim = 10
    layers = _layers(rng, [dim, width, width], "tanh")
    sigma = jnp.asarray(0.5 + rng.random(len(index))) if diagonal else None
    S = _selection(dim, index) * (1.0 if sigma is None else sigma)
    X = jnp.asarray(rng.standard_normal((2, dim)) / np.sqrt(dim))
    ours = projected_deep_bilaplacian(X, layers, diffusion=sigma, coordinates=index)
    for a, b in zip(ours, _reference(layers, X, S=S), strict=True):
        _close(a, b)
    params, specs = _split(layers)

    def loss(ps, fn):
        u, _, lap, bilap, _ = fn(_join(ps, specs))
        return jnp.sum(bilap**2) + jnp.sum(lap * u)

    got = jax.grad(
        lambda ps: loss(
            ps, lambda q: projected_deep_bilaplacian(X, q, diffusion=sigma, coordinates=index)
        )
    )(params)
    ref = jax.grad(lambda ps: loss(ps, lambda q: _reference(q, X, S=S)))(params)
    for a, b in zip(jax.tree_util.tree_leaves(got), jax.tree_util.tree_leaves(ref), strict=True):
        _close(a, b, 1e-11)


@pytest.mark.parametrize(
    ("width", "diagonal", "per_point", "affine_at", "outputs"),
    [
        (5, False, False, None, 1),
        (12, True, True, None, 1),
        (9, False, True, 0, 2),
        (6, True, False, 1, 3),
    ],
)
def test_time_dependent_fourth_order_residual(width, diagonal, per_point, affine_at, outputs):
    rng = np.random.default_rng(width * 7 + outputs)
    dim = 10
    index = list(range(1, dim))
    layers = _layers(rng, [dim, width, width], "tanh", outputs=outputs, affine_at=affine_at)
    sigma = jnp.asarray(0.5 + rng.random(dim - 1)) if diagonal else None
    S = _selection(dim, index) * (1.0 if sigma is None else sigma)
    X = jnp.asarray(rng.standard_normal((3, dim)) / np.sqrt(dim))
    e_t = jnp.eye(dim)[:, :1]
    if per_point:
        V = jnp.stack([jnp.broadcast_to(e_t[:, 0], X.shape), X.at[:, 0].set(0.0)], axis=-1)
    else:
        V = e_t
    ours = projected_deep_bilaplacian(X, layers, diffusion=sigma, coordinates=index, directions=V)
    for a, b in zip(ours, _reference(layers, X, S=S, V=V), strict=True):
        _close(a, b)
    params, specs = _split(layers)
    f = jnp.asarray(rng.standard_normal((3, outputs)))

    def loss(ps, fn):
        _, g2, lap, bilap, dd = fn(_join(ps, specs))
        r = dd[:, 0] + bilap + lap + 0.5 * g2 - f
        return 0.5 * jnp.mean(r * r)

    got = jax.grad(
        lambda ps: loss(
            ps,
            lambda q: projected_deep_bilaplacian(
                X, q, diffusion=sigma, coordinates=index, directions=V
            ),
        )
    )(params)
    ref = jax.grad(lambda ps: loss(ps, lambda q: _reference(q, X, S=S, V=V)))(params)
    for a, b in zip(jax.tree_util.tree_leaves(got), jax.tree_util.tree_leaves(ref), strict=True):
        _close(a, b, 1e-11)


def test_unbatched_directions():
    rng = np.random.default_rng(41)
    dim = 8
    layers = _layers(rng, [dim, 6, 6], "sin", outputs=2)
    X = jnp.asarray(rng.standard_normal((3, dim)) / np.sqrt(dim))
    V = jnp.asarray(rng.standard_normal((dim, 2)))
    batched = projected_deep_bilaplacian(X, layers, directions=V)
    single = projected_deep_bilaplacian(X[2], layers, directions=V)
    assert single[4].shape == (2, 2)
    for s, b in zip(single, batched, strict=True):
        _close(s, b[2])
