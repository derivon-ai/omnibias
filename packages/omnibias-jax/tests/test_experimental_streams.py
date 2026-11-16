# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""The stacked stream engine against the dense Hessian, and its structural guarantees."""

from __future__ import annotations

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
from omnibias.jax.experimental.streams import second_order_streams  # noqa: E402


def _layers(dim, width, depth, out=1, act="tanh", seed=0):
    g = np.random.default_rng(seed)
    sizes = [dim] + [width] * depth + [out]
    layers = []
    for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:], strict=True)):
        layers.append((jnp.asarray(g.standard_normal((b, a)) / np.sqrt(a)), jnp.asarray(0.3 * g.standard_normal(b)),
                       act if i < depth else None))
    return layers


def _forward(layers, z):
    a = z
    for W, b, act in layers:
        a = a @ W.T + b
        if act is not None:
            a = {"tanh": jnp.tanh, "sigmoid": jax.nn.sigmoid, "softplus": jax.nn.softplus}[act](a)
    return a


def _oracle(layers, x, A):
    """Dense reference: ``u``, ``grad u^T A grad u`` and ``tr(A hess u)`` per point (scalar output)."""
    f = lambda z: _forward(layers, z[None])[0, 0]  # noqa: E731
    u = jax.vmap(f)(x)
    g = jax.vmap(jax.grad(f))(x)
    h = jax.vmap(jax.hessian(f))(x)
    return u, jnp.einsum("bi,ij,bj->b", g, A, g), jnp.einsum("ij,bij->b", A, h)


@pytest.mark.parametrize("act", ["tanh", "sigmoid", "softplus"])
@pytest.mark.parametrize(("dim", "width", "depth"), [(2, 8, 2), (5, 8, 3), (40, 8, 2)])
@pytest.mark.parametrize("form", ["auto", "diag", "gram"])
def test_identity_metric_matches_the_dense_hessian(act, dim, width, depth, form):
    layers = _layers(dim, width, depth, act=act, seed=dim)
    x = jnp.asarray(np.random.default_rng(1).standard_normal((6, dim)))
    r = second_order_streams(x, layers, form=form)
    u, g2, lap = _oracle(layers, x, jnp.eye(dim))
    np.testing.assert_allclose(r.value[:, 0], u, atol=1e-12)
    np.testing.assert_allclose(r.grad_sq[:, 0], g2, atol=1e-11)
    np.testing.assert_allclose(r.laplacian[:, 0], lap, atol=1e-11)


@pytest.mark.parametrize("form", ["diag", "gram"])
@pytest.mark.parametrize("dim", [3, 30])
def test_indefinite_metric_any_symmetric_form(form, dim):
    layers = _layers(dim, 8, 2, seed=3)
    x = jnp.asarray(np.random.default_rng(2).standard_normal((5, dim)))
    g = np.random.default_rng(4)
    Q, _ = np.linalg.qr(g.standard_normal((dim, dim)))
    lam = np.where(np.arange(dim) % 2 == 0, 1.0, -1.0) * (0.5 + g.random(dim))
    A = jnp.asarray((Q * lam) @ Q.T)  # indefinite, full rank
    assert float(np.linalg.eigvalsh(np.asarray(A)).min()) < 0 < float(np.linalg.eigvalsh(np.asarray(A)).max())
    r = second_order_streams(x, layers, hess_form=A, form=form)
    _, g2, lap = _oracle(layers, x, A)
    np.testing.assert_allclose(r.grad_sq[:, 0], g2, atol=1e-10)
    np.testing.assert_allclose(r.laplacian[:, 0], lap, atol=1e-10)


def test_traced_hess_form_is_differentiable_in_the_metric():
    layers = _layers(6, 8, 2, seed=5)
    x = jnp.asarray(np.random.default_rng(6).standard_normal((4, 6)))
    A0 = np.random.default_rng(7).standard_normal((6, 6))
    A0 = jnp.asarray(0.5 * (A0 + A0.T))
    ours = jax.jit(jax.grad(lambda A: jnp.sum(second_order_streams(x, layers, hess_form=A).laplacian)))(A0)
    ref = jax.grad(lambda A: jnp.sum(_oracle(layers, x, A)[2]))(A0)
    np.testing.assert_allclose(0.5 * (ours + ours.T), 0.5 * (ref + ref.T), atol=1e-10)


def test_weights_and_diffusion_and_coordinates():
    dim, width = 7, 6
    layers = _layers(dim, width, 2, seed=8)
    x = jnp.asarray(np.random.default_rng(9).standard_normal((5, dim)))
    idx = np.array([1, 3, 4])
    S = jnp.asarray(np.random.default_rng(10).standard_normal((3, 2)))
    A = np.zeros((dim, dim))
    A[np.ix_(idx, idx)] = np.asarray(S @ S.T)
    r = second_order_streams(x, layers, diffusion=S, coordinates=idx)
    _, g2, lap = _oracle(layers, x, jnp.asarray(A))
    np.testing.assert_allclose(r.grad_sq[:, 0], g2, atol=1e-11)
    np.testing.assert_allclose(r.laplacian[:, 0], lap, atol=1e-11)
    w = jnp.asarray([1.0, -1.0])
    r = second_order_streams(x, layers, diffusion=S, coordinates=idx, weights=w)
    A[np.ix_(idx, idx)] = np.asarray(S @ jnp.diag(w) @ S.T)
    _, g2, lap = _oracle(layers, x, jnp.asarray(A))
    np.testing.assert_allclose(r.laplacian[:, 0], lap, atol=1e-11)


def test_drift_directions_constant_and_per_point():
    dim = 5
    layers = _layers(dim, 8, 2, seed=11)
    x = jnp.asarray(np.random.default_rng(12).standard_normal((4, dim)))
    v = jnp.asarray(np.random.default_rng(13).standard_normal((dim, 2)))
    vp = jnp.asarray(np.random.default_rng(14).standard_normal((4, dim, 3)))
    f = lambda z: _forward(layers, z[None])[0, 0]  # noqa: E731
    grads = jax.vmap(jax.grad(f))(x)
    r = second_order_streams(x, layers, directions=v)
    np.testing.assert_allclose(r.directional[:, :, 0], grads @ v, atol=1e-12)
    r = second_order_streams(x, layers, directions=vp)
    np.testing.assert_allclose(r.directional[:, :, 0], jnp.einsum("bi,bij->bj", grads, vp), atol=1e-12)


@pytest.mark.parametrize("dim", [4, 40])
def test_parameter_gradients_to_every_order(dim):
    layers = _layers(dim, 8, 2, seed=15)
    x = jnp.asarray(np.random.default_rng(16).standard_normal((5, dim)))
    flat = [(W, b) for W, b, _ in layers]
    acts = [a for _, _, a in layers]

    def rebuild(p):
        return [(W, b, a) for (W, b), a in zip(p, acts, strict=True)]

    def ours(p):
        r = second_order_streams(x, rebuild(p))
        return jnp.sum(r.laplacian**2) + jnp.sum(r.grad_sq)

    def ref(p):
        _, g2, lap = _oracle(rebuild(p), x, jnp.eye(dim))
        return jnp.sum(lap**2) + jnp.sum(g2)

    g_ours, g_ref = jax.grad(ours)(flat), jax.grad(ref)(flat)
    for (a, b), (c, d) in zip(g_ours, g_ref, strict=True):
        np.testing.assert_allclose(a, c, atol=1e-9)
        np.testing.assert_allclose(b, d, atol=1e-9)
    # second order: Hessian-vector product of the loss in a random parameter direction
    t = jax.tree_util.tree_map(lambda a: jnp.asarray(np.random.default_rng(17).standard_normal(a.shape)), flat)
    hvp_ours = jax.jvp(jax.grad(ours), (flat,), (t,))[1]
    hvp_ref = jax.jvp(jax.grad(ref), (flat,), (t,))[1]
    for (a, b), (c, d) in zip(hvp_ours, hvp_ref, strict=True):
        np.testing.assert_allclose(a, c, atol=1e-8)
        np.testing.assert_allclose(b, d, atol=1e-8)


def test_no_conditional_solver_or_host_transfer_in_the_compiled_program():
    dim, width = 64, 8  # D > 2 H: the Gram form
    layers = _layers(dim, width, 2, seed=18)
    x = jnp.asarray(np.random.default_rng(19).standard_normal((4, dim)))
    params = [(W, b) for W, b, _ in layers]

    def loss(p, z):
        r = second_order_streams(z, [(W, b, a) for (W, b), (_, _, a) in zip(p, layers, strict=True)])
        return jnp.sum(r.laplacian)

    for fn in (loss, jax.grad(loss)):
        text = jax.jit(fn).lower(params, x).compile().as_text().lower()
        for forbidden in ("conditional(", "cholesky", "potrf", "geqrf", "orgqr", "lapack", "cusolver", "callback"):
            assert forbidden not in text, forbidden


def test_gram_form_channel_count_is_independent_of_the_input_dimension():
    layers = _layers(500, 8, 2, seed=20)
    x = jnp.asarray(np.random.default_rng(21).standard_normal((3, 500)))
    r = second_order_streams(x, layers)  # auto -> gram: 2H channels, never D
    assert r.jacobian is None
    _, _, lap = _oracle(layers, x, jnp.eye(500))
    np.testing.assert_allclose(r.laplacian[:, 0], lap, atol=1e-10)


def test_full_jacobian_and_unbatched_and_vmap():
    layers = _layers(4, 6, 2, out=3, seed=22)
    x = jnp.asarray(np.random.default_rng(23).standard_normal((5, 4)))
    r = second_order_streams(x, layers, full_jacobian=True)
    jac = jax.vmap(jax.jacfwd(lambda z: _forward(layers, z[None])[0]))(x)  # (B, C, D)
    np.testing.assert_allclose(jnp.swapaxes(r.jacobian, 1, 2), jac, atol=1e-12)
    r1 = second_order_streams(x[0], layers)
    np.testing.assert_allclose(r1.laplacian, second_order_streams(x, layers).laplacian[0], atol=1e-13)
    vm = jax.vmap(lambda z: second_order_streams(z, layers).laplacian)(x)
    np.testing.assert_allclose(vm, second_order_streams(x, layers).laplacian, atol=1e-13)


def test_refusals():
    layers = _layers(4, 6, 2, seed=24)
    x = jnp.zeros((2, 4))
    with pytest.raises(ValueError, match="layers must be non-empty"):
        second_order_streams(x, [])
    with pytest.raises(ValueError, match="form must be"):
        second_order_streams(x, layers, form="cholesky")
    with pytest.raises(ValueError, match="hess_form must have shape"):
        second_order_streams(x, layers, hess_form=jnp.eye(3))
    with pytest.raises(ValueError, match="diffusion and weights"):
        second_order_streams(x, layers, hess_form=jnp.eye(4), diffusion=jnp.ones(4))
    relu = [(W, b, "relu" if a else None) for W, b, a in layers]
    with pytest.raises(ValueError):
        second_order_streams(x, relu)


# ------------------------------------------------------------------------------------------------ directional jets
def _nested_partial(layers, x, alpha):
    dim = x.shape[1]

    def f(z):
        return _forward(layers, z[None])[0, 0]

    fn = f
    for axis, count in enumerate(alpha):
        e = jnp.zeros(dim).at[axis].set(1.0)
        for _ in range(count):
            fn = (lambda g, e=e: lambda z: jax.jvp(g, (z,), (e,))[1])(fn)
    return jax.vmap(fn)(x)


@pytest.mark.parametrize("act", ["tanh", "sigmoid"])
@pytest.mark.parametrize("dim", [1, 3])
def test_mixed_partials_to_order_four_match_nested_forward_mode(act, dim):
    from omnibias.jax.experimental.streams import partial_derivatives

    layers = _layers(dim, 8, 2, act=act, seed=30 + dim)
    x = jnp.asarray(np.random.default_rng(31).standard_normal((4, dim)))
    alphas = [tuple(int(i == j) * p for j in range(dim)) for i in range(dim) for p in (1, 2, 3, 4)]
    if dim > 1:
        alphas += [(2, 2) + (0,) * (dim - 2), (1, 1) + (0,) * (dim - 2), (1, 0, 2)[:dim] + (0,) * max(0, dim - 3)]
    u, out = partial_derivatives(x, layers, alphas)
    np.testing.assert_allclose(u[:, 0], jax.vmap(lambda z: _forward(layers, z[None])[0, 0])(x), atol=1e-13)
    for r, alpha in enumerate(alphas):
        np.testing.assert_allclose(out[:, r, 0], _nested_partial(layers, x, alpha), atol=1e-10, err_msg=str(alpha))


def test_directional_jet_parameter_gradients_and_unbatched():
    from omnibias.jax.experimental.streams import directional_jet_streams, partial_derivatives

    layers = _layers(2, 8, 2, seed=33)
    x = jnp.asarray(np.random.default_rng(34).standard_normal((3, 2)))
    flat = [(W, b) for W, b, _ in layers]
    acts = [a for _, _, a in layers]

    def ours(p):
        _, d = partial_derivatives(x, [(W, b, a) for (W, b), a in zip(p, acts, strict=True)], [(3, 0), (0, 4)])
        return jnp.sum(d**2)

    def ref(p):
        ls = [(W, b, a) for (W, b), a in zip(p, acts, strict=True)]
        return jnp.sum(_nested_partial(ls, x, (3, 0)) ** 2) + jnp.sum(_nested_partial(ls, x, (0, 4)) ** 2)

    for (a, b), (c, d) in zip(jax.grad(ours)(flat), jax.grad(ref)(flat), strict=True):
        np.testing.assert_allclose(a, c, atol=1e-8, rtol=1e-8)
        np.testing.assert_allclose(b, d, atol=1e-8, rtol=1e-8)
    one = directional_jet_streams(x[0], layers, jnp.eye(2), 3)
    assert one.value.shape == (1,) and one.derivatives.shape == (2, 3, 1)


def test_jet_no_conditional_or_solver_and_refusals():
    from omnibias.jax.experimental.streams import partial_derivatives

    layers = _layers(2, 8, 2, seed=35)
    x = jnp.asarray(np.random.default_rng(36).standard_normal((3, 2)))
    acts = [a for _, _, a in layers]
    params = [(W, b) for W, b, _ in layers]
    fn = lambda p, z: partial_derivatives(z, [(W, b, a) for (W, b), a in zip(p, acts, strict=True)], [(2, 2)])[1]  # noqa: E731
    text = jax.jit(fn).lower(params, x).compile().as_text().lower()
    for forbidden in ("conditional(", "cholesky", "potrf", "geqrf", "callback"):
        assert forbidden not in text
    with pytest.raises(ValueError, match="total derivative order"):
        partial_derivatives(x, layers, [(5, 0)])
    relu = [(W, b, "relu" if a else None) for W, b, a in layers]
    with pytest.raises(ValueError):
        partial_derivatives(x, relu, [(3, 0)])


@pytest.mark.parametrize("dim", [2, 3])
def test_laplacian_gradient_and_bilaplacian_by_jets(dim):
    from omnibias.jax.experimental.streams import bilaplacian_by_jets, laplacian_gradient

    layers = _layers(dim, 8, 2, seed=40 + dim)
    x = jnp.asarray(np.random.default_rng(41).standard_normal((3, dim)))

    def f(z):
        return _forward(layers, z[None])[0, 0]

    lap = lambda z: jnp.trace(jax.hessian(f)(z))  # noqa: E731
    u, g = laplacian_gradient(x, layers)
    np.testing.assert_allclose(g[:, :, 0], jax.vmap(jax.grad(lap))(x), atol=1e-10)
    bil = lambda z: jnp.trace(jax.hessian(lap)(z))  # noqa: E731
    _, b = bilaplacian_by_jets(x, layers)
    np.testing.assert_allclose(b[:, 0], jax.vmap(bil)(x), atol=1e-9)


# ------------------------------------------------------------------------------------------------ performance audit


@pytest.mark.parametrize(("dim", "width"), [(3, 8), (12, 8), (30, 8)])
def test_a_traced_metric_matrix_matches_the_eager_path_and_the_dense_hessian(dim, width):
    """Under ``jit`` a matrix of size ``m <= 2H`` runs as ``m`` coordinate tangents; a larger one keeps the Gram stack."""
    layers = _layers(dim, width, 2, seed=dim)
    x = jnp.asarray(np.random.default_rng(1).standard_normal((5, dim)))
    g = np.random.default_rng(6)
    M = g.standard_normal((dim, dim))
    A = jnp.asarray(M @ M.T - 0.4 * np.trace(M @ M.T) / dim * np.eye(dim))  # indefinite
    traced = jax.jit(lambda a: second_order_streams(x, layers, hess_form=a))(A)
    eager = second_order_streams(x, layers, hess_form=A)
    _, g2, lap = _oracle(layers, x, A)
    for got in (traced, eager):
        np.testing.assert_allclose(got.grad_sq[:, 0], g2, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(got.laplacian[:, 0], lap, rtol=1e-10, atol=1e-10)
    assert traced.jacobian is None  # the traced path reports what it always did: no metric tangents
    jaxpr = str(jax.make_jaxpr(lambda a: second_order_streams(x, layers, hess_form=a).laplacian)(A))
    assert "eigh" not in jaxpr  # no eigendecomposition is traced in either form
    grad = jax.jit(jax.grad(lambda a: jnp.sum(second_order_streams(x, layers, hess_form=a).laplacian)))(A)
    assert np.all(np.isfinite(np.asarray(grad)))


def test_an_accelerator_keeps_the_gram_stack_for_a_traced_metric_above_a_quarter_of_the_width(monkeypatch):
    """The coordinate-tangent route of a traced ``hess_form``: ``m <= 2H`` on a CPU, ``m <= H/4`` elsewhere, ``form="diag"`` always."""
    from omnibias.jax.experimental import streams as st

    assert st._coordinate_metric(192, 96, "auto") and not st._coordinate_metric(193, 96, "auto")
    monkeypatch.setattr(st.jax, "default_backend", lambda: "gpu")
    assert st._coordinate_metric(32, 128, "auto") and not st._coordinate_metric(33, 128, "auto")
    assert not st._coordinate_metric(10, 12, "auto") and not st._coordinate_metric(64, 96, "auto")
    assert st._coordinate_metric(64, 96, "diag")
    # both routes are the same metric: the values of the Gram stack (taken on the "accelerator") are those of the coordinates (on a CPU)
    layers = _layers(6, 8, 2, seed=2)
    x = jnp.asarray(np.random.default_rng(1).standard_normal((5, 6)))
    M = np.random.default_rng(6).standard_normal((6, 6))
    A = jnp.asarray(M @ M.T - 0.4 * np.trace(M @ M.T) / 6 * np.eye(6))
    gram = jax.jit(lambda a: second_order_streams(x, layers, hess_form=a))(A)
    monkeypatch.undo()
    coords = jax.jit(lambda a: second_order_streams(x, layers, hess_form=a))(A)
    np.testing.assert_allclose(gram.grad_sq, coords.grad_sq, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(gram.laplacian, coords.laplacian, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("form", ["diag", "gram"])
def test_remat_keeps_the_values_and_the_gradients(form):
    from omnibias.jax.experimental.streams import directional_jet_streams

    layers = _layers(4, 7, 3, out=2, seed=2)
    x = jnp.asarray(np.random.default_rng(3).standard_normal((6, 4)))
    plain = second_order_streams(x, layers, form=form)
    kept = second_order_streams(x, layers, form=form, remat=True)
    for a, b in zip(jax.tree.leaves(plain), jax.tree.leaves(kept), strict=True):
        np.testing.assert_array_equal(a, b)
    V = jnp.asarray(np.random.default_rng(4).standard_normal((4, 3)))
    for order in (2, 4):
        jets = directional_jet_streams(x, layers, V, order)
        jets_kept = directional_jet_streams(x, layers, V, order, remat=True)
        np.testing.assert_array_equal(jets.derivatives, jets_kept.derivatives)

    def loss(ws, remat, kind):
        ls = [(w, b, s) for w, (_, b, s) in zip(ws, layers, strict=True)]
        if kind == "jets":
            return jnp.sum(directional_jet_streams(x, ls, V, 3, remat=remat).derivatives ** 2)
        r = second_order_streams(x, ls, form=form, remat=remat)
        return jnp.sum(r.laplacian**2) + jnp.sum(r.grad_sq)

    ws = [w for w, _, _ in layers]
    for kind in ("jets", "streams"):
        for a, b in zip(jax.grad(loss)(ws, False, kind), jax.grad(loss)(ws, True, kind), strict=True):
            np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-13)


def test_remat_wraps_each_layer_in_a_checkpoint_and_the_default_does_not():
    layers = _layers(3, 16, 6, seed=1)
    x = jnp.asarray(np.random.default_rng(1).standard_normal((32, 3)))

    def checkpoints(remat):
        def loss(ws):
            ls = [(w, b, s) for w, (_, b, s) in zip(ws, layers, strict=True)]
            return jnp.sum(second_order_streams(x, ls, form="diag", remat=remat).laplacian)

        text = str(jax.make_jaxpr(jax.grad(loss))([w for w, _, _ in layers]))
        return text.count("checkpoint") + text.count("remat")

    assert checkpoints(False) == 0 and checkpoints(True) >= len(layers)


def test_the_identity_block_after_the_first_layer_equals_the_full_product():
    from omnibias.jax.experimental.streams import _after_identity_block

    g = np.random.default_rng(0)
    B, H, n_out, extra = 4, 6, 5, 3
    s1 = jnp.asarray(g.standard_normal((B, H)))
    eye = jnp.einsum("bc,ch->bch", s1, jnp.eye(H))  # channel c is s1[b, c] on unit c: the first layer's identity block
    tail = jnp.asarray(g.standard_normal((B, H + extra + 1, H)))
    head = jnp.asarray(g.standard_normal((B, 1, H)))
    X = jnp.concatenate([head, eye, tail], axis=1)
    W = jnp.asarray(g.standard_normal((n_out, H)))
    np.testing.assert_allclose(
        _after_identity_block(X, W, s1, H), jnp.einsum("bsh,oh->bso", X, W), rtol=1e-14, atol=1e-14
    )


def test_identity_block_is_off_by_default_and_agrees_with_the_default_when_asked_for():
    layers = _layers(3, 16, 6, seed=2)
    x = jnp.asarray(np.random.default_rng(2).standard_normal((8, 3)))
    plain = second_order_streams(x, layers, form="gram")
    asked = second_order_streams(x, layers, form="gram", identity_block=True)
    for a, b in zip((plain.value, plain.grad_sq, plain.laplacian), (asked.value, asked.grad_sq, asked.laplacian), strict=True):
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-13)

    def einsums(**kw):
        return str(jax.make_jaxpr(lambda xx: second_order_streams(xx, layers, form="gram", **kw).laplacian)(x)).count("dot_general")

    assert einsums(identity_block=True) != einsums()  # the opt-in program differs; the default is the unsplit product


# ---------------------------------------------------------------------------------------------------- the split layout
def _same(a, b):
    for name in ("value", "grad_sq", "laplacian", "directional", "jacobian"):
        u, v = getattr(a, name), getattr(b, name)
        assert (u is None) == (v is None), name
        if u is not None:
            np.testing.assert_allclose(u, v, rtol=1e-12, atol=1e-12, err_msg=name)


@pytest.mark.parametrize("act", ["tanh", "sigmoid"])
@pytest.mark.parametrize(("dim", "width", "depth", "out"), [(2, 8, 2, 1), (5, 8, 3, 3), (12, 8, 1, 1)])
def test_split_layout_matches_the_stack_for_every_diagonal_option(act, dim, width, depth, out):
    layers = _layers(dim, width, depth, out=out, act=act, seed=dim + 11)
    x = jnp.asarray(np.random.default_rng(5).standard_normal((4, dim)))
    g = np.random.default_rng(6)
    s_vec = jnp.asarray(0.5 + g.random(dim))
    s_mat = jnp.asarray(g.standard_normal((dim, 3)))
    w = jnp.asarray(np.where(np.arange(dim) % 2 == 0, 1.0, -1.0) * (0.5 + g.random(dim)))
    directions = jnp.asarray(g.standard_normal((dim, 2)))
    options = [
        {},
        {"full_jacobian": True},
        {"diffusion": s_vec},
        {"diffusion": s_mat, "directions": directions},
        {"weights": w, "form": "diag"},
        {"coordinates": [0, dim - 1], "directions": directions},
        {"point_diffusion": jnp.asarray(0.5 + g.random((4, dim))), "form": "diag"},
        {"form": "gram"},
        {"form": "gram", "directions": directions},
        {"point_diffusion": jnp.asarray(0.5 + g.random((4, dim))), "form": "gram"},
    ]
    for kw in options:
        _same(second_order_streams(x, layers, layout="stack", **kw), second_order_streams(x, layers, layout="split", **kw))


def test_split_layout_matches_the_stack_for_a_metric_matrix_traced_or_eager_and_for_skip_blocks():
    from omnibias.jax.experimental._chain import Skip

    dim, width = 6, 8
    layers = _layers(dim, width, 3, seed=9)
    layers[2] = (layers[2][0], layers[2][1], Skip(layers[2][2]))
    x = jnp.asarray(np.random.default_rng(7).standard_normal((3, dim)))
    A = np.random.default_rng(8).standard_normal((dim, dim))
    A = jnp.asarray(0.5 * (A + A.T))
    _same(second_order_streams(x, layers, hess_form=A, layout="stack"), second_order_streams(x, layers, hess_form=A, layout="split"))
    traced = {lay: jax.jit(lambda a, lay=lay: second_order_streams(x, layers, hess_form=a, layout=lay))(A) for lay in ("stack", "split")}
    _same(traced["stack"], traced["split"])
    # the Gram form, and a stack that keeps the identity block
    _same(second_order_streams(x, layers, form="gram"), second_order_streams(x, layers, form="gram", layout="split"))
    _same(
        second_order_streams(x, layers, form="gram", identity_block=True),
        second_order_streams(x, layers, form="gram", identity_block=True, layout="split"),
    )
    # unbatched points, remat and parameter gradients to second order
    _same(second_order_streams(x[0], layers), second_order_streams(x[0], layers, layout="split"))
    _same(second_order_streams(x, layers, layout="split", remat=True), second_order_streams(x, layers, layout="stack"))

    def loss(params, lay):
        ls = [(p[0], p[1], layers[i][2]) for i, p in enumerate(params)]
        return jnp.sum(second_order_streams(x, ls, layout=lay).laplacian ** 2)

    params = [(W, b) for W, b, _ in layers]
    gs = jax.grad(loss)(params, "stack")
    gp = jax.grad(loss)(params, "split")
    for (a, b), (c, d) in zip(gs, gp, strict=True):
        np.testing.assert_allclose(a, c, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(b, d, rtol=1e-10, atol=1e-12)
    hs = jax.hessian(lambda W0: loss([(W0, params[0][1])] + params[1:], "stack"))(params[0][0])
    hp = jax.hessian(lambda W0: loss([(W0, params[0][1])] + params[1:], "split"))(params[0][0])
    np.testing.assert_allclose(hs, hp, rtol=1e-9, atol=1e-11)


def test_split_layout_refuses_an_unknown_layout_and_its_compiled_program_has_no_concatenate():
    layers = _layers(4, 8, 2, seed=2)
    x = jnp.asarray(np.random.default_rng(1).standard_normal((3, 4)))
    with pytest.raises(ValueError, match="layout must be"):
        second_order_streams(x, layers, layout="rows")
    for lay, expected in (("stack", True), ("split", False)):
        text = jax.jit(lambda z, lay=lay: second_order_streams(z, layers, layout=lay).laplacian).lower(x).compile().as_text()
        assert (" concatenate(" in text) is expected, lay


def test_auto_layout_is_split_on_a_cpu_and_in_training_but_stack_on_an_accelerator(monkeypatch):
    from omnibias.jax.experimental import streams

    assert streams.DEFAULT_LAYOUT == "auto" and streams.TRAINING_LAYOUT == "split"
    monkeypatch.setattr(streams.jax, "default_backend", lambda: "cpu")
    assert streams.resolve_layout() == "split"  # a CPU takes the split layout in every pass
    monkeypatch.setattr(streams.jax, "default_backend", lambda: "gpu")
    assert streams.AUTO_SPLIT_MAX_HIDDEN == 2
    assert streams.resolve_layout() == "stack"  # an accelerator keeps the stack for forward-only calls of unknown depth
    assert streams.resolve_layout(hidden=2) == "stack"  # the dtype is not known
    assert streams.resolve_layout(hidden=2, dtype=jnp.float64) == "split"  # float64 takes the split layout for two hidden layers
    assert streams.resolve_layout(hidden=3, dtype=jnp.float64) == "stack"
    assert streams.resolve_layout(hidden=2, dtype=jnp.float32) == "stack"  # float32 keeps the stack
    assert streams.resolve_layout("stack", hidden=1, dtype=jnp.float64) == "stack"  # an explicit layout wins
    monkeypatch.setattr(streams, "AUTO_SPLIT_MAX_HIDDEN", 0)
    assert streams.resolve_layout(hidden=1, dtype=jnp.float64) == "stack"  # zero keeps the stack on an accelerator
    monkeypatch.setattr(streams, "AUTO_SPLIT_MAX_HIDDEN", 2)
    with streams.training_layout():
        assert streams.resolve_layout() == "split"  # and takes the split layout while a training program is traced
        assert streams.resolve_layout("stack") == "stack"  # an explicit layout wins
        assert streams.resolve_layout("split") == "split"
        with pytest.raises(ValueError, match="layout must be"):
            streams.resolve_layout("rows")
    assert streams.resolve_layout() == "stack"  # the context is left again
    monkeypatch.setattr(streams, "DEFAULT_LAYOUT", "split")
    assert streams.resolve_layout() == "split"
    monkeypatch.setattr(streams, "DEFAULT_LAYOUT", "stack")
    monkeypatch.setattr(streams.jax, "default_backend", lambda: "cpu")
    with streams.training_layout():
        assert streams.resolve_layout() == "stack"  # a forced default is not overridden


def test_second_order_streams_takes_the_training_layout_under_the_context(monkeypatch):
    from omnibias.jax.experimental import streams

    monkeypatch.setattr(streams.jax, "default_backend", lambda: "gpu")  # the program is still compiled for the host
    layers = _layers(4, 8, 3, seed=3)  # three hidden layers: a forward-only call on an accelerator keeps the stack
    x = jnp.asarray(np.random.default_rng(2).standard_normal((3, 4)))

    def text(ctx):
        def f(z):
            return second_order_streams(z, layers).laplacian

        with ctx:
            return jax.jit(f).lower(x).compile().as_text()

    import contextlib

    assert " concatenate(" in text(contextlib.nullcontext())
    assert " concatenate(" not in text(streams.training_layout())


def test_training_layout_changes_the_program_but_not_the_values(monkeypatch):
    from omnibias.jax.experimental import streams

    monkeypatch.setattr(streams.jax, "default_backend", lambda: "gpu")  # outside training an accelerator takes the stack (deep nets)
    layers = _layers(4, 8, 3, seed=3)
    x = jnp.asarray(np.random.default_rng(2).standard_normal((5, 4)))

    def lap(z):
        return second_order_streams(z, layers).laplacian

    from omnibias.jax.experimental.streams import training_layout

    plain = jax.jit(lap).lower(x).compile().as_text()
    with training_layout():
        trained = jax.jit(lambda z: second_order_streams(z, layers).laplacian).lower(x).compile().as_text()
        value = jax.jit(lambda z: second_order_streams(z, layers).laplacian)(x)
    assert " concatenate(" in plain and " concatenate(" not in trained
    np.testing.assert_allclose(value, lap(x), rtol=1e-10, atol=1e-12)


# ------------------------------------------------------------------------------- the split layout of directional jets
@pytest.mark.parametrize("order", [1, 2, 3, 4])
@pytest.mark.parametrize("directions_count", [1, 3])
@pytest.mark.parametrize("act", ["tanh", "sigmoid"])
def test_split_jets_match_the_stack_and_the_nested_directional_derivatives(order, directions_count, act):
    from omnibias.jax.experimental.streams import directional_jet_streams

    layers = _layers(5, 7, 3, out=2, act=act, seed=order + 20)
    x = jnp.asarray(np.random.default_rng(order).standard_normal((4, 5)))
    V = jnp.asarray(np.random.default_rng(9).standard_normal((5, directions_count)))
    stack = directional_jet_streams(x, layers, V, order, layout="stack")
    split = directional_jet_streams(x, layers, V, order, layout="split")
    np.testing.assert_allclose(split.value, stack.value, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(split.derivatives, stack.derivatives, rtol=1e-11, atol=1e-12)
    assert split.derivatives.shape == (4, directions_count, order, 2)

    def nested(z, v, p):
        f = lambda s: _forward(layers, (z + s * v)[None])[0]  # noqa: E731
        g = f
        for _ in range(p):
            g = (lambda h: lambda s: jax.jvp(h, (s,), (jnp.ones_like(s),))[1])(g)
        return g(jnp.zeros(()))

    for k in range(directions_count):
        want = jax.vmap(lambda z, k=k: nested(z, V[:, k], order))(x)
        np.testing.assert_allclose(split.derivatives[:, k, order - 1], want, rtol=1e-9, atol=1e-10)


def test_split_jets_with_a_skip_layer_unbatched_input_parameter_gradients_and_remat():
    from omnibias.jax.experimental._chain import Skip
    from omnibias.jax.experimental.streams import directional_jet_streams

    layers = _layers(4, 6, 3, seed=40)
    layers[2] = (layers[2][0], layers[2][1], Skip(layers[2][2]))
    x = jnp.asarray(np.random.default_rng(41).standard_normal((5, 4)))
    V = jnp.asarray(np.random.default_rng(42).standard_normal((4, 2)))
    same = lambda a, b: np.testing.assert_allclose(a, b, rtol=1e-11, atol=1e-12)  # noqa: E731
    same(
        directional_jet_streams(x, layers, V, 3, layout="split").derivatives,
        directional_jet_streams(x, layers, V, 3, layout="stack").derivatives,
    )
    one = directional_jet_streams(x[0], layers, V, 3, layout="split")
    assert one.value.shape == (1,) and one.derivatives.shape == (2, 3, 1)
    same(one.derivatives, directional_jet_streams(x[0], layers, V, 3, layout="stack").derivatives)
    ws = [w for w, _, _ in layers]

    def loss(ws, layout, remat=False):
        ls = [(w, b, s) for w, (_, b, s) in zip(ws, layers, strict=True)]
        return jnp.sum(directional_jet_streams(x, ls, V, 3, layout=layout, remat=remat).derivatives ** 2)

    for a, b, c in zip(
        jax.grad(loss)(ws, "stack"), jax.grad(loss)(ws, "split"), jax.grad(loss)(ws, "split", True), strict=True
    ):
        same(a, b)
        same(a, c)


def test_jet_layout_auto_takes_the_split_layout_on_a_cpu_from_the_threshold_and_refuses_unknown_layouts(monkeypatch):
    from omnibias.jax.experimental import streams

    assert streams.JET_LAYOUT == "auto"
    below, at = streams.JET_SPLIT_MIN_POINTS - 1, streams.JET_SPLIT_MIN_POINTS
    monkeypatch.setattr(streams.jax, "default_backend", lambda: "cpu")
    assert streams._resolve_jet_layout("auto", below) == "stack"
    assert streams._resolve_jet_layout("auto", at) == "split"
    assert streams._resolve_jet_layout("stack", at) == "stack" and streams._resolve_jet_layout("split", 1) == "split"
    monkeypatch.setattr(streams.jax, "default_backend", lambda: "gpu")
    assert streams._resolve_jet_layout("auto", 10 * at) == "stack"  # an accelerator keeps the stack
    with streams.training_layout():
        assert streams._resolve_jet_layout("auto", 10 * at) == "stack"
    with pytest.raises(ValueError, match="layout must be"):
        streams._resolve_jet_layout("rows", at)
    layers = _layers(3, 5, 2, seed=1)
    with pytest.raises(ValueError, match="layout must be"):
        streams.directional_jet_streams(jnp.zeros((2, 3)), layers, jnp.eye(3), 2, layout="rows")


def test_the_default_jet_program_follows_the_layout_rule():
    from omnibias.jax.experimental.streams import directional_jet_streams

    layers = _layers(4, 6, 2, seed=7)
    V = jnp.eye(4)[:, :2]

    def concatenates(points, layout=None):
        x = jnp.ones((points, 4))
        f = lambda z: directional_jet_streams(z, layers, V, 3, layout=layout).derivatives  # noqa: E731
        return jax.jit(f).lower(x).compile().as_text().count(" concatenate(")

    assert concatenates(256, "split") < concatenates(256, "stack")  # the split layout drops the stack's copies
    expected = "split" if jax.default_backend() == "cpu" else "stack"
    assert concatenates(256) == concatenates(256, expected)  # from the threshold on
    assert concatenates(32) == concatenates(32, "stack")  # below it every backend keeps the stack


def test_auto_layout_takes_the_split_layout_while_the_weights_are_differentiated_on_an_accelerator(monkeypatch):
    """On an A100 the split layout took 0.5 to 0.9 of the stack's time in the value-and-gradient pass and the stack stayed ahead in
    a forward-only call; a call traced while ``jax.grad`` differentiates its weights (or points) is a gradient pass."""
    from omnibias.jax.experimental import streams

    monkeypatch.setattr(streams.jax, "default_backend", lambda: "gpu")
    seen: list[str] = []
    real = streams.resolve_layout

    def spy(layout=None, *arrays, hidden=None, dtype=None):
        out = real(layout, *arrays, hidden=hidden, dtype=dtype)
        seen.append(out)
        return out

    monkeypatch.setattr(streams, "resolve_layout", spy)
    layers = _layers(4, 6, 3)
    x = jnp.asarray(np.random.default_rng(3).standard_normal((5, 4)))
    second_order_streams(x, layers)
    assert seen == ["stack"]  # a forward call of three hidden layers keeps the stack
    seen.clear()
    second_order_streams(x, _layers(4, 6, 2))
    assert seen == ["split"]  # and one of two hidden layers takes the split layout

    def loss(ws, z):
        ls = [(w, b, s) for w, (_, b, s) in zip(ws, layers, strict=True)]
        r = second_order_streams(z, ls)
        return jnp.sum(r.laplacian) + jnp.sum(r.grad_sq)

    ws = [w for w, _, _ in layers]
    for transform in (jax.grad, lambda f: jax.jit(jax.grad(f)), jax.value_and_grad):
        seen.clear()
        transform(loss)(ws, x)
        assert seen == ["split"], transform
    seen.clear()
    jax.grad(jax.jit(loss))(ws, x)  # traced before it is differentiated: the call sees no differentiation
    assert seen == ["stack"]
    seen.clear()
    jax.grad(loss, argnums=1)(ws, x)  # the points differentiated: also a gradient pass
    assert seen == ["split"]
    # the two layouts agree under the gradient
    monkeypatch.setattr(streams, "resolve_layout", real)
    g_split = jax.grad(loss)(ws, x)
    g_stack = jax.grad(lambda w, z: loss(w, z) + 0 * jnp.sum(second_order_streams(z, [(a, b, s) for a, (_, b, s) in zip(w, layers, strict=True)], layout="stack").laplacian))(ws, x)
    for a, b in zip(g_split, g_stack, strict=True):
        np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-12)


def test_a_shallow_float64_forward_call_on_an_accelerator_takes_the_split_layout_and_the_values_agree(monkeypatch):
    from omnibias.jax.experimental import streams

    monkeypatch.setattr(streams.jax, "default_backend", lambda: "gpu")
    x = jnp.asarray(np.random.default_rng(5).standard_normal((4, 3)))
    for depth, expect_concat in ((1, False), (2, False), (3, True)):
        layers = _layers(3, 8, depth, seed=depth)
        text = jax.jit(lambda z, layers=layers: second_order_streams(z, layers).laplacian).lower(x).compile().as_text()
        assert (" concatenate(" in text) is expect_concat, depth
    layers = _layers(3, 8, 2, seed=2)
    x32 = x.astype(jnp.float32)
    layers32 = [(W.astype(jnp.float32), b.astype(jnp.float32), s) for W, b, s in layers]
    text = jax.jit(lambda z: second_order_streams(z, layers32).laplacian).lower(x32).compile().as_text()
    assert " concatenate(" in text  # float32 keeps the stack
    auto = second_order_streams(x, layers)
    stack = second_order_streams(x, layers, layout="stack")
    np.testing.assert_allclose(auto.laplacian, stack.laplacian, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(auto.grad_sq, stack.grad_sq, rtol=1e-10, atol=1e-12)
