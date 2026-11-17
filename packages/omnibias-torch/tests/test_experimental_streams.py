# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""The stacked stream engine (PyTorch) against the dense Hessian, and its host-sync freedom."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from omnibias.torch.experimental.streams import second_order_streams


def _layers(dim, width, depth, out=1, act="tanh", seed=0, requires_grad=False):
    g = np.random.default_rng(seed)
    sizes = [dim] + [width] * depth + [out]
    layers = []
    for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:], strict=True)):
        W = torch.tensor(g.standard_normal((b, a)) / np.sqrt(a), requires_grad=requires_grad)
        bias = torch.tensor(0.3 * g.standard_normal(b), requires_grad=requires_grad)
        layers.append((W, bias, act if i < depth else None))
    return layers


def _forward(layers, z):
    a = z
    for W, b, act in layers:
        a = a @ W.T + b
        if act is not None:
            a = {"tanh": torch.tanh, "sigmoid": torch.sigmoid, "softplus": torch.nn.functional.softplus}[act](a)
    return a


def _oracle(layers, x, A):
    f = lambda z: _forward(layers, z[None])[0, 0]  # noqa: E731
    u = torch.func.vmap(f)(x)
    g = torch.func.vmap(torch.func.grad(f))(x)
    h = torch.func.vmap(torch.func.hessian(f))(x)
    return u, torch.einsum("bi,ij,bj->b", g, A, g), torch.einsum("ij,bij->b", A, h)


@pytest.mark.parametrize("act", ["tanh", "sigmoid"])
@pytest.mark.parametrize(("dim", "width", "depth"), [(2, 8, 2), (40, 8, 2)])
@pytest.mark.parametrize("form", ["auto", "diag", "gram"])
def test_identity_metric_matches_the_dense_hessian(act, dim, width, depth, form):
    layers = _layers(dim, width, depth, act=act, seed=dim)
    x = torch.tensor(np.random.default_rng(1).standard_normal((6, dim)))
    r = second_order_streams(x, layers, form=form)
    u, g2, lap = _oracle(layers, x, torch.eye(dim, dtype=torch.float64))
    torch.testing.assert_close(r.value[:, 0], u, atol=1e-12, rtol=0)
    torch.testing.assert_close(r.grad_sq[:, 0], g2, atol=1e-11, rtol=0)
    torch.testing.assert_close(r.laplacian[:, 0], lap, atol=1e-11, rtol=0)


@pytest.mark.parametrize("form", ["diag", "gram"])
def test_indefinite_metric(form):
    dim = 9
    layers = _layers(dim, 6, 2, seed=3)
    x = torch.tensor(np.random.default_rng(2).standard_normal((5, dim)))
    g = np.random.default_rng(4)
    Q, _ = np.linalg.qr(g.standard_normal((dim, dim)))
    lam = np.where(np.arange(dim) % 2 == 0, 1.0, -1.0) * (0.5 + g.random(dim))
    A = torch.tensor((Q * lam) @ Q.T)
    r = second_order_streams(x, layers, hess_form=A, form=form)
    _, g2, lap = _oracle(layers, x, A)
    torch.testing.assert_close(r.grad_sq[:, 0], g2, atol=1e-10, rtol=0)
    torch.testing.assert_close(r.laplacian[:, 0], lap, atol=1e-10, rtol=0)


def test_drift_diffusion_and_coordinates():
    dim = 7
    layers = _layers(dim, 6, 2, seed=8)
    x = torch.tensor(np.random.default_rng(9).standard_normal((5, dim)))
    idx = np.array([1, 3, 4])
    S = torch.tensor(np.random.default_rng(10).standard_normal((3, 2)))
    A = torch.zeros(dim, dim, dtype=torch.float64)
    A[np.ix_(idx, idx)] = S @ S.T
    r = second_order_streams(x, layers, diffusion=S, coordinates=idx)
    _, g2, lap = _oracle(layers, x, A)
    torch.testing.assert_close(r.grad_sq[:, 0], g2, atol=1e-11, rtol=0)
    torch.testing.assert_close(r.laplacian[:, 0], lap, atol=1e-11, rtol=0)
    v = torch.tensor(np.random.default_rng(13).standard_normal((dim, 2)))
    r = second_order_streams(x, layers, directions=v)
    f = lambda z: _forward(layers, z[None])[0, 0]  # noqa: E731
    grads = torch.func.vmap(torch.func.grad(f))(x)
    torch.testing.assert_close(r.directional[:, :, 0], grads @ v, atol=1e-12, rtol=0)


@pytest.mark.parametrize("dim", [4, 40])
def test_parameter_gradients_to_second_order(dim):
    layers = _layers(dim, 8, 2, seed=15, requires_grad=True)
    x = torch.tensor(np.random.default_rng(16).standard_normal((5, dim)))
    params = [t for W, b, _ in layers for t in (W, b)]

    def ours():
        r = second_order_streams(x, layers)
        return (r.laplacian**2).sum() + r.grad_sq.sum()

    def ref():
        _, g2, lap = _oracle(layers, x, torch.eye(dim, dtype=torch.float64))
        return (lap**2).sum() + g2.sum()

    kw = {"allow_unused": True, "materialize_grads": True}
    g1 = torch.autograd.grad(ours(), params, create_graph=True, **kw)
    g2 = torch.autograd.grad(ref(), params, create_graph=True, **kw)
    for a, b in zip(g1, g2, strict=True):
        torch.testing.assert_close(a, b, atol=1e-9, rtol=0)
    t = [torch.tensor(np.random.default_rng(17).standard_normal(tuple(p.shape))) for p in params]
    h1 = torch.autograd.grad(sum((a * c).sum() for a, c in zip(g1, t, strict=True)), params, **kw)
    h2 = torch.autograd.grad(sum((a * c).sum() for a, c in zip(g2, t, strict=True)), params, **kw)
    for a, b in zip(h1, h2, strict=True):
        torch.testing.assert_close(a, b, atol=1e-8, rtol=0)


def test_no_data_dependent_control_flow():
    """Meta tensors have no data: a ``bool(tensor)`` / ``.item()`` (a device-to-host sync) would raise here."""
    layers = [(W.to("meta"), b.to("meta"), a) for W, b, a in _layers(64, 8, 2, seed=1)]
    x = torch.empty(4, 64, device="meta", dtype=torch.float64)
    for form in ("auto", "diag", "gram"):
        r = second_order_streams(x, layers, form=form)
        assert r.laplacian.shape == (4, 1) and r.laplacian.device.type == "meta"


def test_refusals():
    layers = _layers(4, 6, 2, seed=24)
    x = torch.zeros(2, 4, dtype=torch.float64)
    with pytest.raises(ValueError, match="layers must be non-empty"):
        second_order_streams(x, [])
    with pytest.raises(ValueError, match="form must be"):
        second_order_streams(x, layers, form="cholesky")
    with pytest.raises(ValueError, match="hess_form must have shape"):
        second_order_streams(x, layers, hess_form=torch.eye(3, dtype=torch.float64))


def _nested_partial(layers, x, alpha):
    dim = x.shape[1]

    def f(z):
        return _forward(layers, z[None])[0, 0]

    fn = f
    for axis, count in enumerate(alpha):
        e = torch.zeros(dim, dtype=torch.float64)
        e[axis] = 1.0
        for _ in range(count):
            fn = (lambda g, e=e: lambda z: torch.func.jvp(g, (z,), (e,))[1])(fn)
    return torch.func.vmap(fn)(x)


@pytest.mark.parametrize("dim", [1, 3])
def test_mixed_partials_to_order_four(dim):
    from omnibias.torch.experimental.streams import partial_derivatives

    layers = _layers(dim, 8, 2, seed=30 + dim)
    x = torch.tensor(np.random.default_rng(31).standard_normal((4, dim)))
    alphas = [tuple(int(i == j) * p for j in range(dim)) for i in range(dim) for p in (1, 2, 3, 4)]
    if dim > 1:
        alphas += [(2, 2) + (0,) * (dim - 2), (1, 1) + (0,) * (dim - 2)]
    u, out = partial_derivatives(x, layers, alphas)
    for r, alpha in enumerate(alphas):
        torch.testing.assert_close(out[:, r, 0], _nested_partial(layers, x, alpha), atol=1e-10, rtol=0)


def test_directional_jet_meta_and_gradients():
    from omnibias.torch.experimental.streams import partial_derivatives

    layers = [(W.to("meta"), b.to("meta"), a) for W, b, a in _layers(2, 8, 2, seed=1)]
    u, d = partial_derivatives(torch.empty(3, 2, device="meta", dtype=torch.float64), layers, [(2, 2), (3, 0)])
    assert d.shape == (3, 2, 1)
    layers = _layers(2, 8, 2, seed=33, requires_grad=True)
    x = torch.tensor(np.random.default_rng(34).standard_normal((3, 2)))
    params = [t for W, b, _ in layers for t in (W, b)]
    _, d = partial_derivatives(x, layers, [(3, 0), (0, 4)])
    got = torch.autograd.grad((d**2).sum(), params, allow_unused=True, materialize_grads=True)
    ref = torch.autograd.grad(
        (_nested_partial(layers, x, (3, 0)) ** 2).sum() + (_nested_partial(layers, x, (0, 4)) ** 2).sum(),
        params, allow_unused=True, materialize_grads=True,
    )
    for a, b in zip(got, ref, strict=True):
        torch.testing.assert_close(a, b, atol=1e-8, rtol=1e-8)


@pytest.mark.parametrize("dim", [2, 3])
def test_laplacian_gradient_and_bilaplacian_by_jets(dim):
    from omnibias.torch.experimental.streams import bilaplacian_by_jets, laplacian_gradient

    layers = _layers(dim, 8, 2, seed=40 + dim)
    x = torch.tensor(np.random.default_rng(41).standard_normal((3, dim)))

    def f(z):
        return _forward(layers, z[None])[0, 0]

    lap = lambda z: torch.trace(torch.func.hessian(f)(z))  # noqa: E731
    _, g = laplacian_gradient(x, layers)
    torch.testing.assert_close(g[:, :, 0], torch.func.vmap(torch.func.grad(lap))(x), atol=1e-10, rtol=0)
    bil = lambda z: torch.trace(torch.func.hessian(lap)(z))  # noqa: E731
    _, b = bilaplacian_by_jets(x, layers)
    torch.testing.assert_close(b[:, 0], torch.func.vmap(bil)(x), atol=1e-9, rtol=0)


# -- the split layout: the value, the tangents and the second-order stream in their own tensors ------------------------


def _both(x, layers, **kw):
    return second_order_streams(x, layers, layout="stack", **kw), second_order_streams(x, layers, layout="split", **kw)


@pytest.mark.parametrize("act", ["tanh", "sigmoid"])
@pytest.mark.parametrize(("dim", "width", "depth"), [(2, 8, 2), (5, 6, 1), (3, 7, 3), (40, 8, 2)])
def test_split_layout_matches_the_stack_and_the_dense_hessian(act, dim, width, depth):
    layers = _layers(dim, width, depth, act=act, seed=3)
    x = torch.tensor(np.random.default_rng(4).standard_normal((6, dim)))
    a, b = _both(x, layers)
    for p, q in zip((a.value, a.grad_sq, a.laplacian), (b.value, b.grad_sq, b.laplacian), strict=True):
        torch.testing.assert_close(p, q, atol=1e-12, rtol=0)
    if a.jacobian is not None:
        torch.testing.assert_close(a.jacobian, b.jacobian, atol=1e-12, rtol=0)
    assert b.directional is None
    u, g2, lap = _oracle(layers, x, torch.eye(dim, dtype=torch.float64))
    torch.testing.assert_close(b.laplacian[:, 0], lap, atol=1e-9, rtol=0)
    torch.testing.assert_close(b.grad_sq[:, 0], g2, atol=1e-9, rtol=0)


def test_split_layout_with_a_metric_weights_a_selection_the_full_jacobian_and_an_unbatched_point():
    dim = 4
    layers = _layers(dim, 6, 2, seed=5)
    x = torch.tensor(np.random.default_rng(6).standard_normal((3, dim)))
    S = torch.tensor(np.random.default_rng(7).standard_normal((dim, 3)))
    cases = [
        dict(diffusion=S),
        dict(coordinates=[0, 2]),
        dict(full_jacobian=True),
        dict(diffusion=S, weights=torch.tensor([1.0, -1.0, 0.5], dtype=torch.float64)),
        dict(point_diffusion=torch.tensor(np.random.default_rng(8).standard_normal((3, dim)))),
    ]
    for kw in cases:
        a, b = _both(x, layers, **kw)
        for p, q in zip(a[:3], b[:3], strict=True):
            torch.testing.assert_close(p, q, atol=1e-12, rtol=0)
        if a.jacobian is not None:
            torch.testing.assert_close(a.jacobian, b.jacobian, atol=1e-12, rtol=0)
    a, b = _both(x[0], layers)
    assert b.laplacian.shape == a.laplacian.shape == (1,)
    torch.testing.assert_close(a.laplacian, b.laplacian, atol=1e-12, rtol=0)


def test_split_layout_falls_back_to_the_stack_for_drift_directions_and_the_gram_form():
    dim = 4
    layers = _layers(dim, 6, 2, seed=9)
    x = torch.tensor(np.random.default_rng(10).standard_normal((3, dim)))
    v = torch.tensor(np.random.default_rng(11).standard_normal((dim, 2)))
    a, b = _both(x, layers, directions=v)
    torch.testing.assert_close(a.directional, b.directional, atol=0, rtol=0)
    a, b = _both(x, layers, form="gram")
    torch.testing.assert_close(a.laplacian, b.laplacian, atol=0, rtol=0)


def test_split_layout_parameter_gradients_to_second_order_and_skip_blocks():
    from omnibias.torch.experimental._chain import Skip

    layers = _layers(4, 8, 3, seed=12, requires_grad=True)
    W, b, act = layers[2]
    layers[2] = (W, b, Skip(act)) if not isinstance(act, Skip) else layers[2]
    x = torch.tensor(np.random.default_rng(13).standard_normal((5, 4)))
    params = [t for W, b, _ in layers for t in (W, b)]

    def objective(layout):
        r = second_order_streams(x, layers, layout=layout)
        return (r.laplacian**2).sum() + r.grad_sq.sum()

    kw = {"allow_unused": True, "materialize_grads": True}
    g1 = torch.autograd.grad(objective("stack"), params, create_graph=True, **kw)
    g2 = torch.autograd.grad(objective("split"), params, create_graph=True, **kw)
    for p, q in zip(g1, g2, strict=True):
        torch.testing.assert_close(p, q, atol=1e-10, rtol=0)
    t = [torch.tensor(np.random.default_rng(14).standard_normal(tuple(p.shape))) for p in params]
    h1 = torch.autograd.grad(sum((p * c).sum() for p, c in zip(g1, t, strict=True)), params, **kw)
    h2 = torch.autograd.grad(sum((p * c).sum() for p, c in zip(g2, t, strict=True)), params, **kw)
    for p, q in zip(h1, h2, strict=True):
        torch.testing.assert_close(p, q, atol=1e-9, rtol=0)


def test_stream_layout_refuses_an_unknown_name_and_the_default_is_the_module_setting(monkeypatch):
    from omnibias.torch.experimental import streams

    layers = _layers(3, 5, 2, seed=15)
    x = torch.tensor(np.random.default_rng(16).standard_normal((2, 3)))
    with pytest.raises(ValueError, match="layout must be"):
        second_order_streams(x, layers, layout="rows")
    ref = second_order_streams(x, layers, layout="stack")
    for name in ("stack", "split", "auto"):
        monkeypatch.setattr(streams, "STREAM_LAYOUT", name)
        torch.testing.assert_close(second_order_streams(x, layers).laplacian, ref.laplacian, atol=1e-12, rtol=0)


def test_auto_layout_is_split_when_called_eagerly_and_stack_under_cuda_graph_capture_or_compile(monkeypatch):
    """Replayed from a CUDA graph the host dispatch is gone and the split layout's extra kernels show (1.04 to 1.19 of the stack's
    time at small batches on an A100); called eagerly the split layout issues fewer operations and is faster."""
    from omnibias.torch.experimental import streams

    assert streams.STREAM_LAYOUT == "auto"
    layers = _layers(3, 5, 2, seed=17)
    x = torch.tensor(np.random.default_rng(18).standard_normal((4, 3)))
    ref = second_order_streams(x, layers, layout="stack")
    used: list[str] = []
    real_split, real_resolve = streams._split_loop, streams._resolve_layout
    monkeypatch.setattr(streams, "_split_loop", lambda *a, **k: (used.append("split"), real_split(*a, **k))[1])
    monkeypatch.setattr(streams, "_resolve_layout", lambda layout, a, differentiated=False: (used.append(real_resolve(layout, a, differentiated)), used[-1])[1])
    out = second_order_streams(x, layers)
    assert used[0] == "split" and "split" in used[1:]
    torch.testing.assert_close(out.laplacian, ref.laplacian, atol=1e-12, rtol=0)
    used.clear()
    monkeypatch.setattr(streams, "_capturing", lambda a: True)
    out = second_order_streams(x, layers)
    assert used == ["stack"]  # a captured call keeps the stack
    torch.testing.assert_close(out.laplacian, ref.laplacian, atol=1e-12, rtol=0)
    used.clear()
    monkeypatch.setattr(streams, "_capturing", lambda a: False)
    monkeypatch.setattr(streams.torch.compiler, "is_compiling", lambda: True)
    second_order_streams(x, layers)
    assert used == ["stack"]


def test_a_captured_call_that_a_gradient_flows_through_takes_the_split_layout(monkeypatch):
    """Replayed from a CUDA graph the backward pass of the stack's concatenations and slices cost far more kernel time than the
    split layout's (a value-and-gradient replay took 0.58 to 0.75 of the stack's time at 2,048 points on an A100), so a captured
    call takes the split layout when grad mode is on and a weight, a bias or the points require grad; a forward-only call keeps
    the stack."""
    from omnibias.torch.experimental import streams

    layers = _layers(3, 5, 2, seed=19)
    x = torch.tensor(np.random.default_rng(20).standard_normal((4, 3)))
    monkeypatch.setattr(streams, "_capturing", lambda a: True)
    used: list[str] = []
    real = streams._resolve_layout
    monkeypatch.setattr(streams, "_resolve_layout", lambda layout, a, differentiated=False: (used.append(real(layout, a, differentiated)), used[-1])[1])
    second_order_streams(x, layers)
    assert used == ["stack"]  # nothing requires grad
    trainable = [(W.clone().requires_grad_(True), b, s) for W, b, s in layers]
    used.clear()
    second_order_streams(x, trainable)
    assert used == ["split"]
    used.clear()
    with torch.no_grad():
        second_order_streams(x, trainable)
    assert used == ["stack"]  # grad mode off: forward-only
    used.clear()
    second_order_streams(x.clone().requires_grad_(True), layers)
    assert used == ["split"]  # the points are differentiated
    monkeypatch.setattr(streams, "_resolve_layout", real)
    ref = second_order_streams(x, layers, layout="stack")
    torch.testing.assert_close(second_order_streams(x, trainable).laplacian, ref.laplacian, atol=1e-12, rtol=0)


def test_split_layout_keeps_every_bias_in_the_graph_so_autograd_grad_reaches_the_readout_bias_with_zeros():
    """The stack's concatenation put every bias in the graph of every stream; the readout bias does not enter the Laplacian, and
    ``torch.autograd.grad`` over all parameters must return a zero for it under both layouts rather than raise."""
    dim = 4
    x = torch.tensor(np.random.default_rng(5).standard_normal((5, dim)))
    results = {}
    for layout in ("stack", "split"):
        layers = _layers(dim, 6, 2, act="tanh", seed=3)
        params = [t.detach().clone().requires_grad_(True) for W, b, _ in layers for t in (W, b)]
        it = iter(params)
        wired = [(next(it), next(it), spec) for _, _, spec in layers]
        for quantity in ("laplacian", "grad_sq", "jacobian"):
            out = getattr(second_order_streams(x, wired, layout=layout), quantity)
            grads = torch.autograd.grad(out.sum(), params, retain_graph=True)
            assert all(g is not None and torch.isfinite(g).all() for g in grads), (layout, quantity)
            results[(layout, quantity)] = grads
        assert torch.count_nonzero(results[(layout, "laplacian")][-1]) == 0  # the readout bias: zero, not missing
    for quantity in ("laplacian", "grad_sq", "jacobian"):
        for g, h in zip(results[("stack", quantity)], results[("split", quantity)], strict=True):
            torch.testing.assert_close(g, h, atol=1e-10, rtol=1e-8)


@pytest.mark.parametrize("specs", [(None, None, None), (None, "tanh", None), ("tanh", None, None), (None, None, "tanh")])
def test_affine_layers_anywhere_give_the_same_streams_in_both_layouts_and_a_zero_laplacian_for_an_affine_network(specs):
    """An affine first layer has a zero second-order stream: the split layout must not seed it with the value."""
    dim = 4
    x = torch.tensor(np.random.default_rng(8).standard_normal((5, dim)))
    base = _layers(dim, 6, 2, act="tanh", seed=9, requires_grad=True)
    layers = [(W, b, spec) for (W, b, _), spec in zip(base, specs, strict=True)]
    params = [t for W, b, _ in layers for t in (W, b)]
    out = {layout: second_order_streams(x, layers, layout=layout) for layout in ("stack", "split")}
    for p, q in zip(
        (out["stack"].value, out["stack"].grad_sq, out["stack"].laplacian),
        (out["split"].value, out["split"].grad_sq, out["split"].laplacian),
        strict=True,
    ):
        torch.testing.assert_close(p, q, atol=1e-12, rtol=0)
    u, g2, lap = _oracle(layers, x, torch.eye(dim, dtype=torch.float64))
    torch.testing.assert_close(out["split"].laplacian[:, 0], lap, atol=1e-9, rtol=0)
    if all(s is None for s in specs):
        assert torch.count_nonzero(out["split"].laplacian) == 0
    for quantity in ("laplacian", "grad_sq"):
        grads = {
            layout: torch.autograd.grad(getattr(out[layout], quantity).sum(), params, retain_graph=True, allow_unused=True)
            for layout in out
        }
        for g, h in zip(grads["stack"], grads["split"], strict=True):
            assert (g is None) == (h is None)
            if g is not None:
                torch.testing.assert_close(g, h, atol=1e-10, rtol=1e-8)
