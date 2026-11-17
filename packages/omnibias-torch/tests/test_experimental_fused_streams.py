# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Fused Triton stream kernels against the autograd stream engine (CUDA, or the Triton interpreter on CPU)."""

import os

import pytest

pytest.importorskip("triton")
import torch  # noqa: E402

if not torch.cuda.is_available():
    os.environ["TRITON_INTERPRET"] = "1"

import numpy as np  # noqa: E402
from omnibias.torch.experimental.fused_streams import (  # noqa: E402
    fused_directional_jets,
    fused_partial_derivatives,
    fused_second_order,
    fused_unsupported_reason,
)
from omnibias.torch.experimental.streams import (  # noqa: E402
    directional_jet_streams,
    partial_derivatives,
    second_order_streams,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ACTS = ["tanh", "sigmoid", "sin", "cos", "softplus"]


def _t(a, dtype=torch.float64, grad=False):
    return torch.as_tensor(np.asarray(a), dtype=dtype, device=DEVICE).requires_grad_(grad)


def _net(D, widths, C=1, seed=0, acts="tanh", grad=False):
    rng = np.random.default_rng(seed)
    layers, fan = [], D
    for h in widths:
        layers.append(
            [
                _t(rng.standard_normal((h, fan)) / np.sqrt(fan), grad=grad),
                _t(rng.standard_normal(h) * 0.3, grad=grad),
                acts,
            ]
        )
        fan = h
    layers.append(
        [
            _t(rng.standard_normal((C, fan)) / np.sqrt(fan), grad=grad),
            _t(rng.standard_normal(C) * 0.1, grad=grad),
            None,
        ]
    )
    return layers


def _x(B, D, seed=1):
    return _t(np.random.default_rng(seed).standard_normal((B, D)) * 0.7)


def _close(a, b, rtol=1e-11, atol=1e-12):
    np.testing.assert_allclose(
        a.detach().cpu().numpy(), b.detach().cpu().numpy(), rtol=rtol, atol=atol
    )


@pytest.mark.parametrize("act", ACTS)
@pytest.mark.parametrize("D,widths,B", [(2, (16,), 5), (3, (24, 20), 19)])
def test_second_order_matches_stream_engine(act, D, widths, B):
    layers, x = _net(D, widths, 2, acts=act), _x(B, D)
    got = fused_second_order(x, layers)
    ref = second_order_streams(x, layers, form="diag")
    _close(got.value, ref.value)
    _close(got.grad_sq, ref.grad_sq)
    _close(got.laplacian, ref.laplacian)


def test_signed_weights_coordinates_and_unbatched():
    layers, x = _net(4, (16, 16), 1), _x(7, 4)
    w = _t([1.0, -2.0, 0.5])
    got = fused_second_order(x, layers, coordinates=[0, 2, 3], weights=w)
    ref = second_order_streams(x, layers, coordinates=[0, 2, 3], weights=w, form="diag")
    _close(got.laplacian, ref.laplacian)
    _close(got.grad_sq, ref.grad_sq)
    one = fused_second_order(x[0], layers)
    _close(one.laplacian, second_order_streams(x[0], layers, form="diag").laplacian)


@pytest.mark.parametrize("act", ["tanh", "sin", "softplus"])
def test_parameter_and_input_gradients_match_stream_engine(act):
    x = _x(21, 3).requires_grad_(True)
    target = _t(np.random.default_rng(7).standard_normal(21))

    def run(fn, layers):
        r = fn(x, layers)
        res = r.laplacian[:, 0] + 0.3 * r.grad_sq[:, 0] + 0.1 * r.value[:, 0] ** 2 - target
        loss = 0.5 * (res * res).mean()
        params = [t for W, b, _ in layers for t in (W, b)]
        return torch.autograd.grad(loss, [*params, x])

    ga = run(fused_second_order, _net(3, (16, 16), 1, seed=3, acts=act, grad=True))
    gb = run(
        lambda x, ls: second_order_streams(x, ls, form="diag"),
        _net(3, (16, 16), 1, seed=3, acts=act, grad=True),
    )
    for u, v in zip(ga, gb, strict=True):
        _close(u, v, rtol=1e-9, atol=1e-11)


def test_diffusion_gradient_flows_through_the_prologue():
    layers, x = _net(3, (16,), 1, seed=5), _x(9, 3)
    S1 = _t(np.random.default_rng(2).standard_normal((3, 2)), grad=True)
    S2 = S1.detach().clone().requires_grad_(True)
    g1 = torch.autograd.grad(fused_second_order(x, layers, diffusion=S1).laplacian.sum(), S1)[0]
    g2 = torch.autograd.grad(
        second_order_streams(x, layers, diffusion=S2, form="diag").laplacian.sum(), S2
    )[0]
    _close(g1, g2, rtol=1e-9, atol=1e-11)


@pytest.mark.parametrize("order", [1, 2, 3, 4])
@pytest.mark.parametrize("act", ["tanh", "sin"])
def test_directional_jets_match_stream_engine(order, act):
    layers, x = _net(2, (16, 16), 1, seed=4, acts=act), _x(11, 2)
    V = _t([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
    got = fused_directional_jets(x, layers, V, order)
    ref = directional_jet_streams(x, layers, V, order)
    _close(got.value, ref.value)
    _close(got.derivatives, ref.derivatives, rtol=1e-10, atol=1e-11)


def test_jet_parameter_gradients_order_four():
    V = _t([[1.0, 0.0], [0.0, 1.0]])
    x = _x(13, 2)

    def run(fn, layers):
        j = fn(x, layers, V, 4)
        loss = (
            (j.derivatives[:, 0, 3, 0] + 2.0 * j.derivatives[:, 1, 1, 0] + j.value[:, 0]) ** 2
        ).mean()
        return torch.autograd.grad(loss, [t for W, b, _ in layers for t in (W, b)])

    ga = run(fused_directional_jets, _net(2, (16, 16), 1, seed=6, grad=True))
    gb = run(directional_jet_streams, _net(2, (16, 16), 1, seed=6, grad=True))
    for u, v in zip(ga, gb, strict=True):
        _close(u, v, rtol=1e-8, atol=1e-10)


def test_partial_derivatives_by_polarization():
    layers, x = _net(2, (16, 16), 1, seed=8), _x(6, 2)
    idx = [(3, 0), (2, 2), (1, 1)]
    u, d = fused_partial_derivatives(x, layers, idx)
    ur, dr = partial_derivatives(x, layers, idx)
    _close(u, ur)
    _close(d, dr, rtol=1e-9, atol=1e-10)


def test_float32_matches_float64_within_single_precision():
    l64, x64 = _net(3, (16, 16), 1), _x(17, 3)
    l32 = [[W.detach().float(), b.detach().float(), s] for W, b, s in l64]
    got = fused_second_order(x64.float(), l32)
    ref = second_order_streams(x64, l64, form="diag")
    assert got.laplacian.dtype == torch.float32
    _close(got.laplacian.double(), ref.laplacian, rtol=2e-4, atol=2e-5)


def test_second_derivative_through_the_kernel_is_refused():
    layers, x = _net(2, (16,), 1, grad=True), _x(5, 2)
    loss = fused_second_order(x, layers).laplacian.sum()
    (g,) = torch.autograd.grad(loss, layers[0][0], create_graph=True)
    with pytest.raises(NotImplementedError, match="first order"):
        torch.autograd.grad(g.sum(), layers[0][0])


@pytest.mark.parametrize(
    "layers,channels,fragment",
    [
        (_net(2, (16,), 1, acts="gelu"), 3, "no fused kernel tower"),
        (_net(2, (256,), 1), 3, "exceeds the fused limit"),
        (_net(2, (16,), 1), 100, "stream channels"),
        (_net(2, (16,), 1)[1:], 3, "hidden layer"),
    ],
)
def test_unsupported_configurations_are_named(layers, channels, fragment):
    reason = fused_unsupported_reason(layers, channels=channels)
    assert reason is not None and fragment in reason


def test_generated_program_is_straight_line_and_exact_constants():
    from omnibias.torch.experimental._triton_program import Cfg, Graph, kernel_sources

    src = kernel_sources(Cfg("jet", 4, 2, ("tanh", "tanh"), 16, 16, 16, 16, "float64", True))
    assert "def fused_forward" in src and "def fused_backward" in src
    assert "tl.cond" not in src and "if " not in src.split("def fused_forward", 1)[1]
    with pytest.raises(ValueError, match="float32"):
        Graph().input("a") * 0.1


def _loss_fn(x, layers):
    r = fused_second_order(x, layers)
    return (r.laplacian[:, 0] + 0.5 * r.grad_sq[:, 0] + r.value[:, 0] ** 2).pow(2).mean()


def test_the_custom_ops_compose_with_torch_compile_graph_capture():
    """Fake kernels and registered autograd let the compiler trace the fused ops without running them."""
    layers, x = _net(3, (16, 16), 1, seed=9, grad=True), _x(11, 3)
    params = [t for W, b, _ in layers for t in (W, b)]
    eager = torch.autograd.grad(_loss_fn(x, layers), params)
    compiled = torch.compile(_loss_fn, backend="aot_eager", fullgraph=True)
    got = torch.autograd.grad(compiled(x, layers), params)
    for u, v in zip(got, eager, strict=True):
        _close(u, v, rtol=1e-12, atol=1e-13)


def test_the_ops_have_fake_and_autograd_registrations():
    from omnibias.torch.experimental import fused_streams as fs

    for name in ("fused_stream_forward", "fused_stream_backward"):
        op = getattr(torch.ops.omnibias, name)
        assert op is not None
    assert fs._forward_op is not None and fs._backward_op is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs need a GPU")
def test_cuda_graph_replays_the_fused_forward_and_backward_without_host_sync():
    layers, x = _net(3, (32, 32), 1, seed=10, grad=True), _x(64, 3)
    params = [t for W, b, _ in layers for t in (W, b)]
    static_x = x.clone()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):  # warm up: compile the Triton kernels outside capture
            torch.autograd.grad(_loss_fn(static_x, layers), params)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        grads = torch.autograd.grad(_loss_fn(static_x, layers), params)
    new_x = _x(64, 3, seed=77)
    static_x.copy_(new_x)
    graph.replay()
    torch.cuda.synchronize()
    ref = torch.autograd.grad(_loss_fn(new_x, layers), params)
    for u, v in zip(grads, ref, strict=True):
        _close(u, v, rtol=1e-11, atol=1e-12)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="inductor needs a GPU")
def test_inductor_compiles_a_training_loss_through_the_fused_ops():
    layers, x = _net(3, (32, 32), 1, seed=11, grad=True), _x(64, 3)
    params = [t for W, b, _ in layers for t in (W, b)]
    eager = torch.autograd.grad(_loss_fn(x, layers), params)
    compiled = torch.compile(_loss_fn, fullgraph=True)
    got = torch.autograd.grad(compiled(x, layers), params)
    for u, v in zip(got, eager, strict=True):
        _close(u, v, rtol=1e-10, atol=1e-11)


def test_unnamed_block_shrinks_to_fit_shared_memory():
    from omnibias.torch.experimental.fused_streams import _fit_block

    assert _fit_block("float32", 128, 4, 32, explicit=False) == 16
    assert _fit_block("float32", 128, 4, 32, explicit=True) == 32
    assert _fit_block("float32", 32, 4, 32, explicit=False) == 32
    with pytest.raises(NotImplementedError, match="shared memory"):
        _fit_block("float32", 128, 14, 16, explicit=False)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the limit exists on the compiled lowering only")
def test_shapes_beyond_shared_memory_raise_a_named_error():
    layers, x = _net(12, (128, 128), 1, grad=True), _x(32, 12)
    l32 = [[t.detach().float() for t in (W, b)] + [s] for W, b, s in layers]
    with pytest.raises(NotImplementedError, match="shared memory"):
        fused_second_order(x.float(), l32)


# ------------------------------------------------------------------------------------------------ performance audit


def test_precision_names_select_the_kernel_dot(monkeypatch):
    from omnibias.torch.experimental import fused_streams as fs
    from omnibias.torch.experimental._triton_program import kernel_sources

    for dt in (torch.float64,):  # a float64 product is full precision under every name
        assert all(fs._full_precision(p, dt) for p in ("highest", "tf32", "default"))
    assert fs._full_precision("highest", torch.float32)
    assert not fs._full_precision("tf32", torch.float32)
    monkeypatch.setattr(fs, "_process_matmul_is_ieee", lambda: True)
    assert fs._full_precision("default", torch.float32)  # "default" leaves the process setting, which is full precision
    monkeypatch.setattr(fs, "_process_matmul_is_ieee", lambda: False)
    assert not fs._full_precision("default", torch.float32)
    with pytest.raises(ValueError, match="precision must be one of"):
        fs._full_precision("fast", torch.float32)
    layers, x = _net(3, (16, 16), 1), _x(5, 3)
    with pytest.raises(ValueError, match="precision must be one of"):
        fused_second_order(x, layers, precision="fast")
    cfg = fs.Cfg("lap", 2, 3, ("tanh", "tanh"), 16, 16, 16, 16, "float32", True)
    assert 'input_precision="ieee"' in kernel_sources(cfg) and 'input_precision="tf32"' not in kernel_sources(cfg)
    assert 'input_precision="tf32"' in kernel_sources(cfg._replace(highest=False))


def test_the_process_setting_is_read_the_way_pytorch_holds_it():
    from omnibias.torch.experimental import fused_streams as fs

    matmul = torch.backends.cuda.matmul
    if not hasattr(matmul, "fp32_precision"):
        pytest.skip("PyTorch before 2.9 holds the mode in set_float32_matmul_precision")
    before = matmul.fp32_precision
    try:
        for value, ieee in (("ieee", True), ("tf32", False)):
            matmul.fp32_precision = value
            assert fs._process_matmul_is_ieee() is ieee
    finally:
        matmul.fp32_precision = before


def test_the_planned_fused_program_follows_the_policy_of_its_operator():
    from omnibias.core.planner import PlanCache
    from omnibias.torch.experimental import compile_planned_operator

    layers, x = _net(3, (16, 16), 1), _x(5, 3)
    modes = {}
    for name in ("highest", "tf32", "default"):
        op = compile_planned_operator(
            layers, x, operator="laplacian", force="fused_streams", precision=name, cache=PlanCache()
        )
        modes[name] = op._ctx.matmul
    assert modes == {"highest": "highest", "tf32": "tf32", "default": "default"}


def test_padded_parameters_are_reused_only_while_unchanged_and_never_in_a_recorded_graph():
    from omnibias.torch.experimental import fused_streams as fs

    layers = _net(3, (20, 24), 1, seed=4)  # widths below the tile are padded
    x = _x(9, 3)
    W = layers[1][0]
    saved = W.clone()
    with torch.no_grad():
        first = fused_second_order(x, layers)
        pad_a = fs._padded_param(W, W.dtype, W.device, (32, 32))
        assert fs._padded_param(W, W.dtype, W.device, (32, 32)) is pad_a  # unchanged: reused
        W.mul_(1.25)  # an optimizer step updates in place
        pad_b = fs._padded_param(W, W.dtype, W.device, (32, 32))
        assert pad_b is not pad_a and torch.equal(pad_b[:24, :20], W)
        W.copy_(saved)  # restored exactly: the copy of the first call is valid again
        again = fused_second_order(x, layers)
    _close(again.laplacian, first.laplacian, rtol=1e-13, atol=1e-13)
    assert torch.equal(again.value, first.value)
    with torch.no_grad():
        replaced = layers[1][0].clone().mul_(0.5)
        layers[1][0] = replaced
        assert not torch.equal(fs._padded_param(replaced, replaced.dtype, replaced.device, (32, 32)), pad_a)
    # a call that records gradients pads afresh: two consecutive backward passes work and reach every parameter
    live = _net(3, (20, 24), 1, seed=4, grad=True)
    params = [t for W_, b, _ in live for t in (W_, b)]
    with torch.no_grad():
        fused_second_order(x, live)  # fills the cache with the parameters' padded copies
    for _ in range(2):
        loss = fused_second_order(x, live).laplacian.sum()
        grads = torch.autograd.grad(loss, params)
        assert all(g is not None and torch.isfinite(g).all() for g in grads)
    ref = _net(3, (20, 24), 1, seed=4, grad=True)
    ref_loss = second_order_streams(x, ref, form="diag").laplacian.sum()
    for u, v in zip(grads, torch.autograd.grad(ref_loss, [t for W_, b, _ in ref for t in (W_, b)]), strict=True):
        _close(u, v, rtol=1e-9, atol=1e-11)


def test_the_padded_parameter_cache_does_not_hold_a_dead_parameter():
    from omnibias.torch.experimental import fused_streams as fs

    before = len(fs._PADDED)
    W = _t(np.random.default_rng(0).standard_normal((20, 20)))
    with torch.no_grad():
        fs._padded_param(W, W.dtype, W.device, (32, 32))
    assert len(fs._PADDED) == before + 1
    del W
    assert len(fs._PADDED) == before


def test_the_stacked_reductions_match_the_per_channel_sums():
    layers, x = _net(4, (16, 16, 16), 1, seed=6), _x(11, 4)
    w = _t([1.0, -2.0, 0.5, 0.25])
    for weights in (None, w):
        got = fused_second_order(x, layers, weights=weights)
        ref = second_order_streams(x, layers, weights=weights, form="diag")
        _close(got.grad_sq, ref.grad_sq)
        _close(got.laplacian, ref.laplacian)


@pytest.mark.parametrize("hidden", [(16,), (16, 16), (24, 20, 16)])
def test_gradients_of_every_hidden_layer_come_from_one_grouped_reduction(hidden):
    x = _x(21, 3)
    params_a = _net(3, hidden, 1, seed=9, grad=True)
    params_b = _net(3, hidden, 1, seed=9, grad=True)

    def grads(fn, layers):
        r = fn(x, layers)
        loss = (r.laplacian**2).sum() + r.grad_sq.sum() + r.value.sum()
        return torch.autograd.grad(loss, [t for W, b, _ in layers for t in (W, b)])

    ga = grads(fused_second_order, params_a)
    gb = grads(lambda x_, ls: second_order_streams(x_, ls, form="diag"), params_b)
    for u, v in zip(ga, gb, strict=True):
        _close(u, v, rtol=1e-9, atol=1e-11)
    first, second = ga[0], ga[2] if len(ga) > 4 else ga[0]
    assert first.shape == params_a[0][0].shape  # each layer's gradient has its own parameter's shape
    assert second.shape == params_a[min(1, len(hidden) - 1)][0].shape or len(hidden) == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="TF32 matrix engines need a GPU")
def test_tf32_is_used_only_when_named_and_default_follows_the_process_setting():
    layers32 = [[t.detach().float() if hasattr(t, "detach") else t for t in layer[:2]] + [layer[2]] for layer in _net(3, (64, 64), 1, seed=2)]
    x = _x(256, 3).float()
    highest = fused_second_order(x, layers32, precision="highest")
    default = fused_second_order(x, layers32, precision="default")
    tf32 = fused_second_order(x, layers32, precision="tf32")
    assert torch.equal(default.laplacian, highest.laplacian)  # the process setting is full precision
    assert not torch.equal(tf32.laplacian, highest.laplacian)  # TF32 rounds the products
    torch.testing.assert_close(tf32.laplacian, highest.laplacian, rtol=5e-2, atol=5e-2)
    matmul = torch.backends.cuda.matmul
    if hasattr(matmul, "fp32_precision"):
        before = matmul.fp32_precision
        matmul.fp32_precision = "tf32"
        try:
            assert not torch.equal(fused_second_order(x, layers32, precision="default").laplacian, highest.laplacian)
        finally:
            matmul.fp32_precision = before
