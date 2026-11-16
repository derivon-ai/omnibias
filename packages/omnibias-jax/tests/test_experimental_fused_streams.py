# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Fused Pallas stream kernels against the XLA stream engine (compiled on GPU, interpreted elsewhere)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from omnibias.jax.experimental.fused_streams import (  # noqa: E402
    fused_directional_jets,
    fused_partial_derivatives,
    fused_second_order,
    fused_unsupported_reason,
)
from omnibias.jax.experimental.streams import (  # noqa: E402
    directional_jet_streams,
    partial_derivatives,
    second_order_streams,
)

INTERPRET = jax.default_backend() != "gpu"
ACTS = ["tanh", "sigmoid", "sin", "cos", "softplus"]


def _net(D, widths, C=1, seed=0, acts="tanh"):
    rng = np.random.default_rng(seed)
    layers, fan = [], D
    for h in widths:
        layers.append(
            (
                jnp.asarray(rng.standard_normal((h, fan)) / np.sqrt(fan)),
                jnp.asarray(rng.standard_normal(h) * 0.3),
                acts,
            )
        )
        fan = h
    layers.append(
        (
            jnp.asarray(rng.standard_normal((C, fan)) / np.sqrt(fan)),
            jnp.asarray(rng.standard_normal(C) * 0.1),
            None,
        )
    )
    return layers


def _x(B, D, seed=1):
    return jnp.asarray(np.random.default_rng(seed).standard_normal((B, D)) * 0.7)


def _params(layers):
    return [(W, b) for W, b, _ in layers]


def _rebuild(layers, params):
    return [(W, b, spec) for (W, b), (_, _, spec) in zip(params, layers, strict=True)]


@pytest.mark.parametrize("act", ACTS)
@pytest.mark.parametrize("D,widths,B", [(2, (16,), 5), (3, (24, 20), 19), (5, (16, 16, 16), 33)])
def test_second_order_matches_stream_engine(act, D, widths, B):
    layers, x = _net(D, widths, 2, acts=act), _x(B, D)
    got = fused_second_order(x, layers, interpret=INTERPRET)
    ref = second_order_streams(x, layers, form="diag")
    for g, r in (
        (got.value, ref.value),
        (got.grad_sq, ref.grad_sq),
        (got.laplacian, ref.laplacian),
    ):
        np.testing.assert_allclose(np.asarray(g), np.asarray(r), rtol=1e-11, atol=1e-12)


def test_signed_weights_coordinates_and_unbatched():
    layers, x = _net(4, (16, 16), 1), _x(7, 4)
    w = jnp.asarray([1.0, -2.0, 0.5])
    got = fused_second_order(x, layers, coordinates=[0, 2, 3], weights=w, interpret=INTERPRET)
    ref = second_order_streams(x, layers, coordinates=[0, 2, 3], weights=w, form="diag")
    np.testing.assert_allclose(
        np.asarray(got.laplacian), np.asarray(ref.laplacian), rtol=1e-11, atol=1e-12
    )
    np.testing.assert_allclose(
        np.asarray(got.grad_sq), np.asarray(ref.grad_sq), rtol=1e-11, atol=1e-12
    )
    one = fused_second_order(x[0], layers, interpret=INTERPRET)
    ref1 = second_order_streams(x[0], layers, form="diag")
    np.testing.assert_allclose(
        np.asarray(one.laplacian), np.asarray(ref1.laplacian), rtol=1e-11, atol=1e-12
    )
    assert one.value.shape == ref1.value.shape


@pytest.mark.parametrize("act", ["tanh", "sin"])
def test_parameter_and_input_gradients_match_stream_engine(act):
    layers, x = _net(3, (16, 16), 1, seed=3, acts=act), _x(21, 3)
    target = jnp.asarray(np.random.default_rng(7).standard_normal(21))

    def loss(fn):
        def f(params, x):
            r = fn(x, _rebuild(layers, params))
            res = r.laplacian[:, 0] + 0.3 * r.grad_sq[:, 0] + 0.1 * r.value[:, 0] ** 2 - target
            return 0.5 * jnp.mean(res * res)

        return f

    fused = loss(lambda x, ls: fused_second_order(x, ls, interpret=INTERPRET))
    xla = loss(lambda x, ls: second_order_streams(x, ls, form="diag"))
    params = _params(layers)
    (gp, gx), (rp, rx) = (
        jax.grad(fused, argnums=(0, 1))(params, x),
        jax.grad(xla, argnums=(0, 1))(params, x),
    )
    for (gW, gb), (rW, rb) in zip(gp, rp, strict=True):
        np.testing.assert_allclose(np.asarray(gW), np.asarray(rW), rtol=1e-9, atol=1e-11)
        np.testing.assert_allclose(np.asarray(gb), np.asarray(rb), rtol=1e-9, atol=1e-11)
    np.testing.assert_allclose(np.asarray(gx), np.asarray(rx), rtol=1e-9, atol=1e-11)


def test_diffusion_gradient_flows_through_the_prologue():
    layers, x = _net(3, (16,), 1, seed=5), _x(9, 3)

    def f(S, fn):
        return jnp.sum(fn(x, layers, diffusion=S).laplacian)

    S = jnp.asarray(np.random.default_rng(2).standard_normal((3, 2)))
    g1 = jax.grad(
        lambda S: f(S, lambda x, ls, **k: fused_second_order(x, ls, interpret=INTERPRET, **k))
    )(S)
    g2 = jax.grad(lambda S: f(S, lambda x, ls, **k: second_order_streams(x, ls, form="diag", **k)))(
        S
    )
    np.testing.assert_allclose(np.asarray(g1), np.asarray(g2), rtol=1e-9, atol=1e-11)


@pytest.mark.parametrize("order", [1, 2, 3, 4])
@pytest.mark.parametrize("act", ["tanh", "sin"])
def test_directional_jets_match_stream_engine(order, act):
    layers, x = _net(2, (16, 16), 1, seed=4, acts=act), _x(11, 2)
    V = jnp.asarray([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
    got = fused_directional_jets(x, layers, V, order, interpret=INTERPRET)
    ref = directional_jet_streams(x, layers, V, order)
    np.testing.assert_allclose(np.asarray(got.value), np.asarray(ref.value), rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(
        np.asarray(got.derivatives), np.asarray(ref.derivatives), rtol=1e-10, atol=1e-11
    )


def test_jet_parameter_gradients_order_four():
    layers, x = _net(2, (16, 16), 1, seed=6), _x(13, 2)
    V = jnp.asarray([[1.0, 0.0], [0.0, 1.0]])

    def loss(fn):
        def f(params):
            j = fn(x, _rebuild(layers, params), V, 4)
            return jnp.mean(
                (j.derivatives[:, 0, 3, 0] + 2.0 * j.derivatives[:, 1, 1, 0] + j.value[:, 0]) ** 2
            )

        return f

    g = jax.grad(
        loss(lambda x, ls, V, n: fused_directional_jets(x, ls, V, n, interpret=INTERPRET))
    )(_params(layers))
    r = jax.grad(loss(directional_jet_streams))(_params(layers))
    for (gW, gb), (rW, rb) in zip(g, r, strict=True):
        np.testing.assert_allclose(np.asarray(gW), np.asarray(rW), rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(np.asarray(gb), np.asarray(rb), rtol=1e-8, atol=1e-10)


def test_partial_derivatives_by_polarization():
    layers, x = _net(2, (16, 16), 1, seed=8), _x(6, 2)
    idx = [(3, 0), (2, 2), (1, 1)]
    u, d = fused_partial_derivatives(x, layers, idx, interpret=INTERPRET)
    ur, dr = partial_derivatives(x, layers, idx)
    np.testing.assert_allclose(np.asarray(u), np.asarray(ur), rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(np.asarray(d), np.asarray(dr), rtol=1e-9, atol=1e-10)


def test_jit_and_vmap_over_models_forward_and_backward():
    layers, x = _net(3, (16, 16), 1), _x(17, 3)
    f = jax.jit(lambda x: fused_second_order(x, layers, interpret=INTERPRET).laplacian)
    ref = second_order_streams(x, layers, form="diag").laplacian
    np.testing.assert_allclose(np.asarray(f(x)), np.asarray(ref), rtol=1e-11, atol=1e-12)
    W0 = layers[0][0]
    stack = jnp.stack([W0, 0.9 * W0, 1.1 * W0])

    def total(fn):
        return lambda W: jnp.sum(fn(x, [(W, layers[0][1], layers[0][2])] + layers[1:]).laplacian)

    fused = total(lambda x, ls: fused_second_order(x, ls, interpret=INTERPRET))
    xla = total(lambda x, ls: second_order_streams(x, ls, form="diag"))
    np.testing.assert_allclose(
        np.asarray(jax.vmap(jax.grad(fused))(stack)),
        np.asarray(jax.vmap(jax.grad(xla))(stack)),
        rtol=1e-9,
        atol=1e-11,
    )


def test_float32_matches_float64_within_single_precision():
    layers, x = _net(3, (16, 16), 1), _x(17, 3)
    l32 = [(W.astype(jnp.float32), b.astype(jnp.float32), s) for W, b, s in layers]
    got = fused_second_order(x.astype(jnp.float32), l32, interpret=INTERPRET)
    ref = second_order_streams(x, layers, form="diag")
    assert got.laplacian.dtype == jnp.float32
    np.testing.assert_allclose(
        np.asarray(got.laplacian), np.asarray(ref.laplacian), rtol=2e-4, atol=2e-5
    )


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


def test_second_derivative_through_the_kernel_is_refused():
    layers, x = _net(2, (16,), 1), _x(5, 2)

    def loss(W):
        ls = [(W, layers[0][1], layers[0][2])] + layers[1:]
        return jnp.sum(fused_second_order(x, ls, interpret=INTERPRET).laplacian)

    W = layers[0][0]
    jax.grad(loss)(W)
    jax.jacrev(
        lambda W: fused_second_order(
            x, [(W, layers[0][1], layers[0][2])] + layers[1:], interpret=INTERPRET
        ).laplacian
    )(W)
    with pytest.raises(NotImplementedError, match="first order"):
        jax.jvp(jax.grad(loss), (W,), (jnp.ones_like(W),))


def test_unnamed_block_shrinks_to_fit_shared_memory():
    from omnibias.jax.experimental.fused_streams import _fit_block

    assert (
        _fit_block("float32", 128, 4, 32, explicit=False) == 16
    )  # the backward at 32 would need ~213 kB
    assert (
        _fit_block("float32", 128, 4, 32, explicit=True) == 32
    )  # a named block is the caller's choice
    assert _fit_block("float32", 32, 4, 32, explicit=False) == 32
    with pytest.raises(NotImplementedError, match="shared memory"):
        _fit_block("float32", 128, 14, 16, explicit=False)


@pytest.mark.skipif(
    INTERPRET, reason="the shared-memory limit exists on the compiled GPU lowering only"
)
def test_shapes_beyond_shared_memory_raise_a_named_error_not_an_xla_failure():
    layers, x = _net(12, (128, 128), 1), _x(32, 12)
    l32 = [(W.astype(jnp.float32), b.astype(jnp.float32), s) for W, b, s in layers]
    with pytest.raises(NotImplementedError, match="shared memory"):
        fused_second_order(x.astype(jnp.float32), l32)
    # a shape whose default tile fits but whose larger tile does not trains through the default tile
    small, xs = _net(2, (128, 128), 1), _x(64, 2)
    wb = [(W.astype(jnp.float32), b.astype(jnp.float32)) for W, b, _ in small]
    acts = [s for _, _, s in small]

    def loss(params):
        ls = [(W, b, a) for (W, b), a in zip(params, acts, strict=True)]
        return fused_second_order(xs.astype(jnp.float32), ls).laplacian.sum()

    got = jax.grad(loss)(wb)
    assert all(bool(jnp.all(jnp.isfinite(W))) for W, _ in got)


def test_precision_names_select_the_kernel_dot():
    from omnibias.jax.experimental import fused_streams as fs

    assert fs._full_precision("highest") and not fs._full_precision("tf32")
    assert not fs._full_precision("default")  # unset: the backend default, TF32 where the device has it
    with jax.default_matmul_precision("highest"):
        assert fs._full_precision("default")  # "default" leaves the process setting, here full precision
    with jax.default_matmul_precision("tensorfloat32"):
        assert not fs._full_precision("default")
    with pytest.raises(ValueError, match="precision must be one of"):
        fs._full_precision("fast")
    layers, x = _net(3, (16, 16), 1), _x(5, 3)
    with pytest.raises(ValueError, match="precision must be one of"):
        fused_second_order(x, layers, precision="fast", interpret=True)



def test_padded_parameters_are_formed_once_for_concrete_arrays_and_follow_replacement():
    from omnibias.jax.experimental import fused_streams as fs

    layers, x = _net(3, (20, 24), 1, seed=4), _x(9, 3)
    W = layers[1][0]
    pad_a = fs._padded_param(W, W.dtype, (32, 32))
    assert fs._padded_param(W, W.dtype, (32, 32)) is pad_a
    np.testing.assert_array_equal(pad_a[:24, :20], W)
    first = fused_second_order(x, layers, interpret=INTERPRET)
    replaced = [(layers[0]), (W * 0.5, layers[1][1], layers[1][2]), *layers[2:]]
    changed = fused_second_order(x, replaced, interpret=INTERPRET)
    assert not np.allclose(first.laplacian, changed.laplacian)
    ref = second_order_streams(x, replaced, form="diag")
    np.testing.assert_allclose(changed.laplacian, ref.laplacian, rtol=1e-11, atol=1e-12)
    before = len(fs._PADDED)
    del pad_a, replaced
    import gc

    gc.collect()
    assert len(fs._PADDED) <= before
    acts = [a for _, _, a in layers]

    def under_jit(wb):  # traced parameters are padded inside their program, never cached
        ls = [(W_, b_, a) for (W_, b_), a in zip(wb, acts, strict=True)]
        return fused_second_order(x, ls, interpret=INTERPRET).laplacian

    traced = jax.jit(under_jit)([(W_, b_) for W_, b_, _ in layers])
    np.testing.assert_allclose(traced, first.laplacian, rtol=1e-11, atol=1e-12)


def test_a_concrete_weight_closed_over_by_a_trace_is_not_cached_as_a_tracer():
    from omnibias.jax.experimental import fused_streams as fs

    layers, x = _net(3, (20, 24), 1, seed=6), _x(9, 3)
    fs._PADDED.clear()
    eager = fused_second_order(x, layers, interpret=INTERPRET).laplacian
    fs._PADDED.clear()
    traced = jax.jit(lambda xx: fused_second_order(xx, layers, interpret=INTERPRET).laplacian)(x)
    assert all(not isinstance(v[2], jax.core.Tracer) for v in fs._PADDED.values())
    again = fused_second_order(x, layers, interpret=INTERPRET).laplacian
    np.testing.assert_allclose(traced, eager, rtol=1e-11, atol=1e-12)
    np.testing.assert_array_equal(again, eager)
