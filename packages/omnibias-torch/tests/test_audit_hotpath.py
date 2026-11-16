# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Hot-path changes of the performance audit: each keeps the values of the code it replaced, and its caches never go stale."""

from __future__ import annotations

import gc

import numpy as np
import pytest
import torch
from omnibias.torch.experimental import planned
from omnibias.torch.experimental.planned import compile_planned_operator
from omnibias.torch.experimental.projected import capture_safe_branches, input_factor
from omnibias.torch.experimental.streams import (
    activation_tower,
    directional_jet_streams,
    has_row_tower,
    plan_constant,
    second_order_streams,
)
from omnibias.torch.fastpath import eulerian, legendre


def _layers(dims, act="tanh", seed=0, bias=True):
    g = np.random.default_rng(seed)
    out = []
    for i, (a, b) in enumerate(zip(dims[:-1], dims[1:], strict=True)):
        W = torch.tensor(g.standard_normal((b, a)) / np.sqrt(a))
        out.append((W, torch.tensor(0.3 * g.standard_normal(b)) if bias else None, act if i < len(dims) - 2 else None))
    return out


def _reference_horner(coeffs, x):
    """The original evaluation: a filled tensor times ``x`` and one add per coefficient, zeros included."""
    result = torch.full_like(x, coeffs[-1])
    for k in range(len(coeffs) - 2, -1, -1):
        result = result * x + coeffs[k]
    return result


# ------------------------------------------------------------------------------------------------ activation towers


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("order", range(1, 13))
def test_horner_without_the_filled_start_or_zero_adds_is_bit_identical(dtype, order):
    z = torch.randn(6, 9, dtype=dtype, generator=torch.Generator().manual_seed(order)) * 3
    z[0, :4] = torch.tensor([0.0, -0.0, 40.0, -40.0], dtype=dtype)  # exact zeros of either sign, and saturation
    for module, coeff_fn, base in (
        (legendre, legendre.tanh_polynomial_coeffs, torch.tanh),
        (eulerian, eulerian.sigmoid_polynomial_coeffs, torch.sigmoid),
    ):
        t = base(z)
        coeffs = coeff_fn(order)
        got, want = module._horner(coeffs, t), _reference_horner(coeffs, t)
        assert got.view(torch.int64 if dtype == torch.float64 else torch.int32).equal(
            want.view(torch.int64 if dtype == torch.float64 else torch.int32)
        )


@pytest.mark.parametrize("name", ["tanh", "sigmoid", "softplus"])
def test_tower_rows_are_the_rows_of_the_stacked_tower(name):
    from omnibias.torch.activations.registry import get_activation

    spec = get_activation(name)
    z = torch.randn(5, 7, dtype=torch.float64)
    assert has_row_tower(spec)
    rows = activation_tower(spec, z, 4)
    stacked = spec.tower(z, 4)
    assert len(rows) == 5 and all(torch.equal(rows[n], stacked[n]) for n in range(5))
    # and they are what the unstacked forward and fastpath functions return one at a time
    assert torch.equal(rows[0], spec.forward(z))
    assert all(torch.equal(rows[n], spec.fastpath(z, n)) for n in range(1, 5))


# ------------------------------------------------------------------------------------------------------ stream engine


def _dense_oracle(layers, x, A):
    def f(z):
        a = z
        for W, b, act in layers:
            a = a @ W.T + (0 if b is None else b)
            if act is not None:
                a = getattr(torch, act)(a)
        return a[0]

    h = torch.func.vmap(torch.func.hessian(f))(x)
    return torch.einsum("ij,bij->b", A, h)


@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("form", ["auto", "diag", "gram"])
def test_first_layer_special_cases_match_the_dense_hessian(bias, form):
    layers = _layers((5, 7, 7, 1), bias=bias, seed=2)
    x = torch.randn(4, 5, dtype=torch.float64, generator=torch.Generator().manual_seed(1))
    r = second_order_streams(x, layers, form=form)
    torch.testing.assert_close(r.laplacian[:, 0], _dense_oracle(layers, x, torch.eye(5, dtype=torch.float64)), atol=1e-11, rtol=0)


@pytest.mark.parametrize("order", [1, 2, 3, 4])
def test_directional_jets_match_repeated_forward_mode_differentiation(order):
    layers = _layers((3, 6, 6, 1), seed=5)
    x = torch.randn(3, 3, dtype=torch.float64, generator=torch.Generator().manual_seed(3))
    V = torch.eye(3, dtype=torch.float64)

    def along(i, direction):
        def h(t):
            a = x[i] + t * V[:, direction]
            for W, b, act in layers:
                a = a @ W.T + (0 if b is None else b)
                if act is not None:
                    a = getattr(torch, act)(a)
            return a[0]

        return h

    jet = directional_jet_streams(x, layers, V, order)
    for i in range(3):
        for direction in (0, 2):
            h = along(i, direction)
            for p in range(1, order + 1):
                h = torch.func.jacfwd(h)
                torch.testing.assert_close(
                    jet.derivatives[i, direction, p - 1, 0], h(torch.zeros((), dtype=torch.float64)), atol=1e-10, rtol=1e-10
                )


def test_gram_identity_block_as_an_elementwise_scale_matches_the_matmul():
    layers = _layers((9, 6, 6, 1), seed=4)
    x = torch.randn(5, 9, dtype=torch.float64, generator=torch.Generator().manual_seed(8))
    got = second_order_streams(x, layers, form="gram")
    ref = second_order_streams(x, layers, form="diag")  # a different program (9 tangent channels) for the same numbers
    torch.testing.assert_close(got.value, ref.value, atol=1e-13, rtol=0)
    torch.testing.assert_close(got.grad_sq, ref.grad_sq, atol=1e-12, rtol=0)
    torch.testing.assert_close(got.laplacian, ref.laplacian, atol=1e-12, rtol=0)


def test_hess_form_split_is_reused_and_follows_an_in_place_edit():
    layers = _layers((6, 5, 5, 1), seed=6)
    x = torch.randn(4, 6, dtype=torch.float64, generator=torch.Generator().manual_seed(2))
    g = torch.Generator().manual_seed(0)
    A = torch.randn(6, 6, dtype=torch.float64, generator=g)
    A = A + A.T
    first = second_order_streams(x, layers, hess_form=A)
    again = second_order_streams(x, layers, hess_form=A)
    assert torch.equal(first.laplacian, again.laplacian)
    with torch.no_grad():
        A.mul_(2.0)  # same tensor, new version
    doubled = second_order_streams(x, layers, hess_form=A)
    torch.testing.assert_close(doubled.laplacian, 2.0 * first.laplacian, atol=1e-11, rtol=1e-11)
    fresh = second_order_streams(x, layers, hess_form=A.clone())
    assert torch.equal(doubled.laplacian, fresh.laplacian)


def test_plan_constants_are_kept_on_the_plan_and_are_not_shared_between_dtypes():
    from omnibias.core.polarization import bilaplacian_plan

    plan = bilaplacian_plan(5)
    a = plan_constant(plan, "dense", torch.float64, torch.device("cpu"), plan.dense_floats)
    assert plan_constant(plan, "dense", torch.float64, torch.device("cpu"), plan.dense_floats) is a
    b = plan_constant(plan, "dense", torch.float32, torch.device("cpu"), plan.dense_floats)
    assert b is not a and b.dtype == torch.float32


# ----------------------------------------------------------------------------------------------- planned operators


def test_context_constants_follow_the_metric_they_were_built_from():
    layers = _layers((6, 5, 5, 1), seed=7)
    x = torch.randn(4, 6, dtype=torch.float64, generator=torch.Generator().manual_seed(4))
    diffusion = torch.linspace(0.5, 1.5, 6, dtype=torch.float64)
    op = compile_planned_operator(layers, x, operator="laplacian", force="taylor_closed", diffusion=diffusion)
    before = op(x)
    assert torch.equal(op(x), before)  # served from the context's cache
    with torch.no_grad():
        diffusion.mul_(2.0)
    after = op(x)
    torch.testing.assert_close(after, 4.0 * before, atol=1e-10, rtol=1e-10)  # Delta with metric S S^T scales as s^2
    ndiff = diffusion.numpy().copy()
    op_np = compile_planned_operator(layers, x, operator="laplacian", force="taylor_closed", diffusion=ndiff)
    first = op_np(x)
    ndiff *= 3.0
    torch.testing.assert_close(op_np(x), 9.0 * first, atol=1e-10, rtol=1e-10)  # an edited array is noticed


def test_projected_factor_decision_is_the_same_with_one_read_and_with_the_branch_on_the_device():
    g = torch.Generator().manual_seed(11)
    W = torch.randn(4, 9, dtype=torch.float64, generator=g)
    healthy = input_factor(W)
    with capture_safe_branches():
        same = input_factor(W)
    assert torch.equal(healthy, same)
    deficient = W.clone()
    deficient[1:] = deficient[0]  # rank one: the Cholesky pivots collapse and the Householder branch must answer
    expected = input_factor(deficient)
    with capture_safe_branches():
        got = input_factor(deficient)
    assert torch.equal(expected, got)
    torch.testing.assert_close(got @ got.T, deficient @ deficient.T, atol=1e-10, rtol=0)


def test_graph_option_falls_back_to_eager_off_cuda_and_says_why():
    layers = _layers((4, 6, 6, 1), seed=9)
    x = torch.randn(5, 4, dtype=torch.float64, generator=torch.Generator().manual_seed(5))
    plain = compile_planned_operator(layers, x, operator="laplacian")
    auto = compile_planned_operator(layers, x, operator="laplacian", graph="auto")
    assert torch.equal(auto(x), plain(x))
    info = auto.explain()["graph"]
    assert info["captured"] == 0 and "CUDA" in info["eager_reason"]
    assert "graph" not in plain.explain()  # the default report is unchanged
    with pytest.raises(RuntimeError, match="cannot be replayed"):
        compile_planned_operator(layers, x, operator="laplacian", graph=True)(x)
    with pytest.raises(ValueError, match="graph must be"):
        compile_planned_operator(layers, x, operator="laplacian", graph="always")


# ------------------------------------------------------------------------------------------- conditioning and graphs


def test_conditioning_report_is_formed_on_reading_and_equals_the_eager_report():
    from omnibias.core.planner import PlanCache
    from omnibias.torch.experimental.conditioning import condition_report
    from omnibias.torch.experimental.residual_compiler import Residual, compile_residual

    g = torch.Generator().manual_seed(0)
    mods = [torch.nn.Linear(3, 6), torch.nn.Tanh(), torch.nn.Linear(6, 6), torch.nn.Tanh(), torch.nn.Linear(6, 1)]
    torch.manual_seed(0)
    model = torch.nn.Sequential(*mods).double()
    poisson = Residual(lambda q: q.lap[:, 0] + torch.sin(q.x[:, 0]), ["value", "lap"])
    pts = torch.randn(5, 3, generator=g, dtype=torch.float64)
    op = compile_residual(model, [(pts, poisson)], cache=PlanCache())
    assert op.explain()["conditioning"]["status"] == "no call yet"
    r, K, step = op(damping=1e-3)
    assert op._last_condition is None and op._pending_condition is not None  # nothing was formed by the call
    report = op.last_conditioning
    assert op._pending_condition is None and report == condition_report(K.detach(), 1e-3, op._ctx.policy)
    assert op.last_conditioning is report  # formed once
    op(damping=1e-3)
    assert op._last_condition is None  # a new call supersedes the old report
    assert op.explain()["conditioning"]["status"] in {"ok", "conditioning_limited", "unreliable"}


def test_a_hybrid_bridge_output_is_freed_without_the_garbage_collector():
    import weakref

    from omnibias.torch.experimental import _parameter_ad as pad
    from torch.utils import _pytree

    leaves, structure = _pytree.tree_flatten(((torch.ones(3),), {}))
    x = torch.ones(3, requires_grad=True)
    gc.collect()
    gc.disable()
    try:
        invocation = pad._Invocation(
            native=None,  # type: ignore[arg-type]
            compiled=lambda args, kwargs: args[0] * 2.0,
            leaves=leaves,
            structure=structure,
            slots={0: 0},
        )
        outputs = pad._HybridBridge.apply(invocation, x)
        assert torch.equal(outputs[0], torch.full((3,), 2.0, dtype=torch.float32))
        ref = weakref.ref(outputs[0])
        del outputs
        assert ref() is None, "the output tensor is still referenced from the invocation: a cycle only gc can free"
    finally:
        gc.enable()
