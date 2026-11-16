# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""The planner front end of PyTorch: dispatch, refusals, generic fallback, ``explain()`` and gradients."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from omnibias.core.planner import PlanCache
from omnibias.torch.experimental import compile_operator, compile_planned_operator
from torch import nn

EXPLAIN_KEYS = {
    "selected_strategy",
    "mode",
    "device",
    "operator",
    "descriptor",
    "predicted_seconds",
    "cold_cost_seconds",
    "expected_uses",
    "candidates",
    "refused",
    "measured_seconds",
    "memory_budget_bytes",
    "versions",
    "backend",
    "matched_patterns",
    "fallback_reason",
    "chain",
    "request",
    "first_call_seconds",
    "compute_dtype",
    "executor_failures",
    "cold_cost",
}


def layers(dims, acts, seed=0):
    g = torch.Generator().manual_seed(seed)
    out = []
    for i, (a, b) in enumerate(zip(dims[:-1], dims[1:], strict=True)):
        W = (
            torch.randn(b, a, generator=g, dtype=torch.float64) * 0.5 / np.sqrt(a)
        ).requires_grad_()
        bias = (torch.randn(b, generator=g, dtype=torch.float64) * 0.1).requires_grad_()
        out.append((W, bias, acts[i]))
    return out


def forward(ls, x):
    from omnibias.torch.activations.registry import get_activation

    a = x
    for W, b, s in ls:
        a = a @ W.T + b
        if s is not None:
            a = get_activation(s).forward(a)
    return a


def pts(n, d, seed=1):
    return torch.randn(n, d, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)


def dense_lap(ls, x):
    return torch.stack(
        [
            torch.einsum(
                "ckk->c", torch.func.jacrev(torch.func.jacrev(lambda z: forward(ls, z)))(xi)
            )
            for xi in x
        ]
    )


def test_laplacian_default_plan_matches_dense_and_explains():
    ls = layers([14, 6, 6, 1], ("tanh", "tanh", None))
    x = pts(5, 14)
    op = compile_planned_operator(ls, x, cache=PlanCache())
    out = op.evaluate(x)
    assert torch.allclose(out.value, dense_lap(ls, x), atol=1e-10)
    explained = op.explain()
    assert EXPLAIN_KEYS <= set(explained)
    assert explained["backend"] == "torch" and explained["matched_patterns"] == ["dense_chain"]
    assert explained["selected_strategy"] == op.algorithm
    assert {c["name"] for c in explained["candidates"]} >= {
        "projected",
        "forward_laplacian",
        "nested_autodiff",
    }
    assert explained["first_call_seconds"] is not None


def test_every_feasible_laplacian_algorithm_agrees_and_gradients_reach_parameters():
    ls = layers([5, 7, 7, 1], ("silu", "tanh", None))
    x = pts(4, 5)
    want = dense_lap(ls, x)
    ref = torch.autograd.grad(want.sum(), [ls[0][0], ls[1][0]])
    for name in ("projected", "forward_laplacian", "taylor_jet", "nested_autodiff"):
        op = compile_planned_operator(ls, x, force=name, cache=PlanCache())
        assert op.algorithm == name
        value = op(x)
        assert torch.allclose(value, want, atol=1e-10), name
        grads = torch.autograd.grad(value.sum(), [ls[0][0], ls[1][0]])
        for a, b in zip(grads, ref, strict=True):
            assert torch.allclose(a, b, atol=1e-8, rtol=1e-8), name


def test_higher_powers_are_served_by_the_projected_programs_on_torch():
    ls = layers([3, 5, 1], ("tanh", None))
    x = pts(3, 3)
    for power in (2, 3):
        op = compile_planned_operator(ls, x, power=power, cache=PlanCache())
        refused = {r["algorithm"] for r in op.explain()["refused"]}
        assert "projected" not in refused
        reference = compile_planned_operator(
            ls, x, power=power, force="nested_autodiff", cache=PlanCache()
        )(x)
        forced = compile_planned_operator(ls, x, power=power, force="projected", cache=PlanCache())
        assert forced.algorithm == "projected"
        assert torch.allclose(forced(x), reference, atol=1e-7, rtol=1e-7), power
        assert torch.allclose(op(x), reference, atol=1e-7, rtol=1e-7), power


def test_sequential_module_is_matched_and_a_generic_callable_falls_back():
    torch.manual_seed(0)
    net = nn.Sequential(
        nn.Linear(4, 8), nn.Tanh(), nn.Linear(8, 8), nn.SiLU(), nn.Linear(8, 1)
    ).double()
    x = pts(3, 4)
    op = compile_planned_operator(net, x, cache=PlanCache())
    assert op.explain()["matched_patterns"] == ["dense_chain"]
    want = torch.stack([torch.func.hessian(lambda z: net(z[None])[0, 0])(xi).trace() for xi in x])
    assert op(x).shape == (3, 1) and torch.allclose(op(x)[:, 0], want, atol=1e-10)

    class Opaque(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = net

        def forward(self, z):
            return (self.net(z) ** 2).squeeze(-1)

    generic = compile_planned_operator(Opaque(), x, cache=PlanCache())
    info = generic.explain()
    assert info["matched_patterns"] == [] and info["fallback_reason"]
    assert generic.algorithm in {"taylor_jet", "nested_autodiff"}
    assert info["mode"] == "fallback"
    ref = torch.stack([torch.func.hessian(lambda z: Opaque()(z[None])[0])(xi).trace() for xi in x])
    assert torch.allclose(generic(x), ref, atol=1e-10)


def test_divergence_logdet_and_solve_plans():
    ls = layers([6, 9, 9, 6], ("tanh", "tanh", None))
    x = pts(3, 6)
    rhs = pts(3, 6, seed=5)
    jac = [torch.func.jacrev(lambda z: forward(ls, z))(xi) for xi in x]
    div = compile_planned_operator(ls, x, operator="divergence", cache=PlanCache())
    assert torch.allclose(div(x), torch.stack([j.trace() for j in jac]), atol=1e-10)
    ld = compile_planned_operator(ls, x, operator="logdet", cache=PlanCache())
    out = ld.evaluate(x)
    want = torch.stack(
        [torch.linalg.slogdet(torch.eye(6, dtype=torch.float64) + j)[1] for j in jac]
    )
    assert torch.allclose(out.value, want, atol=1e-10) and "sign" in out.aux
    solve = compile_planned_operator(ls, (x, rhs), operator="solve", cache=PlanCache())
    y = solve(x, rhs)
    want_y = torch.stack(
        [
            torch.linalg.solve(torch.eye(6, dtype=torch.float64) + j, r)
            for j, r in zip(jac, rhs, strict=True)
        ]
    )
    assert torch.allclose(y, want_y, atol=1e-9)
    for name in ("jacobian_trace",):
        dense = compile_planned_operator(
            ls, x, operator="divergence", force=name, cache=PlanCache()
        )
        assert torch.allclose(dense(x), div(x), atol=1e-10)


def test_find_mode_measures_and_reports_and_memory_budget_refuses():
    ls = layers([8, 6, 1], ("tanh", None))
    x = pts(4, 8)
    op = compile_planned_operator(ls, x, mode="find", cache=PlanCache())
    info = op.explain()
    assert info["mode"] == "find" and info["measured_seconds"]
    tiny = compile_planned_operator(ls, x, memory_budget=2.0**40, cache=PlanCache())
    assert tiny.explain()["memory_budget_bytes"] == 2.0**40
    with pytest.raises(ValueError, match="memory_budget"):
        compile_planned_operator(ls, x, memory_budget=1.0, cache=PlanCache())


def test_find_mode_survives_a_candidate_that_cannot_run(monkeypatch):
    from omnibias.torch.experimental import planned

    key = ("laplacian", "projected")
    real = planned._EXECUTORS[key]

    def broken(ctx):
        real(ctx)

        def boom(*_args, **_kwargs):
            raise RuntimeError("out of memory (injected)")

        return boom

    monkeypatch.setitem(planned._EXECUTORS, key, broken)
    ls = layers([8, 6, 1], ("tanh", None))
    info = compile_planned_operator(ls, pts(4, 8), mode="find", cache=PlanCache()).explain()
    assert info["selected_strategy"] != "projected"
    assert any(
        f["algorithm"] == "projected" and "find: RuntimeError" in f["reason"]
        for f in info["executor_failures"]
    )


def test_hutchinson_is_refused_unless_inexact_is_allowed():
    ls = layers([8, 6, 1], ("tanh", None))
    x = pts(4, 8)
    exact = compile_planned_operator(ls, x, cache=PlanCache())
    assert any(r["algorithm"] == "hutchinson" for r in exact.explain()["refused"])
    estimator = compile_planned_operator(
        ls, x, exact=False, force="hutchinson", n_probes=64, cache=PlanCache()
    )
    got = estimator(x, key=3)
    assert got.shape == (4, 1) and torch.isfinite(got).all()


def test_compute_dtype_and_unbatched_points_and_coordinates():
    ls32 = [
        (W.detach().float(), b.detach().float(), s) for W, b, s in layers([6, 5, 1], ("tanh", None))
    ]
    x = pts(3, 6).float()
    wide = compile_planned_operator(
        ls32, x, compute_dtype="float64", force="projected", cache=PlanCache()
    )
    out = wide(x)
    assert out.dtype == torch.float32
    ref = dense_lap([(W.double(), b.double(), s) for W, b, s in ls32], x.double())
    assert torch.allclose(out.double(), ref, atol=1e-6)
    ls = layers([7, 5, 1], ("tanh", None))
    xi = pts(1, 7)[0]
    one = compile_planned_operator(ls, xi, cache=PlanCache())
    assert one(xi).shape == (1,)
    coords = [0, 2, 5]
    sub = compile_planned_operator(ls, pts(3, 7), coordinates=coords, cache=PlanCache())
    H = torch.func.hessian(lambda z: forward(ls, z)[0])(xi)
    assert torch.allclose(sub(xi[None])[0, 0], sum(H[i, i] for i in coords), atol=1e-10)


def test_string_operator_routes_through_compile_operator_and_operator_spec_path_is_unchanged():
    ls = layers([4, 5, 1], ("tanh", None))
    x = pts(2, 4)
    op = compile_operator(ls, x, operator="laplacian", cache=PlanCache())
    assert op.explain()["backend"] == "torch"
    from omnibias.core.operator_spec import OperatorSpec

    net = nn.Sequential(nn.Linear(3, 4), nn.Tanh(), nn.Linear(4, 1)).double()
    with pytest.raises(TypeError, match="unexpected keyword"):
        compile_operator(
            net, (pts(2, 3),), operator=OperatorSpec.laplacian(), batch_ndim=1, mode="find"
        )


def test_plan_cache_keys_include_the_backend():
    ls = layers([4, 5, 1], ("tanh", None))
    x = pts(2, 4)
    cache = PlanCache()
    compile_planned_operator(ls, x, cache=cache)
    compile_planned_operator(ls, x, cache=cache)
    assert cache.hits == 1 and len(cache) == 1
    assert next(iter(cache._entries)) and "torch" in next(iter(cache._entries.keys())).split("|")


def test_find_with_top_k_none_times_every_feasible_program_and_the_cache_is_keyed_on_it():
    ls = layers([8, 6, 1], ("tanh", None))
    x = pts(4, 8)
    cache = PlanCache()
    default = compile_planned_operator(ls, x, mode="find", cache=cache).explain()
    every = compile_planned_operator(ls, x, mode="find", top_k=None, cache=cache).explain()
    feasible = {c["name"] for c in every["candidates"]}
    assert len(default["measured_seconds"]) == 2 < len(feasible)
    assert set(every["measured_seconds"]) == feasible
    assert every["selected_strategy"] == min(every["measured_seconds"], key=every["measured_seconds"].get)
    assert len(cache) == 2
    compile_planned_operator(ls, x, mode="find", top_k=None, cache=cache)
    assert cache.hits == 1 and len(cache) == 2


def test_find_keys_carry_the_library_versions(monkeypatch):
    from omnibias.torch.experimental import planned

    ls = layers([4, 5, 1], ("tanh", None))
    x = pts(2, 4)
    cache = PlanCache()
    compile_planned_operator(ls, x, mode="find", cache=cache)
    compile_planned_operator(ls, x, mode="find", cache=cache)
    assert cache.hits == 1 and len(cache) == 1
    real = planned._versions
    monkeypatch.setattr(planned, "_versions", lambda: {**real(), "torch": "0.0.0-another"})
    compile_planned_operator(ls, x, mode="find", cache=cache)
    assert cache.hits == 1 and len(cache) == 2


def test_find_cache_keys_name_the_precision_compute_dtype_and_compile_mode():
    ls = layers([4, 5, 1], ("tanh", None))
    x = pts(2, 4)
    cache = PlanCache()
    compile_planned_operator(ls, x, mode="find", cache=cache)
    compile_planned_operator(ls, x, mode="find", cache=cache)
    assert cache.hits == 1 and len(cache) == 1  # the defaults keep the key they always had
    compile_planned_operator(ls, x, mode="find", cache=cache, precision="tf32")
    compile_planned_operator(ls, x, mode="find", cache=cache, compute_dtype="float64")
    assert len(cache) == 3


def test_find_stops_a_candidate_after_one_run_above_its_deadline(monkeypatch):
    from omnibias.torch.experimental import planned

    ls = layers([8, 6, 1], ("tanh", None))
    x = pts(4, 8)
    captured = {}
    real_select = planned.select

    def spy(*args, **kwargs):
        captured["timer"] = kwargs["timer"]
        captured["key_extra"] = kwargs["key_extra"]
        return real_select(*args, **kwargs)

    monkeypatch.setattr(planned, "select", spy)
    compile_planned_operator(ls, x, mode="find", cache=PlanCache())
    assert captured["key_extra"] == ""
    runs = []
    key = ("laplacian", "projected")
    real = planned._EXECUTORS[key]

    def counting(ctx):
        inner = real(ctx)

        def run(*a, **k):
            runs.append(1)
            return inner(*a, **k)

        return run

    monkeypatch.setitem(planned._EXECUTORS, key, counting)
    timer = captured["timer"]
    assert timer("projected") < float("inf")
    assert len(runs) == 1 + 5  # one warm-up, five timed runs
    runs.clear()
    assert timer("projected", deadline=0.0) < float("inf")
    assert len(runs) == 1 + 1  # the first timed run is already above the deadline: not repeated


def test_find_candidates_run_under_the_operators_matmul_precision(monkeypatch):
    from omnibias.torch.experimental import planned

    ls = layers([8, 6, 1], ("tanh", None))
    x = pts(4, 8)
    seen = []
    real = planned.matmul_precision

    def spy(policy):
        seen.append(policy.matmul)
        return real(policy)

    monkeypatch.setattr(planned, "matmul_precision", spy)
    compile_planned_operator(ls, x, mode="find", cache=PlanCache(), precision="tf32")
    assert seen and set(seen) == {"tf32"}
