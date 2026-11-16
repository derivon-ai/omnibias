# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""The PyTorch residual planner: dispatch, refusals, generic fallback and explain()."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from omnibias.core.planner import PlanCache
from omnibias.torch.experimental import Residual, compile_residual
from torch import nn

REQUIRED = {
    "selected_strategy",
    "mode",
    "device",
    "operator",
    "descriptor",
    "candidates",
    "refused",
    "backend",
    "matched_patterns",
    "fallback_reason",
    "chain",
    "request",
    "first_call_seconds",
    "executor_failures",
    "conditioning",
    "gram_precision",
    "compute_dtype",
    "cold_cost",
}


def net(dims=(3, 6, 6, 1), seed=0):
    torch.manual_seed(seed)
    mods = []
    for i, (a, b) in enumerate(zip(dims[:-1], dims[1:], strict=True)):
        mods.append(nn.Linear(a, b))
        if i < len(dims) - 2:
            mods.append(nn.Tanh())
    return nn.Sequential(*mods).double()


def pts(n, d, seed):
    return torch.randn(n, d, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)


poisson = Residual(lambda q: q.lap[:, 0] + torch.sin(q.x[:, 0]), ["value", "lap"])
boundary = Residual(lambda q: q.u[:, 0] - q.x.sum(-1), ["value"])


def blocks(D=3):
    return [(pts(5, D, 1), poisson), (pts(3, D, 2), boundary)]


def test_every_feasible_program_agrees_on_gram_step_and_diagonal():
    model = net()
    gram = {}
    for name in ("full_streams", "autodiff_rows"):
        op = compile_residual(model, blocks(), quantity="gram", force=name, cache=PlanCache())
        assert op.algorithm == name
        gram[name] = op()
    (r1, K1), (r2, K2) = gram["full_streams"], gram["autodiff_rows"]
    assert torch.allclose(r1, r2, atol=1e-10) and torch.allclose(K1, K2, atol=1e-9, rtol=1e-9)
    steps = {}
    for name in ("full_streams", "autodiff_rows"):
        op = compile_residual(model, blocks(), force=name, cache=PlanCache())
        r, K, step = op(damping=1e-3)
        assert len(step) == len(op.parameters)
        steps[name] = torch.cat([s.reshape(-1) for s in step])
        assert all(s.shape == p.shape for s, p in zip(step, op.parameters, strict=True))
    assert torch.allclose(steps["full_streams"], steps["autodiff_rows"], atol=1e-8, rtol=1e-8)
    diag = {
        name: compile_residual(
            model, blocks(), quantity="ntk_diagonal", force=name, cache=PlanCache()
        )()[1]
        for name in ("full_streams", "autodiff_rows")
    }
    assert torch.allclose(diag["full_streams"], diag["autodiff_rows"], atol=1e-9, rtol=1e-9)
    assert torch.allclose(diag["full_streams"], torch.diagonal(K1), atol=1e-9, rtol=1e-9)


def test_default_plan_explain_refusals_and_gauss_newton_decreases_the_loss():
    model = net((2, 8, 8, 1), seed=1)
    bl = [(pts(12, 2, 3), poisson), (pts(6, 2, 4), boundary)]
    op = compile_residual(model, bl, cache=PlanCache())
    info = op.explain()
    assert REQUIRED <= set(info) and info["backend"] == "torch"
    assert info["matched_patterns"] == ["dense_chain"]
    refused = {r["algorithm"]: r["reason"] for r in info["refused"]}
    assert "does not port the first-layer projected stream program" in refused["projected_streams"]
    assert info["conditioning"]["status"] == "no call yet"
    losses = []
    for _ in range(4):
        r, K, step = op(damping=5.0)
        losses.append(float((r**2).sum()))
        op.apply_step(step)
    assert losses[-1] < losses[0]
    assert op.explain()["conditioning"]["status"] in {"ok", "conditioning_limited", "unreliable"}


def test_vector_damping_stacks_steps_and_floor_policy_changes_nothing_when_unneeded():
    model = net()
    op = compile_residual(model, blocks(), cache=PlanCache(), conditioning="floor")
    r, K, steps = op(damping=torch.tensor([1e-1, 1.0], dtype=torch.float64))
    assert all(s.shape[0] == 2 for s in steps)
    _, _, single = op(damping=torch.tensor(1.0, dtype=torch.float64))
    assert torch.allclose(steps[0][1], single[0], atol=1e-12)


def test_generic_module_falls_back_to_autodiff_rows_and_gauss_newton_works():
    class Opaque(nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = net((3, 5, 1))

        def forward(self, x):
            return self.inner(x) ** 1

    model = Opaque().double()
    op = compile_residual(model, blocks(), cache=PlanCache())
    info = op.explain()
    assert info["algorithm" if "algorithm" in info else "selected_strategy"] == "autodiff_rows"
    assert info["mode"] == "fallback" and info["fallback_reason"]
    r, K, step = op(damping=1e-3)
    ref = compile_residual(net((3, 5, 1)), blocks(), force="autodiff_rows", cache=PlanCache())
    assert K.shape == (8, 8) and len(step) == len(list(model.parameters()))
    with pytest.raises(ValueError, match="only autodiff_rows"):
        compile_residual(model, blocks(), force="full_streams", cache=PlanCache())
    assert ref.algorithm == "autodiff_rows"


def test_demand_the_full_engine_cannot_serve_is_refused_with_a_reason():
    model = net()
    bil = Residual(lambda q: q.bilap[:, 0] - 1.0, ["value", "bilap"])
    op = compile_residual(model, [(pts(4, 3, 5), bil)], quantity="gram", cache=PlanCache())
    assert op.algorithm == "autodiff_rows"
    refused = {r["algorithm"]: r["reason"] for r in op.explain()["refused"]}
    assert "full_streams" in refused and "bilap" in refused["full_streams"]
    r, K = op()
    assert K.shape == (4, 4) and torch.isfinite(K).all()


def test_old_style_callbacks_are_rejected_and_gauss_newton_needs_damping():
    model = net()
    with pytest.raises(TypeError, match="Residual"):
        compile_residual(
            model, [(pts(3, 3, 1), lambda u, g, lap, x: lap[:, 0], 2)], cache=PlanCache()
        )
    op = compile_residual(model, blocks(), cache=PlanCache())
    with pytest.raises(ValueError, match="needs `damping`"):
        op()


def test_find_mode_and_cache_behaviour():
    model = net()
    cache = PlanCache()
    op = compile_residual(model, blocks(), mode="find", cache=cache)
    assert op.explain()["mode"] == "find" and op.explain()["measured_seconds"]
    compile_residual(model, blocks(), cache=cache)
    compile_residual(model, blocks(), cache=cache)
    assert cache.hits >= 1


def test_find_times_every_program_with_top_k_none():
    model = net()
    cache = PlanCache()
    leader_only = compile_residual(model, blocks(), mode="find", top_k=1, cache=cache).explain()
    every = compile_residual(model, blocks(), mode="find", top_k=None, cache=cache).explain()
    feasible = {c["name"] for c in every["candidates"]}
    assert len(leader_only["measured_seconds"]) == 1 < len(feasible) == len(every["measured_seconds"])
    assert len(cache) == 2  # a different count is a different measurement


def test_a_program_that_fails_when_timed_is_a_failed_measurement_not_an_error(monkeypatch):
    from omnibias.torch.experimental import residual_compiler as rc

    model = net()
    leader = compile_residual(model, blocks(), cache=PlanCache()).algorithm

    def broken(ctx):
        def run(*_args, **_kwargs):
            raise RuntimeError("out of memory (injected)")

        return run

    monkeypatch.setitem(rc._RUNNERS, leader, broken)
    cache = PlanCache()
    info = compile_residual(model, blocks(), mode="find", top_k=None, cache=cache).explain()
    assert info["selected_strategy"] != leader
    assert any(
        f["algorithm"] == leader and "find: RuntimeError" in f["reason"] for f in info["executor_failures"]
    )
    assert len(cache) == 0


def test_layer_list_models_and_coordinates():
    g = torch.Generator().manual_seed(2)
    layers = [
        (
            torch.randn(5, 4, generator=g, dtype=torch.float64) * 0.4,
            torch.zeros(5, dtype=torch.float64),
            "tanh",
        ),
        (
            torch.randn(1, 5, generator=g, dtype=torch.float64) * 0.4,
            torch.zeros(1, dtype=torch.float64),
            None,
        ),
    ]
    x = pts(6, 4, 9)
    op = compile_residual(
        layers, [(x, poisson)], quantity="gram", coordinates=[0, 1, 2], cache=PlanCache()
    )
    ref = compile_residual(
        layers,
        [(x, poisson)],
        quantity="gram",
        coordinates=[0, 1, 2],
        force="autodiff_rows",
        cache=PlanCache(),
    )
    assert np.isclose(float(torch.linalg.norm(op()[1] - ref()[1])), 0.0, atol=1e-9)


def _flat_grads(model, r):
    rows = []
    params = list(model.parameters())
    for k in range(r.shape[0]):
        g = torch.autograd.grad(r[k], params, retain_graph=True, allow_unused=True)
        rows.append(
            torch.cat(
                [
                    (x if x is not None else torch.zeros_like(p)).reshape(-1)
                    for x, p in zip(g, params, strict=True)
                ]
            )
        )
    return torch.stack(rows)


@pytest.mark.parametrize("name", ["full_streams", "autodiff_rows"])
def test_residual_values_and_parameter_diagonal_match_dense_jacobian(name):
    model = net((3, 6, 5, 1), seed=3)
    op = compile_residual(model, blocks(), force=name, cache=PlanCache())
    r_only = op.residual()
    r, K = compile_residual(model, blocks(), quantity="gram", force=name, cache=PlanCache())()
    assert torch.allclose(r_only, r, atol=1e-12)
    r2, parts = op.parameter_diagonal()
    assert torch.allclose(r2, r, atol=1e-12)
    assert [p.shape for p in parts] == [p.shape for p in op.parameters]
    # dense reference: the Jacobian from the generic route's rows
    gen = compile_residual(
        model, blocks(), quantity="gram", force="autodiff_rows", cache=PlanCache()
    )
    from omnibias.torch.experimental.residual_compiler import _autodiff_jacobian

    _, J = _autodiff_jacobian(gen._ctx, gen.parameters, gen._ctx.xs_of(None))
    flat = torch.cat([p.reshape(-1) for p in parts])
    assert torch.allclose(flat, (J * J).sum(0), atol=1e-9, rtol=1e-9)
    assert torch.allclose(K, J @ J.T, atol=1e-9, rtol=1e-9)


def test_diagonal_curvature_with_planned_gn_diagonal_matches_the_row_loop():
    from omnibias.torch.optim import DiagonalCurvature, planned_gn_diagonal

    pts_ = [(pts(6, 3, 1), poisson), (pts(4, 3, 2), boundary)]

    def run(planned):
        model = net(seed=5)
        op = compile_residual(model, pts_, cache=PlanCache())
        kw = {"gn_diagonal": planned_gn_diagonal(op)} if planned else {}
        opt = DiagonalCurvature(model.parameters(), lr=1e-2, **kw)

        def closure():
            return _closure_residual(model, pts_)

        for _ in range(3):
            opt.step(closure)
        return torch.cat([p.detach().reshape(-1) for p in model.parameters()])

    assert torch.allclose(run(True), run(False), atol=1e-9, rtol=1e-9)


def _closure_residual(model, blocks_):
    parts = []
    for x, res in blocks_:
        x = x.clone().requires_grad_(True)
        u = model(x)
        g = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        lap = sum(
            torch.autograd.grad(g[:, k].sum(), x, create_graph=True)[0][:, k]
            for k in range(x.shape[1])
        )
        from omnibias.core.demand import Quantities

        parts.append(
            torch.as_tensor(
                res(Quantities(u=u, x=x, grad=g[..., None], grad_sq=None, lap=lap[:, None]))
            ).reshape(-1)
        )
    return torch.cat(parts)


def test_planned_levenberg_marquardt_reduces_the_loss_and_survives_a_bad_start():
    from omnibias.torch.optim import PlannedGaussNewton

    model = net((2, 10, 10, 1), seed=2)
    bl = [(pts(14, 2, 3), poisson), (pts(8, 2, 4), boundary)]
    op = compile_residual(model, bl, cache=PlanCache())
    lm = PlannedGaussNewton(op, damping=1e-4)
    first = lm.step()
    for _ in range(6):
        info = lm.step()
    assert first.accepted and first.loss_after < first.loss_before
    assert lm.history[-1].loss_after < lm.history[0].loss_before * 0.9
    assert all(h.loss_after <= h.loss_before for h in lm.history)
    assert info.algorithm in {"full_streams", "autodiff_rows"}
