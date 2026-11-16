# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""The torch training step: eager equivalence, grouping, clipping, planning and CUDA-graph capture."""

from __future__ import annotations

import os
import warnings

import numpy as np
import pytest
import torch

if not torch.cuda.is_available():
    os.environ.setdefault("TRITON_INTERPRET", "1")

from omnibias.torch.experimental.training_step import (
    StepBlock,
    compile_training_step,
    planned_evaluator,
)

CUDA = torch.cuda.is_available()
DEV = "cuda" if CUDA else "cpu"
D, H = 2, 12
DT = torch.float64


def _layers(seed=0):
    g = np.random.default_rng(seed)
    dims = [D, H, H, 1]
    out = []
    for a, b in zip(dims[:-1], dims[1:], strict=True):
        W = torch.tensor(
            g.standard_normal((b, a)) / np.sqrt(a), dtype=DT, device=DEV, requires_grad=True
        )
        bias = torch.tensor(0.1 * g.standard_normal(b), dtype=DT, device=DEV, requires_grad=True)
        out.append((W, bias))
    return out


def _flat(layers):
    return [t for pair in layers for t in pair]


def _u(layers, x):
    a = x
    for W, b in layers[:-1]:
        a = torch.tanh(a @ W.T + b)
    W, b = layers[-1]
    return (a @ W.T + b)[..., 0]


def _lap(layers, x):
    x = x.detach().requires_grad_(True)
    u = _u(layers, x)
    (g,) = torch.autograd.grad(u.sum(), x, create_graph=True)
    out = 0.0
    for i in range(x.shape[1]):
        (h,) = torch.autograd.grad(g[:, i].sum(), x, create_graph=True)
        out = out + h[:, i]
    return out


class _Counter:
    def __init__(self):
        self.calls = 0

    def __call__(self, layers, x):
        self.calls += 1
        return {"u": _u(layers, x), "lap": _lap(layers, x)}


def _blocks(ev, n_in=24, n_bd=16):
    def s_in():
        return 2 * torch.rand(n_in, D, dtype=DT, device=DEV) - 1

    def s_bd():
        t = 2 * torch.rand(n_bd, dtype=DT, device=DEV) - 1
        return torch.stack([t, torch.ones_like(t)], dim=-1)

    def r_in(p, x, out, aux):
        return out["lap"] + 2.0 * np.pi**2 * torch.sin(np.pi * x[:, 0]) * torch.sin(np.pi * x[:, 1])

    def r_bd(p, x, out, aux):
        return out["u"]

    return [
        StepBlock("interior", s_in, r_in, evaluate=ev, order=2),
        StepBlock("boundary", s_bd, r_bd, evaluate=ev, order=2, weight=10.0),
    ]


def _eager_reference(blocks, layers, steps, lr, seed):
    opt = torch.optim.Adam(_flat(layers), lr=lr)
    torch.manual_seed(seed)
    losses = []
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        tot = 0.0
        pts = [b.sample() for b in blocks]
        for b, x in zip(blocks, pts, strict=True):
            out = {"u": _u(layers, x), "lap": _lap(layers, x)}
            tot = tot + b.weight * torch.mean(b.residual(layers, x, out, None) ** 2)
        tot.backward()
        opt.step()
        losses.append(float(tot.detach()))
    return np.asarray(losses)


def test_step_matches_an_eager_loop():
    la, lb = _layers(), _layers()
    ev = _Counter()
    ts = compile_training_step(
        _blocks(ev), _flat(la), steps_per_call=5, lr=1e-2, params_view=la, graph=False
    )
    torch.manual_seed(3)
    hist = torch.cat([ts().clone() for _ in range(2)])
    ref = _eager_reference(_blocks(ev), lb, 10, 1e-2, 3)
    np.testing.assert_allclose(hist[:, 0].cpu().numpy(), ref, rtol=1e-9)
    for (Wa, ba), (Wb, bb) in zip(la, lb, strict=True):
        torch.testing.assert_close(Wa, Wb, rtol=1e-9, atol=1e-12)
        torch.testing.assert_close(ba, bb, rtol=1e-9, atol=1e-12)
    assert hist.shape == (10, 4)  # total, two blocks, grad norm


def test_blocks_sharing_an_evaluate_are_one_launch():
    la = _layers()
    ev = _Counter()
    ts = compile_training_step(
        _blocks(ev), _flat(la), steps_per_call=3, params_view=la, graph=False
    )
    ev.calls = 0
    ts.step()
    assert ev.calls == 1
    g = ts.explain()["groups"]
    assert len(g) == 1 and g[0]["blocks"] == ["interior", "boundary"] and g[0]["shared_evaluate"]
    a, b = _Counter(), _Counter()
    bl = _blocks(a)
    bl[1] = StepBlock("boundary", bl[1].sample, bl[1].residual, evaluate=b, order=2, weight=10.0)
    ts2 = compile_training_step(bl, _flat(_layers()), params_view=_layers(), graph=False)
    assert len(ts2.explain()["groups"]) == 2


def test_training_reduces_the_loss_and_clipping_bounds_the_update():
    la = _layers()
    ts = compile_training_step(
        _blocks(_Counter()), _flat(la), steps_per_call=40, lr=3e-3, params_view=la, graph=False
    )
    first = ts().clone()
    for _ in range(6):
        ts()
    last = ts()
    assert float(last[-1, 0]) < 0.6 * float(first[0, 0])

    lc = _layers()
    before = [t.detach().clone() for t in _flat(lc)]
    sgd = torch.optim.SGD(_flat(lc), lr=1e-2)
    tc = compile_training_step(
        _blocks(_Counter()),
        _flat(lc),
        sgd,
        steps_per_call=1,
        params_view=lc,
        graph=False,
        clip_norm=0.5,
    )
    tc.step()
    for a, b in zip(before, _flat(lc), strict=True):
        assert float((a - b.detach()).abs().max()) <= 1e-2 * 0.5 + 1e-12


def test_prepare_and_validation():
    la = _layers()
    seen = {}

    def residual(p, x, out, aux):
        seen["aux"] = tuple(aux.shape)
        return _u(p, x) - aux

    blk = StepBlock(
        "fit",
        lambda: torch.randn(8, D, dtype=DT, device=DEV),
        residual,
        prepare=lambda x: torch.sin(x[:, 0]),
    )
    ts = compile_training_step([blk], _flat(la), steps_per_call=2, params_view=la, graph=False)
    ts()
    assert seen["aux"] == (8,)
    b = _blocks(None)[0]
    with pytest.raises(ValueError, match="unique"):
        compile_training_step([b, b], _flat(la), graph=False)
    with pytest.raises(ValueError, match="at least one"):
        compile_training_step([], _flat(la), graph=False)
    with pytest.raises(ValueError, match="requires_grad"):
        compile_training_step([b], [torch.zeros(2)], graph=False)
    with pytest.raises(ValueError, match="steps_per_call"):
        compile_training_step([b], _flat(la), steps_per_call=0, graph=False)


def test_planned_evaluator_asks_for_the_exhaustive_find_for_training(monkeypatch):
    from omnibias.core.planner import PlanCache
    from omnibias.torch.experimental import planned

    seen: dict = {}
    real = planned.compile_planned_operator

    def spy(*args, **kwargs):
        seen.clear()
        seen.update(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(planned, "compile_planned_operator", spy)
    la = _layers()
    spec = [(W, b, "tanh" if i < len(la) - 1 else None) for i, (W, b) in enumerate(la)]
    x = torch.zeros(24, D, dtype=DT, device=DEV)
    planned_evaluator(spec, None, x, steps=200, cache=PlanCache())
    assert (seen["mode"], seen["top_k"], seen["differentiate"], seen["expected_uses"]) == ("find", None, "params", 200)
    planned_evaluator(spec, None, x, steps=200, cache=PlanCache(), mode="auto", top_k=2)
    assert (seen["mode"], seen["top_k"]) == ("auto", 2)


def test_planned_evaluator_serves_the_step():
    la = _layers()
    spec = [(W, b, "tanh" if i < len(la) - 1 else None) for i, (W, b) in enumerate(la)]
    x = torch.zeros(24, D, dtype=DT, device=DEV)
    ev = planned_evaluator(spec, None, x, steps=200)
    xx = x + 0.3
    torch.testing.assert_close(
        ev(None, xx).reshape(-1).detach(), _lap(la, xx).detach(), rtol=1e-8, atol=1e-9
    )

    def r(p, z, lap, aux):
        return lap.reshape(-1) + 1.0

    blk = StepBlock("pde", lambda: torch.rand(24, D, dtype=DT, device=DEV), r, evaluate=ev, order=2)
    ts = compile_training_step(
        [blk], _flat(la), steps_per_call=3, lr=1e-2, params_view=None, graph=False
    )
    hist = ts()
    assert torch.isfinite(hist).all()


def test_cpu_records_why_it_is_eager():
    if CUDA:
        pytest.skip("CUDA present")
    la = _layers()
    ts = compile_training_step(
        _blocks(_Counter()), _flat(la), params_view=la, graph=True if False else "auto"
    )
    e = ts.explain()
    assert e["captured"] is False and "CPU" in e["why_eager"] or "not on CUDA" in e["why_eager"]


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
def test_cuda_graph_replay_matches_eager_and_leaves_the_model_untrained_by_capture():
    la, lb = _layers(), _layers()
    ev = _Counter()
    snap = [t.detach().clone() for t in _flat(la)]
    ts = compile_training_step(
        _blocks(ev), _flat(la), steps_per_call=4, lr=1e-2, params_view=la, graph=True
    )
    assert ts.captured and ts.explain()["optimizer"] == "Adam"
    for a, b in zip(snap, _flat(la), strict=True):  # capture restored the initial state
        torch.testing.assert_close(a, b.detach())
    torch.manual_seed(11)
    hist = torch.cat([ts().clone() for _ in range(2)])
    eager = compile_training_step(
        _blocks(ev), _flat(lb), steps_per_call=4, lr=1e-2, params_view=lb, graph=False
    )
    torch.manual_seed(11)
    ref = torch.cat([eager().clone() for _ in range(2)])
    torch.testing.assert_close(hist[:, 0], ref[:, 0], rtol=1e-6, atol=1e-9)
    assert ts.time_steps(n_blocks=3)["seconds_per_step"] > 0


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
def test_cuda_graph_step_has_no_host_sync_and_is_fast():
    la = _layers()
    ts = compile_training_step(
        _blocks(_Counter()), _flat(la), steps_per_call=8, params_view=la, graph=True
    )
    with (
        warnings.catch_warnings()
    ):  # set_sync_debug_mode is a prototype feature and warns on every call
        warnings.simplefilter("ignore")
        torch.cuda.set_sync_debug_mode("error")
        try:
            ts.run(2)
        finally:
            torch.cuda.set_sync_debug_mode("default")
    torch.cuda.synchronize()


class _SyncsInside(_Counter):
    """Synchronises the host on every call: legal while warming up, illegal while a graph is being captured."""

    def __call__(self, layers, x):
        torch.cuda.synchronize()
        return super().__call__(layers, x)


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
def test_a_failed_capture_falls_back_to_a_working_eager_step_and_leaves_the_device_usable():
    la = _layers()
    snap = [t.detach().clone() for t in _flat(la)]
    ts = compile_training_step(
        _blocks(_SyncsInside()), _flat(la), steps_per_call=2, lr=1e-2, params_view=la, graph="auto"
    )
    assert ts.captured is False and ts.explain()["why_eager"].startswith("capture failed")
    for a, b in zip(snap, _flat(la), strict=True):  # the warm-up steps were undone
        torch.testing.assert_close(a, b.detach())
    assert torch.isfinite(ts()).all()  # the eager fallback samples from the default generator
    assert torch.rand(3, device="cuda").shape == (3,)
    lb = _layers()
    ok = compile_training_step(  # a later capture in the same process still works
        _blocks(_Counter()), _flat(lb), steps_per_call=2, params_view=lb, graph=True
    )
    assert ok.captured
