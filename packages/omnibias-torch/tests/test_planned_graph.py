# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Opt-in CUDA-graph replay of the planned operator (``graph=``): runs on a GPU, skipped without one."""

from __future__ import annotations

import pytest
import torch
from omnibias.core.planner import PlanCache
from omnibias.torch.experimental import compile_planned_operator
from torch import nn

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs need a GPU")

DEVICE = "cuda"


def _model(dims=(4, 16, 16, 1), seed=0, dtype=torch.float64):
    torch.manual_seed(seed)
    mods: list[nn.Module] = []
    for i, (a, b) in enumerate(zip(dims[:-1], dims[1:], strict=True)):
        mods.append(nn.Linear(a, b))
        if i < len(dims) - 2:
            mods.append(nn.Tanh())
    return nn.Sequential(*mods).to(device=DEVICE, dtype=dtype)


def _x(n=12, d=4, dtype=torch.float64, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, d, generator=g, dtype=torch.float64).to(device=DEVICE, dtype=dtype)


def _pair(model, x, **kw):
    kw.setdefault("operator", "laplacian")
    eager = compile_planned_operator(model, x, cache=PlanCache(), **kw)
    replay = compile_planned_operator(model, x, cache=PlanCache(), graph="auto", **kw)
    return eager, replay


def _captured(op):
    return op.explain()["graph"]["captured"]


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("power", [1, 2, 3])
def test_replay_equals_eager_and_is_captured(dtype, power):
    model, x = _model(dtype=dtype), _x(dtype=dtype)
    eager, replay = _pair(model, x, power=power)
    with torch.no_grad():
        want = eager(x)
        got = [replay(x) for _ in range(3)]
    info = replay.explain()["graph"]
    assert info["mode"] == "auto"
    if info["captured"] == 0:  # a program that cannot be captured must say why and still be right
        assert "capture failed" in info["eager_reason"], info
    else:
        assert info["eager_reason"] is None
    tol = 1e-12 if dtype == torch.float64 else 1e-4
    for g in got:
        torch.testing.assert_close(g, want, atol=tol, rtol=tol)


def test_replay_outputs_are_copies_and_follow_the_input():
    model, x = _model(), _x()
    eager, replay = _pair(model, x)
    with torch.no_grad():
        first = replay(x)
        kept = first.clone()
        other = _x(seed=7)
        second = replay(other)
        torch.testing.assert_close(first, kept)  # the next call did not overwrite the earlier value
        torch.testing.assert_close(second, eager(other), atol=1e-12, rtol=1e-12)
    assert first.data_ptr() != second.data_ptr()


def test_parameters_updated_in_place_are_followed_without_a_new_capture():
    model, x = _model(), _x()
    eager, replay = _pair(model, x)
    with torch.no_grad():
        replay(x)
        before = _captured(replay)
        for p in model.parameters():
            p.mul_(1.1)  # what an optimizer step does
        torch.testing.assert_close(replay(x), eager(x), atol=1e-12, rtol=1e-12)
    assert _captured(replay) == before == 1


def test_a_replaced_parameter_tensor_triggers_a_new_capture():
    model, x = _model(), _x()
    eager, replay = _pair(model, x)
    with torch.no_grad():
        old = replay(x)
        model[0].weight = nn.Parameter(model[0].weight.detach().clone() * 0.5)
        new = replay(x)
        torch.testing.assert_close(new, eager(x), atol=1e-12, rtol=1e-12)
    assert not torch.allclose(new, old)
    assert _captured(replay) == 1  # the stale graph was released, one entry remains


def test_calls_that_need_gradients_run_eagerly_and_graph_true_refuses_them():
    model, x = _model(), _x()
    _, auto = _pair(model, x)
    value = auto(x)  # gradients on: eager, differentiable
    assert value.requires_grad
    assert _captured(auto) == 0 and "gradients" in auto.explain()["graph"]["eager_reason"]
    strict = compile_planned_operator(model, x, operator="laplacian", cache=PlanCache(), graph=True)
    with pytest.raises(RuntimeError, match="cannot be replayed"):
        strict(x)
    with torch.no_grad():
        assert torch.isfinite(strict(x)).all()


def test_a_failed_capture_leaves_the_device_usable(monkeypatch):
    model, x = _model(), _x()
    eager, replay = _pair(model, x)

    def boom(self, *args, **kwargs):
        raise RuntimeError("synthetic capture failure")

    with torch.no_grad():
        monkeypatch.setattr(type(replay), "_call_impl", boom)
        got = None
        try:
            got = replay._capture(x, ())
        finally:
            monkeypatch.undo()
        assert got is None and "synthetic capture failure" in replay.explain()["graph"]["eager_reason"]
        torch.testing.assert_close(replay(x), eager(x), atol=1e-12, rtol=1e-12)  # random draws and capture work again
        assert torch.rand(2, device=DEVICE).shape == (2,)
