# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Planner v2 on the PyTorch executors: new pool members, training objective, estimators and auto modes."""

from __future__ import annotations

import os

import numpy as np
import pytest

if "TRITON_INTERPRET" not in os.environ:
    import torch as _t

    if not _t.cuda.is_available():
        os.environ["TRITON_INTERPRET"] = "1"  # the fused kernels run in the interpreter on a CPU

import torch  # noqa: E402
from omnibias.core.planner import AUTO_TRAINING_USES, PlanCache  # noqa: E402
from omnibias.torch.activations.registry import get_activation  # noqa: E402
from omnibias.torch.experimental import compile_planned_operator  # noqa: E402


def _layers(dims, act="tanh", seed=0):
    g = torch.Generator().manual_seed(seed)
    out = []
    for i, (a, b) in enumerate(zip(dims[:-1], dims[1:], strict=True)):
        W = (
            torch.randn(b, a, generator=g, dtype=torch.float64) * 0.6 / np.sqrt(a)
        ).requires_grad_()
        bias = (torch.randn(b, generator=g, dtype=torch.float64) * 0.1).requires_grad_()
        out.append((W, bias, act if i < len(dims) - 2 else None))
    return out


def _forward(ls, x):
    a = x
    for W, b, s in ls:
        a = a @ W.T + b
        if s is not None:
            a = get_activation(s).forward(a)
    return a


def _x(n, d, seed=1):
    return torch.randn(n, d, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)


def _hess_trace(ls, x, power=1):
    def f(z):
        return _forward(ls, z[None])[0, 0]

    g = f
    for _ in range(power):
        g = (lambda h: lambda z: torch.trace(torch.func.hessian(h)(z)))(g)
    return torch.stack([g(xi) for xi in x])


def _op(ls, x, **kw):
    kw.setdefault("cache", PlanCache())
    return compile_planned_operator(ls, x, **kw)


@pytest.mark.parametrize("name", ["projected", "taylor_closed", "fused_streams"])
def test_exact_laplacian_programs_agree_with_the_dense_hessian(name):
    ls, x = _layers([3, 16, 16, 1]), _x(6, 3)
    op = _op(ls, x, force=name)
    assert op.algorithm == name
    torch.testing.assert_close(
        op(x).detach().reshape(-1), _hess_trace(ls, x).detach(), rtol=1e-9, atol=1e-10
    )


def test_taylor_closed_bilaplacian_with_coordinate_subset():
    ls, x = _layers([3, 12, 12, 1], seed=2), _x(4, 3, seed=2)
    full = _op(ls, x, power=2, force="taylor_closed")
    torch.testing.assert_close(
        full(x).detach().reshape(-1), _hess_trace(ls, x, power=2).detach(), rtol=1e-7, atol=1e-8
    )
    sub = _op(ls, x, power=2, coordinates=[0, 2], force="taylor_closed")

    def point(xi):
        def lap(z):
            H = torch.func.hessian(lambda w: _forward(ls, w[None])[0, 0])(z)
            return H[0, 0] + H[2, 2]

        H2 = torch.func.hessian(lap)(xi)
        return H2[0, 0] + H2[2, 2]

    ref = torch.stack([point(xi) for xi in x])
    torch.testing.assert_close(sub(x).detach().reshape(-1), ref.detach(), rtol=1e-7, atol=1e-8)


def test_fused_gradients_match_the_stream_engine_and_are_planned_for_training():
    x = _x(9, 3, seed=5)
    a, b = _layers([3, 16, 16, 1], seed=5), _layers([3, 16, 16, 1], seed=5)
    fused = _op(a, x, force="fused_streams", differentiate="params")
    streams = _op(b, x, force="projected", differentiate="params")
    ga = torch.autograd.grad((fused(x) ** 2).sum(), [t for W, bb, _ in a for t in (W, bb)])
    gb = torch.autograd.grad((streams(x) ** 2).sum(), [t for W, bb, _ in b for t in (W, bb)])
    for u, v in zip(ga, gb, strict=True):
        torch.testing.assert_close(u, v, rtol=1e-8, atol=1e-10)
    both = _op(a, x, differentiate="params2")
    assert "fused_streams" not in [c["name"] for c in both.explain()["candidates"]]


@pytest.mark.parametrize("name", ["hutchinson", "stde", "sdgd"])
def test_estimators_are_unbiased(name):
    D = 6
    ls, x = _layers([D, 10, 1], seed=6), _x(3, D, seed=6)
    ref = _hess_trace(ls, x).detach().numpy()
    op = _op(ls, x, exact=False, force=name, accuracy=0.9 if name == "sdgd" else 0.3)
    draws = np.stack([op(x, key=k).detach().numpy().reshape(-1) for k in range(256)])
    err = draws.std(axis=0, ddof=1) / np.sqrt(len(draws))
    assert np.all(np.abs(draws.mean(axis=0) - ref) <= 5.0 * err + 1e-9)
    assert np.all(draws.std(axis=0) > 0)


def test_estimator_needs_a_key_in_training_and_probes_differ_per_point():
    ls = _layers([4, 8, 1], seed=7)
    x = _x(1, 4, seed=7).repeat(6, 1)
    train = _op(ls, x, exact=False, force="hutchinson", differentiate="params")
    with pytest.raises(ValueError, match="key"):
        train(x)
    out = train(x, key=3).detach().numpy().reshape(-1)
    assert np.ptp(out) > 0


def test_training_objective_auto_mode_is_timed_and_recorded(tmp_path):
    ls, x = _layers([3, 16, 16, 1]), _x(32, 3)
    path = str(tmp_path / "plans.json")
    op = _op(
        ls,
        x,
        differentiate="params",
        mode="auto",
        expected_uses="auto",
        cache=path,
    )
    info = op.explain()
    assert info["objective"] == "params" and info["mode"] == "find"
    assert info["uses_source"] == "auto" and info["expected_uses"] == AUTO_TRAINING_USES
    assert info["measured_seconds"]
    quiet = _op(ls, x, differentiate="params", mode="auto", expected_uses="auto", uses_hint=4)
    assert quiet.explain()["mode"] == "heuristic" and quiet.explain()["uses_source"] == "hint"


def test_training_objective_needs_parameters_that_require_grad():
    ls = [(W.detach(), b.detach(), s) for W, b, s in _layers([3, 8, 1])]
    with pytest.raises(ValueError, match="require grad"):
        _op(ls, _x(4, 3), differentiate="params")
