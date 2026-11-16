# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Modules that are not dense chains: the input-product subspace programs of ``compile_planned_operator``."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
from omnibias.torch.experimental import compile_planned_operator  # noqa: E402
from omnibias.torch.experimental.planned import analyze_input_weights  # noqa: E402
from torch import nn  # noqa: E402

D, H = 20, 5


class Residual(nn.Module):
    def __init__(self, d=D, h=H, blocks=2):
        super().__init__()
        self.inp = nn.Linear(d, h)
        self.blocks = nn.ModuleList(nn.Linear(h, h) for _ in range(blocks))
        self.out = nn.Linear(h, 1)

    def forward(self, x):
        h = torch.tanh(self.inp(x))
        for blk in self.blocks:
            h = h + torch.tanh(blk(h))
        return self.out(h)[..., 0]


class Fourier(nn.Module):
    def __init__(self, d=D, m=4, h=6):
        super().__init__()
        self.B = nn.Parameter(torch.randn(d, m, dtype=torch.float64))
        self.mid = nn.Linear(2 * m, h)
        self.out = nn.Linear(h, 1)

    def forward(self, x):
        z = x @ self.B
        phi = torch.cat([torch.sin(z), torch.cos(z)], dim=-1)
        return self.out(torch.tanh(self.mid(phi)))[..., 0]


class Attention(nn.Module):
    def __init__(self, d=D, tokens=3, width=2):
        super().__init__()
        self.t, self.w = tokens, width
        self.inp = nn.Linear(d, tokens * width, bias=False)
        self.q, self.k, self.v = (nn.Linear(width, width, bias=False) for _ in range(3))
        self.out = nn.Linear(tokens * width, 1)

    def forward(self, x):
        z = self.inp(x).unflatten(-1, (self.t, self.w))
        a = torch.softmax(self.q(z) @ self.k(z).transpose(-1, -2) / self.w**0.5, dim=-1)
        return self.out(torch.tanh((a @ self.v(z)).flatten(-2)))[..., 0]


class Untraceable(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = Residual()

    def forward(self, x):
        z = self.net.inp(x)
        return self.net.out(torch.tanh(z.reshape(*x.shape[:-1], -1)))[..., 0]


class ReadsInput(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = Residual()

    def forward(self, x):
        return self.net(x) + 0.3 * x[..., 0] ** 2


def _double(module: nn.Module) -> nn.Module:
    torch.manual_seed(0)
    return module.double()


def _x(batch=4, dim=D):
    g = torch.Generator().manual_seed(1)
    return torch.randn(batch, dim, generator=g, dtype=torch.float64)


def _reference(model, x, power=1):
    def point(z):
        return model(z[None])[0]

    g = point
    for _ in range(power):
        g = (lambda g: lambda z: torch.func.hessian(g)(z).diagonal().sum())(g)
    return torch.func.vmap(g)(x)


@pytest.mark.parametrize("make", [Residual, Fourier, Attention])
def test_non_chain_modules_reduce_the_directions(make) -> None:
    model, x = _double(make()), _x()
    op = compile_planned_operator(model, x, operator="laplacian")
    assert not op.matched and op.algorithm == "subspace_jet"
    info = op.explain()
    assert 0 < info["input_products"]["rows"] < D
    assert [c["name"] for c in info["candidates"]][:2] == ["subspace_jet", "subspace_nested"]
    ref = _reference(model, x)
    torch.testing.assert_close(op(x), ref, rtol=1e-9, atol=1e-9)
    forced = compile_planned_operator(model, x, operator="laplacian", force="taylor_jet")
    torch.testing.assert_close(forced(x), ref, rtol=1e-9, atol=1e-9)


def test_subspace_nested_serves_the_biharmonic() -> None:
    model, x = _double(Residual(d=9, h=3, blocks=1)), _x(batch=2, dim=9)
    op = compile_planned_operator(model, x, operator="laplacian", power=2)
    assert op.algorithm == "subspace_nested"
    torch.testing.assert_close(op(x), _reference(model, x, power=2), rtol=1e-8, atol=1e-8)


def test_subspace_with_a_metric_factor() -> None:
    model, x = _double(Residual()), _x()
    S = torch.randn(D, 40, generator=torch.Generator().manual_seed(3), dtype=torch.float64) / 6.0
    op = compile_planned_operator(model, x, operator="laplacian", diffusion=S.numpy())
    assert op.algorithm == "subspace_jet"

    def point(z):
        return model(z[None])[0]

    ref = torch.func.vmap(lambda z: ((S @ S.T) * torch.func.hessian(point)(z)).sum())(x)
    torch.testing.assert_close(op(x), ref, rtol=1e-9, atol=1e-9)


def test_subspace_parameter_gradient_matches_the_full_trace() -> None:
    model, x = _double(Residual()), _x()
    op = compile_planned_operator(model, x, operator="laplacian")
    ref_op = compile_planned_operator(model, x, operator="laplacian", force="taylor_jet")
    params = list(model.parameters())
    got = torch.autograd.grad((op(x) ** 2).sum(), params, allow_unused=True)
    want = torch.autograd.grad((ref_op(x) ** 2).sum(), params, allow_unused=True)
    for p, g, w in zip(params, got, want, strict=True):
        g = torch.zeros_like(p) if g is None else g
        w = torch.zeros_like(p) if w is None else w
        torch.testing.assert_close(g, w, rtol=1e-8, atol=1e-10)


def test_refused_when_the_input_is_read_elsewhere() -> None:
    model, x = _double(ReadsInput()), _x()
    rows, why = analyze_input_weights(model)
    assert rows is None and "also read" in why
    op = compile_planned_operator(model, x, operator="laplacian")
    assert op.algorithm == "taylor_jet"
    assert any(r["algorithm"] == "subspace" for r in op.explain()["refused"])
    torch.testing.assert_close(op(x), _reference(model, x), rtol=1e-9, atol=1e-9)


def test_refused_without_a_reduction() -> None:
    model, x = _double(Residual(d=4, h=6)), _x(dim=4)
    op = compile_planned_operator(model, x, operator="laplacian")
    assert op.algorithm == "taylor_jet"
    refused = {r["algorithm"]: r["reason"] for r in op.explain()["refused"]}
    assert "no reduction" in refused["subspace"]


def test_untraceable_module_is_refused_with_the_reason() -> None:
    model, x = _double(Untraceable()), _x()
    rows, why = analyze_input_weights(model)
    assert rows is None and "torch.fx" in why
    op = compile_planned_operator(model, x, operator="laplacian")
    assert op.algorithm == "taylor_jet"


def test_a_forced_program_that_cannot_run_is_an_error_and_an_unforced_one_is_retried(monkeypatch) -> None:
    from omnibias.torch.experimental import planned

    def broken(ctx):
        raise RuntimeError("out of memory")

    model, x = _double(Residual()), _x()
    monkeypatch.setitem(planned._EXECUTORS, ("laplacian", "taylor_jet"), broken)
    with pytest.raises(ValueError, match=r"no executor could run this model.*taylor_jet"):
        compile_planned_operator(model, x, operator="laplacian", force="taylor_jet")

    monkeypatch.setitem(planned._EXECUTORS, ("laplacian", "subspace_jet"), broken)
    op = compile_planned_operator(model, x, operator="laplacian")
    assert op.algorithm == "subspace_nested" and op.explain()["mode"].endswith("+retry")
