# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""PyTorch residual Gram, Gauss-Newton step and NTK diagonal against explicit Jacobians."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from omnibias.core.demand import Residual
from omnibias.torch.activations.registry import get_activation
from omnibias.torch.experimental.conditioning import ConditioningPolicy, condition_report
from omnibias.torch.experimental.residual_gram import (
    deep_residual_gauss_newton,
    deep_residual_gram,
    deep_residual_ntk_diagonal,
)


def make_layers(dims, acts, seed=0):
    g = torch.Generator().manual_seed(seed)
    out = []
    for i, (a, b) in enumerate(zip(dims[:-1], dims[1:], strict=True)):
        W = (
            torch.randn(b, a, generator=g, dtype=torch.float64) * 0.6 / np.sqrt(a)
        ).requires_grad_()
        bias = (torch.randn(b, generator=g, dtype=torch.float64) * 0.1).requires_grad_()
        out.append((W, bias, acts[i]))
    return out


def forward(layers, x):
    a = x
    for W, b, s in layers:
        a = a @ W.T + b
        if s is not None:
            a = get_activation(s).forward(a)
    return a


def pts(n, d, seed):
    return torch.randn(n, d, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)


def explicit(layers, blocks):
    """Residual rows and the dense Jacobian over every parameter by autograd (nested for the Laplacian)."""
    params = [p for W, b, _ in layers for p in (W, b)]

    def rvec(*ps):
        ls = [(ps[2 * i], ps[2 * i + 1], s) for i, (_, _, s) in enumerate(layers)]
        rows = []
        for x, res, order in blocks:
            pieces = []
            for xi in x:
                f = lambda z, ls=ls: forward(ls, z[None])[0]  # noqa: E731
                u = f(xi)
                g = torch.func.jacrev(f)(xi).T if order >= 1 else None
                lap = (
                    torch.einsum("ckk->c", torch.func.jacrev(torch.func.jacrev(f))(xi))
                    if order == 2
                    else None
                )
                pieces.append(
                    res(
                        u[None],
                        None if g is None else g[None],
                        None if lap is None else lap[None],
                        xi[None],
                    )
                )
            rows.append(torch.cat([p.reshape(-1) for p in pieces]))
        return torch.cat(rows)

    J = torch.func.jacrev(rvec, argnums=tuple(range(len(params))))(*params)
    n = J[0].shape[0]
    return rvec(*params).detach(), torch.cat([j.reshape(n, -1) for j in J], dim=1).detach()


def residual_two(u, grad, lap, x):
    return lap[:, 0] + 0.5 * (grad[:, :, 0] ** 2).sum(-1) - torch.sin(x[:, 0]) * u[:, 0]


def residual_boundary(u, grad, lap, x):
    return u[:, 0] - x.sum(-1)


@pytest.mark.parametrize("strategy", ["auto", "streams", "explicit"])
@pytest.mark.parametrize(
    "dims,acts", [((3, 6, 6, 1), ("tanh", "tanh", None)), ((5, 4, 1), ("silu", None))]
)
def test_gram_matches_explicit_jacobian(dims, acts, strategy):
    layers = make_layers(dims, acts)
    D = dims[0]
    blocks = [
        (pts(4, D, 1), residual_two, 2),
        (pts(3, D, 2), residual_boundary, 0),
    ]
    r_ref, J = explicit(layers, blocks)
    r, K = deep_residual_gram(blocks, layers, strategy=strategy)
    assert torch.allclose(r, r_ref, atol=1e-10)
    assert torch.allclose(K, J @ J.T, atol=1e-9, rtol=1e-9)


def test_gauss_newton_step_and_ntk_diagonal_match_the_dense_formulas():
    layers = make_layers((4, 7, 7, 1), ("tanh", "tanh", None), seed=3)
    blocks = [(pts(5, 4, 4), residual_two, 2), (pts(3, 4, 5), residual_boundary, 0)]
    r_ref, J = explicit(layers, blocks)
    damping = 1e-3
    r, K, step = deep_residual_gauss_newton(blocks, layers, damping)
    flat = torch.linalg.solve(J @ J.T + damping * torch.eye(J.shape[0], dtype=torch.float64), r_ref)
    want = J.T @ flat
    got = torch.cat([torch.cat([dW.reshape(-1), db.reshape(-1)]) for dW, db in step])
    # the Jacobian columns are ordered (W0, b0, W1, b1, ...), as are the step blocks
    assert torch.allclose(got, want, atol=1e-8, rtol=1e-8)
    _, diag = deep_residual_ntk_diagonal(blocks, layers)
    assert torch.allclose(diag, (J * J).sum(1), atol=1e-9, rtol=1e-9)


def test_residual_object_declares_its_demand_and_wrong_demands_fail_loudly():
    layers = make_layers((3, 5, 1), ("tanh", None))
    x = pts(4, 3, 7)
    lap_only = Residual(lambda q: q.lap[:, 0] - q.u[:, 0], ["value", "lap"])
    r, K = deep_residual_gram([(x, lap_only)], layers)
    assert K.shape == (4, 4) and torch.allclose(K, K.T)
    grad_sq = Residual(lambda q: q.grad_sq[:, 0] + q.lap[:, 0], ["value", "grad_sq", "lap"])
    _, K2 = deep_residual_gram([(x, grad_sq)], layers)
    assert torch.isfinite(K2).all()
    bilap = Residual(lambda q: q.bilap[:, 0], ["value", "bilap"])
    with pytest.raises(ValueError, match="cannot provide"):
        deep_residual_gram([(x, bilap)], layers)


def test_conditioning_report_and_floor():
    layers = make_layers((3, 5, 1), ("tanh", None))
    x = pts(6, 3, 8)
    _, K = deep_residual_gram([(x, residual_two, 2)], layers)
    report = condition_report(K, 1e-9, ConditioningPolicy("floor"))
    assert report["floor_applied"] and report["effective_damping"] > 1e-9
    assert condition_report(K, 1e-3, ConditioningPolicy("report"))["status"] in {
        "ok",
        "conditioning_limited",
    }
    from omnibias.torch.experimental.conditioning import damped_solve

    r, _ = deep_residual_gram([(x, residual_two, 2)], layers)[0], None
    _, d_plain = damped_solve(K, r, 1e-9, None)
    _, d_floor = damped_solve(K, r, 1e-9, ConditioningPolicy("floor"))
    assert float(d_plain) == 1e-9 and float(d_floor) > 1e-9
    _, _, step = deep_residual_gauss_newton(
        [(x, residual_two, 2)], layers, 1e-9, conditioning=ConditioningPolicy("floor")
    )
    assert all(torch.isfinite(dW).all() for dW, _ in step)
