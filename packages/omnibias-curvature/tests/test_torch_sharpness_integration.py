# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Integration tests: exact-curvature sharpness on losses that *already contain
a derivative* -- the PINN residual / CNF divergence structure.

The sharpness HVP is a reverse-over-reverse pass. When the loss itself is built
from an inner ``torch.autograd.grad`` (a spatial derivative, as in a PDE
residual or a continuous-normalizing-flow log-density), differentiating it for
curvature is a genuine *third*-order autograd. These tests pin that this works
and stays exact:

1. ``test_hvp_exact_through_inner_derivative_loss`` (no extra deps, runs in CI)
   -- a PDE-residual loss ``L(theta)`` depending on ``du_theta/dx``; the
   matrix-free HVP must equal the dense Hessian action.
2. ``test_sharpness_objectives_through_cnf_nll`` (skipped unless ``omnibias.score.flow``
   is installed) -- the real exact-divergence CNF negative log-likelihood: the
   differentiable ``sharpness_aware_loss`` / ``sam_objective`` yield finite,
   parameter-shaped gradients, and the flatness readout is finite.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")
from omnibias.curvature.torch import sharpness as S  # noqa: E402
from omnibias.torch.architectures import JetMLP  # noqa: E402


def test_hvp_exact_through_inner_derivative_loss():
    torch.manual_seed(0)
    net = JetMLP(1, 4, 1, depth=2, base="tanh").double()
    x = torch.linspace(-1.0, 1.0, 8, dtype=torch.float64).reshape(-1, 1).requires_grad_(True)
    params = [p for p in net.parameters() if p.requires_grad]

    def residual_loss():
        u = net(x)
        (du,) = torch.autograd.grad(u.sum(), x, create_graph=True)  # du/dx, keep graph
        r = du - u  # toy first-order PDE residual du/dx = u
        return (r ** 2).mean()

    # Matrix-free HVP == dense Hessian action, even though the loss contains an
    # inner spatial derivative (so this is a third-order autograd overall).
    Hd = S.dense_hessian(residual_loss(), params)
    v = S._rand_like(params, generator=torch.Generator().manual_seed(2))
    Hv = S.hvp(residual_loss(), params, v)
    Hv_flat = torch.cat([h.reshape(-1) for h in Hv]).detach()
    v_flat = torch.cat([vi.reshape(-1) for vi in v])
    assert float((Hv_flat - Hd @ v_flat).abs().max()) < 1e-9

    # The differentiable penalty backprops to finite, parameter-shaped grads.
    obj = S.sharpness_aware_loss(residual_loss(), params, lam=1e-3, measure="frobenius",
                                 n_samples=2, generator=torch.Generator().manual_seed(3))
    net.zero_grad()
    obj.backward()
    for p in params:
        assert p.grad is not None and p.grad.shape == p.shape
        assert torch.isfinite(p.grad).all()


def test_sharpness_objectives_through_cnf_nll():
    pytest.importorskip("omnibias.score.flow.torch.ops")
    from omnibias.score.flow.torch.ops import log_prob

    torch.manual_seed(0)
    log2pi = math.log(2.0 * math.pi)

    def base_lp(z):
        return -0.5 * (z.pow(2).sum(-1) + z.shape[-1] * log2pi)

    class Vel(torch.nn.Module):
        def __init__(self, d=2, h=4):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(d + 1, h), torch.nn.Tanh(), torch.nn.Linear(h, d),
            ).double()
            with torch.no_grad():  # near-identity, well-conditioned flow
                for m in self.net:
                    if isinstance(m, torch.nn.Linear):
                        m.weight.mul_(0.1)
                        m.bias.mul_(0.1)

        def forward(self, t, x):
            tt = torch.full((x.shape[0], 1), float(t), dtype=x.dtype)
            return self.net(torch.cat([x, tt], dim=-1))

    net = Vel()
    params = [p for p in net.parameters() if p.requires_grad]
    x = 0.5 * torch.randn(6, 2, dtype=torch.float64)

    def nll():
        return -log_prob(net, x, 1.0, 0.0, base_lp, steps=2, method="euler").mean()

    # differentiable frobenius penalty through the exact-divergence NLL
    obj = S.sharpness_aware_loss(nll(), params, lam=1e-3, measure="frobenius",
                                 n_samples=1, generator=torch.Generator().manual_seed(1))
    net.zero_grad()
    obj.backward()
    for p in params:
        assert p.grad is not None and p.grad.shape == p.shape
        assert torch.isfinite(p.grad).all()

    # exact second-order SAM objective also differentiates end-to-end
    net.zero_grad()
    S.sam_objective(nll(), params, rho=0.05, iters=2,
                    generator=torch.Generator().manual_seed(2)).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)

    # cheap flatness readout of the CNF NLL is finite
    lam_max = float(S.top_eigenvalue(nll(), params, iters=6,
                                     generator=torch.Generator().manual_seed(0)))
    assert math.isfinite(lam_max)
