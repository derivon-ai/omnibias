# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Hard-constraint boundary/initial-condition ansatz ``u = g + b N`` (torch).

What is checked:

1. *Exactness of the constraint*: the Dirichlet / box / initial conditions hold to
   machine precision for arbitrary network parameters -- they are baked into the
   architecture, not learned.
2. *Exactness of the derivatives*: ``D^alpha u`` of the wrapped field matches
   ``torch.func`` autograd to ~machine precision (float64), because the ansatz is
   assembled with the jet-level Leibniz product :func:`omnibias.torch.jet_multiply`.
3. *The payoff*: a Gauss-Newton PINN (``omnibias.torch.optim``) drives the 1-D
   Poisson residual to ~zero using **only** the interior residual -- no boundary
   loss term -- and the boundary values stay exact throughout.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn
from omnibias.torch import jet_multiply
from omnibias.torch.architectures import JetMLP
from omnibias.torch.architectures.hardbc import (
    AffineFactor,
    AffineLift,
    BoundaryMask,
    HardConstraintField,
    dirichlet_interval,
    homogeneous_box,
    initial_value,
)
from omnibias.torch.jet_mv import identity_jet, jet_gradient, jet_hessian
from omnibias.torch.optim import GaussNewton, functional_residual_fn
from torch.func import hessian, jacrev

# --- jet_multiply: the jet-level Leibniz rule ---------------------------------


def test_jet_multiply_is_polynomial_product() -> None:
    """jet(x) * jet(y) is the jet of x*y: grad = [y, x], Hessian = [[0,1],[1,0]]."""
    x0 = torch.tensor([0.3, -0.7], dtype=torch.float64)
    idj = identity_jet(x0, 2)
    prod = jet_multiply(idj[:, 0], idj[:, 1], 2, 2)
    assert torch.allclose(jet_gradient(prod, 2, 1), torch.tensor([-0.7, 0.3]).double())
    h = jet_hessian(prod, 2, 2)
    assert torch.allclose(h, torch.tensor([[0.0, 1.0], [1.0, 0.0]]).double())


def test_jet_multiply_broadcasts_scalar_times_vector() -> None:
    """A scalar mask jet ``(M,)`` multiplies a vector field jet ``(M, C)``."""
    x0 = torch.tensor([0.2, 0.5], dtype=torch.float64)
    idj = identity_jet(x0, 2)
    vec = torch.randn(idj.shape[0], 3, dtype=torch.float64)
    out = jet_multiply(idj[:, 0], vec, 2, 2)
    assert out.shape == (idj.shape[0], 3)
    assert torch.allclose(out[0], x0[0] * vec[0])


def test_jet_multiply_rejects_wrong_rows() -> None:
    x0 = torch.tensor([0.1, 0.2], dtype=torch.float64)
    idj = identity_jet(x0, 2)
    with pytest.raises(ValueError, match="rows"):
        jet_multiply(idj[:, 0], torch.zeros(2), 2, 2)


# --- Dirichlet on an interval -------------------------------------------------


def _scalar_value(field: HardConstraintField):
    def u(xi: torch.Tensor) -> torch.Tensor:
        return field.value(xi.unsqueeze(0))[0, 0]

    return u


def test_dirichlet_interval_endpoints_exact() -> None:
    net = JetMLP(1, 16, 1, depth=3, base="tanh").double()
    field = dirichlet_interval(net, 0.0, 2.0, lower_value=1.0, upper_value=-3.0)
    ends = torch.tensor([[0.0], [2.0]], dtype=torch.float64)
    u = field.value(ends)
    assert torch.allclose(u[:, 0], torch.tensor([1.0, -3.0]).double(), atol=1e-12)


def test_dirichlet_homogeneous_has_no_lift() -> None:
    net = JetMLP(1, 8, 1, depth=2).double()
    field = dirichlet_interval(net, -1.0, 1.0)
    assert field.lift is None
    ends = torch.tensor([[-1.0], [1.0]], dtype=torch.float64)
    assert torch.allclose(field.value(ends), torch.zeros(2, 1, dtype=torch.float64), atol=1e-13)


def test_dirichlet_derivatives_match_autograd() -> None:
    net = JetMLP(1, 16, 1, depth=3, base="tanh").double()
    field = dirichlet_interval(net, 0.0, 1.0, lower_value=0.5, upper_value=2.0)
    x = torch.tensor([[0.13], [0.42], [0.86]], dtype=torch.float64)
    u = _scalar_value(field)
    g_ad = torch.stack([jacrev(u)(xi) for xi in x])  # (B,1)
    h_ad = torch.stack([hessian(u)(xi) for xi in x])  # (B,1,1)
    assert torch.allclose(field.gradient(x)[:, :, 0], g_ad, atol=1e-11)
    assert torch.allclose(field.hessian(x)[:, :, :, 0], h_ad, atol=1e-11)


# --- Homogeneous Dirichlet on a box (2-D) -------------------------------------


def test_homogeneous_box_boundary_is_zero() -> None:
    net = JetMLP(2, 12, 1, depth=2, base="tanh").double()
    field = homogeneous_box(net, [0.0, 0.0], [1.0, 1.0])
    boundary = torch.tensor(
        [[0.0, 0.3], [1.0, 0.5], [0.4, 0.0], [0.6, 1.0]], dtype=torch.float64
    )
    assert torch.allclose(field.value(boundary), torch.zeros(4, 1, dtype=torch.float64), atol=1e-13)
    interior = torch.tensor([[0.5, 0.5]], dtype=torch.float64)
    assert field.value(interior).abs().item() > 0.0  # free in the interior


def test_box_derivatives_match_autograd() -> None:
    net = JetMLP(2, 10, 1, depth=2, base="tanh").double()
    field = homogeneous_box(net, [-1.0, 0.0], [1.0, 2.0])
    x = torch.tensor([[0.1, 0.7], [-0.4, 1.3]], dtype=torch.float64)
    u = _scalar_value(field)
    g_ad = torch.stack([jacrev(u)(xi) for xi in x])  # (B,2)
    h_ad = torch.stack([hessian(u)(xi) for xi in x])  # (B,2,2)
    assert torch.allclose(field.gradient(x)[:, :, 0], g_ad, atol=1e-10)
    assert torch.allclose(field.hessian(x)[:, :, :, 0], h_ad, atol=1e-10)


# --- Initial condition --------------------------------------------------------


def test_initial_value_exact_on_slice() -> None:
    net = JetMLP(2, 10, 1, depth=2, base="tanh").double()
    field = initial_value(net, t_axis=1, t0=0.0, value=2.0)
    ic = torch.tensor([[0.1, 0.0], [0.9, 0.0], [0.5, 0.0]], dtype=torch.float64)
    assert torch.allclose(field.value(ic), 2.0 * torch.ones(3, 1, dtype=torch.float64), atol=1e-13)
    later = torch.tensor([[0.5, 0.7]], dtype=torch.float64)
    assert (field.value(later) - 2.0).abs().item() > 0.0  # unconstrained for t > 0


def test_initial_value_negative_axis() -> None:
    net = JetMLP(2, 8, 1, depth=2).double()
    field = initial_value(net, t_axis=-1, t0=0.5, value=0.0)
    assert field.lift is None
    ic = torch.tensor([[0.3, 0.5], [0.8, 0.5]], dtype=torch.float64)
    assert torch.allclose(field.value(ic), torch.zeros(2, 1, dtype=torch.float64), atol=1e-13)


# --- readout consistency / arbitrary order ------------------------------------


def test_value_grad_hessian_consistent() -> None:
    net = JetMLP(2, 10, 1, depth=2, base="tanh").double()
    field = homogeneous_box(net, [0.0, 0.0], [1.0, 1.0])
    x = torch.rand(5, 2, dtype=torch.float64)
    v, g, h = field.value_grad_hessian(x)
    assert torch.allclose(v, field.value(x), atol=1e-12)
    assert torch.allclose(g, field.gradient(x), atol=1e-12)
    assert torch.allclose(h, field.hessian(x), atol=1e-12)


def test_partials_third_order_match_autograd() -> None:
    net = JetMLP(1, 14, 1, depth=2, base="tanh").double()
    field = dirichlet_interval(net, 0.0, 1.0, lower_value=1.0, upper_value=0.0)
    x = torch.tensor([[0.37]], dtype=torch.float64)
    u = _scalar_value(field)
    d3_ad = jacrev(jacrev(jacrev(u)))(x[0])  # (1,1,1)
    parts = field.partials(x, 3)
    assert torch.allclose(parts[(3,)][:, 0], d3_ad.reshape(()), atol=1e-9)


# --- training plumbing --------------------------------------------------------


def test_parameters_are_only_the_network() -> None:
    net = JetMLP(1, 8, 1, depth=2).double()
    field = dirichlet_interval(net, 0.0, 1.0, lower_value=1.0, upper_value=2.0)
    field_params = {id(p) for p in field.parameters()}
    net_params = {id(p) for p in net.parameters()}
    assert field_params == net_params  # mask + lift carry no trainable state


def test_param_gradients_flow_through_residual() -> None:
    net = JetMLP(1, 10, 1, depth=2, base="tanh").double()
    field = dirichlet_interval(net, 0.0, 1.0)
    x = torch.linspace(0, 1, 12, dtype=torch.float64).unsqueeze(1)
    loss = (field.hessian(x)[:, 0, 0, 0] ** 2).mean()
    loss.backward()
    assert all(p.grad is not None for p in field.parameters())
    assert any(p.grad.abs().sum().item() > 0 for p in field.parameters())


def test_gauss_newton_solves_poisson_without_boundary_loss() -> None:
    """u'' = -pi^2 sin(pi x), u(0)=u(1)=0; only the interior residual is trained.

    The boundary condition is structural, so the loss has a single term. Gauss-Newton
    drives it down many orders of magnitude and the endpoints stay exactly zero.
    """
    torch.manual_seed(0)
    net = JetMLP(1, 24, 1, depth=2, base="tanh").double()
    field = dirichlet_interval(net, 0.0, 1.0)
    x = torch.linspace(0, 1, 40, dtype=torch.float64).unsqueeze(1)
    f = -(math.pi**2) * torch.sin(math.pi * x[:, 0])

    class Resid(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.field = field
            self.register_buffer("x", x)
            self.register_buffer("f", f)

        def forward(self) -> torch.Tensor:
            uxx = self.field.hessian(self.x)[:, 0, 0, 0]
            return uxx - self.f

    resid = Resid()
    flat0, rfn = functional_residual_fn(resid)
    init_loss = 0.5 * float((rfn(flat0) ** 2).mean())
    gn = GaussNewton(damping=1e-2)
    final, history = gn.minimize(rfn, flat0, steps=40)
    assert history[-1] < init_loss * 1e-4  # >= 4 orders of magnitude

    torch.nn.utils.vector_to_parameters(final, list(resid.parameters()))
    xx = torch.linspace(0, 1, 101, dtype=torch.float64).unsqueeze(1)
    u = field.value(xx)[:, 0]
    u_star = torch.sin(math.pi * xx[:, 0])
    assert (u - u_star).abs().max().item() < 1e-3
    # boundary stays *exactly* satisfied, no penalty needed
    bc = field.value(torch.tensor([[0.0], [1.0]], dtype=torch.float64))
    assert bc.abs().max().item() < 1e-12


# --- validation ---------------------------------------------------------------


def test_dirichlet_interval_rejects_bad_bounds() -> None:
    net = JetMLP(1, 8, 1, depth=2).double()
    with pytest.raises(ValueError, match="must exceed"):
        dirichlet_interval(net, 1.0, 0.0)


def test_homogeneous_box_rejects_wrong_length() -> None:
    net = JetMLP(2, 8, 1, depth=2).double()
    with pytest.raises(ValueError, match="length in_dim"):
        homogeneous_box(net, [0.0], [1.0])


def test_mask_factor_axis_out_of_range() -> None:
    net = JetMLP(1, 8, 1, depth=2).double()
    bad = BoundaryMask((AffineFactor(3, 1.0, 0.0),))
    with pytest.raises(ValueError, match="out of range"):
        HardConstraintField(net, bad)


def test_lift_dim_mismatch_rejected() -> None:
    net = JetMLP(2, 8, 1, depth=2).double()
    mask = BoundaryMask((AffineFactor(0, 1.0, 0.0),))
    bad_lift = AffineLift(((0.0,), (0.0,)), (0.0, 0.0))  # in_dim 1, out_dim 2
    with pytest.raises(ValueError, match="out_dim|in_dim"):
        HardConstraintField(net, mask, bad_lift)


def test_empty_mask_and_zero_scale_rejected() -> None:
    with pytest.raises(ValueError, match="at least one factor"):
        BoundaryMask(())
    with pytest.raises(ValueError, match="non-zero"):
        AffineFactor(0, 0.0, 1.0)
