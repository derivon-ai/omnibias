# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Independent initial/boundary problem checks for quotient-compiled profiles."""

import torch
from omnibias.core.operator_spec import OperatorSpec
from omnibias.core.profile_system import ProfileSystem
from omnibias.torch.profile_system import profile_solution


def test_forced_heat_with_zero_initial_and_boundary_data() -> None:
    heat = OperatorSpec.polynomial({(1, 0): 1, (0, 2): -1})
    forced = ProfileSystem.from_operators(((heat,),), (0, 1), carrier=(-2, 0),
                                         annihilator=(1, 0, 1)).solve(((1,),))
    homogeneous = ProfileSystem.from_operators(((heat,),), (0, 1), carrier=(-1, 0),
                                              annihilator=(1, 0, 1)).solve()
    assert forced.consistent and forced.particular is not None
    t, x = torch.meshgrid(torch.linspace(0, 1, 9, dtype=torch.float64),
                          torch.linspace(0, torch.pi, 19, dtype=torch.float64), indexing="ij")
    points = torch.stack((t, x), dim=-1).requires_grad_()
    u = (profile_solution(points, forced.particular, [0., 1.])
         + profile_solution(points, homogeneous.homogeneous_basis[0], [0., 1.]))[..., 0]
    expected = (torch.exp(-t) - torch.exp(-2*t)) * torch.sin(x)
    torch.testing.assert_close(u, expected, atol=2e-13, rtol=2e-13)
    first = torch.autograd.grad(u.sum(), points, create_graph=True)[0]
    second = torch.autograd.grad(first[..., 1].sum(), points)[0]
    source = torch.exp(-2*t) * torch.sin(x)
    residual = first[..., 0] - second[..., 1] - source
    assert residual.abs().max().item() < 2e-12
    assert u[0].abs().max().item() < 2e-13
    assert u[:, [0, -1]].abs().max().item() < 2e-13
    print({"forced_heat_field_error": (u - expected).abs().max().item(),
           "forced_heat_residual": residual.abs().max().item()})


def test_laplace_dirichlet_problem_from_opposite_carriers() -> None:
    operator = OperatorSpec.laplacian()
    packs = [ProfileSystem.from_operators(((operator,),), (0, 1), carrier=(sign, 0),
                                          annihilator=(1, 0, 1)).solve().homogeneous_basis[0]
             for sign in (1, -1)]
    x, y = torch.meshgrid(torch.linspace(0, 1, 9, dtype=torch.float64),
                          torch.linspace(0, torch.pi, 19, dtype=torch.float64), indexing="ij")
    points = torch.stack((x, y), dim=-1).requires_grad_()
    normalization = 2 * torch.sinh(x.new_tensor(1.0))
    u = (profile_solution(points, packs[0], [0., 1.])
         - profile_solution(points, packs[1], [0., 1.]))[..., 0] / normalization
    expected = torch.sinh(x) * torch.sin(y) / torch.sinh(x.new_tensor(1.0))
    torch.testing.assert_close(u, expected, atol=2e-13, rtol=2e-13)
    first = torch.autograd.grad(u.sum(), points, create_graph=True)[0]
    dxx = torch.autograd.grad(first[..., 0].sum(), points, retain_graph=True)[0][..., 0]
    dyy = torch.autograd.grad(first[..., 1].sum(), points)[0][..., 1]
    assert (dxx + dyy).abs().max().item() < 2e-12
    assert u[0].abs().max().item() < 2e-13
    torch.testing.assert_close(u[-1], torch.sin(y[-1]), atol=2e-13, rtol=2e-13)
    assert u[:, [0, -1]].abs().max().item() < 2e-13
    print({"laplace_field_error": (u - expected).abs().max().item(),
           "laplace_residual": (dxx + dyy).abs().max().item()})
