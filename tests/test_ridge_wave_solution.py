# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Fit wave data in a compiled family and independently check the full problem."""

import torch
from omnibias.core.operator_spec import OperatorSpec
from omnibias.core.ridge_system import RidgeSystem
from omnibias.torch.ridge_system import ridge_solution


def test_compiled_wave_family_solves_initial_boundary_problem() -> None:
    identity = OperatorSpec.mixed((0, 0))
    wave = (
        (OperatorSpec.polynomial({(1, 0): -1}), identity, None),
        (OperatorSpec.polynomial({(0, 1): -1}), None, identity),
        (None, OperatorSpec.mixed((1, 0)), OperatorSpec.polynomial({(0, 1): -4})),
    )
    frequencies = (1, 3)
    pairs = tuple(
        tuple(RidgeSystem.from_operators(wave, (sign * 2 * k, k)).kernel_basis(1)[0]
              for sign in (1, -1))
        for k in frequencies
    )

    def features(points: torch.Tensor) -> torch.Tensor:
        return torch.stack([
            sum(ridge_solution(points, pack, activation="sin")
                / float(pack.coefficients[0][0]) for pack in pair) / 2
            for pair in pairs
        ], dim=-1)

    initial_x = torch.linspace(0, torch.pi, 17, dtype=torch.float64)
    initial_points = torch.stack((torch.zeros_like(initial_x), initial_x), dim=-1)
    initial_values = initial_x.sin() + (3 * initial_x).sin() / 3
    design = features(initial_points)[:, 0, :]
    amplitudes = torch.linalg.lstsq(design, initial_values).solution
    torch.testing.assert_close(amplitudes, initial_x.new_tensor([1, 1/3]),
                               atol=2e-14, rtol=0)

    t, x = torch.meshgrid(torch.linspace(0, 1.25, 11, dtype=torch.float64),
                          torch.linspace(0, torch.pi, 23, dtype=torch.float64), indexing="ij")
    points = torch.stack((t, x), dim=-1).requires_grad_()
    actual = features(points) @ amplitudes
    # Analytic oracle constructed directly, independently of syzygy coefficients.
    expected = torch.stack((
        x.sin() * (2*t).cos() + (3*x).sin() * (6*t).cos() / 3,
        -2*x.sin() * (2*t).sin() - 2*(3*x).sin() * (6*t).sin(),
        x.cos() * (2*t).cos() + (3*x).cos() * (6*t).cos(),
    ), dim=-1)
    torch.testing.assert_close(actual, expected, atol=2e-14, rtol=0)

    derivatives = [torch.autograd.grad(actual[..., component].sum(), points,
                                       retain_graph=True)[0] for component in range(3)]
    residual = torch.stack((actual[..., 1] - derivatives[0][..., 0],
                            actual[..., 2] - derivatives[0][..., 1],
                            derivatives[1][..., 0] - 4 * derivatives[2][..., 1]), dim=-1)
    assert residual.abs().max().item() < 2e-14
    assert actual[:, [0, -1], 0].abs().max().item() < 2e-14
    assert actual[0, :, 1].abs().max().item() < 2e-14
    torch.testing.assert_close((features(initial_points) @ amplitudes)[:, 0],
                               initial_values, atol=2e-14, rtol=0)
    print({"field_max_error": (actual - expected).abs().max().item(),
           "pde_max_residual": residual.abs().max().item(),
           "boundary_max_error": actual[:, [0, -1], 0].abs().max().item(),
           "initial_velocity_max_error": actual[0, :, 1].abs().max().item()})
