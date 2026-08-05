# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Closed-form activation antiderivative tests."""

from __future__ import annotations

import pytest
import torch
from omnibias.torch import OperatorBlock, OperatorMultiBiasUnit
from omnibias.torch.activations import get_activation

SUPPORTED_INTEGRALS = [
    "sigmoid",
    "tanh",
    "gaussian",
    "exp",
    "huber",
    "arctan",
    "log1pu2",
    "relu",
    "gelu",
    "sin",
    "cos",
    "sinh",
    "cosh",
    "tan",
    "cot",
    "sech",
    "coth",
    "softabs",
    "smooth_sign",
]


def _safe_points(name: str) -> torch.Tensor:
    if name in {"cot", "coth"}:
        return torch.linspace(0.35, 1.35, 17, dtype=torch.double)
    if name in {"huber", "relu"}:
        return torch.tensor([-1.4, -0.7, -0.2, 0.2, 0.7, 1.4], dtype=torch.double)
    return torch.linspace(-0.8, 0.8, 17, dtype=torch.double)


@pytest.mark.parametrize("name", SUPPORTED_INTEGRALS)
def test_integral_derivative_recovers_activation(name: str) -> None:
    spec = get_activation(name)
    assert spec.integral is not None
    z = _safe_points(name).requires_grad_(True)
    primitive = spec.integral(z)
    (grad,) = torch.autograd.grad(primitive.sum(), z)
    ref = spec.forward(z.detach())
    assert torch.allclose(grad, ref, atol=1e-9, rtol=1e-9), name


def test_multibias_analytic_integral_matches_antiderivative_window() -> None:
    ombu = OperatorMultiBiasUnit(
        num_channels=2,
        K=2,
        base="sigmoid",
        biases_init=torch.tensor([[-0.5, 0.5], [0.25, 0.75]]),
        signs_init=torch.tensor([1.0, -1.0]),
    )
    z = torch.linspace(-1.0, 1.0, 8).unsqueeze(-1).expand(8, 2).contiguous()
    out = ombu.analytic_integral(z)
    ref = torch.nn.functional.softplus(z + ombu.biases[:, 0])
    ref = ref - torch.nn.functional.softplus(z + ombu.biases[:, 1])
    assert torch.allclose(out, ref, atol=1e-6)


def test_integral_window_derivative_matches_literal_difference() -> None:
    ombu = OperatorMultiBiasUnit(
        num_channels=1,
        K=2,
        base="tanh",
        biases_init=torch.tensor([[-0.4, 0.6]]),
        signs_init=torch.tensor([1.0, -1.0]),
    )
    z = torch.linspace(-1.0, 1.0, 21, dtype=torch.double).unsqueeze(-1).requires_grad_(True)
    ombu = ombu.double()
    window = ombu.analytic_integral(z)
    (grad,) = torch.autograd.grad(window.sum(), z)
    assert torch.allclose(grad, ombu(z), atol=1e-10, rtol=1e-10)


def test_operator_integral_derivative_recovers_band() -> None:
    integral = OperatorBlock(op="integral", base="tanh", channels=1, init_delta=0.75).double()
    band = OperatorBlock(op="band", base="tanh", channels=1, init_delta=0.75).double()
    with torch.no_grad():
        band.ombu.biases[:, 0] = integral.ombu.biases[:, 0] - 0.5 * torch.nn.functional.softplus(
            integral.ombu.biases[:, 1]
        )
        band.ombu.biases[:, 1] = integral.ombu.biases[:, 0] + 0.5 * torch.nn.functional.softplus(
            integral.ombu.biases[:, 1]
        )
    z = torch.linspace(-1.0, 1.0, 21, dtype=torch.double).unsqueeze(-1).requires_grad_(True)
    area = integral(z)
    (grad,) = torch.autograd.grad(area.sum(), z)
    assert torch.allclose(grad, band(z), atol=1e-10, rtol=1e-10)
    assert torch.allclose(integral.literal_forward(z.detach()), band(z.detach()), atol=1e-10, rtol=1e-10)


def test_normalized_integral_collapses_to_activation_for_small_width() -> None:
    block = OperatorBlock(
        op="integral",
        base="sigmoid",
        channels=2,
        init_delta=0.0,
        normalize_integral=True,
        integral_small_width=1e-3,
    )
    z = torch.randn(5, 2)
    out = block(z)
    assert torch.allclose(out, torch.sigmoid(z), atol=1e-6)


# --- closed-form antiderivative vs an independent numeric quadrature ----------
# The tests above check the integral operator the *derivative* way (autodiff of
# the antiderivative recovers the integrand, i.e. d/dz of FTC). These check the
# *integral* way: the closed-form definite integral ``Sigma(b) - Sigma(a)`` must
# equal an independent composite-Simpson quadrature of the integrand, with no
# shared code between the two.

# Smooth integrands only: Simpson's O(h^4) error needs a continuous 4th
# derivative, so the C^0/C^1 kinks of relu/huber are excluded here.
_SMOOTH_QUAD = [
    "sigmoid", "tanh", "gaussian", "exp", "arctan", "log1pu2",
    "sin", "cos", "sinh", "cosh", "sech", "tan", "cot", "coth",
    "softabs", "smooth_sign",
]


def _quad_interval(name: str) -> tuple[float, float]:
    """An asymmetric integration window kept clear of each activation's poles."""
    if name in {"cot", "coth"}:
        return (0.35, 1.35)
    if name == "tan":
        return (-0.8, 0.8)
    return (-0.9, 0.7)


def _simpson(f, a: float, b: float, n: int = 4000) -> torch.Tensor:  # type: ignore[no-untyped-def]
    """Composite Simpson rule on ``n`` (even) panels -- an independent oracle."""
    xs = torch.linspace(a, b, n + 1, dtype=torch.double)
    ys = f(xs)
    w = torch.ones(n + 1, dtype=torch.double)
    w[1:-1:2] = 4.0
    w[2:-1:2] = 2.0
    return (b - a) / n / 3.0 * (w * ys).sum()


@pytest.mark.parametrize("name", _SMOOTH_QUAD)
def test_closed_form_integral_matches_independent_simpson(name: str) -> None:
    spec = get_activation(name)
    assert spec.integral is not None
    a, b = _quad_interval(name)
    quad = _simpson(lambda z: spec.forward(z), a, b)
    closed = spec.integral(torch.tensor(b, dtype=torch.double)) - spec.integral(
        torch.tensor(a, dtype=torch.double)
    )
    assert torch.allclose(closed, quad, atol=1e-7, rtol=1e-7), name


def test_ombu_analytic_integral_matches_independent_simpson() -> None:
    ombu = OperatorMultiBiasUnit(
        num_channels=1,
        K=2,
        base="sigmoid",
        biases_init=torch.tensor([[-0.5, 0.5]]),
        signs_init=torch.tensor([1.0, -1.0]),
    ).double()
    a, b = -1.3, 0.9
    quad = _simpson(lambda z: ombu(z.unsqueeze(-1)).squeeze(-1), a, b)

    def antiderivative(x: float) -> torch.Tensor:
        return ombu.analytic_integral(torch.tensor([[x]], dtype=torch.double)).reshape(())

    closed = antiderivative(b) - antiderivative(a)
    assert torch.allclose(closed, quad, atol=1e-7, rtol=1e-7)
