# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Beta-tempered smooth-surrogate tower tests (PyTorch).

Checks the differentiable ``beta``-tempered family that complements the hard
a.e. towers:

* beta-scaling identities (``soft_relu``, ``soft_step``),
* the ``tempered()`` combinator wiring (``forward`` / ``derivative`` route
  through the tower),
* convergence to the hard activation as ``beta -> inf`` (and the emerging
  Dirac bump),
* ``swish(beta=1) == silu``; ``soft_leaky_relu -> leaky_relu``,
* the ``TemperedActivation`` / ``LearnablePReLU`` blocks and a learnable-``beta``
  gradient (finite-difference).
"""

from __future__ import annotations

import pytest
import torch
from omnibias.torch import LearnablePReLU, TemperedActivation
from omnibias.torch.activations import get_activation
from omnibias.torch.activations.tempered import (
    make_soft_leaky_relu_spec,
    make_soft_relu_spec,
    make_soft_step_spec,
    make_swish_spec,
)
from torch import Tensor


def _nth_autograd(fn, z: Tensor, n: int) -> Tensor:
    z = z.clone().requires_grad_(True)
    y = fn(z)
    for _ in range(n):
        if not y.requires_grad:
            return torch.zeros_like(z)
        (y,) = torch.autograd.grad(y.sum(), z, create_graph=True)
    return y.detach()


# --- registration ----------------------------------------------------------


def test_soft_relu_soft_step_registered() -> None:
    assert get_activation("soft_relu").name == "soft_relu"
    assert get_activation("soft_step").name == "soft_step"
    assert get_activation("soft_heaviside").name == "soft_step"


def test_soft_sign_soft_abs_aliases() -> None:
    assert get_activation("soft_sign").name == "smooth_sign"
    assert get_activation("soft_abs").name == "softabs"


# --- beta-scaling identities ----------------------------------------------


@pytest.mark.parametrize("beta", [0.5, 1.0, 2.0, 4.0])
def test_soft_relu_beta_scaling_identity(beta: float) -> None:
    """``soft_relu^(n)(z) == beta**(n-1) * softplus^(n)(beta z)``."""
    spec = make_soft_relu_spec(beta)
    softplus = get_activation("softplus")
    z = torch.linspace(-3.0, 3.0, 41, dtype=torch.float64)
    for n in range(0, 6):
        lhs = spec.fastpath(z, n)
        rhs = (beta ** (n - 1)) * softplus.fastpath(beta * z, n)
        torch.testing.assert_close(lhs, rhs, msg=f"soft_relu identity n={n}")


@pytest.mark.parametrize("beta", [0.5, 1.0, 2.0, 4.0])
def test_soft_step_beta_scaling_identity(beta: float) -> None:
    """``soft_step^(n)(z) == beta**n * sigmoid^(n)(beta z)``."""
    spec = make_soft_step_spec(beta)
    sigmoid = get_activation("sigmoid")
    z = torch.linspace(-3.0, 3.0, 41, dtype=torch.float64)
    for n in range(0, 6):
        lhs = spec.fastpath(z, n)
        rhs = (beta**n) * sigmoid.fastpath(beta * z, n)
        torch.testing.assert_close(lhs, rhs, msg=f"soft_step identity n={n}")


# --- combinator wiring -----------------------------------------------------


def test_forward_and_derivative_route_through_tower() -> None:
    z = torch.linspace(-3.0, 3.0, 41, dtype=torch.float64)
    for spec in (make_soft_relu_spec(2.0), make_soft_step_spec(1.5)):
        torch.testing.assert_close(spec.forward(z), spec.fastpath(z, 0))
        assert spec.derivative is not None
        torch.testing.assert_close(spec.derivative(z), spec.fastpath(z, 1))


def test_tempered_tower_matches_autograd() -> None:
    z = torch.linspace(-3.0, 3.0, 41, dtype=torch.float64)
    for spec in (make_soft_relu_spec(2.0), make_soft_step_spec(1.5)):
        for n in range(1, 5):
            closed = spec.fastpath(z, n)
            auto = _nth_autograd(spec.forward, z, n)
            torch.testing.assert_close(
                closed, auto, rtol=1e-6, atol=1e-6,
                msg=f"{spec.name!r} order {n}: tower disagrees with autograd",
            )


# --- convergence to the hard limit ----------------------------------------


def test_soft_relu_converges_to_relu() -> None:
    relu = get_activation("relu")
    z = torch.tensor([-3.0, -1.0, -0.3, 0.3, 1.0, 3.0], dtype=torch.float64)
    prev = None
    for beta in (2.0, 8.0, 32.0, 128.0):
        spec = make_soft_relu_spec(beta)
        err = (spec.forward(z) - relu.forward(z)).abs().max().item()
        if prev is not None:
            assert err < prev, "soft_relu should approach relu as beta grows"
        prev = err
    assert prev < 1e-2
    # derivative -> Heaviside away from 0.
    d = make_soft_relu_spec(128.0).fastpath(z, 1)
    torch.testing.assert_close(d, (z > 0).double(), rtol=0, atol=1e-2)


def test_soft_relu_second_order_bump_is_unit_mass() -> None:
    """The n=2 bump sharpens (peak grows) but keeps unit mass -> delta."""
    zz = torch.linspace(-40.0, 40.0, 400001, dtype=torch.float64)
    dz = float(zz[1] - zz[0])
    peaks = []
    for beta in (1.0, 4.0, 16.0):
        bump = make_soft_relu_spec(beta).fastpath(zz, 2)
        mass = float(bump.sum() * dz)
        assert abs(mass - 1.0) < 1e-3, f"bump mass {mass} != 1 (beta={beta})"
        peaks.append(float(bump.max()))
    assert peaks[0] < peaks[1] < peaks[2], "peak should grow with beta"


def test_swish_beta1_is_silu() -> None:
    swish = make_swish_spec(1.0)
    silu = get_activation("silu")
    z = torch.linspace(-4.0, 4.0, 41, dtype=torch.float64)
    for n in range(0, 6):
        torch.testing.assert_close(
            swish.fastpath(z, n), silu.fastpath(z, n), msg=f"swish(beta=1) != silu at n={n}"
        )


def test_swish_matches_autograd() -> None:
    spec = make_swish_spec(1.5)
    z = torch.linspace(-4.0, 4.0, 41, dtype=torch.float64)
    for n in range(1, 5):
        closed = spec.fastpath(z, n)
        auto = _nth_autograd(spec.forward, z, n)
        torch.testing.assert_close(closed, auto, rtol=1e-6, atol=1e-6, msg=f"swish n={n}")


def test_soft_leaky_relu_converges() -> None:
    alpha = 0.1
    leaky = get_activation("leaky_relu")  # slope 0.01; build matching alpha spec
    from omnibias.torch.activations.piecewise import make_leaky_relu_spec

    hard = make_leaky_relu_spec(alpha)
    z = torch.tensor([-3.0, -1.0, -0.3, 0.3, 1.0, 3.0], dtype=torch.float64)
    err = (make_soft_leaky_relu_spec(alpha, 128.0).forward(z) - hard.forward(z)).abs().max()
    assert err.item() < 1e-2
    # derivative arms: alpha on z<0, 1 on z>0.
    d = make_soft_leaky_relu_spec(alpha, 128.0).fastpath(z, 1)
    expected = torch.where(z > 0, torch.ones_like(z), torch.full_like(z, alpha))
    torch.testing.assert_close(d, expected, rtol=0, atol=1e-2)
    assert leaky.name == "leaky_relu"


# --- blocks ----------------------------------------------------------------


def test_tempered_activation_module_matches_spec() -> None:
    z = torch.linspace(-3.0, 3.0, 41, dtype=torch.float64)
    mod = TemperedActivation("softplus", beta=2.0, scale="one_over_beta")
    spec = make_soft_relu_spec(2.0)
    for n in range(0, 5):
        torch.testing.assert_close(mod.fastpath(z, n), spec.fastpath(z, n))
    torch.testing.assert_close(mod(z), spec.forward(z))


def test_tempered_activation_learnable_beta_gradient() -> None:
    z = torch.linspace(-3.0, 3.0, 41, dtype=torch.float64)
    mod = TemperedActivation("softplus", beta=2.0, scale="one_over_beta", learnable_beta=True)
    assert isinstance(mod.beta, torch.nn.Parameter)
    out = mod(z).sum()
    out.backward()
    ana = float(mod.beta.grad)

    # finite-difference reference on the (stateless) spec.
    eps = 1e-6

    def loss(beta: float) -> float:
        return float(make_soft_relu_spec(beta).forward(z).sum())

    fd = (loss(2.0 + eps) - loss(2.0 - eps)) / (2 * eps)
    assert abs(ana - fd) < 1e-4, f"grad {ana} vs finite-diff {fd}"


def test_frozen_beta_is_buffer_not_parameter() -> None:
    mod = TemperedActivation("sigmoid", beta=3.0, scale="unit", learnable_beta=False)
    assert not isinstance(mod.beta, torch.nn.Parameter)
    assert "beta" in dict(mod.named_buffers())


def test_learnable_prelu() -> None:
    z = torch.tensor([-2.0, -0.5, 0.5, 2.0], dtype=torch.float64)
    lp = LearnablePReLU(0.25, learnable=True)
    assert isinstance(lp.alpha, torch.nn.Parameter)
    torch.testing.assert_close(lp(z), torch.where(z > 0, z, 0.25 * z))
    torch.testing.assert_close(
        lp.fastpath(z, 1), torch.where(z > 0, torch.ones_like(z), torch.full_like(z, 0.25))
    )
    assert torch.equal(lp.fastpath(z, 2), torch.zeros_like(z))
    lp(z).sum().backward()
    assert lp.alpha.grad is not None
