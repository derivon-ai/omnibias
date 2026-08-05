# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Piecewise (almost-everywhere) activation tower tests (PyTorch).

Covers the "hard" non-smooth family (``relu`` relatives, clamps, shrinkage,
``abs`` / ``sign`` / ``step``) plus the in-place extensions (``huber`` a.e.,
``silu`` / ``gelu`` exact). Checks:

* forward against a PyTorch reference,
* integer-order reduction (``fastpath(z, 0) == forward``, ``fastpath(z, 1) ==
  derivative``),
* the a.e. tower against autograd on every *open* piece (breakpoints excluded),
* the dropped-singular-part convention (linear pieces ``n >= 2 -> 0``;
  ``sign`` / ``step`` ``n >= 1 -> 0``; ``H(0) = 0``, ``sign(0) = 0``).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from omnibias.torch.activations import get_activation, list_activations
from torch import Tensor

torch.manual_seed(0)

_PIECEWISE_NAMES = [
    "leaky_relu",
    "prelu",
    "relu6",
    "hardtanh",
    "hardsigmoid",
    "hardswish",
    "elu",
    "selu",
    "celu",
    "softshrink",
    "hardshrink",
    "threshold",
    "abs",
    "sign",
    "step",
    "softsign",
]

# Linear-piece activations: the tower is exactly zero for every order >= 2.
_LINEAR_PIECE_NAMES = [
    "leaky_relu",
    "prelu",
    "relu6",
    "hardtanh",
    "hardsigmoid",
    "softshrink",
    "hardshrink",
    "threshold",
    "abs",
]

# Highest order compared against autograd, per activation, plus the finite
# breakpoint set to avoid when sampling open pieces.
_AUTOGRAD_SPEC = {
    "relu": (3, [0.0]),
    "leaky_relu": (3, [0.0]),
    "prelu": (3, [0.0]),
    "relu6": (3, [0.0, 6.0]),
    "hardtanh": (3, [-1.0, 1.0]),
    "hardsigmoid": (3, [-3.0, 3.0]),
    "hardswish": (3, [-3.0, 3.0]),
    "softshrink": (3, [-0.5, 0.5]),
    "hardshrink": (3, [-0.5, 0.5]),
    "threshold": (3, [0.0]),
    "abs": (3, [0.0]),
    "sign": (3, [0.0]),
    "step": (3, [0.0]),
    "elu": (4, [0.0]),
    "selu": (4, [0.0]),
    "celu": (4, [0.0]),
    "softsign": (4, [0.0]),
    "huber": (3, [-1.0, 1.0]),
    "silu": (5, []),
    "gelu": (5, []),
}


def _safe_samples(breakpoints: list[float], *, pad: float = 0.08) -> Tensor:
    grid = torch.linspace(-5.0, 5.0, 221, dtype=torch.float64)
    if breakpoints:
        keep = torch.ones_like(grid, dtype=torch.bool)
        for bp in breakpoints:
            keep &= (grid - bp).abs() > pad
        grid = grid[keep]
    return grid


def _nth_autograd(fn, z: Tensor, n: int) -> Tensor:
    """Elementwise ``n``-th derivative via autograd (0 once the graph is flat)."""
    z = z.clone().requires_grad_(True)
    y = fn(z)
    for _ in range(n):
        if not y.requires_grad:
            return torch.zeros_like(z)
        (y,) = torch.autograd.grad(y.sum(), z, create_graph=True)
    return y.detach()


# --- registry --------------------------------------------------------------


def test_all_piecewise_registered() -> None:
    registered = set(list_activations())
    for name in _PIECEWISE_NAMES:
        assert name in registered, f"{name!r} not registered"


def test_heaviside_alias_resolves() -> None:
    assert get_activation("heaviside").name == "step"


# --- forward references ----------------------------------------------------


def test_forward_matches_reference() -> None:
    z = torch.linspace(-4.0, 4.0, 65, dtype=torch.float64)
    refs = {
        "leaky_relu": F.leaky_relu(z, 0.01),
        "prelu": F.leaky_relu(z, 0.25),
        "relu6": F.relu6(z),
        "hardtanh": F.hardtanh(z),
        "hardsigmoid": F.hardsigmoid(z),
        "hardswish": F.hardswish(z),
        "elu": F.elu(z),
        "selu": F.selu(z),
        "celu": F.celu(z),
        "softshrink": F.softshrink(z),
        "hardshrink": F.hardshrink(z),
        "threshold": F.threshold(z, 0.0, 0.0),
        "abs": z.abs(),
        "sign": torch.sign(z),
        "step": (z > 0).double(),
        "softsign": F.softsign(z),
    }
    for name, ref in refs.items():
        out = get_activation(name).forward(z)
        torch.testing.assert_close(out, ref, msg=f"forward mismatch for {name!r}")


# --- integer-order reduction ----------------------------------------------


@pytest.mark.parametrize("name", _PIECEWISE_NAMES)
def test_order0_is_forward_order1_is_derivative(name: str) -> None:
    spec = get_activation(name)
    z = torch.linspace(-4.0, 4.0, 51, dtype=torch.float64)
    torch.testing.assert_close(spec.fastpath(z, 0), spec.forward(z))
    assert spec.derivative is not None
    torch.testing.assert_close(spec.fastpath(z, 1), spec.derivative(z))


# --- a.e. tower vs autograd on open pieces --------------------------------


@pytest.mark.parametrize("name", sorted(_AUTOGRAD_SPEC))
def test_tower_matches_autograd_on_open_pieces(name: str) -> None:
    max_order, breakpoints = _AUTOGRAD_SPEC[name]
    spec = get_activation(name)
    z = _safe_samples(breakpoints)
    for n in range(1, max_order + 1):
        closed = spec.fastpath(z, n)
        auto = _nth_autograd(spec.forward, z, n)
        torch.testing.assert_close(
            closed, auto, rtol=1e-6, atol=1e-6,
            msg=f"{name!r} order {n}: closed form disagrees with autograd",
        )


# --- linear pieces: exactly zero from order 2 -----------------------------


@pytest.mark.parametrize("name", _LINEAR_PIECE_NAMES)
def test_linear_pieces_zero_from_order_two(name: str) -> None:
    spec = get_activation(name)
    z = torch.linspace(-4.0, 4.0, 51, dtype=torch.float64)
    for n in (2, 3, 4):
        out = spec.fastpath(z, n)
        assert torch.equal(out, torch.zeros_like(out)), f"{name!r} order {n} not zero"


# --- relu / huber a.e. specifics ------------------------------------------


def test_relu_all_orders_ae() -> None:
    spec = get_activation("relu")
    z = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0], dtype=torch.float64)
    torch.testing.assert_close(spec.fastpath(z, 1), (z > 0).double())  # H(0)=0
    for n in (2, 3, 5):
        out = spec.fastpath(z, n)
        assert torch.equal(out, torch.zeros_like(out))


def test_huber_all_orders_ae() -> None:
    spec = get_activation("huber")
    z = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0], dtype=torch.float64)
    # order 2 is the indicator |z| <= tau (tau = 1).
    torch.testing.assert_close(spec.fastpath(z, 2), (z.abs() <= 1.0).double())
    for n in (3, 4):
        out = spec.fastpath(z, n)
        assert torch.equal(out, torch.zeros_like(out))


# --- breakpoint a.e. conventions ------------------------------------------


def test_breakpoint_conventions() -> None:
    z0 = torch.zeros(1, dtype=torch.float64)
    assert float(get_activation("sign").forward(z0)) == 0.0
    assert float(get_activation("step").forward(z0)) == 0.0
    assert float(get_activation("relu").fastpath(z0, 1)) == 0.0  # H(0) = 0
    # sign / step: whole tower (n >= 1) is zero (singular delta dropped).
    for name in ("sign", "step"):
        spec = get_activation(name)
        for n in (1, 2, 3):
            out = spec.fastpath(torch.linspace(-2, 2, 9, dtype=torch.float64), n)
            assert torch.equal(out, torch.zeros_like(out))


# --- silu / gelu exact high-order -----------------------------------------


@pytest.mark.parametrize("name", ["silu", "gelu"])
def test_silu_gelu_exact_high_order(name: str) -> None:
    spec = get_activation(name)
    z = torch.linspace(-4.0, 4.0, 61, dtype=torch.float64)
    for n in range(0, 6):
        closed = spec.fastpath(z, n)
        auto = _nth_autograd(spec.forward, z, n)
        torch.testing.assert_close(
            closed, auto, rtol=1e-6, atol=1e-6,
            msg=f"{name!r} order {n}: exact tower disagrees with autograd",
        )


# --- errors ----------------------------------------------------------------


@pytest.mark.parametrize("name", ["leaky_relu", "elu", "abs", "sign", "step", "softsign"])
def test_negative_order_raises(name: str) -> None:
    spec = get_activation(name)
    with pytest.raises(ValueError):
        spec.fastpath(torch.zeros(3, dtype=torch.float64), -1)
