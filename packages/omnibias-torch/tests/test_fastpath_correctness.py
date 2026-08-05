# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Fast-path correctness tests.

For specs with a closed-form derivative tower (sigmoid, tanh, softplus,
gaussian, exp), the fast-path output ``sigma^(n)(z)`` must match the
literal multi-bias forward in the bias-collapse stencil regime to a
tight tolerance.

We construct the literal forward in the central-difference stencil with
rescaled signs, so its limit as ``delta -> 0`` is exactly
``sigma^(K-1)(z + b_mid)`` and the fast-path can be compared directly
without any auxiliary scaling.
"""

from __future__ import annotations

import warnings

import pytest
import torch
from omnibias.torch.activations.registry import get_activation
from omnibias.torch.fastpath.dispatch import (
    multibias_forward,
    multibias_literal_forward,
)
from omnibias.torch.stencil import (
    central_bias_offsets,
    central_difference_signs,
)

SMOOTH_SPECS_ALL_K = [
    ("sigmoid", [2, 3, 4, 5]),
    ("tanh", [2, 3, 4, 5]),
    ("softplus", [2, 3, 4, 5]),
    ("gaussian", [2, 3, 4, 5]),
    ("exp", [2, 3, 4, 5]),
]


def _stencil_ombu_args(num_channels: int, K: int, delta: float, bias_value: float = 0.0):
    """Build (biases, signs) tensors in central-difference rescaled form."""
    offsets = central_bias_offsets(K, delta)  # (K,)
    biases = bias_value + offsets.unsqueeze(0).expand(num_channels, K).contiguous()
    signs = central_difference_signs(K, delta).unsqueeze(0).expand(num_channels, K).contiguous()
    return biases, signs


@pytest.mark.parametrize("name,Ks", SMOOTH_SPECS_ALL_K)
def test_fastpath_matches_literal_in_collapse_stencil(name: str, Ks: list[int]) -> None:
    """At delta = 1e-2 the literal stencil forward agrees with the
    closed-form ``sigma^(K-1)`` to ``rtol=1e-4`` (limited by the O(delta^2)
    stencil error, not by the fast-path itself).
    """
    spec = get_activation(name)
    delta = 1e-2
    z = torch.linspace(-1.0, 1.0, 32).unsqueeze(-1).expand(32, 4).contiguous().double()
    bias_value = 0.0
    for K in Ks:
        biases, signs = _stencil_ombu_args(4, K, delta, bias_value=bias_value)
        biases = biases.double()
        signs = signs.double()
        literal = multibias_literal_forward(z, biases, signs, spec.forward)
        analytic = spec.fastpath(z + bias_value, K - 1)
        rel = (literal - analytic).abs().max().item() / (analytic.abs().max().item() + 1e-12)
        # The central-difference stencil at delta=1e-2 has truncation error
        # O(delta^2) = 1e-4 for smooth activations.
        assert rel < 5e-4, f"{name} K={K}: rel={rel:.2e}"


def test_multibias_forward_collapse_threshold_is_deprecated_noop() -> None:
    """``collapse_threshold`` never affected dispatch: a non-zero value warns
    (DeprecationWarning) but leaves the result equal to the literal forward,
    and the default value is silent."""
    spec = get_activation("sigmoid")
    z = torch.linspace(-1.0, 1.0, 6).unsqueeze(-1).double()  # (6, 1)
    biases = torch.zeros(1, 3, dtype=torch.float64)
    signs = torch.tensor([[1.0, -1.0, 1.0]], dtype=torch.float64)
    literal = multibias_literal_forward(z, biases, signs, spec.forward)

    with pytest.warns(DeprecationWarning, match="collapse_threshold"):
        out = multibias_forward(z, biases, signs, spec, collapse_threshold=1e-3)
    assert torch.equal(out, literal)

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning here would fail the test
        out_default = multibias_forward(z, biases, signs, spec)
    assert torch.equal(out_default, literal)


@pytest.mark.parametrize("name", ["sigmoid", "tanh", "softplus", "gaussian", "exp"])
def test_fastpath_recovers_first_derivative(name: str) -> None:
    """``spec.fastpath(z, 1)`` equals ``spec.derivative(z)``."""
    spec = get_activation(name)
    z = torch.linspace(-2, 2, 50).double()
    out = spec.fastpath(z, 1)
    ref = spec.derivative(z)
    assert torch.allclose(out, ref, atol=1e-10)


def test_fastpath_zero_order_returns_forward() -> None:
    """``spec.fastpath(z, 0)`` equals ``spec.forward(z)``."""
    z = torch.linspace(-2, 2, 16).double()
    for name in ("sigmoid", "tanh", "softplus", "gaussian", "exp"):
        spec = get_activation(name)
        out = spec.fastpath(z, 0)
        ref = spec.forward(z)
        assert torch.allclose(out, ref, atol=1e-12), f"{name} order 0 mismatch"


def test_exp_fastpath_is_constant_across_orders() -> None:
    """``exp^(n)(z) = exp(z)`` for every order."""
    spec = get_activation("exp")
    z = torch.linspace(-1, 1, 16).double()
    ref = torch.exp(z)
    for n in range(8):
        out = spec.fastpath(z, n)
        assert torch.allclose(out, ref, atol=1e-12), f"order {n} mismatch"


def test_sigmoid_known_polynomials() -> None:
    """Spot-check ``P_2(s) = s - 3 s^2 + 2 s^3`` and ``P_3(s) = s - 7 s^2 + 12 s^3 - 6 s^4``."""
    from omnibias.torch.fastpath.eulerian import sigmoid_polynomial_coeffs

    assert sigmoid_polynomial_coeffs(0) == (0.0, 1.0)
    assert sigmoid_polynomial_coeffs(1) == (0.0, 1.0, -1.0)
    assert sigmoid_polynomial_coeffs(2) == (0.0, 1.0, -3.0, 2.0)
    assert sigmoid_polynomial_coeffs(3) == (0.0, 1.0, -7.0, 12.0, -6.0)


def test_hermite_known_polynomials() -> None:
    """Spot-check ``He_0..He_4``."""
    from omnibias.torch.fastpath.hermite import hermite_coeffs

    assert hermite_coeffs(0) == (1.0,)
    assert hermite_coeffs(1) == (0.0, 1.0)
    assert hermite_coeffs(2) == (-1.0, 0.0, 1.0)
    assert hermite_coeffs(3) == (0.0, -3.0, 0.0, 1.0)
    assert hermite_coeffs(4) == (3.0, 0.0, -6.0, 0.0, 1.0)
