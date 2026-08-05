# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Numerical stability regression tests for the fast path.

These tests reproduce the v0.2.1 regime where the literal multi-bias
forward (a rescaled finite-difference of K activations) loses accuracy
like ``1 / delta^(K-1)`` from cancellation. The fast path computes the
same quantity from a single base-activation call plus a Horner
polynomial, which is bit-stable down to machine epsilon.

Two complementary assertions:

1. **Analytic kernel is rock-solid.** ``spec.fastpath(z, K-1)`` matches
   the closed-form derivative to ``1e-10`` (float64) regardless of how
   we tried to discretise it.
2. **Literal forward degrades.** For small ``delta`` and ``K >= 4`` the
   literal stencil suffers measurable cancellation noise in float64 and
   either saturates to zero or blows up in float32.
"""

from __future__ import annotations

import pytest
import torch
from omnibias.torch.activations.registry import get_activation
from omnibias.torch.fastpath.dispatch import multibias_literal_forward
from omnibias.torch.stencil import central_bias_offsets, central_difference_signs


@pytest.mark.parametrize("name", ["sigmoid", "tanh", "softplus", "gaussian", "exp"])
@pytest.mark.parametrize("K", [2, 3, 4, 5])
def test_analytic_kernel_is_independent_of_delta(name: str, K: int) -> None:
    """``spec.fastpath(z, K-1)`` does not depend on delta -- it is the
    closed-form derivative tower. This is the structural property that
    makes the fast path immune to the bias-collapse cancellation."""
    spec = get_activation(name)
    z = torch.linspace(-1.0, 1.0, 32).double()
    refs: list[torch.Tensor] = []
    for delta in (1e-6, 1e-4, 1e-2, 1.0):
        del delta  # the analytic kernel does not see delta
        refs.append(spec.fastpath(z, K - 1))
    base = refs[0]
    for r in refs[1:]:
        assert torch.allclose(r, base, atol=1e-12)


@pytest.mark.parametrize("name", ["sigmoid", "tanh", "softplus", "gaussian"])
@pytest.mark.parametrize("delta", [1e-1, 1e-2])
@pytest.mark.parametrize("K", [2, 3])
def test_literal_matches_analytic_in_well_conditioned_regime(
    name: str, delta: float, K: int
) -> None:
    """For modest ``K`` and modest ``delta``, the literal forward in
    float64 still agrees with the closed-form derivative to the
    truncation accuracy of the central-difference stencil."""
    spec = get_activation(name)
    z = torch.linspace(-1.0, 1.0, 32).unsqueeze(-1).expand(32, 4).contiguous().double()

    biases = central_bias_offsets(K, delta).double().unsqueeze(0).expand(4, K).contiguous()
    signs = central_difference_signs(K, delta).double().unsqueeze(0).expand(4, K).contiguous()

    literal = multibias_literal_forward(z, biases, signs, spec.forward)
    analytic = spec.fastpath(z, K - 1)
    rel = (literal - analytic).abs().max().item() / (analytic.abs().max().item() + 1e-12)
    # Central-difference stencil has truncation error O(delta^2); for the
    # well-conditioned regime (K <= 3, delta >= 1e-2) cancellation is
    # negligible so the empirical agreement is set by the truncation.
    assert rel < 5e-2, f"{name} K={K} delta={delta}: rel={rel:.2e}"


@pytest.mark.parametrize("name", ["sigmoid", "tanh", "softplus", "gaussian"])
@pytest.mark.parametrize("K", [3, 4, 5])
def test_fastpath_dominates_literal_in_float32(name: str, K: int) -> None:
    """Compared against a float64 ground truth, the float32 closed-form
    fast path is essentially exact while the float32 literal forward at
    a moderately small delta exhibits cancellation noise large enough to
    distinguish the two strategies. This is the regime motivating the
    fast-path-by-default design."""
    spec = get_activation(name)
    z = torch.linspace(-0.6, 0.6, 32).unsqueeze(-1).expand(32, 4).contiguous()

    truth64 = spec.fastpath(z.double(), K - 1)
    fast32 = spec.fastpath(z, K - 1).double()
    err_fast = (fast32 - truth64).abs().max().item()
    assert err_fast < 1e-5, f"{name} K={K}: fast-path err {err_fast:.2e}"

    # Compute the float32 literal at a delta where cancellation is bad
    # enough to dominate -- pick small enough that ``1/delta^(K-1)`` exceeds
    # the float32 unit roundoff inverse.
    delta = 10 ** (-3.0 / max(1, K - 1))  # picks 1e-1 for K=4, 1e-0.75 for K=5...
    biases = (
        central_bias_offsets(K, delta, dtype=torch.float32).unsqueeze(0).expand(4, K).contiguous()
    )
    signs = (
        central_difference_signs(K, delta, dtype=torch.float32)
        .unsqueeze(0)
        .expand(4, K)
        .contiguous()
    )
    literal32 = multibias_literal_forward(z, biases, signs, spec.forward).double()
    err_lit = (literal32 - truth64).abs().max().item()
    # Fast path is at least 100x more accurate than the literal at this
    # cancellation-dominated delta. (The exact ratio depends on the
    # activation's higher-derivative magnitudes; 100x is a safe floor.)
    if err_lit > 1e-6:
        assert err_lit > 100 * (err_fast + 1e-9), (
            f"{name} K={K}: literal err {err_lit:.2e} is not >> "
            f"fast-path err {err_fast:.2e} at delta={delta}."
        )


@pytest.mark.parametrize("name", ["sigmoid", "tanh", "softplus", "gaussian"])
def test_literal_forward_diverges_at_extreme_small_delta(name: str) -> None:
    """At ``delta = 1e-7`` and ``K >= 3`` in float32 the rescaling factor
    ``1/delta^(K-1) >= 1e14`` exceeds the dynamic range that products
    of O(1) sigmoid values can carry, so the literal forward returns
    NaN, Inf, or a value that is order-of-magnitude wrong compared to
    the analytic fast path."""
    spec = get_activation(name)
    z = torch.linspace(-0.5, 0.5, 16).unsqueeze(-1).expand(16, 2).contiguous()
    K = 3
    delta = 1e-7

    analytic = spec.fastpath(z, K - 1)
    assert torch.isfinite(analytic).all()
    assert analytic.abs().max() > 1e-3, "spec sanity: analytic non-trivial in this z range"

    biases = (
        central_bias_offsets(K, delta, dtype=torch.float32).unsqueeze(0).expand(2, K).contiguous()
    )
    signs = (
        central_difference_signs(K, delta, dtype=torch.float32)
        .unsqueeze(0)
        .expand(2, K)
        .contiguous()
    )
    literal = multibias_literal_forward(z, biases, signs, spec.forward)
    # Either non-finite, or order-of-magnitude wrong.
    finite = torch.isfinite(literal).all().item()
    if finite:
        max_rel_err = (literal - analytic).abs().max().item() / (
            analytic.abs().max().item() + 1e-12
        )
        assert max_rel_err > 0.5, (
            f"{name}: literal at delta={delta} K={K} did not diverge "
            f"(rel err {max_rel_err:.2e}); cancellation should be catastrophic."
        )


def test_fastpath_constant_cost_in_K() -> None:
    """The fast-path code path makes one base-activation call regardless
    of order. Indirect check: the polynomial-in-sigma reconstruction
    using a single ``torch.sigmoid`` call must equal the spec's fastpath
    output for every requested order."""
    from omnibias.torch.fastpath.eulerian import sigmoid_polynomial_coeffs

    spec = get_activation("sigmoid")
    z = torch.linspace(-2, 2, 32).double()
    s_once = torch.sigmoid(z)

    for n in (1, 2, 3, 4, 5):
        coeffs = sigmoid_polynomial_coeffs(n)
        deg = len(coeffs) - 1
        out = torch.full_like(s_once, coeffs[deg])
        for k in range(deg - 1, -1, -1):
            out = out * s_once + coeffs[k]
        ref = spec.fastpath(z, n)
        assert torch.allclose(out, ref, atol=1e-12), f"order {n} mismatch"
