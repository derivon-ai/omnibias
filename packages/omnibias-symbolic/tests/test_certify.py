# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Rationalize-and-certify: verify-and-seal a discovered coefficient vector.

The worked example throughout is the shipped Lie-symmetry determining
equation for the transport PDE (``u_t + u_x = 0``, ``pr_transport``):
``x d/dx + t d/dt`` (a genuine textbook scaling symmetry, integer generator
weights) is the positive case; a lone ``x d/dx`` and a coefficient corrupted
with an irrational float are the negative cases a broken implementation would
falsely certify.
"""

from __future__ import annotations

import math
from fractions import Fraction

import pytest
from omnibias.core.proof.certificate import (
    THEOREM_PROVER_VERIFIED_KEY,
    schema_errors_v1,
    verify_certificate_digest,
)
from omnibias.symbolic.certify import (
    DiscoveryCertificate,
    ResidualFn,
    honesty_payload,
    lie_symmetry_residual_fn,
    rationalize_and_certify_discovery,
)
from omnibias.symbolic.symmetry import SymmetryBasis, affine_basis, designed_samples
from omnibias.symbolic.symmetry._core import Sample, _restrict_transport, pr_transport

# x d/dx is basis index 1, t d/dt is basis index 5 (see affine_basis()).
_X_DX = 1
_T_DT = 5


def _transport_residual_fn() -> tuple[SymmetryBasis, tuple[Sample, ...], ResidualFn]:
    basis = affine_basis()
    samples = designed_samples(12)
    return basis, samples, lie_symmetry_residual_fn(pr_transport, _restrict_transport, basis, samples)


# --------------------------------------------------------------------------- #
# Positive: a genuine, textbook, exactly-rational Lie symmetry is certified.
# --------------------------------------------------------------------------- #
def test_worked_example_transport_scaling_symmetry_is_certified() -> None:
    basis, samples, residual_fn = _transport_residual_fn()
    coeffs = [0.0] * basis.n_coeff
    coeffs[_X_DX] = 1.0
    coeffs[_T_DT] = 1.0

    result = rationalize_and_certify_discovery(
        coeffs,
        residual_fn,
        max_denominator=1000,
        claim="transport scaling symmetry x d/dx + t d/dt",
        meta={"pde": "transport", "generator": "x d/dx + t d/dt"},
    )

    assert isinstance(result, DiscoveryCertificate)
    assert result.certified is True
    assert result.n_checks == len(samples)
    assert result.residuals == tuple(Fraction(0) for _ in samples)
    assert result.rationalized_coefficients[_X_DX] == Fraction(1)
    assert result.rationalized_coefficients[_T_DT] == Fraction(1)
    assert all(c == Fraction(0) for i, c in enumerate(result.rationalized_coefficients) if i not in (_X_DX, _T_DT))
    assert len(result.certificates) == len(samples)


def test_certified_payload_contains_exact_coefficients_and_zero_residual() -> None:
    basis, samples, residual_fn = _transport_residual_fn()
    coeffs = [0.0] * basis.n_coeff
    coeffs[_X_DX] = 1.0
    coeffs[_T_DT] = 1.0

    result = rationalize_and_certify_discovery(
        coeffs, residual_fn, max_denominator=1000, claim="transport scaling symmetry"
    )
    assert result.certified
    for cert in result.certificates:
        payload = cert["payload"]
        assert payload["type"] == "rational_identity"
        assert payload["rhs"] == 0
        # The exact Int identity the Lean kernel would check: sum_i c_i m_i = 0.
        assert sum(c * m for c, m in payload["lhs_terms"]) == 0


def test_noisy_float_coefficients_snap_to_the_exact_integer_generator() -> None:
    """A realistic numerical proposer's tiny float noise must still rationalize."""
    basis, samples, residual_fn = _transport_residual_fn()
    coeffs = [0.0] * basis.n_coeff
    coeffs[_X_DX] = 1.0 + 1e-9
    coeffs[_T_DT] = 1.0 - 1e-9

    result = rationalize_and_certify_discovery(
        coeffs, residual_fn, max_denominator=1000, claim="noisy transport scaling symmetry"
    )
    assert result.certified is True
    assert result.rationalized_coefficients[_X_DX] == Fraction(1)
    assert result.rationalized_coefficients[_T_DT] == Fraction(1)


# --------------------------------------------------------------------------- #
# Negative: real refusals a broken implementation would falsely certify.
# --------------------------------------------------------------------------- #
def test_lone_x_ddx_is_not_a_transport_symmetry_and_is_refused() -> None:
    """x d/dx alone (already-exact integer coefficients) breaks the residual."""
    basis, samples, residual_fn = _transport_residual_fn()
    coeffs = [0.0] * basis.n_coeff
    coeffs[_X_DX] = 1.0

    result = rationalize_and_certify_discovery(
        coeffs, residual_fn, max_denominator=1000, claim="bogus: x d/dx alone"
    )
    assert result.certified is False
    assert result.certificates == ()
    assert any(r != 0 for r in result.residuals)
    assert "refused" in result.detail


def test_irrational_coefficient_is_refused_not_falsely_certified() -> None:
    """sqrt(2) in place of the true rational weight 1 must not rationalize away."""
    basis, samples, residual_fn = _transport_residual_fn()
    coeffs = [0.0] * basis.n_coeff
    coeffs[_X_DX] = 1.0
    coeffs[_T_DT] = math.sqrt(2.0)

    result = rationalize_and_certify_discovery(
        coeffs, residual_fn, max_denominator=1000, claim="bogus: sqrt(2) generator"
    )
    assert result.certified is False
    assert result.certificates == ()
    # The snap itself must not silently equal the true rational value.
    assert result.rationalized_coefficients[_T_DT] != Fraction(1)
    assert any(r != 0 for r in result.residuals)


def test_near_miss_float_does_not_falsely_certify_via_float_tolerance() -> None:
    """A tiny but genuinely nonzero residual must be refused, not rounded away.

    This is the test that would catch a broken implementation that checks
    ``abs(residual) < eps`` instead of an exact ``== 0``: the snap at a large
    denominator bound recovers a fraction close to, but not equal to, the true
    integer weight, so the exact residual is small but never zero.
    """
    basis, samples, residual_fn = _transport_residual_fn()
    coeffs = [0.0] * basis.n_coeff
    coeffs[_X_DX] = 1.0
    coeffs[_T_DT] = 1.0 + 1e-7

    result = rationalize_and_certify_discovery(
        coeffs, residual_fn, max_denominator=10**7, claim="bogus: near-miss generator"
    )
    assert result.certified is False
    assert result.rationalized_coefficients[_T_DT] != Fraction(1)
    nonzero = [r for r in result.residuals if r != 0]
    assert nonzero
    # Genuinely tiny (not the O(1) residual of the lone-x_dx case) but exactly nonzero.
    assert all(0 < abs(r) < Fraction(1, 1000) for r in nonzero)


# --------------------------------------------------------------------------- #
# Certificate well-formedness against the existing v1 schema.
# --------------------------------------------------------------------------- #
def test_certificates_are_well_formed_per_v1_schema() -> None:
    basis, samples, residual_fn = _transport_residual_fn()
    coeffs = [0.0] * basis.n_coeff
    coeffs[_X_DX] = 1.0
    coeffs[_T_DT] = 1.0

    result = rationalize_and_certify_discovery(
        coeffs, residual_fn, max_denominator=1000, claim="transport scaling symmetry"
    )
    assert result.certificates
    for cert in result.certificates:
        assert schema_errors_v1(cert) == []
        assert verify_certificate_digest(cert)


def test_never_asserts_theorem_prover_verified() -> None:
    basis, samples, residual_fn = _transport_residual_fn()
    coeffs = [0.0] * basis.n_coeff
    coeffs[_X_DX] = 1.0
    coeffs[_T_DT] = 1.0

    result = rationalize_and_certify_discovery(
        coeffs, residual_fn, max_denominator=1000, claim="transport scaling symmetry"
    )
    for cert in result.certificates:
        assert THEOREM_PROVER_VERIFIED_KEY not in cert["honesty"]
    payload = honesty_payload()
    assert THEOREM_PROVER_VERIFIED_KEY not in payload
    assert payload["exact_rational_check"] is True
    assert payload["float_tolerance_used"] is False


# --------------------------------------------------------------------------- #
# Implementation safety nets.
# --------------------------------------------------------------------------- #
def test_rejects_float_residual_pairs() -> None:
    def bogus_residual_fn(coeffs):  # type: ignore[no-untyped-def]
        return [[(coeffs[0], 0.5)]]  # a float value, not exact

    with pytest.raises(TypeError):
        rationalize_and_certify_discovery(
            [1.0], bogus_residual_fn, max_denominator=10, claim="bogus float residual"
        )


def test_rejects_zero_checks() -> None:
    def empty_residual_fn(coeffs):  # type: ignore[no-untyped-def]
        return []

    with pytest.raises(ValueError):
        rationalize_and_certify_discovery(
            [1.0], empty_residual_fn, max_denominator=10, claim="bogus empty"
        )


def test_rejects_nonpositive_max_denominator() -> None:
    def residual_fn(coeffs):  # type: ignore[no-untyped-def]
        return [[(Fraction(1), Fraction(0))]]

    with pytest.raises(ValueError):
        rationalize_and_certify_discovery(
            [1.0], residual_fn, max_denominator=0, claim="bogus denominator"
        )


def test_lie_symmetry_residual_fn_rejects_dimension_mismatch() -> None:
    basis, samples, residual_fn = _transport_residual_fn()
    with pytest.raises(ValueError):
        residual_fn([Fraction(1)])


def test_lie_symmetry_residual_fn_rejects_complex_step_residual() -> None:
    """pr_kdv_linear routes through the complex-step eta_xxx: refused, not silently floated."""
    from omnibias.symbolic.symmetry._core import _restrict_kdv_linear, pr_kdv_linear

    basis = affine_basis()
    samples = designed_samples(4)
    residual_fn = lie_symmetry_residual_fn(pr_kdv_linear, _restrict_kdv_linear, basis, samples)
    with pytest.raises(TypeError):
        residual_fn([Fraction(1)] + [Fraction(0)] * (basis.n_coeff - 1))


def test_lie_symmetry_residual_fn_requires_nonempty_basis_and_samples() -> None:
    from omnibias.symbolic.symmetry import SymmetryBasis

    with pytest.raises(ValueError):
        lie_symmetry_residual_fn(pr_transport, _restrict_transport, SymmetryBasis((), "empty"), designed_samples(2))
    with pytest.raises(ValueError):
        lie_symmetry_residual_fn(pr_transport, _restrict_transport, affine_basis(), [])
