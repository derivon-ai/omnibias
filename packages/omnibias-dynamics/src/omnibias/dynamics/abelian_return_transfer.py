# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Conditional transfer from an Abelian zero count to a true return map.

Suppose the physical displacement has the uniform expansion

.. math::

   \Delta_\varepsilon(h)
      = \varepsilon I(h) + \varepsilon^2 R_\varepsilon(h).

If certified root boxes contain all zeros of ``I``, its endpoint and
off-root margins are positive, its derivative is separated from zero in
the root boxes, and uniform value/derivative bounds on ``R`` are supplied,
then the same root count persists for sufficiently small nonzero
``epsilon``.

This module certifies the exact rational threshold arithmetic.  It does not
derive the expansion or remainder bounds for a DRR graphic.  In particular,
the open ``I_2^1`` and ``I_4^1`` graphics are not Hamiltonian regular ovals,
so a Picard--Fuchs certificate alone cannot close either case.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction

from omnibias.core.collapse.verdict import adjudicate_residual
from omnibias.core.proof.certificate import (
    Cert,
    make_certificate,
    verify_certificate_digest,
)
from omnibias.core.verified.interval import Interval
from omnibias.dynamics.honesty_inventory import build_honesty
from omnibias.holonomic import certify_picard_fuchs, verify_picard_fuchs

Rational = int | Fraction

__all__ = [
    "AbelianDRRTransferReport",
    "ConditionalReturnTransferCertificate",
    "FirstOrderReturnBounds",
    "certify_conditional_return_transfer",
    "identity_verdicts",
    "report",
    "residual_derivative_threshold",
    "residual_normalized_displacement",
    "residual_value_threshold",
    "verify_conditional_return_transfer",
]


def _q(value: Rational, name: str) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, int | Fraction):
        raise TypeError(f"{name} must be an exact rational")
    return Fraction(value)


def residual_value_threshold(
    margin: Fraction,
    remainder_bound: Fraction,
    epsilon_threshold: Fraction,
) -> Fraction:
    """Residual of ``M*epsilon_value = margin``."""
    return remainder_bound * epsilon_threshold - margin


def residual_derivative_threshold(
    derivative_margin: Fraction,
    derivative_remainder_bound: Fraction,
    epsilon_threshold: Fraction,
) -> Fraction:
    """Residual of ``K*epsilon_derivative = derivative_margin``."""
    return derivative_remainder_bound * epsilon_threshold - derivative_margin


def residual_normalized_displacement(
    displacement: Fraction,
    epsilon: Fraction,
    abelian: Fraction,
    remainder: Fraction,
) -> Fraction:
    """Residual of ``Delta=epsilon*I+epsilon^2*R``."""
    return displacement - epsilon * abelian - epsilon**2 * remainder


def _verdict(residual: Fraction) -> str:
    box = (
        Interval.point(0.0)
        if residual == 0
        else Interval.from_rational(residual)
    )
    return adjudicate_residual(box, existential=False).status


def identity_verdicts() -> dict[str, str]:
    margin, m_bound = Fraction(1, 20), Fraction(2)
    derivative_margin, k_bound = Fraction(1, 4), Fraction(3)
    epsilon = Fraction(1, 40)
    abelian, remainder = Fraction(2, 5), Fraction(1, 7)
    displacement = epsilon * abelian + epsilon**2 * remainder
    return {
        "abelian_transfer_value": _verdict(
            residual_value_threshold(margin, m_bound, margin / m_bound)
        ),
        "abelian_transfer_derivative": _verdict(
            residual_derivative_threshold(
                derivative_margin,
                k_bound,
                derivative_margin / k_bound,
            )
        ),
        "abelian_transfer_normalized": _verdict(
            residual_normalized_displacement(
                displacement,
                epsilon,
                abelian,
                remainder,
            )
        ),
    }


@dataclass(frozen=True)
class FirstOrderReturnBounds:
    """Exact hypotheses for a conditional first-order return transfer."""

    zero_count: int
    endpoint_margin: Fraction
    off_root_margin: Fraction
    derivative_margin: Fraction
    remainder_value_bound: Fraction
    remainder_derivative_bound: Fraction

    @classmethod
    def create(
        cls,
        *,
        zero_count: int,
        endpoint_margin: Rational,
        off_root_margin: Rational,
        derivative_margin: Rational,
        remainder_value_bound: Rational,
        remainder_derivative_bound: Rational,
    ) -> FirstOrderReturnBounds:
        if type(zero_count) is not int or zero_count < 0:
            raise ValueError("zero_count must be a nonnegative integer")
        values = (
            _q(endpoint_margin, "endpoint_margin"),
            _q(off_root_margin, "off_root_margin"),
            _q(derivative_margin, "derivative_margin"),
            _q(remainder_value_bound, "remainder_value_bound"),
            _q(remainder_derivative_bound, "remainder_derivative_bound"),
        )
        if any(value <= 0 for value in values):
            raise ValueError("all transfer margins and bounds must be positive")
        return cls(zero_count, *values)

    @property
    def epsilon_upper(self) -> Fraction:
        """Strict threshold preserving signs, uniqueness, and exclusion."""
        return min(
            self.endpoint_margin / self.remainder_value_bound,
            self.off_root_margin / self.remainder_value_bound,
            self.derivative_margin / self.remainder_derivative_bound,
        )

    def to_payload(self) -> dict[str, object]:
        def pair(value: Fraction) -> list[int]:
            return [value.numerator, value.denominator]

        return {
            "zero_count": self.zero_count,
            "endpoint_margin": pair(self.endpoint_margin),
            "off_root_margin": pair(self.off_root_margin),
            "derivative_margin": pair(self.derivative_margin),
            "remainder_value_bound": pair(self.remainder_value_bound),
            "remainder_derivative_bound": pair(
                self.remainder_derivative_bound
            ),
            "epsilon_upper_strict": pair(self.epsilon_upper),
        }


@dataclass(frozen=True)
class ConditionalReturnTransferCertificate:
    """Sealed exact threshold, conditional on analytic source bounds."""

    bounds: FirstOrderReturnBounds
    epsilon_upper: Fraction
    seal: Cert


def certify_conditional_return_transfer(
    bounds: FirstOrderReturnBounds,
) -> ConditionalReturnTransferCertificate:
    """Seal the exact epsilon threshold without asserting its hypotheses."""
    if not isinstance(bounds, FirstOrderReturnBounds):
        raise TypeError("bounds must be FirstOrderReturnBounds")
    threshold = bounds.epsilon_upper
    seal = make_certificate(
        claim=(
            "Conditional exact epsilon threshold preserving a supplied Abelian "
            "zero count under a supplied first-order return remainder."
        ),
        payload={
            "type": "abelian_first_order_return_transfer",
            "bounds": bounds.to_payload(),
            "scope": (
                "exact threshold arithmetic; physical expansion, root cover, "
                "and remainder bounds are external analytic premises"
            ),
        },
        honesty={
            "conditional_transfer_arithmetic": True,
            "physical_remainder_derived": False,
            "drr_graphic_membership_proved": False,
            "full_hilbert16_solved": False,
        },
        meta={"transcend_backend": "not_used"},
    )
    return ConditionalReturnTransferCertificate(bounds, threshold, seal)


def verify_conditional_return_transfer(
    certificate: ConditionalReturnTransferCertificate,
) -> bool:
    """Replay the exact threshold and digest."""
    if not isinstance(certificate, ConditionalReturnTransferCertificate):
        return False
    replay = certify_conditional_return_transfer(certificate.bounds)
    return (
        verify_certificate_digest(certificate.seal)
        and replay == certificate
    )


@dataclass(frozen=True)
class AbelianDRRTransferReport:
    """H4 result: strong infinitesimal artifact, no DRR physical transfer."""

    identities: Mapping[str, str]
    picard_fuchs_verified: bool
    conditional_transfer_verified: bool
    epsilon_upper: Fraction
    open_drr_cases: tuple[str, ...]
    regular_hamiltonian_oval: bool
    named_graphic_reduction_supplied: bool
    physical_remainder_derived: bool
    endpoint_capture_proved: bool
    drr_transfer_certified: bool
    honesty: Mapping[str, object]

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": "hilbert16-abelian-drr-transfer-v1",
            "identities": dict(self.identities),
            "picard_fuchs_verified": self.picard_fuchs_verified,
            "conditional_transfer_verified": self.conditional_transfer_verified,
            "epsilon_upper": [
                self.epsilon_upper.numerator,
                self.epsilon_upper.denominator,
            ],
            "open_drr_cases": list(self.open_drr_cases),
            "regular_hamiltonian_oval": self.regular_hamiltonian_oval,
            "named_graphic_reduction_supplied": (
                self.named_graphic_reduction_supplied
            ),
            "physical_remainder_derived": self.physical_remainder_derived,
            "endpoint_capture_proved": self.endpoint_capture_proved,
            "drr_transfer_certified": self.drr_transfer_certified,
            "full_hilbert16_solved": False,
            "honesty": dict(self.honesty),
            "scope": (
                "Exact Picard-Fuchs identity and conditional first-order "
                "epsilon arithmetic. No physical DRR return membership, "
                "singular endpoint capture, or Hilbert-XVI theorem."
            ),
        }


def report() -> AbelianDRRTransferReport:
    """Test H4 against the two open nilpotent saddle-node graphics."""
    picard_fuchs = certify_picard_fuchs(-1)
    bounds = FirstOrderReturnBounds.create(
        zero_count=2,
        endpoint_margin=Fraction(1, 10),
        off_root_margin=Fraction(1, 20),
        derivative_margin=Fraction(1, 4),
        remainder_value_bound=2,
        remainder_derivative_bound=3,
    )
    conditional = certify_conditional_return_transfer(bounds)
    pf_verified = verify_picard_fuchs(picard_fuchs)
    conditional_verified = verify_conditional_return_transfer(conditional)
    physical = False
    honesty = build_honesty(
        picard_fuchs_instance_certified=pf_verified,
        conditional_abelian_return_transfer=conditional_verified,
        drr_graphic_abelian_reduction=False,
        drr_return_remainder_derived=False,
        drr_endpoint_capture=False,
        drr_abelian_transfer_certified=False,
        full_hilbert16_solved=False,
    )
    return AbelianDRRTransferReport(
        identities=identity_verdicts(),
        picard_fuchs_verified=pf_verified,
        conditional_transfer_verified=conditional_verified,
        epsilon_upper=conditional.epsilon_upper,
        open_drr_cases=("I_2^1", "I_4^1"),
        regular_hamiltonian_oval=True,
        named_graphic_reduction_supplied=physical,
        physical_remainder_derived=physical,
        endpoint_capture_proved=physical,
        drr_transfer_certified=physical,
        honesty=honesty,
    )
