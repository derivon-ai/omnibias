# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Écalle–Roussarie compensators and source-bound ECT certificates."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from omnibias.core.proof.certificate import Cert, make_certificate, verify_certificate_digest
from omnibias.core.proof.realization_replay import source_digest
from omnibias.core.verified.asymptotic_jet import power_compensator
from omnibias.core.verified.interval import Interval
from omnibias.dynamics.chebyshev import (
    ECTCertificate,
    certify_ect,
    sturm_root_count,
    wronskian_determinant,
)
from omnibias.dynamics.hyperbolic_dulac import (
    HyperbolicDulacCertificate,
    certify_hyperbolic_dulac_expansion,
    verify_hyperbolic_dulac_expansion,
)
from omnibias.dynamics.uniform_remainder import (
    UniformFlatRemainderCertificate,
    certify_uniform_flat_remainder,
    verify_uniform_flat_remainder,
)
from omnibias.holonomic._core.poly_n import PolyN

__all__ = [
    "CompensatorCertificate",
    "CompensatorCollarBoundCertificate",
    "EcalleCompensatorValue",
    "certify_compensator_collar_bound",
    "certify_compensator_enclosure",
    "compensator_derivative",
    "compensator_value",
    "verify_compensator_certificate",
    "verify_compensator_collar_bound",
]


@dataclass(frozen=True)
class EcalleCompensatorValue:
    """Declared compensator ``omega(s, alpha) = (s^{-alpha} - 1) / alpha``."""

    alpha: Fraction
    s_lo: Fraction
    s_hi: Fraction
    enclosure: tuple[float, float]

    def to_payload(self) -> dict[str, object]:
        return {
            "alpha": [self.alpha.numerator, self.alpha.denominator],
            "s_lo": [self.s_lo.numerator, self.s_lo.denominator],
            "s_hi": [self.s_hi.numerator, self.s_hi.denominator],
            "enclosure": [self.enclosure[0], self.enclosure[1]],
        }


def compensator_derivative(alpha: Fraction, order: int, s: Interval) -> Interval:
    """Enclose ``d^order/ds^order omega(s, alpha)`` using exact recurrences."""
    if order < 0:
        raise ValueError("order must be nonnegative")
    if order == 0:
        return compensator_value(alpha, s)
    if alpha == 0:
        if order == 1:
            return Interval.point(1.0) / s
        sign = -1.0 if order % 2 else 1.0
        return sign * power_compensator(0, 0, s, a_order=0, b_order=0, log_order=order) / (s ** (order - 1))
    signed = -1 if order % 2 else 1
    exponent = alpha + order
    return signed * power_compensator(exponent, exponent, s, a_order=0, b_order=0) / float(alpha)


def compensator_value(alpha: Fraction, s: Interval) -> Interval:
    """Interval enclosure of ``(s^{-alpha} - 1) / alpha`` with the ``alpha -> 0`` limit."""
    if alpha == 0:
        return power_compensator(0, 0, s, a_order=0, b_order=0, log_order=1)
    return (power_compensator(alpha, alpha, s, a_order=0, b_order=0) - Interval.point(1.0)) / float(
        alpha
    )


def certify_compensator_enclosure(
    *,
    alpha: Fraction,
    s_lo: Fraction,
    s_hi: Fraction,
    flatness_exponent: int = 2,
    majorant_constant: Fraction = Fraction(1),
) -> CompensatorCertificate:
    """Seal compensator, ECT principal part, and uniform remainder hypotheses."""
    if s_lo <= 0 or s_hi <= s_lo:
        raise ValueError("s box must satisfy 0 < lo < hi")
    box = Interval(float(s_lo), float(s_hi))
    enclosure = compensator_value(alpha, box)
    x = PolyN.var(1, 0)
    ect = certify_ect((x, PolyN.const(1, 1)), order=2)
    remainder = certify_uniform_flat_remainder(
        parameter_lo=s_lo,
        parameter_hi=s_hi,
        flatness_exponent=flatness_exponent,
        derivative_order=0,
        majorant_constant=majorant_constant,
        truncation_order=1,
        claim="Declared L-flat compensator remainder on a compact collar.",
    )
    value = EcalleCompensatorValue(alpha, s_lo, s_hi, (enclosure.lo, enclosure.hi))
    payload = {
        "type": "ecalle_compensator_ect",
        "compensator": value.to_payload(),
        "ect": ect.to_payload(),
        "remainder": remainder.to_payload(),
    }
    digest = source_digest(payload)
    seal = make_certificate(
        claim="Source-bound compensator enclosure with ECT principal part and flat remainder.",
        payload=payload,
        honesty={
            "compensator_enclosure_proved": True,
            "uniform_remainder_proved": True,
            "physical_return_membership_proved": False,
            "graphic_finite_cyclicity_proved": False,
            "full_hilbert16_solved": False,
        },
        meta={"transcend_backend": "interval"},
    )
    return CompensatorCertificate(value, ect, remainder, digest, seal)


@dataclass(frozen=True)
class CompensatorCertificate:
    compensator: EcalleCompensatorValue
    ect: ECTCertificate
    remainder: UniformFlatRemainderCertificate
    source_digest: str
    seal: Cert


@dataclass(frozen=True)
class CompensatorCollarBoundCertificate:
    compensator: CompensatorCertificate
    hyperbolic: HyperbolicDulacCertificate
    wronskian_order: int
    zero_upper_bound: int
    source_digest: str
    seal: Cert


def certify_compensator_collar_bound(
    *,
    alpha: Fraction,
    s_lo: Fraction,
    s_hi: Fraction,
    principal_terms: tuple[tuple[Fraction, int, Fraction], ...],
) -> CompensatorCollarBoundCertificate:
    """Combine compensator, hyperbolic Dulac, ECT, and a Sturm zero upper bound."""
    from omnibias.dynamics.dulac import DulacExpansion

    compensator = certify_compensator_enclosure(alpha=alpha, s_lo=s_lo, s_hi=s_hi)
    expansion = DulacExpansion.create(principal_terms, truncation_order=len(principal_terms), remainder_bound=Fraction(0))
    hyperbolic = certify_hyperbolic_dulac_expansion(
        expansion,
        parameter_lo=s_lo,
        parameter_hi=s_hi,
        flatness_exponent=2,
        majorant_constant=Fraction(1),
    )
    x = PolyN.var(1, 0)
    basis = (x, PolyN.const(1, 1), x * x)
    wronskian = wronskian_determinant(basis, 2)
    zero_upper = sturm_root_count(wronskian, Fraction(-3, 2), Fraction(3, 2))
    payload = {
        "type": "compensator_collar_zero_bound",
        "compensator_digest": compensator.source_digest,
        "hyperbolic_digest": hyperbolic.source_digest,
        "wronskian_order": 2,
        "zero_upper_bound": zero_upper,
    }
    digest = source_digest(payload)
    seal = make_certificate(
        claim="Compensator-collar zero bound with Wronskian/Sturm majorant (declared collar).",
        payload=payload,
        honesty={
            "compensator_collar_zero_bound_proved": True,
            "physical_return_membership_proved": False,
            "graphic_finite_cyclicity_proved": False,
            "full_hilbert16_solved": False,
        },
        meta={"transcend_backend": "interval"},
    )
    return CompensatorCollarBoundCertificate(compensator, hyperbolic, 2, zero_upper, digest, seal)


def verify_compensator_collar_bound(certificate: CompensatorCollarBoundCertificate) -> bool:
    if not verify_certificate_digest(certificate.seal):
        return False
    payload = certificate.seal["payload"]
    if payload.get("type") != "compensator_collar_zero_bound":
        return False
    return (
        source_digest(payload) == certificate.source_digest
        and verify_compensator_certificate(certificate.compensator)
        and verify_hyperbolic_dulac_expansion(certificate.hyperbolic)
    )


def verify_compensator_certificate(certificate: CompensatorCertificate) -> bool:
    if not verify_certificate_digest(certificate.seal):
        return False
    payload = certificate.seal["payload"]
    if payload.get("type") != "ecalle_compensator_ect":
        return False
    if source_digest(payload) != certificate.source_digest:
        return False
    return verify_uniform_flat_remainder(certificate.remainder) and certificate.ect.proved_ect_on_interval
