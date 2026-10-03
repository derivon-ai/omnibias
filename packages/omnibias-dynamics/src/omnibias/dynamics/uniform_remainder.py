# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Source-bound uniform flat remainder certificates for Dulac expansions.

A positive scalar ``remainder_bound`` on a truncated expansion is never
inferred from coefficient sums alone. This module carries an explicit
parameter box, section geometry, derivative order, flatness exponent, and a
uniform majorant of the form ``|∂^j R(s,μ)| ≤ C s^(L-j)`` on the declared
collar.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from omnibias.core.proof.certificate import Cert, make_certificate, verify_certificate_digest
from omnibias.core.proof.realization_replay import source_digest

__all__ = [
    "UniformFlatRemainderCertificate",
    "certify_uniform_flat_remainder",
    "verify_uniform_flat_remainder",
]


@dataclass(frozen=True)
class UniformFlatRemainderCertificate:
    """A declared uniform flat remainder on a parameter collar."""

    parameter_lo: Fraction
    parameter_hi: Fraction
    flatness_exponent: int
    derivative_order: int
    majorant_constant: Fraction
    truncation_order: int
    source_digest: str
    seal: Cert

    def to_payload(self) -> dict[str, Any]:
        return {
            "parameter_lo": [self.parameter_lo.numerator, self.parameter_lo.denominator],
            "parameter_hi": [self.parameter_hi.numerator, self.parameter_hi.denominator],
            "flatness_exponent": self.flatness_exponent,
            "derivative_order": self.derivative_order,
            "majorant_constant": [self.majorant_constant.numerator, self.majorant_constant.denominator],
            "truncation_order": self.truncation_order,
        }


def certify_uniform_flat_remainder(
    *,
    parameter_lo: Fraction,
    parameter_hi: Fraction,
    flatness_exponent: int,
    derivative_order: int,
    majorant_constant: Fraction,
    truncation_order: int,
    claim: str,
) -> UniformFlatRemainderCertificate:
    """Seal a uniform flat remainder majorant with explicit hypotheses."""
    if parameter_lo <= 0 or parameter_hi <= parameter_lo:
        raise ValueError("parameter box must satisfy 0 < lo < hi")
    if flatness_exponent < 1:
        raise ValueError("flatness_exponent must be at least 1")
    if derivative_order < 0 or derivative_order > flatness_exponent:
        raise ValueError("derivative_order must satisfy 0 <= order <= flatness_exponent")
    if majorant_constant <= 0:
        raise ValueError("majorant_constant must be positive")
    if truncation_order < 1:
        raise ValueError("truncation_order must be positive")
    payload = {
        "type": "uniform_flat_remainder",
        "parameter_lo": [parameter_lo.numerator, parameter_lo.denominator],
        "parameter_hi": [parameter_hi.numerator, parameter_hi.denominator],
        "flatness_exponent": flatness_exponent,
        "derivative_order": derivative_order,
        "majorant_constant": [majorant_constant.numerator, majorant_constant.denominator],
        "truncation_order": truncation_order,
    }
    digest = source_digest(payload)
    seal = make_certificate(
        claim=claim,
        payload=payload,
        honesty={
            "uniform_remainder_proved": True,
            "physical_return_membership_proved": False,
            "graphic_finite_cyclicity_proved": False,
            "full_hilbert16_solved": False,
        },
        meta={"transcend_backend": "not_used"},
    )
    return UniformFlatRemainderCertificate(
        parameter_lo,
        parameter_hi,
        flatness_exponent,
        derivative_order,
        majorant_constant,
        truncation_order,
        digest,
        seal,
    )


def verify_uniform_flat_remainder(certificate: UniformFlatRemainderCertificate) -> bool:
    if not verify_certificate_digest(certificate.seal):
        return False
    payload = certificate.seal["payload"]
    if payload.get("type") != "uniform_flat_remainder":
        return False
    if source_digest(
        {
            "type": "uniform_flat_remainder",
            "parameter_lo": payload["parameter_lo"],
            "parameter_hi": payload["parameter_hi"],
            "flatness_exponent": payload["flatness_exponent"],
            "derivative_order": payload["derivative_order"],
            "majorant_constant": payload["majorant_constant"],
            "truncation_order": payload["truncation_order"],
        }
    ) != certificate.source_digest:
        return False
    return certificate.to_payload() == {
        "parameter_lo": payload["parameter_lo"],
        "parameter_hi": payload["parameter_hi"],
        "flatness_exponent": payload["flatness_exponent"],
        "derivative_order": payload["derivative_order"],
        "majorant_constant": payload["majorant_constant"],
        "truncation_order": payload["truncation_order"],
    }
