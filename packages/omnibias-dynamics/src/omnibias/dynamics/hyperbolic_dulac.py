# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Parameter-uniform hyperbolic Dulac expansions with explicit L-flat remainders."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from omnibias.core.proof.certificate import Cert, make_certificate, verify_certificate_digest
from omnibias.core.proof.realization_replay import source_digest
from omnibias.dynamics.dulac import DulacExpansion
from omnibias.dynamics.uniform_remainder import (
    UniformFlatRemainderCertificate,
    certify_uniform_flat_remainder,
    verify_uniform_flat_remainder,
)

__all__ = [
    "HyperbolicDulacCertificate",
    "certify_hyperbolic_dulac_expansion",
    "verify_hyperbolic_dulac_expansion",
]


@dataclass(frozen=True)
class HyperbolicDulacCertificate:
    """A finitely L-flat hyperbolic Dulac principal part plus uniform remainder."""

    expansion: DulacExpansion
    flatness_exponent: int
    parameter_lo: Fraction
    parameter_hi: Fraction
    remainder: UniformFlatRemainderCertificate
    source_digest: str
    seal: Cert

    def to_payload(self) -> dict[str, object]:
        return {
            "flatness_exponent": self.flatness_exponent,
            "parameter_lo": [self.parameter_lo.numerator, self.parameter_lo.denominator],
            "parameter_hi": [self.parameter_hi.numerator, self.parameter_hi.denominator],
            "term_count": len(self.expansion.terms),
            "remainder": self.remainder.to_payload(),
        }


def certify_hyperbolic_dulac_expansion(
    expansion: DulacExpansion,
    *,
    parameter_lo: Fraction,
    parameter_hi: Fraction,
    flatness_exponent: int,
    majorant_constant: Fraction,
    derivative_order: int = 0,
) -> HyperbolicDulacCertificate:
    """Seal a declared hyperbolic Dulac expansion with an explicit L-flat majorant."""
    if flatness_exponent < 1:
        raise ValueError("flatness_exponent must be at least 1")
    if not expansion.terms:
        raise ValueError("expansion must carry at least one principal term")
    remainder = certify_uniform_flat_remainder(
        parameter_lo=parameter_lo,
        parameter_hi=parameter_hi,
        flatness_exponent=flatness_exponent,
        derivative_order=derivative_order,
        majorant_constant=majorant_constant,
        truncation_order=expansion.truncation_order,
        claim="Declared parameter-uniform L-flat remainder on a hyperbolic Dulac collar.",
    )
    payload = {
        "type": "hyperbolic_dulac_l_flat",
        "expansion_terms": len(expansion.terms),
        "flatness_exponent": flatness_exponent,
        "parameter_lo": [parameter_lo.numerator, parameter_lo.denominator],
        "parameter_hi": [parameter_hi.numerator, parameter_hi.denominator],
        "remainder": remainder.to_payload(),
    }
    digest = source_digest(payload)
    seal = make_certificate(
        claim="Hyperbolic Dulac principal part with explicit uniform L-flat remainder.",
        payload=payload,
        honesty={
            "hyperbolic_dulac_l_flat_proved": True,
            "uniform_remainder_proved": True,
            "physical_return_membership_proved": False,
            "graphic_finite_cyclicity_proved": False,
            "full_hilbert16_solved": False,
        },
        meta={"transcend_backend": "not_used"},
    )
    return HyperbolicDulacCertificate(
        expansion,
        flatness_exponent,
        parameter_lo,
        parameter_hi,
        remainder,
        digest,
        seal,
    )


def verify_hyperbolic_dulac_expansion(certificate: HyperbolicDulacCertificate) -> bool:
    if not verify_certificate_digest(certificate.seal):
        return False
    payload = certificate.seal["payload"]
    if payload.get("type") != "hyperbolic_dulac_l_flat":
        return False
    if source_digest(
        {
            "type": "hyperbolic_dulac_l_flat",
            "expansion_terms": len(certificate.expansion.terms),
            "flatness_exponent": certificate.flatness_exponent,
            "parameter_lo": payload["parameter_lo"],
            "parameter_hi": payload["parameter_hi"],
            "remainder": payload["remainder"],
        }
    ) != certificate.source_digest:
        return False
    return verify_uniform_flat_remainder(certificate.remainder)
