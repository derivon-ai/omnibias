# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Conditional uniform cyclicity on a hyperbolic Dulac collar."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from omnibias.core.verified.interval import Interval

from omnibias.core.proof.certificate import Cert, make_certificate, verify_certificate_digest
from omnibias.core.proof.lean_check import LeanCheckResult, check_certificate
from omnibias.core.proof.realization_replay import source_digest
from omnibias.dynamics.compensator import certify_compensator_collar_bound
from omnibias.dynamics.cyclicity import certify_exponential_cyclicity
from omnibias.dynamics.dulac import DulacExpansion, dulac_to_confluent
from omnibias.dynamics.hyperbolic_dulac import certify_hyperbolic_dulac_expansion
from omnibias.dynamics.majorant_derive import derive_collar_majorant_from_variational_enclosure

__all__ = [
    "HyperbolicUniformCyclicityCertificate",
    "HyperbolicUniformFormalVerification",
    "certify_hyperbolic_uniform_cyclicity",
    "default_hyperbolic_smoke_expansion",
    "verify_hyperbolic_uniform_cyclicity",
    "verify_hyperbolic_uniform_cyclicity_formally",
]

MARIN_VILLADELPRAT_PREMISE = (
    "Marin-Villadelprat (2024) uniform hyperbolic Dulac coefficient bound on the "
    "declared collar"
)


@dataclass(frozen=True)
class HyperbolicUniformFormalVerification:
    box_cover: LeanCheckResult
    verified: bool


@dataclass(frozen=True)
class HyperbolicUniformCyclicityCertificate:
    expansion: DulacExpansion
    parameter_lo: Fraction
    parameter_hi: Fraction
    zero_bound: int
    external_premises: tuple[str, ...]
    seal: Cert
    box_cover_seal: Cert | None = None

    def to_payload(self) -> dict[str, object]:
        return {
            "parameter_lo": [self.parameter_lo.numerator, self.parameter_lo.denominator],
            "parameter_hi": [self.parameter_hi.numerator, self.parameter_hi.denominator],
            "zero_bound": self.zero_bound,
            "external_premises": list(self.external_premises),
            "status": "CONDITIONAL" if self.external_premises else "DISCHARGED",
        }


def _principal_terms(expansion: DulacExpansion) -> tuple[tuple[Fraction, int, Fraction], ...]:
    output: list[tuple[Fraction, int, Fraction]] = []
    for term in expansion.terms:
        if not term.exact:
            raise ValueError("hyperbolic uniform cyclicity requires exact Dulac terms")
        output.append((term.exponent.lo, term.log_power, term.coefficient.lo))
    return tuple(output)


def certify_hyperbolic_uniform_cyclicity(
    expansion: DulacExpansion,
    *,
    parameter_lo: Fraction,
    parameter_hi: Fraction,
    flatness_exponent: int,
    majorant_constant: Fraction,
    derivative_order: int = 0,
    derive_majorant: bool = False,
    variational_sup: Interval | None = None,
) -> HyperbolicUniformCyclicityCertificate:
    """Compose hyperbolic Dulac, compensator, and Rolle zero bounds on a collar."""
    active_majorant = majorant_constant
    premises: tuple[str, ...] = ()
    if derive_majorant:
        if variational_sup is None:
            variational_sup = Interval(0.0, float(majorant_constant))
        derived = derive_collar_majorant_from_variational_enclosure(
            parameter_lo=parameter_lo,
            parameter_hi=parameter_hi,
            flatness_exponent=flatness_exponent,
            derivative_order=derivative_order,
            truncation_order=expansion.truncation_order,
            variational_sup=variational_sup,
        )
        active_majorant = derived.majorant_constant
    else:
        premises = (MARIN_VILLADELPRAT_PREMISE,)
    hyperbolic = certify_hyperbolic_dulac_expansion(
        expansion,
        parameter_lo=parameter_lo,
        parameter_hi=parameter_hi,
        flatness_exponent=flatness_exponent,
        majorant_constant=active_majorant,
        derivative_order=derivative_order,
    )
    principal = _principal_terms(expansion)
    collar = certify_compensator_collar_bound(
        alpha=principal[0][0],
        s_lo=parameter_lo,
        s_hi=parameter_hi,
        principal_terms=principal,
    )
    exponential = certify_exponential_cyclicity(dulac_to_confluent(expansion))
    zero_bound = exponential.upper_bound
    exponential_digest = exponential.seal["payload"].get("source_digest", "")
    box_source = {
        "parameter_lo": [parameter_lo.numerator, parameter_lo.denominator],
        "parameter_hi": [parameter_hi.numerator, parameter_hi.denominator],
        "uniform_bound": zero_bound,
        "hyperbolic_digest": hyperbolic.source_digest,
        "collar_digest": collar.source_digest,
        "exponential_digest": exponential_digest,
    }
    box_digest = source_digest(box_source)
    box_cover_seal = make_certificate(
        claim="Single-cell parameter box tiling for a hyperbolic uniform cyclicity bound.",
        payload={
            "type": "box_cover_tiling",
            "parent": box_source,
            "tree": {"depth": 0, "children": []},
            "uniform_bound": zero_bound,
            "source": box_source,
            "source_digest": box_digest,
            "leaf_winding_digests": [],
            "finite_formal_scope": "one rational parameter box; analytic collars are trusted inputs",
        },
        honesty={
            "parameter_uniform_hyperbolic_bound": True,
            "finite_parameter_box": True,
            "full_hilbert16_solved": False,
            "analytic_enclosures_are_lean_verified": False,
        },
        meta={"transcend_backend": "not_used"},
    )
    payload: dict[str, Any] = {
        "type": "hyperbolic_uniform_cyclicity",
        "zero_bound": zero_bound,
        "hyperbolic_digest": hyperbolic.source_digest,
        "collar_digest": collar.source_digest,
        "exponential_digest": exponential_digest,
        "box_cover_digest": box_digest,
        "external_premises": list(premises),
    }
    seal = make_certificate(
        claim="Uniform hyperbolic collar cyclicity bound on a declared parameter box.",
        payload=payload,
        honesty={
            "uniform_cyclicity_proved": not premises,
            "conditional_on_external_premises": bool(premises),
            "physical_return_membership_proved": False,
            "graphic_finite_cyclicity_proved": False,
            "full_hilbert16_solved": False,
        },
        meta={"transcend_backend": "not_used"},
    )
    return HyperbolicUniformCyclicityCertificate(
        expansion,
        parameter_lo,
        parameter_hi,
        zero_bound,
        premises,
        seal,
        box_cover_seal,
    )


def verify_hyperbolic_uniform_cyclicity(certificate: HyperbolicUniformCyclicityCertificate) -> bool:
    if not verify_certificate_digest(certificate.seal):
        return False
    if certificate.box_cover_seal is not None:
        return verify_certificate_digest(certificate.box_cover_seal)
    return True


def verify_hyperbolic_uniform_cyclicity_formally(
    certificate: HyperbolicUniformCyclicityCertificate,
) -> HyperbolicUniformFormalVerification:
    if certificate.box_cover_seal is None:
        return HyperbolicUniformFormalVerification(
            LeanCheckResult(
                verified=False,
                available=False,
                obligation="box_cover_tiling",
                detail="box cover seal missing",
            ),
            False,
        )
    box_cover = check_certificate(certificate.box_cover_seal)
    return HyperbolicUniformFormalVerification(
        box_cover,
        verify_hyperbolic_uniform_cyclicity(certificate) and box_cover.available,
    )


def default_hyperbolic_smoke_expansion() -> DulacExpansion:
    """A one-term hyperbolic principal part for regression tests."""
    return DulacExpansion.create(((Fraction(1), 0, Fraction(1)),), truncation_order=1)
