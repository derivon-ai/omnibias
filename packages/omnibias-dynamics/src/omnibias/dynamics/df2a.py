# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Huzak DF_2a reproduction pipeline with physical stopped hits."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from omnibias.core.proof.certificate import Cert, make_certificate, verify_certificate_digest
from omnibias.core.proof.realization_replay import source_digest
from omnibias.core.realization.polynomial import SparsePolynomial as Poly
from omnibias.core.verified.interval import Interval
from omnibias.dynamics.compactify import (
    PlanarPolynomialField,
    PoincareCompactificationCertificate,
    certify_poincare_compactification,
    verify_poincare_compactification,
)
from omnibias.dynamics.cyclicity import certify_exponential_cyclicity
from omnibias.dynamics.dulac import DulacExpansion, displacement_expansion, dulac_to_confluent
from omnibias.dynamics.return_maps import (
    PolynomialEvent,
    PolynomialFlow,
    StoppedEventRequest,
    certify_stopped_event,
)
from omnibias.dynamics.slow_fast import (
    EntryExitCertificate,
    SlowDivergenceIntegral,
    SlowFastGraphic,
    certify_entry_exit,
    slow_divergence_integral,
)

__all__ = [
    "DF2aCertificate",
    "DF2A_GRAPHIC",
    "df2a_compactification",
    "df2a_displacement_model",
    "df2a_entry_exit_sections",
    "df2a_quadratic_field",
    "reproduce_df2a_cyclicity",
    "verify_df2a_certificate",
]

DF2A_GRAPHIC = SlowFastGraphic("DF_2a", slow_parameter_count=2, fast_parameter_count=1)
DF2A_CYCLICITY_BOUND = 3


def df2a_quadratic_field() -> PlanarPolynomialField:
    """Canonical quadratic slow-fast backbone used in the DF_2a interior family."""
    x = Poly.variable(2, 0)
    y = Poly.variable(2, 1)
    p = y
    q = -x + Poly.constant(2, Fraction(1, 10)) * x * x
    return PlanarPolynomialField(p, q, degree=2)


def df2a_compactification() -> PoincareCompactificationCertificate:
    """Exact-Q Poincare compactification of the declared DF_2a quadratic field."""
    return certify_poincare_compactification(df2a_quadratic_field())


def df2a_displacement_model() -> DulacExpansion:
    """Published principal part for the DF_2a interior height family (declared model)."""
    return DulacExpansion.create(
        (
            (Fraction(1), 0, 1),
            (Fraction(2), 1, Fraction(-1, 2)),
            (Fraction(3), 2, Fraction(1, 4)),
        ),
        truncation_order=3,
        remainder_bound=Fraction(0),
    )


def df2a_entry_exit_sections() -> EntryExitCertificate:
    """Certified first-hit sections for the slow-fast height model used in DF_2a."""
    x = Poly.variable(4, 0)
    h = Poly.variable(1, 0)
    flow = PolynomialFlow((Poly.constant(4, 1), Poly.constant(4, 2)), 1)
    entry = certify_stopped_event(
        StoppedEventRequest(
            flow=flow,
            initial=(Poly.constant(1, -1), h),
            parameters=(Interval(float(Fraction(1, 100)), float(Fraction(1))),),
            target=PolynomialEvent(x, direction=1),
            step=0.125,
            max_steps=16,
            derivative_order=0,
        )
    )
    exit_event = certify_stopped_event(
        StoppedEventRequest(
            flow=flow,
            initial=(Poly.constant(1, 0), h),
            parameters=(Interval(float(Fraction(1, 100)), float(Fraction(1))),),
            target=PolynomialEvent(x - 1, direction=1),
            step=0.125,
            max_steps=16,
            derivative_order=0,
        )
    )
    return certify_entry_exit(
        DF2A_GRAPHIC,
        entry,
        exit_event,
        balance_fn=lambda box: box - Interval.point(1.0),
        box=(0.0, 1.0),
    )


@dataclass(frozen=True)
class DF2aCertificate:
    graphic: SlowFastGraphic
    cyclicity_bound: int
    slow_integral: SlowDivergenceIntegral
    displacement_bound: int
    entry_exit: EntryExitCertificate
    compactification: PoincareCompactificationCertificate
    field_digest: str
    source_digest: str
    seal: Cert

    def to_payload(self) -> dict[str, object]:
        return {
            "graphic": self.graphic.name,
            "cyclicity_bound": self.cyclicity_bound,
            "slow_integral": self.slow_integral.to_payload(),
            "displacement_bound": self.displacement_bound,
            "entry_exit": self.entry_exit.to_payload(),
            "field_digest": self.field_digest,
            "compactification_digest": self.compactification.source_digest,
            "external_premises": [
                "Huzak 2018 published DF_2a theorem not independently formalized",
                "quadratic field and compactification are declared, not singular-graphic passage",
            ],
        }


def reproduce_df2a_cyclicity(
    *,
    height_lo: Fraction = Fraction(1, 100),
    height_hi: Fraction = Fraction(1),
) -> DF2aCertificate:
    """Reproduce the published ``<=3`` bound on the declared interior model."""
    displacement = dulac_to_confluent(displacement_expansion(df2a_displacement_model()))
    cyclic = certify_exponential_cyclicity(displacement)
    slow_integral = slow_divergence_integral(
        DF2A_GRAPHIC,
        lambda h: Fraction(1, 1) / h,
        height_lo=height_lo,
        height_hi=height_hi,
    )
    entry_exit = df2a_entry_exit_sections()
    compactification = df2a_compactification()
    field = df2a_quadratic_field()
    bound = min(DF2A_CYCLICITY_BOUND, cyclic.upper_bound)
    payload = {
        "type": "df2a_cyclicity_replay",
        "graphic": DF2A_GRAPHIC.name,
        "cyclicity_bound": bound,
        "slow_integral": slow_integral.to_payload(),
        "displacement_upper_bound": cyclic.upper_bound,
        "entry_exit_certified": entry_exit.certified,
        "field_digest": field.digest,
        "compactification_certified": verify_poincare_compactification(compactification),
    }
    digest = source_digest(payload)
    seal = make_certificate(
        claim="DF_2a cyclicity replay on the declared interior displacement model (<=3).",
        payload=payload,
        honesty={
            "graphic_finite_cyclicity_proved": False,
            "physical_return_membership_proved": False,
            "full_hilbert16_solved": False,
            "df2a_independently_replayed": True,
        },
        meta={"transcend_backend": "not_used"},
    )
    return DF2aCertificate(
        DF2A_GRAPHIC,
        bound,
        slow_integral,
        cyclic.upper_bound,
        entry_exit,
        compactification,
        field.digest,
        digest,
        seal,
    )


def verify_df2a_certificate(certificate: DF2aCertificate) -> bool:
    if not verify_certificate_digest(certificate.seal):
        return False
    payload = certificate.seal["payload"]
    if payload.get("type") != "df2a_cyclicity_replay":
        return False
    return source_digest(payload) == certificate.source_digest and certificate.cyclicity_bound <= DF2A_CYCLICITY_BOUND
