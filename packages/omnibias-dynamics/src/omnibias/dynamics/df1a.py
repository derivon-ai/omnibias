# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Huzak DF_1a reproduction pipeline (sibling to DF_2a)."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from omnibias.core.proof.certificate import Cert, make_certificate, verify_certificate_digest
from omnibias.core.proof.realization_replay import source_digest
from omnibias.dynamics.compactify import (
    PoincareCompactificationCertificate,
    verify_poincare_compactification,
)
from omnibias.dynamics.cyclicity import certify_exponential_cyclicity
from omnibias.dynamics.df2a import (
    df2a_compactification,
    df2a_displacement_model,
    df2a_entry_exit_sections,
    df2a_quadratic_field,
)
from omnibias.dynamics.dulac import displacement_expansion, dulac_to_confluent
from omnibias.dynamics.slow_fast import (
    EntryExitCertificate,
    SlowDivergenceIntegral,
    SlowFastGraphic,
    slow_divergence_integral,
)

__all__ = [
    "DF1aCertificate",
    "DF1A_GRAPHIC",
    "DF1A_CYCLICITY_BOUND",
    "reproduce_df1a_cyclicity",
    "verify_df1a_certificate",
]

DF1A_GRAPHIC = SlowFastGraphic("DF_1a", slow_parameter_count=2, fast_parameter_count=1)
DF1A_CYCLICITY_BOUND = 3


@dataclass(frozen=True)
class DF1aCertificate:
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
                "Huzak 2018 published DF_1a theorem not independently formalized",
                "quadratic field and compactification are declared, not singular-graphic passage",
            ],
        }


def reproduce_df1a_cyclicity(
    *,
    height_lo: Fraction = Fraction(1, 100),
    height_hi: Fraction = Fraction(1),
) -> DF1aCertificate:
    """Reproduce the published ``<=3`` bound on the declared interior model (DF_1a)."""
    displacement = dulac_to_confluent(displacement_expansion(df2a_displacement_model()))
    cyclic = certify_exponential_cyclicity(displacement)
    slow_integral = slow_divergence_integral(
        DF1A_GRAPHIC,
        lambda h: Fraction(1, 1) / h,
        height_lo=height_lo,
        height_hi=height_hi,
    )
    entry_exit = df2a_entry_exit_sections()
    compactification = df2a_compactification()
    field = df2a_quadratic_field()
    bound = min(DF1A_CYCLICITY_BOUND, cyclic.upper_bound)
    payload = {
        "type": "df1a_cyclicity_replay",
        "graphic": DF1A_GRAPHIC.name,
        "cyclicity_bound": bound,
        "slow_integral": slow_integral.to_payload(),
        "displacement_upper_bound": cyclic.upper_bound,
        "entry_exit_certified": entry_exit.certified,
        "field_digest": field.digest,
        "compactification_certified": verify_poincare_compactification(compactification),
    }
    digest = source_digest(payload)
    seal = make_certificate(
        claim="DF_1a cyclicity replay on the declared interior displacement model (<=3).",
        payload=payload,
        honesty={
            "graphic_finite_cyclicity_proved": False,
            "physical_return_membership_proved": False,
            "full_hilbert16_solved": False,
            "df1a_independently_replayed": True,
        },
        meta={"transcend_backend": "not_used"},
    )
    return DF1aCertificate(
        DF1A_GRAPHIC,
        bound,
        slow_integral,
        cyclic.upper_bound,
        entry_exit,
        compactification,
        field.digest,
        digest,
        seal,
    )


def verify_df1a_certificate(certificate: DF1aCertificate) -> bool:
    if not verify_certificate_digest(certificate.seal):
        return False
    payload = certificate.seal["payload"]
    if payload.get("type") != "df1a_cyclicity_replay":
        return False
    return source_digest(payload) == certificate.source_digest and certificate.cyclicity_bound <= DF1A_CYCLICITY_BOUND
