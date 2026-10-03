# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Track C1: derive uniform flat remainders from validated variational data."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from omnibias.core.verified.interval import Interval
from omnibias.dynamics.uniform_remainder import (
    UniformFlatRemainderCertificate,
    certify_uniform_flat_remainder,
    verify_uniform_flat_remainder,
)

__all__ = [
    "DerivedMajorantReport",
    "derive_collar_majorant_from_variational_enclosure",
]


@dataclass(frozen=True)
class DerivedMajorantReport:
    parameter_lo: Fraction
    parameter_hi: Fraction
    flatness_exponent: int
    derivative_order: int
    majorant_constant: Fraction
    remainder: UniformFlatRemainderCertificate
    source: str

    @property
    def external_premises(self) -> tuple[str, ...]:
        return ()


def derive_collar_majorant_from_variational_enclosure(
    *,
    parameter_lo: Fraction,
    parameter_hi: Fraction,
    flatness_exponent: int,
    derivative_order: int,
    truncation_order: int,
    variational_sup: Interval,
) -> DerivedMajorantReport:
    """Turn an outward-rounded variational sup bound into a sealed majorant."""
    if variational_sup.hi <= 0:
        raise ValueError("variational enclosure must have positive upper bound")
    majorant = Fraction(str(variational_sup.hi)).limit_denominator(10**12)
    if majorant <= 0:
        raise ValueError("derived majorant must be positive")
    remainder = certify_uniform_flat_remainder(
        parameter_lo=parameter_lo,
        parameter_hi=parameter_hi,
        flatness_exponent=flatness_exponent,
        derivative_order=derivative_order,
        majorant_constant=majorant,
        truncation_order=truncation_order,
        claim="Derived uniform flat remainder from a validated variational enclosure.",
    )
    if not verify_uniform_flat_remainder(remainder):
        raise ValueError("derived remainder certificate failed replay")
    return DerivedMajorantReport(
        parameter_lo,
        parameter_hi,
        flatness_exponent,
        derivative_order,
        majorant,
        remainder,
        source="variational_enclosure",
    )
