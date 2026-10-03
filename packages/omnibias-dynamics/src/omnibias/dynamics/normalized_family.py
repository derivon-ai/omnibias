# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Phase 1-2: normalized coefficient family ``K_n`` for degree-``n`` fields."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import comb

from omnibias.core.realization.polynomial import SparsePolynomial
from omnibias.dynamics.compactify import PlanarPolynomialField, certify_poincare_compactification

__all__ = [
    "NormalizedFamily",
    "coefficient_dimension",
    "normalization_preserves_cycle_count",
    "sphere_dimension",
]


def coefficient_dimension(degree: int) -> int:
    """Dimension of ``(P,Q)`` with ``max(deg P, deg Q) <= degree``."""
    if type(degree) is not int or degree < 0:
        raise ValueError("degree must be a nonnegative integer")
    return 2 * (degree + 1) * (degree + 2)


def sphere_dimension(degree: int) -> int:
    """Dimension of the unit sphere after coefficient normalization."""
    return coefficient_dimension(degree) - 1


@dataclass(frozen=True)
class NormalizedFamily:
    """Compact normalized parameter space for a fixed polynomial degree."""

    degree: int
    coefficient_dimension: int
    sphere_dimension: int
    time_rescaling_identified: bool = True
    degree_drop_strata_declared: bool = True
    zero_field_excluded: bool = True

    def __post_init__(self) -> None:
        if self.coefficient_dimension != coefficient_dimension(self.degree):
            raise ValueError("coefficient_dimension does not match degree")
        if self.sphere_dimension != sphere_dimension(self.degree):
            raise ValueError("sphere_dimension does not match degree")

    @property
    def harnack_upper_bound(self) -> int:
        d = self.degree
        return (d - 1) * (d - 2) // 2 + 1 if d >= 2 else 0


def normalization_preserves_cycle_count() -> dict[str, bool]:
    """Recorded invariants for Phase 1.5 (not a continuum proof)."""
    return {
        "affine_chart_change_preserves_count": True,
        "positive_time_rescale_preserves_geometry": True,
        "coefficient_norm_rescale_preserves_zero_set": True,
        "theorem_prover_verified": False,
        "continuum_pde_claim": False,
    }


def example_compactified_field() -> PlanarPolynomialField:
    """A degree-2 smoke field with a nonempty equator used in atlas regression tests."""
    return PlanarPolynomialField(
        SparsePolynomial(2, {(0, 1): Fraction(1)}),
        SparsePolynomial(2, {(1, 0): Fraction(1), (2, 0): Fraction(-1)}),
        degree=2,
    )


def certify_example_atlas() -> dict[str, object]:
    """Seal a finite Poincare atlas for the smoke field."""
    field = example_compactified_field()
    certificate = certify_poincare_compactification(field)
    return {
        "field_digest": field.digest,
        "infinite_singularity_count": len(certificate.singularities),
        "honesty": {
            "full_hilbert16_solved": False,
            "physical_return_membership_proved": False,
        },
    }
