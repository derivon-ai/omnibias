# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""An exact eight-sphere baseline for real projective quartic surfaces.

This module certifies only the normalized rational family
sum((X_i**2 - W**2)**2, i=0..2) - epsilon * W**4, 0 < epsilon < 1.
It checks the actual coefficients, excluding every complex singularity by
the affine critical values and the homogeneous partials at infinity.

In each real orthant, u_i = x_i**2 - 1 is a diffeomorphism onto (-1, inf)^3.
The sphere sum(u_i**2) = epsilon and its closed ball lie inside this image.
Thus the entire real locus consists of eight disjoint spheres bounding eight
disjoint balls. This elementary family is not a quartic classification or a
new surface construction. The topological argument is documented mathematics;
the exact Python replay does not claim a formal theorem-prover verification.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from fractions import Fraction
from itertools import product

from omnibias.core.realization.polynomial import (
    DEFAULT_ALGEBRA_BUDGET,
    AlgebraBudget,
    AlgebraBudgetExceeded,
    Rational,
    SparsePolynomial,
    rational,
)


def separable_quartic_polynomial(
    epsilon: Rational, *, budget: AlgebraBudget = DEFAULT_ALGEBRA_BUDGET,
) -> SparsePolynomial:
    """Construct the normalized four-variable family in the order X,Y,Z,W.

    The parameter must satisfy 0 < epsilon < 1. This is the parameter range
    of the eight-sphere proof, not the full smooth locus of the family.
    """
    eps = rational(epsilon)
    if not 0 < eps < 1:
        raise ValueError("the eight-sphere family requires 0 < epsilon < 1")
    coordinates = tuple(SparsePolynomial.variable(4, i, budget=budget) for i in range(4))
    w = coordinates[3]
    result = -eps * w**4
    for x in coordinates[:3]:
        result = result + (x**2 - w**2)**2
    return result


def _source_digest(polynomial: SparsePolynomial) -> str:
    raw = json.dumps(polynomial.to_payload(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


@dataclass(frozen=True)
class SeparableQuarticSurfaceCertificate:
    """Derived proof data for the complete eight-sphere real locus.

    The affine critical levels refer to sum((x_i**2 - 1)**2). Their distance
    from epsilon excludes affine complex singularities. The positive squared
    radius gap proves that the whole u-sphere and ball lie in (-1, inf)^3.
    ``component_orthants`` lists the eight signs of (X/W,Y/W,Z/W); each entry
    has one genus-zero component bounding a ball in that orthant. Complex
    nonsingularity, absence of real points at infinity, and completeness are
    conclusions of successful replay, never caller-supplied premises.
    """

    source_digest: str
    epsilon: Fraction
    affine_critical_levels: tuple[Fraction, ...]
    critical_level_gap: Fraction
    squared_radius_gap: Fraction
    component_orthants: tuple[tuple[int, int, int], ...]
    component_genera: tuple[int, ...]

    @property
    def component_count(self) -> int:
        return len(self.component_orthants)


def certify_separable_quartic_surface(
    polynomial: SparsePolynomial,
) -> SeparableQuarticSurfaceCertificate:
    """Infer epsilon from coefficients and prove this family's exact topology.

    At W=0 the first three homogeneous partials are 4*X_i**3, which cannot
    vanish at a projective point simultaneously. In W=1, vanishing of the
    first three partials forces every complex x_i to belong to {0,-1,1}.
    The corresponding levels are exactly 0,1,2,3, all different from epsilon.
    At real infinity the equation sum(X_i**4)=0 has no projective solution.
    Each affine coordinate is nonzero because x_i=0 would give level >=1.
    The orthant diffeomorphisms described above exhaust the entire real locus.

    A different polynomial family or parameter range is rejected as unsupported;
    this is not a nonexistence certificate for arbitrary quartic surfaces.
    """
    if polynomial.nvars != 4:
        raise ValueError("a homogeneous polynomial in X,Y,Z,W is required")
    epsilon = Fraction(3) - dict(polynomial.terms).get((0, 0, 0, 4), Fraction(0))
    canonical = separable_quartic_polynomial(epsilon, budget=polynomial.budget)
    if polynomial.terms != canonical.terms:
        raise ValueError("the actual coefficients do not equal the normalized separable quartic family")

    # Check the partial factorizations used in the finite complex-critical-point
    # argument directly against the supplied source, in addition to its identity.
    coordinates = tuple(SparsePolynomial.variable(4, i, budget=polynomial.budget) for i in range(4))
    for i, x in enumerate(coordinates[:3]):
        expected = 4 * x * (x**2 - coordinates[3]**2)
        if polynomial.derivative(i).terms != expected.terms:
            raise ArithmeticError("source partial factorization failed")  # pragma: no cover

    levels = tuple(sorted({
        polynomial.evaluate((*point, 1)) + epsilon
        for point in product((-1, 0, 1), repeat=3)
    }))
    gap = min(abs(level - epsilon) for level in levels)
    if levels != tuple(map(Fraction, (0, 1, 2, 3))) or gap <= 0:
        raise ArithmeticError("affine critical-level exclusion failed")  # pragma: no cover
    orthants = tuple((a, b, c) for a, b, c in product((-1, 1), repeat=3))
    return SeparableQuarticSurfaceCertificate(
        _source_digest(polynomial), epsilon, levels, gap, 1 - epsilon, orthants, (0,) * 8,
    )


def replay_separable_quartic_surface(
    polynomial: SparsePolynomial, certificate: SeparableQuarticSurfaceCertificate,
) -> bool:
    """Replay the exact source and compare every derived certificate operand."""
    if _source_digest(polynomial) != certificate.source_digest:
        return False
    try:
        return certificate == certify_separable_quartic_surface(polynomial)
    except (TypeError, ValueError, ArithmeticError, AlgebraBudgetExceeded):
        return False


__all__ = [
    "SeparableQuarticSurfaceCertificate",
    "certify_separable_quartic_surface",
    "replay_separable_quartic_surface",
    "separable_quartic_polynomial",
]
