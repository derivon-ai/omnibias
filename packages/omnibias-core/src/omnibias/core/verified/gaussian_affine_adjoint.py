# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Exact reusable normal geometry for affine Gaussian parameter adjoints."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING

from omnibias.core._rational_linear import factor_positive_definite
from omnibias.core.gaussian_jet import _Arithmetic
from omnibias.core.verified.gaussian_affine import GaussianAffineRestriction, _dot
from omnibias.core.verified.inverse_gaussian_affine import _intact
from omnibias.core.verified.inverse_gaussian_heat import _pair

if TYPE_CHECKING:
    from omnibias.core.proof.certificate import Cert

Vector = tuple[Fraction, ...]
Matrix = tuple[Vector, ...]


@dataclass(frozen=True, init=False)
class GaussianAffineAdjointGeometry:
    r"""Reusable exact lift for the declared independent constraint rows.

    ``left_inverse`` is L=(C C^T)^(-1) C, with shape rows(C) by dimension.
    Thus L C^T=I and C^T L=I-P. In particular, L^T maps a constraint-value
    perturbation to its minimum-norm normal displacement. Row coordinates
    remain those of the supplied C; changing the row basis changes L.

    The artifact certifies geometry only. It makes no contraction, inverse,
    gradient or optimization claim. A later query can reuse L without a new
    factorization. Construct with :func:`compile_gaussian_affine_adjoint`.
    """

    restriction: GaussianAffineRestriction
    left_inverse: Matrix
    certificate: Cert

    def __init__(self) -> None:
        raise TypeError("use compile_gaussian_affine_adjoint")

    @property
    def dimension(self) -> int:
        return self.restriction.dimension

    @property
    def constraint_count(self) -> int:
        return len(self.restriction.constraints)

    @property
    def tangent_dimension(self) -> int:
        return self.restriction.tangent_dimension


def _check_identities(constraints: Matrix, left: Matrix, projector: Matrix, arithmetic: _Arithmetic) -> None:
    rows, dimension = len(constraints), len(projector)
    for i in range(rows):
        for j in range(rows):
            if _dot(left[i], constraints[j], arithmetic) != int(i == j):
                raise ArithmeticError("constraint left inverse identity failed")
    columns = tuple(tuple(row[j] for row in left) for j in range(dimension))
    for i in range(dimension):
        row = tuple(constraint[i] for constraint in constraints)
        for j in range(dimension):
            normal = arithmetic.add(Fraction(int(i == j)), -projector[i][j])
            if _dot(row, columns[j], arithmetic) != normal:
                raise ArithmeticError("constraint normal projector identity failed")


def compile_gaussian_affine_adjoint(
    restriction: GaussianAffineRestriction,
) -> GaussianAffineAdjointGeometry | None:
    r"""Compile L=(C C^T)^(-1) C using one bounded exact LDLT factorization.

    C must have full row rank. Consistent redundant rows are accepted by the
    restriction compiler, but this factory raises ValueError for them:
    arbitrary local derivatives with respect to all C and rhs entries are
    not defined on that redundant representation. Empty C is valid and
    returns an empty left inverse. No Gaussian contraction bound is needed.

    Both defining identities are replayed exactly before sealing. Invalid
    restriction types raise TypeError; damaged dependencies, arithmetic or
    resource failures return None. Work is O(r^3+r^2*d+r*d^2+d^2+m*d), including
    identity checks and retained family metadata; storage is O(d^2+r*d+r^2
    +m*d). All rational intermediates use the restriction's shared budget.
    No transcendental is evaluated, and prior fallback history is preserved.
    The shared strict sealing policy may still require a rigorous backend.
    A certificate digest is not a theorem-prover verification.
    """
    if not isinstance(restriction, GaussianAffineRestriction):
        raise TypeError("restriction must be a GaussianAffineRestriction")
    dimension, rows = restriction.dimension, len(restriction.constraints)
    count = restriction.source_model.mode_count
    arithmetic = _Arithmetic(restriction.source_model.budget)
    try:
        # Include retained restriction/family data, exact Gram/factor/solve
        # scratch, transposes and paired-integer serialization. These count
        # scalar metadata slots, not Python byte allocations.
        arithmetic._entries(128*(count+rows+1)*(dimension+1) + 32*(dimension+1)**2
                            + 16*(rows+1)**2 + 512)
        arithmetic._work(128*(count+rows+1)*(dimension+1) + 128*(dimension+1)**2
                         + 64*(rows+1)**2 + 512)
        if not _intact(restriction.certificate):
            return None
    except (ArithmeticError, ValueError):
        return None
    if restriction.constraint_rank != rows:
        raise ValueError("adjoint geometry requires full row rank; redundant constraint rows have no arbitrary local parameter derivative")
    try:
        if rows:
            constraints = restriction.constraints
            gram = tuple(tuple(_dot(left, right, arithmetic) for right in constraints) for left in constraints)
            factor = factor_positive_definite(gram, arithmetic)
            columns = tuple(factor.solve(tuple(row[axis] for row in constraints), arithmetic)
                            for axis in range(dimension))
            left_inverse = tuple(tuple(column[row] for column in columns) for row in range(rows))
        else:
            left_inverse = ()
        _check_identities(restriction.constraints, left_inverse, restriction.projector, arithmetic)
        from omnibias.core.proof.certificate import make_certificate

        certificate = make_certificate(
            claim="Exact reusable left inverse and normal projector for independent affine constraint rows.",
            payload={
                "type": "gaussian_affine_adjoint_geometry",
                "restriction_certificate": deepcopy(restriction.certificate),
                "constraint_count": rows, "dimension": dimension,
                "left_inverse": [[_pair(value) for value in row] for row in left_inverse],
                "identities": ["L*C^T=I", "C^T*L=I-P"],
                "argument": "The intact exact restriction records rank(C)=rows(C). Hence C*C^T is positive definite. One exact LDLT factorization and one solve per coordinate construct L=(C*C^T)^(-1)*C. Both L*C^T=I and C^T*L=I-P are replayed with bounded rational arithmetic. Empty constraints give the empty L and the identity tangent projector.",
                "scope": "Geometry for the declared full-row-rank constraint representation only; no contraction, optimizer or derivative enclosure claim. Arbitrary local C/rhs derivatives are outside the domain at redundant rows.",
            },
            honesty={"unproven_claim": False, "continuum_pde_claim": False},
            meta={"transcend_backend": "not_used"},
        )
        result = object.__new__(GaussianAffineAdjointGeometry)
        object.__setattr__(result, "restriction", restriction)
        object.__setattr__(result, "left_inverse", left_inverse)
        object.__setattr__(result, "certificate", certificate)
        return result
    except (ArithmeticError, ValueError):
        return None


__all__ = ["GaussianAffineAdjointGeometry", "compile_gaussian_affine_adjoint"]
