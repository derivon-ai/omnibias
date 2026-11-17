# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Exact fixed-time restriction of Gaussian heat mixtures to affine sets.

Rational elimination determines feasibility and rank. A nullspace Gram
factorization gives the orthogonal tangent projector without floating rank
thresholds, square roots, or a new linear solver.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING

from omnibias.core._rational_linear import factor_positive_definite, solve_affine
from omnibias.core.gaussian_jet import _Arithmetic
from omnibias.core.verified.gaussian_heat import GaussianHeatMap
from omnibias.core.verified.inverse_gaussian_heat import _pair

if TYPE_CHECKING:
    from omnibias.core.proof.certificate import Cert

Rational = int | Fraction
Vector = tuple[Fraction, ...]
Matrix = tuple[Vector, ...]


class InconsistentAffineConstraints(ValueError):
    """Exact rational elimination proved that the affine feasible set is empty."""


def _validate_vector(values: Sequence[Rational], dimension: int, name: str) -> None:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a finite exact sequence")
    if len(values) != dimension:
        raise ValueError(f"{name} has the wrong dimension")
    if any(isinstance(value, bool) or not isinstance(value, (int, Fraction)) for value in values):
        raise TypeError(f"{name} entries must be exact int or Fraction, excluding bool")


def _dot(left: Vector, right: Vector, arithmetic: _Arithmetic) -> Fraction:
    total = Fraction(0)
    for a, b in zip(left, right, strict=True):
        total = arithmetic.add(total, arithmetic.mul(a, b))
    return total


def _matvec(matrix: Matrix, vector: Vector, arithmetic: _Arithmetic) -> Vector:
    return tuple(_dot(row, vector, arithmetic) for row in matrix)


def _storage(rows: int, dimension: int, count: int) -> int:
    # Resident source/transformed metadata, immutable geometry, elimination
    # and Gram scratch, and paired-integer certificate serialization. Slots,
    # not Python byte estimates; dimensions are checked before allocation.
    return (8 * rows * (dimension + 1) + 16 * dimension * dimension
            + 32 * count * dimension + 24 * count + 32 * dimension + 256)


@dataclass(frozen=True, init=False)
class GaussianAffineRestriction:
    r"""The Gaussian family restricted to ``constraints @ x = rhs``.

    ``anchor`` is the minimum-norm feasible point, and ``projector`` is the
    orthogonal projector onto the tangent space. The transformed ``model``
    has centers ``anchor + projector @ a_j``. On the feasible set its density
    equals the source density times a positive x-independent factor. Its
    inverse at ``project_target(v)`` therefore solves the original potential
    minimization with linear tilt v, whenever that inverse is certified.

    The transformation itself makes no contraction or optimizer claim.
    Transformed biases depend on the source time: time derivatives or aging
    of this fixed-bias model do not automatically describe the source family
    with fixed constraints. Rank-zero restrictions reuse the original model.

    Construct with :func:`restrict_gaussian_heat`. The certificate records
    exact finite identities, not a theorem-prover verification.
    """

    source_model: GaussianHeatMap
    constraints: Matrix
    rhs: Vector
    constraint_rank: int
    anchor: Vector
    projector: Matrix
    normal_cost_differences: Vector
    model: GaussianHeatMap
    certificate: Cert

    def __init__(self) -> None:
        raise TypeError("use restrict_gaussian_heat")

    @property
    def dimension(self) -> int:
        return self.source_model.dimension

    @property
    def tangent_dimension(self) -> int:
        return self.dimension - self.constraint_rank

    def _project(self, values: Sequence[Rational], *, point: bool) -> Vector:
        _validate_vector(values, self.dimension, "point" if point else "target")
        arithmetic = _Arithmetic(self.source_model.budget)
        arithmetic._entries(_storage(len(self.constraints), self.dimension, self.source_model.mode_count))
        arithmetic._work(8 * self.dimension * self.dimension + 8 * self.dimension)
        frozen = tuple(arithmetic.exact(value) for value in values)
        projected = _matvec(self.projector, frozen, arithmetic)
        if point:
            return tuple(arithmetic.add(a, value) for a, value in zip(self.anchor, projected, strict=True))
        return projected

    def project_target(self, target: Sequence[Rational]) -> Vector:
        """Return P target exactly; invalid data or exhausted budgets raise."""
        return self._project(target, point=False)

    def project_point(self, point: Sequence[Rational]) -> Vector:
        """Return the orthogonal projection x0+P point; failures raise."""
        return self._project(point, point=True)


def _geometry(
    constraints: Matrix, rhs: Vector, dimension: int, arithmetic: _Arithmetic,
) -> tuple[int, Vector, Matrix]:
    zero = (Fraction(0),) * dimension
    if not constraints:
        identity = tuple(tuple(Fraction(int(i == j)) for j in range(dimension)) for i in range(dimension))
        return 0, zero, identity
    solved = solve_affine(constraints, rhs, arithmetic)
    if not solved.consistent:
        raise InconsistentAffineConstraints("exact affine constraints are inconsistent")
    assert solved.particular is not None
    if solved.rank == dimension:
        return solved.rank, solved.particular, (zero,) * dimension
    if solved.rank == 0:
        identity = tuple(tuple(Fraction(int(i == j)) for j in range(dimension)) for i in range(dimension))
        return 0, zero, identity
    basis = solved.homogeneous_basis  # Basis vectors are rows here, N's columns mathematically.
    gram = tuple(tuple(_dot(left, right, arithmetic) for right in basis) for left in basis)
    factor = factor_positive_definite(gram, arithmetic)
    coordinates = tuple(tuple(vector[axis] for vector in basis) for axis in range(dimension))
    columns = tuple(factor.solve(coordinates[axis], arithmetic) for axis in range(dimension))
    projector = tuple(tuple(_dot(coordinates[i], columns[j], arithmetic) for j in range(dimension))
                      for i in range(dimension))
    projected = _matvec(projector, solved.particular, arithmetic)
    anchor = tuple(arithmetic.add(value, -part) for value, part in zip(solved.particular, projected, strict=True))
    return solved.rank, anchor, projector


def _check_geometry(
    constraints: Matrix, rhs: Vector, rank: int, anchor: Vector, projector: Matrix, arithmetic: _Arithmetic,
) -> None:
    """Replay exact witness identities before making a certificate claim."""
    dimension = len(anchor)
    zero = (Fraction(0),) * dimension
    if _matvec(constraints, anchor, arithmetic) != rhs or _matvec(projector, anchor, arithmetic) != zero:
        raise ArithmeticError("affine anchor identities failed")
    columns = tuple(tuple(row[j] for row in projector) for j in range(dimension))
    if columns != projector:
        raise ArithmeticError("tangent projector is not symmetric")
    for column in columns:
        if any(_matvec(constraints, column, arithmetic)) or _matvec(projector, column, arithmetic) != column:
            raise ArithmeticError("tangent projector identities failed")
    if arithmetic.sum(tuple(projector[i][i] for i in range(dimension))) != dimension - rank:
        raise ArithmeticError("tangent projector rank identity failed")


def restrict_gaussian_heat(
    source_model: GaussianHeatMap, constraints: Sequence[Sequence[Rational]], rhs: Sequence[Rational],
) -> GaussianAffineRestriction | None:
    r"""Compile an exact affine restriction at the source model's fixed time.

    Empty constraints and redundant consistent rows are supported. Invalid
    input types/shapes raise; exact inconsistency raises
    :class:`InconsistentAffineConstraints`. Exhausted exact arithmetic or
    resource bounds return None. Construction does not require q<1.

    Write P for the tangent projector, x0 for the minimum-norm anchor,
    r_j=a_j-a_0, n_j=r_j-P r_j, and n0=(I-P)a_0-x0. The differences
    c_j-c_0=2*n0.n_j+||n_j||² are computed without squaring the common normal
    offset. Positive-rank transformed biases are
    (b_j-b_0)-(c_j-c_0)/(4*nu*time). This exact gauge avoids both large common
    bias subtraction in floating arithmetic and unnecessary common energies.

    No transcendental evaluation occurs. Sealing retains the shared strict
    backend policy, which may raise RuntimeError if its backend is absent.
    """
    if not isinstance(source_model, GaussianHeatMap):
        raise TypeError("source_model must be a GaussianHeatMap")
    if not isinstance(constraints, Sequence) or isinstance(constraints, (str, bytes)):
        raise TypeError("constraints must be a finite exact matrix")
    dimension, count, rows = source_model.dimension, source_model.mode_count, len(constraints)
    _validate_vector(rhs, rows, "rhs")
    for row in constraints:
        _validate_vector(row, dimension, "constraint row")
    arithmetic = _Arithmetic(source_model.budget)
    try:
        arithmetic._entries(_storage(rows, dimension, count))
        arithmetic._work(32 * (rows + count + dimension + 1) * (dimension + 1) ** 2 + 256)
        matrix = tuple(tuple(arithmetic.exact(value) for value in row) for row in constraints)
        right = tuple(arithmetic.exact(value) for value in rhs)
        rank, anchor, projector = _geometry(matrix, right, dimension, arithmetic)
        _check_geometry(matrix, right, rank, anchor, projector, arithmetic)
        if rank == 0:
            transformed = source_model
            differences = (Fraction(0),) * count
            normal_reference = (Fraction(0),) * dimension
        else:
            projected_reference = _matvec(projector, source_model.centers[0], arithmetic)
            center_reference = tuple(arithmetic.add(a, b) for a, b in zip(anchor, projected_reference, strict=True))
            normal_reference = tuple(arithmetic.add(a, -b) for a, b in zip(
                source_model.centers[0], center_reference, strict=True))
            denominator = arithmetic.mul(arithmetic.mul(Fraction(4), source_model.viscosity), source_model.time)
            centers: list[Vector] = []
            biases: list[Fraction] = []
            costs: list[Fraction] = []
            for center, bias in zip(source_model.centers, source_model.biases, strict=True):
                relative = tuple(arithmetic.add(a, -a0) for a, a0 in zip(center, source_model.centers[0], strict=True))
                tangent = _matvec(projector, relative, arithmetic)
                normal = tuple(arithmetic.add(a, -b) for a, b in zip(relative, tangent, strict=True))
                cost = arithmetic.add(arithmetic.mul(Fraction(2), _dot(normal_reference, normal, arithmetic)),
                                      _dot(normal, normal, arithmetic))
                costs.append(cost)
                centers.append(tuple(arithmetic.add(a, b) for a, b in zip(center_reference, tangent, strict=True)))
                shifted = arithmetic.add(bias, -source_model.biases[0])
                biases.append(arithmetic.add(shifted, -arithmetic._coefficient(cost / denominator)))
            differences = tuple(costs)
            # Both constructors below (GaussianHeatMap and its MeanMap) have
            # local budgets. Reserve their combined conservative construction
            # cost in this operation's cumulative meter before calling them.
            arithmetic._work(512 * (count + 1) * (dimension + 1) ** 2 + 128 * count * count * dimension)
            transformed = GaussianHeatMap(tuple(centers), tuple(biases), viscosity=source_model.viscosity,
                                          time=source_model.time, budget=source_model.budget)
        from omnibias.core.proof.certificate import make_certificate

        certificate = make_certificate(
            claim="Exact fixed-time restriction of a Gaussian heat mixture to a consistent rational affine set.",
            payload={
                "type": "gaussian_heat_affine_restriction",
                "source_centers": [[_pair(value) for value in row] for row in source_model.centers],
                "source_biases": [_pair(value) for value in source_model.biases],
                "viscosity": _pair(source_model.viscosity), "time": _pair(source_model.time),
                "constraints": [[_pair(value) for value in row] for row in matrix],
                "rhs": [_pair(value) for value in right], "constraint_rank": rank,
                "anchor": [_pair(value) for value in anchor],
                "projector": [[_pair(value) for value in row] for row in projector],
                "normal_reference": [_pair(value) for value in normal_reference],
                "normal_cost_differences": [_pair(value) for value in differences],
                "centers": [[_pair(value) for value in row] for row in transformed.centers],
                "biases": [_pair(value) for value in transformed.biases],
                "source_diameter_squared": _pair(source_model.diameter_squared),
                "diameter_squared": _pair(transformed.diameter_squared),
                "contraction_bound": _pair(transformed.contraction_bound), "source_model_reused": rank == 0,
                "identities": ["C*x0=d", "C*P=0", "P^T=P", "P^2=P", "P*x0=0", "trace(P)=dimension-rank(C)"],
                "density_gauge": "1" if rank == 0 else "tau_restricted=exp(-b_0+c_0/(4*nu*time))*tau_source on C*x=d; c_0=||normal_reference||^2",
                "argument": "Exact affine elimination establishes consistency and rank. The nullspace Gram factorization gives the symmetric idempotent tangent projector and minimum-norm anchor, whose identities are replayed exactly. Orthogonal decomposition of each squared Gaussian distance on the feasible set proves the density gauge, using c_j-c_0=2*normal_reference.n_j+||n_j||^2. Tangential gradients therefore agree. The transformed inverse at P*v, if certified, is feasible and minimizes the original potential minus v.x on the affine set.",
                "scope": "fixed source time; no contraction, uniqueness, time-evolution, or inequality-constraint claim",
            },
            honesty={"unproven_claim": False, "continuum_pde_claim": False},
            meta={"transcend_backend": "not_used"},
        )
        result = object.__new__(GaussianAffineRestriction)
        for name, value in (("source_model", source_model), ("constraints", matrix), ("rhs", right),
                            ("constraint_rank", rank), ("anchor", anchor), ("projector", projector),
                            ("normal_cost_differences", differences), ("model", transformed), ("certificate", certificate)):
            object.__setattr__(result, name, value)
        return result
    except InconsistentAffineConstraints:
        raise
    except (ArithmeticError, ValueError):
        return None


__all__ = ["GaussianAffineRestriction", "InconsistentAffineConstraints", "restrict_gaussian_heat"]
