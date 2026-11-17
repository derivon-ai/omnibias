# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Bounded exact rational geometry of affine equality sets.

This is Euclidean projection, not an optimization algorithm for inequalities.
All inputs and outputs are exact integers or :class:`fractions.Fraction`.
The implementation uses the shared rational elimination and LDLT routines;
it does not import Gaussian models, numerical backends, or transcendental code.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction

from omnibias.core._rational_linear import factor_positive_definite, solve_affine

Rational = int | Fraction
Vector = tuple[Fraction, ...]
Matrix = tuple[Vector, ...]


class AffineBudgetExceeded(ValueError):
    """An explicit exact-arithmetic resource limit was exhausted."""


class InconsistentAffineConstraints(ValueError):
    """An empty affine set has no Euclidean projection or finite distance."""


@dataclass(frozen=True)
class AffineBudget:
    """Per-operation bounds, not a guarantee about Python allocator bytes.

    ``max_entries`` bounds a conservative resident/scratch rational-slot count.
    ``max_work`` bounds charged arithmetic work, including shared elimination
    reservations. Every rational intermediate has bounded numerator and
    denominator bit lengths. Each construction or query starts a fresh meter.
    """

    max_dimension: int = 128
    max_rows: int = 512
    max_entries: int = 1_000_000
    max_work: int = 10_000_000
    max_coefficient_bits: int = 16_384

    def __post_init__(self) -> None:
        for name in ("max_dimension", "max_rows", "max_entries", "max_work", "max_coefficient_bits"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")


class _Arithmetic:
    def __init__(self, limits: AffineBudget) -> None:
        self.limits = limits
        self.work = 0

    def _entries(self, count: int) -> None:
        if count > self.limits.max_entries:
            raise AffineBudgetExceeded("affine rational storage exceeds max_entries")

    def _work(self, count: int) -> None:
        self.work += count
        if self.work > self.limits.max_work:
            raise AffineBudgetExceeded("affine rational arithmetic exceeds max_work")

    def _coefficient(self, value: Fraction) -> Fraction:
        self._work(1)
        limit = self.limits.max_coefficient_bits
        if value.numerator.bit_length() > limit or value.denominator.bit_length() > limit:
            raise AffineBudgetExceeded("affine rational coefficient exceeds max_coefficient_bits")
        return value

    def exact(self, value: Rational) -> Fraction:
        if isinstance(value, bool) or not isinstance(value, (int, Fraction)):
            raise TypeError("affine data must be exact int or Fraction, excluding bool")
        return self._coefficient(Fraction(value))

    def add(self, left: Fraction, right: Fraction) -> Fraction:
        return self._coefficient(left + right)

    def mul(self, left: Fraction, right: Fraction) -> Fraction:
        return self._coefficient(left * right)


def _sequence(value: object, name: str) -> None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a finite exact sequence")


def _vector(values: Sequence[Rational], dimension: int, arithmetic: _Arithmetic, name: str) -> Vector:
    _sequence(values, name)
    if len(values) != dimension:
        raise ValueError(f"{name} has the wrong dimension")
    return tuple(arithmetic.exact(value) for value in values)


def _dot(left: Vector, right: Vector, arithmetic: _Arithmetic) -> Fraction:
    result = Fraction(0)
    for a, b in zip(left, right, strict=True):
        result = arithmetic.add(result, arithmetic.mul(a, b))
    return result


def _matvec(matrix: Matrix, vector: Vector, arithmetic: _Arithmetic) -> Vector:
    return tuple(_dot(row, vector, arithmetic) for row in matrix)


def _storage(rows: int, dimension: int) -> int:
    # Input, augmented elimination, basis, Gram factors, projector, witness
    # checks and their simultaneous immutable/scratch copies.
    return 8 * rows * (dimension + 1) + 16 * dimension * dimension + 32 * dimension + 64


@dataclass(frozen=True, init=False)
class ExactAffineGeometry:
    """Exact geometry of ``constraints @ x = rhs`` in Euclidean coordinates.

    ``anchor`` is the minimum-norm feasible point and ``tangent_projector``
    is the symmetric orthogonal projector onto ``ker(constraints)``. Both
    are ``None`` when exact elimination proves inconsistency. Redundant
    consistent equations are supported without a numerical rank tolerance.

    Construct with :func:`exact_affine_geometry`. Witness identities are
    replayed using exact arithmetic; this is not theorem-prover verification.
    """

    constraints: Matrix
    rhs: Vector
    dimension: int
    rank: int
    consistent: bool
    anchor: Vector | None
    tangent_projector: Matrix | None
    budget: AffineBudget

    def __init__(self) -> None:
        raise TypeError("use exact_affine_geometry")

    @property
    def tangent_dimension(self) -> int:
        """Dimension of the homogeneous kernel, including an inconsistent set."""
        return self.dimension - self.rank

    def _point(self, point: Sequence[Rational]) -> tuple[Vector, _Arithmetic]:
        arithmetic = _Arithmetic(self.budget)
        arithmetic._entries(_storage(len(self.constraints), self.dimension))
        return _vector(point, self.dimension, arithmetic, "point"), arithmetic

    def residual(self, point: Sequence[Rational]) -> Vector:
        """Return the exact residual in the supplied equation coordinates."""
        frozen, arithmetic = self._point(point)
        return tuple(arithmetic.add(value, -rhs) for value, rhs in
                     zip(_matvec(self.constraints, frozen, arithmetic), self.rhs, strict=True))

    def contains(self, point: Sequence[Rational]) -> bool:
        """Exact membership; no tolerance or floating equality is used."""
        return not any(self.residual(point))

    def project(self, point: Sequence[Rational]) -> Vector:
        """Return the exact closest point; raise for inconsistency or limits."""
        frozen, arithmetic = self._point(point)
        if not self.consistent:
            raise InconsistentAffineConstraints("exact affine constraints are inconsistent")
        assert self.anchor is not None and self.tangent_projector is not None
        tangent = _matvec(self.tangent_projector, frozen, arithmetic)
        return tuple(arithmetic.add(a, value) for a, value in zip(self.anchor, tangent, strict=True))

    def _squared_distance(self, point: Sequence[Rational]) -> tuple[Fraction, _Arithmetic]:
        frozen, arithmetic = self._point(point)
        if not self.consistent:
            raise InconsistentAffineConstraints("exact affine constraints are inconsistent")
        assert self.anchor is not None and self.tangent_projector is not None
        tangent = _matvec(self.tangent_projector, frozen, arithmetic)
        correction = tuple(arithmetic.add(arithmetic.add(value, -a), -p)
                           for value, a, p in zip(frozen, self.anchor, tangent, strict=True))
        return _dot(correction, correction, arithmetic), arithmetic

    def squared_distance(self, point: Sequence[Rational]) -> Fraction:
        """Return the exact squared Euclidean distance; raise for an empty set."""
        return self._squared_distance(point)[0]

    def half_squared_distance(self, point: Sequence[Rational]) -> Fraction:
        """Return one half the exact squared distance, the geometric loss."""
        value, arithmetic = self._squared_distance(point)
        return arithmetic._coefficient(value / 2)


def _check_witness(
    matrix: Matrix, rhs: Vector, anchor: Vector, projector: Matrix, rank: int,
    arithmetic: _Arithmetic,
) -> None:
    dimension = len(anchor)
    zero = (Fraction(0),) * dimension
    if _matvec(matrix, anchor, arithmetic) != rhs or _matvec(projector, anchor, arithmetic) != zero:
        raise ArithmeticError("exact affine anchor identities failed")
    columns = tuple(tuple(row[j] for row in projector) for j in range(dimension))
    if projector != columns:
        raise ArithmeticError("exact tangent projector is not symmetric")
    for column in columns:
        if any(_matvec(matrix, column, arithmetic)) or _matvec(projector, column, arithmetic) != column:
            raise ArithmeticError("exact tangent projector identities failed")
    trace = Fraction(0)
    for i in range(dimension):
        trace = arithmetic.add(trace, projector[i][i])
    if trace != dimension - rank:
        raise ArithmeticError("exact tangent projector rank identity failed")


def exact_affine_geometry(
    constraints: Sequence[Sequence[Rational]], rhs: Sequence[Rational], *,
    dimension: int | None = None, budget: AffineBudget | None = None,
) -> ExactAffineGeometry:
    """Construct bounded exact geometry, retaining a proven empty-set verdict.

    Infer dimension from a nonempty matrix; an empty matrix requires an
    explicit positive ``dimension``. Empty and redundant consistent rows are
    supported. Shape/type errors raise; work, storage and coefficient limits
    raise :class:`AffineBudgetExceeded`. No resource refusal is interpreted
    as inconsistency. No floating-point numbers are accepted.
    """
    limits = AffineBudget() if budget is None else budget
    if not isinstance(limits, AffineBudget):
        raise TypeError("budget must be an AffineBudget")
    _sequence(constraints, "constraints")
    _sequence(rhs, "rhs")
    rows = len(constraints)
    if rows > limits.max_rows:
        raise AffineBudgetExceeded("affine row count exceeds max_rows")
    if rows:
        _sequence(constraints[0], "constraint row")
        inferred = len(constraints[0])
        if dimension is None:
            dimension = inferred
        elif dimension != inferred:
            raise ValueError("dimension disagrees with the constraint matrix")
    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 1:
        raise ValueError("dimension must be a positive integer, explicit for empty constraints")
    if dimension > limits.max_dimension:
        raise AffineBudgetExceeded("affine dimension exceeds max_dimension")
    arithmetic = _Arithmetic(limits)
    arithmetic._entries(_storage(rows, dimension))
    right = _vector(rhs, rows, arithmetic, "rhs")
    matrix = tuple(_vector(row, dimension, arithmetic, "constraint row") for row in constraints)
    zero = (Fraction(0),) * dimension
    identity = tuple(tuple(Fraction(int(i == j)) for j in range(dimension)) for i in range(dimension))
    if matrix:
        solved = solve_affine(matrix, right, arithmetic)
        consistent, rank = solved.consistent, solved.rank
        anchor = solved.particular
        if not consistent:
            projector: Matrix | None = None
        elif rank == dimension:
            projector = (zero,) * dimension
        elif rank == 0:
            anchor, projector = zero, identity
        else:
            basis = solved.homogeneous_basis
            gram = tuple(tuple(_dot(left, other, arithmetic) for other in basis) for left in basis)
            factor = factor_positive_definite(gram, arithmetic)
            coordinates = tuple(tuple(vector[axis] for vector in basis) for axis in range(dimension))
            columns = tuple(factor.solve(row, arithmetic) for row in coordinates)
            projector = tuple(tuple(_dot(left, right_column, arithmetic) for right_column in columns)
                              for left in coordinates)
            assert anchor is not None
            tangent_anchor = _matvec(projector, anchor, arithmetic)
            anchor = tuple(arithmetic.add(value, -tangent) for value, tangent in
                           zip(anchor, tangent_anchor, strict=True))
    else:
        consistent, rank, anchor, projector = True, 0, zero, identity
    if consistent:
        assert anchor is not None and projector is not None
        _check_witness(matrix, right, anchor, projector, rank, arithmetic)
    result = object.__new__(ExactAffineGeometry)
    for name, value in (("constraints", matrix), ("rhs", right), ("dimension", dimension),
                        ("rank", rank), ("consistent", consistent), ("anchor", anchor),
                        ("tangent_projector", projector), ("budget", limits)):
        object.__setattr__(result, name, value)
    return result


__all__ = ["AffineBudget", "AffineBudgetExceeded", "ExactAffineGeometry",
           "InconsistentAffineConstraints", "exact_affine_geometry"]
