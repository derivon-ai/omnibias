# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Budgeted exact Euclidean projection onto finite rational polyhedra."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations
from typing import TYPE_CHECKING

from omnibias.core._rational_linear import (
    RationalBudget,
    RationalLDLT,
    factor_positive_definite,
    solve_affine,
)
from omnibias.core.ridge_system import RidgeBudget

if TYPE_CHECKING:
    from omnibias.core.proof.certificate import Cert

Rational = int | Fraction
Vector = tuple[Fraction, ...]
Matrix = tuple[Vector, ...]


class _Arithmetic:
    def __init__(self, budget: RidgeBudget) -> None:
        self.budget, self.used = budget, 0

    def _entries(self, count: int) -> None:
        self.budget._entries(count)

    def _work(self, count: int) -> None:
        self.used += count
        self.budget._work(self.used)

    def _coefficient(self, value: Fraction) -> Fraction:
        self._work(1)
        return self.budget._coefficient(value)


def _validate_vector(values: Sequence[Rational], size: int, label: str) -> None:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise TypeError(f"{label} must be a finite exact sequence")
    if len(values) != size:
        raise ValueError(f"{label} must have length {size}")
    if any(isinstance(value, bool) or not isinstance(value, (int, Fraction)) for value in values):
        raise TypeError(f"{label} entries must be int or Fraction, excluding bool")


def _dot(left: Vector, right: Vector, arithmetic: RationalBudget) -> Fraction:
    total = Fraction(0)
    for a, b in zip(left, right, strict=True):
        total = arithmetic._coefficient(total+arithmetic._coefficient(a*b))
    return total


def _pair(value: Fraction) -> list[int]:
    return [value.numerator, value.denominator]


def _intact(certificate: Cert) -> bool:
    from omnibias.core.proof.certificate import verify_certificate_digest

    try:
        return verify_certificate_digest(certificate)
    except (TypeError, ValueError, RecursionError):
        return False


@dataclass(frozen=True)
class _Basis:
    active_indices: tuple[int, ...]
    factor: RationalLDLT | None


def _source_slots(projector: PolyhedralProjector) -> int:
    """Conservative scalar slots for source metadata, factors and a snapshot."""
    dimension, rows = projector.dimension, len(projector.inequalities)
    rank = min(dimension, rows)
    return 32*(rows+1)*(dimension+1)+64*len(projector.bases)*(rank*rank+rank+1)+2048


def _query_work_bound(projector: PolyhedralProjector) -> int:
    """Bound every stored-basis solve and original-row KKT check in one query.

    Includes exact input bit checks and public-result distance/multiplier work,
    but not hashing/copying source certificates; reserve _source_slots too.
    """
    dimension, rows = projector.dimension, len(projector.inequalities)
    rank = min(dimension, rows)
    per_basis = 64*(rank+1)**2+32*(rank+1)*(dimension+1)+8*(rows+1)*(dimension+1)
    return len(projector.bases)*per_basis+64*(dimension+rows+1)


def _project_exact(
    projector: PolyhedralProjector, point: Vector, arithmetic: RationalBudget,
) -> tuple[Vector, tuple[int, ...], Vector] | None:
    """Exact projection kernel; caller owns validation, integrity and reservation.

    Uses the caller's cumulative meter and stored factors. No certificate
    copying/sealing or transcendental evaluation occurs here. A failed budget
    raises ValueError for the caller to handle; no valid basis returns None.
    """
    checked = arithmetic._coefficient
    source = tuple(checked(value) for value in point)
    for basis in projector.bases:
        arithmetic._work(8*(projector.dimension+len(projector.inequalities)+len(basis.active_indices)+1))
        selected = tuple(projector.inequalities[index] for index in basis.active_indices)
        if basis.factor is None:
            multipliers: Vector = ()
        else:
            rhs = tuple(checked(_dot(row, source, arithmetic)-projector.upper_bounds[index])
                        for row, index in zip(selected, basis.active_indices, strict=True))
            multipliers = basis.factor.solve(rhs, arithmetic)
        if any(value < 0 for value in multipliers):
            continue
        correction = tuple(_dot(tuple(row[j] for row in selected), multipliers, arithmetic)
                           for j in range(projector.dimension))
        candidate = tuple(checked(value-change) for value, change in zip(source, correction, strict=True))
        slacks = tuple(checked(bound-_dot(row, candidate, arithmetic))
                       for row, bound in zip(projector.inequalities, projector.upper_bounds, strict=True))
        if any(value < 0 for value in slacks) or any(slacks[index] for index in basis.active_indices):
            continue
        if any(checked(checked(value-origin)+change) for value, origin, change in zip(candidate, source, correction, strict=True)):
            continue
        return candidate, basis.active_indices, multipliers
    return None


@dataclass(frozen=True, init=False)
class PolyhedralProjector:
    """Compiled complete rational active-basis family for G*x<=h.

    Compilation includes every independent subset of at most dimension rows,
    including the empty set, and factorizes each Gram matrix once. Redundant,
    zero and opposite inequalities and lower-dimensional polyhedra are allowed.
    Enumeration can be combinatorial and is bounded before allocation.
    Compilation asserts neither feasibility nor an optimizer at any query.
    """

    dimension: int
    inequalities: Matrix
    upper_bounds: Vector
    budget: RidgeBudget
    bases: tuple[_Basis, ...]
    enumerated_bases: int
    certificate: Cert

    def __init__(self) -> None:
        raise TypeError("use compile_polyhedral_projector")

    def project(self, point: Sequence[Rational]) -> PolyhedralProjection | None:
        """Certify the unique exact Euclidean projection of an exact point.

        Invalid types/shapes raise. Damaged proof data, resource exhaustion,
        or no valid KKT basis return None without asserting infeasibility.
        This rational query needs no transcendental backend and preserves
        shared strict-sealing policy and sticky provenance without resetting it.
        """
        _validate_vector(point, self.dimension, "point")
        arithmetic = _Arithmetic(self.budget)
        try:
            slots = _source_slots(self)
            arithmetic._entries(slots+64*(self.dimension+len(self.inequalities)+1))
            self.budget._work(4*slots+_query_work_bound(self))
            arithmetic._work(4*slots)
            if not _intact(self.certificate):
                return None
            source = tuple(arithmetic._coefficient(Fraction(value)) for value in point)
            solved = _project_exact(self, source, arithmetic)
            if solved is None:
                return None
            projected, active, selected = solved
            multipliers = [Fraction(0)]*len(self.inequalities)
            for index, value in zip(active, selected, strict=True):
                multipliers[index] = value
            displacement = tuple(arithmetic._coefficient(value-origin) for value, origin in zip(projected, source, strict=True))
            distance = _dot(displacement, displacement, arithmetic)
            from omnibias.core.proof.certificate import make_certificate

            certificate = make_certificate(
                claim="Unique exact Euclidean projection onto the original rational polyhedron.",
                payload={
                    "type": "polyhedral_projection", "projector_certificate": deepcopy(self.certificate),
                    "source_point": [_pair(value) for value in source], "point": [_pair(value) for value in projected],
                    "active_indices": list(active), "multipliers": [_pair(value) for value in multipliers],
                    "squared_distance": _pair(distance),
                    "argument": "The exact returned point satisfies every original inequality. Stored selected multipliers are nonnegative, omitted multipliers are zero, selected slacks vanish, and x-z+G^T*eta=0 is replayed exactly. These KKT conditions prove the global minimum of ||x-z||²/2 on the convex polyhedron. Strict convexity makes this projection unique.",
                    "scope": "Exact finite rational constraints and query. No feasibility conclusion follows from refusal, and no differentiability, approximate-data robustness, or external formal proof-checker claim is made.",
                }, honesty={"unproven_claim": False, "continuum_pde_claim": False}, meta={"transcend_backend": "not_used"},
            )
            result = object.__new__(PolyhedralProjection)
            for name, stored in (("projector", self), ("source_point", source), ("point", projected), ("active_indices", active),
                                ("multipliers", tuple(multipliers)), ("squared_distance", distance), ("certificate", certificate)):
                object.__setattr__(result, name, stored)
            return result
        except (ArithmeticError, ValueError):
            return None


@dataclass(frozen=True, init=False)
class PolyhedralProjection:
    """Exact primal projection and nonnegative original-row KKT witness."""

    projector: PolyhedralProjector
    source_point: Vector
    point: Vector
    active_indices: tuple[int, ...]
    multipliers: Vector
    squared_distance: Fraction
    certificate: Cert

    def __init__(self) -> None:
        raise TypeError("use PolyhedralProjector.project")


def compile_polyhedral_projector(
    dimension: int, inequalities: Sequence[Sequence[Rational]], upper_bounds: Sequence[Rational], *,
    budget: RidgeBudget | None = None,
) -> PolyhedralProjector | None:
    """Compile every independent rational active set, with complete preflight.

    Positive integer dimension and exact int/Fraction constraints are required;
    bool and inexact data are refused. Invalid inputs raise. Exceeding exact
    resource limits returns None, never a partial active-set family. Infeasible
    polyhedra may compile successfully; queries return no projection witness.
    No Gaussian, tensor or transcendental computations are required.
    """
    if type(dimension) is not int:
        raise TypeError("dimension must be an integer excluding bool")
    if dimension < 1:
        raise ValueError("dimension must be positive")
    if not isinstance(inequalities, Sequence) or isinstance(inequalities, (str, bytes)):
        raise TypeError("inequalities must be a finite exact matrix")
    limits = RidgeBudget() if budget is None else budget
    if not isinstance(limits, RidgeBudget):
        raise TypeError("budget must be a RidgeBudget")
    rows, rank = len(inequalities), min(len(inequalities), dimension)
    _validate_vector(upper_bounds, rows, "upper_bounds")
    for row in inequalities:
        _validate_vector(row, dimension, "inequality row")
    arithmetic = _Arithmetic(limits)
    try:
        source_slots = 32*(rows+1)*(dimension+1)+2048
        per_basis_slots = 64*(rank*rank+rank+1)
        per_basis_work = 64*(rank+1)**3+64*(dimension+1)*(rank+1)**2
        arithmetic._entries(source_slots+per_basis_slots)
        arithmetic._work(4*source_slots)
    except ValueError:
        return None
    try:
        count = term = 1
        for size in range(1, rank+1):
            term = term*(rows-size+1)//size
            count += term
            arithmetic._entries(source_slots+count*per_basis_slots)
            limits._work(arithmetic.used+count*per_basis_work)
        arithmetic._entries(source_slots+count*per_basis_slots)
        limits._work(arithmetic.used+count*per_basis_work)
        matrix = tuple(tuple(arithmetic._coefficient(Fraction(value)) for value in row) for row in inequalities)
        upper = tuple(arithmetic._coefficient(Fraction(value)) for value in upper_bounds)
        bases = [_Basis((), None)]
        enumerated = 1
        for size in range(1, rank+1):
            for active in combinations(range(rows), size):
                enumerated += 1
                arithmetic._work(8*(dimension+size+1))
                selected = tuple(matrix[index] for index in active)
                if solve_affine(selected, None, arithmetic).rank != size:
                    continue
                gram = tuple(tuple(_dot(left, right, arithmetic) for right in selected) for left in selected)
                bases.append(_Basis(active, factor_positive_definite(gram, arithmetic)))
        if enumerated != count:
            return None
        from omnibias.core.proof.certificate import make_certificate

        certificate = make_certificate(
            claim="Complete compiled independent active-basis family for exact rational polyhedral projection.",
            payload={
                "type": "polyhedral_projector", "dimension": dimension,
                "inequalities": [[_pair(value) for value in row] for row in matrix], "upper_bounds": [_pair(value) for value in upper],
                "enumerated_bases": count,
                "bases": [{"active_indices": list(basis.active_indices),
                           "lower": None if basis.factor is None else [[_pair(value) for value in row] for row in basis.factor.lower],
                           "diagonal": None if basis.factor is None else [_pair(value) for value in basis.factor.diagonal]} for basis in bases],
                "argument": "Every subset of up to the ambient dimension rows was enumerated after complete binomial resource preflight, including the empty subset. Exact elimination retains every independent subset, whose symmetric positive-definite Gram matrix is factored by rational LDLT. For any nonempty closed polyhedron a unique Euclidean projection exists. Its normal lies in the cone of active rows; conic Caratheodory gives an independent representation of at most dimension rows. Consequently some stored basis supports exact feasible nonnegative KKT multipliers for every query with a projection. Compilation alone asserts neither feasibility nor a projected point.",
                "scope": "Exact fixed finite rational data. Enumeration can be combinatorial; no fast general optimization, differentiability, infeasibility or external formal proof-checker claim.",
            }, honesty={"unproven_claim": False, "continuum_pde_claim": False}, meta={"transcend_backend": "not_used"},
        )
        result = object.__new__(PolyhedralProjector)
        for name, value in (("dimension", dimension), ("inequalities", matrix), ("upper_bounds", upper), ("budget", limits),
                            ("bases", tuple(bases)), ("enumerated_bases", enumerated), ("certificate", certificate)):
            object.__setattr__(result, name, value)
        return result
    except (ArithmeticError, ValueError):
        return None


__all__ = ["PolyhedralProjection", "PolyhedralProjector", "compile_polyhedral_projector"]
