# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Exact derivative packs satisfying constant-coefficient PDE systems.

For a nonzero rational direction ``w``, restriction replaces ``D**alpha`` by
``w**alpha * s**sum(alpha)``. A polynomial vector ``Q(s)`` in the right kernel
of the resulting matrix ``A_w(s)`` supplies solutions
``u(x) = Q(d/dz) phi(z)``, ``z = w.x + b``, for every sufficiently smooth
scalar profile. Derivatives are RAW, not Taylor-normalized coefficients.

The kernel routine returns a rational vector-space basis through a requested
degree. It does not compute a minimal polynomial-module basis, find suitable
directions, or solve boundary conditions. All construction is bounded exact
host arithmetic; no numerical backend is imported.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from operator import index as integer_index
from typing import Any

from omnibias.core._rational_linear import solve_affine
from omnibias.core.operator_spec import OperatorSpec

Rational = int | Fraction
Vector = tuple[Fraction, ...]
Matrix = tuple[Vector, ...]


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    try:
        result = integer_index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer >= {minimum}") from exc
    if result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def _exact(value: Rational) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (int, Fraction)):
        raise TypeError("ridge metadata must be int or Fraction")
    return Fraction(value)


def _direction(values: Sequence[Rational]) -> Vector:
    result = tuple(_exact(value) for value in values)
    if not result or not any(result):
        raise ValueError("ridge direction must be nonempty and nonzero")
    return result


@dataclass(frozen=True)
class RidgeBudget:
    """Limits on exact host construction, including intermediate rational bits.

    ``max_matrix_entries`` bounds each dense coefficient/convolution matrix
    and the total returned basis storage. ``max_elimination_work`` bounds a
    conservative scalar-operation estimate before elimination begins.
    """

    max_degree: int = 32
    max_order: int = 128
    max_matrix_entries: int = 262_144
    max_elimination_work: int = 16_000_000
    max_coefficient_bits: int = 4096

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            minimum = 0 if name in {"max_degree", "max_order"} else 1
            object.__setattr__(self, name, _integer(getattr(self, name), name, minimum))

    def _entries(self, count: int) -> None:
        if count > self.max_matrix_entries:
            raise ValueError("ridge matrix entries exceed the resource budget")

    def _work(self, count: int) -> None:
        if count > self.max_elimination_work:
            raise ValueError("ridge arithmetic work exceeds the resource budget")

    def _coefficient(self, value: Fraction) -> Fraction:
        if max(value.numerator.bit_length(), value.denominator.bit_length()) > self.max_coefficient_bits:
            raise ValueError("ridge coefficient exceeds the exact bit budget")
        return value


def _power(value: Fraction, order: int, budget: RidgeBudget) -> Fraction:
    result = Fraction(1)
    for _ in range(order):
        result = budget._coefficient(result * value)
    return result


@dataclass(frozen=True)
class RidgeSyzygy:
    """Vector derivative pack: ``coefficients[j][c]`` multiplies ``phi^(j)``.

    Coefficients are derivative-major. Trailing all-zero rows are removed;
    the zero pack retains one row and consequently has degree zero.
    Direct construction checks metadata only; use a system's
    ``residual_coefficients`` to check membership in its kernel.
    """

    direction: Vector
    coefficients: tuple[Vector, ...]

    def __post_init__(self) -> None:
        direction = _direction(self.direction)
        rows = tuple(tuple(_exact(value) for value in row) for row in self.coefficients)
        if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
            raise ValueError("ridge pack coefficients must be nonempty and rectangular")
        while len(rows) > 1 and not any(rows[-1]):
            rows = rows[:-1]
        object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "coefficients", rows)

    @property
    def degree(self) -> int:
        return len(self.coefficients) - 1

    @property
    def outputs(self) -> int:
        return len(self.coefficients[0])


@dataclass(frozen=True)
class RidgeSystem:
    """Restricted matrix polynomial ``sum(matrices[k] * s**k)`` over rationals.

    Matrix rows are equations and columns are output-field components. Bias
    does not enter the restriction and remains free in any resulting pack.
    """

    direction: Vector
    matrices: tuple[Matrix, ...]
    budget: RidgeBudget = RidgeBudget()

    def __post_init__(self) -> None:
        if not isinstance(self.budget, RidgeBudget):
            raise TypeError("budget must be a RidgeBudget")
        limits = self.budget
        limits._entries(len(self.direction))
        direction = _direction(self.direction)
        for value in direction:
            limits._coefficient(value)
        if not self.matrices or not self.matrices[0] or not self.matrices[0][0]:
            raise ValueError("ridge matrices must have positive equation and output counts")
        equations, outputs = len(self.matrices[0]), len(self.matrices[0][0])
        if len(self.matrices) - 1 > limits.max_order:
            raise ValueError("ridge operator order exceeds the resource budget")
        limits._entries(len(self.matrices) * equations * outputs)
        matrices: list[Matrix] = []
        for matrix in self.matrices:
            if len(matrix) != equations or any(len(row) != outputs for row in matrix):
                raise ValueError("ridge matrices must share a rectangular shape")
            matrices.append(tuple(tuple(limits._coefficient(_exact(v)) for v in row) for row in matrix))
        while len(matrices) > 1 and not any(v for row in matrices[-1] for v in row):
            matrices.pop()
        object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "matrices", tuple(matrices))

    @classmethod
    def from_operators(
        cls, operators: Sequence[Sequence[OperatorSpec | None]],
        direction: Sequence[Rational], *, budget: RidgeBudget | None = None,
    ) -> RidgeSystem:
        """Restrict scalar operators; ``None`` denotes a zero matrix entry.

        Directional requests have a separate runtime direction and are refused.
        Laplacian powers use ``(sum(w_i**2))**power`` without symbol expansion.
        """
        limits = budget if budget is not None else RidgeBudget()
        if not isinstance(limits, RidgeBudget):
            raise TypeError("budget must be a RidgeBudget")
        limits._entries(len(direction))
        normal = _direction(direction)
        for value in normal:
            limits._coefficient(value)
        if not operators or not operators[0] or any(len(row) != len(operators[0]) for row in operators):
            raise ValueError("operators must form a nonempty rectangular matrix")
        equations, outputs = len(operators), len(operators[0])
        limits._entries(equations * outputs)
        order = 0
        work = len(normal)
        limits._work(work)
        for row in operators:
            for operator in row:
                if operator is None:
                    continue
                if not isinstance(operator, OperatorSpec):
                    raise TypeError("system entries must be OperatorSpec or None")
                if operator.mode == "directional":
                    raise ValueError("directional operators are ambiguous in a ridge system")
                operator.check_dimension(len(normal))
                order = max(order, operator.order)
                if order > limits.max_order:
                    raise ValueError("ridge operator order exceeds the resource budget")
                if operator.mode == "laplacian":
                    assert operator.power is not None
                    work += 2 * len(normal) + operator.power
                else:
                    terms = len(operator.terms) if operator.mode == "polynomial" else 1
                    work += terms * (len(normal) + 2 * operator.order + 1)
                limits._work(work)
        limits._entries((order + 1) * equations * outputs)
        coefficients = [[[Fraction(0) for _ in range(outputs)] for _ in range(equations)]
                        for _ in range(order + 1)]
        for i, row in enumerate(operators):
            for j, operator in enumerate(row):
                if operator is None:
                    continue
                if operator.mode == "laplacian":
                    assert operator.power is not None
                    norm = Fraction(0)
                    if operator.power:
                        for value in normal:
                            norm = limits._coefficient(norm + limits._coefficient(value * value))
                    coefficients[operator.order][i][j] = _power(norm, operator.power, limits)
                    continue
                for alpha, value in operator.resolve_terms(len(normal)):
                    value = limits._coefficient(value)
                    for coordinate, exponent in zip(normal, alpha, strict=True):
                        value = limits._coefficient(value * _power(coordinate, exponent, limits))
                    degree = sum(alpha)
                    coefficients[degree][i][j] = limits._coefficient(coefficients[degree][i][j] + value)
        matrices = tuple(tuple(tuple(row) for row in matrix) for matrix in coefficients)
        return cls(normal, matrices, limits)

    @property
    def equations(self) -> int:
        return len(self.matrices[0])

    @property
    def outputs(self) -> int:
        return len(self.matrices[0][0])

    @property
    def order(self) -> int:
        """Degree after restriction/cancellation, not the original ambient PDE order."""
        return len(self.matrices) - 1

    def residual_coefficients(self, pack: RidgeSyzygy) -> tuple[Vector, ...]:
        """Exact derivative-major coefficients of ``A_w(s) Q(s)``.

        The result has ``order + pack.degree + 1`` rows, including trailing
        zeros. A zero result proves the algebraic identity for every profile.
        """
        if not isinstance(pack, RidgeSyzygy):
            raise TypeError("pack must be a RidgeSyzygy")
        if pack.direction != self.direction or pack.outputs != self.outputs:
            raise ValueError("pack direction and output count must match the ridge system")
        if pack.degree > self.budget.max_degree:
            raise ValueError("ridge pack degree exceeds the resource budget")
        self.budget._entries((self.order + pack.degree + 1) * self.equations)
        self.budget._work((self.order + 1) * (pack.degree + 1) * self.equations * self.outputs * 2)
        for row in pack.coefficients:
            for value in row:
                self.budget._coefficient(value)
        result = [[Fraction(0)] * self.equations for _ in range(self.order + pack.degree + 1)]
        for k, matrix in enumerate(self.matrices):
            for j, vector in enumerate(pack.coefficients):
                for equation, row in enumerate(matrix):
                    for value, coefficient in zip(row, vector, strict=True):
                        term = self.budget._coefficient(value * coefficient)
                        result[k + j][equation] = self.budget._coefficient(result[k + j][equation] + term)
        return tuple(tuple(row) for row in result)

    def kernel_basis(self, max_degree: int = 0) -> tuple[RidgeSyzygy, ...]:
        """Basis of polynomial right-kernel vectors of degree at most the bound.

        Shifts of a lower-degree generator can appear as independent basis
        elements. Empty means no nonzero pack at this direction and degree;
        it does not rule out particular profiles or other directions.
        """
        degree = _integer(max_degree, "max_degree")
        limits = self.budget
        if degree > limits.max_degree:
            raise ValueError("ridge kernel degree exceeds the resource budget")
        rows = self.equations * (self.order + degree + 1)
        columns = self.outputs * (degree + 1)
        limits._entries(rows * columns)
        limits._work(3 * rows * columns * min(rows, columns) + rows * columns)
        matrix = [[Fraction(0)] * columns for _ in range(rows)]
        for k, coefficient in enumerate(self.matrices):
            for j in range(degree + 1):
                for i, row in enumerate(coefficient):
                    for c, value in enumerate(row):
                        matrix[(k + j) * self.equations + i][j * self.outputs + c] = value
        solution = solve_affine(matrix, None, limits)
        basis: list[RidgeSyzygy] = []
        for vector in solution.homogeneous_basis:
            coefficients = tuple(tuple(vector[j * self.outputs:(j + 1) * self.outputs])
                                 for j in range(degree + 1))
            basis.append(RidgeSyzygy(self.direction, coefficients))
        return tuple(basis)


__all__ = ["RidgeBudget", "RidgeSystem", "RidgeSyzygy"]
