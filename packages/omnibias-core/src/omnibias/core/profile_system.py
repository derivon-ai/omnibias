# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Exact PDE derivative packs for profiles with a declared annihilating ODE.

Let ``R(D_z) phi = 0`` and ``u(x) = exp(a.x) Q(D_z) phi(w.x+b)``. Then a
constant-coefficient system acts as ``exp(a.x) A(a+w*D_z) Q(D_z) phi``.
Coefficient matching in ``Q[s]/(R)`` therefore constructs homogeneous packs
or solves forcing of the same form, for EVERY profile in the kernel of R.

Repeated roots are retained. This finite quotient concerns differential
operators acting on profiles, not pointwise multiplication of functions.
Inconsistency excludes only the declared carrier, direction and profile space;
resonant solutions outside that space may exist. Profile smoothness and the
annihilator identity are assumptions, not inferred from arbitrary callables.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction

from omnibias.core._rational_linear import solve_affine
from omnibias.core.operator_spec import OperatorSpec
from omnibias.core.ridge_system import Matrix, Rational, RidgeBudget, Vector, _direction, _exact


def _monic(values: Sequence[Rational], budget: RidgeBudget | None = None) -> Vector:
    if budget is not None:
        budget._entries(len(values))
        if len(values) - 1 > budget.max_order:
            raise ValueError("profile annihilator order exceeds the resource budget")
    coefficients = tuple(_exact(value) for value in values)
    if budget is not None:
        for value in coefficients:
            budget._coefficient(value)
    while coefficients and coefficients[-1] == 0:
        coefficients = coefficients[:-1]
    if len(coefficients) < 2:
        raise ValueError("profile annihilator must be a nonconstant nonzero polynomial")
    if budget is not None and len(coefficients) - 2 > budget.max_degree:
        raise ValueError("profile quotient degree exceeds the resource budget")
    lead = coefficients[-1]
    result = tuple(value / lead for value in coefficients)
    if budget is not None:
        for value in result:
            budget._coefficient(value)
    return result


def _metadata(
    direction: Sequence[Rational], carrier: Sequence[Rational], annihilator: Sequence[Rational],
    budget: RidgeBudget | None = None,
) -> tuple[Vector, Vector, Vector]:
    if budget is not None:
        budget._entries(len(direction) + len(carrier) + len(annihilator))
        budget._work(len(direction) + len(carrier) + 2 * len(annihilator))
    normal = _direction(direction)
    shift = tuple(_exact(value) for value in carrier)
    if len(shift) != len(normal):
        raise ValueError("profile carrier must match the direction dimension")
    modulus = _monic(annihilator, budget)
    if budget is not None:
        for value in (*normal, *shift):
            budget._coefficient(value)
    return normal, shift, modulus


class _Arithmetic:
    """Cumulative rational-work meter for each public construction/solve call."""

    def __init__(self, annihilator: Vector, budget: RidgeBudget, initial_work: int = 0):
        self.annihilator = annihilator
        self.order = len(annihilator) - 1
        self.budget = budget
        self.work = initial_work
        budget._work(initial_work)

    def _entries(self, count: int) -> None:
        self.budget._entries(count)

    def _work(self, count: int) -> None:
        self.budget._work(self.work + count)

    def _coefficient(self, value: Fraction) -> Fraction:
        return self.budget._coefficient(value)

    def charge(self, count: int) -> None:
        self._work(count)
        self.work += count

    def add(self, left: Fraction, right: Fraction) -> Fraction:
        self.charge(1)
        return self._coefficient(left + right)

    def multiply(self, left: Fraction, right: Fraction) -> Fraction:
        self.charge(1)
        return self._coefficient(left * right)

    @property
    def zero(self) -> Vector:
        return (Fraction(0),) * self.order

    @property
    def one(self) -> Vector:
        return (Fraction(1),) + (Fraction(0),) * (self.order - 1)

    def reduce(self, values: Sequence[Fraction]) -> Vector:
        self._entries(max(len(values), self.order))
        self.charge(len(values) + self.order)
        remainder = [self._coefficient(v) for v in values]
        remainder.extend([Fraction(0)] * max(0, self.order - len(remainder)))
        for degree in range(len(remainder) - 1, self.order - 1, -1):
            head = remainder[degree]
            if not head:
                continue
            for j, coefficient in enumerate(self.annihilator[:-1]):
                target = degree - self.order + j
                remainder[target] = self.add(remainder[target], -self.multiply(head, coefficient))
        return tuple(remainder[:self.order])

    def plus(self, left: Vector, right: Vector) -> Vector:
        return tuple(self.add(a, b) for a, b in zip(left, right, strict=True))

    def product(self, left: Vector, right: Vector) -> Vector:
        self._entries(2 * self.order - 1)
        self.charge(self.order * self.order)
        values = [Fraction(0)] * (2 * self.order - 1)
        for i, a in enumerate(left):
            if not a:
                continue
            for j, b in enumerate(right):
                if b:
                    values[i + j] = self.add(values[i + j], self.multiply(a, b))
        return self.reduce(values)

    def power(self, base: Vector, exponent: int) -> Vector:
        result = self.one
        while exponent:
            if exponent & 1:
                result = self.product(result, base)
            exponent //= 2
            if exponent:
                base = self.product(base, base)
        return result


@dataclass(frozen=True)
class ProfilePack:
    """``exp(carrier.x) * sum(coefficients[j] * phi^(j)(direction.x+b))``.

    The scalar profile must satisfy the ascending-coefficient annihilator.
    Exactly ``profile_order`` derivative-major rows are stored, including zero
    rows. Construction validates metadata only, not membership in a PDE system.
    """

    direction: Vector
    carrier: Vector
    annihilator: Vector
    coefficients: tuple[Vector, ...]

    def __post_init__(self) -> None:
        normal, shift, modulus = _metadata(self.direction, self.carrier, self.annihilator)
        if len(self.coefficients) != len(modulus) - 1 or not self.coefficients[0]:
            raise ValueError("profile pack needs one nonempty coefficient row per profile order")
        outputs = len(self.coefficients[0])
        if any(len(row) != outputs for row in self.coefficients):
            raise ValueError("profile pack coefficients must be rectangular")
        rows = tuple(tuple(_exact(value) for value in row) for row in self.coefficients)
        object.__setattr__(self, "direction", normal)
        object.__setattr__(self, "carrier", shift)
        object.__setattr__(self, "annihilator", modulus)
        object.__setattr__(self, "coefficients", rows)

    @property
    def profile_order(self) -> int:
        return len(self.annihilator) - 1

    @property
    def outputs(self) -> int:
        return len(self.coefficients[0])


@dataclass(frozen=True)
class ProfileSolveResult:
    """Exact affine family within the declared carrier and profile quotient.

    ``consistent=False`` means that this finite ansatz cannot produce the
    forcing for every profile in ``ker R``. It is not general PDE impossibility.
    The homogeneous basis remains available even for inconsistent forcing.
    """

    consistent: bool
    particular: ProfilePack | None
    homogeneous_basis: tuple[ProfilePack, ...]
    rank: int


def _restrict(
    operator: OperatorSpec, normal: Vector, carrier: Vector, arithmetic: _Arithmetic,
) -> Vector:
    if operator.mode == "laplacian":
        assert operator.power is not None
        if not operator.power:
            return arithmetic.one
        constant = linear = quadratic = Fraction(0)
        for a, w in zip(carrier, normal, strict=True):
            constant = arithmetic.add(constant, arithmetic.multiply(a, a))
            linear = arithmetic.add(linear, arithmetic.multiply(a, w))
            quadratic = arithmetic.add(quadratic, arithmetic.multiply(w, w))
        linear = arithmetic.multiply(Fraction(2), linear)
        return arithmetic.power(arithmetic.reduce((constant, linear, quadratic)), operator.power)
    result = arithmetic.zero
    for alpha, coefficient in operator.resolve_terms(len(normal)):
        arithmetic.charge(len(alpha))
        term = arithmetic.reduce((coefficient,))
        for a, w, exponent in zip(carrier, normal, alpha, strict=True):
            if exponent:
                term = arithmetic.product(term, arithmetic.power(arithmetic.reduce((a, w)), exponent))
        result = arithmetic.plus(result, term)
    return result


@dataclass(frozen=True)
class ProfileSystem:
    """Matrix polynomial ``A(carrier + direction*s)`` modulo its annihilator.

    ``matrices[k][i][j]`` is the coefficient of ``s**k`` in equation i,
    output-field component j. There are exactly ``profile_order`` matrices.
    ``RidgeBudget.max_degree`` bounds the maximum quotient degree ``m-1``;
    ``max_order`` bounds both annihilator and ambient operator orders.
    """

    direction: Vector
    carrier: Vector
    annihilator: Vector
    matrices: tuple[Matrix, ...]
    budget: RidgeBudget = RidgeBudget()

    def __post_init__(self) -> None:
        if not isinstance(self.budget, RidgeBudget):
            raise TypeError("budget must be a RidgeBudget")
        normal, shift, modulus = _metadata(self.direction, self.carrier, self.annihilator, self.budget)
        if len(self.matrices) != len(modulus) - 1 or not self.matrices[0] or not self.matrices[0][0]:
            raise ValueError("profile system needs one nonempty matrix per profile order")
        equations, outputs = len(self.matrices[0]), len(self.matrices[0][0])
        self.budget._entries(len(self.matrices) * equations * outputs)
        matrices: list[Matrix] = []
        for matrix in self.matrices:
            if len(matrix) != equations or any(len(row) != outputs for row in matrix):
                raise ValueError("profile matrices must share a rectangular shape")
            matrices.append(tuple(tuple(self.budget._coefficient(_exact(v)) for v in row) for row in matrix))
        object.__setattr__(self, "direction", normal)
        object.__setattr__(self, "carrier", shift)
        object.__setattr__(self, "annihilator", modulus)
        object.__setattr__(self, "matrices", tuple(matrices))

    @classmethod
    def from_operators(
        cls, operators: Sequence[Sequence[OperatorSpec | None]], direction: Sequence[Rational], *,
        annihilator: Sequence[Rational], carrier: Sequence[Rational] | None = None,
        budget: RidgeBudget | None = None,
    ) -> ProfileSystem:
        """Build a bounded exact quotient; ``None`` entries are zero operators.

        Laplacian powers remain compact through the quadratic shifted symbol.
        Runtime directional requests are ambiguous and therefore refused.
        """
        limits = RidgeBudget() if budget is None else budget
        if not isinstance(limits, RidgeBudget):
            raise TypeError("budget must be a RidgeBudget")
        limits._entries(len(direction))
        shift_input = (Fraction(0),) * len(direction) if carrier is None else carrier
        normal, shift, modulus = _metadata(direction, shift_input, annihilator, limits)
        if not operators or not operators[0] or any(len(row) != len(operators[0]) for row in operators):
            raise ValueError("operators must form a nonempty rectangular matrix")
        equations, outputs, order = len(operators), len(operators[0]), len(modulus) - 1
        limits._entries(order * equations * outputs)
        arithmetic = _Arithmetic(modulus, limits, len(normal) + len(shift) + 2 * len(modulus))
        arithmetic.charge(equations * outputs)
        for row in operators:
            for operator in row:
                if operator is None:
                    continue
                if not isinstance(operator, OperatorSpec):
                    raise TypeError("system entries must be OperatorSpec or None")
                if operator.mode == "directional":
                    raise ValueError("directional operators are ambiguous in a profile system")
                operator.check_dimension(len(normal))
                if operator.order > limits.max_order:
                    raise ValueError("profile operator order exceeds the resource budget")
        matrices = [[[Fraction(0)] * outputs for _ in range(equations)] for _ in range(order)]
        for i, row in enumerate(operators):
            for j, operator in enumerate(row):
                if operator is not None:
                    coefficients = _restrict(operator, normal, shift, arithmetic)
                    for k, value in enumerate(coefficients):
                        matrices[k][i][j] = value
        return cls(normal, shift, modulus, tuple(tuple(tuple(row) for row in matrix) for matrix in matrices), limits)

    @property
    def profile_order(self) -> int:
        return len(self.annihilator) - 1

    @property
    def equations(self) -> int:
        return len(self.matrices[0])

    @property
    def outputs(self) -> int:
        return len(self.matrices[0][0])

    def _arithmetic(self) -> _Arithmetic:
        return _Arithmetic(self.annihilator, self.budget)

    def _forcing(self, forcing: Sequence[Sequence[Rational]] | None, arithmetic: _Arithmetic) -> tuple[Vector, ...]:
        order = self.profile_order
        if forcing is None:
            return ((Fraction(0),) * self.equations,) * order
        if len(forcing) - 1 > self.budget.max_order:
            raise ValueError("profile forcing order exceeds the resource budget")
        arithmetic._entries(len(forcing) * self.equations)
        if any(len(row) != self.equations for row in forcing):
            raise ValueError("forcing rows must match the equation count")
        columns = [arithmetic.reduce(tuple(_exact(row[i]) for row in forcing)) for i in range(self.equations)]
        return tuple(tuple(column[k] for column in columns) for k in range(order))

    def _linear_matrix(self, arithmetic: _Arithmetic) -> tuple[Vector, ...]:
        order = self.profile_order
        rows, columns = order * self.equations, order * self.outputs
        arithmetic._entries(rows * columns)
        arithmetic.charge(rows * columns)
        result = [[Fraction(0)] * columns for _ in range(rows)]
        for i in range(self.equations):
            for c in range(self.outputs):
                polynomial = tuple(matrix[i][c] for matrix in self.matrices)
                for j in range(order):
                    values = arithmetic.reduce((Fraction(0),) * j + polynomial)
                    for k, value in enumerate(values):
                        result[k * self.equations + i][j * self.outputs + c] = value
        return tuple(tuple(row) for row in result)

    def _pack(self, vector: Vector) -> ProfilePack:
        rows = tuple(tuple(vector[k * self.outputs:(k + 1) * self.outputs]) for k in range(self.profile_order))
        return ProfilePack(self.direction, self.carrier, self.annihilator, rows)

    def solve(self, forcing: Sequence[Sequence[Rational]] | None = None) -> ProfileSolveResult:
        """Solve ``A Q = forcing mod R`` as a finite rational affine system.

        Forcing is derivative-major and represents
        ``exp(carrier.x) sum(forcing[j] phi^(j)(direction.x+b))``. Terms of
        degree at least ``profile_order`` are reduced exactly before solving.
        Inconsistent forcing still returns the homogeneous kernel basis.
        """
        arithmetic = self._arithmetic()
        target = self._forcing(forcing, arithmetic)
        matrix = self._linear_matrix(arithmetic)
        rhs = None if forcing is None else tuple(value for row in target for value in row)
        solution = solve_affine(matrix, rhs, arithmetic)
        count = len(solution.homogeneous_basis) + int(solution.consistent)
        pack_size = (2 * len(self.direction) + len(self.annihilator)
                     + self.profile_order * self.outputs)
        self.budget._entries(count * pack_size)
        particular = self._pack(solution.particular) if solution.particular is not None else None
        basis = tuple(self._pack(vector) for vector in solution.homogeneous_basis)
        return ProfileSolveResult(solution.consistent, particular, basis, solution.rank)

    def residual_coefficients(self, pack: ProfilePack) -> tuple[Vector, ...]:
        """Canonical ``profile_order × equations`` remainder of ``A Q mod R``.

        This is the operator output; it does not subtract a forcing target.
        It proves an identity for the declared profile space only.
        """
        if not isinstance(pack, ProfilePack):
            raise TypeError("pack must be a ProfilePack")
        if (pack.direction != self.direction or pack.carrier != self.carrier
                or pack.annihilator != self.annihilator or pack.outputs != self.outputs):
            raise ValueError("pack metadata and output count must match the profile system")
        arithmetic = self._arithmetic()
        arithmetic._entries(self.profile_order * (self.equations + self.outputs))
        for row in pack.coefficients:
            for value in row:
                arithmetic._coefficient(value)
        result = [arithmetic.zero for _ in range(self.equations)]
        for i in range(self.equations):
            for c in range(self.outputs):
                operator = tuple(matrix[i][c] for matrix in self.matrices)
                profile = tuple(row[c] for row in pack.coefficients)
                result[i] = arithmetic.plus(result[i], arithmetic.product(operator, profile))
        return tuple(tuple(row[k] for row in result) for k in range(self.profile_order))


__all__ = ["ProfilePack", "ProfileSolveResult", "ProfileSystem"]
