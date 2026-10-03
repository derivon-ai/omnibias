# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Exact Picard--Fuchs certificates for the depressed-cubic elliptic family.

For

.. math::

   H(x,y)=y^2+x^3+p x+q,\qquad e=h-q,

the periods ``J0 = integral dx/y`` and ``J1 = integral x dx/y`` satisfy an
exact two-dimensional Gauss--Manin system.  This module derives the scalar
annihilator of ``J0`` from that system and accepts it only when an exact
``Q[h]`` determining matrix has a non-trivial kernel and the proposed
operator lies in that kernel.

The exact algebra certifies the Picard--Fuchs identity.  It does not certify
that a requested contour is a physical oval, that analytic continuation
avoids every singular value, or that an Abelian integral has a stated zero
count.  Those are separate consumers in :mod:`omnibias.dynamics.abelian`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from math import gcd

from omnibias.core.proof.certificate import (
    Cert,
    make_certificate,
    verify_certificate_digest,
)
from omnibias.holonomic._core.ore import OrePolynomial, diff_algebra
from omnibias.holonomic._core.oreops import symmetric_product
from omnibias.holonomic._core.ratfunc import (
    RatFunc,
    rf_add,
    rf_div,
    rf_from_poly,
    rf_from_rational,
    rf_is_zero,
    rf_mul,
    rf_normalize,
    rf_sub,
)
from omnibias.holonomic._core.rational_poly import (
    Poly,
    padd,
    pderiv,
    pdivmod,
    pgcd,
    pmul,
    pscale,
    to_poly,
)
from omnibias.holonomic.rank_syzygy import (
    certify_holonomic_syzygy,
    integerize_matrix,
)

Rational = Fraction | int


def _as_fraction(value: Rational) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, int | Fraction):
        raise TypeError("Picard--Fuchs coefficients must be exact rationals")
    return Fraction(value)


def _rf_derivative(value: RatFunc) -> RatFunc:
    """Exact derivative of a rational function in ``Q(h)``."""

    num, den = value
    return rf_normalize(
        padd(pmul(pderiv(num), den), pscale(pmul(num, pderiv(den)), -1)),
        pmul(den, den),
    )


def _rf_sum(values: Sequence[RatFunc]) -> RatFunc:
    result = rf_from_rational(0)
    for value in values:
        result = rf_add(result, value)
    return result


def _poly_power_x(power: int) -> Poly:
    return (Fraction(0),) * power + (Fraction(1),)


def _gauss_manin_rows(p: Fraction, q: Fraction) -> tuple[tuple[RatFunc, RatFunc], ...]:
    """Rows representing ``J0``, ``J0'`` and ``J0''`` in the ``(J0,J1)`` basis."""

    e = (-q, Fraction(1))
    discriminant = padd(pscale(pmul(e, e), 27), (4 * p**3,))
    two_delta = pscale(discriminant, 2)
    matrix = (
        (
            rf_normalize(pscale(e, -9), two_delta),
            rf_normalize((6 * p,), two_delta),
        ),
        (
            rf_normalize((2 * p**2,), two_delta),
            rf_normalize(pscale(e, 9), two_delta),
        ),
    )
    rows: list[tuple[RatFunc, RatFunc]] = [
        (rf_from_rational(1), rf_from_rational(0))
    ]
    for _ in range(2):
        left, right = rows[-1]
        rows.append(
            (
                rf_add(
                    _rf_derivative(left),
                    rf_add(rf_mul(left, matrix[0][0]), rf_mul(right, matrix[1][0])),
                ),
                rf_add(
                    _rf_derivative(right),
                    rf_add(rf_mul(left, matrix[0][1]), rf_mul(right, matrix[1][1])),
                ),
            )
        )
    return tuple(rows)


def _solve_ratfunc_system(
    matrix: Sequence[Sequence[RatFunc]],
    rhs: Sequence[RatFunc],
) -> tuple[RatFunc, ...]:
    """Exact Gaussian elimination over ``Q(h)``."""

    size = len(matrix)
    if size == 0 or len(rhs) != size or any(len(row) != size for row in matrix):
        raise ValueError("a square rational-function system is required")
    augmented = [list(row) + [rhs[index]] for index, row in enumerate(matrix)]
    for column in range(size):
        pivot = next(
            (
                row
                for row in range(column, size)
                if not rf_is_zero(augmented[row][column])
            ),
            None,
        )
        if pivot is None:
            raise ArithmeticError("hyperelliptic de Rham reduction is singular")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        pivot_value = augmented[column][column]
        augmented[column] = [
            rf_div(value, pivot_value) for value in augmented[column]
        ]
        for row in range(size):
            if row == column or rf_is_zero(augmented[row][column]):
                continue
            factor = augmented[row][column]
            augmented[row] = [
                rf_sub(value, rf_mul(factor, pivot_entry))
                for value, pivot_entry in zip(
                    augmented[row],
                    augmented[column],
                    strict=True,
                )
            ]
    return tuple(augmented[row][-1] for row in range(size))


def _hyperelliptic_gauss_manin(potential: Poly) -> tuple[tuple[RatFunc, ...], ...]:
    r"""Derive ``J' = G(h)J`` from ``d(P/y)`` exactness for ``y^2=h-V(x)``."""

    degree = len(potential) - 1
    basis_size = degree - 1
    equation_size = 2 * degree - 1
    zero = rf_from_rational(0)
    f_coeffs: list[Poly] = []
    for power in range(degree + 1):
        coefficient = potential[power] if power < len(potential) else Fraction(0)
        f_coeffs.append(
            (-coefficient, Fraction(1))
            if power == 0
            else ((-coefficient,) if coefficient else ())
        )

    columns: list[list[RatFunc]] = []
    # J_j = integral x^j (h-V) dx/y^3.
    for basis_power in range(basis_size):
        column = [zero for _ in range(equation_size)]
        for power, coefficient in enumerate(f_coeffs):
            target = power + basis_power
            if target < equation_size and coefficient:
                column[target] = rf_from_poly(coefficient)
        columns.append(column)

    # 2 P'(h-V) + P V' is the numerator of the exact form 2 d(P/y).
    for p_power in range(degree):
        column = [zero for _ in range(equation_size)]
        if p_power:
            for power, coefficient in enumerate(f_coeffs):
                target = power + p_power - 1
                if target < equation_size and coefficient:
                    column[target] = rf_add(
                        column[target],
                        rf_mul(
                            rf_from_rational(2 * p_power),
                            rf_from_poly(coefficient),
                        ),
                    )
        for v_power in range(1, len(potential)):
            target = p_power + v_power - 1
            if target < equation_size and potential[v_power]:
                column[target] = rf_add(
                    column[target],
                    rf_from_rational(v_power * potential[v_power]),
                )
        columns.append(column)

    coefficient_matrix = tuple(
        tuple(columns[column][row] for column in range(equation_size))
        for row in range(equation_size)
    )
    gauss_manin: list[tuple[RatFunc, ...]] = []
    for moment_power in range(basis_size):
        rhs = tuple(
            rf_from_rational(1 if row == moment_power else 0)
            for row in range(equation_size)
        )
        solution = _solve_ratfunc_system(coefficient_matrix, rhs)
        # J_k' = -1/2 K_k.
        gauss_manin.append(
            tuple(
                rf_mul(rf_from_rational(Fraction(-1, 2)), solution[index])
                for index in range(basis_size)
            )
        )
    return tuple(gauss_manin)


def _derivative_rows(
    gauss_manin: Sequence[Sequence[RatFunc]],
) -> tuple[tuple[RatFunc, ...], ...]:
    """Represent successive derivatives of ``J0`` in the moment basis."""

    size = len(gauss_manin)
    rows: list[tuple[RatFunc, ...]] = [
        tuple(rf_from_rational(1 if index == 0 else 0) for index in range(size))
    ]
    for _ in range(size):
        previous = rows[-1]
        rows.append(
            tuple(
                rf_add(
                    _rf_derivative(previous[column]),
                    _rf_sum(
                        tuple(
                            rf_mul(previous[row], gauss_manin[row][column])
                            for row in range(size)
                        )
                    ),
                )
                for column in range(size)
            )
        )
    return tuple(rows)


def _relation_matrix(
    derivatives: Sequence[Sequence[RatFunc]],
    degree_bound: int,
) -> tuple[tuple[Fraction, ...], ...]:
    """Exact coefficient matrix for polynomial relations among derivative rows."""

    columns: list[tuple[RatFunc, ...]] = []
    for derivative in derivatives:
        for degree in range(degree_bound + 1):
            factor = rf_from_poly(_poly_power_x(degree))
            columns.append(tuple(rf_mul(factor, value) for value in derivative))

    denominators = [
        value[1] for column in columns for value in column if value[0]
    ]
    common: Poly = (Fraction(1),)
    for denominator in denominators:
        divisor = pgcd(common, denominator)
        common = pdivmod(pmul(common, denominator), divisor)[0]

    cleared: list[tuple[Poly, ...]] = []
    max_degree = 0
    for column in columns:
        parts: list[Poly] = []
        for numerator, denominator in column:
            quotient, remainder = pdivmod(common, denominator)
            if remainder:  # pragma: no cover - exact polynomial LCM invariant
                raise ArithmeticError("failed to clear a rational-function denominator")
            value = pmul(numerator, quotient)
            parts.append(value)
            max_degree = max(max_degree, len(value) - 1)
        cleared.append(tuple(parts))

    rows: list[tuple[Fraction, ...]] = []
    for component in range(len(derivatives[0])):
        for degree in range(max_degree + 1):
            row = tuple(
                column[component][degree]
                if degree < len(column[component])
                else Fraction(0)
                for column in cleared
            )
            if any(row):
                rows.append(row)
    return tuple(rows)


def _determining_matrix(p: Fraction, q: Fraction) -> tuple[tuple[Fraction, ...], ...]:
    """Coefficient matrix for ``sum c_ij h^j D^i J0 = 0``, ``i,j <= 2``."""

    return _relation_matrix(_gauss_manin_rows(p, q), 2)


def _period_operator(p: Fraction, q: Fraction) -> OrePolynomial:
    e = (-q, Fraction(1))
    discriminant = padd(pscale(pmul(e, e), 27), (4 * p**3,))
    return diff_algebra().operator(
        (
            (15,),
            pscale(e, 216),
            pscale(discriminant, 4),
        )
    )


def _area_operator(period: OrePolynomial) -> OrePolynomial:
    return period.algebra.operator(((), *period.coeffs))


def _factor_operator(factor: Poly) -> OrePolynomial:
    return diff_algebra().operator((pscale(pderiv(factor), -1), factor))


def _operator_vector(operator: OrePolynomial) -> tuple[Fraction, ...] | None:
    """Flatten an order-two, degree-two operator into the determining basis."""

    if operator.algebra.name != "differential" or operator.order > 2:
        return None
    values: list[Fraction] = []
    for order in range(3):
        polynomial = operator.coeffs[order] if order < len(operator.coeffs) else ()
        if len(polynomial) > 3:
            return None
        values.extend(
            polynomial[degree] if degree < len(polynomial) else Fraction(0)
            for degree in range(3)
        )
    return tuple(values)


def _primitive_integer_vector(values: Sequence[Fraction]) -> tuple[int, ...]:
    denominator = 1
    for value in values:
        denominator = denominator * value.denominator // gcd(denominator, value.denominator)
    integers = [int(value * denominator) for value in values]
    divisor = 0
    for value in integers:
        divisor = gcd(divisor, abs(value))
    if divisor:
        integers = [value // divisor for value in integers]
    for value in reversed(integers):
        if value:
            if value < 0:
                integers = [-entry for entry in integers]
            break
    return tuple(integers)


def _matrix_residuals(
    matrix: Sequence[Sequence[int]], vector: Sequence[int]
) -> tuple[int, ...]:
    return tuple(sum(entry * value for entry, value in zip(row, vector, strict=True)) for row in matrix)


@dataclass(frozen=True)
class PicardFuchsCertificate:
    """Exact depressed-cubic Picard--Fuchs relation and its syzygy witness."""

    p: Fraction
    q: Fraction
    energy_factor: Poly
    period_operator: OrePolynomial
    area_operator: OrePolynomial
    integral_operator: OrePolynomial
    determining_matrix: tuple[tuple[int, ...], ...]
    candidate_vector: tuple[int, ...]
    rank_kernel: tuple[tuple[int, ...], ...]
    verified: bool
    detail: str
    seal: Cert | None

    @property
    def critical_polynomial(self) -> Poly:
        """``27 (h-q)^2 + 4 p^3``; its zeros are the singular energies."""

        e = (-self.q, Fraction(1))
        return padd(pscale(pmul(e, e), 27), (4 * self.p**3,))


def certify_picard_fuchs(
    p: Rational,
    q: Rational = 0,
    *,
    energy_factor: Sequence[Rational] = (1,),
    candidate: OrePolynomial | None = None,
) -> PicardFuchsCertificate:
    """Certify the Picard--Fuchs operator for ``y^2+x^3+p*x+q``.

    ``candidate`` is the proposed scalar operator on ``J0 = integral dx/y``.
    When absent, the exact closed-form operator is proposed.  Acceptance
    requires both an exact non-trivial determining-matrix kernel and zero
    residual for this particular candidate; finding some unrelated kernel is
    insufficient.
    """

    pp, qq = _as_fraction(p), _as_fraction(q)
    if pp == 0:
        raise ValueError("the depressed cubic requires p != 0")
    factor = to_poly(tuple(_as_fraction(value) for value in energy_factor))
    if not factor:
        raise ValueError("energy_factor must be a nonzero polynomial")

    proposed = _period_operator(pp, qq) if candidate is None else candidate
    rational_matrix = _determining_matrix(pp, qq)
    int_matrix = tuple(tuple(row) for row in integerize_matrix(rational_matrix))
    rank = certify_holonomic_syzygy(rational_matrix)
    flat = _operator_vector(proposed)
    vector = () if flat is None else _primitive_integer_vector(flat)
    residuals = () if not vector else _matrix_residuals(int_matrix, vector)
    verified = bool(rank.proved and vector and all(value == 0 for value in residuals))

    canonical_period = _period_operator(pp, qq)
    area = _area_operator(proposed)
    integral = (
        area
        if factor == (Fraction(1),)
        else symmetric_product(area, _factor_operator(factor))
    )
    if verified:
        seal = make_certificate(
            claim=(
                "Exact Q[h] Picard-Fuchs syzygy for the declared depressed-cubic "
                "period; no contour or Hilbert-number claim."
            ),
            payload={
                "type": "integer_matrix_syzygy",
                "matrix": [list(row) for row in int_matrix],
                "vector": list(vector),
                "picard_fuchs": {
                    "p": [pp.numerator, pp.denominator],
                    "q": [qq.numerator, qq.denominator],
                },
            },
            honesty={
                "exact_picard_fuchs_identity": True,
                "analytic_continuation_certified": False,
                "zero_count_claim": False,
                "full_hilbert16_solved": False,
            },
            meta={"transcend_backend": "not_used"},
        )
        detail = "candidate is an exact Q[h] Picard-Fuchs syzygy"
    else:
        seal = None
        detail = "candidate failed the exact Q[h] determining-matrix check"

    # This is a useful invariant when a caller supplied a candidate: the
    # canonical operator must itself always pass the same exact matrix.
    canonical_vector = _primitive_integer_vector(_operator_vector(canonical_period) or ())
    if not canonical_vector or any(_matrix_residuals(int_matrix, canonical_vector)):
        raise ArithmeticError("internal Picard-Fuchs derivation failed its exact replay")

    return PicardFuchsCertificate(
        p=pp,
        q=qq,
        energy_factor=factor,
        period_operator=proposed,
        area_operator=area,
        integral_operator=integral,
        determining_matrix=int_matrix,
        candidate_vector=vector,
        rank_kernel=rank.kernel,
        verified=verified,
        detail=detail,
        seal=seal,
    )


def verify_picard_fuchs(certificate: PicardFuchsCertificate) -> bool:
    """Replay the determining matrix and certificate seal from exact sources."""

    if not certificate.verified or certificate.seal is None:
        return False
    try:
        if not verify_certificate_digest(certificate.seal):
            return False
        replay = certify_picard_fuchs(
            certificate.p,
            certificate.q,
            energy_factor=certificate.energy_factor,
            candidate=certificate.period_operator,
        )
        return replay == certificate
    except (ArithmeticError, TypeError, ValueError, ZeroDivisionError):
        return False


@dataclass(frozen=True)
class HyperellipticPicardFuchsCertificate:
    """Exact de Rham reduction and scalar period syzygy for ``y^2=h-V(x)``."""

    potential: Poly
    energy_factor: Poly
    basis_dimension: int
    coefficient_degree_bound: int
    period_operator: OrePolynomial
    area_operator: OrePolynomial
    integral_operator: OrePolynomial
    determining_matrix: tuple[tuple[int, ...], ...]
    candidate_vector: tuple[int, ...]
    rank_kernel: tuple[tuple[int, ...], ...]
    verified: bool
    detail: str
    seal: Cert | None


def _operator_from_relation(
    vector: Sequence[int],
    derivative_count: int,
    degree_bound: int,
) -> OrePolynomial:
    width = degree_bound + 1
    if len(vector) != derivative_count * width:
        raise ValueError("relation vector has the wrong dimension")
    return diff_algebra().operator(
        tuple(
            tuple(Fraction(value) for value in vector[order * width : (order + 1) * width])
            for order in range(derivative_count)
        )
    )


def _discover_hyperelliptic_relation(
    potential: Poly,
    max_degree: int,
) -> tuple[
    OrePolynomial,
    tuple[tuple[int, ...], ...],
    tuple[int, ...],
    tuple[tuple[int, ...], ...],
    int,
]:
    gauss_manin = _hyperelliptic_gauss_manin(potential)
    derivatives = _derivative_rows(gauss_manin)
    for degree_bound in range(max_degree + 1):
        rational_matrix = _relation_matrix(derivatives, degree_bound)
        rank = certify_holonomic_syzygy(rational_matrix)
        if not rank.proved:
            continue
        width = degree_bound + 1
        candidates = [
            tuple(vector)
            for vector in rank.kernel
            if any(vector[(len(derivatives) - 1) * width :])
        ]
        if not candidates:
            candidates = [tuple(vector) for vector in rank.kernel if any(vector)]
        if not candidates:
            continue
        vector = min(candidates)
        int_matrix = tuple(tuple(row) for row in integerize_matrix(rational_matrix))
        if any(_matrix_residuals(int_matrix, vector)):
            raise ArithmeticError("rank-collapse kernel failed exact replay")
        operator = _operator_from_relation(
            vector,
            len(derivatives),
            degree_bound,
        )
        return operator, int_matrix, vector, rank.kernel, degree_bound
    raise ArithmeticError(
        f"no scalar Picard-Fuchs relation found through coefficient degree {max_degree}"
    )


def certify_hyperelliptic_picard_fuchs(
    potential: Sequence[Rational],
    *,
    energy_factor: Sequence[Rational] = (1,),
    max_degree: int | None = None,
) -> HyperellipticPicardFuchsCertificate:
    """Derive and certify a scalar Picard--Fuchs operator for ``y^2=h-V(x)``.

    The moment basis is ``J_k = integral x^k dx/y`` for
    ``0 <= k <= deg(V)-2``.  Exact forms ``d(P/y)`` reduce every ``J_k'`` to
    that basis over ``Q(h)``.  Successive derivatives of ``J0`` then form an
    exact determining matrix whose integerized kernel supplies the scalar
    operator.
    """

    polynomial = to_poly(tuple(_as_fraction(value) for value in potential))
    if len(polynomial) < 3:
        raise ValueError("a hyperelliptic potential must have degree >= 2")
    factor = to_poly(tuple(_as_fraction(value) for value in energy_factor))
    if not factor:
        raise ValueError("energy_factor must be a nonzero polynomial")
    degree = len(polynomial) - 1
    degree_limit = 4 * degree if max_degree is None else max_degree
    if type(degree_limit) is not int or degree_limit < 0:
        raise ValueError("max_degree must be a nonnegative integer")

    period, matrix, vector, kernel, used_degree = _discover_hyperelliptic_relation(
        polynomial,
        degree_limit,
    )
    area = _area_operator(period)
    integral = (
        area
        if factor == (Fraction(1),)
        else symmetric_product(area, _factor_operator(factor))
    )
    seal = make_certificate(
        claim=(
            "Exact cleared-denominator Q[h] Picard-Fuchs syzygy from "
            "hyperelliptic de Rham reduction; no contour or zero-count claim."
        ),
        payload={
            "type": "integer_matrix_syzygy",
            "matrix": [list(row) for row in matrix],
            "vector": list(vector),
            "hyperelliptic": {
                "potential": [
                    [value.numerator, value.denominator] for value in polynomial
                ],
                "basis_dimension": degree - 1,
                "coefficient_degree_bound": used_degree,
            },
        },
        honesty={
            "exact_picard_fuchs_identity": True,
            "analytic_continuation_certified": False,
            "zero_count_claim": False,
            "full_hilbert16_solved": False,
        },
        meta={"transcend_backend": "not_used"},
    )
    return HyperellipticPicardFuchsCertificate(
        potential=polynomial,
        energy_factor=factor,
        basis_dimension=degree - 1,
        coefficient_degree_bound=used_degree,
        period_operator=period,
        area_operator=area,
        integral_operator=integral,
        determining_matrix=matrix,
        candidate_vector=vector,
        rank_kernel=kernel,
        verified=True,
        detail="exact hyperelliptic de Rham reduction and Q[h] rank syzygy",
        seal=seal,
    )


def verify_hyperelliptic_picard_fuchs(
    certificate: HyperellipticPicardFuchsCertificate,
) -> bool:
    """Replay a general hyperelliptic Picard--Fuchs certificate."""

    if not certificate.verified or certificate.seal is None:
        return False
    try:
        if not verify_certificate_digest(certificate.seal):
            return False
        replay = certify_hyperelliptic_picard_fuchs(
            certificate.potential,
            energy_factor=certificate.energy_factor,
            max_degree=certificate.coefficient_degree_bound,
        )
        return (
            replay.potential == certificate.potential
            and replay.energy_factor == certificate.energy_factor
            and replay.basis_dimension == certificate.basis_dimension
            and replay.coefficient_degree_bound
            == certificate.coefficient_degree_bound
            and replay.period_operator.coeffs == certificate.period_operator.coeffs
            and replay.area_operator.coeffs == certificate.area_operator.coeffs
            and replay.integral_operator.coeffs == certificate.integral_operator.coeffs
            and replay.determining_matrix == certificate.determining_matrix
            and replay.candidate_vector == certificate.candidate_vector
            and replay.rank_kernel == certificate.rank_kernel
            and replay.seal == certificate.seal
        )
    except (ArithmeticError, TypeError, ValueError, ZeroDivisionError):
        return False


__all__ = [
    "HyperellipticPicardFuchsCertificate",
    "PicardFuchsCertificate",
    "certify_hyperelliptic_picard_fuchs",
    "certify_picard_fuchs",
    "verify_hyperelliptic_picard_fuchs",
    "verify_picard_fuchs",
]
