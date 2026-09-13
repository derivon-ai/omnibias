# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Complete real membership for three declared polynomial architecture families.

All classifications are global for the named homogeneous architecture only.
The returned parameters/path are replayed against an exact coefficient map.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations
from typing import Literal

from omnibias.core.realization.algebraic import AlgebraicNumber, RealAlgebraicField
from omnibias.core.realization.polynomial import (
    PolynomialNetworkSpec,
    PolynomialRealizationMap,
    Rational,
    compile_coefficient_map,
)
from omnibias.core.realization.rank import Matrix, certify_matrix_rank, determinant, matrix
from omnibias.core.realization.witness import (
    AlgebraicRealizationWitness,
    LaurentClosureWitness,
    LaurentPolynomial,
    RationalRealizationWitness,
    rational_realization_witness,
)


@dataclass(frozen=True)
class MembershipReport:
    status: Literal["realizable", "closure_only", "excluded"]
    family: str
    source: PolynomialRealizationMap
    target: tuple[Fraction, ...]
    witness: RationalRealizationWitness | AlgebraicRealizationWitness | None
    closure_path: LaurentClosureWitness | None
    detail: str


def _target(
    source: PolynomialRealizationMap,
    rows: Sequence[Sequence[Fraction]],
    indices: Sequence[tuple[int, ...]],
) -> tuple[Fraction, ...]:
    return tuple(
        dict(zip(indices, row, strict=True)).get(idx, Fraction(0))
        for row in rows
        for idx in source.input_indices
    )


def classify_linear(
    coefficients: Sequence[Sequence[Rational]], hidden_widths: Sequence[int] = ()
) -> MembershipReport:
    a = matrix(coefficients)
    dims = (len(a[0]), *hidden_widths, len(a))
    source = compile_coefficient_map(PolynomialNetworkSpec.monomial(dims, 1))
    indices = [tuple(int(i == j) for i in range(dims[0])) for j in range(dims[0])]
    target = _target(source, a, indices)
    rank = certify_matrix_rank(a)
    if rank.rank > min(dims):
        return MembershipReport(
            "excluded",
            "linear",
            source,
            target,
            None,
            None,
            "target rank exceeds the narrowest layer; the rank-bounded image is closed",
        )
    if not hidden_widths:
        theta = tuple(x for row in a for x in row)
    else:
        weights: list[Fraction] = []
        for layer, (n, m) in enumerate(zip(dims[:-1], dims[1:], strict=True)):
            if layer == 0:
                w = [
                    [rank.right[i][j] if i < rank.rank else Fraction(0) for j in range(n)]
                    for i in range(m)
                ]
            elif layer == len(dims) - 2:
                w = [
                    [rank.left[i][j] if j < rank.rank else Fraction(0) for j in range(n)]
                    for i in range(m)
                ]
            else:
                w = [[Fraction(int(i == j and i < rank.rank)) for j in range(n)] for i in range(m)]
            weights.extend(x for row in w for x in row)
        theta = tuple(weights)
    witness = rational_realization_witness(source, theta, target)
    return MembershipReport(
        "realizable",
        "linear",
        source,
        target,
        witness,
        None,
        "exact rank factorization padded through every layer",
    )


def _signed_squares(coefficients: Matrix) -> tuple[tuple[Fraction, ...], Matrix]:
    """Rational congruence decomposition A=sum_i d_i l_i l_i^T."""
    n = len(coefficients)
    a = [list(row) for row in coefficients]
    if any(len(row) != n for row in a) or any(
        a[i][j] != a[j][i] for i in range(n) for j in range(n)
    ):
        raise ValueError("symmetric square coefficient matrix required")
    transform = [[Fraction(int(i == j)) for j in range(n)] for i in range(n)]
    forms, values = [], []
    while any(v for row in a for v in row):
        pivot = next((i for i in range(n) if a[i][i]), None)
        if pivot is None:
            i, j = next((i, j) for i in range(n) for j in range(i + 1, n) if a[i][j])
            # y=S z, S=I+e_j e_i^T. The coordinate forms become S^-1 y.
            for k in range(n):
                a[k][i] += a[k][j]
            for k in range(n):
                a[i][k] += a[j][k]
            transform[j] = [x - y for x, y in zip(transform[j], transform[i], strict=True)]
            pivot = i
        d = a[pivot][pivot]
        linear = [x / d for x in a[pivot]]
        form = tuple(
            sum((linear[k] * transform[k][j] for k in range(n)), Fraction(0)) for j in range(n)
        )
        values.append(d)
        forms.append(form)
        a = [[a[i][j] - d * linear[i] * linear[j] for j in range(n)] for i in range(n)]
    return tuple(values), tuple(forms)


def classify_scalar_quadratic(
    coefficients: Sequence[Sequence[Rational]], hidden_width: int
) -> MembershipReport:
    a = matrix(coefficients)
    values, forms = _signed_squares(a)
    n = len(a)
    source = compile_coefficient_map(PolynomialNetworkSpec.monomial((n, hidden_width, 1), 2))
    coeffs = {}
    for i in range(n):
        for j in range(i, n):
            idx = tuple(int(k == i) + int(k == j) for k in range(n))
            coeffs[idx] = a[i][j] * (1 if i == j else 2)
    target = tuple(coeffs.get(idx, Fraction(0)) for idx in source.input_indices)
    if len(values) > hidden_width:
        return MembershipReport(
            "excluded",
            "scalar_quadratic",
            source,
            target,
            None,
            None,
            "symmetric rank exceeds hidden width; rank-bounded image is closed",
        )
    theta = tuple(x for form in forms for x in form) + (Fraction(0),) * (
        (hidden_width - len(forms)) * n
    )
    theta += values + (Fraction(0),) * (hidden_width - len(values))
    witness = rational_realization_witness(source, theta, target)
    return MembershipReport(
        "realizable",
        "scalar_quadratic",
        source,
        target,
        witness,
        None,
        "exact rational signed-square congruence",
    )


def _row_coordinates(
    row: Sequence[Fraction], basis: Sequence[Sequence[Fraction]]
) -> tuple[Fraction, Fraction]:
    for i, j in combinations(range(len(row)), 2):
        d = basis[0][i] * basis[1][j] - basis[0][j] * basis[1][i]
        if d:
            return (
                (row[i] * basis[1][j] - row[j] * basis[1][i]) / d,
                (basis[0][i] * row[j] - basis[0][j] * row[i]) / d,
            )
    raise ValueError("basis is rank deficient")


def classify_binary_quadratic(coefficients: Sequence[Sequence[Rational]]) -> MembershipReport:
    """Shared two-square units, any output count; rows use (x², xy, y²)."""
    c = matrix(coefficients)
    if len(c[0]) != 3:
        raise ValueError("binary quadratic coefficient rows must have three entries")
    source = compile_coefficient_map(PolynomialNetworkSpec.monomial((2, 2, len(c)), 2))
    target = _target(source, c, ((2, 0), (1, 1), (0, 2)))
    rank = certify_matrix_rank(c)
    if rank.rank > 2:
        return MembershipReport(
            "excluded",
            "binary_quadratic_width2",
            source,
            target,
            None,
            None,
            "coefficient rank exceeds two, including in the closure",
        )
    if rank.rank <= 1:
        row = next((row for row in c if any(row)), (Fraction(0),) * 3)
        d, forms = _signed_squares(((row[0], row[1] / 2), (row[1] / 2, row[2])))
        forms = forms + ((Fraction(0), Fraction(0)),) * (2 - len(forms))
        d = d + (Fraction(0),) * (2 - len(d))
        pivot = next((i for i, x in enumerate(row) if x), None)
        out = [(Fraction(0) if pivot is None else r[pivot] / row[pivot]) for r in c]
        theta = tuple(v for f in forms for v in f) + tuple(scale * v for scale in out for v in d)
        witness = rational_realization_witness(source, theta, target)
        return MembershipReport(
            "realizable",
            "binary_quadratic_width2",
            source,
            target,
            witness,
            None,
            "rank-one output span with rational signed squares",
        )
    b = [c[i] for i in rank.minor_rows]
    m12, m13, m23 = (determinant([[r[i], r[j]] for r in b]) for i, j in ((0, 1), (0, 2), (1, 2)))
    delta = m13 * m13 - m12 * m23
    if delta < 0:
        return MembershipReport(
            "excluded",
            "binary_quadratic_width2",
            source,
            target,
            None,
            None,
            "negative real square-direction discriminant; outside the Euclidean closure",
        )
    if delta == 0:
        u, v = (Fraction(1), m13 / m12) if m12 else (Fraction(0), Fraction(1))
        s, t = (Fraction(0), Fraction(1)) if u else (Fraction(1), Fraction(0))
        basis = ((u * u, 2 * u * v, v * v), (2 * u * s, 2 * (u * t + v * s), 2 * v * t))
        coords = [_row_coordinates(row, basis) for row in c]
        path: tuple[LaurentPolynomial, ...] = (
            LaurentPolynomial({0: u}),
            LaurentPolynomial({0: v}),
            LaurentPolynomial({0: u, 1: s}),
            LaurentPolynomial({0: v, 1: t}),
        )
        path += tuple(
            p
            for a, b0 in coords
            for p in (LaurentPolynomial({0: a, -1: -b0}), LaurentPolynomial({-1: b0}))
        )
        closure = LaurentClosureWitness(source.fingerprint, path, target)
        if not closure.verify(source):
            raise ArithmeticError("contact-plane Laurent witness failed")
        return MembershipReport(
            "closure_only",
            "binary_quadratic_width2",
            source,
            target,
            None,
            closure,
            "rank-two tangent plane has only one square direction; explicit colliding-square limit",
        )
    field = RealAlgebraicField.sqrt(delta)
    one, zero = field.constant(1), field.constant(0)
    if m12:
        forms_a = (
            (one, (field.constant(m13) + field.generator) / m12),
            (one, (field.constant(m13) - field.generator) / m12),
        )
    else:
        forms_a = ((zero, one), (one, field.constant(m23 / (2 * m13))))
    squares = tuple((u * u, 2 * u * v, v * v) for u, v in forms_a)
    pair = next(
        (i, j)
        for i, j in combinations(range(3), 2)
        if not (squares[0][i] * squares[1][j] - squares[0][j] * squares[1][i]).is_zero()
    )
    i, j = pair
    det = squares[0][i] * squares[1][j] - squares[0][j] * squares[1][i]
    out_a: list[AlgebraicNumber] = []
    for row in c:
        out_a.extend(
            (
                (row[i] * squares[1][j] - row[j] * squares[1][i]) / det,
                (row[j] * squares[0][i] - row[i] * squares[0][j]) / det,
            )
        )
    witness_a = AlgebraicRealizationWitness(
        source.fingerprint, field, tuple(v for f in forms_a for v in f) + tuple(out_a), target
    )
    if not witness_a.verify(source):
        raise ArithmeticError("real algebraic square witness failed")
    return MembershipReport(
        "realizable",
        "binary_quadratic_width2",
        source,
        target,
        witness_a,
        None,
        "two distinct real square directions with Sturm-isolated algebraic parameters",
    )


__all__ = [
    "MembershipReport",
    "classify_binary_quadratic",
    "classify_linear",
    "classify_scalar_quadratic",
]
