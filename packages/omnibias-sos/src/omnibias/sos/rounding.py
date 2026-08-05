# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Peyrl-Parrilo rational projection of a float Gram onto exact coefficients.

The SDP proposes a floating-point Gram ``Q_float`` with ``z^T Q_float z approx p``.
To turn that into a *proof* the Gram must match ``p``'s coefficients **exactly**
(over the rationals) while staying positive definite.  This module does the exact
part: round ``Q_float`` to a rational matrix at a chosen denominator, then project
that rational matrix orthogonally (in the trace inner product) onto the affine set
``{Q : z^T Q z = p}``.

The projection is unusually clean here because each Gram entry ``(i, j)`` feeds
**exactly one** product monomial ``alpha = basis[i] + basis[j]``, so the
coefficient-matching constraints act on **disjoint** groups of entries.  The
orthogonal projection therefore decouples per monomial ``alpha``:

    delta_alpha = (p_alpha - <A_alpha, Q>) / <A_alpha, A_alpha>

is added to every entry of group ``alpha`` (``A_alpha`` is the 0/1 symmetric
indicator of the group), a single exact rational correction per monomial.  All of
this is :class:`fractions.Fraction` arithmetic -- no floating point, so the
resulting matrix matches the coefficients to the last bit.  Whether it is still
positive definite is decided downstream by the rigorous interval ``LDL^T`` check.
"""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction

from omnibias.sos.monomials import gram_products
from omnibias.sos.problem import Exponent, Polynomial, RationalPolynomial

RationalGram = list[list[Fraction]]
#: Anything exposing ``coefficient(exponent)`` and ``support`` (float or exact).
CoefficientSource = Polynomial | RationalPolynomial


def _round_fraction(value: float, denominator: int) -> Fraction:
    """Nearest rational to ``value`` with the given (power-of-two-friendly) denominator."""
    return Fraction(int(round(float(value) * denominator)), denominator)


def project_to_exact_gram(
    gram: Sequence[Sequence[float]],
    polynomial: CoefficientSource,
    basis: Sequence[Exponent],
    *,
    denominator: int,
) -> RationalGram:
    r"""Round ``gram`` at ``denominator`` and exactly project onto ``z^T Q z = p``.

    Returns a symmetric rational Gram matrix whose expansion ``z(x)^T Q z(x)``
    equals ``polynomial`` **exactly** (verify with :func:`exact_coefficient_residual`,
    which is ``0`` by construction).  Positive-definiteness is *not* checked here.
    """
    size = len(basis)
    rational: RationalGram = [
        [_round_fraction(gram[i][j], denominator) for j in range(size)] for i in range(size)
    ]
    # Symmetrise exactly (rounding may perturb the two triangles differently).
    for i in range(size):
        for j in range(i + 1, size):
            avg = (rational[i][j] + rational[j][i]) / 2
            rational[i][j] = rational[j][i] = avg

    for alpha, pairs in gram_products(basis).items():
        target = Fraction(polynomial.coefficient(alpha))
        current = sum((mult * rational[i][j] for i, j, mult in pairs), Fraction(0))
        norm_sq = sum(1 if i == j else 2 for i, j, _mult in pairs)
        delta = (target - current) / norm_sq
        for i, j, _mult in pairs:
            rational[i][j] = rational[i][j] + delta
            if i != j:
                rational[j][i] = rational[i][j]
    return rational


def solve_rational_system(
    matrix: Sequence[Sequence[Fraction]], rhs: Sequence[Fraction]
) -> list[Fraction] | None:
    r"""Exact solution of ``matrix @ x = rhs`` over the rationals (free vars = 0).

    Gauss-Jordan elimination in :class:`~fractions.Fraction` arithmetic; returns a
    particular solution (non-pivot variables set to zero) or ``None`` when the
    system is inconsistent.  Used for the exact coefficient-matching projections in
    the Positivstellensatz and auxiliary-functional methods, where a Gram entry can
    feed several monomials (so the disjoint-group shortcut does not apply).
    """
    n_eq = len(matrix)
    n_var = len(matrix[0]) if n_eq else 0
    rows = [[Fraction(v) for v in row] + [Fraction(rhs[i])] for i, row in enumerate(matrix)]
    pivot_cols: list[int] = []
    pivot_row = 0
    for col in range(n_var):
        pivot = next((i for i in range(pivot_row, n_eq) if rows[i][col] != 0), None)
        if pivot is None:
            continue
        rows[pivot_row], rows[pivot] = rows[pivot], rows[pivot_row]
        inv = rows[pivot_row][col]
        rows[pivot_row] = [v / inv for v in rows[pivot_row]]
        for i in range(n_eq):
            if i != pivot_row and rows[i][col] != 0:
                factor = rows[i][col]
                rows[i] = [rows[i][k] - factor * rows[pivot_row][k] for k in range(n_var + 1)]
        pivot_cols.append(col)
        pivot_row += 1
        if pivot_row == n_eq:
            break
    for i in range(n_eq):
        if all(rows[i][c] == 0 for c in range(n_var)) and rows[i][n_var] != 0:
            return None
    solution = [Fraction(0)] * n_var
    for idx, col in enumerate(pivot_cols):
        solution[col] = rows[idx][n_var]
    return solution


def min_norm_correction(
    matrix: Sequence[Sequence[Fraction]], residual: Sequence[Fraction]
) -> list[Fraction] | None:
    r"""Exact ``delta`` with ``matrix @ delta = residual`` in the row space of ``matrix``.

    Solves the normal equations ``(A A^T) z = residual`` exactly and returns
    ``delta = A^T z`` (the correction lying in ``A``'s row space, hence the small
    perturbation used to project rounded Gram entries back onto the exact
    coefficient-matching set).  ``None`` if the normal equations are inconsistent.
    """
    n_eq = len(matrix)
    if n_eq == 0:
        return []
    n_var = len(matrix[0])
    gram = [
        [sum((matrix[i][k] * matrix[j][k] for k in range(n_var)), Fraction(0)) for j in range(n_eq)]
        for i in range(n_eq)
    ]
    z = solve_rational_system(gram, list(residual))
    if z is None:
        return None
    return [sum((matrix[i][c] * z[i] for i in range(n_eq)), Fraction(0)) for c in range(n_var)]


def project_coefficients_exact(
    rounded: Sequence[Fraction],
    matrix: Sequence[Sequence[Fraction]],
    target: Sequence[Fraction],
) -> list[Fraction] | None:
    r"""Project ``rounded`` onto ``{u : matrix @ u = target}`` exactly (min-norm step)."""
    residual = [
        Fraction(target[i]) - sum((matrix[i][c] * rounded[c] for c in range(len(rounded))), Fraction(0))
        for i in range(len(matrix))
    ]
    delta = min_norm_correction(matrix, residual)
    if delta is None:
        return None
    return [rounded[c] + delta[c] for c in range(len(rounded))]


def exact_coefficient_residual(
    gram: Sequence[Sequence[Fraction]],
    polynomial: CoefficientSource,
    basis: Sequence[Exponent],
) -> Fraction:
    r"""Max ``|coeff(z^T Q z) - coeff(p)|`` as an exact rational (``0`` iff matched).

    A soundness guard: certification only proceeds when this is exactly ``0``, so a
    bug in the projection can never let an unmatched Gram masquerade as a proof.
    """
    residual = Fraction(0)
    products = gram_products(basis)
    monomials = set(products) | set(polynomial.support)
    for alpha in monomials:
        pairs = products.get(alpha, [])
        value = sum((mult * gram[i][j] for i, j, mult in pairs), Fraction(0))
        diff = abs(value - Fraction(polynomial.coefficient(alpha)))
        residual = max(residual, diff)
    return residual


__all__ = [
    "RationalGram",
    "exact_coefficient_residual",
    "min_norm_correction",
    "project_coefficients_exact",
    "project_to_exact_gram",
    "solve_rational_system",
]
