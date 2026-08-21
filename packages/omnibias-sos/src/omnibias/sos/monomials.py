# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Monomial bases and the Gram <-> coefficient linear maps.

An SOS decomposition writes ``p(x) = z(x)^T Q z(x)`` where ``z(x)`` is the vector
of monomials up to half the degree of ``p``.  Multiplying two basis monomials
``z_i z_j`` yields the monomial with exponent ``basis[i] + basis[j]``, so the
coefficient of a monomial ``alpha`` in ``z^T Q z`` is

    sum over pairs (i, j) with basis[i] + basis[j] == alpha of  mult * Q[i, j]

with ``mult = 1`` on the diagonal and ``2`` off it (``Q`` symmetric).  This module
builds the basis and that pair table -- the linear "coefficient-matching"
constraints ``<A_alpha, Q> = p_alpha`` an SOS SDP must satisfy.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from omnibias.sos.problem import Exponent, Polynomial


def _compositions(n_vars: int, total: int) -> Iterator[Exponent]:
    """Yield every length-``n_vars`` exponent tuple summing to ``total``."""
    if n_vars <= 0:
        if total == 0:
            yield ()
        return
    if n_vars == 1:
        yield (total,)
        return
    for first in range(total, -1, -1):
        for rest in _compositions(n_vars - 1, total - first):
            yield (first, *rest)


def monomial_basis(n_vars: int, degree: int) -> tuple[Exponent, ...]:
    """All monomials in ``n_vars`` variables of total degree ``0..degree``.

    Returned in graded order (degree ascending); the constant monomial is first.
    """
    if n_vars < 0 or degree < 0:
        raise ValueError("n_vars and degree must be non-negative")
    out: list[Exponent] = []
    for total in range(degree + 1):
        out.extend(_compositions(n_vars, total))
    return tuple(out)


def arrangement_adapted_basis(
    polynomial: Polynomial,
    arrangement: Sequence[Sequence[float]],
    *,
    degree: int,
) -> MonomialBasis:
    """Sparse monomial basis concentrated on an axis-aligned arrangement.

    Ambient total-degree monomials up to ``degree`` are kept. Each
    hyperplane ``a·x + b = 0`` contributes extra pure powers of its
    dominant coordinate up to half the polynomial's degree in that
    variable, so a well that is a high even power of one arrangement
    form can be certified at a *lower* ambient degree than the
    total-degree basis.

    This chooses tightness, never soundness. Failures are reported
    (theory 07-05 G5), not excluded.
    """
    if degree < 0:
        raise ValueError(f"degree must be >= 0, got {degree}")
    n = polynomial.n_vars
    exponents = set(monomial_basis(n, degree))
    max_power = [0] * n
    for exp in polynomial.support:
        for i, power in enumerate(exp):
            if power > max_power[i]:
                max_power[i] = int(power)
    for form in arrangement:
        coords = [float(c) for c in form]
        if len(coords) < n:
            raise ValueError("each arrangement form must have at least n_vars coefficients")
        # Dominant coordinate of a·x (+ optional offset).
        axis = max(range(n), key=lambda i: abs(coords[i]))
        half = (max_power[axis] + 1) // 2
        for power in range(degree + 1, half + 1):
            exp = [0] * n
            exp[axis] = power
            exponents.add(tuple(exp))
    ordered = tuple(sorted(exponents, key=lambda e: (sum(e), e)))
    return MonomialBasis(n, ordered)


@dataclass(frozen=True)
class MonomialBasis:
    """The monomial vector ``z(x)`` used to build a Gram matrix."""

    n_vars: int
    exponents: tuple[Exponent, ...]

    @classmethod
    def up_to_degree(cls, n_vars: int, degree: int) -> MonomialBasis:
        return cls(n_vars, monomial_basis(n_vars, degree))

    @property
    def size(self) -> int:
        return len(self.exponents)

    def evaluate(self, point: Sequence[float]) -> list[float]:
        """The numeric vector ``z(point)``."""
        out: list[float] = []
        for exp in self.exponents:
            term = 1.0
            for value, power in zip(point, exp, strict=True):
                if power:
                    term *= float(value) ** power
            out.append(term)
        return out


def gram_products(
    basis: Sequence[Exponent],
) -> dict[Exponent, list[tuple[int, int, int]]]:
    r"""Map each product monomial ``alpha`` to the ``(i, j, mult)`` that build it.

    ``mult`` is ``1`` when ``i == j`` and ``2`` otherwise, so that
    ``sum mult * Q[i, j]`` over the list equals the coefficient of ``alpha`` in
    ``z(x)^T Q z(x)`` for a symmetric ``Q`` (only ``i <= j`` are listed).
    """
    products: dict[Exponent, list[tuple[int, int, int]]] = {}
    size = len(basis)
    for i in range(size):
        for j in range(i, size):
            alpha = tuple(a + b for a, b in zip(basis[i], basis[j], strict=True))
            mult = 1 if i == j else 2
            products.setdefault(alpha, []).append((i, j, mult))
    return products


def gram_to_poly(
    gram: Sequence[Sequence[float]], basis: Sequence[Exponent], n_vars: int
) -> Polynomial:
    r"""Expand ``z(x)^T Q z(x)`` back into a :class:`Polynomial`."""
    coeffs: dict[Exponent, float] = {}
    for alpha, pairs in gram_products(basis).items():
        acc = 0.0
        for i, j, mult in pairs:
            acc += mult * float(gram[i][j])
        if acc != 0.0:
            coeffs[alpha] = acc
    return Polynomial(n_vars, coeffs)


@dataclass(frozen=True)
class SOSProblem:
    r"""The problem "is ``polynomial`` a sum of squares in ``basis``?".

    ``basis`` defaults to the full monomial basis up to ``ceil(deg(p) / 2)``.
    """

    polynomial: Polynomial
    basis: MonomialBasis

    @classmethod
    def for_polynomial(
        cls, polynomial: Polynomial, *, half_degree: int | None = None
    ) -> SOSProblem:
        if half_degree is None:
            half_degree = (max(polynomial.degree(), 0) + 1) // 2
        basis = MonomialBasis.up_to_degree(polynomial.n_vars, half_degree)
        return cls(polynomial, basis)

    def representable(self) -> bool:
        """Whether every monomial of the polynomial can appear in ``z^T Q z``.

        A monomial outside the product support (e.g. an odd-degree term the basis
        cannot build) makes the SOS problem infeasible.
        """
        products = set(gram_products(self.basis.exponents))
        return self.polynomial.support <= products


__all__ = [
    "MonomialBasis",
    "SOSProblem",
    "arrangement_adapted_basis",
    "gram_products",
    "gram_to_poly",
    "monomial_basis",
]
