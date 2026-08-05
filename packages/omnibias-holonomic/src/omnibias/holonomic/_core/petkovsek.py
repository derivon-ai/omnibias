# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Petkovsek's ``Hyper``: hypergeometric-term solutions of a shift recurrence.

Given a P-recursive recurrence ``sum_{i=0}^r p_i(n) y(n+i) = 0`` (a **shift**-algebra
:class:`~.ore.OrePolynomial`), :func:`hyper` returns every hypergeometric solution -- each
described by its exact term ratio ``r(n) = y(n+1)/y(n)`` as a :data:`~.ratfunc.RatFunc`.
It is the companion of Gosper's algorithm (indefinite summation): where Gosper decides one
term, ``Hyper`` enumerates the whole hypergeometric solution space of a recurrence of any
order.

The algorithm is the classic Gosper-Petkovsek construction (Petkovsek 1992; "A=B", ch. 8):
a hypergeometric ratio factors as

.. math::

    r(n) = z\,\frac{a(n)}{b(n)}\,\frac{c(n+1)}{c(n)},

with ``a`` a monic factor of the trailing coefficient ``p_0``, ``b`` a monic factor of the
shifted leading coefficient ``p_r(n+r-1)``, ``z`` a non-zero constant, and ``c`` an unknown
polynomial. Substituting and clearing denominators turns the recurrence into

.. math::

    \sum_{i=0}^r p_i(n)\, z^i \Bigl(\prod_{l=0}^{i-1} a(n+l)\Bigr)
        \Bigl(\prod_{l=i}^{r-1} b(n+l)\Bigr) c(n+i) = 0,

whose top-degree balance fixes the candidate ``z`` (rational roots of an indicial polynomial)
and whose remaining coefficients are an exact homogeneous linear system for ``c`` (solved with
the rational null space of :mod:`.linalg`).

**Honesty / scope.** Exact within the *linear / rational-root factorisation regime*: the
monic divisors of ``p_0`` and ``p_r`` are enumerated from their rational-root linear factors
(:mod:`.factor`), and ``z`` is searched among rational roots -- so solutions requiring an
irreducible non-linear factor or an irrational/algebraic ``z`` (e.g. Fibonacci's ``phi^n``)
are out of scope and simply not returned. Every returned ratio is **verified exactly**
(:func:`term_ratio_annihilates`) as a rational-function identity against the operator, so the
output never contains a spurious solution.
"""

from __future__ import annotations

from fractions import Fraction

from omnibias.holonomic._core.factor import monic_linear_divisors, rational_roots
from omnibias.holonomic._core.linalg import null_space
from omnibias.holonomic._core.ore import OrePolynomial
from omnibias.holonomic._core.ratfunc import (
    RatFunc,
    rf_add,
    rf_from_poly,
    rf_from_rational,
    rf_is_zero,
    rf_mul,
    rf_normalize,
    rf_shift,
    rf_zero,
)
from omnibias.holonomic._core.rational_poly import (
    Poly,
    degree,
    is_zero,
    padd,
    pmul,
    pscale,
    pshift,
    to_poly,
)

_ONE: Poly = (Fraction(1),)


def _monomial(e: int) -> Poly:
    """The polynomial ``n^e``."""
    return tuple(Fraction(0) for _ in range(e)) + (Fraction(1),)


def _prod(polys: list[Poly]) -> Poly:
    out: Poly = _ONE
    for p in polys:
        out = pmul(out, p)
    return out


def _build_pi(coeffs: list[Poly], a: Poly, b: Poly, r: int) -> list[Poly]:
    r"""``P_i(n) = p_i(n) prod_{l<i} a(n+l) prod_{i<=l<r} b(n+l)`` for ``i = 0..r``."""
    out: list[Poly] = []
    for i in range(r + 1):
        a_part = _prod([pshift(a, j) for j in range(i)])
        b_part = _prod([pshift(b, j) for j in range(i, r)])
        out.append(pmul(coeffs[i], pmul(a_part, b_part)))
    return out


def _indicial_roots(pis: list[Poly]) -> list[Fraction]:
    """Candidate non-zero ``z``: rational roots of the top-degree balance ``sum lc_i z^i``."""
    degs = [degree(p) for p in pis]
    top = max(degs)
    if top < 0:
        return []
    q_coeffs = [Fraction(0)] * (len(pis))
    for i, p in enumerate(pis):
        if degree(p) == top:
            q_coeffs[i] = p[-1]
    q = to_poly(q_coeffs)
    if is_zero(q):
        return []
    return [z for z in rational_roots(q) if z != 0]


def _solve_c(pis: list[Poly], z: Fraction, max_c_degree: int) -> Poly | None:
    r"""Minimal non-zero ``c`` with ``sum_i P_i(n) z^i c(n+i) = 0``, or ``None``."""
    r = len(pis) - 1
    z_pow = [Fraction(1)]
    for _ in range(r):
        z_pow.append(z_pow[-1] * z)
    for delta in range(max_c_degree + 1):
        columns: list[Poly] = []
        for e in range(delta + 1):
            col: Poly = ()
            mono = _monomial(e)
            for i in range(r + 1):
                if is_zero(pis[i]):
                    continue
                shifted = pshift(mono, i)  # (n + i)^e
                col = padd(col, pscale(pmul(pis[i], shifted), z_pow[i]))
            columns.append(col)
        max_power = max((len(c) for c in columns), default=0)
        rows: list[list[Fraction]] = []
        for power in range(max_power):
            rows.append([col[power] if power < len(col) else Fraction(0) for col in columns])
        if not rows:
            continue
        basis = null_space(rows)
        for sol in basis:
            c = to_poly(sol)
            if not is_zero(c):
                lead = c[-1]
                return tuple(coeff / lead for coeff in c)
    return None


def term_ratio_annihilates(op: OrePolynomial, ratio: RatFunc) -> bool:
    r"""Whether ``r(n) = ratio`` solves ``op`` exactly: ``sum_i p_i(n) prod_{l<i} r(n+l) = 0``.

    A rational-function identity check (no sampling): the hypergeometric term with
    ``y(n+1)/y(n) = ratio`` annihilates the operator iff this sum is the zero rational
    function.
    """
    if op.algebra.name != "shift":
        raise ValueError("Petkovsek Hyper is defined for the shift algebra")
    total = rf_zero()
    running = rf_from_rational(1)  # prod_{l<i} r(n+l); i = 0 -> empty product = 1
    for i, coeff in enumerate(op.coeffs):
        if i > 0:
            running = rf_mul(running, rf_shift(ratio, i - 1))
        if not is_zero(coeff):
            total = rf_add(total, rf_mul(rf_from_poly(coeff), running))
    return bool(rf_is_zero(total))


def hyper(op: OrePolynomial, *, max_c_degree: int = 4) -> list[RatFunc]:
    r"""All hypergeometric term ratios ``y(n+1)/y(n)`` solving the shift recurrence ``op``.

    Returns a list of exact :data:`~.ratfunc.RatFunc` ratios (deduplicated, each verified by
    :func:`term_ratio_annihilates`). The empty list is a genuine finding: no hypergeometric
    solution exists within the rational-factor / rational-``z`` regime. ``max_c_degree`` caps
    the degree of the unknown Petkovsek polynomial ``c`` (default 4 covers the classical
    factorial / binomial / geometric families).

    Raises :class:`NotImplementedError` when the trailing/leading coefficient vanishes (the
    recurrence must be reduced first -- out of scope here) and :class:`ValueError` for a
    non-shift algebra.
    """
    if op.algebra.name != "shift":
        raise ValueError("Petkovsek Hyper is defined for the shift algebra")
    r = op.order
    if r < 1:
        raise ValueError("hyper needs a recurrence of order >= 1")
    coeffs = [op.coeffs[i] if i < len(op.coeffs) else () for i in range(r + 1)]
    p0, pr = coeffs[0], coeffs[r]
    if is_zero(p0) or is_zero(pr):
        raise NotImplementedError(
            "hyper requires non-zero trailing and leading coefficients; reduce the recurrence first"
        )
    a_divs = monic_linear_divisors(p0)
    b_divs = monic_linear_divisors(pshift(pr, r - 1))
    found: list[RatFunc] = []
    for a in a_divs:
        for b in b_divs:
            pis = _build_pi(coeffs, a, b, r)
            for z in _indicial_roots(pis):
                c = _solve_c(pis, z, max_c_degree)
                if c is None:
                    continue
                num = pscale(pmul(a, pshift(c, 1)), z)
                den = pmul(b, c)
                if is_zero(num) or is_zero(den):
                    continue
                ratio = rf_normalize(num, den)
                if rf_is_zero(ratio):
                    continue
                if not term_ratio_annihilates(op, ratio):
                    continue
                if all(existing != ratio for existing in found):
                    found.append(ratio)
    return found


__all__ = [
    "hyper",
    "term_ratio_annihilates",
]
