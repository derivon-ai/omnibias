# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""q-holonomic bridge: the q-shift algebra, q-Gosper, and q-Zeilberger.

The q-analogue of the holonomic machinery, working in the variable ``x = q^k`` (for the
summation index) or ``x = q^n`` (for the recurrence index) over the rationals, with the
**q-shift** acting multiplicatively, ``(sigma p)(x) = p(q x)`` -- the exact operator behind
the Jackson q-calculus of :mod:`omnibias.qcalculus`.

* :func:`q_shift_algebra` builds the Ore algebra ``R[S; sigma, 0]`` with ``sigma`` the
  q-dilation, so a q-recurrence ``sum_i c_i(q^n) y(n+i) = 0`` is an
  :class:`~.ore.OrePolynomial` whose coefficients are polynomials in ``x = q^n``.
* :func:`q_gosper` is the q-analogue of Gosper's algorithm: it decides whether a
  q-hypergeometric term ``t`` (ratio ``t(k+1)/t(k)`` a fixed rational function of ``x=q^k``)
  has a q-hypergeometric antidifference and returns the exact rational certificate ``R(x)``
  with ``sum_{k=a}^{b-1} t(k) = R(q^b)t(b) - R(q^a)t(a)`` (:func:`q_gosper_definite_sum`).
* :func:`q_zeilberger` produces a q-recurrence for a q-hypergeometric single sum
  ``S(n) = sum_k F(n,k)`` -- **guessed** (exact rational null space) then **verified** on an
  extended range (the same discipline as the ordinary ``creative_telescoping``).

**Honesty / scope.** ``q`` is a fixed exact **rational** parameter: every certificate is an
exact rational function of ``x`` (no rounding), sound at that ``q``. This is *q-numeric*, not
a single symbolic-in-``q`` certificate; probes exercise a **q-sweep** and the **distinct**
``q -> 1`` limit (``[n]_q -> n``), which is a different limit from the ``delta -> 0`` founding
collapse of :mod:`omnibias.difference` and the ``beta -> inf`` feasibility penalty. q-Gosper's
antidifference is closed-form / exact; q-Zeilberger's recurrence is guessed-then-verified.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from fractions import Fraction

from omnibias.holonomic._core.linalg import null_space, solve_exact
from omnibias.holonomic._core.ore import OreAlgebra, OrePolynomial
from omnibias.holonomic._core.rational_poly import (
    Poly,
    degree,
    is_zero,
    pdivmod,
    peval,
    pgcd,
    pmul,
    pscale,
    psub,
    to_poly,
)

Rational = Fraction | int


def _frac(v: Rational) -> Fraction:
    return v if isinstance(v, Fraction) else Fraction(v)


def _check_q(q: Fraction) -> None:
    if q == 0:
        raise ValueError("q must be non-zero")
    if q == 1:
        raise ValueError("q must not be 1 (the q -> 1 limit is the ordinary/holonomic case)")


def q_dilate(p: Poly, s: Fraction) -> Poly:
    r"""Substitute ``x -> s x`` in ``p`` (so ``q_dilate(p, q)`` is ``p(qx)``); exact."""
    return tuple(c * s**i for i, c in enumerate(p))


def _monomial(e: int) -> Poly:
    return tuple(Fraction(0) for _ in range(e)) + (Fraction(1),)


def q_shift_algebra(q: Rational) -> OreAlgebra:
    r"""The q-shift Ore algebra ``sigma(p)(x) = p(q x)``, ``delta = 0`` (q-recurrences).

    The generator ``S`` is the forward q-shift; an operator ``sum_i c_i(x) S^i`` with
    ``x = q^n`` is a q-recurrence. Applied to a sequence by :func:`q_apply`.
    """
    qf = _frac(q)
    _check_q(qf)
    return OreAlgebra("q-shift", lambda p: q_dilate(p, qf), lambda _p: ())


def q_apply(op: OrePolynomial, values: Callable[[int], Fraction], n: int, q: Rational) -> Fraction:
    r"""Apply a q-shift operator to a sequence: ``sum_i c_i(q^n) values(n+i)`` (exact)."""
    qf = _frac(q)
    x = qf**n
    total = Fraction(0)
    for i, c in enumerate(op.coeffs):
        if c:
            total += peval(c, x) * values(n + i)
    return total


# --------------------------------------------------------------------------- #
# q-Gosper.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class QGosperResult:
    """Outcome of q-Gosper on a q-hypergeometric ratio ``num/den`` (variable ``x = q^k``)."""

    summable: bool
    q: Fraction = Fraction(2)
    cert_num: Poly = ()
    cert_den: Poly = ()

    def certificate(self, k: int) -> Fraction:
        """Evaluate ``R(q^k)`` with ``T(k) = R(q^k) t(k)`` (raises if not summable)."""
        if not self.summable:
            raise ValueError("not q-Gosper-summable; no certificate")
        x = self.q**k
        return Fraction(peval(self.cert_num, x) / peval(self.cert_den, x))


def _q_dispersion(a: Poly, b: Poly, q: Fraction, *, bound: int = 64) -> list[int]:
    """Non-negative ``j`` with ``deg gcd(a(x), b(q^j x)) > 0`` (the q-dispersion set)."""
    if is_zero(a) or is_zero(b):
        return []
    out: list[int] = []
    for j in range(bound + 1):
        if degree(pgcd(a, q_dilate(b, q**j))) > 0:
            out.append(j)
    return out


def q_gosper_normal_form(num: Poly, den: Poly, q: Fraction) -> tuple[Poly, Poly, Poly]:
    r"""q-Gosper-Petkovsek form ``(a, b, c)`` with ``num/den = (a/b)(c(qx)/c(x))``.

    Guarantees ``gcd(a(x), b(q^j x)) = 1`` for every non-negative integer ``j`` (the
    q-analogue of the Gosper-Petkovsek normal form).
    """
    a, b = num, den
    g0 = pgcd(a, b)
    if degree(g0) > 0:
        a = pdivmod(a, g0)[0]
        b = pdivmod(b, g0)[0]
    c: Poly = (Fraction(1),)
    while True:
        shifts = [j for j in _q_dispersion(a, b, q) if j >= 1]
        if not shifts:
            break
        j = shifts[0]
        g = pgcd(a, q_dilate(b, q**j))
        a = pdivmod(a, g)[0]
        b = pdivmod(b, q_dilate(g, q ** (-j)))[0]
        for i in range(1, j + 1):
            c = pmul(c, q_dilate(g, q ** (-i)))
    return a, b, c


def _q_degree_bound(bigA: Poly, bigB: Poly, bigC: Poly, q: Fraction) -> int:
    dA, dB, dC = degree(bigA), degree(bigB), degree(bigC)
    base = dC - min(dA, dB) if dC >= 0 else 0
    cand = max(base, dC, 0)
    # In A(x) f(qx) - B(x) f(x) the top term is (lcA q^m - lcB) x^{dA+m} when dA == dB;
    # a genuine cancellation at degree m needs q^m = lcB/lcA, which admits a higher f.
    if dA == dB and dA >= 0:
        ratio = bigB[-1] / bigA[-1]
        power = Fraction(1)
        for m in range(cand + 65):
            if power == ratio:
                cand = max(cand, m)
                break
            power *= q
    return int(cand)


def _q_verify(num: Poly, den: Poly, cert_num: Poly, cert_den: Poly, q: Fraction) -> bool:
    """Exactly verify the q-Gosper relation ``(num/den) R(qx) - R(x) = 1`` (``R = cN/cD``)."""
    rn_q = q_dilate(cert_num, q)
    rd_q = q_dilate(cert_den, q)
    # numerator of (num/den)(rn_q/rd_q) - cert_num/cert_den - 1 over den*rd_q*cert_den:
    n1 = pmul(pmul(num, rn_q), cert_den)
    n2 = pmul(pmul(cert_num, den), rd_q)
    n3 = pmul(pmul(den, rd_q), cert_den)
    return bool(is_zero(psub(psub(n1, n2), n3)))


def q_gosper(num: Poly, den: Poly, q: Rational) -> QGosperResult:
    r"""Run q-Gosper on the q-hypergeometric ratio ``t(k+1)/t(k) = num(x)/den(x)``, ``x=q^k``.

    Returns a :class:`QGosperResult`; ``summable`` is ``True`` iff a q-hypergeometric
    antidifference exists (the returned certificate is verified exactly), and ``False`` -- a
    genuine finding within the search bounds -- otherwise.
    """
    qf = _frac(q)
    _check_q(qf)
    num, den = to_poly(num), to_poly(den)
    if degree(den) < 0:
        raise ValueError("den must be non-zero")
    if degree(num) < 0:
        return QGosperResult(summable=False, q=qf)
    a, b, c = q_gosper_normal_form(num, den, qf)
    bigA, bigB, bigC = a, q_dilate(b, qf ** (-1)), c  # bigB = b(x/q)
    bound = _q_degree_bound(bigA, bigB, bigC, qf)
    if bound < 0:
        return QGosperResult(summable=False, q=qf)
    # Solve A(x) f(qx) - B(x) f(x) = C(x) for f of degree <= bound.
    cols: list[Poly] = []
    for e in range(bound + 1):
        contrib = pmul(psub(pscale(bigA, qf**e), bigB), _monomial(e))
        cols.append(contrib)
    max_deg = max((degree(col) for col in cols), default=-1)
    max_deg = max(max_deg, degree(bigC), 0)
    matrix = [[(col[p] if p < len(col) else Fraction(0)) for col in cols] for p in range(max_deg + 1)]
    rhs = [(bigC[p] if p < len(bigC) else Fraction(0)) for p in range(max_deg + 1)]
    sol = solve_exact(matrix, rhs)
    if sol is None:
        return QGosperResult(summable=False, q=qf)
    f = to_poly(sol)
    cert_num = pmul(bigB, f)
    cert_den = c
    if degree(cert_num) < 0 or not _q_verify(num, den, cert_num, cert_den, qf):
        return QGosperResult(summable=False, q=qf)
    return QGosperResult(summable=True, q=qf, cert_num=cert_num, cert_den=cert_den)


def q_gosper_definite_sum(
    num: Poly, den: Poly, term0: Rational, a: int, b: int, q: Rational
) -> Fraction | None:
    r"""Exact definite q-sum ``sum_{k=a}^{b-1} t(k)`` via q-Gosper (``t(a) = term0``).

    ``t`` is the q-hypergeometric term with ``t(k+1)/t(k) = num(q^k)/den(q^k)``. When
    q-Gosper finds an antidifference ``T = R t`` the sum telescopes exactly to
    ``R(q^b) t(b) - R(q^a) t(a)``. Returns ``None`` (a genuine finding) when ``t`` is **not**
    q-Gosper-summable. Unconditional and closed-form -- no fitting.
    """
    qf = _frac(q)
    result = q_gosper(num, den, qf)
    if not result.summable:
        return None
    if b == a:
        return Fraction(0)
    num_p, den_p = to_poly(num), to_poly(den)
    t: dict[int, Fraction] = {a: _frac(term0)}
    for k in range(a, b):
        x = qf**k
        d = peval(den_p, x)
        if d == 0:
            raise ValueError(f"term ratio denominator vanishes at k={k}")
        t[k + 1] = t[k] * peval(num_p, x) / d
    return Fraction(result.certificate(b) * t[b] - result.certificate(a) * t[a])


# --------------------------------------------------------------------------- #
# q-Zeilberger (guessed-then-verified single-sum recurrence).
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class QRecurrence:
    """A guessed-then-verified q-recurrence ``sum_i c_i(q^n) S(n+i) = 0`` for ``S(n)``."""

    operator: OrePolynomial
    q: Fraction
    checked_upto: int
    values: list[Fraction] = field(default_factory=list)

    @property
    def order(self) -> int:
        """The order of the q-recurrence operator."""
        return int(self.operator.order)

    def residual(self, n: int) -> Fraction:
        """The q-recurrence residual ``sum_i c_i(q^n) S(n+i)`` at ``n`` (exact)."""
        return q_apply(self.operator, lambda m: self.values[m], n, self.q)

    def max_residual(self) -> Fraction:
        """Max ``|residual(n)|`` over the verified range (exactly 0 for a true recurrence)."""
        top = self.checked_upto - self.order
        return max((abs(self.residual(n)) for n in range(top + 1)), default=Fraction(0))


def q_zeilberger(
    summand: Callable[[int, int], Rational],
    q: Rational,
    *,
    max_order: int = 4,
    max_index_degree: int = 3,
    n_max: int = 20,
    k_bound: Callable[[int], tuple[int, int]] | None = None,
) -> QRecurrence | None:
    r"""Guess-then-verify a q-recurrence for ``S(n) = sum_k F(n, k)`` (``F = summand``).

    Fits the minimal ``sum_{i=0}^d c_i(q^n) S(n+i) = 0`` with ``c_i`` polynomials in
    ``x = q^n`` of degree ``<= max_index_degree`` (exact rational null space over the sampled
    ``S(n)``), then **verifies** it on the remaining sampled range. Returns ``None`` when no
    such recurrence fits within the bounds -- a genuine finding, not a guess.
    """
    qf = _frac(q)
    _check_q(qf)

    def bounds(n: int) -> tuple[int, int]:
        return k_bound(n) if k_bound is not None else (0, n)

    def S(n: int) -> Fraction:
        lo, hi = bounds(n)
        return sum((_frac(summand(n, k)) for k in range(lo, hi + 1)), Fraction(0))

    values = [S(n) for n in range(n_max + 1)]
    for d in range(1, max_order + 1):
        for D in range(max_index_degree + 1):
            n_unknowns = (d + 1) * (D + 1)
            # Need enough equations (rows) to pin a (hopefully 1-dim) null space.
            fit_rows = min(n_max - d, n_unknowns + 2)
            if fit_rows < n_unknowns - 1:
                continue
            matrix: list[list[Fraction]] = []
            for n in range(fit_rows):
                x = qf**n
                row: list[Fraction] = []
                for i in range(d + 1):
                    for e in range(D + 1):
                        row.append(x**e * values[n + i])
                matrix.append(row)
            for sol in null_space(matrix):
                coeffs: list[Poly] = []
                for i in range(d + 1):
                    block = sol[i * (D + 1) : (i + 1) * (D + 1)]
                    coeffs.append(to_poly(block))
                if is_zero(coeffs[d]):
                    continue  # leading coefficient must be non-trivial for order exactly d
                op = q_shift_algebra(qf).operator(coeffs)
                rec = QRecurrence(operator=op, q=qf, checked_upto=n_max, values=values)
                if rec.max_residual() == 0 and op.order == d:
                    return rec
    return None


__all__ = [
    "QGosperResult",
    "QRecurrence",
    "q_apply",
    "q_dilate",
    "q_gosper",
    "q_gosper_definite_sum",
    "q_gosper_normal_form",
    "q_shift_algebra",
    "q_zeilberger",
]
