# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Transforms between the D-finite and P-recursive registers, plus D-finite closures.

Two exact, structure-preserving transforms and three exact closures:

* :func:`dfinite_to_precursive` -- turn the ODE ``sum_i c_i(x) D^i f = 0`` into the
  P-recurrence its Taylor coefficients obey (``D^i x^m = m^{\underline i} x^{m-i}``);
* :func:`precursive_to_dfinite` -- the inverse, via the Euler operator ``theta = x D``
  (``theta^b = sum_i S(b,i) x^i D^i`` with Stirling numbers of the second kind);
* :func:`dfinite_derivative`, :func:`dfinite_integral`, :func:`dfinite_compose_poly` --
  ``f'``, ``\int_0^x f``, and ``f(p(x))`` (polynomial ``p`` with ``p(0)=0``) are again
  D-finite; their annihilators come from the **cyclic-vector** construction: reduce the
  derivatives of the target onto the finite basis ``{f, Df, ..., D^{r-1} f}`` over ``Q(x)``
  and read off the minimal operator with :func:`~.relations.find_poly_relation`.

Every returned object is **verified exactly** -- the transformed annihilator is required to
regenerate the transformed sequence / series over a checked prefix (a false transform raises
rather than returning an unsound result). Honesty: **closed-form / exact**. Composition is
scoped to polynomial substitutions with ``p(0)=0`` (formal-power-series composition);
non-polynomial algebraic substitution is out of scope and raises.
"""

from __future__ import annotations

from fractions import Fraction

from omnibias.holonomic._core.dfinite import DFinite, PRecursive
from omnibias.holonomic._core.ore import OrePolynomial, diff_algebra, shift_algebra
from omnibias.holonomic._core.ratfunc import (
    RatFunc,
    rf_add,
    rf_div,
    rf_from_poly,
    rf_from_rational,
    rf_is_zero,
    rf_mul,
    rf_neg,
    rf_zero,
)
from omnibias.holonomic._core.rational_poly import (
    Poly,
    is_zero,
    padd,
    pderiv,
    pmul,
    to_poly,
)
from omnibias.holonomic._core.relations import find_poly_relation

_ONE: Poly = (Fraction(1),)


# --------------------------------------------------------------------------- #
# Small polynomial / rational-function helpers.
# --------------------------------------------------------------------------- #
def _falling(shift: int, i: int) -> Poly:
    r"""``(n + shift)^{\underline i} = prod_{t=0}^{i-1}(n + shift - t)`` as a polynomial in ``n``."""
    out: Poly = _ONE
    for t in range(i):
        out = pmul(out, (Fraction(shift - t), Fraction(1)))
    return out


def _pcompose(p: Poly, q: Poly) -> Poly:
    """Polynomial composition ``p(q(x))`` (Horner)."""
    acc: Poly = ()
    for c in reversed(p):
        acc = padd(pmul(acc, q), (c,))
    return acc


def _rf_deriv(rf: RatFunc) -> RatFunc:
    """Exact derivative ``d/dx (num/den)`` as a rational function."""
    from omnibias.holonomic._core.ratfunc import rf_normalize
    from omnibias.holonomic._core.rational_poly import psub

    num, den = rf
    n2 = psub(pmul(pderiv(num), den), pmul(num, pderiv(den)))
    d2 = pmul(den, den)
    return rf_normalize(n2, d2)


def _rf_compose_poly(rf: RatFunc, q: Poly) -> RatFunc:
    """Substitute ``x -> q(x)`` into a rational function ``num/den``."""
    from omnibias.holonomic._core.ratfunc import rf_normalize

    num, den = rf
    return rf_normalize(_pcompose(num, q), _pcompose(den, q))


def _reductions(op: OrePolynomial) -> list[RatFunc]:
    r"""``R_i`` with ``D^r f = sum_{i<r} R_i D^i f`` (``r = op.order``), over ``Q(x)``."""
    r = op.order
    lead = rf_from_poly(op.coeffs[r])
    out: list[RatFunc] = []
    for i in range(r):
        ci = op.coeffs[i] if i < len(op.coeffs) else ()
        out.append(rf_neg(rf_div(rf_from_poly(ci), lead)))
    return out


def _stirling2_row(b: int) -> list[int]:
    """Row ``[S(b,0), ..., S(b,b)]`` of Stirling numbers of the second kind."""
    prev = [1]  # row 0: S(0,0) = 1
    for m in range(1, b + 1):
        cur = [0] * (m + 1)
        for i in range(1, m + 1):
            left = i * prev[i] if i < len(prev) else 0
            below = prev[i - 1] if i - 1 < len(prev) else 0
            cur[i] = left + below
        prev = cur
    return prev


def _strip_common_x(coeffs: list[Poly]) -> list[Poly]:
    r"""Divide the operator by the highest common ``x^t`` (left factor; preserves the kernel)."""
    vals = []
    for c in coeffs:
        if not is_zero(c):
            vals.append(next(idx for idx, v in enumerate(c) if v != 0))
    if not vals:
        return coeffs
    t = min(vals)
    if t == 0:
        return coeffs
    return [() if is_zero(c) else c[t:] for c in coeffs]


# --------------------------------------------------------------------------- #
# ODE <-> recurrence.
# --------------------------------------------------------------------------- #
def dfinite_to_precursive(d: DFinite, *, terms: int = 40) -> PRecursive:
    r"""The P-recurrence obeyed by the Taylor coefficients of a D-finite series (exact)."""
    from omnibias.holonomic._core.dfinite import _wrap_with_op

    op = d.annihilator
    shifts: list[int] = []
    for i, c in enumerate(op.coeffs):
        for j, cij in enumerate(c):
            if cij != 0:
                shifts.append(i - j)
    if not shifts:
        raise ValueError("empty differential operator")
    smin, smax = min(shifts), max(shifts)
    order = smax - smin
    if order < 1:
        raise ValueError("differential operator does not induce a recurrence of order >= 1")
    qs: list[Poly] = [()] * (order + 1)
    for i, c in enumerate(op.coeffs):
        for j, cij in enumerate(c):
            if cij == 0:
                continue
            p = (i - j) - smin
            qs[p] = padd(qs[p], tuple(cij * v for v in _falling(p, i)))
    rec = shift_algebra().operator(qs)
    samples = d.taylor(terms)
    return _wrap_with_op(rec, samples)


def precursive_to_dfinite(p: PRecursive, *, terms: int = 40) -> DFinite:
    r"""The ODE obeyed by the generating function of a P-recursive sequence (exact).

    Uses ``sum_p q_p(n) a_{n+p} = 0`` -> ``sum_p x^{P-p} q_p(theta - p)`` with ``theta = x D``
    and ``theta^b = sum_i S(b,i) x^i D^i``; the common ``x`` factor is stripped and the result
    is verified against the series it must annihilate.
    """
    op = p.annihilator
    P = op.order
    # theta-form accumulator: (x-power a, D-power i) -> coefficient.
    dcoeffs: dict[int, Poly] = {}

    def add(i: int, xpow: int, val: Fraction) -> None:
        if val == 0:
            return
        cur = dcoeffs.get(i, ())
        pad = (Fraction(0),) * xpow + (val,)
        dcoeffs[i] = padd(cur, pad)

    for shift in range(P + 1):
        qp = op.coeffs[shift] if shift < len(op.coeffs) else ()
        xbase = P - shift
        for c_deg, qc in enumerate(qp):  # coefficient of n^{c_deg} in q_shift
            if qc == 0:
                continue
            # (theta - shift)^{c_deg} = sum_b C(c_deg, b) theta^b (-shift)^{c_deg - b}
            for b in range(c_deg + 1):
                from math import comb

                scalar = qc * comb(c_deg, b) * Fraction(-shift) ** (c_deg - b)
                if scalar == 0:
                    continue
                stir = _stirling2_row(b)
                for i, s in enumerate(stir):
                    if s:
                        add(i, xbase + i, scalar * s)
    max_i = max(dcoeffs) if dcoeffs else 0
    coeffs = [dcoeffs.get(i, ()) for i in range(max_i + 1)]
    coeffs = _strip_common_x(coeffs)
    ode = diff_algebra().operator(coeffs)
    order = ode.order
    if order < 1:
        raise ValueError("recurrence does not induce an ODE of order >= 1")
    target = p.terms(terms)
    return _verify_dfinite(ode, target)


# --------------------------------------------------------------------------- #
# D-finite closures via the cyclic-vector construction.
# --------------------------------------------------------------------------- #
def _verify_dfinite(op: OrePolynomial, target: list[Fraction]) -> DFinite:
    """Build a :class:`DFinite` for ``target`` and verify the operator regenerates it."""
    coeffs = _strip_common_x(list(op.coeffs))
    op = diff_algebra().operator(coeffs)
    order = op.order
    if order < 1 or len(target) < order:
        raise ValueError("degenerate operator or too few series terms to verify")
    candidate = DFinite(op, tuple(target[:order]))
    try:
        regen = candidate.taylor(len(target))
    except ValueError as exc:
        raise ValueError(f"transformed ODE is singular at 0 (out of scope): {exc}") from exc
    if regen != target:
        raise ValueError("transformed ODE failed exact verification on the series prefix")
    return candidate


def _annihilator(columns: list[list[RatFunc]], *, max_degree: int) -> list[Poly]:
    rel = find_poly_relation(columns, max_degree=max_degree)
    if rel is None:
        raise ValueError("no D-finite annihilator found within the degree bound")
    return list(rel)


def _coeff_degrees(op: OrePolynomial) -> int:
    return max((len(c) - 1 for c in op.coeffs if c), default=0)


def dfinite_derivative(d: DFinite, *, terms: int = 40) -> DFinite:
    r"""The D-finite annihilator of ``f'`` (exact, verified)."""
    op = d.annihilator
    r = op.order
    R = _reductions(op)

    def deriv(w: list[RatFunc]) -> list[RatFunc]:
        out = [rf_zero() for _ in range(r)]
        for i in range(r):
            out[i] = rf_add(out[i], _rf_deriv(w[i]))
        for i in range(r):
            if rf_is_zero(w[i]):
                continue
            if i < r - 1:
                out[i + 1] = rf_add(out[i + 1], w[i])
            else:
                for k in range(r):
                    out[k] = rf_add(out[k], rf_mul(w[i], R[k]))
        return out

    start = [rf_zero() for _ in range(r)]
    if r >= 2:
        start[1] = rf_from_rational(1)
    else:
        start[0] = R[0]
    columns = [start]
    cur = start
    for _ in range(r):
        cur = deriv(cur)
        columns.append(cur)
    rel = _annihilator(columns, max_degree=_coeff_degrees(op) + 3)
    ode = diff_algebra().operator(rel)
    a = d.taylor(terms + 1)
    target = [Fraction(m + 1) * a[m + 1] for m in range(terms)]
    return _verify_dfinite(ode, target)


def dfinite_integral(d: DFinite, *, terms: int = 40) -> DFinite:
    r"""The D-finite annihilator of ``\int_0^x f`` (exact, verified)."""
    op = d.annihilator
    r = op.order
    R = _reductions(op)
    dim = r + 1  # index 0 = G = integral; index m (1..r) = D^{m-1} f

    def deriv(w: list[RatFunc]) -> list[RatFunc]:
        out = [rf_zero() for _ in range(dim)]
        for i in range(dim):
            out[i] = rf_add(out[i], _rf_deriv(w[i]))
        if not rf_is_zero(w[0]):
            out[1] = rf_add(out[1], w[0])  # D G = f
        for m in range(1, dim):
            if rf_is_zero(w[m]):
                continue
            if m < r:
                out[m + 1] = rf_add(out[m + 1], w[m])
            else:
                for k in range(r):
                    out[k + 1] = rf_add(out[k + 1], rf_mul(w[m], R[k]))
        return out

    start = [rf_zero() for _ in range(dim)]
    start[0] = rf_from_rational(1)
    columns = [start]
    cur = start
    for _ in range(dim):
        cur = deriv(cur)
        columns.append(cur)
    rel = _annihilator(columns, max_degree=_coeff_degrees(op) + 3)
    ode = diff_algebra().operator(rel)
    a = d.taylor(terms)
    target = [Fraction(0)] + [a[m - 1] / m for m in range(1, terms)]
    return _verify_dfinite(ode, target)


def dfinite_compose_poly(d: DFinite, poly: Poly, *, terms: int = 40) -> DFinite:
    r"""The D-finite annihilator of ``f(p(x))`` for a polynomial ``p`` with ``p(0)=0`` (exact)."""
    poly = to_poly(poly)
    if is_zero(poly) or poly[0] != 0:
        raise ValueError("dfinite_compose_poly needs a polynomial p with p(0) = 0")
    if len(poly) < 2:
        raise ValueError("dfinite_compose_poly needs a non-constant polynomial")
    op = d.annihilator
    r = op.order
    R = _reductions(op)
    Rc = [_rf_compose_poly(R[k], poly) for k in range(r)]
    pd = rf_from_poly(pderiv(poly))

    def deriv(w: list[RatFunc]) -> list[RatFunc]:
        out = [rf_zero() for _ in range(r)]
        for i in range(r):
            out[i] = rf_add(out[i], _rf_deriv(w[i]))
        for i in range(r):
            if rf_is_zero(w[i]):
                continue
            term = rf_mul(w[i], pd)
            if i < r - 1:
                out[i + 1] = rf_add(out[i + 1], term)
            else:
                for k in range(r):
                    out[k] = rf_add(out[k], rf_mul(term, Rc[k]))
        return out

    start = [rf_zero() for _ in range(r)]
    start[0] = rf_from_rational(1)
    columns = [start]
    cur = start
    for _ in range(r):
        cur = deriv(cur)
        columns.append(cur)
    rel = _annihilator(columns, max_degree=_coeff_degrees(op) + 2 * (len(poly) - 1) + 3)
    ode = diff_algebra().operator(rel)
    a = d.taylor(terms)
    target = _series_compose(a, poly, terms)
    return _verify_dfinite(ode, target)


def _series_compose(a: list[Fraction], poly: Poly, terms: int) -> list[Fraction]:
    """Formal composition ``sum_i a_i poly(x)^i`` truncated to ``terms`` coefficients."""
    result = [Fraction(0)] * terms
    power: list[Fraction] = [Fraction(0)] * terms
    if terms > 0:
        power[0] = Fraction(1)  # poly^0 = 1
    poly_trunc = [Fraction(poly[j]) if j < len(poly) else Fraction(0) for j in range(terms)]
    for i, ai in enumerate(a):
        if i >= terms and all(v == 0 for v in power):
            break
        for s in range(terms):
            if power[s]:
                result[s] += ai * power[s]
        # power *= poly (truncated)
        nxt = [Fraction(0)] * terms
        for s in range(terms):
            if power[s] == 0:
                continue
            for t in range(terms - s):
                if poly_trunc[t]:
                    nxt[s + t] += power[s] * poly_trunc[t]
        power = nxt
    return result


__all__ = [
    "dfinite_compose_poly",
    "dfinite_derivative",
    "dfinite_integral",
    "dfinite_to_precursive",
    "precursive_to_dfinite",
]
