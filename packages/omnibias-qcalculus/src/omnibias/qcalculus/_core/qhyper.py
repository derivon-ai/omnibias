# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Basic (q-)hypergeometric series with a certified geometric tail.

The basic hypergeometric series

.. math::

    {}_r\phi_s(a_1,\dots,a_r; b_1,\dots,b_s; q, z)
      = \sum_{n\ge 0}
        \frac{(a_1;q)_n \cdots (a_r;q)_n}{(b_1;q)_n \cdots (b_s;q)_n\, (q;q)_n}
        \bigl((-1)^n q^{\binom{n}{2}}\bigr)^{1+s-r} z^n .

Two registers:

* :func:`basic_hypergeometric` -- the plain float **direct-summation baseline**.
* :func:`basic_hypergeometric_enclosure` -- a **certified** :class:`Interval`: the retained
  terms are exact rationals and the omitted tail is majorised geometrically (reusing
  :func:`omnibias.core.verified.certified_ratio_series_sum`). Valid for ``0 < q < 1`` with
  ``r <= s + 1`` and a rigorous ratio bound ``< 1``; otherwise it raises rather than return
  an unsound bound.

The q-exponential enclosure :func:`q_exp_enclosure` is the ``e_q`` special case.
"""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction

from omnibias.core.verified import Interval, certified_ratio_series_sum
from omnibias.qcalculus._core.qnumbers import q_bracket, q_factorial, q_pochhammer

Rational = Fraction | int


def _frac(v: Rational) -> Fraction:
    return v if isinstance(v, Fraction) else Fraction(v)


def basic_hypergeometric(
    a: Sequence[float],
    b: Sequence[float],
    q: float,
    z: float,
    *,
    terms: int = 64,
) -> float:
    r"""Numerical ``_r phi_s`` by direct truncated summation (the baseline)."""
    if not 0.0 < q < 1.0:
        raise ValueError(f"basic_hypergeometric needs 0 < q < 1, got q={q}")
    if terms < 1:
        raise ValueError(f"terms must be >= 1, got {terms}")
    exponent = 1 + len(b) - len(a)
    total = 0.0
    for n in range(terms):
        num = 1.0
        for ai in a:
            num *= _poch_float(ai, q, n)
        den = 1.0
        for bj in b:
            den *= _poch_float(bj, q, n)
        den *= _poch_float(q, q, n)  # (q; q)_n
        sign_pow = ((-1.0) ** n * q ** (n * (n - 1) / 2.0)) ** exponent
        total += (num / den) * sign_pow * z**n
    return total


def _poch_float(a: float, q: float, n: int) -> float:
    result = 1.0
    power = 1.0
    for _ in range(n):
        result *= 1.0 - a * power
        power *= q
    return result


def _term_exact(
    a: Sequence[Fraction],
    b: Sequence[Fraction],
    q: Fraction,
    z: Fraction,
    exponent: int,
    n: int,
) -> Fraction:
    num = Fraction(1)
    for ai in a:
        num *= q_pochhammer(ai, q, n)
    den = q_pochhammer(q, q, n)  # (q; q)_n
    for bj in b:
        den *= q_pochhammer(bj, q, n)
    # ((-1)^n q^{C(n,2)})^exponent, exponent >= 0 (r <= s + 1)
    sign = Fraction((-1) ** (n * exponent))
    qpow = q ** (exponent * (n * (n - 1) // 2))
    value: Fraction = (num / den) * sign * qpow * z**n
    return value


def basic_hypergeometric_enclosure(
    a: Sequence[Rational],
    b: Sequence[Rational],
    q: Rational,
    z: Rational,
    *,
    terms: int = 32,
) -> Interval:
    r"""Certified :class:`Interval` enclosure of ``_r phi_s`` (``0 < q < 1``, ``r <= s+1``).

    Sums ``terms`` exact-rational retained terms and majorises the omitted tail by a
    rigorous geometric bound on the consecutive term ratio. Raises ``ValueError`` if the
    ratio cannot be certified ``< 1`` at this truncation (raise ``terms`` and retry) or if
    the parameters place the series outside the convergent regime.
    """
    af = [_frac(x) for x in a]
    bf = [_frac(x) for x in b]
    qf, zf = _frac(q), _frac(z)
    if not 0 < qf < 1:
        raise ValueError(f"enclosure needs 0 < q < 1, got q={qf}")
    if terms < 1:
        raise ValueError(f"terms must be >= 1, got {terms}")
    exponent = 1 + len(bf) - len(af)
    if exponent < 0:
        raise ValueError(
            f"r <= s+1 required for a convergent enclosure (r={len(af)}, s={len(bf)})"
        )

    ratio = _ratio_bound(af, bf, qf, zf, exponent, terms)
    if not ratio < 1:
        raise ValueError(
            f"could not certify ratio < 1 at terms={terms} (got {float(ratio):.4g}); "
            "increase terms"
        )

    def term(k: int) -> Interval:
        return Interval.from_rational(_term_exact(af, bf, qf, zf, exponent, k))

    return certified_ratio_series_sum(term, ratio, num_terms=terms)


def _ratio_bound(
    a: Sequence[Fraction],
    b: Sequence[Fraction],
    q: Fraction,
    z: Fraction,
    exponent: int,
    terms: int,
) -> Fraction:
    r"""Rigorous upper bound on ``|t_{n+1}/t_n|`` for all ``n >= terms-1`` (``0 < q < 1``).

    Over ``t = q^n in (0, cap]`` with ``cap = q^{terms-1}``, each linear factor ``1 - c t``
    is monotone, so its magnitude sup / inf lie at the endpoints (with inf ``0`` if it
    changes sign inside). The ``q^{n(1+s-r)}`` factor is ``<= cap^{exponent}`` and the
    ``1/(q;q)`` step contributes ``1/(1 - q^{terms})``.
    """
    cap = q ** (terms - 1)

    num_sup = Fraction(1)
    for ai in a:
        num_sup *= _linear_abs_sup(ai, cap)
    den_inf = Fraction(1)
    for bj in b:
        low = _linear_abs_inf(bj, cap)
        if low == 0:
            raise ValueError(f"denominator factor 1 - {bj} q^n can vanish; cannot certify")
        den_inf *= low
    # (q;q) step: |1 - q^{n+1}| >= 1 - q^{terms} for n >= terms-1
    last_low = 1 - q**terms
    pf = cap**exponent if exponent > 0 else Fraction(1)
    return num_sup * pf * abs(z) / (den_inf * last_low)


def _linear_abs_sup(c: Fraction, cap: Fraction) -> Fraction:
    """sup over t in [0, cap] of |1 - c t| (linear -> attained at an endpoint)."""
    return max(abs(Fraction(1)), abs(1 - c * cap))


def _linear_abs_inf(c: Fraction, cap: Fraction) -> Fraction:
    """inf over t in [0, cap] of |1 - c t| (0 if 1 - c t changes sign inside)."""
    hi = 1 - c * cap
    if (Fraction(1) >= 0) != (hi >= 0):  # sign change inside [0, cap]
        return Fraction(0)
    return min(abs(Fraction(1)), abs(hi))


def q_exp_enclosure(z: Rational, q: Rational, *, terms: int = 32) -> Interval:
    r"""Certified :class:`Interval` enclosure of ``e_q(z) = sum_n z^n / [n]_q!``.

    Exact retained terms with a geometric tail bounded by ``|z| / [terms]_q`` (the q-number
    ``[n+1]_q`` is increasing, so ``|t_{n+1}/t_n| = |z|/[n+1]_q <= |z|/[terms]_q``).
    """
    zf, qf = _frac(z), _frac(q)
    if not 0 < qf < 1:
        raise ValueError(f"q_exp_enclosure needs 0 < q < 1, got q={qf}")
    if terms < 1:
        raise ValueError(f"terms must be >= 1, got {terms}")
    ratio = abs(zf) / q_bracket(terms, qf)
    if not ratio < 1:
        raise ValueError(
            f"could not certify ratio < 1 at terms={terms} (got {float(ratio):.4g}); "
            "increase terms"
        )

    def term(k: int) -> Interval:
        return Interval.from_rational(zf**k / q_factorial(k, qf))

    return certified_ratio_series_sum(term, ratio, num_terms=terms)


__all__ = [
    "basic_hypergeometric",
    "basic_hypergeometric_enclosure",
    "q_exp_enclosure",
]
