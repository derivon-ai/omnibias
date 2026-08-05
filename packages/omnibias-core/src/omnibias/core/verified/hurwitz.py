# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Certified Hurwitz zeta -- exact negative-integer values + Euler-Maclaurin.

The Hurwitz zeta ``zeta(s, a) = sum_{k>=0} (k + a)^{-s}`` (``a > 0``) generalises
the Riemann zeta (``zeta(s) = zeta(s, 1)``). This module gives two honestly
labelled registers, both built on the verified substrate:

* **exact negative-integer values** (:func:`hurwitz_zeta_negative_integer`):
  ``zeta(-n, a) = -B_{n+1}(a)/(n+1)`` for integer ``n >= 0`` and rational ``a`` --
  a **closed-form rational**, read off the exact Bernoulli polynomial
  (:func:`~omnibias.core.verified.coeffs.bernoulli_polynomial_exact`);
* a **numerical** Euler-Maclaurin continuation (:func:`hurwitz_zeta`) valid for
  complex ``s`` with ``Re(s) > -(2 order + 1)``, ``s != 1``, mirroring
  :func:`omnibias.core.verified.dirichlet.zeta_euler_maclaurin` with the shift
  ``a`` (DLMF 25.11.5) and a rigorous DLMF-25.11.5 remainder.

Honesty / scope
---------------
The continuation is a verified enclosure of the analytically-continued *value*;
no statement is made about zeros of ``zeta(s, a)``. See ``dirichlet.py`` for the
matching honesty note on the Riemann Hypothesis (an external obligation, never
inferred).
"""

from __future__ import annotations

import math
from fractions import Fraction

from omnibias.core.verified.coeffs import bernoulli_number_exact, bernoulli_polynomial_exact
from omnibias.core.verified.complex_interval import ComplexInterval, ComplexLike
from omnibias.core.verified.dirichlet import complex_exp
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.transcend import ln_iv


def hurwitz_zeta_negative_integer(n: int, a: Fraction | int) -> Interval:
    r"""Exact ``zeta(-n, a) = -B_{n+1}(a)/(n+1)`` as a certified interval (``n >= 0``).

    A **closed-form rational** for rational ``a``, read off the exact Bernoulli
    polynomial. ``zeta(0, a) = 1/2 - a``; ``zeta(-1, a) = -B_2(a)/2``. With
    ``a = 1`` this is ``zeta(-n) = -B_{n+1}/(n+1)`` (matching
    :func:`~omnibias.core.verified.dirichlet.zeta_negative_odd` on odd negatives).
    """
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    return Interval.from_rational(-bernoulli_polynomial_exact(n + 1, a) / Fraction(n + 1))


def _pos_base_power_neg_s(base: Interval, s: ComplexInterval) -> ComplexInterval:
    r"""Enclosure of ``base^{-s} = exp(-s ln base)`` for a strictly positive ``base``."""
    ln_b = ln_iv(base)
    return complex_exp(ComplexInterval(-s.re * ln_b, -s.im * ln_b))


def _rising_factorial_ci(s: ComplexInterval, count: int) -> ComplexInterval:
    r"""Pochhammer ``(s)_count = s (s+1) ... (s + count - 1)`` as a complex interval."""
    prod = ComplexInterval.one()
    for j in range(count):
        prod = prod * (s + ComplexInterval.from_parts(Interval.point(float(j))))
    return prod


def hurwitz_zeta(
    s: ComplexLike, a: float, *, num_sum_terms: int = 20, order: int = 6
) -> ComplexInterval:
    r"""Numerical Euler-Maclaurin enclosure of ``zeta(s, a)`` for ``a > 0``.

    With ``N = num_sum_terms``, ``M = N + a``, and ``n = order`` correction terms
    (DLMF 25.11.5):

    .. math::

        \zeta(s, a) = \sum_{k=0}^{N-1} (k+a)^{-s} + \frac{M^{1-s}}{s-1}
            + \tfrac12 M^{-s}
            + \sum_{k=1}^{n} \frac{B_{2k}}{(2k)!}\,(s)_{2k-1}\,M^{-s-2k+1} + R_n,

    with the rigorous remainder bounded by
    ``|(s+2n+1)/(sigma+2n+1)| |B_{2n+2}/(2n+2)! (s)_{2n+1} M^{-s-2n-1}|``.
    Valid (and enclosed) for ``Re(s) > -(2n+1)`` and ``s != 1``.
    """
    if a <= 0.0:
        raise ValueError(f"hurwitz_zeta requires a > 0, got a={a}")
    if num_sum_terms < 1:
        raise ValueError(f"num_sum_terms must be >= 1, got {num_sum_terms}")
    if order < 1:
        raise ValueError(f"order must be >= 1, got {order}")
    s_ci = ComplexInterval.from_value(s)
    denom_re_lo = s_ci.re.lo + (2 * order + 1)
    if denom_re_lo <= 0.0:
        raise ValueError(
            f"Hurwitz Euler-Maclaurin needs Re(s) > -(2*order+1) = {-(2 * order + 1)}; "
            f"got Re(s).lo={s_ci.re.lo!r}"
        )
    s_minus_1 = s_ci - ComplexInterval.one()
    if s_minus_1.modulus().lo <= 0.0:
        raise ValueError("hurwitz_zeta: s-1 straddles the pole at s = 1")

    # Partial sum sum_{k=0}^{N-1} (k+a)^{-s}.
    total = _pos_base_power_neg_s(Interval.point(a), s_ci)
    for k in range(1, num_sum_terms):
        total = total + _pos_base_power_neg_s(Interval.point(a + k), s_ci)

    m = a + float(num_sum_terms)
    m_iv = Interval.point(m)
    m_pow_neg_s = _pos_base_power_neg_s(m_iv, s_ci)
    # M^{1-s} = M * M^{-s}.
    m_pow_1_minus_s = ComplexInterval.from_parts(m_iv) * m_pow_neg_s
    total = total + m_pow_1_minus_s / s_minus_1
    # Partial sum stops at k = N-1, so the Euler-Maclaurin boundary half-term is ADDED:
    # sum_{k=N}^inf (k+a)^{-s} = M^{1-s}/(s-1) + (1/2) M^{-s} + corrections + R.
    total = total + ComplexInterval.from_parts(Interval.point(0.5)) * m_pow_neg_s

    for k in range(1, order + 1):
        b_over_fac = Interval.from_rational(
            bernoulli_number_exact(2 * k) / Fraction(math.factorial(2 * k))
        )
        poch = _rising_factorial_ci(s_ci, 2 * k - 1)
        shifted = s_ci + ComplexInterval.from_parts(Interval.point(float(2 * k - 1)))
        m_term = _pos_base_power_neg_s(m_iv, shifted)  # M^{-(s+2k-1)}
        total = total + ComplexInterval.from_value(b_over_fac) * poch * m_term

    b_rem = abs(bernoulli_number_exact(2 * order + 2) / Fraction(math.factorial(2 * order + 2)))
    poch_rem = _rising_factorial_ci(s_ci, 2 * order + 1)
    shifted_rem = s_ci + ComplexInterval.from_parts(Interval.point(float(2 * order + 1)))
    m_rem = _pos_base_power_neg_s(m_iv, shifted_rem)  # M^{-(s+2n+1)}
    num_factor = (s_ci + ComplexInterval.from_parts(Interval.point(float(2 * order + 1)))).mag
    factor_bound = num_factor / denom_re_lo
    r_bound = float(Interval.from_rational(b_rem).hi) * poch_rem.mag * m_rem.mag * factor_bound
    remainder = ComplexInterval.from_parts(Interval(-r_bound, r_bound), Interval(-r_bound, r_bound))
    return total + remainder


__all__ = [
    "hurwitz_zeta",
    "hurwitz_zeta_negative_integer",
]
