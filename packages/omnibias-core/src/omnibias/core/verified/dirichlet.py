# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Certified analytic-number-theory enclosures -- Dirichlet series on ``Re(s) > 1``.

A thin, pure-Python slice of analytic number theory built on the verified
substrate (:mod:`~omnibias.core.verified.interval`,
:mod:`~omnibias.core.verified.complex_interval`,
:mod:`~omnibias.core.verified.transcend`, :mod:`~omnibias.core.verified.series`).
Every routine returns a **rigorous enclosure** -- an outward-rounded interval that
*provably contains* the true value -- of a Dirichlet series

.. math::

    D(s) = \sum_{n \ge 1} a_n\, n^{-s}, \qquad n^{-s} = e^{-s \ln n},

in the region of absolute convergence ``Re(s) > 1``. The partial sum over the
retained terms is enclosed term-by-term in :class:`ComplexInterval` arithmetic;
the omitted tail is bounded by a **caller-proved (or, for zeta / L, a
function-supplied) majorant**:

* :func:`p_series_tail_bound` -- the integral-test bound
  ``sum_{n>N} n^{-sigma} <= N^{1-sigma}/(sigma-1)`` (``sigma > 1``), the natural
  majorant for `n^{-s}` coefficients bounded by ``1``;
* :func:`certified_dirichlet_series` -- the general contract mirroring
  :func:`omnibias.core.verified.series.certified_geometric_series_sum`: the caller
  supplies a rigorous absolute bound on the omitted-tail magnitude.

The Dirichlet-*series* routines are **verified-interval numerics with a mandatory
convergence majorant**, valid only on ``Re(s) > 1``. Two honest extensions past
that wall are provided and clearly labelled:

* **exact special values** -- ``zeta(2m)``, ``zeta(1-2m) = -B_{2m}/(2m)``, and
  ``beta(2m+1)`` (:func:`zeta_even`, :func:`zeta_negative_odd`,
  :func:`dirichlet_beta_odd`) are **closed-form** rational (multiples of a certified
  power of ``pi``), read off the Bernoulli / Euler numbers;
* an **attempted critical-strip enclosure** (:func:`zeta_euler_maclaurin`) is a
  **numerical** verified enclosure of the analytically-continued value via the
  Euler-Maclaurin formula (DLMF 25.2.3) with a rigorous remainder.

Honesty / scope
---------------
The critical strip is entered only as a *numerical enclosure of a value*; the
functional equation and, above all, the **Riemann Hypothesis** remain a recorded
**external proof obligation**, never inferred here. A tight enclosure straddling
``0`` at a candidate point is *not* a statement that ``Re(s) = 1/2``. Nothing in
this module makes a statement about zeros of ``zeta`` or ``L`` functions,
primality, factoring, or any cryptographic hardness assumption; see
``docs/scope-and-guarantees.md`` s6.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from fractions import Fraction

from omnibias.core.verified.coeffs import (
    bernoulli_number_exact,
    euler_number_exact,
    generalized_bernoulli_exact,
)
from omnibias.core.verified.complex_interval import ComplexInterval, ComplexLike
from omnibias.core.verified.interval import Interval, IntervalLike
from omnibias.core.verified.series import geometric_tail_enclosure
from omnibias.core.verified.transcend import PI_IV, cos_iv, exp_iv, ln_iv, sin_iv


def complex_exp(z: ComplexInterval) -> ComplexInterval:
    r"""Rigorous enclosure of ``exp(z)`` for a complex interval ``z = a + i b``.

    ``exp(a + i b) = e^{a}(\cos b + i \sin b)``, each factor an outward-rounded
    real interval (:func:`exp_iv` / :func:`cos_iv` / :func:`sin_iv`).
    """
    ea = exp_iv(z.re)
    return ComplexInterval(ea * cos_iv(z.im), ea * sin_iv(z.im))


def n_power_neg_s(n: int, s: ComplexLike) -> ComplexInterval:
    r"""Rigorous enclosure of ``n^{-s} = e^{-s \ln n}`` for integer ``n >= 1``.

    ``|n^{-s}| = n^{-Re(s)}``; the phase is ``e^{-i Im(s) \ln n}``. ``n = 1`` gives
    exactly ``1`` (``ln 1 = 0``).
    """
    if n < 1:
        raise ValueError(f"n must be a positive integer, got {n}")
    s_ci = ComplexInterval.from_value(s)
    ln_n = ln_iv(Interval.point(float(n)))
    # -s ln n = (-Re(s) ln n) + i(-Im(s) ln n)
    exponent = ComplexInterval(-s_ci.re * ln_n, -s_ci.im * ln_n)
    return complex_exp(exponent)


def p_series_tail_bound(n_terms: int, sigma: IntervalLike) -> Interval:
    r"""Integral-test tail bound ``sum_{n>N} n^{-sigma} <= N^{1-sigma}/(sigma-1)``.

    ``N = n_terms`` is the number of retained terms; the bound is evaluated at the
    **smallest** admissible ``sigma`` (``Interval.from_value(sigma).lo``) so it is a
    valid upper bound over the whole real-part range. Requires ``sigma.lo > 1``.
    The returned interval's ``.hi`` is the certified magnitude bound.
    """
    if n_terms < 1:
        raise ValueError(f"n_terms must be >= 1, got {n_terms}")
    sigma_lo = Interval.from_value(sigma).lo
    if sigma_lo <= 1.0:
        raise ValueError(
            f"p-series tail requires Re(s) > 1; got sigma.lo={sigma_lo!r}"
        )
    ln_n = ln_iv(Interval.point(float(n_terms)))
    # N^{1-sigma_lo} = exp((1 - sigma_lo) ln N), with 1 - sigma_lo < 0.
    numerator = exp_iv(Interval.point(1.0 - sigma_lo) * ln_n)
    denom = Interval.point(sigma_lo - 1.0)
    return numerator * denom.reciprocal()


def certified_dirichlet_series(
    terms: Sequence[ComplexLike], tail_bound: IntervalLike
) -> ComplexInterval:
    r"""Enclosure of ``sum_{n>=1} a_n`` from retained terms plus a tail magnitude.

    ``terms`` are enclosures of the retained coefficients ``a_1, ..., a_N`` (each an
    already-formed ``a_n n^{-s}`` value); ``tail_bound`` is a rigorous **absolute
    upper bound** on the magnitude of the omitted tail
    ``|sum_{n>N} a_n|``. The tail is enclosed by the axis-aligned square
    ``[-B, B] + i[-B, B]`` that contains the disc ``|z| <= B``.

    The caller owns the correctness of ``tail_bound`` (e.g. via
    :func:`p_series_tail_bound` or
    :func:`omnibias.core.verified.series.geometric_tail_enclosure`), exactly as
    :func:`~omnibias.core.verified.series.certified_geometric_series_sum` owns its
    ``ratio``.
    """
    ivs = [ComplexInterval.from_value(t) for t in terms]
    if not ivs:
        raise ValueError("certified_dirichlet_series needs >= 1 retained term")
    partial = ivs[0]
    for term in ivs[1:]:
        partial = partial + term
    b = Interval.from_value(tail_bound).abs().hi
    tail = ComplexInterval.from_parts(Interval(-b, b), Interval(-b, b))
    return partial + tail


def zeta_enclosure(s: ComplexLike, *, num_terms: int = 1000) -> ComplexInterval:
    r"""Rigorous enclosure of the Riemann zeta function ``zeta(s)`` on ``Re(s) > 1``.

    ``zeta(s) = sum_{n>=1} n^{-s}``. Returns the enclosure of the first
    ``num_terms`` terms plus the integral-test ``p``-series tail
    (:func:`p_series_tail_bound`). Requires ``Re(s) > 1``; raises otherwise.
    """
    s_ci = ComplexInterval.from_value(s)
    if s_ci.re.lo <= 1.0:
        raise ValueError(
            f"zeta_enclosure requires Re(s) > 1 (absolute convergence); "
            f"got Re(s).lo={s_ci.re.lo!r}"
        )
    terms = [n_power_neg_s(n, s_ci) for n in range(1, num_terms + 1)]
    tail_b = p_series_tail_bound(num_terms, s_ci.re).hi
    return certified_dirichlet_series(terms, tail_b)


def l_function_enclosure(
    character: Sequence[ComplexLike], s: ComplexLike, *, num_terms: int = 1000
) -> ComplexInterval:
    r"""Rigorous enclosure of a Dirichlet ``L``-function ``L(chi, s)`` on ``Re(s) > 1``.

    ``L(chi, s) = sum_{n>=1} chi(n) n^{-s}`` with ``chi`` supplied as one period of
    values ``character = [chi(0), chi(1), ..., chi(q-1)]`` (so ``chi(n) =
    character[n mod q]``; for a genuine Dirichlet character ``chi(0) = 0``). The
    tail uses ``|chi(n)| <= max_r |chi(r)|`` times the ``p``-series bound, so it is
    valid for any bounded periodic coefficient sequence. Requires ``Re(s) > 1``.
    """
    if len(character) == 0:
        raise ValueError("character must have period >= 1")
    s_ci = ComplexInterval.from_value(s)
    if s_ci.re.lo <= 1.0:
        raise ValueError(
            f"l_function_enclosure requires Re(s) > 1; got Re(s).lo={s_ci.re.lo!r}"
        )
    period = len(character)
    chi = [ComplexInterval.from_value(c) for c in character]
    terms = [chi[n % period] * n_power_neg_s(n, s_ci) for n in range(1, num_terms + 1)]
    chi_max = max(c.mag for c in chi)
    tail_b = (Interval.point(chi_max) * p_series_tail_bound(num_terms, s_ci.re)).hi
    return certified_dirichlet_series(terms, tail_b)


def zeta_even(m: int) -> Interval:
    r"""Exact special value ``zeta(2m)`` as a certified interval (``m >= 1``).

    .. math::

        \zeta(2m) = (-1)^{m+1}\,\frac{B_{2m}\,(2\pi)^{2m}}{2\,(2m)!}
                  = \Bigl[(-1)^{m+1}\frac{B_{2m}\,2^{2m}}{2\,(2m)!}\Bigr]\,\pi^{2m},

    an exact rational multiple of ``pi^{2m}`` (:data:`PI_IV`). **Closed-form**: the
    only inexactness is the certified enclosure of ``pi``. E.g. ``zeta(2)=pi^2/6``,
    ``zeta(4)=pi^4/90``.
    """
    if m < 1:
        raise ValueError(f"m must be >= 1 (zeta(2m) for positive even 2m), got {m}")
    coeff = (
        Fraction((-1) ** (m + 1))
        * bernoulli_number_exact(2 * m)
        * Fraction(2 ** (2 * m))
        / Fraction(2 * math.factorial(2 * m))
    )
    return Interval.from_rational(coeff) * PI_IV.pow_int(2 * m)


def zeta_negative_odd(m: int) -> Interval:
    r"""Exact special value ``zeta(1 - 2m) = -B_{2m}/(2m)`` (``m >= 1``).

    A **closed-form rational** (no ``pi``): ``zeta(-1) = -1/12``,
    ``zeta(-3) = 1/120``, ``zeta(-5) = -1/252``. Returned as the tightest interval
    enclosing the exact rational.
    """
    if m < 1:
        raise ValueError(f"m must be >= 1 (zeta(1-2m) for negative odd 1-2m), got {m}")
    return Interval.from_rational(-bernoulli_number_exact(2 * m) / Fraction(2 * m))


def dirichlet_beta_odd(m: int) -> Interval:
    r"""Exact Dirichlet ``beta(2m+1)`` as a certified interval (``m >= 0``).

    .. math::

        \beta(2m+1) = (-1)^{m}\,\frac{E_{2m}\,(\pi/2)^{2m+1}}{2\,(2m)!}
                    = \Bigl[(-1)^{m}\frac{E_{2m}}{2^{2m+2}\,(2m)!}\Bigr]\,\pi^{2m+1},

    an exact rational multiple of ``pi^{2m+1}`` (:data:`PI_IV`) via the Euler
    numbers ``E_{2m}`` (:func:`~omnibias.core.verified.coeffs.euler_number_exact`).
    **Closed-form**. E.g. ``beta(1)=pi/4``, ``beta(3)=pi^3/32``.
    """
    if m < 0:
        raise ValueError(f"m must be >= 0, got {m}")
    coeff = (
        Fraction((-1) ** m * euler_number_exact(2 * m))
        / Fraction(2 ** (2 * m + 2) * math.factorial(2 * m))
    )
    return Interval.from_rational(coeff) * PI_IV.pow_int(2 * m + 1)


def dirichlet_l_negative_integer(n: int, character: Sequence[int | Fraction]) -> Interval:
    r"""Exact ``L(1-n, chi) = -B_{n,chi}/n`` for a **real** character (``n >= 1``).

    A **closed-form rational** read off the generalized Bernoulli number
    :func:`~omnibias.core.verified.coeffs.generalized_bernoulli_exact`. ``character``
    is one period ``[chi(0), ..., chi(q-1)]`` of a real Dirichlet character. For the
    non-principal character mod 4 this gives ``L(0) = 1/2`` and
    ``L(-2m) = E_{2m}/2`` (Euler numbers); the numerical ``Re(s) > 1`` values are
    :func:`l_function_enclosure`.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1 (L(1-n, chi)), got {n}")
    return Interval.from_rational(-generalized_bernoulli_exact(n, character) / Fraction(n))


def _rising_factorial_ci(s: ComplexInterval, count: int) -> ComplexInterval:
    r"""Pochhammer ``(s)_count = s (s+1) ... (s + count - 1)`` as a complex interval."""
    prod = ComplexInterval.one()
    for j in range(count):
        prod = prod * (s + ComplexInterval.from_parts(Interval.point(float(j))))
    return prod


def zeta_euler_maclaurin(
    s: ComplexLike, *, num_sum_terms: int = 20, order: int = 6
) -> ComplexInterval:
    r"""**Attempted** critical-strip enclosure of ``zeta(s)`` via Euler-Maclaurin.

    Extends the ``Re(s) > 1`` wall of :func:`zeta_enclosure` using the
    Euler-Maclaurin continuation (DLMF 25.2.3): with ``N = num_sum_terms`` and
    ``n = order`` correction terms,

    .. math::

        \zeta(s) = \sum_{k=1}^{N} k^{-s} + \frac{N^{1-s}}{s-1} - \tfrac12 N^{-s}
        + \sum_{k=1}^{n} \frac{B_{2k}}{(2k)!}\,(s)_{2k-1}\,N^{-s-2k+1} + R_{n},

    valid (and here rigorously enclosed) for ``Re(s) > -(2n+1)`` and ``s != 1``.
    The remainder is bounded by DLMF 25.2.4,

    .. math::

        |R_n| \le \Bigl|\tfrac{s+2n+1}{\sigma+2n+1}\Bigr|\,
                  \Bigl|\tfrac{B_{2n+2}}{(2n+2)!}\,(s)_{2n+1}\,N^{-s-2n-1}\Bigr|,

    enclosed as the axis-aligned square containing that disc.

    Honesty / scope
    ---------------
    This is a **numerical** verified enclosure of the analytically-continued value,
    *not* a statement about the zeros of ``zeta``. The Riemann Hypothesis remains an
    **external obligation, never inferred** here; a tight enclosure straddling ``0``
    at a candidate point says nothing about whether ``Re(s) = 1/2``.
    """
    if num_sum_terms < 1:
        raise ValueError(f"num_sum_terms must be >= 1, got {num_sum_terms}")
    if order < 1:
        raise ValueError(f"order must be >= 1, got {order}")
    s_ci = ComplexInterval.from_value(s)
    # Remainder denominator sigma + 2n + 1 must be provably positive.
    denom_re_lo = s_ci.re.lo + (2 * order + 1)
    if denom_re_lo <= 0.0:
        raise ValueError(
            f"Euler-Maclaurin continuation needs Re(s) > -(2*order+1) = "
            f"{-(2 * order + 1)}; got Re(s).lo={s_ci.re.lo!r}"
        )
    # s - 1 must be bounded away from the pole at s = 1.
    s_minus_1 = s_ci - ComplexInterval.one()
    if s_minus_1.modulus().lo <= 0.0:
        raise ValueError("zeta_euler_maclaurin: s-1 straddles the pole at s = 1")

    n_big = float(num_sum_terms)
    # Partial sum sum_{k=1}^{N} k^{-s}.
    total = n_power_neg_s(1, s_ci)
    for k in range(2, num_sum_terms + 1):
        total = total + n_power_neg_s(k, s_ci)
    # N^{1-s}/(s-1) - (1/2) N^{-s}.
    n_pow_neg_s = n_power_neg_s(num_sum_terms, s_ci)
    n_pow_1_minus_s = ComplexInterval.from_parts(Interval.point(n_big)) * n_pow_neg_s
    total = total + n_pow_1_minus_s / s_minus_1
    total = total - ComplexInterval.from_parts(Interval.point(0.5)) * n_pow_neg_s

    # Correction terms sum_{k=1}^{n} B_{2k}/(2k)! (s)_{2k-1} N^{-s-2k+1}.
    for k in range(1, order + 1):
        b_over_fac = Interval.from_rational(
            bernoulli_number_exact(2 * k) / Fraction(math.factorial(2 * k))
        )
        poch = _rising_factorial_ci(s_ci, 2 * k - 1)
        shifted = s_ci + ComplexInterval.from_parts(Interval.point(float(2 * k - 1)))
        n_term = n_power_neg_s(num_sum_terms, shifted)  # N^{-(s+2k-1)}
        total = total + ComplexInterval.from_value(b_over_fac) * poch * n_term

    # Rigorous remainder bound (DLMF 25.2.4).
    b_rem = abs(bernoulli_number_exact(2 * order + 2) / Fraction(math.factorial(2 * order + 2)))
    poch_rem = _rising_factorial_ci(s_ci, 2 * order + 1)
    shifted_rem = s_ci + ComplexInterval.from_parts(Interval.point(float(2 * order + 1)))
    n_rem = n_power_neg_s(num_sum_terms, shifted_rem)  # N^{-(s+2n+1)}
    num_factor = (s_ci + ComplexInterval.from_parts(Interval.point(float(2 * order + 1)))).mag
    factor_bound = num_factor / denom_re_lo  # |(s+2n+1)/(sigma+2n+1)| upper bound
    r_bound = float(Interval.from_rational(b_rem).hi) * poch_rem.mag * n_rem.mag * factor_bound
    remainder = ComplexInterval.from_parts(Interval(-r_bound, r_bound), Interval(-r_bound, r_bound))
    return total + remainder


def theta_enclosure(u: IntervalLike, t: float, *, num_terms: int = 50) -> Interval:
    r"""Rigorous enclosure of the Jacobi theta / heat kernel on the circle.

    .. math::

        \vartheta(u; t) = \sum_{n \in \mathbb Z} e^{-t n^2} e^{i n u}
        = 1 + 2 \sum_{n \ge 1} e^{-t n^2} \cos(n u), \qquad t > 0,

    a **real**, strictly positive function. The retained terms are enclosed
    directly; the omitted tail is bounded by the majorant ``b_n = 2 e^{-t n^2}``
    (whose consecutive ratio ``e^{-t(2n+1)} < 1`` gives a geometric tail via
    :func:`~omnibias.core.verified.series.geometric_tail_enclosure`), which
    dominates ``|2 e^{-t n^2} \cos(n u)|`` regardless of the cosine sign. This is
    the public form of the ``U(1)`` heat-kernel pattern used in lattice
    transfer-operator bounds.
    """
    if t <= 0.0:
        raise ValueError(f"theta_enclosure requires t > 0, got {t}")
    if num_terms < 1:
        raise ValueError(f"num_terms must be >= 1, got {num_terms}")
    u_iv = Interval.from_value(u)
    total = Interval.point(1.0)
    last_majorant = Interval.point(2.0)
    for n in range(1, num_terms + 1):
        decay = exp_iv(Interval.point(-t * float(n) * float(n)))
        term = Interval.point(2.0) * decay * cos_iv(Interval.point(float(n)) * u_iv)
        total = total + term
        last_majorant = Interval.point(2.0) * decay
    # Majorant ratio for the omitted terms n > N: b_{n+1}/b_n = e^{-t(2n+1)},
    # maximised at n = N, so q = e^{-t(2N+1)} < 1 bounds every omitted ratio.
    ratio = exp_iv(Interval.point(-t * (2.0 * float(num_terms) + 1.0)))
    tail = geometric_tail_enclosure(last_majorant, ratio)
    return total + tail


__all__ = [
    "certified_dirichlet_series",
    "complex_exp",
    "dirichlet_beta_odd",
    "dirichlet_l_negative_integer",
    "l_function_enclosure",
    "n_power_neg_s",
    "p_series_tail_bound",
    "theta_enclosure",
    "zeta_enclosure",
    "zeta_euler_maclaurin",
    "zeta_even",
    "zeta_negative_odd",
]
