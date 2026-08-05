# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Certified polylogarithm and Lerch transcendent enclosures (``|z| < 1``).

The polylogarithm ``Li_s(z) = sum_{k>=1} z^k / k^s`` and the Lerch transcendent
``Phi(z, s, a) = sum_{k>=0} z^k / (k + a)^s`` are the workhorse special functions
of analytic combinatorics and statistical mechanics. In the disc of absolute
convergence ``|z| < 1`` both are enclosed here as a term-by-term
:class:`~omnibias.core.verified.complex_interval.ComplexInterval` partial sum plus
a rigorous geometric tail: past the retained terms the consecutive-ratio
magnitude is bounded a-priori by

.. math::

    q = |z| \cdot \Bigl(1 + \tfrac1N\Bigr)^{\max(0,\,-\Re s)} < 1

(the ``k^{-s}`` factor grows at most like ``(1 + 1/k)^{|\Re s|}`` when
``Re(s) < 0``, and shrinks otherwise), so
:func:`~omnibias.core.verified.series.geometric_tail_enclosure` bounds the omitted
tail. **Numerical** (verified enclosure); the coefficients and the certified
``pi``/base-point enclosures are the only inexactness.

Scope: this is the ``|z| < 1`` series register only; analytic continuation past
the unit circle (the inversion / Landen relations) is out of scope here.
"""

from __future__ import annotations

from fractions import Fraction

from omnibias.core.verified.complex_interval import ComplexInterval, ComplexLike
from omnibias.core.verified.dirichlet import n_power_neg_s
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.series import geometric_tail_enclosure
from omnibias.core.verified.transcend import exp_iv, ln_iv


def _pos_base_power_neg_s(base: Interval, s: ComplexInterval) -> ComplexInterval:
    r"""Enclosure of ``base^{-s} = exp(-s ln base)`` for a strictly positive ``base``."""
    from omnibias.core.verified.dirichlet import complex_exp

    ln_b = ln_iv(base)
    return complex_exp(ComplexInterval(-s.re * ln_b, -s.im * ln_b))


def _complex_square(bound: float) -> ComplexInterval:
    r"""The axis-aligned square ``[-B, B] + i[-B, B]`` containing the disc ``|z| <= B``."""
    return ComplexInterval.from_parts(Interval(-bound, bound), Interval(-bound, bound))


def polylog_enclosure(s: ComplexLike, z: ComplexLike, *, num_terms: int = 80) -> ComplexInterval:
    r"""Rigorous enclosure of the polylogarithm ``Li_s(z) = sum_{k>=1} z^k / k^s``.

    Requires ``|z| < 1`` and a ratio bound ``q < 1`` (raise otherwise -- increase
    ``num_terms`` or reduce ``|z|`` / ``|Re(s)|``). ``Li_1(z) = -ln(1 - z)`` and
    ``Li_2`` is the dilogarithm; verified against ``mpmath.polylog``.
    """
    if num_terms < 1:
        raise ValueError(f"num_terms must be >= 1, got {num_terms}")
    z_ci = ComplexInterval.from_value(z)
    s_ci = ComplexInterval.from_value(s)
    zmag = z_ci.mag
    if zmag >= 1.0:
        raise ValueError(f"polylog_enclosure requires |z| < 1, got |z| ~ {zmag!r}")

    sigma_lo = s_ci.re.lo
    if sigma_lo >= 0.0:
        growth = Interval.point(1.0)
    else:
        base = Interval.point(1.0) + Interval.from_rational(Fraction(1, num_terms))
        growth = exp_iv(Interval.point(-sigma_lo) * ln_iv(base))
    q = (Interval.point(zmag) * growth).hi
    if q >= 1.0:
        raise ValueError(
            f"polylog ratio bound q={q!r} is not < 1; increase num_terms or reduce |z|/|Re(s)|"
        )

    total = ComplexInterval.zero()
    z_pow = ComplexInterval.one()
    last = ComplexInterval.zero()
    for k in range(1, num_terms + 1):
        z_pow = z_pow * z_ci  # z^k
        term = z_pow * n_power_neg_s(k, s_ci)  # z^k k^{-s}
        total = total + term
        last = term
    tail_b = geometric_tail_enclosure(Interval.point(last.mag), q).hi
    return total + _complex_square(tail_b)


def lerch_transcendent(
    z: ComplexLike, s: ComplexLike, a: float, *, num_terms: int = 80
) -> ComplexInterval:
    r"""Rigorous enclosure of the Lerch transcendent ``Phi(z, s, a) = sum_{k>=0} z^k/(k+a)^s``.

    Requires ``a > 0``, ``|z| < 1``, and a ratio bound ``q < 1``.
    ``Phi(z, s, 1) = Li_s(z)/z`` and ``Phi(1, s, a) = zeta(s, a)`` (on the boundary,
    out of scope here). Verified against ``mpmath.lerchphi``.
    """
    if num_terms < 1:
        raise ValueError(f"num_terms must be >= 1, got {num_terms}")
    if a <= 0.0:
        raise ValueError(f"lerch_transcendent requires a > 0, got a={a}")
    z_ci = ComplexInterval.from_value(z)
    s_ci = ComplexInterval.from_value(s)
    zmag = z_ci.mag
    if zmag >= 1.0:
        raise ValueError(f"lerch_transcendent requires |z| < 1, got |z| ~ {zmag!r}")

    sigma_lo = s_ci.re.lo
    if sigma_lo >= 0.0:
        growth = Interval.point(1.0)
    else:
        ratio_step = Interval.point(a + num_terms) / Interval.point(a + num_terms - 1.0)
        growth = exp_iv(Interval.point(-sigma_lo) * ln_iv(ratio_step))
    q = (Interval.point(zmag) * growth).hi
    if q >= 1.0:
        raise ValueError(
            f"lerch ratio bound q={q!r} is not < 1; increase num_terms or reduce |z|/|Re(s)|"
        )

    total = ComplexInterval.zero()
    z_pow = ComplexInterval.one()
    last = ComplexInterval.zero()
    for k in range(num_terms):
        term = z_pow * _pos_base_power_neg_s(Interval.point(a + k), s_ci)  # z^k (a+k)^{-s}
        total = total + term
        last = term
        z_pow = z_pow * z_ci
    tail_b = geometric_tail_enclosure(Interval.point(last.mag), q).hi
    return total + _complex_square(tail_b)


__all__ = [
    "lerch_transcendent",
    "polylog_enclosure",
]
