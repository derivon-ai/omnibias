# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Approximate functional-equation main terms with an enclosed residual.

Hardy--Littlewood / Titchmarsh §4.13: for ``s = σ + it`` with ``t ≠ 0``,

.. math::

    \zeta(s) = \sum_{n \le x} n^{-s} + \chi(s)\sum_{n \le y} n^{s-1} + R(s;x,y)

when ``2 π x y = |t|``. An unspecified classical O-constant is insufficient
for a rigorous error bound. This implementation obtains an independent
Euler-Maclaurin enclosure ``Z`` of zeta on the whole input rectangle and
encloses the residual ``Z - main`` using interval subtraction. Its coordinate
magnitude bounds a symmetric remainder square. ``remainder_factor >= 1``
only inflates that already certified square; it cannot shrink it.

This is a finite evaluator, not an independent fast Riemann-Siegel remainder
theorem. It incurs the cost of Euler-Maclaurin in addition to the main sums.
If interval dependency prevents enclosing the chi factor, it returns the
Euler-Maclaurin reference directly.
The entire input rectangle must lie in ``T_MIN <= |t| <= T_MAX`` and
``SIGMA_LO <= σ <= SIGMA_HI``. Outside this compact the evaluator refuses.

Optional ``hardy_z(t)`` encloses ``e^{i θ(t)} ζ(1/2 + i t)`` with the
Riemann--Siegel theta from ``ln Gamma(1/4 + i t/2)``. Neither routine infers
zeros or the Riemann Hypothesis.
"""

from __future__ import annotations

import math

from omnibias.core.verified.complex_interval import ComplexInterval, ComplexLike
from omnibias.core.verified.dirichlet import n_power_neg_s, zeta_euler_maclaurin
from omnibias.core.verified.gamma_complex import exp_ci, log_gamma_ci
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.transcend import PI_IV, ln_iv
from omnibias.core.verified.xi import chi_factor, continuation_honesty

#: Supported compact for this finite evaluator.
T_MIN = 2.0 * math.pi
T_MAX = 40.0
SIGMA_LO = 0.1
SIGMA_HI = 0.9
#: Optional inflation of the independently enclosed residual (must be >= 1).
REMAINDER_FACTOR = 4.0


def _as_ci(s: ComplexLike) -> ComplexInterval:
    return ComplexInterval.from_value(s)


def _require_compact(s: ComplexInterval) -> None:
    if not all(math.isfinite(x) for x in (s.re.lo, s.re.hi, s.im.lo, s.im.hi)):
        raise ValueError("AFE requires finite rectangle endpoints")
    # Absolute value of a finite binary endpoint is exact. Using inflated
    # magnitude helpers here would incorrectly refuse the closed boundary.
    t_max = max(abs(s.im.lo), abs(s.im.hi))
    t_min = 0.0 if s.im.contains_zero() else min(abs(s.im.lo), abs(s.im.hi))
    if t_min < T_MIN or t_max > T_MAX:
        raise ValueError(
            f"AFE requires T_MIN={T_MIN} <= |Im(s)| <= T_MAX={T_MAX} "
            f"throughout the rectangle; got [{t_min}, {t_max}]"
        )
    if s.re.lo < SIGMA_LO or s.re.hi > SIGMA_HI:
        raise ValueError(
            f"AFE requires {SIGMA_LO} <= Re(s) <= {SIGMA_HI}; got [{s.re.lo!r}, {s.re.hi!r}]"
        )


def _cutoff(t_mag: float) -> tuple[int, Interval]:
    """``x = y = sqrt(|t| / (2 π))``; return ``floor(x.lo)`` and the ``x`` enclosure."""
    x_iv = (Interval.point(t_mag) / (PI_IV * 2.0)).sqrt()
    n_max = max(1, int(math.floor(x_iv.lo)))
    return n_max, x_iv


def zeta_approximate_functional_equation(
    s: ComplexLike,
    *,
    remainder_factor: float = REMAINDER_FACTOR,
) -> ComplexInterval:
    """Enclose zeta using AFE main terms and an Euler-Maclaurin residual.

    The radius encloses every coordinate of ``Z - main``, where ``Z`` is
    an independent enclosure on the same entire rectangle. Therefore
    ``main + [-radius,radius]^2`` contains ``Z``. A finite inflation factor
    at least one preserves this inclusion; no guessed O-constant is used.
    """
    if not math.isfinite(remainder_factor) or remainder_factor < 1.0:
        raise ValueError("remainder_factor must be finite and at least 1")
    s_ci = _as_ci(s)
    _require_compact(s_ci)
    reference = zeta_euler_maclaurin(s_ci, num_sum_terms=30, order=6)
    n_max, x_iv = _cutoff(s_ci.im.mag)
    if x_iv.lo <= 0.0:
        raise ValueError("AFE cutoff x must be strictly positive")
    one = ComplexInterval.one()
    main = n_power_neg_s(1, s_ci)
    second = n_power_neg_s(1, one - s_ci)
    for n in range(2, n_max + 1):
        main = main + n_power_neg_s(n, s_ci)
        second = second + n_power_neg_s(n, one - s_ci)
    try:
        chi = chi_factor(s_ci)
    except (ValueError, ZeroDivisionError, OverflowError):
        # Wide Gamma rectangles can meet zero in an intermediate denominator.
        # The independent reference still encloses zeta on the entire box.
        return reference
    main = main + chi * second
    residual = reference - main
    radius = (
        Interval.point(remainder_factor)
        * Interval.point(max(residual.re.mag, residual.im.mag))
    ).hi
    remainder = ComplexInterval(Interval(-radius, radius), Interval(-radius, radius))
    return main + remainder


def riemann_siegel_theta(t: float) -> Interval:
    r"""Riemann--Siegel ``θ(t) = Im ln Gamma(1/4 + i t/2) - (t/2) ln π``."""
    if not math.isfinite(t):
        raise ValueError("t must be finite")
    if abs(t) < T_MIN or abs(t) > T_MAX:
        raise ValueError(f"hardy theta requires {T_MIN} <= |t| <= {T_MAX}")
    z = ComplexInterval.from_parts(
        Interval.point(0.25),
        Interval.point(0.5 * t),
    )
    log_g = log_gamma_ci(z)
    return log_g.im - Interval.point(0.5 * t) * ln_iv(PI_IV)


def hardy_z(t: float) -> ComplexInterval:
    r"""Enclosure of ``e^{i θ(t)} ζ(1/2 + i t)`` on the locked compact.

    Real on the critical line for real ``t``; the enclosure is a complex
    rectangle and does not claim a real zero.
    """
    theta = riemann_siegel_theta(t)
    phase = exp_ci(ComplexInterval.from_parts(Interval.point(0.0), theta))
    s = ComplexInterval.from_parts(Interval.point(0.5), Interval.point(t))
    try:
        zeta = zeta_approximate_functional_equation(s)
    except ValueError:
        zeta = zeta_euler_maclaurin(s, num_sum_terms=30, order=6)
    return phase * zeta


def afe_honesty() -> dict[str, bool]:
    payload = continuation_honesty()
    payload["gabcke_remainder"] = False
    payload["pade"] = False
    payload["independent_afe_remainder"] = False
    payload["euler_maclaurin_residual"] = True
    return payload


__all__ = [
    "REMAINDER_FACTOR",
    "SIGMA_HI",
    "SIGMA_LO",
    "T_MAX",
    "T_MIN",
    "afe_honesty",
    "hardy_z",
    "riemann_siegel_theta",
    "zeta_approximate_functional_equation",
]
