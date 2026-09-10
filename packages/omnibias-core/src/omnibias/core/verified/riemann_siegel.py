# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Approximate functional equation for zeta on a named compact.

Hardy--Littlewood / Titchmarsh §4.13: for ``s = σ + it`` with ``t ≠ 0``,

.. math::

    \zeta(s) = \sum_{n \le x} n^{-s} + \chi(s)\sum_{n \le y} n^{s-1} + R(s;x,y)

when ``2 π x y = |t|``. This module encloses the two finite sums and adds an
axis-aligned remainder square whose radius is the inflated Titchmarsh §4.13
majorant

.. math::

    |R| \le C\bigl(x^{-\sigma} + |\chi(s)|\, y^{\sigma-1}\bigr)

with ``C = 4`` on the locked compact ``T_MIN <= |t| <= T_MAX`` and
``SIGMA_LO <= σ <= SIGMA_HI``. The constant is an inflation of the classical
O-constant on that compact, not Padé / Borel and not an mpmath residual.
Outside the compact the evaluator **refuses**. Gabcke-class remainders that
need ``|t| >= 200`` are out of scope here; CI tests the refusal.

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
from omnibias.core.verified.transcend import PI_IV, exp_iv, ln_iv
from omnibias.core.verified.xi import chi_factor, continuation_honesty

#: Locked compact (Titchmarsh §4.13 remainder majorant).
T_MIN = 2.0 * math.pi
T_MAX = 40.0
SIGMA_LO = 0.1
SIGMA_HI = 0.9
#: Inflated O-constant on the compact above.
REMAINDER_FACTOR = 4.0


def _as_ci(s: ComplexLike) -> ComplexInterval:
    return ComplexInterval.from_value(s)


def _require_compact(s: ComplexInterval) -> None:
    t_mag = s.im.mag
    if t_mag < T_MIN or t_mag > T_MAX:
        raise ValueError(f"AFE requires T_MIN={T_MIN} <= |Im(s)| <= T_MAX={T_MAX}; got |t|={t_mag}")
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
    """Enclosure of zeta via the AFE main terms plus the compact remainder square."""
    if remainder_factor <= 0.0:
        raise ValueError("remainder_factor must be positive")
    s_ci = _as_ci(s)
    _require_compact(s_ci)
    n_max, x_iv = _cutoff(s_ci.im.mag)
    if x_iv.lo <= 0.0:
        raise ValueError("AFE cutoff x must be strictly positive")
    one = ComplexInterval.one()
    main = n_power_neg_s(1, s_ci)
    second = n_power_neg_s(1, one - s_ci)
    for n in range(2, n_max + 1):
        main = main + n_power_neg_s(n, s_ci)
        second = second + n_power_neg_s(n, one - s_ci)
    chi = chi_factor(s_ci)
    main = main + chi * second
    # |R| <= C (x^{-σ} + |χ| y^{σ-1}) with x = y on this compact.
    ln_x = ln_iv(x_iv)
    sigma_lo = Interval.point(s_ci.re.lo)
    sigma_hi = Interval.point(s_ci.re.hi)
    x_neg_sigma = exp_iv(-(sigma_lo * ln_x))
    y_pow = exp_iv((sigma_hi - Interval.point(1.0)) * ln_x)
    radius_iv = Interval.point(remainder_factor) * (x_neg_sigma + Interval.point(chi.mag) * y_pow)
    radius = radius_iv.hi
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
