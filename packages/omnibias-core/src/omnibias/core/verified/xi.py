# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Completed ``xi`` / ``chi`` and a finite functional-equation zeta evaluator.

The Dirichlet *series* register stays :mod:`omnibias.core.verified.dirichlet`
(``Re(s) > 1`` only). This module encloses the completed function

.. math::

    \xi(s) = \tfrac12 s(s-1)\,\pi^{-s/2}\,\Gamma(s/2)\,\zeta(s)

and the factor ``chi(s)`` in ``zeta(s) = chi(s) zeta(1-s)`` on named compact
rectangles. For ``Re(s) < 0`` the evaluator ``zeta_via_functional_equation``
uses ``zeta_enclosure`` on the right half-plane -- a numerical enclosure of
the continued *value*, not a continuation theorem and not a zero locator.

Honesty: ``rh_claim`` is frozen ``False``. A residual enclosure of
``xi(s) - xi(1-s)`` that contains ``0`` is Enclosure Collapse of a finite
obligation, not ``identity_collapse`` over ``Q``, and never infers the
Riemann Hypothesis.
"""

from __future__ import annotations

from typing import Any

from omnibias.core.verified.complex_interval import ComplexInterval, ComplexLike
from omnibias.core.verified.dirichlet import (
    n_power_neg_s,
    zeta_enclosure,
    zeta_euler_maclaurin,
    zeta_negative_odd,
)
from omnibias.core.verified.gamma_complex import exp_ci, gamma_ci, log_ci
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.transcend import PI_IV, ln_iv

_HALF = Interval.point(0.5)
_XI_ZETA_TERMS = 240
_WINDING_ZETA_TERMS = 80


def continuation_honesty() -> dict[str, bool]:
    """Runtime honesty for the finite evaluator. Never infers RH."""
    return {
        "rh_claim": False,
        "continuation_theorem": False,
        "zeros": False,
    }


def _as_ci(s: ComplexLike) -> ComplexInterval:
    return ComplexInterval.from_value(s)


def _contains_one(s: ComplexInterval) -> bool:
    return (s - ComplexInterval.one()).modulus().lo <= 0.0


def _is_near_nonpositive_even_integer(s: ComplexInterval, *, tol: float = 1e-12) -> bool:
    if s.re.width > tol or s.im.width > tol:
        return False
    if not s.im.contains(0.0):
        return False
    x = s.re.mid
    if x > tol:
        return False
    n = round(x)
    return n % 2 == 0 and abs(x - n) <= tol


def chi_factor(s: ComplexLike) -> ComplexInterval:
    r"""Enclosure of ``chi(s) = pi^{s-1/2} Gamma((1-s)/2) / Gamma(s/2)``.

    Poles of the denominator (``s = 0, -2, -4, ...``) are refused unless the
    caller is evaluating a known trivial zero of zeta separately.
    """
    s_ci = _as_ci(s)
    half = ComplexInterval.from_parts(_HALF)
    pi_c = ComplexInterval.from_parts(PI_IV)
    log_pi = log_ci(pi_c)
    exp_part = exp_ci((s_ci - half) * log_pi)
    num = gamma_ci((ComplexInterval.one() - s_ci) * half)
    den = gamma_ci(s_ci * half)
    return exp_part * num / den


def zeta_via_functional_equation(
    s: ComplexLike, *, num_terms: int = _XI_ZETA_TERMS
) -> ComplexInterval:
    r"""``zeta(s) = chi(s) zeta(1-s)`` for ``Re(s) < 0``, using the series on the right.

    Requires ``Re(s).hi < 0`` so that ``Re(1-s).lo > 1``. Negative even integers
    (trivial zeros) return the exact ``0`` without evaluating ``chi``.
    """
    s_ci = _as_ci(s)
    if _is_near_nonpositive_even_integer(s_ci):
        return ComplexInterval.zero()
    if s_ci.re.hi >= 0.0:
        raise ValueError(
            f"zeta_via_functional_equation requires Re(s) < 0; got Re.hi={s_ci.re.hi!r}"
        )
    right = zeta_enclosure(ComplexInterval.one() - s_ci, num_terms=num_terms)
    return chi_factor(s_ci) * right


def zeta_continued(
    s: ComplexLike,
    *,
    num_terms: int = _XI_ZETA_TERMS,
    em_order: int = 6,
) -> ComplexInterval:
    r"""Finite dispatcher for an enclosure of the continued *value* of ``zeta(s)``.

    * pole at ``s = 1`` is refused
    * ``Re(s) > 1`` uses :func:`zeta_enclosure`
    * ``Re(s) < 0`` uses :func:`zeta_via_functional_equation`
    * otherwise Euler-Maclaurin on the strip (including the line ``Re = 1``
      away from the pole)

    Not a continuation theorem. Honesty keys stay false.
    """
    s_ci = _as_ci(s)
    if _contains_one(s_ci):
        raise ValueError("zeta_continued: s-1 meets the pole at s = 1")
    if s_ci.re.lo > 1.0:
        return zeta_enclosure(s_ci, num_terms=num_terms)
    if s_ci.re.hi < 0.0:
        return zeta_via_functional_equation(s_ci, num_terms=num_terms)
    return zeta_euler_maclaurin(s_ci, num_sum_terms=max(num_terms // 8, 20), order=em_order)


def xi_enclosure(
    s: ComplexLike,
    *,
    num_terms: int = _XI_ZETA_TERMS,
    em_order: int = 6,
) -> ComplexInterval:
    r"""Enclosure of ``xi(s) = (1/2) s (s-1) pi^{-s/2} Gamma(s/2) zeta(s)``."""
    s_ci = _as_ci(s)
    half = ComplexInterval.from_parts(_HALF)
    log_pi = log_ci(ComplexInterval.from_parts(PI_IV))
    pi_pow = exp_ci(-(s_ci * half) * log_pi)
    pre = half * s_ci * (s_ci - ComplexInterval.one()) * pi_pow * gamma_ci(s_ci * half)
    return pre * zeta_continued(s_ci, num_terms=num_terms, em_order=em_order)


def xi_functional_equation_residual(
    s: ComplexLike,
    *,
    num_terms: int = _XI_ZETA_TERMS,
) -> ComplexInterval:
    r"""Enclosure of ``xi(s) - xi(1-s)`` on a compact. Contains ``0`` when both sides match.

    Intended for ``Re(s) > 1`` so one side is the series register. Not an
    identity over ``Q`` and not an RH claim.
    """
    s_ci = _as_ci(s)
    return xi_enclosure(s_ci, num_terms=num_terms) - xi_enclosure(
        ComplexInterval.one() - s_ci, num_terms=num_terms
    )


def _winding_on_rectangle(
    evaluator: Any,
    center: complex,
    half_width: float,
    half_height: float,
    *,
    segments: int,
) -> Interval | None:
    # Lazy: collapse.winding must not load while omnibias.core.verified is
    # still importing (circular proof.engine -> collapse.identity).
    from omnibias.core.collapse.winding import winding_enclosure_function

    if half_width <= 0.0 or half_height <= 0.0:
        raise ValueError("half_width and half_height must be positive")
    if segments < 4 or segments % 4:
        raise ValueError("segments must be divisible by 4 and >= 4")
    return winding_enclosure_function(
        evaluator,
        center,
        half_width,
        segments=segments,
        contour="rectangle",
        half_width=half_width,
        half_height=half_height,
    )


def zeta_winding_enclosure(
    center: complex,
    half_width: float,
    half_height: float,
    *,
    segments: int = 8,
    num_terms: int = _WINDING_ZETA_TERMS,
) -> Interval | None:
    """Winding of ``zeta`` on an axis-aligned rectangle that stays in ``Re(s) > 1``.

    A unique integer ``0`` is the expected count (no zeros in ``Re > 1``).
    Returning ``None`` is BLOCKED, never a zero certificate.
    """
    if center.real - half_width <= 1.0:
        raise ValueError("zeta_winding_enclosure requires the whole rectangle in Re(s) > 1")

    def _eval(z: ComplexInterval) -> ComplexInterval:
        return zeta_enclosure(z, num_terms=num_terms)

    return _winding_on_rectangle(_eval, center, half_width, half_height, segments=segments)


def zeta_strip_winding_enclosure(
    center: complex,
    half_width: float,
    half_height: float,
    *,
    segments: int = 8,
    num_sum_terms: int = 20,
    em_order: int = 6,
) -> Interval | None:
    """Winding of the Euler-Maclaurin zeta evaluator on a strip rectangle.

    Returning ``None`` is the allowed BLOCKED outcome (image meets 0 or the
    arg cut). It is never a zero certificate.
    """

    def _eval(z: ComplexInterval) -> ComplexInterval:
        return zeta_euler_maclaurin(z, num_sum_terms=num_sum_terms, order=em_order)

    return _winding_on_rectangle(_eval, center, half_width, half_height, segments=segments)


def zeta_winding_count(
    center: complex,
    half_width: float,
    half_height: float,
    **kwargs: Any,
) -> tuple[int | None, Interval | None]:
    """Isolate a unique winding integer, or ``(None, enclosure_or_None)`` if BLOCKED."""
    from omnibias.core.collapse.winding import integers_in

    winding = zeta_winding_enclosure(center, half_width, half_height, **kwargs)
    if winding is None:
        return None, None
    hits = integers_in(winding)
    if len(hits) != 1:
        return None, winding
    return hits[0], winding


def trivial_zero_enclosure(m: int) -> Interval:
    """Closed-form ``zeta(1-2m)`` for ``m >= 1`` (negative odd integers)."""
    return zeta_negative_odd(m)


def two_to_the_s(s: ComplexLike) -> ComplexInterval:
    """``2^s = exp(s ln 2)``."""
    return exp_ci(_as_ci(s) * ComplexInterval.from_parts(ln_iv(Interval.point(2.0))))


def n_power_s(n: int, s: ComplexLike) -> ComplexInterval:
    """``n^s = exp(s ln n)`` for integer ``n >= 1`` (inverse of ``n^{-s}``)."""
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    inv = n_power_neg_s(n, s)
    return ComplexInterval.one() / inv


__all__ = [
    "chi_factor",
    "continuation_honesty",
    "n_power_s",
    "trivial_zero_enclosure",
    "two_to_the_s",
    "xi_enclosure",
    "xi_functional_equation_residual",
    "zeta_continued",
    "zeta_strip_winding_enclosure",
    "zeta_via_functional_equation",
    "zeta_winding_count",
    "zeta_winding_enclosure",
]
