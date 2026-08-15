# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Verified whole-line Hilbert transform on the generalized Cauchy-Hardy pair.

For ``a > 0`` and real ``alpha``, the branch

    F_{a,alpha}(y) = (a - i y)^{-alpha}

is analytic and single-valued in the upper half-plane.  Writing
``r = sqrt(a^2 + y^2)`` and ``phi = atan(y / a)``,

    P_{a,alpha}(y) = r^{-alpha} cos(alpha phi)   (even),
    Q_{a,alpha}(y) = r^{-alpha} sin(alpha phi)   (odd).

Under the repo convention ``H[f](x) = (1/pi) p.v. int f(t)/(x-t) dt``,

    H[P] = Q,      H[Q] = -P.

Derivatives close in the family:

    P' = -alpha Q_{a, alpha+1},
    Q' =  alpha P_{a, alpha+1}.

The classical Poisson / conjugate-Poisson pair of
:mod:`omnibias.core.verified.line` is the special case ``alpha = 1``:

    P_{a,1} = a / (a^2 + y^2),    Q_{a,1} = y / (a^2 + y^2).

For CCF self-similar profiles the physical far-field exponent is
``alpha = 1/(1+lambda)``, so a finite Hardy sum matches the algebraic decay
that a finite Poisson sum (``|y|^{-2}``) structurally cannot.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from omnibias.core.verified.interval import Interval
from omnibias.core.verified.transcend import (
    atan_iv,
    cos_iv,
    exp_iv,
    ln_iv,
    sin_iv,
)


def _validate_scale_alpha(a: float, alpha: float) -> None:
    if not (a > 0.0 and math.isfinite(a)):
        raise ValueError(f"Hardy scale a must be finite and > 0, got {a!r}")
    if not math.isfinite(alpha):
        raise ValueError(f"Hardy exponent alpha must be finite, got {alpha!r}")


def hardy_radius(y: float, a: float) -> Interval:
    """Verified ``r = sqrt(a^2 + y^2)``."""
    _validate_scale_alpha(a, 1.0)
    return (Interval.point(a).pow_int(2) + Interval.point(y).pow_int(2)).sqrt()


def hardy_radius_iv(y: Interval, a: float) -> Interval:
    """Verified ``r = sqrt(a^2 + y^2)`` for an interval ``y`` (cell covering)."""
    _validate_scale_alpha(a, 1.0)
    return (Interval.point(a).pow_int(2) + y.pow_int(2)).sqrt()


def hardy_angle(y: float, a: float) -> Interval:
    """Verified ``phi = atan(y / a)``."""
    _validate_scale_alpha(a, 1.0)
    return atan_iv(Interval.point(y) * Interval.point(a).reciprocal())


def hardy_angle_iv(y: Interval, a: float) -> Interval:
    """Verified ``phi = atan(y / a)`` for interval ``y``."""
    _validate_scale_alpha(a, 1.0)
    return atan_iv(y * Interval.point(a).reciprocal())


def hardy_even(y: float, a: float, alpha: float) -> Interval:
    r"""Verified ``P_{a,alpha}(y) = r^{-alpha} cos(alpha phi)`` (even)."""
    _validate_scale_alpha(a, alpha)
    r = hardy_radius(y, a)
    # r^{-alpha} = exp(-alpha * ln(r)); r > 0 for a > 0.
    r_pow = exp_iv(Interval.point(-alpha) * ln_iv(r))
    return r_pow * cos_iv(Interval.point(alpha) * hardy_angle(y, a))


def hardy_even_iv(y: Interval, a: float, alpha: float) -> Interval:
    r"""Interval-``y`` enclosure of ``P_{a,alpha}`` (sound over a cell)."""
    _validate_scale_alpha(a, alpha)
    r = hardy_radius_iv(y, a)
    r_pow = exp_iv(Interval.point(-alpha) * ln_iv(r))
    return r_pow * cos_iv(Interval.point(alpha) * hardy_angle_iv(y, a))


def hardy_odd(y: float, a: float, alpha: float) -> Interval:
    r"""Verified ``Q_{a,alpha}(y) = r^{-alpha} sin(alpha phi)`` (odd)."""
    _validate_scale_alpha(a, alpha)
    r = hardy_radius(y, a)
    r_pow = exp_iv(Interval.point(-alpha) * ln_iv(r))
    return r_pow * sin_iv(Interval.point(alpha) * hardy_angle(y, a))


def hardy_odd_iv(y: Interval, a: float, alpha: float) -> Interval:
    r"""Interval-``y`` enclosure of ``Q_{a,alpha}`` (sound over a cell)."""
    _validate_scale_alpha(a, alpha)
    r = hardy_radius_iv(y, a)
    r_pow = exp_iv(Interval.point(-alpha) * ln_iv(r))
    return r_pow * sin_iv(Interval.point(alpha) * hardy_angle_iv(y, a))


def hardy_pair(y: float, a: float, alpha: float) -> tuple[Interval, Interval]:
    """Return ``(P, Q)`` at ``y``."""
    return hardy_even(y, a, alpha), hardy_odd(y, a, alpha)


def hardy_even_deriv(y: float, a: float, alpha: float) -> Interval:
    r"""Verified ``P' = -alpha Q_{a, alpha+1}``."""
    _validate_scale_alpha(a, alpha)
    return Interval.point(-alpha) * hardy_odd(y, a, alpha + 1.0)


def hardy_even_deriv_iv(y: Interval, a: float, alpha: float) -> Interval:
    """Interval-``y`` enclosure of ``P'``."""
    _validate_scale_alpha(a, alpha)
    return Interval.point(-alpha) * hardy_odd_iv(y, a, alpha + 1.0)


def hardy_odd_deriv(y: float, a: float, alpha: float) -> Interval:
    r"""Verified ``Q' = alpha P_{a, alpha+1}``."""
    _validate_scale_alpha(a, alpha)
    return Interval.point(alpha) * hardy_even(y, a, alpha + 1.0)


def hardy_odd_deriv_iv(y: Interval, a: float, alpha: float) -> Interval:
    """Interval-``y`` enclosure of ``Q'``."""
    _validate_scale_alpha(a, alpha)
    return Interval.point(alpha) * hardy_even_iv(y, a, alpha + 1.0)


def hardy_even_dalpha(y: float, a: float, alpha: float) -> Interval:
    r"""Verified ``dP/dalpha = -ln(r) P - phi Q``."""
    _validate_scale_alpha(a, alpha)
    p, q = hardy_pair(y, a, alpha)
    return -ln_iv(hardy_radius(y, a)) * p - hardy_angle(y, a) * q


def hardy_odd_dalpha(y: float, a: float, alpha: float) -> Interval:
    r"""Verified ``dQ/dalpha = -ln(r) Q + phi P``."""
    _validate_scale_alpha(a, alpha)
    p, q = hardy_pair(y, a, alpha)
    return -ln_iv(hardy_radius(y, a)) * q + hardy_angle(y, a) * p


def hilbert_of_hardy_even(y: float, a: float, alpha: float) -> Interval:
    """Exact line Hilbert: ``H[P] = Q``."""
    return hardy_odd(y, a, alpha)


def hilbert_of_hardy_odd(y: float, a: float, alpha: float) -> Interval:
    """Exact line Hilbert: ``H[Q] = -P``."""
    return -hardy_even(y, a, alpha)


def _profile_sum(
    kernel: object,
    y: float,
    coeffs: Sequence[float],
    scales: Sequence[float],
    alpha: float,
) -> Interval:
    f = kernel
    assert callable(f)
    if len(coeffs) != len(scales):
        raise ValueError("coeffs and scales must have the same length")
    if not coeffs:
        raise ValueError("need at least one (coeff, scale) term")
    acc = Interval.point(0.0)
    for c, a in zip(coeffs, scales, strict=True):
        acc = acc + Interval.point(float(c)) * f(float(y), float(a), float(alpha))
    return acc


def hardy_even_profile(
    y: float, coeffs: Sequence[float], scales: Sequence[float], alpha: float
) -> Interval:
    r"""Verified even profile ``Theta = sum_i c_i P_{a_i, alpha}``."""
    return _profile_sum(hardy_even, y, coeffs, scales, alpha)


def hardy_even_profile_deriv(
    y: float, coeffs: Sequence[float], scales: Sequence[float], alpha: float
) -> Interval:
    r"""Verified ``Theta' = sum_i c_i P'_{a_i, alpha}``."""
    return _profile_sum(hardy_even_deriv, y, coeffs, scales, alpha)


def hilbert_hardy_even_profile(
    y: float, coeffs: Sequence[float], scales: Sequence[float], alpha: float
) -> Interval:
    r"""Verified exact ``H[Theta] = sum_i c_i Q_{a_i, alpha}``."""
    return _profile_sum(hardy_odd, y, coeffs, scales, alpha)


def hilbert_hardy_even_profile_deriv(
    y: float, coeffs: Sequence[float], scales: Sequence[float], alpha: float
) -> Interval:
    r"""Verified exact ``(H Theta)' = sum_i c_i Q'_{a_i, alpha}``."""
    return _profile_sum(hardy_odd_deriv, y, coeffs, scales, alpha)


def hardy_tail_constant(
    coeffs: Sequence[float], scales: Sequence[float], alpha: float
) -> tuple[float, float]:
    r"""Far-field decay ``(C, p)`` with ``|Theta(y)| <= C |y|^{-p}``.

    Asymptotically ``P_{a,alpha}(y) ~ cos(pi alpha / 2) |y|^{-alpha}`` (leading),
    and ``|P| <= r^{-alpha} <= |a|^{-0} |y|^{-alpha}`` is too loose near zero;
    for the Hilbert tail we use the uniform bound
    ``|P_{a,alpha}(y)| <= |y|^{-alpha}`` for ``|y| >= a`` is not always true.
    A safe envelope is ``|P| <= r^{-alpha} <= a^{-alpha}`` near the origin and
    ``|P| <= |y|^{-alpha}`` at infinity; feeding
    :func:`~omnibias.core.verified.line.hilbert_tail_bound` we take
    ``C = sum_i |c_i|`` and ``p = alpha`` (outward-rounded), which is sound for
    ``|y| >= max_i a_i`` after absorbing the angular factor into a slightly
    larger ``C`` via ``max |cos| <= 1``.
    """
    if len(coeffs) != len(scales):
        raise ValueError("coeffs and scales must have the same length")
    if alpha <= 0.0:
        raise ValueError("alpha must be > 0 for an algebraic tail")
    acc = Interval.point(0.0)
    for c in coeffs:
        acc = acc + Interval.point(float(c)).abs()
    return acc.hi, float(alpha)


def _matrix(
    fn: object,
    y_nodes: Sequence[float],
    a_values: Sequence[float],
    alpha: float,
) -> list[list[Interval]]:
    f = fn
    assert callable(f)
    return [[f(float(y), float(a), float(alpha)) for a in a_values] for y in y_nodes]


def hardy_even_matrix(
    y_nodes: Sequence[float], a_values: Sequence[float], alpha: float
) -> list[list[Interval]]:
    """Design matrix ``[[P_{a_i, alpha}(y_j)]]``."""
    return _matrix(hardy_even, y_nodes, a_values, alpha)


def hardy_odd_matrix(
    y_nodes: Sequence[float], a_values: Sequence[float], alpha: float
) -> list[list[Interval]]:
    """Design matrix ``[[Q_{a_i, alpha}(y_j)]]``."""
    return _matrix(hardy_odd, y_nodes, a_values, alpha)


def hardy_even_deriv_matrix(
    y_nodes: Sequence[float], a_values: Sequence[float], alpha: float
) -> list[list[Interval]]:
    """Design matrix ``[[P'_{a_i, alpha}(y_j)]]``."""
    return _matrix(hardy_even_deriv, y_nodes, a_values, alpha)


def hardy_odd_deriv_matrix(
    y_nodes: Sequence[float], a_values: Sequence[float], alpha: float
) -> list[list[Interval]]:
    """Design matrix ``[[Q'_{a_i, alpha}(y_j)]]``."""
    return _matrix(hardy_odd_deriv, y_nodes, a_values, alpha)


def pochhammer(alpha: float, n: int) -> float:
    """Rising factorial ``(alpha)_n``. Exact for integer ``alpha``; float otherwise."""
    if n < 0:
        raise ValueError(f"Pochhammer order n must be >= 0, got {n}")
    acc = 1.0
    for k in range(n):
        acc *= alpha + k
    return acc


def pochhammer_iv(alpha: float, n: int) -> Interval:
    """Outward-rounded enclosure of ``(alpha)_n``."""
    if n < 0:
        raise ValueError(f"Pochhammer order n must be >= 0, got {n}")
    acc = Interval.point(1.0)
    for k in range(n):
        acc = acc * Interval.point(alpha + float(k))
    return acc


def _table_kind(n: int) -> tuple[int, str, int, str]:
    """Signs and P/Q kinds for ``(P^(n), Q^(n))`` from ``n mod 4``."""
    r = n % 4
    if r == 0:
        return 1, "even", 1, "odd"
    if r == 1:
        return -1, "odd", 1, "even"
    if r == 2:
        return -1, "even", -1, "odd"
    return 1, "odd", -1, "even"


def _eval_pq(kind: str, y: float, a: float, alpha: float) -> Interval:
    if kind == "even":
        return hardy_even(y, a, alpha)
    return hardy_odd(y, a, alpha)


def _eval_pq_iv(kind: str, y: Interval, a: float, alpha: float) -> Interval:
    if kind == "even":
        return hardy_even_iv(y, a, alpha)
    return hardy_odd_iv(y, a, alpha)


def hardy_even_deriv_n(y: float, a: float, alpha: float, n: int) -> Interval:
    """Closed-form ``P^(n)`` via the Hardy table. ``n = 1`` matches ``hardy_even_deriv``."""
    if n < 0:
        raise ValueError(f"derivative order n must be >= 0, got {n}")
    if n == 1:
        return hardy_even_deriv(y, a, alpha)
    if n == 0:
        return hardy_even(y, a, alpha)
    _validate_scale_alpha(a, alpha)
    p_sign, p_kind, _, _ = _table_kind(n)
    factor = pochhammer_iv(alpha, n)
    atom = _eval_pq(p_kind, y, a, alpha + float(n))
    return (Interval.point(-1.0) * factor * atom) if p_sign < 0 else (factor * atom)


def hardy_odd_deriv_n(y: float, a: float, alpha: float, n: int) -> Interval:
    """Closed-form ``Q^(n)`` via the Hardy table. ``n = 1`` matches ``hardy_odd_deriv``."""
    if n < 0:
        raise ValueError(f"derivative order n must be >= 0, got {n}")
    if n == 1:
        return hardy_odd_deriv(y, a, alpha)
    if n == 0:
        return hardy_odd(y, a, alpha)
    _validate_scale_alpha(a, alpha)
    _, _, q_sign, q_kind = _table_kind(n)
    factor = pochhammer_iv(alpha, n)
    atom = _eval_pq(q_kind, y, a, alpha + float(n))
    return (Interval.point(-1.0) * factor * atom) if q_sign < 0 else (factor * atom)


def hardy_even_deriv_n_iv(y: Interval, a: float, alpha: float, n: int) -> Interval:
    """Interval-``y`` enclosure of ``P^(n)``."""
    if n < 0:
        raise ValueError(f"derivative order n must be >= 0, got {n}")
    if n == 1:
        return hardy_even_deriv_iv(y, a, alpha)
    if n == 0:
        return hardy_even_iv(y, a, alpha)
    _validate_scale_alpha(a, alpha)
    p_sign, p_kind, _, _ = _table_kind(n)
    factor = pochhammer_iv(alpha, n)
    atom = _eval_pq_iv(p_kind, y, a, alpha + float(n))
    return (Interval.point(-1.0) * factor * atom) if p_sign < 0 else (factor * atom)


def hardy_odd_deriv_n_iv(y: Interval, a: float, alpha: float, n: int) -> Interval:
    """Interval-``y`` enclosure of ``Q^(n)``."""
    if n < 0:
        raise ValueError(f"derivative order n must be >= 0, got {n}")
    if n == 1:
        return hardy_odd_deriv_iv(y, a, alpha)
    if n == 0:
        return hardy_odd_iv(y, a, alpha)
    _validate_scale_alpha(a, alpha)
    _, _, q_sign, q_kind = _table_kind(n)
    factor = pochhammer_iv(alpha, n)
    atom = _eval_pq_iv(q_kind, y, a, alpha + float(n))
    return (Interval.point(-1.0) * factor * atom) if q_sign < 0 else (factor * atom)


def hilbert_of_hardy_even_deriv_n(y: float, a: float, alpha: float, n: int) -> Interval:
    """Exact line Hilbert: ``H[P^(n)] = Q^(n)``. Commutation needs ``alpha > 0``."""
    if not (alpha > 0.0):
        raise ValueError(
            "Hilbert-derivative commutation needs decay (alpha > 0); "
            f"got alpha={alpha!r}"
        )
    return hardy_odd_deriv_n(y, a, alpha, n)


def hilbert_of_hardy_odd_deriv_n(y: float, a: float, alpha: float, n: int) -> Interval:
    """Exact line Hilbert: ``H[Q^(n)] = -P^(n)``. Commutation needs ``alpha > 0``."""
    if not (alpha > 0.0):
        raise ValueError(
            "Hilbert-derivative commutation needs decay (alpha > 0); "
            f"got alpha={alpha!r}"
        )
    return -hardy_even_deriv_n(y, a, alpha, n)


__all__ = [
    "hardy_angle",
    "hardy_angle_iv",
    "hardy_even",
    "hardy_even_dalpha",
    "hardy_even_deriv",
    "hardy_even_deriv_iv",
    "hardy_even_deriv_matrix",
    "hardy_even_deriv_n",
    "hardy_even_deriv_n_iv",
    "hardy_even_iv",
    "hardy_even_matrix",
    "hardy_even_profile",
    "hardy_even_profile_deriv",
    "hardy_odd",
    "hardy_odd_dalpha",
    "hardy_odd_deriv",
    "hardy_odd_deriv_iv",
    "hardy_odd_deriv_matrix",
    "hardy_odd_deriv_n",
    "hardy_odd_deriv_n_iv",
    "hardy_odd_iv",
    "hardy_odd_matrix",
    "hardy_pair",
    "hardy_radius",
    "hardy_radius_iv",
    "hardy_tail_constant",
    "hilbert_hardy_even_profile",
    "hilbert_hardy_even_profile_deriv",
    "hilbert_of_hardy_even",
    "hilbert_of_hardy_even_deriv_n",
    "hilbert_of_hardy_odd",
    "hilbert_of_hardy_odd_deriv_n",
    "pochhammer",
    "pochhammer_iv",
]
