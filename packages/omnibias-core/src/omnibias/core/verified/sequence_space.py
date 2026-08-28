# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Weighted :math:`\ell^1_\nu` sequence spaces with rigorous tail bounds.

Modern computer-assisted proofs work in a Banach space of Fourier / Chebyshev /
Taylor coefficients with a *geometric* weight ``nu > 1`` (analytic data) or
``nu in (0, 1]`` (formal series):

.. math::

    \|a\|_\nu = \sum_{k} |a_k|\,\nu^{|k|}.

This norm makes coefficient convolution **submultiplicative**
(``||a * b||_nu <= ||a||_nu ||b||_nu``), i.e. the sequence space is a Banach
*algebra* -- the property the Newton-Kantorovich / radii-polynomial machinery in
:mod:`omnibias.core.verified.kantorovich` relies on.

This module provides the rigorous (outward-rounded :class:`Interval`) norm,
geometric tail bounds, and a :class:`ValidatedSeries` -- a finite vector of
interval coefficients plus a non-negative *tail* radius bounding the weighted
norm of everything truncated away -- closed under ``+``, ``*`` (Banach-algebra
convolution) and scalar operations.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from omnibias.core.verified.interval import Interval, IntervalLike


def _nu_power(nu: Interval, k: int) -> Interval:
    return nu.pow_int(k)


def _nonneg(iv: Interval) -> Interval:
    """Clamp a magnitude/tail enclosure to ``[0, inf)``.

    Outward rounding can push the lower endpoint of an exactly-zero magnitude one
    ulp below zero (``nextafter(0, -inf)`` is a negative subnormal, not ``0.0``),
    which would otherwise make ``ValidatedSeries.__post_init__`` spuriously reject
    a mathematically valid exact-zero tail. Since the enclosed quantity is a
    non-negative norm/tail, raising the lower bound to ``0`` stays rigorous. This
    mirrors :func:`omnibias.core.verified.fourier._nonneg`, used for exactly the
    same reason on the two-sided sibling.
    """
    return Interval(max(iv.lo, 0.0), max(iv.hi, 0.0))


def ell1_nu_norm(
    coeffs: Sequence[IntervalLike],
    nu: float,
    *,
    chebyshev: bool = False,
) -> Interval:
    r"""Rigorous one-sided weighted norm ``sum_{k>=0} w_k |a_k| nu^k``.

    With ``chebyshev=True`` the Chebyshev convention ``w_0 = 1``, ``w_k = 2``
    (``k >= 1``) is used; otherwise ``w_k = 1`` (Taylor / power series).
    """
    nu_iv = Interval.point(float(nu))
    acc = Interval.point(0.0)
    power = Interval.point(1.0)
    for k, c in enumerate(coeffs):
        mag = Interval.from_value(c).mag
        term = Interval.point(mag) * power
        if chebyshev and k >= 1:
            term = term * 2
        acc = acc + term
        power = power * nu_iv
    return acc


def fourier_nu_norm(coeffs: Mapping[int, IntervalLike], nu: float) -> Interval:
    r"""Rigorous two-sided weighted norm ``sum_{k in Z} |a_k| nu^{|k|}``."""
    nu_iv = Interval.point(float(nu))
    acc = Interval.point(0.0)
    for k, c in coeffs.items():
        mag = Interval.from_value(c).mag
        acc = acc + Interval.point(mag) * _nu_power(nu_iv, abs(int(k)))
    return acc


def geometric_tail_bound(
    coeff_bound: float,
    ratio: float,
    nu: float,
    n_trunc: int,
) -> Interval:
    r"""Bound ``sum_{k>n_trunc} |a_k| nu^k`` given ``|a_k| <= coeff_bound*ratio^k``.

    Requires the *weighted* ratio ``r = nu*ratio < 1``; the geometric tail is then
    ``coeff_bound * r^{n_trunc+1} / (1 - r)`` (outward rounded).
    """
    if coeff_bound < 0.0:
        raise ValueError("coeff_bound must be non-negative")
    r = Interval.point(float(nu)) * Interval.point(float(ratio))
    if r.hi >= 1.0:
        raise ValueError(f"weighted ratio nu*ratio={r.hi!r} must be < 1 for a finite tail")
    one_minus = Interval.point(1.0) - r
    if one_minus.lo <= 0.0:
        raise ValueError("1 - nu*ratio must be strictly positive")
    return Interval.point(coeff_bound) * r.pow_int(n_trunc + 1) / one_minus


def convolve(a: Sequence[Interval], b: Sequence[Interval]) -> list[Interval]:
    """Rigorous discrete convolution ``(a * b)_k = sum_{i+j=k} a_i b_j``."""
    if not a or not b:
        return []
    out = [Interval.point(0.0) for _ in range(len(a) + len(b) - 1)]
    for i, ai in enumerate(a):
        for j, bj in enumerate(b):
            out[i + j] = out[i + j] + ai * bj
    return out


def _chebyshev_product(a: Sequence[Interval], b: Sequence[Interval]) -> list[Interval]:
    r"""Rigorous product coefficients in the ``T_k`` Chebyshev basis.

    The basis identity is ``T_i T_j = (T_{i+j} + T_{|i-j|}) / 2`` for positive
    indices; terms involving ``T_0`` are copied once. Cauchy convolution is
    therefore not the Chebyshev product.
    """
    if not a or not b:
        return []
    out = [Interval.point(0.0) for _ in range(len(a) + len(b) - 1)]
    half = Interval.point(0.5)
    for i, ai in enumerate(a):
        for j, bj in enumerate(b):
            term = ai * bj
            if i == 0:
                out[j] = out[j] + term
            elif j == 0:
                out[i] = out[i] + term
            else:
                half_term = term * half
                out[i + j] = out[i + j] + half_term
                out[abs(i - j)] = out[abs(i - j)] + half_term
    return out


@dataclass
class ValidatedSeries:
    r"""A rigorous element of :math:`\ell^1_\nu`: finite coefficients + tail radius.

    ``coeffs[k]`` encloses the ``k``-th coefficient for ``k = 0..N``; ``tail`` is a
    non-negative bound on ``sum_{k>N} |a_k| nu^k`` (the weighted norm of the
    truncated remainder).  All operations keep ``coeffs`` exact for ``k <= N`` and
    fold every overflow / cross term rigorously into ``tail``.
    """

    coeffs: list[Interval]
    tail: Interval
    nu: float
    chebyshev: bool = False

    def __post_init__(self) -> None:
        if self.tail.lo < 0.0:
            raise ValueError("tail radius must be non-negative")
        if self.chebyshev and self.nu < 1.0:
            raise ValueError("Chebyshev ValidatedSeries requires nu >= 1")

    @property
    def order(self) -> int:
        return len(self.coeffs) - 1

    @classmethod
    def from_coeffs(
        cls,
        coeffs: Sequence[IntervalLike],
        nu: float,
        *,
        tail: IntervalLike = 0.0,
        chebyshev: bool = False,
    ) -> ValidatedSeries:
        return cls(
            [Interval.from_value(c) for c in coeffs],
            Interval.from_value(tail),
            float(nu),
            chebyshev,
        )

    def low_norm(self) -> Interval:
        """Weighted norm of the finite (kept) part only."""
        return ell1_nu_norm(self.coeffs, self.nu, chebyshev=self.chebyshev)

    def norm(self) -> Interval:
        """Rigorous total weighted norm ``||a||_nu`` (finite part + tail)."""
        return self.low_norm() + self.tail

    def _check_compatible(self, other: ValidatedSeries) -> None:
        if self.nu != other.nu or self.chebyshev != other.chebyshev:
            raise ValueError("ValidatedSeries operands must share nu and basis")

    def __add__(self, other: ValidatedSeries) -> ValidatedSeries:
        self._check_compatible(other)
        n = max(len(self.coeffs), len(other.coeffs))
        coeffs: list[Interval] = []
        for k in range(n):
            a = self.coeffs[k] if k < len(self.coeffs) else Interval.point(0.0)
            b = other.coeffs[k] if k < len(other.coeffs) else Interval.point(0.0)
            coeffs.append(a + b)
        return ValidatedSeries(coeffs, _nonneg(self.tail + other.tail), self.nu, self.chebyshev)

    def __mul__(self, other: ValidatedSeries) -> ValidatedSeries:
        self._check_compatible(other)
        n = min(self.order, other.order)  # keep coefficients up to this order
        conv = (
            _chebyshev_product(self.coeffs, other.coeffs)
            if self.chebyshev
            else convolve(self.coeffs, other.coeffs)
        )
        kept = conv[: n + 1]
        overflow = conv[n + 1 :]
        # Weighted norm of the overflow coefficients (their true index starts at n+1).
        nu_iv = Interval.point(self.nu)
        overflow_norm = Interval.point(0.0)
        power = _nu_power(nu_iv, n + 1)
        for k, c in enumerate(overflow):
            w = Interval.point(2.0) if self.chebyshev and (n + 1 + k) >= 1 else Interval.point(1.0)
            overflow_norm = overflow_norm + Interval.point(c.mag) * power * w
            power = power * nu_iv
        cross = (
            self.low_norm() * other.tail
            + self.tail * other.low_norm()
            + self.tail * other.tail
        )
        if self.chebyshev and cross.hi > 0.0:
            # A Chebyshev tail can fold back to every low mode through
            # T_i*T_j = (T_{i+j}+T_|i-j|)/2. Enclose that possible contribution
            # in each retained coefficient; the same norm bound covers the
            # unknown high modes held in ``tail``.
            cross_bound = Interval.point(cross.hi)
            for k, coefficient in enumerate(kept):
                weight = Interval.point(1.0 if k == 0 else 2.0)
                mode_bound = cross_bound / (weight * _nu_power(nu_iv, k))
                radius = mode_bound.hi
                kept[k] = coefficient + Interval(-radius, radius)
        return ValidatedSeries(kept, _nonneg(overflow_norm + cross), self.nu, self.chebyshev)

    def scale(self, factor: IntervalLike) -> ValidatedSeries:
        """Multiply by a scalar (constant) rigorously."""
        f = Interval.from_value(factor)
        coeffs = [c * f for c in self.coeffs]
        return ValidatedSeries(
            coeffs, _nonneg(self.tail * Interval.point(f.mag)), self.nu, self.chebyshev
        )

    def banach_algebra_bound(self, other: ValidatedSeries) -> Interval:
        """The submultiplicative bound ``||a||_nu * ||b||_nu`` on the product norm."""
        self._check_compatible(other)
        return self.norm() * other.norm()


__all__ = [
    "ValidatedSeries",
    "convolve",
    "ell1_nu_norm",
    "fourier_nu_norm",
    "geometric_tail_bound",
]
