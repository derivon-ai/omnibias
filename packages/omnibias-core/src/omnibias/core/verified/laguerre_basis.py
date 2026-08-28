# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Verified Laguerre-function basis on ``[0, inf)`` -- sibling of Hermite.

Convention
----------
This module fixes the **standard Laguerre polynomials** ``L_n = L_n^{(0)}``,

.. math::

    L_0(x) = 1,\quad L_1(x) = 1 - x,
    \quad L_{n+1}(x) = \frac{(2n+1-x)\,L_n(x) - n\,L_{n-1}(x)}{n+1},

with **exact rational** coefficients (the recurrence introduces a division by
``n+1``, so the coefficients live in ``Q``, not ``Z``).  The (unnormalized)
**Laguerre function** used here is

.. math::

    \ell_n(x) = L_n(x)\, e^{-x/2}, \qquad x \ge 0,

evaluated as an interval Horner scheme over :class:`~fractions.Fraction`
coefficients times a rigorous enclosure of ``exp(-x/2)``.

This is a **sibling** of :mod:`omnibias.core.verified.hermite_basis`, not a
drop-in Fourier eigenbasis: Laguerre functions are orthogonal on ``[0, inf)``
with weight ``e^{-x}``, and they are **not** self-dual under the Fourier
transform (there is no analogue of the exact ``(-i)^n`` multiplier).  A tail
bound is still available under a caller-supplied uniform ``|ell_n(x)|``
hypothesis, via the same geometric-decay contract as
:class:`~omnibias.core.verified.hermite_basis.HermiteExpansion`.

Scope.  This module never claims a continuum spectral theorem, a Cohn-Elkies
packing bound, or a Fourier inversion identity.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache

from omnibias.core.verified.complex_interval import ComplexInterval, ComplexLike
from omnibias.core.verified.interval import Interval, IntervalLike
from omnibias.core.verified.sequence_space import geometric_tail_bound
from omnibias.core.verified.transcend import exp_iv

_CACHE_SIZE: int = 256


@lru_cache(maxsize=_CACHE_SIZE)
def laguerre_poly_coeffs_exact(n: int) -> tuple[Fraction, ...]:
    """Exact rational coefficients of ``L_n``, low-degree first.

    ``L_n(x) = sum_{k=0}^n c_k x^k`` with each ``c_k`` a
    :class:`~fractions.Fraction`.
    """
    if n < 0:
        raise ValueError(f"order n must be >= 0, got {n}")
    if n == 0:
        return (Fraction(1),)
    if n == 1:
        return (Fraction(1), Fraction(-1))
    prev2: tuple[Fraction, ...] = (Fraction(1),)
    prev1: tuple[Fraction, ...] = (Fraction(1), Fraction(-1))
    for m in range(1, n):
        # L_{m+1} = ((2m+1) L_m - x L_m - m L_{m-1}) / (m+1)
        scale = Fraction(2 * m + 1)
        denom = Fraction(m + 1)
        shifted = (Fraction(0),) + prev1  # x * L_m
        length = max(len(prev1), len(shifted), len(prev2))
        out = [Fraction(0)] * length
        for k, c in enumerate(prev1):
            out[k] += scale * c
        for k, c in enumerate(shifted):
            out[k] -= c
        for k, c in enumerate(prev2):
            out[k] -= Fraction(m) * c
        prev2, prev1 = prev1, tuple(c / denom for c in out)
    return prev1


def _horner_rational(coeffs: Sequence[Fraction], x: Interval) -> Interval:
    acc = Interval.from_rational(coeffs[-1])
    for coeff in reversed(coeffs[:-1]):
        acc = acc * x + Interval.from_rational(coeff)
    return acc


def laguerre_weight(x: IntervalLike) -> Interval:
    """Rigorous enclosure of ``exp(-x/2)`` for ``x >= 0``."""
    xi = Interval.from_value(x)
    if xi.lo < 0.0:
        raise ValueError(f"Laguerre weight requires x >= 0, got x.lo={xi.lo!r}")
    return exp_iv(Interval.point(-0.5) * xi)


def laguerre_function(n: int, x: IntervalLike) -> Interval:
    r"""Rigorous enclosure of ``ell_n(x) = L_n(x) exp(-x/2)`` for ``x >= 0``."""
    xi = Interval.from_value(x)
    return _horner_rational(laguerre_poly_coeffs_exact(n), xi) * laguerre_weight(xi)


@dataclass
class LaguerreExpansion:
    """A finite Laguerre-function expansion ``f = sum_{n=0}^{N} c_n ell_n``.

    No stored tail radius: as with :class:`~omnibias.core.verified.hermite_basis.HermiteExpansion`,
    the tail depends on the evaluation point through ``|ell_n(x)|``.
    """

    coeffs: list[ComplexInterval]

    def __post_init__(self) -> None:
        if not self.coeffs:
            raise ValueError("LaguerreExpansion needs at least one coefficient")

    @property
    def order(self) -> int:
        return len(self.coeffs) - 1

    @classmethod
    def from_coeffs(cls, coeffs: Sequence[ComplexLike]) -> LaguerreExpansion:
        return cls([ComplexInterval.from_value(c) for c in coeffs])

    def evaluate_kept(self, x: IntervalLike) -> ComplexInterval:
        xi = Interval.from_value(x)
        acc = ComplexInterval.zero()
        for n, coeff in enumerate(self.coeffs):
            acc = acc + coeff * ComplexInterval.from_value(laguerre_function(n, xi))
        return acc

    def tail_bound(
        self,
        x: IntervalLike,
        *,
        coeff_bound: float,
        ratio: float,
        psi_bound: float,
    ) -> Interval:
        r"""Geometric tail ``sum_{n>N} |c_n ell_n(x)|`` under a uniform ``psi_bound``.

        Requires ``|c_n| <= coeff_bound * ratio^n`` past the truncation and
        ``|ell_n(x)| <= psi_bound`` for those ``n``.  The weight ``nu`` of
        :func:`~omnibias.core.verified.sequence_space.geometric_tail_bound` is
        ``1`` (unweighted coefficient decay).
        """
        del x  # bound is uniform in x by hypothesis on psi_bound
        if psi_bound < 0.0:
            raise ValueError(f"psi_bound must be >= 0, got {psi_bound!r}")
        return geometric_tail_bound(coeff_bound, ratio, 1.0, self.order) * Interval.point(
            psi_bound
        )

    def evaluate(
        self,
        x: IntervalLike,
        *,
        coeff_bound: float,
        ratio: float,
        psi_bound: float,
    ) -> ComplexInterval:
        kept = self.evaluate_kept(x)
        tail = self.tail_bound(x, coeff_bound=coeff_bound, ratio=ratio, psi_bound=psi_bound)
        pad = Interval(-tail.hi, tail.hi)
        return ComplexInterval(kept.re + pad, kept.im + pad)


__all__ = [
    "LaguerreExpansion",
    "laguerre_function",
    "laguerre_poly_coeffs_exact",
    "laguerre_weight",
]
