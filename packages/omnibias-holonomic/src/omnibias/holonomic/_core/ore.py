# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Ore (skew-polynomial) algebra ``R[d; sigma, delta]`` over rational polynomials.

An Ore operator is a polynomial ``sum_i c_i(x) d^i`` in an operator ``d`` whose coefficients
``c_i`` are rational polynomials, with the non-commutative product rule

.. math::

    d \cdot r = \sigma(r)\, d + \delta(r),

where ``sigma`` is a ring endomorphism and ``delta`` a ``sigma``-derivation. Two standard
specialisations power the holonomic engine:

* the **shift** algebra ``sigma(p)(x) = p(x+1)``, ``delta = 0`` -- ``d = S`` is the shift
  ``(S a)(n) = a(n+1)``; operators here are the P-recursive recurrences;
* the **differential** algebra ``sigma = id``, ``delta = d/dx`` -- ``d = D`` is
  differentiation; operators here are the D-finite ODEs.

Everything is exact rational arithmetic.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from fractions import Fraction

from omnibias.holonomic._core.rational_poly import (
    Poly,
    padd,
    pderiv,
    peval,
    pmul,
    pshift,
    to_poly,
)

PolyMap = Callable[[Poly], Poly]


@dataclass(frozen=True)
class OreAlgebra:
    """An Ore algebra defined by its ``sigma`` endomorphism and ``delta`` derivation."""

    name: str
    sigma: PolyMap
    delta: PolyMap

    def operator(self, coeffs: Sequence[Sequence[Fraction | int]]) -> OrePolynomial:
        """Build an operator ``sum_i c_i(x) d^i`` from per-order coefficient polynomials."""
        return OrePolynomial(self, tuple(to_poly(c) for c in coeffs))


@dataclass(frozen=True)
class OrePolynomial:
    """An element ``sum_i coeffs[i] d^i`` of an :class:`OreAlgebra`."""

    algebra: OreAlgebra
    coeffs: tuple[Poly, ...]

    @property
    def order(self) -> int:
        """The highest power of ``d`` with a non-zero coefficient (``-1`` for zero)."""
        for i in range(len(self.coeffs) - 1, -1, -1):
            if self.coeffs[i]:
                return i
        return -1

    def __add__(self, other: OrePolynomial) -> OrePolynomial:
        n = max(len(self.coeffs), len(other.coeffs))
        out: list[Poly] = []
        for i in range(n):
            a = self.coeffs[i] if i < len(self.coeffs) else ()
            b = other.coeffs[i] if i < len(other.coeffs) else ()
            out.append(padd(a, b))
        return OrePolynomial(self.algebra, tuple(out))

    def _theta_power_times_poly(self, i: int, p: Poly) -> list[Poly]:
        """``d^i * p`` as a coefficient list (index l = coefficient of ``d^l``)."""
        # d^0 * p = p
        result: list[Poly] = [p]
        for _ in range(i):
            # multiply current operator (sum r_l d^l) on the LEFT by d:
            # d * (r_l d^l) = sigma(r_l) d^{l+1} + delta(r_l) d^l
            nxt: list[Poly] = [()] * (len(result) + 1)
            for l_idx, r in enumerate(result):
                if not r:
                    continue
                nxt[l_idx + 1] = padd(nxt[l_idx + 1], self.algebra.sigma(r))
                nxt[l_idx] = padd(nxt[l_idx], self.algebra.delta(r))
            result = nxt
        return result

    def __mul__(self, other: OrePolynomial) -> OrePolynomial:
        acc: list[Poly] = []

        def _add_into(target: list[Poly], idx: int, poly: Poly) -> None:
            while len(target) <= idx:
                target.append(())
            target[idx] = padd(target[idx], poly)

        for i, a in enumerate(self.coeffs):
            if not a:
                continue
            for j, b in enumerate(other.coeffs):
                if not b:
                    continue
                # a d^i * b d^j = a * (d^i * b) * d^j
                theta_b = self._theta_power_times_poly(i, b)
                for l_idx, coeff in enumerate(theta_b):
                    if not coeff:
                        continue
                    _add_into(acc, l_idx + j, pmul(a, coeff))
        return OrePolynomial(self.algebra, tuple(acc))

    def apply_sequence(self, seq: Callable[[int], Fraction], n: int) -> Fraction:
        """Apply a **shift**-algebra operator to a sequence: ``sum_i c_i(n) seq(n+i)``."""
        total = Fraction(0)
        for i, c in enumerate(self.coeffs):
            if c:
                total += peval(c, n) * seq(n + i)
        return total

    def apply_series(self, coeffs: Sequence[Fraction], x_order: int) -> Fraction:
        """Apply a **differential**-algebra operator to a power series' Taylor coefficients.

        ``coeffs[m]`` is the ``x^m`` coefficient; returns the ``x^{x_order}`` coefficient of
        the image (``D`` lowers the index and multiplies by the falling factorial).
        """
        total = Fraction(0)
        for i, c in enumerate(self.coeffs):
            if not c:
                continue
            # D^i x^{x_order + i} contributes; coefficient of x^{x_order} in c(x) D^i f
            for d_deg, cc in enumerate(c):
                m = x_order - d_deg + i  # source Taylor index
                if 0 <= m < len(coeffs):
                    falling = Fraction(1)
                    for t in range(i):
                        falling *= m - t
                    total += cc * falling * coeffs[m]
        return total


def shift_algebra() -> OreAlgebra:
    """The shift Ore algebra ``sigma(p)(x)=p(x+1)``, ``delta=0`` (P-recursive recurrences)."""
    return OreAlgebra("shift", lambda p: pshift(p, 1), lambda _p: ())


def diff_algebra() -> OreAlgebra:
    """The differential Ore algebra ``sigma=id``, ``delta=d/dx`` (D-finite ODEs)."""
    return OreAlgebra("differential", lambda p: p, pderiv)


__all__ = [
    "OreAlgebra",
    "OrePolynomial",
    "diff_algebra",
    "shift_algebra",
]
