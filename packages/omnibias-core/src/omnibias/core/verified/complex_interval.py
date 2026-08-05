# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Rigorous *complex* interval arithmetic (rectangular enclosures).

A :class:`ComplexInterval` is a rectangle ``[re] + i[im]`` in the complex plane,
built from two real :class:`~omnibias.core.verified.interval.Interval`\s, that
*guaranteed* encloses the true complex result of every operation (each part is
computed with the outward-rounded real interval algebra).  This is the substrate
the verified Fourier series (:mod:`omnibias.core.verified.fourier`) needs:
Fourier coefficients are complex, and the nonlocal symbols that matter -- Riesz
``i k_j/|k|`` and Leray ``\delta_{ab} - k_a k_b/|k|^2`` -- act on them by complex
multiplication.

Rectangular (as opposed to disc/midpoint-radius) enclosures keep the
implementation purely on top of the existing real :class:`Interval`, at the cost
of some dependency overestimation in :meth:`__mul__` (the usual price of the
naive complex product); for the per-mode multiplier application and the
coefficient products in a truncated convolution this is tight enough, and the
weighted-norm tail accounting never relies on cancellation anyway.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

from omnibias.core.verified.interval import Interval, IntervalLike

#: Anything promotable to a :class:`ComplexInterval`.
ComplexLike = Union["ComplexInterval", Interval, complex, float, int]


@dataclass(frozen=True)
class ComplexInterval:
    """A rigorous rectangular enclosure ``[re] + i[im]`` of a complex quantity."""

    re: Interval
    im: Interval

    # ----- constructors -------------------------------------------------- #
    @classmethod
    def point(cls, value: complex | float | int) -> ComplexInterval:
        """Exact degenerate enclosure of the complex/real number ``value``."""
        z = complex(value)
        return cls(Interval.point(z.real), Interval.point(z.imag))

    @classmethod
    def from_parts(cls, re: IntervalLike, im: IntervalLike = 0.0) -> ComplexInterval:
        """Enclosure from real/imaginary :data:`IntervalLike` parts."""
        return cls(Interval.from_value(re), Interval.from_value(im))

    @classmethod
    def zero(cls) -> ComplexInterval:
        return cls(Interval.point(0.0), Interval.point(0.0))

    @classmethod
    def one(cls) -> ComplexInterval:
        return cls(Interval.point(1.0), Interval.point(0.0))

    @classmethod
    def imag_unit(cls) -> ComplexInterval:
        """The imaginary unit ``i``."""
        return cls(Interval.point(0.0), Interval.point(1.0))

    @classmethod
    def from_value(cls, value: ComplexLike) -> ComplexInterval:
        if isinstance(value, ComplexInterval):
            return value
        if isinstance(value, Interval):
            return cls(value, Interval.point(0.0))
        if isinstance(value, complex):
            return cls.point(value)
        return cls(Interval.from_value(value), Interval.point(0.0))

    # ----- arithmetic ---------------------------------------------------- #
    def __add__(self, other: ComplexLike) -> ComplexInterval:
        o = ComplexInterval.from_value(other)
        return ComplexInterval(self.re + o.re, self.im + o.im)

    __radd__ = __add__

    def __neg__(self) -> ComplexInterval:
        return ComplexInterval(-self.re, -self.im)

    def __sub__(self, other: ComplexLike) -> ComplexInterval:
        return self.__add__(-ComplexInterval.from_value(other))

    def __rsub__(self, other: ComplexLike) -> ComplexInterval:
        return ComplexInterval.from_value(other).__add__(-self)

    def __mul__(self, other: ComplexLike) -> ComplexInterval:
        o = ComplexInterval.from_value(other)
        # (a + bi)(c + di) = (ac - bd) + (ad + bc) i
        re = self.re * o.re - self.im * o.im
        im = self.re * o.im + self.im * o.re
        return ComplexInterval(re, im)

    __rmul__ = __mul__

    def __truediv__(self, other: ComplexLike) -> ComplexInterval:
        o = ComplexInterval.from_value(other)
        # z / w = z * conj(w) / |w|^2
        denom = o.re * o.re + o.im * o.im  # real, must exclude 0
        num = self * o.conj()
        return ComplexInterval(num.re / denom, num.im / denom)

    def conj(self) -> ComplexInterval:
        return ComplexInterval(self.re, -self.im)

    # ----- magnitude ----------------------------------------------------- #
    @property
    def mag(self) -> float:
        """Outward-rounded **upper** bound on ``|z|`` over the rectangle.

        The modulus is maximised at a corner, ``sqrt(mag(re)^2 + mag(im)^2)``.
        """
        rr = Interval.point(self.re.mag)
        ii = Interval.point(self.im.mag)
        arg = rr.pow_int(2) + ii.pow_int(2)
        # squaring an exact 0 rounds the lower endpoint one ulp below 0; the
        # radicand is a true sum of squares, so clamp before the sqrt.
        arg = Interval(max(arg.lo, 0.0), arg.hi)
        return arg.sqrt().hi

    def modulus(self) -> Interval:
        """Enclosure ``[min |z|, max |z|]`` of the modulus over the rectangle.

        The minimum is the origin-to-rectangle distance
        ``sqrt(mig(re)^2 + mig(im)^2)`` (``0`` when the rectangle straddles the
        respective axis), the maximum the corner value :attr:`mag`.
        """
        arg = (
            Interval.point(self.re.mig).pow_int(2)
            + Interval.point(self.im.mig).pow_int(2)
        )
        arg = Interval(max(arg.lo, 0.0), arg.hi)
        return Interval(max(arg.sqrt().lo, 0.0), self.mag)

    def contains(self, z: complex | float | int) -> bool:
        zz = complex(z)
        return self.re.contains(zz.real) and self.im.contains(zz.imag)

    def __repr__(self) -> str:
        return f"ComplexInterval(re={self.re!r}, im={self.im!r})"


__all__ = ["ComplexInterval", "ComplexLike"]
