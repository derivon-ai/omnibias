# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Rigorous floating-point interval arithmetic (pure stdlib).

Every operation returns an :class:`Interval` that is *guaranteed* to enclose the
true real result for **all** points in the input intervals.  Rigor comes from
*outward directed rounding*: an endpoint computed in IEEE round-to-nearest is
pushed one representable step outward with :func:`math.nextafter`.  For a single
IEEE-754 operation the rounding error is ``<= 0.5 ulp`` while one
:func:`math.nextafter` step is ``1 ulp`` in the outward direction, so the
inflated endpoints provably bracket the exact result.

This is the dependency-free substrate named in the omnibias certified-evidence backend
contract (``mpfr_or_arb_outward_interval_backend``); the transcendental
enclosures in :mod:`omnibias.core.verified.transcend` optionally sharpen the
base-point evaluations with ``mpmath`` when it is installed, but the algebra
here needs only the standard library.

Notes
-----
* Powers use binary exponentiation by repeated interval multiplication so the
  enclosure is rigorous for *every* exponent (a single ``nextafter`` step would
  not cover the multi-rounding error of ``float.__pow__``).
* Division and reciprocal require the denominator interval to be bounded away
  from zero; otherwise :class:`ZeroDivisionError` is raised.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import inf, isnan, nextafter
from typing import Union

#: Anything that can be promoted to an :class:`Interval`.
IntervalLike = Union["Interval", int, float, Fraction]


def _pred(x: float) -> float:
    """Next representable double below ``x`` (towards ``-inf``)."""
    return nextafter(x, -inf)


def _succ(x: float) -> float:
    """Next representable double above ``x`` (towards ``+inf``)."""
    return nextafter(x, inf)


@dataclass(frozen=True)
class Interval:
    """A closed real interval ``[lo, hi]`` with ``lo <= hi`` and no ``NaN``.

    The constructor performs no rounding -- it assumes the endpoints already
    bracket the intended quantity.

    Injecting scalars (mind the difference):

    - :meth:`from_rational` -- and :meth:`from_value` for ``int`` / ``Fraction``
      -- returns the *tightest outward-rounded* float interval enclosing the
      exact rational, e.g. ``from_rational(Fraction(1, 10))`` straddles the true
      ``1/10`` with ``lo < 1/10 < hi``.
    - :meth:`point` -- and :meth:`from_value` for ``float`` -- injects the float
      as an *exact point* ``[x, x]`` with **no rounding**. This is rigorous when
      the float itself **is** the intended datum (a matrix entry, a measured
      coefficient). It does *not* enclose a different real that the float merely
      approximates: ``point(0.1)`` is the double ``0.1``, not the rational
      ``1/10`` -- use ``from_rational(Fraction(1, 10))`` for that.
    """

    lo: float
    hi: float

    def __post_init__(self) -> None:
        if isnan(self.lo) or isnan(self.hi):
            raise ValueError("interval endpoints must not be NaN")
        if self.lo > self.hi:
            raise ValueError(f"empty interval: lo={self.lo!r} > hi={self.hi!r}")

    # ----- constructors -------------------------------------------------- #
    @classmethod
    def point(cls, x: float) -> Interval:
        """Exact degenerate interval ``[x, x]`` for the float ``x`` itself.

        No rounding is applied, so this is rigorous only when ``x`` is the
        intended datum. To enclose a real that ``x`` merely approximates (e.g.
        the rational ``1/10``), use :meth:`from_rational` instead.
        """
        xf = float(x)
        return cls(xf, xf)

    @classmethod
    def from_rational(cls, q: Fraction | int) -> Interval:
        """Tightest float interval enclosing an exact rational/integer ``q``."""
        frac = Fraction(q)
        f = float(frac)
        lo = f
        hi = f
        as_frac = Fraction(f)
        if as_frac > frac:
            lo = _pred(f)
        elif as_frac < frac:
            hi = _succ(f)
        return cls(lo, hi)

    @classmethod
    def from_value(cls, value: IntervalLike) -> Interval:
        """Promote any :data:`IntervalLike` to an :class:`Interval`.

        ``int`` / ``Fraction`` go through :meth:`from_rational` (tightest
        outward-rounded enclosure of the exact rational); a ``float`` is
        injected as an *exact point* ``[x, x]`` via :meth:`point` -- no
        rounding, which is correct when the float is the literal datum (see the
        class docstring for the rational-vs-float caveat).
        """
        if isinstance(value, Interval):
            return value
        if isinstance(value, int | Fraction):
            return cls.from_rational(value)
        return cls.point(float(value))

    @classmethod
    def hull(cls, *values: IntervalLike) -> Interval:
        """Smallest interval containing every argument."""
        if not values:
            raise ValueError("hull requires at least one value")
        ivs = [cls.from_value(v) for v in values]
        return cls(min(iv.lo for iv in ivs), max(iv.hi for iv in ivs))

    # ----- queries ------------------------------------------------------- #
    @property
    def mid(self) -> float:
        """A representable point near the midpoint (always inside the interval)."""
        m = 0.5 * (self.lo + self.hi)
        if m < self.lo:
            return self.lo
        if m > self.hi:
            return self.hi
        return m

    @property
    def rad(self) -> float:
        """An outward-rounded radius: ``hi - mid`` (>= true radius)."""
        m = self.mid
        return _succ(max(m - self.lo, self.hi - m))

    @property
    def width(self) -> float:
        """Outward-rounded width ``hi - lo``."""
        return _succ(self.hi - self.lo)

    @property
    def mag(self) -> float:
        """Magnitude ``max |x|`` over the interval (outward rounded)."""
        return _succ(max(abs(self.lo), abs(self.hi)))

    @property
    def mig(self) -> float:
        """Mignitude ``min |x|`` over the interval (0 if it straddles 0)."""
        if self.lo <= 0.0 <= self.hi:
            return 0.0
        return _pred(min(abs(self.lo), abs(self.hi)))

    def contains(self, x: float) -> bool:
        return self.lo <= x <= self.hi

    def contains_zero(self) -> bool:
        return self.lo <= 0.0 <= self.hi

    # ----- arithmetic ---------------------------------------------------- #
    def __add__(self, other: IntervalLike) -> Interval:
        o = Interval.from_value(other)
        return Interval(_pred(self.lo + o.lo), _succ(self.hi + o.hi))

    __radd__ = __add__

    def __neg__(self) -> Interval:
        return Interval(-self.hi, -self.lo)

    def __sub__(self, other: IntervalLike) -> Interval:
        o = Interval.from_value(other)
        return Interval(_pred(self.lo - o.hi), _succ(self.hi - o.lo))

    def __rsub__(self, other: IntervalLike) -> Interval:
        return Interval.from_value(other).__sub__(self)

    def __mul__(self, other: IntervalLike) -> Interval:
        o = Interval.from_value(other)
        products = (
            self.lo * o.lo,
            self.lo * o.hi,
            self.hi * o.lo,
            self.hi * o.hi,
        )
        return Interval(_pred(min(products)), _succ(max(products)))

    __rmul__ = __mul__

    def reciprocal(self) -> Interval:
        """``1 / self``; requires the interval to exclude zero."""
        if self.contains_zero():
            raise ZeroDivisionError("interval reciprocal requires 0 outside [lo, hi]")
        return Interval(_pred(1.0 / self.hi), _succ(1.0 / self.lo))

    def __truediv__(self, other: IntervalLike) -> Interval:
        return self.__mul__(Interval.from_value(other).reciprocal())

    def __rtruediv__(self, other: IntervalLike) -> Interval:
        return Interval.from_value(other).__mul__(self.reciprocal())

    def __pow__(self, n: int) -> Interval:
        return self.pow_int(n)

    def pow_int(self, n: int) -> Interval:
        """Rigorous integer power via interval binary exponentiation.

        For an even ``n`` the base is first folded through :meth:`abs`, so a
        sign-straddling interval yields the tight ``[0, mag**n]`` enclosure rather
        than a loose one polluted by spurious negative corner products (``x**n`` is
        non-negative for even ``n``). This only tightens the result; non-straddling
        intervals are unaffected.
        """
        if n < 0:
            return self.pow_int(-n).reciprocal()
        result = Interval.point(1.0)
        base = self.abs() if n % 2 == 0 else self
        e = n
        while e > 0:
            if e & 1:
                result = result * base
            e >>= 1
            if e > 0:
                base = base * base
        return result

    def abs(self) -> Interval:
        """Enclosure of ``|x|`` over the interval."""
        if self.lo >= 0.0:
            return self
        if self.hi <= 0.0:
            return -self
        return Interval(0.0, _succ(max(-self.lo, self.hi)))

    def sqrt(self) -> Interval:
        """Enclosure of ``sqrt(x)``; requires ``lo >= 0``.

        ``math.sqrt`` is an IEEE-754 *correctly rounded* operation (error
        ``<= 0.5 ulp``), so a single outward ``nextafter`` step per endpoint is a
        rigorous bracket. This is unlike the libm transcendentals (``exp`` /
        ``log``), which are not guaranteed correctly rounded and therefore need the
        wider :mod:`omnibias.core.verified.transcend` enclosures.
        """
        from math import sqrt as _sqrt

        if self.lo < 0.0:
            raise ValueError("sqrt requires a non-negative interval")
        return Interval(_pred(_sqrt(self.lo)), _succ(_sqrt(self.hi)))

    def intersect(self, other: Interval) -> Interval:
        lo = max(self.lo, other.lo)
        hi = min(self.hi, other.hi)
        if lo > hi:
            raise ValueError("empty intersection")
        return Interval(lo, hi)

    def __repr__(self) -> str:
        return f"Interval([{self.lo!r}, {self.hi!r}])"


def hull(values: list[IntervalLike]) -> Interval:
    """Functional alias for :meth:`Interval.hull`."""
    return Interval.hull(*values)


def sum_intervals(values: list[Interval]) -> Interval:
    """Outward-rounded sum of a list of intervals (left fold)."""
    acc = Interval.point(0.0)
    for v in values:
        acc = acc + v
    return acc


__all__ = [
    "Interval",
    "IntervalLike",
    "hull",
    "sum_intervals",
]
