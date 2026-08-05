# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Rigorous affine arithmetic (zonotopes) on top of :class:`Interval`.

Interval arithmetic loses every correlation between operands -- it cannot see
that ``x - x = 0`` -- so repeated operations suffer the *dependency* and
*wrapping* blow-up.  Affine arithmetic (Stolfi & de Figueiredo) fixes this by
tracking each quantity as a first-order form in shared *noise symbols*
``eps_i in [-1, 1]``:

.. math::

    \hat x = x_0 + \sum_{i} x_i\,\varepsilon_i \; (\pm\, e\,\varepsilon_e),

where the named ``x_i`` carry correlations exactly and ``e \ge 0`` is an
*anonymous* radius that rigorously absorbs every rounding error and every
non-affine remainder.  Two forms that share ``eps_i`` cancel that component
exactly under subtraction, which is what defeats wrapping in the validated
Lohner flow and tightens the Taylor-model neural-net verifier.

Rigor
-----
Every floating-point coefficient is computed through :class:`Interval`'s
outward-directed rounding: the *midpoint* becomes the stored ``float`` and the
*radius* is accumulated into the anonymous error term ``e``.  Hence the stored
form provably encloses the true real quantity for **all** admissible noise
assignments.  :meth:`AffineForm.to_interval` collapses the form back to a
guaranteed enclosure ``[x_0 - R, x_0 + R]`` with
``R = sum_i |x_i| + e`` (rounded outward).

The non-affine operations (:meth:`AffineForm.apply_scalar` and the
:meth:`reciprocal` / :meth:`sqrt` built on it) linearise ``f`` with an arbitrary
slope ``alpha`` and bound the residual ``f(x) - alpha x`` by the *interval*
enclosure ``f([X]) - alpha[X]``; correctness is independent of the slope, so any
``alpha`` is sound and the secant slope merely minimises the added radius.
"""

from __future__ import annotations

from collections.abc import Callable
from itertools import count
from math import inf, nextafter

from omnibias.core.verified.interval import Interval, IntervalLike

_symbol_counter = count(1)


def new_noise_symbol() -> int:
    """Return a fresh, never-reused noise-symbol identifier."""
    return next(_symbol_counter)


def _succ(x: float) -> float:
    return nextafter(x, inf)


def _add_up(*xs: float) -> float:
    """Outward (upper) rounded sum of non-negative floats."""
    acc = 0.0
    for x in xs:
        acc = _succ(acc + x)
    return acc


def _mul_up(a: float, b: float) -> float:
    """Outward (upper) rounded product of non-negative floats."""
    return _succ(a * b)


class AffineForm:
    r"""A rigorous first-order affine form in shared noise symbols.

    Parameters
    ----------
    center:
        The central value ``x_0``.
    deviations:
        Mapping ``symbol_id -> coefficient`` of the *named* partial deviations.
        Zero coefficients are dropped on construction.
    error:
        The non-negative anonymous radius ``e`` accumulating rounding and
        non-affine remainders.  Treated as an independent symbol on every use.
    """

    __slots__ = ("center", "deviations", "error")

    def __init__(
        self,
        center: float,
        deviations: dict[int, float] | None = None,
        error: float = 0.0,
    ) -> None:
        self.center = float(center)
        if deviations:
            self.deviations = {k: float(v) for k, v in deviations.items() if v != 0.0}
        else:
            self.deviations = {}
        e = float(error)
        if e < 0.0:
            raise ValueError("affine error radius must be non-negative")
        self.error = e

    # ----- constructors -------------------------------------------------- #
    @classmethod
    def constant(cls, value: IntervalLike) -> AffineForm:
        """A degenerate form for a constant (rounding pushed into ``e``)."""
        iv = Interval.from_value(value)
        return cls(iv.mid, {}, iv.rad)

    @classmethod
    def symbol(cls, center: float, radius: float, symbol_id: int | None = None) -> AffineForm:
        """A form ``center + radius * eps`` on a fresh (or given) noise symbol."""
        if radius < 0.0:
            raise ValueError("symbol radius must be non-negative")
        sid = new_noise_symbol() if symbol_id is None else symbol_id
        return cls(center, {sid: float(radius)})

    @classmethod
    def from_interval(cls, iv: IntervalLike, symbol_id: int | None = None) -> AffineForm:
        """Convert an interval to a form, introducing one *fresh* noise symbol.

        Correlations start fresh: two intervals converted separately get
        independent symbols (no spurious cancellation).
        """
        interval = Interval.from_value(iv)
        return cls.symbol(interval.mid, interval.rad, symbol_id)

    # ----- queries ------------------------------------------------------- #
    def named_radius(self) -> float:
        """Outward-rounded ``sum_i |x_i|`` over the named deviations only."""
        return _add_up(*(abs(c) for c in self.deviations.values()))

    def radius(self) -> float:
        """Total outward-rounded radius ``sum_i |x_i| + e``."""
        return _add_up(self.named_radius(), self.error)

    def to_interval(self) -> Interval:
        """Guaranteed enclosure ``[center - R, center + R]`` (outward rounded)."""
        r = self.radius()
        return Interval.point(self.center) + Interval(-r, r)

    # ----- affine algebra ------------------------------------------------ #
    def __neg__(self) -> AffineForm:
        # Negation is exact in IEEE-754 (no rounding).
        return AffineForm(-self.center, {k: -v for k, v in self.deviations.items()}, self.error)

    def __add__(self, other: AffineForm | IntervalLike) -> AffineForm:
        o = other if isinstance(other, AffineForm) else AffineForm.constant(other)
        c_iv = Interval.point(self.center) + Interval.point(o.center)
        carry = c_iv.rad
        devs: dict[int, float] = {}
        for sid in self.deviations.keys() | o.deviations.keys():
            term = Interval.point(self.deviations.get(sid, 0.0)) + Interval.point(
                o.deviations.get(sid, 0.0)
            )
            if term.lo != 0.0 or term.hi != 0.0:
                devs[sid] = term.mid
                carry = _add_up(carry, term.rad)
        return AffineForm(c_iv.mid, devs, _add_up(self.error, o.error, carry))

    __radd__ = __add__

    def __sub__(self, other: AffineForm | IntervalLike) -> AffineForm:
        o = other if isinstance(other, AffineForm) else AffineForm.constant(other)
        return self.__add__(-o)

    def __rsub__(self, other: AffineForm | IntervalLike) -> AffineForm:
        o = other if isinstance(other, AffineForm) else AffineForm.constant(other)
        return o.__add__(-self)

    def __mul__(self, other: AffineForm | IntervalLike) -> AffineForm:
        o = other if isinstance(other, AffineForm) else AffineForm.constant(other)
        x0, y0 = self.center, o.center
        z_iv = Interval.point(x0) * Interval.point(y0)
        carry = z_iv.rad
        devs: dict[int, float] = {}
        for sid in self.deviations.keys() | o.deviations.keys():
            xi = self.deviations.get(sid, 0.0)
            yi = o.deviations.get(sid, 0.0)
            term = Interval.point(x0) * Interval.point(yi) + Interval.point(y0) * Interval.point(xi)
            if term.lo != 0.0 or term.hi != 0.0:
                devs[sid] = term.mid
                carry = _add_up(carry, term.rad)
        # Non-affine remainder (x - x0)(y - y0) plus the linear error-symbol
        # contributions |x0| e_y + |y0| e_x, all bounded outward.
        rx = _add_up(self.named_radius(), self.error)
        ry = _add_up(o.named_radius(), o.error)
        quad = _mul_up(rx, ry)
        lin_err = _add_up(_mul_up(abs(x0), o.error), _mul_up(abs(y0), self.error))
        return AffineForm(z_iv.mid, devs, _add_up(lin_err, quad, carry))

    __rmul__ = __mul__

    # ----- non-affine operations ----------------------------------------- #
    def apply_scalar(
        self,
        f_interval: Callable[[Interval], Interval],
        *,
        slope: float | None = None,
    ) -> AffineForm:
        r"""Rigorous affine enclosure of ``f(self)`` for a scalar function.

        ``f_interval`` must return a guaranteed enclosure of ``f`` over any input
        interval.  The form is linearised as ``alpha * self + beta`` and the
        residual ``f(x) - alpha x`` is bounded by the interval
        ``f([X]) - alpha[X]``; the slope only affects tightness, never soundness.
        """
        x = self.to_interval()
        if slope is None:
            if x.hi > x.lo:
                fa = f_interval(Interval.point(x.lo)).mid
                fb = f_interval(Interval.point(x.hi)).mid
                slope = (fb - fa) / (x.hi - x.lo)
            else:
                slope = 0.0
        residual = f_interval(x) - Interval.point(slope) * x
        z = self.__mul__(slope) + Interval.point(residual.mid)
        return AffineForm(z.center, z.deviations, _add_up(z.error, residual.rad))

    def reciprocal(self) -> AffineForm:
        """``1 / self`` (requires the enclosure to exclude zero)."""
        if self.to_interval().contains_zero():
            raise ZeroDivisionError("affine reciprocal requires 0 outside the enclosure")
        return self.apply_scalar(lambda iv: iv.reciprocal())

    def sqrt(self) -> AffineForm:
        """``sqrt(self)`` (requires a non-negative enclosure)."""
        return self.apply_scalar(lambda iv: iv.sqrt())

    def __truediv__(self, other: AffineForm | IntervalLike) -> AffineForm:
        o = other if isinstance(other, AffineForm) else AffineForm.constant(other)
        return self.__mul__(o.reciprocal())

    def __rtruediv__(self, other: AffineForm | IntervalLike) -> AffineForm:
        o = other if isinstance(other, AffineForm) else AffineForm.constant(other)
        return o.__mul__(self.reciprocal())

    def __pow__(self, n: int) -> AffineForm:
        if n < 0:
            return self.reciprocal().__pow__(-n)
        result = AffineForm.constant(1.0)
        base = self
        e = n
        while e > 0:
            if e & 1:
                result = result * base
            e >>= 1
            if e > 0:
                base = base * base
        return result

    def __repr__(self) -> str:
        body = " + ".join(f"{c!r}*e{sid}" for sid, c in sorted(self.deviations.items()))
        tail = f" +/- {self.error!r}" if self.error else ""
        return f"AffineForm({self.center!r}{' + ' + body if body else ''}{tail})"


def from_intervals(ivs: list[IntervalLike]) -> list[AffineForm]:
    """Convert a list of intervals to forms, each on its own fresh symbol."""
    return [AffineForm.from_interval(iv) for iv in ivs]


__all__ = [
    "AffineForm",
    "from_intervals",
    "new_noise_symbol",
]
