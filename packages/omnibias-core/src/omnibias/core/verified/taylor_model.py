# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Taylor models: rigorous function enclosures over a whole interval cell.

A degree-``d`` Taylor model on a cell ``[center - h, center + h]`` is a pair

.. math::

    f(x) \in P(x - center) + R \quad \forall x \in [center - h, center + h],

where ``P`` is a polynomial with :class:`Interval` coefficients (in the *relative*
variable ``X = x - center``) and ``R`` is an :class:`Interval` *remainder* that
rigorously absorbs everything the polynomial omits.  Unlike a bare interval, a
Taylor model keeps the *shape* of the function, so multiplying / composing does
not suffer the wrapping blow-up of naive interval evaluation and the
:meth:`bound` over the cell stays tight.

This is the object that discharges the certified-evidence ``analytic_tail_operator_bound``
and rigorous-quadrature obligations: the truncation remainder ``R`` *is* the
certified tail term.
"""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction

from omnibias.core.verified.interval import Interval, IntervalLike, _pred, _succ


def _conv(a: Sequence[Interval], b: Sequence[Interval]) -> list[Interval]:
    out = [Interval.point(0.0) for _ in range(len(a) + len(b) - 1)]
    for i, ai in enumerate(a):
        for j, bj in enumerate(b):
            out[i + j] = out[i + j] + ai * bj
    return out


def _cell_rel(radius: float) -> Interval:
    """Outward enclosure of the relative cell ``[-radius, radius]``.

    Negating a finite double is exact, but the rigorous register still inflates
    each endpoint one ulp so every consumer (Horner bound, remainder integral,
    geometric tail) sees a *provable* enclosure of the claimed cell rather than
    a bare float injection that could sit on the wrong side of a later rounding.
    """
    r = float(radius)
    if r < 0.0:
        raise ValueError("cell radius must be non-negative")
    if r == 0.0:
        return Interval.point(0.0)
    return Interval(_pred(-r), _succ(r))


class TaylorModel:
    """Degree-``order`` Taylor model on a symmetric cell about ``center``."""

    __slots__ = ("center", "radius", "order", "coeffs", "remainder")

    def __init__(
        self,
        center: float,
        radius: float,
        coeffs: Sequence[Interval],
        remainder: Interval,
    ) -> None:
        if radius < 0.0:
            raise ValueError("cell radius must be non-negative")
        if not coeffs:
            raise ValueError("a Taylor model needs at least one coefficient")
        self.center = float(center)
        self.radius = float(radius)
        self.order = len(coeffs) - 1
        self.coeffs: list[Interval] = list(coeffs)
        self.remainder = remainder

    # ----- constructors -------------------------------------------------- #
    @classmethod
    def constant(cls, value: IntervalLike, center: float, radius: float, order: int) -> TaylorModel:
        coeffs = [Interval.from_value(value)] + [Interval.point(0.0) for _ in range(order)]
        return cls(center, radius, coeffs, Interval.point(0.0))

    @classmethod
    def identity(cls, center: float, radius: float, order: int) -> TaylorModel:
        """The model of ``f(x) = x`` (``= center + X``) on the cell."""
        coeffs = [Interval.point(center)] + [Interval.point(0.0) for _ in range(order)]
        if order >= 1:
            coeffs[1] = Interval.point(1.0)
        return cls(center, radius, coeffs, Interval.point(0.0))

    # ----- helpers ------------------------------------------------------- #
    def _rel(self) -> Interval:
        """The relative variable ``X = x - center`` over the cell.

        Always routed through :func:`_cell_rel` so the cell can never be rebuilt
        inconsistently and every endpoint is outward-inflated.
        """
        return _cell_rel(self.radius)

    def _check(self, other: TaylorModel) -> None:
        if self.center != other.center or self.radius != other.radius:
            raise ValueError("Taylor models must share the same cell")
        if self.order != other.order:
            raise ValueError("Taylor models must share the same truncation order")

    def bound(self) -> Interval:
        """Guaranteed enclosure of ``f`` over the whole cell."""
        x = self._rel()
        acc = self.coeffs[-1]
        for c in reversed(self.coeffs[:-1]):
            acc = acc * x + c
        return acc + self.remainder

    # ----- arithmetic ---------------------------------------------------- #
    def __add__(self, other: TaylorModel | IntervalLike) -> TaylorModel:
        if isinstance(other, TaylorModel):
            self._check(other)
            coeffs = [a + b for a, b in zip(self.coeffs, other.coeffs, strict=True)]
            return TaylorModel(self.center, self.radius, coeffs, self.remainder + other.remainder)
        coeffs = list(self.coeffs)
        coeffs[0] = coeffs[0] + Interval.from_value(other)
        return TaylorModel(self.center, self.radius, coeffs, self.remainder)

    __radd__ = __add__

    def __neg__(self) -> TaylorModel:
        return TaylorModel(
            self.center, self.radius, [-c for c in self.coeffs], -self.remainder
        )

    def __sub__(self, other: TaylorModel | IntervalLike) -> TaylorModel:
        return self.__add__(-other if isinstance(other, TaylorModel) else -Interval.from_value(other))

    def __mul__(self, other: TaylorModel | IntervalLike) -> TaylorModel:
        if not isinstance(other, TaylorModel):
            scal = Interval.from_value(other)
            return TaylorModel(
                self.center, self.radius, [c * scal for c in self.coeffs], self.remainder * scal
            )
        self._check(other)
        d = self.order
        full = _conv(self.coeffs, other.coeffs)  # length 2d + 1
        low = full[: d + 1]
        x = self._rel()
        # Bound the truncated high part  sum_{k=d+1}^{2d} full[k] * X^k  over the cell
        # by factoring out X^{d+1} and Horner-evaluating the inner polynomial.
        high = Interval.point(0.0)
        if d >= 1 and len(full) > d + 1:
            inner = full[2 * d]
            for k in range(2 * d - 1, d, -1):
                inner = inner * x + full[k]
            high = inner * x.pow_int(d + 1)
        # f*g = P*Q + P*Rg + Q*Rf + Rf*Rg, with P,Q the polynomial parts.
        remainder = (
            high
            + self._poly_bound() * other.remainder
            + other._poly_bound() * self.remainder
            + self.remainder * other.remainder
        )
        return TaylorModel(self.center, self.radius, low, remainder)

    __rmul__ = __mul__

    def _poly_bound(self) -> Interval:
        """Enclosure of the polynomial part only (no remainder)."""
        x = self._rel()
        acc = self.coeffs[-1]
        for c in reversed(self.coeffs[:-1]):
            acc = acc * x + c
        return acc

    def antiderivative(self) -> TaylorModel:
        r"""Rigorous antiderivative ``F(x) = \int_{center}^{x} f(t)\,dt`` (so ``F(center)=0``).

        Integration is the natural *sound* operation on a Taylor model (unlike
        differentiation, whose action on a flat remainder is uncontrolled).  The
        polynomial part integrates term-by-term in the relative variable
        ``X = x - center`` -- raising the degree by one, so the result has order
        ``order + 1`` -- and the remainder integrates rigorously,

        .. math::

            \int_0^X R\,dt \in R \cdot [-h, h]
            \qquad (h = \text{radius}),

        because ``\int_0^X e(t)\,dt = X\cdot\bar e`` with the mean value
        ``\bar e \in R`` for any ``e`` enclosed by ``R``.  Composing
        :meth:`antiderivative` with :meth:`bound` gives a certified definite
        integral over the cell.
        """
        new_coeffs = [Interval.point(0.0)]
        for k, c in enumerate(self.coeffs):
            new_coeffs.append(c * Interval.from_rational(Fraction(1, k + 1)))
        rem = self.remainder * self._rel()
        return TaylorModel(self.center, self.radius, new_coeffs, rem)

    def definite_integral(self) -> Interval:
        r"""Certified ``\int_{center-h}^{center+h} f(x)\,dx`` over the whole cell.

        The symmetric cell kills the odd polynomial terms
        (``\int_{-h}^{h} X^k\,dX = 0`` for odd ``k``, ``= 2 h^{k+1}/(k+1)`` for
        even ``k``); the remainder contributes ``R \cdot 2h`` since
        ``\int_{-h}^{h} e(X)\,dX = 2h\,\bar e`` with ``\bar e \in R``.
        """
        # Use the outward cell radius (``.hi`` of ``_rel``) rather than a bare
        # ``Interval.point(radius)`` so the integral width cannot under-cover the
        # claimed cell by a ulp.
        h = Interval.point(self._rel().hi)
        acc = Interval.point(0.0)
        for k, c in enumerate(self.coeffs):
            if k % 2 == 0:
                acc = acc + c * h.pow_int(k + 1) * Interval.from_rational(
                    Fraction(2, k + 1)
                )
        return acc + self.remainder * (h * Interval.from_rational(2))

    def pow_int(self, n: int) -> TaylorModel:
        if n < 0:
            raise ValueError("Taylor-model powers require n >= 0")
        result = TaylorModel.constant(1.0, self.center, self.radius, self.order)
        base = self
        e = n
        while e > 0:
            if e & 1:
                result = result * base
            e >>= 1
            if e > 0:
                base = base * base
        return result

    def reciprocal(self) -> TaylorModel:
        r"""Rigorous Taylor model of ``1 / f`` over the whole cell.

        Factor ``1/f = (1/c0) * 1/(1 + g)`` about the scalar expansion point
        ``c0 = mid(coeffs[0])`` with ``g = (f - c0)/c0`` (which has a near-zero
        constant term), then expand ``1/(1 + g) = sum_{k>=0} (-g)^k``.  The first
        ``order + 1`` terms are kept as a Taylor model; the omitted analytic tail
        is bounded by the geometric series ``tau^{order+1} / (1 - tau)`` where
        ``tau`` rigorously bounds ``|g|`` over the cell.  Requires ``f`` to stay
        bounded away from ``0`` on the cell and the cell to be narrow enough that
        the relative variation ``tau < 1`` (otherwise :class:`ValueError`).
        """
        rng = self.bound()
        if rng.lo <= 0.0 <= rng.hi:
            raise ValueError(
                "TaylorModel.reciprocal requires f to be bounded away from 0 on the cell"
            )
        c0 = Interval.point(self.coeffs[0].mid)
        if c0.lo == 0.0:
            raise ValueError(
                "TaylorModel.reciprocal needs a non-zero expansion point; narrow the cell"
            )
        inv_c0 = c0.reciprocal()
        shifted = [self.coeffs[0] - c0, *self.coeffs[1:]]
        f_minus_c0 = TaylorModel(self.center, self.radius, shifted, self.remainder)
        g = f_minus_c0 * inv_c0
        # `.mag` is the outward-rounded magnitude accessor (a hand-rolled
        # max(abs(lo), abs(hi)) would sit one ulp lower); the geometric tail below
        # grows with tau, so only the outward value keeps the remainder sound.
        tau = g.bound().mag
        if tau >= 1.0:
            raise ValueError(
                "TaylorModel.reciprocal series does not converge on this cell "
                "(narrow the cell so the relative variation is < 1)"
            )
        neg_g = -g
        series = TaylorModel.constant(1.0, self.center, self.radius, self.order)
        term = TaylorModel.constant(1.0, self.center, self.radius, self.order)
        for _ in range(self.order):
            term = term * neg_g
            series = series + term
        tail = float(
            (
                Interval.point(tau).pow_int(self.order + 1)
                * (Interval.point(1.0) - Interval.point(tau)).reciprocal()
            ).hi
        )
        series = TaylorModel(
            self.center,
            self.radius,
            series.coeffs,
            series.remainder + Interval(-tail, tail),
        )
        return series * inv_c0

    def sqrt(self) -> TaylorModel:
        r"""Rigorous Taylor model of ``sqrt(f)`` over the whole cell.

        Factor ``sqrt(f) = sqrt(c0) * sqrt(1 + g)`` about the scalar expansion
        point ``c0 = mid(coeffs[0]) > 0`` with ``g = (f - c0)/c0`` (relative
        variation, ``|g| <= tau``), then keep the first ``order + 1`` terms of the
        binomial series ``sqrt(1 + g) = sum_k binom(1/2, k) g^k``.  The omitted
        analytic tail is bounded by the Lagrange remainder

        .. math::

            |R| \le \Big|\binom{1/2}{m+1}\Big|\,(1 - \tau)^{1/2 - (m+1)}\,\tau^{m+1}
            \qquad (m = \text{order}),

        valid because ``1 + g \ge 1 - tau > 0`` on the cell and the ``(m+1)``-st
        derivative of ``sqrt(1+\cdot)`` is monotone there.  Requires ``f > 0`` on
        the cell and ``tau < 1`` (otherwise :class:`ValueError`).
        """
        rng = self.bound()
        if rng.lo <= 0.0:
            raise ValueError("TaylorModel.sqrt requires f > 0 on the cell")
        c0 = Interval.point(self.coeffs[0].mid)
        if c0.lo <= 0.0:
            raise ValueError("TaylorModel.sqrt needs a positive expansion point; narrow the cell")
        inv_c0 = c0.reciprocal()
        shifted = [self.coeffs[0] - c0, *self.coeffs[1:]]
        g = TaylorModel(self.center, self.radius, shifted, self.remainder) * inv_c0
        tau = g.bound().mag
        if tau >= 1.0:
            raise ValueError(
                "TaylorModel.sqrt series does not converge on this cell "
                "(narrow the cell so the relative variation is < 1)"
            )
        series = TaylorModel.constant(1.0, self.center, self.radius, self.order)
        gpow = TaylorModel.constant(1.0, self.center, self.radius, self.order)
        coeff = Fraction(1, 1)
        for k in range(1, self.order + 1):
            coeff = coeff * (Fraction(1, 2) - (k - 1)) / k
            gpow = gpow * g
            series = series + gpow * Interval.from_rational(coeff)
        next_coeff = abs(coeff * (Fraction(1, 2) - self.order) / (self.order + 1))
        one_minus = Interval.point(1.0) - Interval.point(tau)
        # (1 - tau)^{1/2 - (order+1)} = 1 / ((1 - tau)^order * sqrt(1 - tau))
        denom = one_minus.pow_int(self.order) * one_minus.sqrt()
        tail = float(
            (
                Interval.from_rational(next_coeff)
                * Interval.point(tau).pow_int(self.order + 1)
                * denom.reciprocal()
            ).hi
        )
        series = TaylorModel(
            self.center,
            self.radius,
            series.coeffs,
            series.remainder + Interval(-tail, tail),
        )
        return series * c0.sqrt()

    def __truediv__(self, other: TaylorModel | IntervalLike) -> TaylorModel:
        if isinstance(other, TaylorModel):
            self._check(other)
            return self * other.reciprocal()
        return self * Interval.from_value(other).reciprocal()

    def __repr__(self) -> str:
        return (
            f"TaylorModel(center={self.center!r}, radius={self.radius!r}, "
            f"order={self.order}, remainder={self.remainder!r})"
        )


__all__ = ["TaylorModel"]
