# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Verified initial-value ODE integration -- the rigorous *flow* / shooting engine.

A self-similar blow-up profile is the solution of a boundary-value problem; the
*rate* is fixed by a scalar connection condition ``g(c) = 0`` evaluated on that
solution (see :mod:`omnibias.core.verified.rootfind`).  To feed the rate selector a
*genuine* connection functional we must integrate the profile ODE **rigorously** --
not float64 RK -- from the regular origin out to the matching point.

This module provides a validated Taylor-series integrator for autonomous systems
``Y'(t) = F(Y(t))`` (a non-autonomous field is handled by appending a clock state
``x' = 1``).  One step ``t -> t + h`` at order ``p`` is the textbook Lohner scheme:

1. **Taylor coefficients** ``U_0..U_p`` of the solution at the step start, via the
   recurrence ``(k+1) U_{k+1} = [F(Y(.))]_k`` -- the ``k``-th Taylor coefficient of
   ``F`` composed with the partial series -- evaluated by interval *automatic Taylor
   arithmetic* (:class:`TaylorSeries`).
2. **A-priori enclosure** ``[Z]`` of the whole step: a box with
   ``Y0 + [0,h] F([Z]) subset [Z]`` (Picard / Banach), so the true solution provably
   stays in ``[Z]`` for ``t in [0, h]``.
3. **Validated remainder**: ``Y(h) in sum_{k<=p} U_k h^k + U_{p+1}([Z]) h^{p+1}``,
   where ``U_{p+1}([Z])`` is the order-``p+1`` coefficient recomputed with the
   expansion point ranging over ``[Z]`` (Taylor's theorem, Lagrange form, derivative
   enclosed over the step).

Everything is an outward-rounded :class:`Interval`, so the returned end enclosure is
a theorem-grade bound on the flow.  Wrapping is controlled only by step size / order
here (no QR Lohner), which is ample for scalar / low-dimensional shooting.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from fractions import Fraction

from omnibias.core.verified.interval import Interval, IntervalLike

#: An autonomous vector field acting on a list of component Taylor series.
VectorField = Callable[[list["TaylorSeries"]], list["TaylorSeries"]]


class TaylorSeries:
    """A truncated power series with :class:`Interval` coefficients.

    Supports the algebra needed to express polynomial / rational vector fields:
    ``+``, ``-``, ``*`` (series*series convolution and series*scalar).  All operands
    in a product are kept at a common length; the product is truncated to it.
    """

    __slots__ = ("coeffs",)

    def __init__(self, coeffs: Sequence[Interval]) -> None:
        if not coeffs:
            raise ValueError("TaylorSeries needs at least one coefficient")
        self.coeffs: tuple[Interval, ...] = tuple(coeffs)

    @classmethod
    def constant(cls, value: IntervalLike, order: int) -> TaylorSeries:
        """Series ``[value, 0, ..., 0]`` of length ``order + 1``."""
        c0 = Interval.from_value(value)
        zero = Interval.point(0.0)
        return cls([c0, *([zero] * order)])

    @property
    def order(self) -> int:
        return len(self.coeffs) - 1

    def __len__(self) -> int:
        return len(self.coeffs)

    def __add__(self, other: TaylorSeries | IntervalLike) -> TaylorSeries:
        if isinstance(other, TaylorSeries):
            n = min(len(self.coeffs), len(other.coeffs))
            return TaylorSeries([self.coeffs[i] + other.coeffs[i] for i in range(n)])
        shifted = list(self.coeffs)
        shifted[0] = shifted[0] + Interval.from_value(other)
        return TaylorSeries(shifted)

    __radd__ = __add__

    def __neg__(self) -> TaylorSeries:
        return TaylorSeries([-c for c in self.coeffs])

    def __sub__(self, other: TaylorSeries | IntervalLike) -> TaylorSeries:
        if isinstance(other, TaylorSeries):
            n = min(len(self.coeffs), len(other.coeffs))
            return TaylorSeries([self.coeffs[i] - other.coeffs[i] for i in range(n)])
        shifted = list(self.coeffs)
        shifted[0] = shifted[0] - Interval.from_value(other)
        return TaylorSeries(shifted)

    def __rsub__(self, other: IntervalLike) -> TaylorSeries:
        return (-self).__add__(other)

    def __mul__(self, other: TaylorSeries | IntervalLike) -> TaylorSeries:
        if isinstance(other, TaylorSeries):
            n = min(len(self.coeffs), len(other.coeffs))
            out: list[Interval] = []
            for k in range(n):
                acc = Interval.point(0.0)
                for i in range(k + 1):
                    acc = acc + self.coeffs[i] * other.coeffs[k - i]
                out.append(acc)
            return TaylorSeries(out)
        scalar = Interval.from_value(other)
        return TaylorSeries([c * scalar for c in self.coeffs])

    __rmul__ = __mul__

    def __repr__(self) -> str:
        return f"TaylorSeries({list(self.coeffs)!r})"


def _solution_coeffs(field: VectorField, y0: Sequence[Interval], upto: int,
                     ) -> list[list[Interval]]:
    """Taylor coefficients ``U_0..U_upto`` of the flow with expansion point ``y0``."""
    n = len(y0)
    coeffs: list[list[Interval]] = [[y0[i]] for i in range(n)]
    for k in range(upto):
        length = k + 1
        series = [TaylorSeries([coeffs[i][j] for j in range(length)]) for i in range(n)]
        f_series = field(series)
        inv = Interval.from_rational(Fraction(1, k + 1))
        for i in range(n):
            coeffs[i].append(f_series[i].coeffs[k] * inv)
    return coeffs


def _field_at(field: VectorField, box: Sequence[Interval]) -> list[Interval]:
    """Evaluate ``F`` on a constant (order-0) box -> ``F(box)`` per component."""
    series = [TaylorSeries([box[i]]) for i in range(len(box))]
    return [s.coeffs[0] for s in field(series)]


def _inflate(iv: Interval, atol: float) -> Interval:
    """Outward inflation by one radius (+ ``atol``) -- a conservative Picard widening."""
    r = iv.rad + atol
    return Interval(iv.lo - r, iv.hi + r)


def _apriori_enclosure(field: VectorField, y0: Sequence[Interval], h: float,
                       *, max_iter: int = 60, atol: float = 1e-30,
                       ) -> list[Interval]:
    """Box ``[Z]`` with ``y0 + [0,h] F([Z]) subset [Z]`` -- encloses the flow on ``[0,h]``."""
    n = len(y0)
    step = Interval(0.0, h)
    f0 = _field_at(field, y0)
    z = [y0[i] + step * f0[i] for i in range(n)]
    for _ in range(max_iter):
        z_in = [_inflate(z[i], atol) for i in range(n)]
        f_z = _field_at(field, z_in)
        cand = [y0[i] + step * f_z[i] for i in range(n)]
        if all(z_in[i].lo <= cand[i].lo and cand[i].hi <= z_in[i].hi for i in range(n)):
            return z_in
        z = cand
    raise RuntimeError("a-priori enclosure did not converge; reduce the step size")


def _step(field: VectorField, y0: Sequence[Interval], h: float, order: int,
          ) -> list[Interval]:
    """One validated Taylor step of size ``h`` (order ``order``)."""
    n = len(y0)
    coeffs = _solution_coeffs(field, y0, order)
    box = _apriori_enclosure(field, y0, h)
    rem = _solution_coeffs(field, box, order + 1)
    h_pow = [Interval.point(h).pow_int(k) for k in range(order + 2)]
    out: list[Interval] = []
    for i in range(n):
        acc = Interval.point(0.0)
        for k in range(order + 1):
            acc = acc + coeffs[i][k] * h_pow[k]
        acc = acc + rem[i][order + 1] * h_pow[order + 1]
        out.append(acc)
    return out


def integrate_ivp(field: VectorField, y0: Sequence[IntervalLike], t0: float, t1: float,
                  *, order: int = 14, n_steps: int = 8) -> list[Interval]:
    r"""Rigorously enclose the flow of ``Y' = F(Y)`` from ``t0`` to ``t1``.

    ``field`` maps a list of component :class:`TaylorSeries` to the list of series of
    ``F(Y)`` (autonomous; append a clock state ``x' = 1`` for explicit-time fields).
    Returns a list of :class:`Interval` enclosing ``Y(t1)`` component-wise.  The
    enclosure is guaranteed for **every** initial condition in ``y0``.
    """
    if t1 < t0:
        raise ValueError("integrate_ivp requires t1 >= t0")
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")
    if order < 1:
        raise ValueError("order must be >= 1")
    y = [Interval.from_value(v) for v in y0]
    if t1 == t0:
        return y
    h = (t1 - t0) / n_steps
    for _ in range(n_steps):
        y = _step(field, y, h, order)
    return y


__all__ = [
    "TaylorSeries",
    "VectorField",
    "integrate_ivp",
]
