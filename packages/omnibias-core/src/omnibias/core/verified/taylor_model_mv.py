# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Multivariate Taylor models: rigorous enclosures over a box in ``R^n``.

This is the ``dim``-variable generalisation of
:mod:`omnibias.core.verified.taylor_model`.  A total-degree-``order`` Taylor
model on the box ``[c_i - r_i, c_i + r_i]`` is a pair

.. math::

    f(x) \in \sum_{|\alpha| \le N} a_\alpha\,(x - c)^\alpha + R
    \qquad \forall x \in \prod_i [c_i - r_i, c_i + r_i],

where the coefficients ``a_\alpha`` are :class:`Interval`\s indexed by the
canonical multi-index ordering of :mod:`omnibias.core.multi_index`, and the
:class:`Interval` remainder ``R`` rigorously absorbs every term the polynomial
omits (both the truncation tail of a product and any pre-existing remainders).

Keeping the polynomial *shape* (rather than collapsing to a bare box) is what
defeats the wrapping/dependency blow-up of naive interval evaluation, so the
:meth:`bound` over the box stays tight even after several multiplications.  This
is the substrate the 2-D model-equation certificates (e.g. the Riesz/Leray
operators in :mod:`omnibias.core.verified.riesz`) build on.
"""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction

from omnibias.core.multi_index import (
    MultiIndex,
    index_position,
    multi_indices,
    num_multi_indices,
)
from omnibias.core.verified.interval import Interval, IntervalLike, _pred, _succ


def _rel_box(radius: Sequence[float]) -> list[Interval]:
    """Per-axis outward enclosure of the relative cell ``[-r_i, r_i]``."""
    out: list[Interval] = []
    for r in radius:
        rf = float(r)
        if rf < 0.0:
            raise ValueError("box radii must be non-negative")
        if rf == 0.0:
            out.append(Interval.point(0.0))
        else:
            out.append(Interval(_pred(-rf), _succ(rf)))
    return out


def _box_monomial(alpha: MultiIndex, radius: Sequence[float]) -> Interval:
    r"""Enclosure of ``X^alpha = prod_i X_i^{alpha_i}`` over the relative box."""
    acc = Interval.point(1.0)
    axes = _rel_box(radius)
    for axis, power in enumerate(alpha):
        if power:
            acc = acc * axes[axis].pow_int(power)
    return acc


def _box_integral_monomial(alpha: MultiIndex, radius: Sequence[float]) -> Interval:
    r"""Exact box integral ``\int_{box} X^alpha dX`` over the symmetric relative box.

    ``\int_{-r_i}^{r_i} X_i^{p} dX_i = 0`` for odd ``p`` and ``2 r_i^{p+1}/(p+1)``
    for even ``p``; the monomial integral is the product over axes (so it vanishes
    as soon as any exponent is odd). Outward-rounded.
    """
    acc = Interval.point(1.0)
    axes = _rel_box(radius)
    for axis, power in enumerate(alpha):
        if power % 2 == 1:
            return Interval.point(0.0)
        h = Interval.point(axes[axis].hi)  # outward half-width
        acc = acc * h.pow_int(power + 1) * Interval.from_rational(Fraction(2, power + 1))
    return acc


class TaylorModelMV:
    """Total-degree-``order`` multivariate Taylor model on a box."""

    __slots__ = ("center", "radius", "dim", "order", "coeffs", "remainder")

    def __init__(
        self,
        center: Sequence[float],
        radius: Sequence[float],
        order: int,
        coeffs: Sequence[Interval],
        remainder: Interval,
    ) -> None:
        dim = len(center)
        if dim < 1:
            raise ValueError("a multivariate Taylor model needs dim >= 1")
        if len(radius) != dim:
            raise ValueError("center and radius must have the same length")
        if order < 0:
            raise ValueError("order must be >= 0")
        if any(r < 0.0 for r in radius):
            raise ValueError("box radii must be non-negative")
        expected = num_multi_indices(dim, order)
        if len(coeffs) != expected:
            raise ValueError(
                f"expected {expected} coefficients for dim={dim}, order={order}, "
                f"got {len(coeffs)}"
            )
        self.center: tuple[float, ...] = tuple(float(c) for c in center)
        self.radius: tuple[float, ...] = tuple(float(r) for r in radius)
        self.dim = dim
        self.order = order
        self.coeffs: list[Interval] = list(coeffs)
        self.remainder = remainder

    # ----- constructors -------------------------------------------------- #
    @classmethod
    def constant(
        cls,
        value: IntervalLike,
        center: Sequence[float],
        radius: Sequence[float],
        order: int,
    ) -> TaylorModelMV:
        """The model of the constant function ``f(x) = value``."""
        n = num_multi_indices(len(center), order)
        coeffs = [Interval.point(0.0) for _ in range(n)]
        coeffs[0] = Interval.from_value(value)
        return cls(center, radius, order, coeffs, Interval.point(0.0))

    @classmethod
    def coordinate(
        cls,
        axis: int,
        center: Sequence[float],
        radius: Sequence[float],
        order: int,
    ) -> TaylorModelMV:
        """The model of the coordinate map ``f(x) = x_axis = c_axis + X_axis``."""
        dim = len(center)
        if not 0 <= axis < dim:
            raise ValueError(f"axis {axis} out of range for dim {dim}")
        if order < 1:
            raise ValueError("coordinate models need order >= 1")
        n = num_multi_indices(dim, order)
        coeffs = [Interval.point(0.0) for _ in range(n)]
        coeffs[0] = Interval.point(float(center[axis]))
        unit = tuple(1 if i == axis else 0 for i in range(dim))
        coeffs[index_position(dim, order)[unit]] = Interval.point(1.0)
        return cls(center, radius, order, coeffs, Interval.point(0.0))

    # ----- helpers ------------------------------------------------------- #
    def _check(self, other: TaylorModelMV) -> None:
        if (
            self.center != other.center
            or self.radius != other.radius
            or self.order != other.order
        ):
            raise ValueError("Taylor models must share the same box and order")

    def _poly_bound(self) -> Interval:
        """Enclosure of the polynomial part only (no remainder), over the box."""
        indices = multi_indices(self.dim, self.order)
        acc = Interval.point(0.0)
        for idx, coeff in zip(indices, self.coeffs, strict=True):
            if coeff.lo == 0.0 and coeff.hi == 0.0:
                continue
            acc = acc + coeff * _box_monomial(idx, self.radius)
        return acc

    def bound(self) -> Interval:
        """Guaranteed enclosure of ``f`` over the whole box."""
        return self._poly_bound() + self.remainder

    def eval(self, delta: Sequence[IntervalLike]) -> Interval:
        r"""Enclosure of ``f(center + delta)`` for ``delta`` in the relative box.

        ``delta`` is given relative to ``center`` (a point or sub-interval per
        axis); each must lie within ``[-r_i, r_i]`` for the enclosure to be
        valid.  Passing the full relative box reproduces :meth:`bound`.
        """
        if len(delta) != self.dim:
            raise ValueError("delta must have one entry per dimension")
        dvals = [Interval.from_value(d) for d in delta]
        indices = multi_indices(self.dim, self.order)
        acc = Interval.point(0.0)
        for idx, coeff in zip(indices, self.coeffs, strict=True):
            if coeff.lo == 0.0 and coeff.hi == 0.0:
                continue
            mono = Interval.point(1.0)
            for axis, power in enumerate(idx):
                if power:
                    mono = mono * dvals[axis].pow_int(power)
            acc = acc + coeff * mono
        return acc + self.remainder

    # ----- arithmetic ---------------------------------------------------- #
    def __add__(self, other: TaylorModelMV | IntervalLike) -> TaylorModelMV:
        if isinstance(other, TaylorModelMV):
            self._check(other)
            coeffs = [a + b for a, b in zip(self.coeffs, other.coeffs, strict=True)]
            return TaylorModelMV(
                self.center, self.radius, self.order, coeffs,
                self.remainder + other.remainder,
            )
        coeffs = list(self.coeffs)
        coeffs[0] = coeffs[0] + Interval.from_value(other)
        return TaylorModelMV(self.center, self.radius, self.order, coeffs, self.remainder)

    __radd__ = __add__

    def __neg__(self) -> TaylorModelMV:
        return TaylorModelMV(
            self.center, self.radius, self.order,
            [-c for c in self.coeffs], -self.remainder,
        )

    def __sub__(self, other: TaylorModelMV | IntervalLike) -> TaylorModelMV:
        if isinstance(other, TaylorModelMV):
            return self.__add__(-other)
        return self.__add__(-Interval.from_value(other))

    def __rsub__(self, other: IntervalLike) -> TaylorModelMV:
        return (-self).__add__(Interval.from_value(other))

    def __mul__(self, other: TaylorModelMV | IntervalLike) -> TaylorModelMV:
        if not isinstance(other, TaylorModelMV):
            scal = Interval.from_value(other)
            return TaylorModelMV(
                self.center, self.radius, self.order,
                [c * scal for c in self.coeffs], self.remainder * scal,
            )
        self._check(other)
        indices = multi_indices(self.dim, self.order)
        pos = index_position(self.dim, self.order)
        n = len(self.coeffs)
        low = [Interval.point(0.0) for _ in range(n)]
        high = Interval.point(0.0)
        for ia in range(n):
            a = self.coeffs[ia]
            if a.lo == 0.0 and a.hi == 0.0:
                continue
            alpha = indices[ia]
            for ib in range(n):
                b = other.coeffs[ib]
                if b.lo == 0.0 and b.hi == 0.0:
                    continue
                gamma = tuple(alpha[k] + indices[ib][k] for k in range(self.dim))
                prod = a * b
                if sum(gamma) <= self.order:
                    low[pos[gamma]] = low[pos[gamma]] + prod
                else:
                    high = high + prod * _box_monomial(gamma, self.radius)
        # f*g = P*Q + P*Rg + Q*Rf + Rf*Rg, with the truncated P*Q tail in `high`.
        # Provably-zero remainder terms are skipped: this keeps an exact-polynomial
        # product exactly remainder-free (no spurious subnormal inflation) and is
        # cheaper, while staying rigorous (a [0,0] factor contributes nothing).
        remainder = high
        self_rzero = self.remainder.lo == 0.0 and self.remainder.hi == 0.0
        other_rzero = other.remainder.lo == 0.0 and other.remainder.hi == 0.0
        if not other_rzero:
            remainder = remainder + self._poly_bound() * other.remainder
        if not self_rzero:
            remainder = remainder + other._poly_bound() * self.remainder
        if not self_rzero and not other_rzero:
            remainder = remainder + self.remainder * other.remainder
        return TaylorModelMV(self.center, self.radius, self.order, low, remainder)

    __rmul__ = __mul__

    def antiderivative(self, axis: int) -> TaylorModelMV:
        r"""Rigorous antiderivative along ``axis``: ``F`` with ``d F / d x_axis = f``.

        Integration is the natural *sound* operation on a Taylor model.  The
        polynomial part integrates term-by-term in the relative variable along
        ``axis`` (raising the total degree by one, so the result has order
        ``order + 1``), with the integration constant fixed by ``F = 0`` on the
        slice ``x_axis = center_axis``.  The remainder integrates rigorously to
        ``R \cdot [-r_axis, r_axis]`` (the mean-value enclosure of
        ``\int_0^{X_axis} e\,dt``).
        """
        if not 0 <= axis < self.dim:
            raise ValueError(f"axis {axis} out of range for dim {self.dim}")
        new_order = self.order + 1
        n_new = num_multi_indices(self.dim, new_order)
        pos_new = index_position(self.dim, new_order)
        old_indices = multi_indices(self.dim, self.order)
        new_coeffs = [Interval.point(0.0) for _ in range(n_new)]
        for c, alpha in zip(self.coeffs, old_indices, strict=True):
            if c.lo == 0.0 and c.hi == 0.0:
                continue
            beta = tuple(a + (1 if i == axis else 0) for i, a in enumerate(alpha))
            scale = Interval.from_rational(Fraction(1, alpha[axis] + 1))
            j = pos_new[beta]
            new_coeffs[j] = new_coeffs[j] + c * scale
        r_axis = self.radius[axis]
        rem = self.remainder * _rel_box([r_axis])[0]
        return TaylorModelMV(self.center, self.radius, new_order, new_coeffs, rem)

    def definite_integral(self) -> Interval:
        r"""Certified ``\int_{box} f(x)\,dx`` over the whole (symmetric) cell.

        The multivariate generalisation of
        :meth:`omnibias.core.verified.taylor_model.TaylorModel.definite_integral`.
        On the symmetric box ``\prod_i [c_i - r_i, c_i + r_i]`` a monomial
        ``X^\alpha`` integrates to ``\prod_i \int_{-r_i}^{r_i} X_i^{\alpha_i} dX_i``,
        which is ``0`` whenever any ``\alpha_i`` is odd and
        ``\prod_i 2\,r_i^{\alpha_i+1}/(\alpha_i+1)`` otherwise. The remainder
        contributes ``R \cdot \mathrm{vol}`` with ``\mathrm{vol} = \prod_i 2 r_i``,
        since ``\int_{box} e(x)\,dx = \mathrm{vol}\cdot\bar e`` for a mean value
        ``\bar e \in R``. Everything is outward-rounded, so the returned
        :class:`Interval` provably contains the true integral over the box.
        """
        indices = multi_indices(self.dim, self.order)
        acc = Interval.point(0.0)
        for coeff, alpha in zip(self.coeffs, indices, strict=True):
            if coeff.lo == 0.0 and coeff.hi == 0.0:
                continue
            acc = acc + coeff * _box_integral_monomial(alpha, self.radius)
        vol = _box_integral_monomial(tuple(0 for _ in range(self.dim)), self.radius)
        return acc + self.remainder * vol

    def pow_int(self, n: int) -> TaylorModelMV:
        if n < 0:
            raise ValueError("Taylor-model powers require n >= 0")
        result = TaylorModelMV.constant(1.0, self.center, self.radius, self.order)
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
        return (
            f"TaylorModelMV(center={self.center!r}, radius={self.radius!r}, "
            f"order={self.order}, remainder={self.remainder!r})"
        )


__all__ = ["TaylorModelMV"]
