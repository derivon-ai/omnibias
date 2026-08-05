# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Certified truncation error for finite-difference PDE discretisations.

Standard PDE codes discretise :math:`\partial_x`, :math:`\partial_x^2`, and the
Laplacian with finite-difference stencils whose *consistency order* (the
``O(h^p)`` truncation rate) is quoted from a Taylor expansion but rarely
**certified**. This module closes that gap by reusing the ``omnibias-difference``
remainder engine -- :func:`omnibias.difference.certified_fd_error_general`, now
decoupled from the built-in activation dictionary -- so the truncation error of a
stencil applied to *any* function with a sound derivative-tower / interval-jet
enclosure (a :data:`~omnibias.difference.DerivBound` oracle) is proven, not
assumed:

.. math::

    \Big|\text{stencil}_h[f](x) - f^{(m)}(x)\Big| \le C_{m,p}\,h^{p}\,
        \max_{[\,x-Rh,\;x+Rh\,]} \big|f^{(m+p)}\big|,

with the constant and the ``max`` both enclosed rigorously. :func:`certified_stencil_truncation`
returns that bound for the 1-D first/second-derivative stencils;
:func:`certified_laplacian_truncation` sums the per-axis 1-D bounds for an
axis-aligned discrete Laplacian; :func:`measured_consistency_order` is the
empirical-order baseline the certified order must match.

Honesty labels: the derivative *enclosure* is **closed-form** (from the supplied
tower / jet), the stencil evaluation is **numerical** (plain ``float``), and the
truncation bound is the certified sandwich linking them.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import log
from typing import Literal

from omnibias.core.verified.interval import Interval
from omnibias.difference import DerivBound, certified_fd_error_general

Stencil = Literal["central", "forward"]


@dataclass(frozen=True)
class StencilTruncationCertificate:
    """Certified truncation error of a 1-D finite-difference PDE stencil.

    ``truncation_bound`` proves ``|stencil - f^(deriv_order)(x)| <= truncation_bound``
    and is ``O(step^consistency_order)``. ``derivative_enclosure`` is the closed-form
    enclosure of the true ``f^(deriv_order)(x)``.
    """

    deriv_order: int
    step: float
    stencil: str
    consistency_order: int
    truncation_bound: float
    estimate: float
    derivative_enclosure: Interval
    label: str = "closed-form + numerical"

    @property
    def true_value_interval(self) -> Interval:
        """The numerical estimate widened by the certified truncation bound."""
        return Interval(self.estimate - self.truncation_bound, self.estimate + self.truncation_bound)

    @property
    def consistent(self) -> bool:
        """Whether the certified bracket overlaps the closed-form derivative enclosure."""
        lo = max(self.derivative_enclosure.lo, self.estimate - self.truncation_bound)
        hi = min(self.derivative_enclosure.hi, self.estimate + self.truncation_bound)
        return lo <= hi


def certified_stencil_truncation(
    f_float: Callable[[float], float],
    deriv_bound: DerivBound,
    x: float,
    deriv_order: int,
    step: float,
    stencil: Stencil = "central",
    *,
    name: str = "pde-stencil",
) -> StencilTruncationCertificate:
    r"""Certify the truncation error of a 1-D finite-difference stencil for ``f``.

    ``deriv_order`` is the derivative the stencil targets (``1`` for
    :math:`\partial_x`, ``2`` for :math:`\partial_x^2`); ``stencil`` picks the
    accuracy order (``"central"`` -> ``p = 2``, ``"forward"`` -> ``p = 1``).
    ``deriv_bound(k, box)`` must rigorously enclose ``f^(k)`` over a box; it is the
    only activation-agnostic input, so this works for arbitrary smooth ``f``.
    """
    cert = certified_fd_error_general(
        f_float, deriv_bound, x, deriv_order, step, stencil, name=name
    )
    return StencilTruncationCertificate(
        deriv_order=deriv_order,
        step=float(step),
        stencil=stencil,
        consistency_order=cert.accuracy_order,
        truncation_bound=cert.error_bound,
        estimate=cert.estimate,
        derivative_enclosure=cert.enclosure,
    )


@dataclass(frozen=True)
class LaplacianTruncationCertificate:
    """Certified truncation error of an axis-aligned discrete Laplacian."""

    dimension: int
    step: float
    consistency_order: int
    truncation_bound: float
    estimate: float
    laplacian_enclosure: Interval
    label: str = "closed-form + numerical"

    @property
    def true_value_interval(self) -> Interval:
        return Interval(self.estimate - self.truncation_bound, self.estimate + self.truncation_bound)

    @property
    def consistent(self) -> bool:
        lo = max(self.laplacian_enclosure.lo, self.estimate - self.truncation_bound)
        hi = min(self.laplacian_enclosure.hi, self.estimate + self.truncation_bound)
        return lo <= hi


def certified_laplacian_truncation(
    axis_f_float: Sequence[Callable[[float], float]],
    axis_deriv_bound: Sequence[DerivBound],
    point: Sequence[float],
    step: float,
) -> LaplacianTruncationCertificate:
    r"""Certify the second-order central discrete Laplacian for a **separable** field.

    For :math:`f(x_1,\dots,x_d) = \sum_i g_i(x_i)` the standard 5-point (``2d + 1``
    node) Laplacian decomposes into ``d`` independent 1-D second-derivative
    stencils, so the truncation error is the **sum** of the per-axis certified
    bounds and the Laplacian enclosure is the sum of the per-axis
    ``g_i''(x_i)`` enclosures -- fully rigorous with the 1-D engine.

    Fully coupled (non-separable) fields need mixed fourth partials and a
    :class:`~omnibias.core.verified.taylor_model_mv.TaylorModelMV` derivative
    oracle; that multi-dim generalisation is out of scope here (see the W5
    findings note).
    """
    d = len(point)
    if not (len(axis_f_float) == len(axis_deriv_bound) == d):
        raise ValueError("axis_f_float, axis_deriv_bound and point must share length d")
    if d == 0:
        raise ValueError("Laplacian needs at least one axis")

    estimate = 0.0
    total_bound = 0.0
    lap_lo = 0.0
    lap_hi = 0.0
    for g, gbound, xi in zip(axis_f_float, axis_deriv_bound, point, strict=True):
        cert = certified_stencil_truncation(g, gbound, xi, 2, step, "central")
        estimate += cert.estimate
        total_bound += cert.truncation_bound
        lap_lo += cert.derivative_enclosure.lo
        lap_hi += cert.derivative_enclosure.hi
    return LaplacianTruncationCertificate(
        dimension=d,
        step=float(step),
        consistency_order=2,
        truncation_bound=total_bound,
        estimate=estimate,
        laplacian_enclosure=Interval(lap_lo, lap_hi),
    )


def measured_consistency_order(
    f_float: Callable[[float], float],
    deriv_bound: DerivBound,
    x: float,
    deriv_order: int,
    steps: Sequence[float],
    stencil: Stencil = "central",
) -> float:
    r"""Empirical order of accuracy: the log-log slope of ``|error|`` vs ``step``.

    The classical baseline for the certified consistency order. The reference
    derivative is the midpoint of the tight point enclosure ``deriv_bound(m, {x})``.
    Uses a least-squares fit of ``log|error|`` on ``log(step)`` over the (moderate)
    ``steps`` where truncation -- not float cancellation -- dominates. Raises if
    fewer than two usable (non-zero-error) points remain.
    """
    if len(steps) < 2:
        raise ValueError("need at least two step sizes to estimate an order")
    reference = deriv_bound(deriv_order, Interval.point(float(x))).mid
    xs: list[float] = []
    ys: list[float] = []
    for h in steps:
        cert = certified_stencil_truncation(f_float, deriv_bound, x, deriv_order, h, stencil)
        err = abs(cert.estimate - reference)
        if err > 0.0:
            xs.append(log(float(h)))
            ys.append(log(err))
    if len(xs) < 2:
        raise ValueError("all errors were zero; cannot estimate an empirical order")
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(xs, ys, strict=True))
    var = sum((xi - mean_x) ** 2 for xi in xs)
    return cov / var


__all__ = [
    "LaplacianTruncationCertificate",
    "StencilTruncationCertificate",
    "certified_laplacian_truncation",
    "certified_stencil_truncation",
    "measured_consistency_order",
]
