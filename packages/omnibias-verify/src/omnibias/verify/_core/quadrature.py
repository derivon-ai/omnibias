# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Certified definite integration front-end.

The certified quadrature *rules* live in
:mod:`omnibias.core.verified.quadrature` (Simpson, Gauss-Legendre,
Euler-Maclaurin / Romberg-class, Clenshaw-Curtis), each with a rigorous
remainder derived from a guaranteed derivative enclosure -- no fudge factor.
This module is the ``omnibias-verify`` front-end that drives them from a single
:data:`~omnibias.difference.DerivBound` oracle ``deriv_bound(k, box)`` -- the same
activation-agnostic interface used by :func:`certified_stencil_truncation`. It
encloses ``f`` at the quadrature nodes as ``deriv_bound(0, .)`` and pulls the
order-appropriate derivative bound for the remainder, so
``|certified_integral(...).enclosure - int_a^b f|`` is *proven*, not assumed.

Honesty: the derivative *enclosure* is **closed-form** (from the supplied tower /
jet); the quadrature nodes and weights are certified data; the returned enclosure
is the guaranteed sandwich. For an endpoint-singular integrand where no
derivative bound applies, use :func:`omnibias.core.verified.tanh_sinh_estimate`
(a labelled *numerical* estimate, not a certified enclosure).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

from omnibias.core.verified.interval import Interval, IntervalLike, sum_intervals
from omnibias.core.verified.quadrature import (
    GAUSS_LEGENDRE_MAX_N,
    clenshaw_curtis_integral,
    euler_maclaurin_quadrature,
    gauss_legendre_integral,
    simpson_integral,
)
from omnibias.core.verified.taylor_model_mv import TaylorModelMV
from omnibias.difference import DerivBound
from omnibias.verify._core.network import Network
from omnibias.verify._core.taylor import linear_image, taylor_propagate

QuadMethod = Literal["gauss", "simpson", "euler_maclaurin", "romberg", "clenshaw_curtis"]

#: A rigorous input box: one outward-rounded :class:`Interval` per axis.
Box = tuple[Interval, ...]
#: An integrand-enclosure oracle: a sound TM (or interval) of ``f`` over a cell.
IntegrandModel = Callable[[Box], "TaylorModelMV | Interval"]


@dataclass(frozen=True)
class IntegralCertificate:
    r"""A guaranteed enclosure of ``\int_a^b f`` with the rule that produced it.

    ``enclosure`` provably contains the true integral; ``estimate`` / ``width`` are
    its midpoint and outward-rounded width.
    """

    a: float
    b: float
    method: str
    order: int
    enclosure: Interval
    label: str = "closed-form + numerical"

    @property
    def estimate(self) -> float:
        return self.enclosure.mid

    @property
    def width(self) -> float:
        return self.enclosure.width


def certified_integral(
    deriv_bound: DerivBound,
    a: float,
    b: float,
    *,
    method: QuadMethod = "gauss",
    n: int = 5,
    panels: int = 8,
    terms: int = 4,
) -> IntegralCertificate:
    r"""Certify ``\int_a^b f`` from a single derivative-enclosure oracle.

    ``deriv_bound(k, box)`` must rigorously enclose ``f^{(k)}`` over ``box``
    (``deriv_bound(0, .)`` is ``f`` itself) -- the only activation-agnostic input,
    so this works for any smooth ``f`` with a sound derivative tower / interval jet.

    ``method`` selects the certified rule and hence which derivative bound feeds the
    remainder: ``"gauss"`` (``f^{(2n)}``, default, tightest for smooth ``f``),
    ``"simpson"`` (``f^{(4)}``), ``"clenshaw_curtis"`` (``f^{(n+1)}``), and
    ``"euler_maclaurin"`` / ``"romberg"`` (``f^{(2 terms)}`` endpoint corrections).
    Requires ``a <= b``.
    """
    if a > b:
        raise ValueError(f"certified_integral requires a <= b, got a={a}, b={b}")
    box = Interval(float(a), float(b))

    def f_iv(x: Interval) -> Interval:
        return deriv_bound(0, x)

    if method == "gauss":
        if not 1 <= n <= GAUSS_LEGENDRE_MAX_N:
            raise ValueError(f"gauss method needs 1 <= n <= {GAUSS_LEGENDRE_MAX_N}, got {n}")
        enc = gauss_legendre_integral(f_iv, a, b, n=n, deriv_2n_bound=deriv_bound(2 * n, box))
        order = 2 * n
    elif method == "simpson":
        enc = simpson_integral(f_iv, a, b, panels=panels, fourth_deriv_bound=deriv_bound(4, box))
        order = 4
    elif method in ("euler_maclaurin", "romberg"):
        enc = euler_maclaurin_quadrature(f_iv, deriv_bound, a, b, panels=panels, terms=terms)
        order = 2 * terms
    elif method == "clenshaw_curtis":
        enc = clenshaw_curtis_integral(f_iv, a, b, n=n, deriv_np1_bound=deriv_bound(n + 1, box))
        order = n + 1
    else:  # pragma: no cover - guarded by the Literal
        raise ValueError(f"unknown quadrature method {method!r}")

    return IntegralCertificate(a=float(a), b=float(b), method=method, order=order, enclosure=enc)


# --------------------------------------------------------------------------- #
# Multivariate certified domain integral: int_Omega f dx over a box.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class DomainIntegralCertificate:
    r"""A guaranteed enclosure of ``\int_{box} f\,dx`` over a box in ``R^dim``.

    ``enclosure`` provably contains the true integral for *every* ``f`` inside the
    per-cell integrand enclosures, regardless of the subdivision budget. ``scope``
    records that this is a local box claim (never a global / global-regularity-grade one).
    """

    enclosure: Interval
    dim: int
    order: int
    cells: int
    refined: bool
    scope: str = "box"
    label: str = "closed-form TM + certified subdivision"

    @property
    def estimate(self) -> float:
        return self.enclosure.mid

    @property
    def width(self) -> float:
        return self.enclosure.width


def _widest_axis(box: Box) -> int:
    return max(range(len(box)), key=lambda i: box[i].width)


def _split_axis(box: Box, axis: int) -> tuple[Box, Box]:
    mid = box[axis].mid
    left = tuple(Interval(iv.lo, mid) if i == axis else iv for i, iv in enumerate(box))
    right = tuple(Interval(mid, iv.hi) if i == axis else iv for i, iv in enumerate(box))
    return left, right


def _uniform_cells(box: Box, subdivisions: int | Sequence[int]) -> list[Box]:
    """Partition ``box`` into a tensor grid of sub-boxes (exact tiling).

    Endpoints are pinned to the box edges and interior edges are shared exactly
    between neighbours, so the cells cover ``box`` with only measure-zero overlap
    -- hence ``int_box = sum_cells int_cell`` holds rigorously.
    """
    dim = len(box)
    subs = (subdivisions,) * dim if isinstance(subdivisions, int) else tuple(subdivisions)
    if len(subs) != dim or any(s < 1 for s in subs):
        raise ValueError("subdivisions must be >= 1 per axis and match the box dimension")
    axis_cells: list[list[Interval]] = []
    for i, iv in enumerate(box):
        n = subs[i]
        lo, hi = iv.lo, iv.hi
        edges = [lo, *[lo + (hi - lo) * k / n for k in range(1, n)], hi]
        axis_cells.append([Interval(edges[k], edges[k + 1]) for k in range(n)])
    cells: list[Box] = [()]
    for col in axis_cells:
        cells = [(*prefix, c) for prefix in cells for c in col]
    return cells


def _cell_integral(model: TaylorModelMV | Interval, cell: Box) -> Interval:
    """Certified ``\\int_cell f`` from a per-cell TM (or crude interval) enclosure."""
    if isinstance(model, TaylorModelMV):
        return model.definite_integral()
    # Interval enclosure of f over the cell: int_cell f in enclosure * volume.
    vol = Interval.point(1.0)
    for iv in cell:
        vol = vol * (Interval.point(iv.hi) - Interval.point(iv.lo))
    return model * vol


def certified_domain_integral(
    integrand_model: IntegrandModel,
    box: Sequence[IntervalLike],
    *,
    subdivisions: int | Sequence[int] = 1,
    adaptive: bool = False,
    max_cells: int = 256,
    tol: float = 1e-6,
    scope: str = "box",
) -> DomainIntegralCertificate:
    r"""Certified multivariate integral ``\int_{box} f\,dx`` over a box.

    ``integrand_model(cell)`` must return a sound enclosure of ``f`` over the
    given sub-box -- a :class:`~omnibias.core.verified.taylor_model_mv.TaylorModelMV`
    (integrated exactly by :meth:`TaylorModelMV.definite_integral`) or a bare
    :class:`Interval` (integrated as ``enclosure * volume``). The box is first cut
    into a ``subdivisions`` tensor grid; with ``adaptive=True`` the cell whose
    enclosure is widest is bisected along its widest axis until the total width
    drops below ``tol`` or ``max_cells`` cells have been used. The returned
    enclosure is the outward-rounded sum of the per-cell enclosures -- sound at
    every budget (branch-and-bound only tightens).
    """
    box0: Box = tuple(Interval.from_value(v) for v in box)
    dim = len(box0)
    if dim < 1:
        raise ValueError("certified_domain_integral needs a non-empty box")
    cells = _uniform_cells(box0, subdivisions)
    order = 0
    leaves: list[tuple[Box, Interval]] = []
    for cell in cells:
        m = integrand_model(cell)
        if isinstance(m, TaylorModelMV):
            order = m.order
        leaves.append((cell, _cell_integral(m, cell)))
    refined = False
    if adaptive:
        while len(leaves) < max_cells:
            total = sum_intervals([e for _, e in leaves])
            if total.width <= tol:
                break
            i = max(range(len(leaves)), key=lambda k: leaves[k][1].width)
            box_i, _ = leaves[i]
            if all(iv.width == 0.0 for iv in box_i):
                break
            left, right = _split_axis(box_i, _widest_axis(box_i))
            ml = integrand_model(left)
            mr = integrand_model(right)
            leaves[i] = (left, _cell_integral(ml, left))
            leaves.append((right, _cell_integral(mr, right)))
            refined = True
    enclosure = sum_intervals([e for _, e in leaves])
    return DomainIntegralCertificate(
        enclosure=enclosure,
        dim=dim,
        order=order,
        cells=len(leaves),
        refined=refined,
        scope=scope,
    )


def network_integrand_model(
    net: Network,
    *,
    order: int = 2,
    out_index: int = 0,
    power: int = 1,
    weights: Sequence[float] | None = None,
    bias: float = 0.0,
) -> IntegrandModel:
    r"""Build a per-cell TM oracle for a scalar read-out of ``net`` raised to ``power``.

    With ``weights`` the integrand is the linear read-out ``w . net(x) + bias``;
    otherwise it is output coordinate ``out_index``. ``power`` raises the scalar
    integrand to an integer power (``power=2`` gives the ``L^2``-norm integrand
    ``net(x)^2``), so this feeds both plain domain integrals and the certified
    norm builders.
    """
    if power < 1:
        raise ValueError("power must be >= 1")

    def model(cell: Box) -> TaylorModelMV:
        models = taylor_propagate(net, cell, order=order)
        tm: TaylorModelMV
        if weights is not None:
            tm = linear_image(list(weights), models) + Interval.point(bias)
        else:
            if not 0 <= out_index < len(models):
                raise ValueError(f"out_index must be in 0..{len(models) - 1}")
            tm = models[out_index]
        return tm.pow_int(power) if power != 1 else tm

    return model


def certified_network_integral(
    net: Network,
    box: Sequence[IntervalLike],
    *,
    order: int = 2,
    out_index: int = 0,
    power: int = 1,
    weights: Sequence[float] | None = None,
    bias: float = 0.0,
    subdivisions: int | Sequence[int] = 1,
    adaptive: bool = False,
    max_cells: int = 256,
    tol: float = 1e-6,
) -> DomainIntegralCertificate:
    r"""Certified ``\int_{box} g(x)\,dx`` where ``g`` is a scalar read-out of ``net``.

    Convenience front-end that wires :func:`network_integrand_model` into
    :func:`certified_domain_integral`. ``power=2`` integrates ``net(x)^2`` (the
    ``L^2`` mass), the substrate of :func:`certified_lp_norm`.
    """
    model = network_integrand_model(
        net, order=order, out_index=out_index, power=power, weights=weights, bias=bias
    )
    return certified_domain_integral(
        model,
        box,
        subdivisions=subdivisions,
        adaptive=adaptive,
        max_cells=max_cells,
        tol=tol,
    )


__all__ = [
    "Box",
    "DomainIntegralCertificate",
    "IntegralCertificate",
    "IntegrandModel",
    "QuadMethod",
    "certified_domain_integral",
    "certified_integral",
    "certified_network_integral",
    "network_integrand_model",
]
