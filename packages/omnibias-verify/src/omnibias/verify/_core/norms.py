# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Certified function-space norms of an ingested :class:`Network` over a box.

The genuinely new capability on top of :mod:`omnibias.verify._core.quadrature`:
rigorous, outward-rounded enclosures of a network's ``L^p`` and Sobolev ``H^k``
norms over a box ``Omega`` -- the certified twin of the differentiable
``omnibias.fields.l2_norm`` / ``sobolev_norm``.

Two integrand routes, honestly labelled:

* ``method="taylor"`` -- the ``L^p`` mass ``int |u|^p`` is integrated as
  ``int u^p`` through the closed-form Taylor-model tower (tight, requires an even
  ``p`` so ``u^p = |u|^p``). Best for smooth nets (tanh / sigmoid / GELU).
* ``method="interval"`` -- the **layer-cake / superlevel** variant: the
  non-negative integrand ``|u(x)|^p`` is enclosed *per cell* from the network's
  output range ``[u_lo, u_hi]`` (a sound superlevel-set bound), then integrated as
  ``enclosure * volume``. This never gives a vacuous bound on a non-smooth
  integrand and handles ReLU networks and odd ``p`` where the derivative-remainder
  view degrades.

The Sobolev ``H^1`` seminorm reuses the rigorous Jacobian enclosure
(:func:`omnibias.verify._core.certificates.interval_jacobian`) as the per-cell
gradient integrand -- sound for smooth *and* ReLU nets (the kink contributes the
``[0, 1]`` subgradient). Certificates are ``IntegralCertificate``-style frozen
dataclasses carrying a ``scope`` field (always a local box claim, never global).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import cast

from omnibias.core.verified.interval import Interval, IntervalLike
from omnibias.verify._core.certificates import interval_jacobian
from omnibias.verify._core.network import Network
from omnibias.verify._core.quadrature import (
    Box,
    DomainIntegralCertificate,
    certified_domain_integral,
    network_integrand_model,
)
from omnibias.verify._core.taylor import taylor_output_bounds

NormMethod = str  # "taylor" | "interval"


# --------------------------------------------------------------------------- #
# Certificates.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LpNormCertificate:
    r"""Guaranteed enclosure of ``||u||_{L^p(box)} = (int_box |u|^p dx)^{1/p}``.

    ``mass`` encloses the ``L^p`` mass ``int_box |u|^p``; ``enclosure`` is its
    certified ``p``-th root. ``scope`` records the local box claim.
    """

    enclosure: Interval
    mass: Interval
    p: int
    dim: int
    cells: int
    method: str
    scope: str = "box"
    label: str = "certified L^p norm"

    @property
    def value(self) -> float:
        return self.enclosure.mid

    @property
    def width(self) -> float:
        return self.enclosure.width


@dataclass(frozen=True)
class SobolevNormCertificate:
    r"""Guaranteed enclosure of the Sobolev norm ``||u||_{H^k(box)}``.

    ``squared_mass`` encloses ``sum_{|a|<=k} int_box (D^a u)^2`` and ``enclosure``
    its certified square root. ``value_mass`` / ``seminorm_masses`` expose the
    ``L^2`` term and the per-order squared-seminorm masses.
    """

    enclosure: Interval
    squared_mass: Interval
    value_mass: Interval
    seminorm_masses: tuple[Interval, ...]
    order: int
    dim: int
    cells: int
    method: str
    scope: str = "box"
    label: str = "certified Sobolev (H^k) norm"

    @property
    def value(self) -> float:
        return self.enclosure.mid

    @property
    def width(self) -> float:
        return self.enclosure.width


# --------------------------------------------------------------------------- #
# Helpers.
# --------------------------------------------------------------------------- #
def _clamp_nonneg(mass: Interval) -> Interval:
    """Clamp a mass enclosure to ``>= 0`` (the true ``int |.|^p`` is non-negative).

    Sound: the true mass ``M`` lies in ``[lo, hi]`` and ``M >= 0``, so
    ``[max(0, lo), max(hi, 0)]`` still contains ``M``.
    """
    lo = mass.lo if mass.lo > 0.0 else 0.0
    hi = mass.hi if mass.hi > lo else lo
    return Interval(lo, hi)


def _interval_pth_root(mass: Interval, p: int) -> Interval:
    r"""Certified ``p``-th root of a non-negative mass enclosure.

    Powers of two use the correctly-rounded :meth:`Interval.sqrt` (no transcendental
    backend needed -- the ``p in {2, 4, 8, ...}`` NN norms); a residual odd factor
    uses the rigorous ``exp``/``ln`` enclosures.
    """
    if p < 1:
        raise ValueError("p must be >= 1")
    m = _clamp_nonneg(mass)
    if p == 1:
        return m
    q = p
    r = m
    while q % 2 == 0 and q > 1:
        r = r.sqrt()
        q //= 2
    if q == 1:
        return r
    from omnibias.core.verified.transcend import exp_iv, ln_iv  # rigorous backend

    inv = Interval.from_rational(Fraction(1, q))
    lo_r = 0.0 if r.lo <= 0.0 else exp_iv(ln_iv(Interval.point(r.lo)) * inv).lo
    hi_r = 0.0 if r.hi <= 0.0 else exp_iv(ln_iv(Interval.point(r.hi)) * inv).hi
    return Interval(lo_r, max(hi_r, lo_r))


def _cell_output_enclosure(net: Network, cell: Box, out_index: int, order: int) -> Interval:
    """Tightest sound per-cell output enclosure (Taylor model intersected with IBP)."""
    bounds = taylor_output_bounds(net, cell, order=order)
    if not 0 <= out_index < len(bounds):
        raise ValueError(f"out_index must be in 0..{len(bounds) - 1}")
    return cast("Interval", bounds[out_index])


# --------------------------------------------------------------------------- #
# Certified L^p norm.
# --------------------------------------------------------------------------- #
def certified_lp_norm(
    net: Network,
    box: Sequence[IntervalLike],
    *,
    p: int = 2,
    out_index: int = 0,
    method: NormMethod = "taylor",
    order: int = 2,
    subdivisions: int | Sequence[int] = 1,
    adaptive: bool = False,
    max_cells: int = 256,
    tol: float = 1e-6,
) -> LpNormCertificate:
    r"""Certified ``||u||_{L^p(box)}`` for output ``out_index`` of ``net``.

    ``method="taylor"`` (default) integrates ``u^p`` through the Taylor-model tower
    and requires an even ``p`` (so ``u^p = |u|^p``); it is the tight route for
    smooth activations. ``method="interval"`` is the layer-cake / superlevel
    variant that encloses ``|u|^p`` from the per-cell output range -- robust on
    non-smooth (ReLU) integrands and valid for any ``p >= 1``. Both subdivide the
    box (``subdivisions`` grid, optional ``adaptive`` branch-and-bound) and take a
    certified ``p``-th root of the mass enclosure.
    """
    if p < 1:
        raise ValueError("p must be >= 1")
    if method == "taylor":
        if p % 2 != 0:
            raise ValueError(
                "method='taylor' needs an even p (u^p = |u|^p); use method='interval' "
                "for odd p or non-smooth nets"
            )
        cert = certified_domain_integral(
            network_integrand_model(net, order=order, out_index=out_index, power=p),
            box,
            subdivisions=subdivisions,
            adaptive=adaptive,
            max_cells=max_cells,
            tol=tol,
        )
    elif method == "interval":

        def model(cell: Box) -> Interval:
            out = _cell_output_enclosure(net, cell, out_index, order)
            return out.abs().pow_int(p)

        cert = certified_domain_integral(
            model,
            box,
            subdivisions=subdivisions,
            adaptive=adaptive,
            max_cells=max_cells,
            tol=tol,
        )
    else:  # pragma: no cover - guarded
        raise ValueError(f"unknown method {method!r}; choose 'taylor' or 'interval'")

    mass = _clamp_nonneg(cert.enclosure)
    return LpNormCertificate(
        enclosure=_interval_pth_root(mass, p),
        mass=mass,
        p=p,
        dim=cert.dim,
        cells=cert.cells,
        method=method,
    )


# --------------------------------------------------------------------------- #
# Certified Sobolev (H^k) norm.
# --------------------------------------------------------------------------- #
def certified_sobolev_norm(
    net: Network,
    box: Sequence[IntervalLike],
    *,
    order: int = 1,
    out_index: int = 0,
    tm_order: int = 2,
    subdivisions: int | Sequence[int] = 1,
    adaptive: bool = False,
    max_cells: int = 256,
    tol: float = 1e-6,
) -> SobolevNormCertificate:
    r"""Certified Sobolev norm ``||u||_{H^k(box)}`` for output ``out_index``.

    ``order=0`` is the ``L^2`` norm; ``order=1`` adds the gradient seminorm
    ``||u||_{H^1}^2 = int u^2 + sum_i int (du/dx_i)^2``. The ``L^2`` mass is the
    tight Taylor-model ``int u^2``; each gradient-squared mass uses the rigorous
    per-cell Jacobian enclosure
    (:func:`~omnibias.verify._core.certificates.interval_jacobian`), so the
    certificate is sound for smooth *and* ReLU networks (the kink contributes the
    ``[0, 1]`` subgradient). Orders ``>= 2`` are not yet supported (no certified
    Hessian-integrand path).
    """
    if order not in (0, 1):
        raise NotImplementedError(
            "certified_sobolev_norm currently supports order 0 (L^2) or 1 (H^1)"
        )

    l2 = certified_domain_integral(
        network_integrand_model(net, order=tm_order, out_index=out_index, power=2),
        box,
        subdivisions=subdivisions,
        adaptive=adaptive,
        max_cells=max_cells,
        tol=tol,
    )
    value_mass = _clamp_nonneg(l2.enclosure)
    squared = value_mass
    seminorm_masses: list[Interval] = []
    dim = l2.dim
    cells = l2.cells

    if order == 1:
        for axis in range(dim):

            def grad_sq(cell: Box, _axis: int = axis) -> Interval:
                jac = interval_jacobian(net, cell)
                return cast("Interval", jac[out_index][_axis].pow_int(2))

            grad_cert = certified_domain_integral(
                grad_sq,
                box,
                subdivisions=subdivisions,
                adaptive=adaptive,
                max_cells=max_cells,
                tol=tol,
            )
            m = _clamp_nonneg(grad_cert.enclosure)
            seminorm_masses.append(m)
            squared = squared + m
            cells += grad_cert.cells

    squared = _clamp_nonneg(squared)
    return SobolevNormCertificate(
        enclosure=squared.sqrt(),
        squared_mass=squared,
        value_mass=value_mass,
        seminorm_masses=tuple(seminorm_masses),
        order=order,
        dim=dim,
        cells=cells,
        method="taylor+jacobian",
    )


# --------------------------------------------------------------------------- #
# Certified layer-cake integral of a non-negative transform of the output.
# --------------------------------------------------------------------------- #
def certified_layer_cake_integral(
    net: Network,
    box: Sequence[IntervalLike],
    *,
    out_index: int = 0,
    transform: str = "abs",
    power: int = 1,
    order: int = 2,
    subdivisions: int | Sequence[int] = 1,
    adaptive: bool = False,
    max_cells: int = 256,
    tol: float = 1e-6,
) -> DomainIntegralCertificate:
    r"""Certified ``int_box phi(u(x)) dx`` for a non-negative ``phi`` of the output.

    The certified counterpart of ``omnibias.measure.layer_cake_integral``: the
    non-negative integrand ``phi(u)`` is enclosed *per cell* from the output range
    ``[u_lo, u_hi]`` (a sound superlevel-set bound) and integrated as
    ``enclosure * volume``. ``transform`` selects ``phi``: ``"abs"`` -> ``|u|^power``
    or ``"relu"`` -> ``max(0, u)^power``. Because the enclosure never differentiates
    a flat remainder, it stays sound (never vacuous) on non-smooth / ReLU nets where
    the derivative-remainder quadrature degrades.
    """
    if power < 1:
        raise ValueError("power must be >= 1")
    if transform not in ("abs", "relu"):
        raise ValueError(f"unknown transform {transform!r}; choose 'abs' or 'relu'")

    def model(cell: Box) -> Interval:
        out = _cell_output_enclosure(net, cell, out_index, order)
        if transform == "abs":
            g = out.abs()
        else:  # relu: max(0, out)
            g = Interval(max(0.0, out.lo), max(0.0, out.hi))
        return g.pow_int(power)

    return certified_domain_integral(
        model,
        box,
        subdivisions=subdivisions,
        adaptive=adaptive,
        max_cells=max_cells,
        tol=tol,
        scope="box",
    )


__all__ = [
    "LpNormCertificate",
    "NormMethod",
    "SobolevNormCertificate",
    "certified_layer_cake_integral",
    "certified_lp_norm",
    "certified_sobolev_norm",
]
