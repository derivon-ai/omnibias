# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Boundary and initial conditions (backend-agnostic descriptors).

A condition names a *component*, a *kind*, and a *target value*. The target is
either a Python float or a callable ``value(coords) -> array`` evaluated on the
runtime coordinate tensor (use :func:`omnibias.pinn.solver._core.arrays.array_namespace`
inside the callable for transcendental functions).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

#: A condition target: a constant or a callable of the coordinate tensor.
ValueLike = float | Callable[[Any], Any]

_BC_KINDS = frozenset({"dirichlet", "neumann", "robin", "periodic"})

#: Default derivative orders matched across a periodic seam.
#:
#: A smooth periodic solution matches *every* derivative, so requiring value,
#: slope and curvature is a subset of the truth rather than an extra assumption
#: -- and matching only the value leaves the solution free to kink exactly where
#: nothing is watching.
#:
#: ``(0, 1, 2)`` rather than ``(0, 1)``: the periodic sweep on the gauge-free
#: Poisson seam showed a C¹-only seam leaves an interior-L2 gap under a
#: second-order operator, and matching the second derivative closes it -- hard
#: then wins on every seed. The sweep also carries ``(0, 1, 2, 3)``, which is
#: better again on that problem, but by 1.30x against the third order's 3.34x.
#: A smooth manufactured solution rewards extra orders indefinitely, so
#: diminishing returns are the signal to stop rather than to continue, and a
#: higher default would over-smooth seams on problems with steep gradients --
#: which the periodic-emit measurement already showed for Burgers. See
#: ``benchmarks/hard_conditions_periodic_sweep.py``.
#:
#: What this buys is matching *at these orders and no higher*. The first
#: unmatched order stays genuinely discontinuous, at roughly the magnitude of
#: that derivative itself; both cage test suites pin that, so "exact seam" can
#: never quietly widen into "smooth seam".
PERIODIC_ORDERS = (0, 1, 2)


@dataclass(frozen=True)
class BoundaryCondition:
    """A boundary condition on one component.

    Parameters
    ----------
    component
        Component name the condition constrains.
    kind
        One of ``"dirichlet"`` (``u = value``), ``"neumann"``
        (``du/dn = value``), ``"robin"`` (``alpha*u + beta*du/dn = value``),
        or ``"periodic"`` (seam matching
        ``d^n u(hi) - d^n u(lo) = 0`` for each ``n`` in ``periodic_orders``,
        enforced structurally by the hard path or as residual rows by the soft
        path -- not by a spectral ansatz alone, and only at the declared
        orders: the first unmatched one stays discontinuous).
    value
        Target: a float or ``value(coords) -> array``.
    axis
        For Neumann / Robin conditions: the axis whose (outward) normal
        derivative is constrained. For periodic conditions: the seam axis
        (``None`` means every spatial axis the domain declares periodic).
        ``None`` (Dirichlet default) applies the condition on the whole spatial
        boundary.
    side
        ``"lo"`` or ``"hi"`` to restrict to one face of ``axis``; ``None``
        applies to both faces.
    alpha, beta
        Robin coefficients (ignored otherwise).
    periodic_orders
        Derivative orders matched across a periodic seam; defaults to
        :data:`PERIODIC_ORDERS` when ``kind == "periodic"``. Must be ``None``
        for every other kind.
    """

    component: str
    kind: str = "dirichlet"
    value: ValueLike = 0.0
    axis: str | None = None
    side: str | None = None
    alpha: float = 1.0
    beta: float = 0.0
    periodic_orders: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if self.kind not in _BC_KINDS:
            raise ValueError(
                f"unknown boundary-condition kind {self.kind!r}; "
                f"expected one of {sorted(_BC_KINDS)}"
            )
        if self.side is not None and self.side not in ("lo", "hi"):
            raise ValueError(f"side must be 'lo', 'hi', or None; got {self.side!r}")
        if self.kind in ("neumann", "robin") and self.axis is None:
            raise ValueError(f"{self.kind} boundary condition requires an `axis`")
        if self.kind == "periodic":
            orders = self.periodic_orders
            if orders is None:
                object.__setattr__(self, "periodic_orders", PERIODIC_ORDERS)
            else:
                if not orders:
                    raise ValueError("periodic_orders must be a non-empty tuple of ints")
                for order in orders:
                    if type(order) is not int or order < 0:
                        raise ValueError(
                            "periodic_orders must be non-negative ints; "
                            f"got {orders!r}"
                        )
        elif self.periodic_orders is not None:
            raise ValueError(
                "periodic_orders is only valid when kind='periodic'; "
                f"got kind={self.kind!r}"
            )


@dataclass(frozen=True)
class InitialCondition:
    """An initial condition on one component (time-dependent problems).

    Parameters
    ----------
    component
        Component name.
    value
        Target: a float or ``value(coords) -> array`` evaluated at ``t = t0``.
    order
        ``0`` constrains ``u(x, t0)``; ``1`` constrains ``du/dt(x, t0)`` (used
        by second-order-in-time problems such as the wave equation).
    t0
        Initial time. ``None`` means "the lower bound of the time axis".
    """

    component: str
    value: ValueLike = 0.0
    order: int = 0
    t0: float | None = None

    def __post_init__(self) -> None:
        if self.order not in (0, 1):
            raise ValueError(f"initial-condition order must be 0 or 1, got {self.order}")


__all__ = ["BoundaryCondition", "InitialCondition", "PERIODIC_ORDERS", "ValueLike"]
