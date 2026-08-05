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
        or ``"periodic"`` (enforced by the spectral / periodic ansatz).
    value
        Target: a float or ``value(coords) -> array``.
    axis
        For Neumann / Robin conditions: the axis whose (outward) normal
        derivative is constrained. ``None`` (Dirichlet default) applies the
        condition on the whole spatial boundary.
    side
        ``"lo"`` or ``"hi"`` to restrict to one face of ``axis``; ``None``
        applies to both faces.
    alpha, beta
        Robin coefficients (ignored otherwise).
    """

    component: str
    kind: str = "dirichlet"
    value: ValueLike = 0.0
    axis: str | None = None
    side: str | None = None
    alpha: float = 1.0
    beta: float = 0.0

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


__all__ = ["BoundaryCondition", "InitialCondition", "ValueLike"]
