# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Assemble PDE residual + boundary/initial rows for a torch ansatz.

Everything here composes the closed-form differential operators on a
:class:`FieldState` (``state.ops.*``). No differential operator is
re-implemented -- this module only routes the ``System``'s residual closures
and conditions to the right collocation point sets.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import Any

import numpy as np
import torch
from omnibias.pinn.solver._core.conditions import BoundaryCondition, InitialCondition
from omnibias.pinn.solver._core.hard import HardConditionPlan
from omnibias.pinn.solver._core.sampling import (
    CollocationSpec,
    bc_faces,
    boundary_points,
    initial_slice_points,
    interior_points,
    periodic_axes,
    spatial_boundary_points,
)
from omnibias.pinn.solver._core.system import System
from omnibias.pinn.solver.torch.readout import empty_rows, readout_device, readout_dtype
from torch import Tensor


def to_tensor(pts: np.ndarray, field: Any) -> Tensor:
    """Convert numpy points to a tensor matching the field's dtype/device."""
    return torch.as_tensor(
        pts, dtype=readout_dtype(field), device=readout_device(field)
    )


def _eval_target(value: Any, coords: Tensor) -> Any:
    return value(coords) if callable(value) else value


def interior_residual(field: Any, system: System, coords: Tensor) -> Tensor:
    """Stack every governing-equation residual at ``coords`` -> ``(n_eq*B,)``."""
    state = field(coords)
    parts = [torch.reshape(r(state), (-1,)) for r in system.residuals]
    return torch.cat(parts) if parts else coords.new_zeros(0)


def _periodic_rows(
    field: Any, system: System, bc: BoundaryCondition, spec: CollocationSpec
) -> Tensor:
    """Seam-matching rows ``d^n u(hi, .) - d^n u(lo, .)`` for ``n`` in the orders.

    This used to be a zero-length row on the grounds that the ansatz carried
    periodicity. That holds for the spectral method-of-lines route; the
    mesh-free ansatz here is a plain one-layer field, so nothing enforced the
    seam and a declared periodic condition was silently doing nothing.

    Shares :func:`_periodic_terms` with the linear path rather than repeating
    the construction, because this function is also the anti-silence guard: two
    copies that could disagree about what a seam means would defeat it.
    """
    rows = [_periodic_row(term) for term in _periodic_terms(field, system, bc, spec)]
    return torch.cat(rows) if rows else empty_rows(field)


def _bc_rows(
    field: Any, system: System, bc: BoundaryCondition, spec: CollocationSpec
) -> Tensor:
    if bc.kind == "periodic":
        return _periodic_rows(field, system, bc, spec)
    rows: list[Tensor] = []
    for axis, side in bc_faces(system.domain, bc):
        pts = boundary_points(system.domain, spec, axis=axis, side=side)
        if pts.shape[0] == 0:
            continue
        coords = to_tensor(pts, field)
        state = field(coords)
        target = _eval_target(bc.value, coords)
        u = state.ops.value(state, bc.component)
        if bc.kind == "dirichlet":
            rows.append(u - target)
            continue
        d = state.ops.derivative(state, bc.component, axis=axis, order=1)
        normal = d if side == "hi" else -d
        if bc.kind == "neumann":
            rows.append(normal - target)
        else:  # robin: alpha*u + beta*du/dn = value
            rows.append(bc.alpha * u + bc.beta * normal - target)
    return torch.cat(rows) if rows else empty_rows(field)


def _ic_rows(
    field: Any, system: System, ic: InitialCondition, spec: CollocationSpec
) -> Tensor:
    pts = initial_slice_points(system.domain, spec, t0=ic.t0)
    coords = to_tensor(pts, field)
    state = field(coords)
    target = _eval_target(ic.value, coords)
    ta = system.domain.time_axis
    if ic.order == 0:
        return state.ops.value(state, ic.component) - target
    return state.ops.derivative(state, ic.component, axis=ta, order=1) - target


def condition_residual(
    field: Any,
    system: System,
    spec: CollocationSpec,
    hard: HardConditionPlan | None = None,
) -> Tensor:
    """Stack every boundary + initial condition row.

    Conditions the ``hard`` plan reports absorbed contribute no rows: the
    architecture enforces them, so a penalty could only add noise. Passing
    ``hard=None`` assembles every condition, which is what makes it usable as a
    check on the plan's own claims.
    """
    absorbed_bc = hard.absorbed_boundary if hard else frozenset()
    absorbed_ic = hard.absorbed_initial if hard else frozenset()
    rows: list[Tensor] = []
    for i, bc in enumerate(system.boundary):
        if i in absorbed_bc:
            continue
        r = _bc_rows(field, system, bc, spec)
        if r.numel():
            rows.append(r)
    for i, ic in enumerate(system.initial):
        if i in absorbed_ic:
            continue
        rows.append(_ic_rows(field, system, ic, spec))
    return torch.cat(rows) if rows else empty_rows(field)


def all_rows(
    field: Any,
    system: System,
    coords_interior: Tensor,
    spec: CollocationSpec,
    hard: HardConditionPlan | None = None,
) -> Tensor:
    """Interior residual rows followed by condition rows."""
    interior = interior_residual(field, system, coords_interior)
    conditions = condition_residual(field, system, spec, hard)
    return torch.cat([interior, conditions])


def default_interior(field: Any, system: System, spec: CollocationSpec) -> Tensor:
    """The interior collocation tensor for this domain/spec."""
    return to_tensor(interior_points(system.domain, spec), field)


def residual_norm(
    field: Any, system: System, spec: CollocationSpec | None = None
) -> float:
    """RMS of the full (interior + condition) residual, for diagnostics.

    Deliberately assembled with **every** condition, absorbed or not: a caged
    solve should report the same quantity as an uncaged one, and a cage that
    stopped enforcing something would show up here rather than be hidden by the
    plan that dropped it from the loss.
    """
    spec = spec or CollocationSpec()
    coords = default_interior(field, system, spec)
    with torch.no_grad():
        rows = all_rows(field, system, coords, spec)
        return float(torch.sqrt(torch.mean(rows ** 2)))


# ---------------------------------------------------------------------
# Cached collocation plan (fast frozen-feature assembly for linear solves)
# ---------------------------------------------------------------------
#
# Declaring fields cache only readout-independent quantities (SigmaCache of
# the frozen feature map for a one-layer field; the temporal hidden state ``h``
# for spectral / Chebyshev). The ``FieldState`` can therefore be built ONCE and
# reused while only the linear readout varies -- that is the readout-
# independence invariant. This turns the ``O(#unknowns)`` linear assembly from
# ``O(H^2 N)`` full re-evaluations into cheap readout contractions.


@dataclass
class _BCTerm:
    state: Any
    kind: str
    component: str
    axis: str
    sign: float
    alpha: float
    beta: float
    target: Tensor


@dataclass
class _ICTerm:
    state: Any
    component: str
    order: int
    time_axis: str
    target: Tensor


@dataclass
class _PeriodicTerm:
    """One seam-matching row: the two face states and the order matched across."""

    lo_state: Any
    hi_state: Any
    component: str
    axis: str
    order: int


def _periodic_row(term: _PeriodicTerm) -> Tensor:
    """``d^order u(hi, .) - d^order u(lo, .)``, zero when the seam is closed."""
    lo_s, hi_s = term.lo_state, term.hi_state
    if term.order == 0:
        return hi_s.ops.value(hi_s, term.component) - lo_s.ops.value(
            lo_s, term.component
        )
    return hi_s.ops.derivative(
        hi_s, term.component, axis=term.axis, order=term.order
    ) - lo_s.ops.derivative(lo_s, term.component, axis=term.axis, order=term.order)


@dataclass
class CollocationPlan:
    """Pre-built states + targets so linear assembly reuses cached ``sigma``."""

    interior_state: Any
    residuals: tuple[Any, ...]
    bc_terms: list[_BCTerm]
    ic_terms: list[_ICTerm]
    periodic_terms: list[_PeriodicTerm] = dc_field(default_factory=list)


def _as_target(value: Any, coords: Tensor) -> Tensor:
    target = _eval_target(value, coords)
    if isinstance(target, Tensor):
        return target
    return coords.new_full((coords.shape[0],), float(target))


def build_plan(
    field: Any,
    system: System,
    spec: CollocationSpec,
    hard: HardConditionPlan | None = None,
) -> CollocationPlan:
    """Build a cached collocation plan for a (frozen-feature) field.

    Absorbed conditions contribute no terms, so the column probing in the linear
    solve costs nothing for them -- the cage already carries them, and it is
    affine in the readout, so the probing itself stays valid.
    """
    absorbed_bc = hard.absorbed_boundary if hard else frozenset()
    absorbed_ic = hard.absorbed_initial if hard else frozenset()
    interior_state = field(default_interior(field, system, spec))
    bc_terms: list[_BCTerm] = []
    periodic_terms: list[_PeriodicTerm] = []
    for i, bc in enumerate(system.boundary):
        if i in absorbed_bc:
            continue
        if bc.kind == "periodic":
            periodic_terms.extend(_periodic_terms(field, system, bc, spec))
            continue
        for axis, side in bc_faces(system.domain, bc):
            pts = boundary_points(system.domain, spec, axis=axis, side=side)
            if pts.shape[0] == 0:
                continue
            coords = to_tensor(pts, field)
            bc_terms.append(
                _BCTerm(
                    state=field(coords),
                    kind=bc.kind,
                    component=bc.component,
                    axis=axis,
                    sign=1.0 if side == "hi" else -1.0,
                    alpha=bc.alpha,
                    beta=bc.beta,
                    target=_as_target(bc.value, coords),
                )
            )
    ic_terms: list[_ICTerm] = []
    for i, ic in enumerate(system.initial):
        if i in absorbed_ic:
            continue
        pts = initial_slice_points(system.domain, spec, t0=ic.t0)
        coords = to_tensor(pts, field)
        ic_terms.append(
            _ICTerm(
                state=field(coords),
                component=ic.component,
                order=ic.order,
                time_axis=system.domain.time_axis,  # type: ignore[arg-type]
                target=_as_target(ic.value, coords),
            )
        )
    return CollocationPlan(
        interior_state, tuple(system.residuals), bc_terms, ic_terms, periodic_terms
    )


def _periodic_terms(
    field: Any, system: System, bc: BoundaryCondition, spec: CollocationSpec
) -> list[_PeriodicTerm]:
    """Cached seam states, one pair per (axis, order) the periodic row matches."""
    cs = system.domain.coordinate_spec
    orders = bc.periodic_orders
    assert orders is not None  # BoundaryCondition defaults for kind="periodic"
    out: list[_PeriodicTerm] = []
    for axis in periodic_axes(system.domain, bc):
        pts = boundary_points(system.domain, spec, axis=axis, side="lo")
        if pts.shape[0] == 0:
            continue
        index = cs.axis_index(axis)
        _, hi = system.domain.bounds[index]
        low = to_tensor(pts, field)
        high = low.clone()
        high[:, index] = hi
        lo_state, hi_state = field(low), field(high)
        out.extend(
            _PeriodicTerm(
                lo_state=lo_state,
                hi_state=hi_state,
                component=bc.component,
                axis=axis,
                order=order,
            )
            for order in orders
        )
    return out


def eval_plan_rows(plan: CollocationPlan) -> Tensor:
    """Evaluate the full residual vector for a plan at the field's current readout."""
    parts: list[Tensor] = [
        torch.reshape(r(plan.interior_state), (-1,)) for r in plan.residuals
    ]
    for t in plan.bc_terms:
        s = t.state
        u = s.ops.value(s, t.component)
        if t.kind == "dirichlet":
            parts.append(u - t.target)
            continue
        d = s.ops.derivative(s, t.component, axis=t.axis, order=1)
        normal = t.sign * d
        if t.kind == "neumann":
            parts.append(normal - t.target)
        else:
            parts.append(t.alpha * u + t.beta * normal - t.target)
    for ic in plan.ic_terms:
        s = ic.state
        if ic.order == 0:
            parts.append(s.ops.value(s, ic.component) - ic.target)
        else:
            parts.append(
                s.ops.derivative(s, ic.component, axis=ic.time_axis, order=1) - ic.target
            )
    parts.extend(_periodic_row(term) for term in plan.periodic_terms)
    return torch.cat(parts)


# re-exported for callers that want the spatial-boundary sampler
__all__ = [
    "CollocationPlan",
    "all_rows",
    "build_plan",
    "condition_residual",
    "default_interior",
    "eval_plan_rows",
    "interior_residual",
    "residual_norm",
    "spatial_boundary_points",
    "to_tensor",
]
