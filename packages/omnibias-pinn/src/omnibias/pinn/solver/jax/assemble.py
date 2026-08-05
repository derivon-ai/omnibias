# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Assemble PDE residual + boundary/initial rows for a jax ansatz.

Bit-identical twin of :mod:`omnibias.pinn.solver.torch.assemble`: it composes the same
closed-form operators on a :class:`FieldState` (``state.ops.*``) and routes the
``System`` residuals + conditions to the same collocation point sets.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np
from omnibias.pinn.solver._core.conditions import BoundaryCondition, InitialCondition
from omnibias.pinn.solver._core.sampling import (
    CollocationSpec,
    bc_faces,
    boundary_points,
    initial_slice_points,
    interior_points,
)
from omnibias.pinn.solver._core.system import System


def to_array(pts: np.ndarray, field: Any) -> Any:
    return jnp.asarray(pts, dtype=field.W.dtype)


def _eval_target(value: Any, coords: Any) -> Any:
    return value(coords) if callable(value) else value


def interior_residual(field: Any, system: System, coords: Any) -> Any:
    state = field(coords)
    parts = [jnp.reshape(r(state), (-1,)) for r in system.residuals]
    return jnp.concatenate(parts) if parts else jnp.zeros((0,), dtype=field.W.dtype)


def _bc_rows(
    field: Any, system: System, bc: BoundaryCondition, spec: CollocationSpec
) -> Any:
    if bc.kind == "periodic":
        return jnp.zeros((0,), dtype=field.W.dtype)
    rows: list[Any] = []
    for axis, side in bc_faces(system.domain, bc):
        pts = boundary_points(system.domain, spec, axis=axis, side=side)
        if pts.shape[0] == 0:
            continue
        coords = to_array(pts, field)
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
        else:
            rows.append(bc.alpha * u + bc.beta * normal - target)
    return jnp.concatenate(rows) if rows else jnp.zeros((0,), dtype=field.W.dtype)


def _ic_rows(
    field: Any, system: System, ic: InitialCondition, spec: CollocationSpec
) -> Any:
    coords = to_array(initial_slice_points(system.domain, spec, t0=ic.t0), field)
    state = field(coords)
    target = _eval_target(ic.value, coords)
    ta = system.domain.time_axis
    if ic.order == 0:
        return state.ops.value(state, ic.component) - target
    return state.ops.derivative(state, ic.component, axis=ta, order=1) - target


def condition_residual(field: Any, system: System, spec: CollocationSpec) -> Any:
    rows: list[Any] = []
    for bc in system.boundary:
        r = _bc_rows(field, system, bc, spec)
        if r.shape[0]:
            rows.append(r)
    for ic in system.initial:
        rows.append(_ic_rows(field, system, ic, spec))
    return jnp.concatenate(rows) if rows else jnp.zeros((0,), dtype=field.W.dtype)


def all_rows(field: Any, system: System, coords_interior: Any, spec: CollocationSpec) -> Any:
    return jnp.concatenate(
        [
            interior_residual(field, system, coords_interior),
            condition_residual(field, system, spec),
        ]
    )


def default_interior(field: Any, system: System, spec: CollocationSpec) -> Any:
    return to_array(interior_points(system.domain, spec), field)


def residual_norm(field: Any, system: System, spec: CollocationSpec | None = None) -> float:
    spec = spec or CollocationSpec()
    rows = all_rows(field, system, default_interior(field, system, spec), spec)
    return float(jnp.sqrt(jnp.mean(rows ** 2)))


__all__ = [
    "all_rows",
    "condition_residual",
    "default_interior",
    "interior_residual",
    "residual_norm",
    "to_array",
]
