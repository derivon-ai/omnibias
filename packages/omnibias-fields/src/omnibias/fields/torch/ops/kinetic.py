# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Torch kinetic-theory ops (twin of :mod:`omnibias.fields.jax.ops.kinetic`).

The collisionless transport operator and its relaxation closure on a phase-space
``(x, v)`` field ``f(t, x, v)``:

- :func:`vlasov_residual` -- the Vlasov / collisionless-Boltzmann residual
  :math:`\partial_t f + v\cdot\nabla_x f + (F/m)\cdot\nabla_v f`. Every term is a
  closed-form first derivative of the distribution along a named axis, weighted by
  the *coordinate* velocity ``v`` (read straight off ``state.coords``), so the jax
  twin is bit-identical.
- :func:`bgk_collision` -- the BGK relaxation source :math:`-(f-f_{eq})/\tau`.
- :func:`maxwellian` -- the local Maxwell-Boltzmann equilibrium (a closed-form
  Gaussian of the velocity coordinates), handy as the BGK target and as an
  analytic moment oracle.
- :func:`number_density` / :func:`momentum_density` / :func:`kinetic_energy_density`
  -- velocity moments :math:`\int v^k f\,dv` via a supplied velocity-space
  quadrature rule (evaluate the field at ``quadrature_nodes`` mapped into the
  velocity axes first).

Honesty (``docs/scope-and-guarantees.md`` §1): Vlasov transport, BGK relaxation
and the Maxwellian are **closed-form**; the full quadratic, non-local Boltzmann
collision integral is **numerical** (velocity-space quadrature) and is *not*
provided here -- see :data:`BOLTZMANN_COLLISION_IS_NUMERICAL`.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from omnibias.fields._core.quadrature import QuadratureSpec
from omnibias.fields.torch.ops.basic import derivative, stack_components, value
from torch import Tensor

if TYPE_CHECKING:  # pragma: no cover
    from omnibias.fields._core.state import FieldState

#: The full Boltzmann collision integral is a non-local velocity-space integral
#: (quadrature), not a closed-form sigma-tower object. Enforcement tests assert
#: no closed-form Boltzmann-collision op is exported from this module.
BOLTZMANN_COLLISION_IS_NUMERICAL = True


def _require_time(state: FieldState, op: str) -> str:
    ta = state.coordinate_spec.time_axis
    if ta is None:
        raise ValueError(f"{op} requires a time axis on the spec")
    return ta


def _velocity_columns(state: FieldState, velocity_axes: tuple[str, ...]) -> Tensor:
    """Stack the raw coordinate values along ``velocity_axes``, shape ``(B, d)``."""
    idx = [state.coordinate_spec.axis_index(a) for a in velocity_axes]
    return state.coords[:, idx]


def vlasov_residual(
    state: FieldState,
    name: str,
    *,
    position_axes: tuple[str, ...],
    velocity_axes: tuple[str, ...],
    force: tuple[str, ...] | Tensor | None = None,
    mass: float = 1.0,
) -> Tensor:
    r"""Vlasov residual :math:`\partial_t f + v\cdot\nabla_x f + (F/m)\cdot\nabla_v f`.

    Shape ``(B,)``. ``position_axes`` and ``velocity_axes`` name the phase-space
    axes (equal length ``d``); ``force`` is the ``d``-component force field
    (component names or a ``(B, d)`` tensor) driving the velocity-space
    advection. ``force=None`` is force-free (free-streaming) transport.
    """
    if len(position_axes) != len(velocity_axes):
        raise ValueError(
            f"position_axes ({len(position_axes)}) and velocity_axes "
            f"({len(velocity_axes)}) must have equal length"
        )
    ta = _require_time(state, "vlasov_residual")
    v_cols = _velocity_columns(state, velocity_axes)
    res = derivative(state, name, axis=ta, order=1)
    for i, xa in enumerate(position_axes):
        res = res + v_cols[:, i] * derivative(state, name, axis=xa, order=1)
    if force is not None:
        f = stack_components(state, force) if isinstance(force, tuple) else force
        for i, va in enumerate(velocity_axes):
            res = res + (f[:, i] / mass) * derivative(state, name, axis=va, order=1)
    return res


def bgk_collision(
    state: FieldState,
    name: str,
    *,
    equilibrium: str | Tensor,
    tau: float,
) -> Tensor:
    r"""BGK relaxation source :math:`-(f-f_{eq})/\tau`, shape ``(B,)``."""
    if tau <= 0.0:
        raise ValueError(f"relaxation time tau must be > 0, got {tau}")
    f = value(state, name)
    f_eq = value(state, equilibrium) if isinstance(equilibrium, str) else equilibrium
    return -(f - f_eq) / tau


def bgk_vlasov_residual(
    state: FieldState,
    name: str,
    *,
    position_axes: tuple[str, ...],
    velocity_axes: tuple[str, ...],
    equilibrium: str | Tensor,
    tau: float,
    force: tuple[str, ...] | Tensor | None = None,
    mass: float = 1.0,
) -> Tensor:
    r"""Vlasov-BGK residual :math:`\mathcal L f + (f-f_{eq})/\tau`, shape ``(B,)``.

    ``L f`` is :func:`vlasov_residual`; the BGK source is moved to the left-hand
    side so an exact Vlasov-BGK solution has zero residual.
    """
    lhs = vlasov_residual(
        state, name, position_axes=position_axes, velocity_axes=velocity_axes,
        force=force, mass=mass,
    )
    return lhs - bgk_collision(state, name, equilibrium=equilibrium, tau=tau)


def maxwellian(
    state: FieldState,
    *,
    velocity_axes: tuple[str, ...],
    density: float | Tensor = 1.0,
    bulk_velocity: tuple[float, ...] | Tensor | None = None,
    temperature: float | Tensor = 1.0,
    mass: float = 1.0,
) -> Tensor:
    r"""Local Maxwell-Boltzmann equilibrium, shape ``(B,)`` (natural units, :math:`k_B=1`).

    :math:`f_{eq}=n\,(m/2\pi T)^{d/2}\exp(-m|v-u|^2/2T)`, a closed-form Gaussian of
    the velocity coordinates. ``density``/``temperature`` are scalars or ``(B,)``
    tensors; ``bulk_velocity`` is a ``d``-vector (scalars or ``(B, d)``).
    """
    v = _velocity_columns(state, velocity_axes)
    d = len(velocity_axes)
    if bulk_velocity is None:
        dv = v
    else:
        u = torch.as_tensor(bulk_velocity, dtype=v.dtype, device=v.device)
        dv = v - u
    speed_sq = (dv * dv).sum(dim=-1)
    prefactor = density * (mass / (2.0 * math.pi * temperature)) ** (d / 2.0)
    return prefactor * torch.exp(-mass * speed_sq / (2.0 * temperature))


def _quad_weights(rule: QuadratureSpec, ref: Tensor) -> Tensor:
    return torch.as_tensor(rule.weights, dtype=ref.dtype, device=ref.device)


def _check_nodes(state: FieldState, rule: QuadratureSpec, op: str) -> Tensor:
    vals = value(state, state.components.names[0])
    if vals.shape[0] != rule.n_nodes:
        raise ValueError(
            f"{op}: state has {vals.shape[0]} points but rule has {rule.n_nodes} "
            "nodes; evaluate the field at quadrature_nodes(rule)"
        )
    return vals


def number_density(state: FieldState, name: str, *, rule: QuadratureSpec) -> Tensor:
    r"""Zeroth velocity moment :math:`n=\int f\,dv` (scalar tensor)."""
    _check_nodes(state, rule, "number_density")
    vals = value(state, name)
    w = _quad_weights(rule, vals)
    return torch.tensordot(w, vals, dims=([0], [0]))


def momentum_density(
    state: FieldState, name: str, *, rule: QuadratureSpec, velocity_axes: tuple[str, ...],
) -> Tensor:
    r"""First velocity moment :math:`\int v\,f\,dv`, shape ``(d,)``."""
    _check_nodes(state, rule, "momentum_density")
    vals = value(state, name)
    w = _quad_weights(rule, vals)
    v = _velocity_columns(state, velocity_axes)
    cols = [torch.tensordot(w, v[:, i] * vals, dims=([0], [0])) for i in range(v.shape[-1])]
    return torch.stack(cols, dim=-1)


def kinetic_energy_density(
    state: FieldState, name: str, *, rule: QuadratureSpec, velocity_axes: tuple[str, ...],
) -> Tensor:
    r"""Second velocity moment :math:`\int \tfrac12|v|^2 f\,dv` (scalar tensor)."""
    _check_nodes(state, rule, "kinetic_energy_density")
    vals = value(state, name)
    w = _quad_weights(rule, vals)
    v = _velocity_columns(state, velocity_axes)
    speed_sq = (v * v).sum(dim=-1)
    return torch.tensordot(w, 0.5 * speed_sq * vals, dims=([0], [0]))


__all__ = [
    "BOLTZMANN_COLLISION_IS_NUMERICAL",
    "bgk_collision",
    "bgk_vlasov_residual",
    "kinetic_energy_density",
    "maxwellian",
    "momentum_density",
    "number_density",
    "vlasov_residual",
]
