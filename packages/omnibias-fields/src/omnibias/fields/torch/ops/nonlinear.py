# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Nonlinear ops: ``advection``, ``material_derivative``, ``p_laplacian``,
``directional_derivative``.

These ops are quadratic (or higher) in the field outputs, so they don't
fit cleanly inside the closed-form derivative tower. The implementation
folds them through the basic ops: each is a few-line composition that
the kernel does not need to know about.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from omnibias.fields.torch.ops.basic import (
    derivative,
    divergence,
    gradient,
    laplacian,
    stack_components,
    value,
)
from omnibias.fields.torch.ops.high_order import hessian
from torch import Tensor

if TYPE_CHECKING:  # pragma: no cover
    from omnibias.fields._core.state import FieldState


def advection(
    state: FieldState,
    *,
    velocity: tuple[str, ...],
    target: tuple[str, ...] | None = None,
    scalar: str | None = None,
) -> Tensor:
    """Advection ``(velocity . nabla) target``.

    Three calling modes:

    - ``advection(state, velocity=("u","v","w"))`` -- self-advection,
      target defaults to ``velocity``. Returns ``(B, len(velocity))``.
    - ``advection(state, velocity=("u","v","w"), scalar="phi")`` --
      scalar advection ``(u . nabla) phi``. Returns ``(B,)``.
    - ``advection(state, velocity=("u","v","w"), target=("a","b","c"))``
      -- general vector-on-vector advection. Returns ``(B, len(target))``.

    The velocity-tuple length must equal the number of *spatial* axes.
    """
    sa = state.coordinate_spec.spatial_axes
    if len(velocity) != len(sa):
        raise ValueError(
            f"advection: velocity has {len(velocity)} components, "
            f"but coordinate spec has {len(sa)} spatial axes ({sa!r})"
        )
    u_vec = stack_components(state, velocity)              # (B, n_spatial)

    if scalar is not None:
        if target is not None:
            raise ValueError("advection: pass scalar OR target, not both")
        # (u . nabla) phi = sum_a u_a (d phi / dx_a).
        grad_phi = gradient(state, scalar)                 # (B, n_spatial)
        return (u_vec * grad_phi).sum(dim=-1)              # (B,)

    target = target if target is not None else velocity
    cols = []
    for n in target:
        grad_n = gradient(state, n)                        # (B, n_spatial)
        cols.append((u_vec * grad_n).sum(dim=-1))          # (B,)
    return torch.stack(cols, dim=-1)                       # (B, len(target))


def material_derivative(
    state: FieldState,
    *,
    velocity: tuple[str, ...],
    scalar: str | None = None,
) -> Tensor:
    """``D/Dt = d/dt + (velocity . nabla)``.

    For a vector target (default ``velocity`` itself) returns
    ``(B, C)``; for a scalar target returns ``(B,)``.
    """
    if state.coordinate_spec.time_axis is None:
        raise ValueError(
            "material_derivative requires a time axis on the coordinate spec"
        )
    time_axis = state.coordinate_spec.time_axis
    assert time_axis is not None  # guarded above
    if scalar is not None:
        dt_target = derivative(state, scalar, axis=time_axis)  # (B,)
        return dt_target + advection(state, velocity=velocity, scalar=scalar)
    dt_target = torch.stack(
        [derivative(state, n, axis=time_axis) for n in velocity], dim=-1,
    )                                                       # (B, C)
    return dt_target + advection(state, velocity=velocity)


def p_laplacian(
    state: FieldState, name: str, *, p: float, eps: float = 1e-8,
) -> Tensor:
    """``Delta_p u = nabla . (|nabla u|^{p-2} nabla u)`` of shape ``(B,)``.

    Implementation follows ``docs/pinn-derivations.md`` Section 4:

        Delta_p u = |g|^{p-2} Delta u + (p-2) |g|^{p-4} (g^T H g),

    where ``g = nabla u`` and ``H`` is the Hessian, both restricted to
    spatial axes. ``eps`` regularises ``|g|`` to avoid singularity at
    zero gradient.
    """
    if p < 1:
        raise ValueError(f"p must be >= 1 for p-Laplacian, got {p}")
    sa = state.coordinate_spec.spatial_axes
    g = gradient(state, name)                              # (B, n_spatial)
    L = laplacian(state, name)                             # (B,)
    if abs(p - 2.0) < 1e-15:
        return L
    g2 = (g * g).sum(dim=-1)                               # (B,)
    g_norm_sq_reg = g2 + eps * eps                         # (B,)
    # Hessian restricted to spatial axes.
    H_full = hessian(state, name)                          # (B, D, D)
    spatial_idx = [
        state.coordinate_spec.axis_index(a) for a in sa
    ]
    H_sp = H_full[..., spatial_idx, :][..., :, spatial_idx]  # (B, ns, ns)
    gHg = torch.einsum("bi,bij,bj->b", g, H_sp, g)         # (B,)
    base = g_norm_sq_reg.pow((p - 2.0) / 2.0)              # (B,)
    cross = (p - 2.0) * g_norm_sq_reg.pow((p - 4.0) / 2.0) * gHg
    return base * L + cross


def directional_derivative(
    state: FieldState, name: str, *, direction: Tensor,
) -> Tensor:
    """``d_v u = grad u . direction``.

    ``direction`` has shape ``(B, n_spatial)`` matching the gradient.
    """
    g = gradient(state, name)                              # (B, n_spatial)
    if direction.shape != g.shape:
        raise ValueError(
            f"direction shape {tuple(direction.shape)} != "
            f"gradient shape {tuple(g.shape)}"
        )
    return (g * direction).sum(dim=-1)


def skew_symmetric_advection(
    state: FieldState,
    *,
    velocity: tuple[str, ...],
    scalar: str | None = None,
) -> Tensor:
    r"""Energy/enstrophy-conserving (skew-symmetric) advection.

    :math:`\tfrac12[(u\cdot\nabla)\phi + \nabla\cdot(u\phi)]
    = (u\cdot\nabla)\phi + \tfrac12(\nabla\cdot u)\phi`. For a divergence-free
    ``u`` this equals the standard advection; the extra ``0.5(div u)phi`` term is
    what makes the discrete operator skew-symmetric (conserves the quadratic
    invariant). With ``scalar=None`` it is the vector self-advection form.
    """
    div_u = divergence(state, velocity)
    if scalar is not None:
        adv = advection(state, velocity=velocity, scalar=scalar)
        return adv + 0.5 * div_u * value(state, scalar)
    adv = advection(state, velocity=velocity)
    u_i = stack_components(state, velocity)
    return adv + 0.5 * div_u[..., None] * u_i


__all__ = [
    "advection",
    "directional_derivative",
    "material_derivative",
    "p_laplacian",
    "skew_symmetric_advection",
]
