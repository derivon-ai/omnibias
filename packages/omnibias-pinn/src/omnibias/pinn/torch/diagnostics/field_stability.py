# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Field-level stability diagnostics (torch).

* :func:`derivative_stability` -- closed-form vs autograd parity sweep
  across high derivative orders.
* :func:`autograd_phase_check` -- wallclock curve over derivative orders.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
from omnibias.pinn._core.state import FieldState
from omnibias.pinn.torch import ops as tops
from torch import Tensor


@dataclass(frozen=True)
class StabilityRow:
    """One row of the bit-stability table."""

    order: int          # ``k`` in polylaplacian; 2k = derivative order
    closed_form: float  # mean |Delta^k u| (closed-form)
    autograd: float     # mean |Delta^k u| (autograd)
    rel_diff: float     # max |closed - autograd| / max(|autograd|, eps)
    abs_diff: float     # max |closed - autograd|


def _laplacian_via_autograd(coords: Tensor, value: Tensor, spatial_dims: int) -> Tensor:
    """Compute Laplacian of a scalar field via autograd. Slow, ``O(D)``
    in the inner loop and ``O(2)`` Hessian if no graph reuse."""
    grads = torch.autograd.grad(
        value.sum(), coords, create_graph=True,
    )[0]
    lap = torch.zeros_like(value)
    for d in range(spatial_dims):
        gd = grads[..., d]
        gdd = torch.autograd.grad(
            gd.sum(), coords, create_graph=True,
        )[0][..., d]
        lap = lap + gdd
    return lap


def _polylaplacian_autograd(state: FieldState, name: str, k: int) -> Tensor:
    """Autograd polylaplacian: apply Laplacian k times."""
    coords = state.coords
    if not coords.requires_grad:
        coords = coords.detach().clone()
        coords.requires_grad_(True)
        # Re-run the field on grad-enabled coords.
        state = state.field(coords)
    cur = tops.value(state, name)
    spatial_axes = state.coordinate_spec.spatial_axes
    spatial_idx = [
        state.coordinate_spec.axis_index(a) for a in spatial_axes
    ]
    # All spatial axes are contiguous in the coords layout.
    for _ in range(k):
        grads = torch.autograd.grad(
            cur.sum(), coords, create_graph=True,
        )[0]
        new_lap = torch.zeros_like(cur)
        for d in spatial_idx:
            gd = grads[..., d]
            gdd = torch.autograd.grad(
                gd.sum(), coords, create_graph=True,
            )[0][..., d]
            new_lap = new_lap + gdd
        cur = new_lap
    return cur


def derivative_stability(
    field, coords: Tensor, *,
    component: str = "u",
    max_order: int = 4,
    eps: float = 1e-30,
) -> list[StabilityRow]:
    """Sweep polylaplacian orders and compare closed-form to autograd.

    Parameters
    ----------
    field
        A typed PINN field (must support ``state.ops.polylaplacian``).
    coords
        Input coordinates ``(B, D)``. For autograd we'll set
        ``requires_grad=True`` on a clone.
    component
        Field component name. Default ``"u"``.
    max_order
        Maximum polylaplacian order ``k``. Default 4 (i.e. up to
        ``Delta^4 u`` = 8th-order derivative).
    eps
        Floor on the denominator of the relative-difference metric.

    Returns
    -------
    list[StabilityRow] -- one entry per ``k = 1..max_order``.
    """
    if max_order < 1:
        raise ValueError(f"max_order must be >= 1, got {max_order}")

    out: list[StabilityRow] = []
    for k in range(1, max_order + 1):
        # Closed-form on a separate state (no autograd graph).
        coords_cf = coords.detach().clone()
        state_cf = field(coords_cf)
        try:
            cf = tops.polylaplacian(state_cf, component, k=k)
        except (NotImplementedError, AttributeError):
            cf = None

        # Autograd on a fresh state with grad-enabled coords.
        coords_ag = coords.detach().clone()
        coords_ag.requires_grad_(True)
        state_ag = field(coords_ag)
        ag = _polylaplacian_autograd(state_ag, component, k)

        if cf is None:
            row = StabilityRow(
                order=k,
                closed_form=float("nan"),
                autograd=float(ag.detach().abs().mean()),
                rel_diff=float("nan"),
                abs_diff=float("nan"),
            )
        else:
            cf_d = cf.detach()
            ag_d = ag.detach()
            abs_d = float((cf_d - ag_d).abs().max())
            scale = float(ag_d.abs().max())
            rel_d = abs_d / max(scale, eps)
            row = StabilityRow(
                order=k,
                closed_form=float(cf_d.abs().mean()),
                autograd=float(ag_d.abs().mean()),
                rel_diff=rel_d,
                abs_diff=abs_d,
            )
        out.append(row)
    return out


@dataclass(frozen=True)
class AutogradPhaseRow:
    """One row of the autograd-phase-transition timing curve."""

    order: int                        # k in polylaplacian
    closed_form_seconds: float
    autograd_seconds: float
    speedup: float                    # autograd / closed_form


def autograd_phase_check(
    field, coords: Tensor, *,
    component: str = "u",
    max_order: int = 3,
    repeats: int = 3,
) -> list[AutogradPhaseRow]:
    """Wallclock-vs-derivative-order curve.

    Times the closed-form polylaplacian and the autograd polylaplacian
    at orders ``k = 1..max_order``. Returns one row per order with the
    median wallclock over ``repeats`` runs each.
    """
    if max_order < 1:
        raise ValueError(f"max_order must be >= 1, got {max_order}")
    if repeats < 1:
        raise ValueError(f"repeats must be >= 1, got {repeats}")

    out: list[AutogradPhaseRow] = []
    for k in range(1, max_order + 1):
        # Closed-form timing.
        cf_times: list[float] = []
        for _ in range(repeats):
            coords_cf = coords.detach().clone()
            state_cf = field(coords_cf)
            t0 = time.perf_counter()
            try:
                _ = tops.polylaplacian(state_cf, component, k=k)
            except (NotImplementedError, AttributeError):
                cf_times.append(float("nan"))
                break
            cf_times.append(time.perf_counter() - t0)
        cf_med = sorted(cf_times)[len(cf_times) // 2]

        # Autograd timing.
        ag_times: list[float] = []
        for _ in range(repeats):
            coords_ag = coords.detach().clone()
            coords_ag.requires_grad_(True)
            state_ag = field(coords_ag)
            t0 = time.perf_counter()
            _ = _polylaplacian_autograd(state_ag, component, k)
            ag_times.append(time.perf_counter() - t0)
        ag_med = sorted(ag_times)[len(ag_times) // 2]

        speedup = float("nan") if cf_med != cf_med else (
            ag_med / cf_med if cf_med > 0 else float("inf")
        )
        out.append(AutogradPhaseRow(
            order=k, closed_form_seconds=cf_med,
            autograd_seconds=ag_med, speedup=speedup,
        ))
    return out


__all__ = [
    "AutogradPhaseRow",
    "StabilityRow",
    "autograd_phase_check",
    "derivative_stability",
]
