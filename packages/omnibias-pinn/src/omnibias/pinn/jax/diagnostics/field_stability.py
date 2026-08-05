# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Field-level stability diagnostics (jax twin)."""

from __future__ import annotations

import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array
from omnibias.pinn.jax import ops as jops


@dataclass(frozen=True)
class StabilityRow:
    order: int
    closed_form: float
    autograd: float
    rel_diff: float
    abs_diff: float


def _polylaplacian_autograd(field, coords: Array, *, name: str, k: int) -> Array:
    """Compute :math:`\\Delta^k u_{\\text{name}}` via JAX autograd.

    JAX is functional; we differentiate scalar-output functions and
    `jax.vmap` over the batch axis to keep memory bounded.
    """
    spatial_axes = field.coordinate_spec.spatial_axes
    spatial_idx = [
        field.coordinate_spec.axis_index(a) for a in spatial_axes
    ]

    def value_fn(c_one: Array) -> Array:
        return jops.value(field(c_one[None, :]), name)[0]

    def laplace_scalar(g_fn):
        """Wrap a scalar-output function and return its Laplacian
        (sum of pure 2nd partials over spatial axes)."""
        def lap_g_fn(c_one: Array) -> Array:
            def gd_fn(c1: Array) -> Array:
                grad = jax.grad(g_fn)(c1)
                # Return spatial-grad component as scalar via summing
                # over spatial dimensions inside an outer wrapper.
                return grad
            grad = jax.grad(g_fn)(c_one)                # (D,)
            # 2nd-order partials along each spatial axis.
            total = jnp.zeros((), dtype=grad.dtype)
            for d in spatial_idx:
                # Per-axis directional derivative of grad component d.
                def gd_func(c1, d=d):
                    return jax.grad(g_fn)(c1)[d]
                gdd = jax.grad(gd_func)(c_one)[d]
                total = total + gdd
            return total
        return lap_g_fn

    cur = value_fn
    for _ in range(k):
        cur = laplace_scalar(cur)
    return jax.vmap(cur)(coords)


def derivative_stability(
    field, coords: Array, *,
    component: str = "u",
    max_order: int = 4,
    eps: float = 1e-30,
) -> list[StabilityRow]:
    """Sweep polylaplacian orders, comparing closed-form vs autograd."""
    if max_order < 1:
        raise ValueError(f"max_order must be >= 1, got {max_order}")

    out: list[StabilityRow] = []
    for k in range(1, max_order + 1):
        state = field(coords)
        try:
            cf = jops.polylaplacian(state, component, k=k)
            cf_arr = jnp.asarray(cf)
        except (NotImplementedError, AttributeError):
            cf_arr = None

        ag = _polylaplacian_autograd(field, coords, name=component, k=k)

        if cf_arr is None:
            row = StabilityRow(
                order=k,
                closed_form=float("nan"),
                autograd=float(jnp.mean(jnp.abs(ag))),
                rel_diff=float("nan"),
                abs_diff=float("nan"),
            )
        else:
            abs_d = float(jnp.max(jnp.abs(cf_arr - ag)))
            scale = float(jnp.max(jnp.abs(ag)))
            rel_d = abs_d / max(scale, eps)
            row = StabilityRow(
                order=k,
                closed_form=float(jnp.mean(jnp.abs(cf_arr))),
                autograd=float(jnp.mean(jnp.abs(ag))),
                rel_diff=rel_d,
                abs_diff=abs_d,
            )
        out.append(row)
    return out


@dataclass(frozen=True)
class AutogradPhaseRow:
    order: int
    closed_form_seconds: float
    autograd_seconds: float
    speedup: float


def autograd_phase_check(
    field, coords: Array, *,
    component: str = "u",
    max_order: int = 3,
    repeats: int = 3,
) -> list[AutogradPhaseRow]:
    """Wallclock-vs-derivative-order curve."""
    if max_order < 1:
        raise ValueError(f"max_order must be >= 1, got {max_order}")
    if repeats < 1:
        raise ValueError(f"repeats must be >= 1, got {repeats}")

    out: list[AutogradPhaseRow] = []
    for k in range(1, max_order + 1):
        cf_times: list[float] = []
        for _ in range(repeats):
            state = field(coords)
            t0 = time.perf_counter()
            try:
                _ = jops.polylaplacian(state, component, k=k)
                cf_times.append(time.perf_counter() - t0)
            except (NotImplementedError, AttributeError):
                cf_times.append(float("nan"))
                break
        cf_med = sorted(cf_times)[len(cf_times) // 2]

        ag_times: list[float] = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            _ = _polylaplacian_autograd(
                field, coords, name=component, k=k,
            )
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
